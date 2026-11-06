# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FusedMoEDecode (ops/moe/fused_decode.py) and ExpertDown (ops/moe/expert_down.py): the down kernels ship with the
package, weights round to the LoFi width, unsupported configurations are rejected, and on device one decode token's
MoE output matches torch from HF-style weights, with a shared expert and under expert parallelism."""

import re
from pathlib import Path

import pytest
import torch
import ttnn

from tt_transformers.ops.moe import _program
from tt_transformers.ops.moe.expert_down import ExpertDown
from tt_transformers.ops.moe.fused_decode import FusedMoEDecode, SharedExpert, round_sig

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[3]

MESH_1X1 = pytest.mark.parametrize(
    "ttnn_mesh_device",
    [pytest.param({"mesh_shape": (1, 1), "trace_region_size": 32 << 20}, id="p150-1x1")],
    indirect=True,
)


@pytest.mark.host
def test_down_kernels_ship_and_are_covered_by_package_data():
    globs = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"][
        "tt_transformers.ops.moe"
    ]
    packaged = {path for glob in globs for path in _program.KERNEL_DIR.parent.glob(glob)}
    for name in ("dn_reader.cpp", "dn_writer.cpp", "dn_compute.cpp"):
        kernel = _program.KERNEL_DIR / name
        assert kernel in packaged, name
        for include in re.findall(r'#include "([^"]+)"', kernel.read_text()):
            if (kernel.parent / include).exists():
                assert (kernel.parent / include) in packaged, f"{name}: {include}"
            else:
                assert include.startswith(("api/", "ckernel")), f"{name}: {include}"


@pytest.mark.host
def test_round_sig_keeps_five_significant_bits():
    x = torch.randn(10000) * torch.logspace(-4, 4, 10000)
    r = round_sig(x)
    mantissa, _ = torch.frexp(r)
    assert torch.equal(mantissa * 32, torch.round(mantissa * 32))  # at most 5 significant bits
    assert ((r - x).abs() <= x.abs() * 2**-5).all()  # half an ulp at 5 bits, relative
    assert torch.equal(round_sig(r), r)


@pytest.mark.host
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(weight_dtype=ttnn.bfloat16),
        dict(hidden=2000),
        dict(ep_experts=32),
        dict(shared=False, ep_experts=48),
        dict(k=16),
        dict(inter=96),
        dict(inter=1024),
    ],
    ids=[
        "bf16_weights",
        "hidden_not_tile_aligned",
        "ep_with_shared",
        "ep_not_dividing_experts",
        "k_too_large",
        "odd_inter_tiles",
        "stack_over_256_tiles",
    ],
)
def test_expert_down_rejects_unsupported_configuration(kwargs):
    args = dict(hidden=2048, inter=128, num_experts=64, k=8, weight_dtype=ttnn.bfloat8_b) | kwargs
    with pytest.raises(ValueError):
        ExpertDown(None, **args)


def _hf_experts(E, H, inter, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(E, inter, H, generator=g) * 0.03 for _ in range(2)] + [
        torch.randn(E, H, inter, generator=g) * 0.03
    ]


def _upload(t, dtype, dev):
    tt = ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return tt, ttnn.to_torch(tt).float()


def _routed(dev, gate_proj, up_proj, down_proj, dtype, gate_first, down_dtype=None):
    """The routed tensors as a model holds them for prefill: [1, E, H, 2 I] packed gate|up, [1, E, I, H] down."""
    a, b = (gate_proj, up_proj) if gate_first else (up_proj, gate_proj)
    up_gate, w_ug = _upload(round_sig(torch.cat([a.transpose(1, 2), b.transpose(1, 2)], -1)).unsqueeze(0), dtype, dev)
    down, w_dn = _upload(round_sig(down_proj.transpose(1, 2)).unsqueeze(0), down_dtype or dtype, dev)
    return up_gate, down, w_ug[0], w_dn[0]


def _swiglu_down(x, w_ug, w_dn, inter, gate_first):
    a, b = x @ w_ug[:, :inter], x @ w_ug[:, inter:]
    gate, up = (a, b) if gate_first else (b, a)
    return (torch.nn.functional.silu(gate) * up) @ w_dn


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def _x(dev, H, seed):
    torch.manual_seed(seed)
    xb = torch.randn(1, 1, 1, H).bfloat16()
    return xb, ttnn.from_torch(
        xb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def _routing(op, x_tt, E, k):
    """Top-k ids (ties to the lower id), softmax weights and the shared-gate logit from the op's own router output."""
    logits = op.logits(x_tt)
    row = ttnn.to_torch(logits).float().reshape(-1)
    ttnn.deallocate(logits)
    ids = sorted(range(E), key=lambda i: (-row[i].item(), i))[:k]
    return ids, torch.softmax(row[ids], -1), row[E]


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize(
    "inter, dtype, down_dtype",
    [
        (128, ttnn.bfloat8_b, ttnn.bfloat8_b),
        (768, ttnn.bfloat4_b, ttnn.bfloat4_b),
        (128, ttnn.bfloat8_b, ttnn.bfloat4_b),
    ],
    ids=["bfp8", "bfp4_m6", "bfp8_bfp4_down"],
)
def test_shared_expert_moe_matches_torch(ttnn_mesh_device, inter, dtype, down_dtype):
    """Qwen3.5-MoE layout on one device: softmax top-k routed experts plus a sigmoid-gated shared expert."""
    dev, H, E, k = ttnn_mesh_device, 2048, 64, 8
    gate_proj, up_proj, down_proj = _hf_experts(E, H, inter, seed=1)
    up_gate, down, w_ug, w_dn = _routed(dev, gate_proj, up_proj, down_proj, dtype, False, down_dtype)
    g = torch.Generator().manual_seed(2)
    shared = SharedExpert(
        expert_gate=torch.randn(1, H, generator=g) * 0.03,
        gate_proj=torch.randn(inter, H, generator=g) * 0.03,
        up_proj=torch.randn(inter, H, generator=g) * 0.03,
        down_proj=torch.randn(H, inter, generator=g) * 0.03,
    )
    router = torch.randn(E, H, generator=g) * 0.05
    cache = {}
    kw = dict(hidden=H, inter=inter, num_experts=E, k=k, weight_dtype=dtype, down_dtype=down_dtype, cache=cache)
    op = FusedMoEDecode(dev, router_weight=router, shared_expert=shared, **kw)
    w_sug = ttnn.to_torch(op.shared_up_gate).float()[0, 0]
    w_sdn = ttnn.to_torch(op.shared_down).float()[0, 0]

    for trial in range(3):
        xb, x_tt = _x(dev, H, seed=trial)
        out = ttnn.to_torch(op(x_tt, up_gate, down)).float().reshape(-1)
        ids, weights, gate_logit = _routing(op, x_tt, E, k)
        xf = xb.float().reshape(-1)
        ref = torch.sigmoid(gate_logit) * _swiglu_down(xf, w_sug, w_sdn, inter, False)
        for j, e in enumerate(ids):
            ref += weights[j] * _swiglu_down(xf, w_ug[e], w_dn[e], inter, False)
        assert _pcc(out, ref) > 0.999, (trial, _pcc(out, ref))
        ttnn.deallocate(x_tt)

    second_layer = FusedMoEDecode(dev, router_weight=router, shared_expert=shared, **kw)
    assert second_layer.gate_up is op.gate_up and second_layer.out is op.out  # one set of ops and buffers per model


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize("window", ["mixed", "none", "all"])
def test_expert_parallel_moe_matches_torch(ttnn_mesh_device, window):
    """Qwen3-Coder-30B-A3B layout: this die holds experts [0, 32) of 128 ([gate | up] packing, no shared expert) and
    returns the part of the MoE output from the top-k experts in its window, zero when none of them are."""
    dev, H, inter, E, local, k = ttnn_mesh_device, 2048, 768, 128, 32, 8
    gate_proj, up_proj, down_proj = _hf_experts(local, H, inter, seed=3)
    up_gate, down, w_ug, w_dn = _routed(dev, gate_proj, up_proj, down_proj, ttnn.bfloat4_b, gate_first=True)
    xb, x_tt = _x(dev, H, seed=4)
    xf = xb.float().reshape(-1)
    router = torch.randn(E, H, generator=torch.Generator().manual_seed(5)) * 0.05
    router[:local] += {"mixed": 0, "none": -20, "all": 20}[window] * xf / xf.dot(xf)  # shifts the window's logits

    op = FusedMoEDecode(
        dev,
        hidden=H,
        inter=inter,
        num_experts=E,
        k=k,
        weight_dtype=ttnn.bfloat4_b,
        router_weight=router,
        ep_experts=local,
        gate_first=True,
    )
    out = ttnn.to_torch(op(x_tt, up_gate, down)).float().reshape(-1)
    ids, weights, _ = _routing(op, x_tt, E, k)
    live = [(j, e) for j, e in enumerate(ids) if e < local]
    assert len(live) == {"none": 0, "all": k}.get(window, len(live))
    if not live:
        assert out.abs().max().item() == 0.0
    else:
        ref = sum(weights[j] * _swiglu_down(xf, w_ug[e], w_dn[e], inter, True) for j, e in live)
        assert _pcc(out, ref) > 0.999, _pcc(out, ref)
    ttnn.deallocate(x_tt)


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize("shared_first", [True, False], ids=["shared_first", "routed_first"])
def test_expert_down_shared_and_routed_only_take_separate_cache_entries(ttnn_mesh_device, request, shared_first):
    """shared changes only which slots the kernels read, so it must key the program cache."""
    dev, H, inter, E, k = ttnn_mesh_device, 2048, 128, 64, 4
    dev.enable_program_cache()
    dev.clear_program_cache()
    request.addfinalizer(dev.disable_and_clear_program_cache)
    _, _, down_proj = _hf_experts(E, H, inter, seed=6)
    down, w_dn = _upload(round_sig(down_proj.transpose(1, 2)).unsqueeze(0), ttnn.bfloat8_b, dev)
    shared, w_sdn = _upload(round_sig(torch.randn(1, 1, inter, H) * 0.03), ttnn.bfloat8_b, dev)
    torch.manual_seed(7)
    act_t = torch.randn(1, 1, k + 1, inter).bfloat16()
    ids = [3, 17, 42, 60]
    rec = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
    rec[0, 0, 0, :k] = torch.tensor(ids)
    rec[0, 0, 0, 15] = k
    l1 = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    act = ttnn.from_torch(act_t, dtype=ttnn.bfloat16, **l1)
    route = ttnn.from_torch(rec, dtype=ttnn.uint16, **l1)
    a = act_t.float()[0, 0]
    routed = sum(a[1 + j] @ w_dn[0, e] for j, e in enumerate(ids))
    entries = dev.num_program_cache_entries()
    for use_shared in (shared_first, not shared_first):
        op = ExpertDown(dev, H, inter, E, k, ttnn.bfloat8_b, shared=use_shared)
        out = op(act, route, down, op.allocate_output(), shared_down=shared if use_shared else None)
        got = ttnn.to_torch(out).float().reshape(-1)
        ref = routed + a[0] @ w_sdn[0, 0] if use_shared else routed
        assert _pcc(got, ref) > 0.999, (use_shared, _pcc(got, ref))
    assert dev.num_program_cache_entries() == entries + 2


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize("bad", ["missing_shared", "down_dtype", "out_shape"])
def test_expert_down_rejects_mismatched_inputs(ttnn_mesh_device, bad):
    dev, H, inter, E, k = ttnn_mesh_device, 2048, 128, 64, 4
    op = ExpertDown(dev, H, inter, E, k, ttnn.bfloat8_b)
    down, _ = _upload(torch.zeros(1, E, inter, H), ttnn.bfloat4_b if bad == "down_dtype" else ttnn.bfloat8_b, dev)
    shared, _ = _upload(torch.zeros(1, 1, inter, H), ttnn.bfloat8_b, dev)
    act = ttnn.from_torch(
        torch.zeros(1, 1, k + 1, inter), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    route = ttnn.from_torch(torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint16, device=dev)
    out = (
        ttnn.from_torch(torch.zeros(1, 1, 1, H // 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        if bad == "out_shape"
        else op.allocate_output()
    )
    with pytest.raises(ValueError):
        op(act, route, down, out, shared_down=None if bad == "missing_shared" else shared)
