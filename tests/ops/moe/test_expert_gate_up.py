# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ExpertGateUp (ops/moe/expert_gate_up.py): its kernels ship with the package, its column layout covers every pair
once, it rejects unsupported configurations, and on device it routes exactly (top-k ids, softmax and shared-gate
weights) and writes w_s * silu(gate) * up per slot as torch computes it from the same weights."""

import re
from pathlib import Path

import pytest
import torch
import ttnn

from tt_transformers.ops.moe import _program
from tt_transformers.ops.moe.expert_gate_up import ExpertGateUp, ep_offsets, gu_layout, ring_blocks

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
def test_kernels_and_headers_ship_and_are_covered_by_package_data():
    globs = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"][
        "tt_transformers.ops.moe"
    ]
    packaged = {path for glob in globs for path in _program.KERNEL_DIR.parent.glob(glob)}
    for name in ("gu_reader.cpp", "gu_writer.cpp", "gu_compute.cpp"):
        kernel = _program.KERNEL_DIR / name
        assert kernel in packaged, name
        for include in re.findall(r'#include "([^"]+)"', kernel.read_text()):
            if (kernel.parent / include).exists():
                assert (kernel.parent / include) in packaged, f"{name}: {include}"
            else:
                assert include.startswith(("api/", "ckernel")), f"{name}: {include}"


@pytest.mark.host
@pytest.mark.parametrize("m", [1, 2, 3, 4, 6])
def test_gate_up_layout_covers_every_pair_once(m):
    banks = 8
    _, nb, groups, pairs = gu_layout(banks * m, banks)
    assert nb == (2 if m % 2 else 1) and groups == banks // nb
    assert sorted(p for g in pairs for p in g) == list(range(banks * m // 2))


@pytest.mark.host
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(weight_dtype=ttnn.bfloat16),
        dict(num_experts=96),
        dict(k=16),
        dict(k=0),
        dict(inter=100),
        dict(hidden=32),  # custom_mm: kt in [2, 256] and even
        dict(hidden=2080),
        dict(hidden=8256),
        dict(ep_experts=32),  # expert parallelism without shared=False
        dict(shared=False, ep_experts=48),
    ],
)
def test_rejects_unsupported_configuration(kwargs):
    args = dict(hidden=2048, inter=128, num_experts=256, k=8, weight_dtype=ttnn.bfloat8_b) | kwargs
    with pytest.raises(ValueError):
        ExpertGateUp(None, **args)


@pytest.mark.host
def test_rejects_a_weight_ring_that_outgrows_l1():
    assert ring_blocks(64, 6, 576) == 3
    with pytest.raises(ValueError):
        ring_blocks(256, 2, 1088)  # hidden 8192, m 2, bfp8: 3 blocks are 1.6 MiB


def _round_sig(t, bits=5):
    """Round to `bits` significant bits, as the model stores expert weights for the LoFi matmul."""
    drop = 24 - bits
    i = t.float().contiguous().view(torch.int32)
    return ((i + (1 << (drop - 1))) & ~((1 << drop) - 1)).view(torch.float32)


def _weights(dev, shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    t = ttnn.from_torch(
        _round_sig(torch.randn(*shape, generator=g) * 0.03),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return t, ttnn.to_torch(t).float()


def _l1(t, dev):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def _bf16_bits(vals):
    return ((vals.to(torch.int32) + 32768) % 65536 - 32768).to(torch.int16).view(torch.bfloat16).float()


def _swiglu(x, w, weight, inter, gate_first):
    a, b = x @ w[:, :inter], x @ w[:, inter:]
    gate, up = (a, b) if gate_first else (b, a)
    return weight * torch.nn.functional.silu(gate) * up


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def _top_k(row, k):
    return sorted(range(row.numel()), key=lambda i: (-row[i].item(), i))[:k]


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize(
    "E, inter, dtype",
    [(256, 128, ttnn.bfloat8_b), (64, 384, ttnn.bfloat4_b), (64, 512, ttnn.bfloat4_b), (64, 768, ttnn.bfloat4_b)],
    ids=["m1_bfp8", "m3_bfp4", "m4_bfp4", "m6_bfp4"],
)
def test_shared_expert_gate_up_matches_torch(ttnn_mesh_device, E, inter, dtype):
    dev, H, k = ttnn_mesh_device, 2048, 8
    up_gate, W = _weights(dev, (1, E, H, 2 * inter), dtype, seed=1)
    shared, Ws = _weights(dev, (1, 1, H, 2 * inter), dtype, seed=2)
    op = ExpertGateUp(dev, H, inter, E, k, dtype)
    act, route = op.allocate_outputs()
    for trial in range(3):
        torch.manual_seed(trial)
        x = torch.zeros(1, 1, 32, H)
        x[0, 0, 0] = torch.randn(H)
        lg = torch.zeros(1, 1, 32, E + 32)
        lg[0, 0, 0, : E + 1] = torch.randn(E + 1) * 2
        if trial == 2:  # exact ties at the top and at the edge of the top-k: lower id first
            row = lg[0, 0, 0, :E]
            row[5] = row[E - 3] = row.max()
            kth = row.topk(k).values[-1]
            row[2] = row[E - 1] = kth
        xb, lgb = x.bfloat16(), lg.bfloat16()
        x_tt, lg_tt = _l1(xb, dev), _l1(lgb, dev)
        op(x_tt, lg_tt, up_gate, act, route, shared_up_gate=shared)

        row = lgb[0, 0, 0, :E].float()
        ids = _top_k(row, k)
        ref_w = torch.softmax(row[ids], -1)
        ref_ws = torch.sigmoid(lgb[0, 0, 0, E].float())
        rec = ttnn.to_torch(route).reshape(-1).long()
        assert rec[:k].tolist() == ids, (trial, rec[:k].tolist(), ids)
        assert not rec[k:15].any() and not rec[17 + k :].any(), trial
        got_w = _bf16_bits(rec[16 : 17 + k])
        assert torch.allclose(got_w, torch.cat([ref_ws.reshape(1), ref_w]), rtol=1e-2, atol=1e-3), trial

        got = ttnn.to_torch(act).float()[0, 0]
        xf = xb[0, 0, 0].float()
        ref = torch.stack(
            [_swiglu(xf, Ws[0, 0], got_w[0], inter, False)]
            + [_swiglu(xf, W[0, e], got_w[1 + j], inter, False) for j, e in enumerate(ids)]
        )
        for s in range(k + 1):
            assert _pcc(got[s], ref[s]) > 0.999, (trial, s, _pcc(got[s], ref[s]))
        ttnn.deallocate(x_tt)
        ttnn.deallocate(lg_tt)


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
def test_expert_parallel_window_matches_torch(ttnn_mesh_device):
    """Qwen3-Coder-30B-A3B shapes on one die of four: experts [32, 64) of 128, [gate | up] packing, no shared expert.
    Trial 1 routes nothing to the window, trial 2 the whole top-k."""
    dev, H, inter, E, local, off, k = ttnn_mesh_device, 2048, 768, 128, 32, 32, 8
    up_gate, W = _weights(dev, (1, local, H, 2 * inter), ttnn.bfloat4_b, seed=7)
    offset = ttnn.from_torch(
        torch.tensor([off] + [0] * 31, dtype=torch.int32).reshape(1, 1, 1, 32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    op = ExpertGateUp(dev, H, inter, E, k, ttnn.bfloat4_b, shared=False, ep_experts=local, gate_first=True)
    act, route = op.allocate_outputs()
    for trial in range(3):
        torch.manual_seed(50 + trial)
        x = torch.zeros(1, 1, 32, H)
        x[0, 0, 0] = torch.randn(H)
        lg = torch.zeros(1, 1, 32, E + 32)
        lg[0, 0, 0, :E] = torch.randn(E) * 2
        if trial == 1:
            lg[0, 0, 0, off : off + local] -= 20
        if trial == 2:
            lg[0, 0, 0, off : off + local] += 20
        xb, lgb = x.bfloat16(), lg.bfloat16()
        x_tt, lg_tt = _l1(xb, dev), _l1(lgb, dev)
        op(x_tt, lg_tt, up_gate, act, route, ep_offset=offset)

        row = lgb[0, 0, 0, :E].float()
        ids = _top_k(row, k)
        weights = torch.softmax(row[ids], -1)
        live = [(j, e - off) for j, e in enumerate(ids) if off <= e < off + local]
        assert len(live) == {1: 0, 2: k}.get(trial, len(live))
        rec = ttnn.to_torch(route).reshape(-1).long()
        assert rec[15].item() == len(live), (trial, rec[15].item(), len(live))
        assert rec[: len(live)].tolist() == [le for _, le in live], trial
        assert not rec[len(live) : 15].any() and not rec[17 + len(live) :].any(), trial

        got = ttnn.to_torch(act).float()[0, 0]
        xf = xb[0, 0, 0].float()
        for s, (j, le) in enumerate(live, start=1):
            ref = _swiglu(xf, W[0, le], weights[j], inter, True)
            assert _pcc(got[s], ref) > 0.999, (trial, s, _pcc(got[s], ref))
        ttnn.deallocate(x_tt)
        ttnn.deallocate(lg_tt)
    ttnn.deallocate(offset)


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize("shared_first", [True, False], ids=["shared_first", "routed_first"])
def test_shared_and_routed_only_take_separate_cache_entries(ttnn_mesh_device, request, shared_first):
    """shared only adds runtime args on the slot-0 cores, so it must key the program cache."""
    dev, H, inter, E, k = ttnn_mesh_device, 2048, 128, 64, 4
    dev.enable_program_cache()
    dev.clear_program_cache()
    request.addfinalizer(dev.disable_and_clear_program_cache)
    up_gate, W = _weights(dev, (1, E, H, 2 * inter), ttnn.bfloat8_b, seed=3)
    shared, Ws = _weights(dev, (1, 1, H, 2 * inter), ttnn.bfloat8_b, seed=4)
    torch.manual_seed(0)
    x = torch.zeros(1, 1, 32, H)
    x[0, 0, 0] = torch.randn(H)
    lg = torch.zeros(1, 1, 32, E + 32)
    lg[0, 0, 0, : E + 1] = torch.randn(E + 1) * 2
    xb, lgb = x.bfloat16(), lg.bfloat16()
    x_tt, lg_tt = _l1(xb, dev), _l1(lgb, dev)
    xf, ids = xb[0, 0, 0].float(), _top_k(lgb[0, 0, 0, :E].float(), k)
    entries = dev.num_program_cache_entries()
    for use_shared in (shared_first, not shared_first):
        op = ExpertGateUp(dev, H, inter, E, k, ttnn.bfloat8_b, shared=use_shared)
        act, route = op.allocate_outputs()
        op(x_tt, lg_tt, up_gate, act, route, shared_up_gate=shared if use_shared else None)
        rec = ttnn.to_torch(route).reshape(-1).long()
        assert rec[:k].tolist() == ids, use_shared
        w = _bf16_bits(rec[16 : 17 + k])
        got = ttnn.to_torch(act).float()[0, 0]
        ref = {s: _swiglu(xf, W[0, e], w[s], inter, False) for s, e in enumerate(ids, start=1)}
        if use_shared:
            ref[0] = _swiglu(xf, Ws[0, 0], w[0], inter, False)
        for s, r in ref.items():
            assert _pcc(got[s], r) > 0.999, (use_shared, s, _pcc(got[s], r))
    assert dev.num_program_cache_entries() == entries + 2
    ttnn.deallocate(x_tt)
    ttnn.deallocate(lg_tt)


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
def test_ep_offsets_hold_each_dies_first_expert(ttnn_mesh_device):
    offsets = ep_offsets(ttnn_mesh_device, 32)
    assert ttnn.to_torch(offsets).reshape(-1)[0].item() == 0


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize("bad", ["x_dram", "logits_width", "logits_bfp8", "missing_shared"])
def test_rejects_mismatched_inputs(ttnn_mesh_device, bad):
    dev, H, inter, E, k = ttnn_mesh_device, 2048, 128, 64, 4
    up_gate, _ = _weights(dev, (1, E, H, 2 * inter), ttnn.bfloat8_b, seed=0)
    shared, _ = _weights(dev, (1, 1, H, 2 * inter), ttnn.bfloat8_b, seed=0)
    op = ExpertGateUp(dev, H, inter, E, k, ttnn.bfloat8_b)
    act, route = op.allocate_outputs()
    zeros = torch.zeros(1, 1, 32, H).bfloat16()
    x = (
        ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        if bad == "x_dram"
        else _l1(zeros, dev)
    )
    width = E if bad == "logits_width" else E + 32
    logits = ttnn.from_torch(
        torch.zeros(1, 1, 32, width),
        dtype=ttnn.bfloat8_b if bad == "logits_bfp8" else ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError):
        op(x, logits, up_gate, act, route, shared_up_gate=None if bad == "missing_shared" else shared)
