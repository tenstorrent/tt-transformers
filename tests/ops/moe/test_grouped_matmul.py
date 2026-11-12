# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GroupedExpertMatmul (ops/moe/grouped_matmul.py): the weights stay resident when they fit L1 and stream otherwise,
and on device every real group matches torch, padding groups are skipped, and program-cache hits read fresh routing."""

import pytest
import torch
import ttnn

from tt_transformers.ops.moe.grouped_matmul import GroupedExpertMatmul

MESH_1X1 = pytest.mark.parametrize(
    "ttnn_mesh_device",
    [pytest.param({"mesh_shape": (1, 1)}, id="p150-1x1")],
    indirect=True,
)
PAD = 0xFFFF


def _config(fidelity=ttnn.MathFidelity.LoFi, fp32=False, l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE, math_fidelity=fidelity, fp32_dest_acc_en=fp32, packer_l1_acc=l1_acc
    )


@pytest.mark.host
def test_qwen3_coder_gate_up_streams_and_down_stays_resident():
    gate_up = GroupedExpertMatmul(2048, 1536, 32, ttnn.bfloat4_b, _config())
    down = GroupedExpertMatmul(768, 2048, 32, ttnn.bfloat4_b, _config(), in0_block_w=4, untilize_out=True)
    assert not gate_up.resident and gate_up.in0_block_w == 16
    assert down.resident and down.untilize_out and down.untilize_chunk_w == 16
    with pytest.raises(ValueError, match="resident"):
        GroupedExpertMatmul(2048, 1536, 32, ttnn.bfloat4_b, _config(), untilize_out=True)


def _group_ids(experts, groups, seed):
    """Expert-sorted groups as the dispatch lays them out: runs of one expert, some experts absent, padding last."""
    g = torch.Generator().manual_seed(seed)
    runs = torch.randint(0, 4, (experts,), generator=g)
    ids = torch.repeat_interleave(torch.arange(experts), runs)[: groups - 1]
    return torch.cat([ids, torch.full((groups - len(ids),), PAD)])


def _run(dev, op, x, weights_tt, ids):
    ids_tt = ttnn.from_torch(
        ids.reshape(1, -1).int(),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return ttnn.to_torch(op(x_tt, weights_tt, ids_tt)).float()


def _check(out, x, weights, ids, experts):
    for group, expert in enumerate(ids.tolist()):
        if expert < experts:
            reference = x[0, group].float() @ weights[0, expert]
            pcc = torch.corrcoef(torch.stack([out[0, group].flatten(), reference.flatten()]))[0, 1].item()
            assert pcc > 0.995, (group, expert, pcc)


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize(
    "k, n, dtype, untilize_out, accumulate",
    [
        (2048, 1536, ttnn.bfloat4_b, False, "dst"),
        (768, 2048, ttnn.bfloat4_b, True, "dst"),
        (768, 2048, ttnn.bfloat8_b, False, "dst"),
        (2048, 1536, ttnn.bfloat8_b, False, "fp32"),
        (768, 2048, ttnn.bfloat4_b, True, "fp32"),
        (2048, 1536, ttnn.bfloat4_b, False, "l1"),
        (768, 2048, ttnn.bfloat4_b, True, "l1"),
    ],
    ids=[
        "streamed",
        "resident-row-major",
        "resident-tile",
        "streamed-fp32",
        "resident-row-major-fp32",
        "streamed-l1-acc",
        "resident-row-major-l1-acc",
    ],
)
def test_grouped_matmul_matches_torch(ttnn_mesh_device, k, n, dtype, untilize_out, accumulate):
    dev, experts, groups = ttnn_mesh_device, 32, 48
    config = {
        "dst": _config(),
        "fp32": _config(ttnn.MathFidelity.HiFi2, fp32=True),
        "l1": _config(l1_acc=True),
    }[accumulate]
    op = GroupedExpertMatmul(k, n, experts, dtype, config, in0_block_w=4 if k == 768 else 16, untilize_out=untilize_out)
    weights_tt = ttnn.from_torch(
        torch.randn(1, experts, k, n) * 0.03,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weights = ttnn.to_torch(weights_tt).float()
    ids = _group_ids(experts, groups, seed=k)
    x = torch.randn(1, groups, 32, k).bfloat16()
    _check(_run(dev, op, x, weights_tt, ids), x, weights, ids, experts)


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
@pytest.mark.parametrize("untilize_out", [False, True], ids=["tile", "row-major"])
def test_grouped_matmul_reads_fresh_routing_on_program_cache_hits(ttnn_mesh_device, request, untilize_out):
    dev, k, n, experts, groups = ttnn_mesh_device, 768, 2048, 32, 40
    dev.enable_program_cache()
    dev.clear_program_cache()
    request.addfinalizer(dev.disable_and_clear_program_cache)
    op = GroupedExpertMatmul(k, n, experts, ttnn.bfloat4_b, _config(), in0_block_w=4, untilize_out=untilize_out)
    for call in range(2):
        weights_tt = ttnn.from_torch(
            torch.randn(1, experts, k, n) * 0.03,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ids = _group_ids(experts, groups, seed=10 + call)
        x = torch.randn(1, groups, 32, k).bfloat16()
        _check(_run(dev, op, x, weights_tt, ids), x, ttnn.to_torch(weights_tt).float(), ids, experts)
        if call == 0:
            entries = dev.num_program_cache_entries()
    assert entries > 0 and dev.num_program_cache_entries() == entries


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [pytest.param({"mesh_shape": (1, 1), "trace_region_size": 8 << 20}, id="p150-1x1")],
    indirect=True,
)
@pytest.mark.parametrize(
    "k, n, untilize_out", [(2048, 1536, False), (768, 2048, True)], ids=["streamed", "resident-row-major"]
)
def test_grouped_matmul_replays_a_trace_with_new_routing(ttnn_mesh_device, k, n, untilize_out):
    """One captured program serves any routing: new activations and group ids written into its inputs."""
    dev, experts, groups = ttnn_mesh_device, 32, 40
    op = GroupedExpertMatmul(
        k, n, experts, ttnn.bfloat4_b, _config(), in0_block_w=4 if k == 768 else 16, untilize_out=untilize_out
    )
    weights_tt = ttnn.from_torch(
        torch.randn(1, experts, k, n) * 0.03,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weights = ttnn.to_torch(weights_tt).float()

    def host(x, ids):
        return (
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(ids.reshape(1, -1).int(), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT),
        )

    x_tt, ids_tt = (
        t.to(dev, ttnn.DRAM_MEMORY_CONFIG) for t in host(torch.zeros(1, groups, 32, k), _group_ids(experts, groups, 0))
    )
    op(x_tt, weights_tt, ids_tt)  # compile before capture
    trace = ttnn.begin_trace_capture(dev, cq_id=0)
    out = op(x_tt, weights_tt, ids_tt)
    ttnn.end_trace_capture(dev, trace, cq_id=0)
    try:
        for seed in (20, 21):
            x, ids = torch.randn(1, groups, 32, k).bfloat16(), _group_ids(experts, groups, seed)
            for source, target in zip(host(x, ids), (x_tt, ids_tt)):
                ttnn.copy_host_to_device_tensor(source, target)
            ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
            _check(ttnn.to_torch(out).float(), x, weights, ids, experts)
    finally:
        ttnn.release_trace(dev, trace)
