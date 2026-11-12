# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert-sorted dispatch ops (ops/moe/sorted_dispatch.py): the dispatch layout matches a stable counting sort, the
gather copies each group's token rows, and the un-dispatch sums each token's slots, on device against torch."""

import pytest
import torch
import ttnn

from tt_transformers.ops.moe.sorted_dispatch import (
    GROUP,
    PAD_GROUP_ID,
    dispatch_rows,
    gather_add_rows,
    gather_rows,
    sorted_dispatch,
)

MESH_1X1 = pytest.mark.parametrize(
    "ttnn_mesh_device",
    [pytest.param({"mesh_shape": (1, 1)}, id="p150-1x1")],
    indirect=True,
)


def _routing(tokens, k, experts, local, seed):
    """Top-k ids of `experts` experts mapped to this device's window [0, local) (others -> local, routed elsewhere),
    and softmax scores."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=g)[:k] for _ in range(tokens)])
    ids = torch.where(ids < local, ids, torch.full_like(ids, local))
    scores = torch.softmax(torch.randn(tokens, k, generator=g), -1).bfloat16()
    return ids, scores


def _reference_dispatch(ids, scores, experts, rows):
    tokens, k = ids.shape
    groups = rows // GROUP
    src_tok, weights = torch.zeros(rows, dtype=torch.int64), torch.zeros(rows, dtype=torch.bfloat16)
    group_ids, undispatch = torch.full((groups,), PAD_GROUP_ID), torch.zeros(tokens * k, dtype=torch.int64)
    counts = torch.bincount(ids[ids < experts].flatten(), minlength=experts)
    offsets = (torch.cumsum(-(-counts // GROUP) * GROUP, 0) - -(-counts // GROUP) * GROUP).tolist()
    for e in range(experts):
        first = offsets[e] // GROUP
        group_ids[first : first + -(-int(counts[e]) // GROUP)] = e
    for i, (t, j) in enumerate((t, j) for t in range(tokens) for j in range(k)):
        e = int(ids[t, j])
        if e < experts:
            row, offsets[e] = offsets[e], offsets[e] + 1
            src_tok[row], weights[row] = t, scores[t, j]
        else:
            row = rows - GROUP
        undispatch[i] = row
    return src_tok, weights, group_ids, undispatch


def _tile(t, dtype, dev):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _row_major(t, dtype, dev):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _host(t, dtype=torch.int64):
    return ttnn.to_torch(t).reshape(-1).to(dtype)


@pytest.mark.host
def test_dispatch_rows_covers_the_worst_case_padding():
    tokens, k, experts = 100, 8, 32
    # every pair, each expert's last group with 31 padding rows, the trailing padding group
    assert dispatch_rows(tokens, k, experts) >= tokens * k + 31 * experts + GROUP
    assert dispatch_rows(tokens, k, experts) % GROUP == 0


@pytest.mark.parametrize("tokens", [1, 48, 300])
@pytest.mark.parametrize("ids_dtype", [ttnn.uint16, ttnn.uint32], ids=["uint16", "uint32"])
@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
def test_sorted_dispatch_matches_a_stable_counting_sort(ttnn_mesh_device, tokens, ids_dtype):
    dev, k, experts, local = ttnn_mesh_device, 8, 128, 32
    ids, scores = _routing(tokens, k, experts, local, seed=tokens)
    rows = dispatch_rows(tokens, k, local)
    out = sorted_dispatch(_tile(ids, ids_dtype, dev), _tile(scores, ttnn.bfloat16, dev), local, rows=rows)
    src_tok, weights, group_ids, undispatch = _reference_dispatch(ids, scores, local, rows)
    assert torch.equal(_host(out[0]), src_tok)
    assert torch.equal(_host(out[1], torch.bfloat16), weights)
    assert torch.equal(_host(out[2]), group_ids)
    assert torch.equal(_host(out[3]), undispatch)


@pytest.mark.parametrize("tokens, hidden", [(48, 2048), (300, 768)])
@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
def test_gather_rows_copies_each_groups_token_rows(ttnn_mesh_device, tokens, hidden):
    dev, k, experts, local = ttnn_mesh_device, 8, 128, 32
    ids, scores = _routing(tokens, k, experts, local, seed=1)
    rows = dispatch_rows(tokens, k, local)
    src_tok, _, group_ids, _ = _reference_dispatch(ids, scores, local, rows)
    x = torch.randn(tokens, hidden).bfloat16()
    out = gather_rows(
        _row_major(x, ttnn.bfloat16, dev),
        _row_major(src_tok.reshape(1, -1).int(), ttnn.uint32, dev),
        _row_major(group_ids.reshape(1, -1).int(), ttnn.uint16, dev),
        local,
    )
    got = ttnn.to_torch(out).reshape(-1, GROUP, hidden)
    real = (group_ids < local).nonzero().flatten()
    assert torch.equal(got[real], x[src_tok].reshape(-1, GROUP, hidden)[real])


@pytest.mark.parametrize("tokens, hidden", [(1, 2048), (48, 2048), (300, 768)])
@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
def test_gather_add_rows_sums_each_tokens_slots(ttnn_mesh_device, tokens, hidden):
    dev, k, experts, local = ttnn_mesh_device, 8, 128, 32
    ids, scores = _routing(tokens, k, experts, local, seed=2)
    rows = dispatch_rows(tokens, k, local)
    *_, undispatch = _reference_dispatch(ids, scores, local, rows)
    expert_rows = torch.randn(rows, hidden).bfloat16()  # the trailing group, never written, holds garbage
    out = gather_add_rows(
        _row_major(expert_rows, ttnn.bfloat16, dev),
        _row_major(undispatch.reshape(1, -1).int(), ttnn.uint32, dev),
        tokens,
        k,
    )
    routed_here = (ids < local).reshape(-1, 1)
    reference = (expert_rows[undispatch].float() * routed_here).reshape(tokens, k, hidden).sum(1)
    got = ttnn.to_torch(out).float().reshape(tokens, hidden)
    assert torch.allclose(got, reference, atol=0.1, rtol=0.02), (got - reference).abs().max()


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
def test_dispatch_gather_and_undispatch_round_trip(ttnn_mesh_device):
    """Each dispatched row scaled by its routing weight and summed back gives x times the token's local weight."""
    dev, tokens, hidden, k, experts, local = ttnn_mesh_device, 200, 2048, 8, 128, 32
    ids, scores = _routing(tokens, k, experts, local, seed=3)
    rows = dispatch_rows(tokens, k, local)
    x = torch.randn(tokens, hidden).bfloat16()
    src_tok, weights, group_ids, undispatch = sorted_dispatch(
        _tile(ids, ttnn.uint16, dev), _tile(scores, ttnn.bfloat16, dev), local, rows=rows
    )
    gathered = ttnn.to_torch(gather_rows(_row_major(x, ttnn.bfloat16, dev), src_tok, group_ids, local))
    padding = (_host(group_ids) >= local).repeat_interleave(GROUP)
    weighted = gathered.reshape(rows, hidden).float() * _host(weights, torch.float32)[:, None]
    weighted[padding] = 7.0  # padding rows are never read back
    out = gather_add_rows(_row_major(weighted.bfloat16(), ttnn.bfloat16, dev), undispatch, tokens, k)
    local_weight = torch.where(ids < local, scores.float(), torch.zeros(())).sum(-1, keepdim=True)
    reference = x.float() * local_weight
    got = ttnn.to_torch(out).float().reshape(tokens, hidden)
    assert torch.allclose(got, reference, atol=0.05, rtol=0.02), (got - reference).abs().max()


@pytest.mark.device
@pytest.mark.blackhole
@MESH_1X1
def test_ops_read_fresh_buffers_on_program_cache_hits(ttnn_mesh_device, request):
    """The second call reuses every program with newly allocated inputs and outputs."""
    dev, tokens, hidden, k, experts, local = ttnn_mesh_device, 64, 2048, 8, 128, 32
    dev.enable_program_cache()
    dev.clear_program_cache()
    request.addfinalizer(dev.disable_and_clear_program_cache)
    rows = dispatch_rows(tokens, k, local)
    for call, seed in enumerate((4, 5)):
        ids, scores = _routing(tokens, k, experts, local, seed)
        x = torch.randn(tokens, hidden).bfloat16()
        src_tok, _, group_ids, undispatch = sorted_dispatch(
            _tile(ids, ttnn.uint16, dev), _tile(scores, ttnn.bfloat16, dev), local, rows=rows
        )
        expected = _reference_dispatch(ids, scores, local, rows)
        assert torch.equal(_host(src_tok), expected[0]) and torch.equal(_host(undispatch), expected[3]), call
        gathered = ttnn.to_torch(gather_rows(_row_major(x, ttnn.bfloat16, dev), src_tok, group_ids, local))
        real = (expected[2] < local).nonzero().flatten()
        assert torch.equal(gathered.reshape(-1, GROUP, hidden)[real], x[expected[0]].reshape(-1, GROUP, hidden)[real])
        expert_rows = torch.randn(rows, hidden).bfloat16()
        out = gather_add_rows(_row_major(expert_rows, ttnn.bfloat16, dev), undispatch, tokens, k)
        routed_here = (ids < local).reshape(-1, 1)
        reference = (expert_rows[expected[3]].float() * routed_here).reshape(tokens, k, hidden).sum(1)
        assert torch.allclose(ttnn.to_torch(out).float().reshape(tokens, hidden), reference, atol=0.1, rtol=0.02)
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
def test_ops_replay_a_trace_with_new_routing(ttnn_mesh_device):
    """One captured dispatch -> gather -> un-dispatch serves any routing written into its inputs."""
    dev, tokens, hidden, k, experts, local = ttnn_mesh_device, 96, 2048, 8, 128, 32
    rows = dispatch_rows(tokens, k, local)

    def host(ids, scores, x, expert_rows):
        return (
            ttnn.from_torch(ids, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.from_torch(expert_rows, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        )

    ids, scores = _routing(tokens, k, experts, local, seed=6)
    inputs = [
        t.to(dev, ttnn.DRAM_MEMORY_CONFIG)
        for t in host(ids, scores, torch.zeros(tokens, hidden), torch.zeros(rows, hidden))
    ]
    ids_tt, scores_tt, x_tt, expert_rows_tt = inputs

    def chain():
        src_tok, weights, group_ids, undispatch = sorted_dispatch(ids_tt, scores_tt, local, rows=rows)
        gathered = gather_rows(x_tt, src_tok, group_ids, local)
        summed = gather_add_rows(expert_rows_tt, undispatch, tokens, k)
        return src_tok, weights, group_ids, undispatch, gathered, summed

    chain()  # compile before capture
    trace = ttnn.begin_trace_capture(dev, cq_id=0)
    outputs = chain()
    ttnn.end_trace_capture(dev, trace, cq_id=0)
    try:
        for seed in (7, 8):
            ids, scores = _routing(tokens, k, experts, local, seed)
            x, expert_rows = torch.randn(tokens, hidden).bfloat16(), torch.randn(rows, hidden).bfloat16()
            for source, target in zip(host(ids, scores, x, expert_rows), inputs):
                ttnn.copy_host_to_device_tensor(source, target)
            ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
            src_tok, weights, group_ids, undispatch = _reference_dispatch(ids, scores, local, rows)
            assert torch.equal(_host(outputs[0]), src_tok) and torch.equal(_host(outputs[2]), group_ids), seed
            assert torch.equal(_host(outputs[1], torch.bfloat16), weights) and torch.equal(
                _host(outputs[3]), undispatch
            )
            real = (group_ids < local).nonzero().flatten()
            gathered = ttnn.to_torch(outputs[4]).reshape(-1, GROUP, hidden)
            assert torch.equal(gathered[real], x[src_tok].reshape(-1, GROUP, hidden)[real]), seed
            routed_here = (ids < local).reshape(-1, 1)
            reference = (expert_rows[undispatch].float() * routed_here).reshape(tokens, k, hidden).sum(1)
            got = ttnn.to_torch(outputs[5]).float().reshape(tokens, hidden)
            assert torch.allclose(got, reference, atol=0.1, rtol=0.02), seed
    finally:
        ttnn.release_trace(dev, trace)
