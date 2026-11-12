# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device grammar sampling on N150 and N300: Sampling1D with synthetic packed masks.

The masks are hand-built allowed-token sets; no grammar engine is involved. The
allowed tokens sit at both sides of the 1x2 vocab shard boundary and at the last
real token, so a mask word unpacked on the wrong shard shows up as a wrong token.
"""

import pytest
import torch
import ttnn
from examples.common.auto_compose import to_torch_auto_compose

from tt_transformers.modules.sampling.sampling_1d import Sampling1D
from tt_transformers.sampling.grammar_bitmask import packed_grammar_width

pytestmark = pytest.mark.usefixtures("skip_on_galaxy_system")

_TRACE_REGION_SIZE = 32 << 20
_B = 32
_VOCAB = 128256  # Llama-class vocab: 2004 mask words per device on 1x2
_HOT = 42  # the unconstrained winner

_MESHES = pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        {"mesh_shape": (1, 1), "trace_region_size": _TRACE_REGION_SIZE},
        {"mesh_shape": (1, 2), "trace_region_size": _TRACE_REGION_SIZE},
    ],
    ids=["1x1", "1x2"],
    indirect=True,
)


def _boundary_tokens():
    """Last token of shard 0, first of shard 1, a mid-vocab token, and the last real token."""
    half = _VOCAB // 2
    return half - 1, half, 777, _VOCAB - 1


def _hot_logits():
    torch.manual_seed(0)
    logits = torch.randn(1, 1, _B, _VOCAB, dtype=torch.float32) * 0.1
    shard0_tail, shard1_head, middle, last = _boundary_tokens()
    for token, value in ((_HOT, 10.0), (shard0_tail, 8.0), (shard1_head, 7.0), (middle, 6.0), (last, 5.0)):
        logits[..., token] = value
    return logits.to(torch.bfloat16)


def _mask(allowed: dict[int, list[int]], rows: int = _B) -> torch.Tensor:
    width = packed_grammar_width(_VOCAB)
    mask = torch.full((rows, width), -1, dtype=torch.int32)
    for row, token_ids in allowed.items():
        words = torch.zeros(width, dtype=torch.int64)
        for token_id in token_ids:
            words[token_id // 32] |= 1 << (token_id % 32)
        mask[row] = words.to(torch.int32)
    return mask


def _expected(logits: torch.Tensor, allowed: dict[int, list[int]]) -> list[int]:
    out = []
    for row in range(_B):
        tokens = allowed.get(row)
        if tokens is None:
            out.append(int(logits[0, 0, row].float().argmax()))
        else:
            out.append(max(tokens, key=lambda t: float(logits[0, 0, row, t])))
    return out


def _logits_tt(logits, mesh_device):
    cluster_shape = tuple(mesh_device.shape)
    dims = (None, None) if max(cluster_shape) == 1 else (None, -1)
    return ttnn.from_torch(
        logits,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=cluster_shape),
    )


def _replicated(mesh_device, values, dtype):
    cluster_shape = tuple(mesh_device.shape)
    return ttnn.from_torch(
        values,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=cluster_shape),
    )


def _kpt(mesh_device, *, k, p, temp):
    return (
        _replicated(mesh_device, torch.full((_B,), k, dtype=torch.int32), ttnn.uint32),
        _replicated(mesh_device, torch.full((_B,), p), ttnn.bfloat16),
        _replicated(mesh_device, torch.full((_B,), temp), ttnn.bfloat16),
    )


def _sampler(mesh_device, mode):
    sampler = Sampling1D(
        vocab_size=_VOCAB,
        mesh_device=mesh_device,
        max_batch_size=_B,
        allow_force_argmax=True,
        pad_to_power_of_2=(mode == "topk" and mesh_device.get_num_devices() > 1),
    )
    sampler.load_device_buffers()
    sampler.enable_device_grammar()
    return sampler


def _tokens(tokens_tt):
    return to_torch_auto_compose(tokens_tt).flatten()[:_B].long().tolist()


# Rows 0/1/15 allow one token each; row 16 allows two; row 31 allows both boundary tokens.
def _first_constraints():
    shard0_tail, shard1_head, middle, last = _boundary_tokens()
    return {0: [shard0_tail], 1: [shard1_head], 15: [last], 16: [middle, last], 31: [shard1_head, shard0_tail]}


def _second_constraints():
    shard0_tail, shard1_head, _middle, last = _boundary_tokens()
    return {1: [last], 2: [shard0_tail], 30: [shard1_head]}


@pytest.mark.device
@_MESHES
@pytest.mark.parametrize("mode", ["argmax", "topk"])
def test_eager_mask_selects_allowed_tokens_and_keeps_logits(ttnn_mesh_device, mode):
    mesh_device = ttnn_mesh_device
    sampler = _sampler(mesh_device, mode)
    logits = _hot_logits()
    logits_tt = _logits_tt(logits, mesh_device)
    params = {} if mode == "argmax" else dict(zip(("k", "p", "temp"), _kpt(mesh_device, k=1, p=0.0, temp=1.0)))
    allowed = _first_constraints()

    sampler.update_grammar_bitmask(_mask(allowed))
    tokens_tt, _ = sampler.decode_forward(logits_tt, apply_grammar=True, **params)

    assert _tokens(tokens_tt) == _expected(logits, allowed)
    # Masking is out of place: the caller's logits are unchanged.
    after = to_torch_auto_compose(logits_tt).reshape(1, 1, _B, _VOCAB)
    assert torch.equal(after, logits)


@pytest.mark.device
@_MESHES
def test_eager_turnover_clears_stale_rows(ttnn_mesh_device):
    """Constrained, then all-allowed (None), then a different constraint set."""
    mesh_device = ttnn_mesh_device
    sampler = _sampler(mesh_device, "argmax")
    logits = _hot_logits()
    logits_tt = _logits_tt(logits, mesh_device)

    for allowed in (_first_constraints(), None, _second_constraints()):
        sampler.update_grammar_bitmask(None if allowed is None else _mask(allowed))
        tokens_tt, _ = sampler.decode_forward(logits_tt, apply_grammar=True)
        assert _tokens(tokens_tt) == _expected(logits, allowed or {})


@pytest.mark.device
@_MESHES
@pytest.mark.parametrize("mode", ["argmax", "topk"])
def test_traced_mask_updates_between_replays(ttnn_mesh_device, mode):
    """One captured grammar sample; each replay reads the mask staged before it."""
    mesh_device = ttnn_mesh_device
    sampler = _sampler(mesh_device, mode)
    logits = _hot_logits()
    logits_tt = _logits_tt(logits, mesh_device)
    kpt = _kpt(mesh_device, k=1, p=0.0, temp=1.0) if mode == "topk" else None

    def forward():
        if kpt is None:
            return sampler.decode_forward(logits_tt, apply_grammar=True)
        k, p, temp = kpt
        return sampler.decode_forward(logits_tt, k=k, p=p, temp=temp, apply_grammar=True)

    # Compile outside capture, with the all-allowed mask enable_device_grammar staged.
    forward()
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        tokens_tt, _ = forward()
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

    try:
        for allowed in (_first_constraints(), None, _second_constraints(), _first_constraints()):
            sampler.update_grammar_bitmask(None if allowed is None else _mask(allowed))
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            assert _tokens(tokens_tt) == _expected(logits, allowed or {}), f"constraints={allowed}"
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.device
@_MESHES
def test_random_sampling_stays_in_the_allowed_set(ttnn_mesh_device):
    """top-k 32 at temperature 1 over near-flat logits: every draw is an allowed token."""
    mesh_device = ttnn_mesh_device
    sampler = _sampler(mesh_device, "topk")
    torch.manual_seed(1)
    logits = (torch.randn(1, 1, _B, _VOCAB) * 0.01).to(torch.bfloat16)
    logits_tt = _logits_tt(logits, mesh_device)
    k, p, temp = _kpt(mesh_device, k=32, p=1.0, temp=1.0)
    shard0_tail, shard1_head, middle, last = _boundary_tokens()
    allowed = {row: [shard0_tail, shard1_head, middle, last][: 2 + row % 3] for row in range(0, _B, 2)}

    sampler.update_grammar_bitmask(_mask(allowed))
    for _ in range(4):
        tokens = _tokens(sampler.decode_forward(logits_tt, k=k, p=p, temp=temp, apply_grammar=True)[0])
        for row, token_ids in allowed.items():
            assert tokens[row] in token_ids, f"row {row} drew {tokens[row]}, allowed {token_ids}"
