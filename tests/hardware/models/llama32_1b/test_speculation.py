# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare real Llama verification with ordinary execution and committed KV."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pytest
import torch
import ttnn
from examples.common.runtime import open_mesh_device
from tests.support.marker_policy import selected_topology_marks

from tt_transformers.device_ownership import default_device_scope
from tt_transformers.models.llama32_1b.vllm_generator import (
    Llama32_1BGeneratorConfig,
    build_llama32_1b_generator,
)

pytestmark = [pytest.mark.timeout(7200), *selected_topology_marks(pytest)]

_MODEL = "meta-llama/Llama-3.2-1B"
_BLOCK_SIZE = 32
_ROW_BLOCKS = 4
_CACHE_BLOCKS = 64


@dataclass
class Target:
    generator: object
    cache: list
    batch: int
    tables: tuple[torch.Tensor, ...]


@pytest.fixture(scope="module", params=[1, 2, 4], ids=lambda batch: f"B{batch}")
def target(request):
    selected = os.environ.get("MESH_DEVICE", "").strip().upper()
    if selected not in {"N150", "P150"}:
        pytest.skip("Set MESH_DEVICE=N150 or P150 for the single-device Llama verifier")
    if ttnn.device.is_blackhole() != (selected == "P150"):
        pytest.fail(f"MESH_DEVICE={selected} does not match the device architecture")
    if os.environ.get("HF_MODEL", _MODEL) != _MODEL:
        pytest.fail(f"This regression requires the base checkpoint {_MODEL}")
    from huggingface_hub import try_to_load_from_cache

    revision = os.environ.get("LLAMA_CHECKPOINT_REVISION")
    if not revision or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        pytest.fail("Set LLAMA_CHECKPOINT_REVISION to the exact cached base-checkpoint commit")
    cached_config = try_to_load_from_cache(_MODEL, "config.json", revision="main")
    if not isinstance(cached_config, str) or cached_config.rsplit("/", 2)[-2] != revision:
        pytest.fail("The cached base-checkpoint main reference must resolve to LLAMA_CHECKPOINT_REVISION")
    batch = request.param
    with open_mesh_device({"mesh_shape": (1, 1), "trace_region_size": 0, "num_command_queues": 1}) as mesh:
        with default_device_scope(ttnn, mesh, owner="test_llama32_speculation"):
            mesh.enable_program_cache()
            generator = build_llama32_1b_generator(
                Llama32_1BGeneratorConfig(
                    hf_model=_MODEL,
                    mesh_device=mesh,
                    max_batch_size=batch,
                    max_seq_len=2048,
                    trace_mode="none",
                    device_sampling_enabled=False,
                    optimizations="performance",
                )
            )
            try:
                attention = generator.model.layers[0].attention.config
                cache = generator.allocate_kv_cache(
                    (_CACHE_BLOCKS, attention.n_kv_heads, _BLOCK_SIZE, attention.head_dim),
                    torch.bfloat16,
                    generator.model.config.n_layers,
                )
                tables = tuple(
                    torch.arange(
                        region * batch * _ROW_BLOCKS,
                        (region + 1) * batch * _ROW_BLOCKS,
                        dtype=torch.int32,
                    ).reshape(batch, _ROW_BLOCKS)
                    for region in range(3)
                )
                print(f"LLAMA_SPEC_TARGET: checkpoint={_MODEL}@{revision} B={batch} trace=none")
                yield Target(generator, cache, batch, tables)
            finally:
                generator.cleanup()
                ttnn.synchronize_device(mesh)
                print("LLAMA_SPEC_TEARDOWN: generator cleaned; borrowed mesh synchronized")


def _ordinary(target, tokens, positions, table):
    output = target.generator.decode_forward(
        tokens[:, None],
        positions,
        table,
        enable_trace=False,
        kv_cache=target.cache,
        sampling_params=None,
    )
    logits = output[0] if isinstance(output, tuple) else output
    assert logits.shape[:2] == (target.batch, 1)
    return logits[:, 0].clone()


def _prefix(target, seed=71):
    lengths = torch.tensor([31 - row for row in range(target.batch)], dtype=torch.long)
    tokens = torch.randint(100, 1000, (target.batch, 31), generator=torch.Generator().manual_seed(seed))
    for table in target.tables:
        target.generator.prefill_forward(
            tokens,
            table,
            prompt_lens=lengths,
            enable_trace=False,
            kv_cache=target.cache,
            sampling_params=None,
        )
    return lengths, tokens[:, -1].clone()


def _snapshot(target):
    return tuple(ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone() for pair in target.cache for tensor in pair)


def _row_kv(snapshot, row, region, batch):
    first = (region * batch + row) * _ROW_BLOCKS
    return tuple(
        tensor[first : first + _ROW_BLOCKS].permute(0, 2, 1, 3).reshape(_ROW_BLOCKS * _BLOCK_SIZE, -1)
        for tensor in snapshot
    )


def _assert_no_inactive_writes(target, before, after, positions, counts):
    for row in range(target.batch):
        unchanged = torch.ones(_ROW_BLOCKS * _BLOCK_SIZE, dtype=torch.bool)
        if positions[row, 0] >= 0:
            start = int(positions[row, 0])
            unchanged[start : start + int(counts[row]) + 1] = False
        for previous, current in zip(_row_kv(before, row, 0, target.batch), _row_kv(after, row, 0, target.batch)):
            torch.testing.assert_close(current[unchanged], previous[unchanged], rtol=0, atol=0)


def _verify(target, tokens, positions, counts, accepted_counts=None):
    table = target.tables[0].clone()
    for row in range(target.batch):
        used = 0 if positions[row, 0] < 0 else (int(positions[row, 0]) + int(counts[row])) // _BLOCK_SIZE + 1
        table[row, used:] = -1
    result = target.generator.decode_forward(
        tokens,
        positions,
        table,
        enable_trace=False,
        kv_cache=target.cache,
        sampling_params=None,
        spec_mode="logits",
        num_valid_drafts=counts,
        accepted_counts=torch.ones(target.batch, dtype=torch.int32) if accepted_counts is None else accepted_counts,
    )
    assert result.spec_mode == "logits"
    assert result.logits.device.type == "cpu"
    assert result.logits.is_floating_point()
    assert result.logits.shape == (*tokens.shape, target.generator.model.config.vocab_size)
    return result.logits


def _assert_logits(actual, expected):
    # Both computations use the same eager graph, precision, batch, and prefix.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.softmax(-1), expected.softmax(-1), rtol=0, atol=0)


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3], ids=lambda k: f"K{k}")
def test_candidate_logits_and_inactive_kv(target, k):
    for pattern in ("full", "mixed", "zero", "padding"):
        starts, anchors = _prefix(target, seed=73 + k)
        counts = torch.full((target.batch,), k, dtype=torch.int32)
        if pattern == "mixed":
            counts = torch.tensor([max(k - row, 0) for row in range(target.batch)], dtype=torch.int32)
        elif pattern == "zero":
            counts.zero_()
        positions = starts[:, None] + torch.arange(k + 1)
        positions[torch.arange(k + 1)[None, :] > counts[:, None]] = -1
        if pattern == "padding":
            positions[-1] = -1
            counts[-1] = 0
        tokens = torch.randint(100, 1000, (target.batch, k + 1), generator=torch.Generator().manual_seed(99 + k))
        tokens[:, 0] = anchors
        before = _snapshot(target)
        verified = _verify(target, tokens, positions, counts)
        saved = verified.clone()
        after = _snapshot(target)
        _assert_no_inactive_writes(target, before, after, positions, counts)
        for column in range(k + 1):
            active = (positions[:, 0] >= 0) & (counts >= column)
            if bool(active.any()):
                reference = _ordinary(
                    target,
                    tokens[:, column].masked_fill(~active, 0),
                    positions[:, column].masked_fill(~active, -1),
                    target.tables[1],
                )
                _assert_logits(verified[active, column], reference[active])
            assert torch.count_nonzero(verified[~active, column]) == 0
        _verify(target, tokens + 17, positions, counts)
        torch.testing.assert_close(verified, saved, rtol=0, atol=0)
        print(f"LLAMA_SPEC_COLUMNS: B={target.batch} K={k} pattern={pattern} logits=exact KV_inactive=exact")


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3], ids=lambda k: f"K{k}")
def test_rejection_correction_and_committed_kv(target, k):
    from vllm_tt_plugin.spec_accept import SpecSamplingInputs, SpecTokenFilters, accept_sampled_drafts

    for depth in sorted({0, min(1, k), k}):
        starts, anchors = _prefix(target, seed=107 + k + depth)
        tokens = torch.full((target.batch, k + 1), 101, dtype=torch.long)
        tokens[:, 0] = anchors
        positions = starts[:, None] + torch.arange(k + 1)
        for column in range(depth + 1):
            reference = _ordinary(target, tokens[:, column], positions[:, column], target.tables[2])
            if column == 0:
                correction = reference.argmax(-1)
            if column < depth:
                tokens[:, column + 1] = correction
        if depth < k:
            tokens[:, depth + 1] = (correction + 1) % reference.shape[-1]
        counts = torch.full((target.batch,), k, dtype=torch.int32)
        verified = _verify(target, tokens, positions, counts)
        for column in range(k + 1):
            expected = _ordinary(target, tokens[:, column], positions[:, column], target.tables[1])
            _assert_logits(verified[:, column], expected)
        disallowed = torch.ones((target.batch, verified.shape[-1]), dtype=torch.bool)
        disallowed[torch.arange(target.batch), correction] = False
        accepted = accept_sampled_drafts(
            verified,
            tokens[:, 1:].to(torch.int32),
            counts,
            SpecSamplingInputs(
                vocab_size=verified.shape[-1],
                temperature=torch.tensor([0.7 if row % 2 == 0 else 0.0 for row in range(target.batch)]),
                top_k=torch.ones(target.batch, dtype=torch.int32),
                filters=SpecTokenFilters(allowed_token_ids_mask=disallowed),
            ),
        )
        assert accepted.accepted_counts.tolist() == [depth + 1] * target.batch
        assert torch.equal(accepted.committed_token_ids[:, depth].long(), correction)
        for offset in range(2):
            next_positions = starts + depth + 1 + offset
            actual = _ordinary(target, correction, next_positions, target.tables[0])
            clean = _ordinary(target, correction, next_positions, target.tables[2])
            _assert_logits(actual, clean)
            correction = clean.argmax(-1)
        snapshot = _snapshot(target)
        for row in range(target.batch):
            committed_end = int(starts[row]) + depth + 3
            for actual, clean in zip(_row_kv(snapshot, row, 0, target.batch), _row_kv(snapshot, row, 2, target.batch)):
                torch.testing.assert_close(actual[:committed_end], clean[:committed_end], rtol=0, atol=0)
        print(f"LLAMA_SPEC_ACCEPT: B={target.batch} K={k} accepted_drafts={depth} next_logits=exact committed_KV=exact")
