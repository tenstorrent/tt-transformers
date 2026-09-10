# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone C device coverage for `Llama33_70BGalaxyExecutor` on WH Galaxy `(8, 4)`.

One test per coverage item of `tttv2_milestone_c_briefs/c1_exec_llama.md`:

1. eager prefill at 128, 512 and 2048, single row, logits PCC ≥ 0.99 against the
   same request through `GalaxyDirectRunner`;
2. eager decode, batch 1 and batch 32, first token after prefill;
3. paged KV — late capacity resolution, transactional bind/unbind, per-layer KV
   metadata, and KV PCC ≥ 0.99 against the reference path;
4. prefix-cached and chunked prefill;
5. program compilation and `WarmupCoordinator` completion, with program identity
   keyed on physical geometry rather than the active row count;
6. three startup/serve/cleanup cycles in one process with no retained TT
   resources;
7. teacher-forced accuracy through the executor path.

**The reference is a file, not a second cycle in this process.** The clean
comparison would build a `GalaxyDirectRunner` beside the executor, but Milestone C's
open Llama L1 address clash (`program.cpp:1763`, L1 buffer at 544832) reproduces on
the *second* KV allocation cycle in one process, so a same-process comparison would
measure the clash rather than the executor. The reference was therefore computed
once, through the qualified runner, and written to a file that a later process
loads.

**In this package there is no generator and no reference file.** Both were
`GalaxyDirectRunner`-based, and that runner is deliberately absent (2026-09-02
operator decision); the reference tree it wrote does not migrate either, being
1.37 GB of `.pt` with one file above GitHub's 100 MB hard limit. So
`test_reference_prefill_and_decode` skips, and every test that consumes a
reference skips through `_require_reference`. Port Phase 4 owns making these real
again, which is one task rather than two: the references have to be regenerated
*and* re-expressed against an executor path instead of the runner. `_REFERENCE_ROOT`
below is still the tt-metal-relative path and needs replacing with a fixture or
environment indirection as part of that work.

Run one node id per process, as the house rules require::

    pytest tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py \
        -v -rA --color=no -p no:cacheprovider \
        -k "prefill_matches_reference and 128"
"""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import ttnn
from tests.models.galaxy.galaxy_hardware import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    align_top5,
    hf_config_or_skip,
    load_reference_tokens,
    teacher_forcing_accuracy,
)

from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from tt_transformers.models.galaxy.collectives import deallocate_if_allocated
from tt_transformers.models.galaxy.kv_contract import GalaxyPagedAttentionConfig
from tt_transformers.models.llama33_70b_galaxy.executor import (
    Llama33_70BGalaxyExecutor,
    Llama33_70BGalaxyExecutorConfig,
    default_galaxy_paged_kv_cache_config,
)
from tt_transformers.models.llama33_70b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, from_pretrained

_REFERENCE_NAME = "Llama-3.3-70B-Instruct"
_BLOCK_SIZE = 32
_MAX_SEQ_LEN = 2048
_PREFILL_LENGTHS = (128, 512, 2048)
#: The lengths the *runtime's* planner asks the model for. `_padded_prefill_length`
#: pads <=128 to 128, <=1024 to 1024, and everything else to the next power of two,
#: so a 512-token prompt is a 1024-token device request and the model needs a 1024
#: recipe rather than a 512 one. `GalaxyDirectRunner.padded_prefill_length` resolves
#: against the same registered set, so both sides of every comparison pad alike.
_RECIPE_LENGTHS = (128, 1024, 2048)
_CHUNK_ALIGNMENT = 128
_LOGITS_PCC = 0.99
_KV_PCC = 0.99
_TEACHER_FORCED_TOP1 = 0.91
_TEACHER_FORCED_TOP5 = 0.99
_REFERENCE_ROOT = Path("tttv2_milestone_c_evidence/exec_llama/reference")


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _hf_model() -> str:
    return os.getenv("LLAMA33_70B_HF_MODEL", DEFAULT_HF_MODEL)


def _layers() -> int | None:
    value = os.getenv("LLAMA33_70B_GALAXY_TEST_LAYERS")
    return int(value) if value else None


def _load_hf_subset():
    """Return a layer-subset loader when the environment asks for one.

    A layer subset reads three safetensors shards instead of thirty, which is what
    makes an iteration loop on this model affordable. The accuracy gate never uses
    it: it needs every layer.
    """

    layers = _layers()
    if layers is None:
        return None
    from tests.models.galaxy.galaxy_checkpoint import load_layer_subset_causal_lm

    return lambda: load_layer_subset_causal_lm(_hf_model(), layer_indices=tuple(range(layers)))


def _blocks_per_user(max_seq_len: int = _MAX_SEQ_LEN) -> int:
    return -(-max_seq_len // _BLOCK_SIZE)


def _paged_config(
    active_slots: int = GALAXY_PHYSICAL_BATCH,
    max_seq_len: int = _MAX_SEQ_LEN,
) -> GalaxyPagedAttentionConfig:
    """Static block ownership: every active slot gets a full context, plus sinks."""

    per_user = _blocks_per_user(max_seq_len)
    sinks = GALAXY_PHYSICAL_BATCH - active_slots
    return GalaxyPagedAttentionConfig(block_size=_BLOCK_SIZE, max_num_blocks=per_user * active_slots + sinks)


def _page_table_rows(
    active_slots: int = GALAXY_PHYSICAL_BATCH,
    max_seq_len: int = _MAX_SEQ_LEN,
) -> torch.Tensor:
    """Return the `[32, blocks_per_user]` block ownership table.

    The same static ownership `GalaxyDirectRunner` uses, restated here because it
    is the *caller's* mapping in the executor contract: the runtime is handed a
    page table, it does not invent one. Slot ``u`` owns
    ``[u * blocks_per_user, (u + 1) * blocks_per_user)``; each inactive slot
    repeats its own sink block so anything it is asked to write lands there.
    """

    per_user = _blocks_per_user(max_seq_len)
    rows = torch.empty((GALAXY_PHYSICAL_BATCH, per_user), dtype=torch.int32)
    active_total = active_slots * per_user
    for slot in range(GALAXY_PHYSICAL_BATCH):
        if slot < active_slots:
            rows[slot] = torch.arange(slot * per_user, (slot + 1) * per_user, dtype=torch.int32)
        else:
            rows[slot] = active_total + (slot - active_slots)
    return rows


def _load(mesh_device: ttnn.MeshDevice, **overrides: Any):
    hf_model = _hf_model()
    hf_config_or_skip(hf_model)
    kwargs: dict[str, Any] = dict(
        hf_model=hf_model,
        max_seq_len=_MAX_SEQ_LEN,
        prefill_sequence_lengths=_RECIPE_LENGTHS,
        n_layers=_layers(),
        paged_attention_config=_paged_config(),
        enable_device_sampling=False,
        load_hf_model=_load_hf_subset(),
    )
    kwargs.update(overrides)
    return from_pretrained(mesh_device, **kwargs)


def _close(handle: Any) -> None:
    try:
        handle.close()
    finally:
        del handle
        gc.collect()


def _executor_config(
    model: Any,
    *,
    num_blocks: int | None = None,
    prefill_seq_lens: tuple[int, ...] = (128,),
) -> Llama33_70BGalaxyExecutorConfig:
    paged = default_galaxy_paged_kv_cache_config(model)
    if num_blocks is not None:
        paged = PagedKVCacheConfig(
            block_size=paged.block_size,
            max_num_blocks=paged.max_num_blocks,
            dtype=paged.dtype,
            memory_config=paged.memory_config,
            num_blocks=num_blocks,
        )
    return Llama33_70BGalaxyExecutorConfig(
        trace=TraceConfig(mode="none"),
        warmup=WarmupConfig(prefill_seq_lens=prefill_seq_lens, prefill_batch_sizes=(1,)),
        paged_kv_cache=paged,
        device_sampling_enabled=False,
        sequential_prefill_only=True,
    )


def _open_executor(handle: Any, **kwargs: Any) -> tuple[Llama33_70BGalaxyExecutor, list[list[Any]]]:
    executor = Llama33_70BGalaxyExecutor(handle.model, handle.runtime_config, _executor_config(handle.model, **kwargs))
    return executor, executor.allocate_kv_cache()


def _pcc(expected: torch.Tensor, actual: torch.Tensor, threshold: float):
    from tests.support.comparison import comp_pcc

    return comp_pcc(expected.unsqueeze(0).float(), actual.unsqueeze(0).float(), threshold)


def _prompt(length: int) -> list[int]:
    reference_tokens, _ = load_reference_tokens(_REFERENCE_NAME)
    source = [int(value) for value in reference_tokens]
    if len(source) >= length:
        return source[:length]
    # The reference text holds 1024 tokens and the brief asks for 2048. Repeating
    # the text is still real text and keeps the prompt deterministic; a skip would
    # be a failed run, not a result.
    repeats = -(-length // len(source))
    return (source * repeats)[:length]


# ---------------------------------------------------------------------------
# Reference values (generator not ported — see test_reference_prefill_and_decode)
# ---------------------------------------------------------------------------


#: Sources whose bytes decide what a recorded reference *means*. A cached
#: reference computed before any of these changed is not a reference to the
#: code under test, and `c-defects` attempt 10 §11 found exactly that: attempt
#: 1's PCC numbers compared a HEAD executor against an artifact four fixes old,
#: because `_reference_prefill` silently reloads from disk. The fingerprint is
#: over source files rather than over `git rev-parse HEAD` on purpose — commits
#: that only add evidence must not invalidate a reference that is still valid.
_REFERENCE_SOURCES = (
    "tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py",
    "src/tt_transformers/models/llama33_70b_galaxy/executor.py",
    "src/tt_transformers/models/llama33_70b_galaxy/model.py",
    "src/tt_transformers/models/galaxy/recipes.py",
    "src/tt_transformers/modules/attention/attention_2d.py",
)


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in _REFERENCE_SOURCES:
        digest.update(name.encode())
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()[:16]


def _context_for(length: int) -> int:
    """Return a context that can hold `length` prompt tokens and one decode step.

    The recorded reference prefills `length` tokens and then decodes at position
    `length`. At `length == _MAX_SEQ_LEN` that position does not exist: the last
    addressable one is `_MAX_SEQ_LEN - 1`, block `2048 // 32 = 64` is column 64
    of a 64-wide page table, and `GalaxyDirectRunner.decode_logits` validates the
    position *count* and never a position *value*, so the call returns garbage
    instead of refusing (D-C17, `c-defects` attempt 10 §11: 448 columns at ±inf).
    Lengthening the context is the fix to the *question*; the missing bound in
    `direct_runner.py` is a separate defect and is not fixed here.

    `GalaxyDenseGeometry` requires `max_seq_len % chunk_alignment == 0`, so the
    next legal context above 2048 is 2176.
    """

    if length < _MAX_SEQ_LEN:
        return _MAX_SEQ_LEN
    alignment = _CHUNK_ALIGNMENT
    return -(-(length + 1) // alignment) * alignment


def _reference_path(length: int) -> Path:
    layers = _layers() or 0
    context = _context_for(length)
    return _REFERENCE_ROOT / f"llama_prefill{length}_ctx{context}_layers{layers}.pt"


def _read_kv_user(
    kv_pair: list[Any],
    mesh_device: ttnn.MeshDevice,
    *,
    slot: int,
    length: int,
    blocks_per_user: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one slot's `(K, V)` as `[heads, length, head_dim]` on host.

    The paged cache is replicated at allocation; each mesh **row** then writes its
    own KV head slice, and each mesh **column** serves its own eight users. So the
    rows concatenate on the head axis and the columns stack on the block axis, and
    the authoritative copy of slot ``u`` is in column ``u // 8``. The block range
    is sliced on device first: composing the whole pool would move gigabytes.
    """

    blocks = -(-length // _BLOCK_SIZE)
    first = slot * blocks_per_user
    column = slot // (GALAXY_PHYSICAL_BATCH // GALAXY_MESH_SHAPE[1])
    outputs = []
    for cache in kv_pair:
        shape = tuple(int(value) for value in cache.shape)
        window = None
        try:
            window = ttnn.slice(cache, (first, 0, 0, 0), (first + blocks, shape[1], shape[2], shape[3]))
            composed = ttnn.to_torch(
                window,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=GALAXY_MESH_SHAPE),
            ).float()
        finally:
            deallocate_if_allocated(window)
        # (columns * blocks, rows * local_heads, block_size, head_dim)
        mine = composed[column * blocks : (column + 1) * blocks]
        heads = int(mine.shape[1])
        outputs.append(mine.permute(1, 0, 2, 3).reshape(heads, blocks * _BLOCK_SIZE, -1)[:, :length, :])
    return outputs[0], outputs[1]


def _require_reference(length: int) -> dict[str, torch.Tensor]:
    path = _reference_path(length)
    if not path.exists():
        # Absent is the expected state in this package, not an error: the generator
        # was GalaxyDirectRunner-based and did not migrate, and the reference tree
        # it wrote cannot live in a Git repository (1.37 GB of `.pt`, one file over
        # GitHub's 100 MB limit). Skip until port Phase 4 regenerates these against
        # an executor path. A *stale* reference still fails below -- that is a real
        # error, and this must not start passing vacuously once one exists.
        pytest.skip(
            f"reference file {path} is absent; its generator is not ported "
            f"(port Phase 4 regenerates it against the executor path)"
        )
    values = torch.load(path, map_location="cpu", weights_only=False)
    fingerprint = _code_fingerprint()
    if values.get("code_fingerprint") != fingerprint:
        pytest.fail(
            f"reference file {path} was recorded by code fingerprint "
            f"{values.get('code_fingerprint')}, not by the current {fingerprint}; "
            f"re-run test_reference_prefill_and_decode[{length}] in its own process"
        )
    print(f"[reference] using {path} fingerprint {fingerprint}", flush=True)
    return values


def _report_kv_windows(case: str, expected: torch.Tensor, actual: torch.Tensor, length: int) -> None:
    """Report the KV comparison window by window before asserting on the whole.

    One PCC over the whole prefix cannot separate "every position is slightly
    off" from "one block is garbage", and `test_model_wh_galaxy._report_kv_pcc`
    already established that the distinction is what identifies the defect.
    Reporting is not asserting: the gate below is untouched.
    """

    if expected.shape != actual.shape:
        print(f"[kv] {case}: shape {tuple(expected.shape)} vs {tuple(actual.shape)}", flush=True)
        return
    windows = {
        "all": slice(0, length),
        "first32": slice(0, min(32, length)),
        "last32": slice(max(0, length - 32), length),
    }
    for name, window in windows.items():
        _, message = _pcc(expected[:, window, :], actual[:, window, :], 0.0)
        print(f"[kv] {case} {name}: {message}", flush=True)
    diff = (expected - actual).abs()
    per_position = diff.amax(dim=(0, 2))
    worst = torch.topk(per_position, k=min(8, per_position.numel()))
    print(
        f"[kv] {case}: reference |max|={float(expected.abs().max()):.6g} device |max|="
        f"{float(actual.abs().max()):.6g} maxabsdiff={float(diff.max()):.6g}",
        flush=True,
    )
    print(
        f"[kv] {case}: worst positions {worst.indices.tolist()} values {[round(float(v), 5) for v in worst.values]}",
        flush=True,
    )
    print(
        f"[kv] {case}: per-head maxabsdiff {[round(float(v), 5) for v in diff.amax(dim=(1, 2))]}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Executor request helpers
# ---------------------------------------------------------------------------


def _executor_prefill(
    executor: Llama33_70BGalaxyExecutor,
    kv_cache: Any,
    prompt: list[int],
    *,
    slot: int = 0,
    rows: torch.Tensor | None = None,
    max_seq_len: int = _MAX_SEQ_LEN,
):
    rows = _page_table_rows(max_seq_len=max_seq_len) if rows is None else rows
    return executor.prefill_forward(
        torch.tensor(prompt, dtype=torch.long).reshape(1, -1),
        rows[slot : slot + 1],
        prompt_lens=torch.tensor([len(prompt)], dtype=torch.long),
        empty_slots=[slot],
        kv_cache=kv_cache,
    )


def _executor_decode(
    executor: Llama33_70BGalaxyExecutor,
    kv_cache: Any,
    tokens: list[int],
    positions: list[int],
):
    return executor.decode_forward(
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
        _page_table_rows(),
        kv_cache=kv_cache,
    )


def _decode_logits(result: Any) -> torch.Tensor:
    logits, _ = result if isinstance(result, tuple) else (result, None)
    return logits.float().reshape(GALAXY_PHYSICAL_BATCH, -1)


# ---------------------------------------------------------------------------
# 0. Reference generation (one cycle, no executor)
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("length", _PREFILL_LENGTHS)
def test_reference_prefill_and_decode(mesh_device: ttnn.MeshDevice, length: int) -> None:
    """Regenerate the reference values the executor is checked against.

    The tt-metal original computed these through `GalaxyDirectRunner`, which
    Milestone B had qualified. That runner is deliberately absent from this
    package (2026-09-02 operator decision), so this generator has no
    implementation here and the reference tree it wrote does not migrate either:
    1.37 GB of `.pt`, one file above GitHub's 100 MB hard limit.

    Phase 4 of the port owns making this real again, and it is one task rather
    than two — the references have to be regenerated *and* re-expressed against an
    executor path instead of the runner. Until then this skips, and the tests that
    consume the references skip through `_require_reference`.
    """

    pytest.skip(
        "reference generator was GalaxyDirectRunner-based and is not ported; "
        "regenerate against the executor path (port Phase 4)"
    )


# ---------------------------------------------------------------------------
# 1. Eager prefill
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("length", _PREFILL_LENGTHS)
def test_executor_prefill_matches_reference(mesh_device: ttnn.MeshDevice, length: int) -> None:
    """Coverage 1: single-row eager prefill logits, PCC ≥ 0.99 against the runner."""

    reference = _require_reference(length)
    context = _context_for(length)
    handle = _load(
        mesh_device,
        max_seq_len=context,
        paged_attention_config=_paged_config(max_seq_len=context),
    )
    executor = None
    try:
        executor, kv_cache = _open_executor(handle)
        logits = _executor_prefill(executor, kv_cache, _prompt(length), max_seq_len=context)
        actual = logits.float().reshape(-1)
        expected = reference["prefill_logits"].float().reshape(-1)
        passed, message = _pcc(expected, actual, _LOGITS_PCC)
        print(f"[exec] prefill {length} logits {message}", flush=True)
        assert int(torch.argmax(actual)) == int(torch.argmax(expected)), (
            f"executor prefill argmax {int(torch.argmax(actual))} != reference {int(torch.argmax(expected))}"
        )
        assert passed, f"prefill {length} logits PCC below {_LOGITS_PCC}: {message}"
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 2. Eager decode
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("active_rows", [1, GALAXY_PHYSICAL_BATCH])
def test_executor_decode_first_token(mesh_device: ttnn.MeshDevice, active_rows: int) -> None:
    """Coverage 2: eager decode at batch 1 and batch 32, first token after prefill."""

    length = 128
    reference = _require_reference(length)
    prompt = _prompt(length)
    handle = _load(mesh_device)
    executor = None
    try:
        executor, kv_cache = _open_executor(handle)
        first_tokens = []
        for slot in range(active_rows):
            logits = _executor_prefill(executor, kv_cache, prompt, slot=slot)
            first_tokens.append(int(torch.argmax(logits.float().reshape(-1))))
        assert len(set(first_tokens)) == 1, f"identical prompts produced different first tokens: {first_tokens}"

        tokens = [0] * GALAXY_PHYSICAL_BATCH
        positions = [0] * GALAXY_PHYSICAL_BATCH
        for slot in range(active_rows):
            tokens[slot] = first_tokens[slot]
            positions[slot] = length
        logits = _decode_logits(_executor_decode(executor, kv_cache, tokens, positions))
        assert torch.isfinite(logits[:active_rows]).all(), "decode logits are not finite"

        expected = reference["decode_logits"].float()
        passed, message = _pcc(expected[0], logits[0], _LOGITS_PCC)
        print(f"[exec] decode active_rows={active_rows} row 0 {message}", flush=True)
        assert passed, f"decode row 0 logits PCC below {_LOGITS_PCC}: {message}"
        if active_rows > 1:
            reference_argmax = int(torch.argmax(expected[0]))
            argmaxes = [int(torch.argmax(logits[slot])) for slot in range(active_rows)]
            assert set(argmaxes) == {reference_argmax}, (
                f"batch-32 decode disagrees across slots: {argmaxes} against {reference_argmax}"
            )
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 3. Paged KV
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_paged_kv_contract(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    """Coverage 3: late capacity resolution, bind/unbind, metadata, KV PCC."""

    length = 128
    reference = _require_reference(length)
    handle = _load(mesh_device)
    model = handle.model
    executor = None
    try:
        # --- late capacity resolution: construct against the ceiling only.
        resolved = default_galaxy_paged_kv_cache_config(model)
        unresolved = PagedKVCacheConfig(
            block_size=resolved.block_size,
            max_num_blocks=resolved.max_num_blocks,
            dtype=resolved.dtype,
            memory_config=resolved.memory_config,
            num_blocks=None,
        )
        executor = Llama33_70BGalaxyExecutor(
            model,
            handle.runtime_config,
            Llama33_70BGalaxyExecutorConfig(
                trace=TraceConfig(mode="none"),
                warmup=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
                paged_kv_cache=unresolved,
            ),
        )
        assert executor.paged_kv_cache_config.num_blocks is None
        with expect_error(RuntimeError, "capacity must be resolved before allocation"):
            executor.allocate_kv_cache()

        executor.configure_paged_kv_cache(resolved)
        assert executor.paged_kv_cache_config.num_blocks == resolved.max_num_blocks
        assert executor.page_table_layout.raw_capacity_width == _MAX_SEQ_LEN // _BLOCK_SIZE
        kv_cache = executor.allocate_kv_cache()

        # --- per-layer metadata, derived from the model's own attention specs.
        context = executor.kv_cache_manager.bound_context
        assert context is not None
        assert len(context.tensors) == model.n_layers
        assert len(context.cache_shapes) == model.n_layers
        for shape, spec in zip(context.cache_shapes, model.kv_specs):
            assert shape == (
                resolved.max_num_blocks,
                spec.n_local_kv_heads,
                resolved.block_size,
                spec.head_dim,
            )
        assert set(context.per_layer_dtypes) == {spec.kv_cache_dtype for spec in model.kv_specs}

        # --- transactional binding: only the exact borrowed handle is accepted.
        executor.kv_cache_manager.validate_borrowed_handle(kv_cache)
        with expect_error(ValueError, "exact manager-owned borrowed handle"):
            executor.kv_cache_manager.validate_borrowed_handle([list(pair) for pair in kv_cache])
        assert all(layer.attention.kv_cache_binding is not None for layer in model.layers)

        # --- the KV the executor's prefill wrote, against the reference path.
        _executor_prefill(executor, kv_cache, _prompt(length))
        blocks_per_user = _MAX_SEQ_LEN // _BLOCK_SIZE
        for label, pair in (("first", kv_cache[0]), ("last", kv_cache[-1])):
            actual_k, actual_v = _read_kv_user(
                pair, mesh_device, slot=0, length=length, blocks_per_user=blocks_per_user
            )
            for kind, actual in (("k", actual_k), ("v", actual_v)):
                expected = reference[f"kv_{label}_{kind}"].float()
                _report_kv_windows(f"{label} {kind.upper()}", expected, actual, length)
                passed, message = _pcc(expected, actual, _KV_PCC)
                print(f"[exec] KV {label} layer {kind.upper()} {message}", flush=True)
                assert passed, f"KV {label} layer {kind.upper()} PCC below {_KV_PCC}: {message}"

        # --- release unbinds transactionally and leaves nothing retained.
        executor.cleanup()
        executor_after = executor
        executor = None
        assert all(layer.attention.kv_cache_binding is None for layer in model.layers)
        assert executor_after.kv_cache_manager.bound_context is None
        assert executor_after.terminal
        assert not any(tensor.is_allocated() for pair in kv_cache for tensor in pair)
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_paged_kv_shrinks_to_a_smaller_physical_pool(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 3: a physical pool smaller than the construction ceiling still serves."""

    length = 128
    active_slots = 1
    handle = _load(mesh_device, paged_attention_config=_paged_config(active_slots=GALAXY_PHYSICAL_BATCH))
    executor = None
    try:
        ceiling = default_galaxy_paged_kv_cache_config(handle.model)
        physical = (_MAX_SEQ_LEN // _BLOCK_SIZE) * active_slots + (GALAXY_PHYSICAL_BATCH - active_slots)
        assert physical < ceiling.max_num_blocks
        executor = Llama33_70BGalaxyExecutor(
            handle.model,
            handle.runtime_config,
            Llama33_70BGalaxyExecutorConfig(
                trace=TraceConfig(mode="none"),
                warmup=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
                paged_kv_cache=PagedKVCacheConfig(
                    block_size=ceiling.block_size,
                    max_num_blocks=ceiling.max_num_blocks,
                    dtype=ceiling.dtype,
                    memory_config=ceiling.memory_config,
                    num_blocks=None,
                ),
            ),
        )
        executor.configure_paged_kv_cache(
            PagedKVCacheConfig(
                block_size=ceiling.block_size,
                max_num_blocks=ceiling.max_num_blocks,
                dtype=ceiling.dtype,
                memory_config=ceiling.memory_config,
                num_blocks=physical,
            )
        )
        kv_cache = executor.allocate_kv_cache()
        assert executor.kv_cache_manager.bound_context.cache_shapes[0][0] == physical
        # A pool sized for one active slot can only be addressed by a page table
        # sized for one active slot: slot 0 owns a full context and the other
        # thirty-one own one sink block each.
        rows = _page_table_rows(active_slots=active_slots)
        assert int(rows.max()) < physical
        logits = _executor_prefill(executor, kv_cache, _prompt(length), rows=rows[0:1])
        first = int(torch.argmax(logits.float().reshape(-1)))
        tokens = [0] * GALAXY_PHYSICAL_BATCH
        positions = [0] * GALAXY_PHYSICAL_BATCH
        tokens[0], positions[0] = first, length
        decode = _decode_logits(
            executor.decode_forward(
                torch.tensor(tokens, dtype=torch.long),
                torch.tensor(positions, dtype=torch.long),
                rows,
                kv_cache=kv_cache,
            )
        )
        assert torch.isfinite(decode[0]).all()
        print(f"[exec] shrunk pool {physical} blocks: first token {first}", flush=True)
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 4. Prefix-cached and chunked prefill
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_prefix_cached_prefill(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 4: resume a prompt whose first chunk is already in the cache.

    Blocked for Llama at Milestone B and through `c-defects` attempt 1 by the L1
    address clash, which `c-defects` has since fixed (`llama-address-clash.status`).

    Two things about the shape, both forced by the model and neither a relaxation:

    * the cached prefix is one **chunk alignment** (128), not one page-table
      block (32). `Attention2D._validate_prefill` refuses a `chunk_start` that is
      not a multiple of `recipe.chunk_alignment`, and that alignment *is* the
      flash-SDPA chunk size, so 32 is not a position this model can resume from.
      That is D-C16, and it is the same reason the warmup plan's prefix moved;
    * the prefix has to be **in the cache** for the claim to mean anything, so
      slot 1 is prefilled with the first 128 tokens before the resume. The
      earlier shape resumed into an empty slot, which would have compared a
      cached prefill against a prefill of a prefix that was never written.

    A cached request is a chunked request to the planner, and the 2D attention
    module resolves one frozen recipe per prefill shape, so the prefix/chunked
    attention mode at 128 has to be registered at construction or the request is
    refused on the host before any device work (`logs/i7_prefix_l1.log`).
    """

    length = 2 * _CHUNK_ALIGNMENT
    cached = _CHUNK_ALIGNMENT
    prompt = _prompt(length)
    handle = _load(mesh_device, chunked_prefill_sequence_lengths=(128,))
    executor = None
    try:
        executor, kv_cache = _open_executor(handle)
        full = _executor_prefill(executor, kv_cache, prompt, slot=0)
        rows = _page_table_rows()
        # Slot 1 gets the prefix, as a plain uncached prefill, exactly as a
        # server would have served the first request of the conversation.
        _executor_prefill(executor, kv_cache, prompt[:cached], slot=1)
        resumed = executor.prefill_forward(
            torch.tensor(prompt, dtype=torch.long).reshape(1, -1),
            rows[1:2],
            prompt_lens=torch.tensor([length], dtype=torch.long),
            start_pos=torch.tensor([cached], dtype=torch.long),
            empty_slots=[1],
            kv_cache=kv_cache,
        )
        expected = full.float().reshape(-1)
        actual = resumed.float().reshape(-1)
        passed, message = _pcc(expected, actual, _LOGITS_PCC)
        print(f"[exec] prefix-cached prefill {message}", flush=True)
        assert passed, f"prefix-cached prefill logits PCC below {_LOGITS_PCC}: {message}"
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_chunked_prefill(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 4: a prompt longer than one prefill chunk, planned as chunks."""

    # The planner chunks when the padded request exceeds `max_prefill_chunk_size`,
    # which the model's runtime config fixes at 2048. So a genuinely chunked
    # request needs a context longer than that: 4096 tokens in two 2048-token
    # chunks, with the 2048 prefix/chunked recipe registered.
    max_seq_len = 4096
    length = 4096
    chunk = 2048
    prompt = _prompt(length)
    handle = _load(
        mesh_device,
        max_seq_len=max_seq_len,
        prefill_sequence_lengths=(128, 2048),
        chunked_prefill_sequence_lengths=(2048,),
        paged_attention_config=_paged_config(max_seq_len=max_seq_len),
    )
    executor = None
    try:
        executor, kv_cache = _open_executor(handle, prefill_seq_lens=(128,))
        assert executor.prefill_runtime.config.max_prefill_chunk_size == chunk
        rows = _page_table_rows(max_seq_len=max_seq_len)
        logits = executor.prefill_forward(
            torch.tensor(prompt, dtype=torch.long).reshape(1, -1),
            rows[0:1],
            prompt_lens=torch.tensor([length], dtype=torch.long),
            empty_slots=[0],
            kv_cache=kv_cache,
        )
        actual = logits.float().reshape(-1)
        assert torch.isfinite(actual).all(), "chunked prefill logits are not finite"
        print(f"[exec] chunked prefill {length} argmax {int(torch.argmax(actual))}", flush=True)
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 5. Program compilation and warmup
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("order", ["prefill_first", "decode_first"])
def test_executor_warmup_and_program_identity(mesh_device: ttnn.MeshDevice, order: str) -> None:
    """Coverage 5: warmup completes, and program identity ignores active rows.

    `WarmupCoordinator` documents that the two warmup calls may be made in either
    order, so both orders are measured. They are not equivalent on this mesh:
    `logs/i5_warmup_l1.log` shows one compiled decode program followed by the
    Llama L1 address clash on the very next prefill (`program 922`, L1 buffer at
    543488, core range `[0-0 - 0-3]`, from `Embedding2D._forward`). That is
    c-defects' open defect, reached here by a much cheaper route than its
    141-second probe, and the parametrization is how this file reports it rather
    than avoids it.
    """

    # Every warmup plan includes one cached (prefix) prefill case per configured
    # sequence length, so the prefix/chunked recipe must be registered for the
    # coordinator to be able to complete at all.
    handle = _load(mesh_device, chunked_prefill_sequence_lengths=(128,))
    executor = None
    try:
        executor, kv_cache = _open_executor(handle)
        prompt = _prompt(128)
        tokens = [0] * GALAXY_PHYSICAL_BATCH
        positions = [0] * GALAXY_PHYSICAL_BATCH

        if order == "decode_first":
            executor.warmup_model_decode(kv_cache=kv_cache)
            assert len(executor.program_compiler.compiled_programs) >= 1
            executor.warmup_model_prefill(kv_cache=kv_cache)
            assert executor.already_warmed_up_prefill, "prefill warmup coverage did not complete"
            print(
                f"[exec] warmup (decode_first) compiled {len(executor.program_compiler.compiled_programs)} programs",
                flush=True,
            )
            return

        # Prefill coverage, then the prefill identity check, then decode coverage
        # and the decode identity check. The order is not a convenience: on this
        # mesh a prefill after a decode raises the open Llama L1 address clash,
        # so a serve-everything-after-warming-everything shape would measure the
        # clash instead of program identity. The `decode_first` parametrization
        # above reports that; this one measures what the identity claim is about.
        executor.warmup_model_prefill(kv_cache=kv_cache)
        assert executor.already_warmed_up_prefill, "prefill warmup coverage did not complete"
        after_prefill_warmup = len(executor.program_compiler.compiled_programs)
        print(f"[exec] prefill warmup compiled {after_prefill_warmup} programs", flush=True)

        # Two prefills of the same padded geometry into different slots must
        # reuse one program: the identity carries padded geometry, not the slot.
        _executor_prefill(executor, kv_cache, prompt, slot=0)
        _executor_prefill(executor, kv_cache, prompt, slot=1)
        assert len(executor.program_compiler.compiled_programs) == after_prefill_warmup, (
            "serving prefill after warmup compiled a program warmup did not cover"
        )

        executor.warmup_model_decode(kv_cache=kv_cache)
        after_warmup = len(executor.program_compiler.compiled_programs)
        assert after_warmup > after_prefill_warmup, "decode warmup compiled no program"
        print(f"[exec] warmup (prefill_first) compiled {after_warmup} programs", flush=True)

        # One active decode row and thirty-two must reuse one program: the
        # identity carries the lane's fixed capacity, not the active row count.
        positions[0] = 128
        _executor_decode(executor, kv_cache, tokens, positions)
        for slot in range(GALAXY_PHYSICAL_BATCH):
            positions[slot] = 128
        _executor_decode(executor, kv_cache, tokens, positions)
        assert len(executor.program_compiler.compiled_programs) == after_warmup, (
            "serving decode after warmup compiled a program warmup did not cover"
        )
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 6. Repeated startup, serving and cleanup
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_repeated_startup_and_cleanup(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 6: three startup/serve/cleanup cycles over one model."""

    length = 128
    prompt = _prompt(length)
    handle = _load(mesh_device)
    tokens_seen = []
    try:
        for cycle in range(3):
            executor, kv_cache = _open_executor(handle)
            try:
                logits = _executor_prefill(executor, kv_cache, prompt)
                first = int(torch.argmax(logits.float().reshape(-1)))
                tokens = [0] * GALAXY_PHYSICAL_BATCH
                positions = [0] * GALAXY_PHYSICAL_BATCH
                tokens[0], positions[0] = first, length
                decode = _decode_logits(_executor_decode(executor, kv_cache, tokens, positions))
                tokens_seen.append((first, int(torch.argmax(decode[0]))))
                print(f"[exec] cycle {cycle}: {tokens_seen[-1]}", flush=True)
            finally:
                executor.cleanup()
            assert not any(tensor.is_allocated() for pair in kv_cache for tensor in pair), (
                f"cycle {cycle} retained KV tensors after cleanup"
            )
            assert all(layer.attention.kv_cache_binding is None for layer in handle.model.layers), (
                f"cycle {cycle} left the model bound to a released cache"
            )
            assert executor.prefill_runtime.transient_orphan_count == 0
            assert executor.decode_runtime.transient_orphan_count == 0
            del executor
            gc.collect()
        assert len(set(tokens_seen)) == 1, f"three cycles disagreed: {tokens_seen}"
    finally:
        _close(handle)


# ---------------------------------------------------------------------------
# 7. Teacher-forced accuracy
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_teacher_forced_accuracy(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 7: top-1 ≥ 91% and top-5 ≥ 99% through the executor path.

    The same convention as Milestone B's `GalaxyDirectRunner` gate, which measured
    98.04% / 100.00%: prefill 512, then 511 forced decode steps at batch 1.
    """

    prompt_len = 512
    reference_tokens, top5_tokens = load_reference_tokens(_REFERENCE_NAME)
    reference_tokens = reference_tokens.reshape(-1)
    if len(reference_tokens) <= prompt_len:
        pytest.fail(f"reference sequence has {len(reference_tokens)} tokens, need more than {prompt_len}")
    prompt = [int(value) for value in reference_tokens[:prompt_len]]
    forced = [int(value) for value in reference_tokens[prompt_len:]]
    aligned = align_top5(top5_tokens, reference_tokens, prompt_len)

    # The planner pads a 512-token prompt to a 1024-token device request, so the
    # model is built with the default registered set `(128, 1024, 2048)` rather
    # than with a 512 recipe it would never be asked for.
    handle = _load(mesh_device)
    executor = None
    try:
        executor, kv_cache = _open_executor(handle)
        logits = _executor_prefill(executor, kv_cache, prompt)
        predictions = [int(torch.argmax(logits.float().reshape(-1)))]
        tokens = [0] * GALAXY_PHYSICAL_BATCH
        positions = [0] * GALAXY_PHYSICAL_BATCH
        for index, token in enumerate(forced[:-1]):
            tokens[0] = int(token)
            positions[0] = prompt_len + index
            step = _decode_logits(_executor_decode(executor, kv_cache, tokens, positions))
            predictions.append(int(torch.argmax(step[0])))
            if index % 64 == 0:
                print(f"[exec] teacher-forced step {index}/{len(forced) - 1}", flush=True)
        top1, top5 = teacher_forcing_accuracy(predictions, aligned[: len(predictions)])
        print(f"[exec] teacher-forced top-1 {top1 * 100:.2f}% top-5 {top5 * 100:.2f}%", flush=True)
        assert top1 >= _TEACHER_FORCED_TOP1, f"top-1 {top1 * 100:.2f}% below {_TEACHER_FORCED_TOP1 * 100:.0f}%"
        assert top5 >= _TEACHER_FORCED_TOP5, f"top-5 {top5 * 100:.2f}% below {_TEACHER_FORCED_TOP5 * 100:.0f}%"
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)
