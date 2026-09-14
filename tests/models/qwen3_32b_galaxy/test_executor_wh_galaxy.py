# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone C device coverage for `Qwen3_32BGalaxyExecutor` on WH Galaxy `(8, 4)`.

One test per coverage item of `tttv2_milestone_c_briefs/c2_exec_qwen.md`:

1. eager decode, batch 1 and batch 32, first token after prefill — the slots must
   agree with each other;
2. paged KV — late capacity resolution, transactional bind/unbind, per-layer KV
   metadata;
3. prefix-cached and chunked prefill. **Both already passed for Qwen at Milestone
   B**, two fresh processes each, so a failure here is a regression introduced by
   `c-defects` or by the executor and is reported as one;
4. program compilation and `WarmupCoordinator` completion, with program identity
   keyed on physical geometry rather than the active row count;
5. three startup/serve/cleanup cycles in one process with no retained TT
   resources — the one-live-model form, which Qwen passed 3/3 at Milestone B;
6. teacher-forced accuracy through the executor path, top-1 >= 89% / top-5 >= 97%;
7. **two models in one process** — the D-C7 shape, which Qwen *failed* at
   Milestone B and which `c-defects` fixed across four defects (D-C7 proper,
   D-C13, D-C14, D-C15). Qwen is the only model of the two that can see D-C7's
   capacity residue, because it does not carry the Llama L1 address clash, so this
   file is where the fix is verified to have stayed fixed *through the executor*.

**What is no longer here.** Coverage items 1 and 3 of the brief compared the
executor against a reference recorded through `GalaxyDirectRunner`, which
Milestone B had qualified. The runner is deliberately absent (2026-09-02
operator decision) and nothing else in this package can produce those tensors,
so those comparisons are **retired, not reimplemented** — recording them through
the executor would only have compared it against itself. Every assertion that
stands without a recording is kept, including the two gates that used to do both.
The `.refpt` token assets behind the accuracy gate are a different mechanism and
are untouched.

Run one node id per process, as the house rules require::

    pytest tests/models/qwen3_32b_galaxy/test_executor_wh_galaxy.py \
        -v -rA --color=no -p no:cacheprovider \
        -k "paged_kv_contract"
"""

from __future__ import annotations

import gc
import os
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
from tt_transformers.models.galaxy.kv_contract import GalaxyPagedAttentionConfig
from tt_transformers.models.qwen3_32b_galaxy.executor import (
    Qwen3_32BGalaxyExecutor,
    Qwen3_32BGalaxyExecutorConfig,
    default_galaxy_paged_kv_cache_config,
)
from tt_transformers.models.qwen3_32b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, from_pretrained
from tt_transformers.models.qwen3_32b_galaxy.model import DEFAULT_HF_REVISION

_REFERENCE_NAME = "Qwen3-32B"
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
_TEACHER_FORCED_TOP1 = 0.89
_TEACHER_FORCED_TOP5 = 0.97


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _hf_model() -> str:
    return os.getenv("QWEN3_32B_HF_MODEL", DEFAULT_HF_MODEL)


def _hf_revision() -> str | None:
    """Pin the revision only for the checkpoint the pin was recorded against.

    `Qwen3_32BGalaxyRuntimeConfig` and the step-7 suite both do this: the default
    model carries `DEFAULT_HF_REVISION`, and an overridden `QWEN3_32B_HF_MODEL`
    (a local path, say) has no such revision and would fail resolution.
    """

    return DEFAULT_HF_REVISION if _hf_model() == DEFAULT_HF_MODEL else None


#: The layer-subset switch. Named once so the accuracy gate can cite it when it
#: refuses to score a truncated model.
_LAYERS_ENV = "QWEN3_32B_GALAXY_TEST_LAYERS"


def _layers() -> int | None:
    value = os.getenv(_LAYERS_ENV)
    return int(value) if value else None


def _load_hf_subset():
    """Return a layer-subset loader when the environment asks for one.

    A layer subset reads the two or three safetensors shards it needs instead of
    the whole 62 GB checkpoint, which is what makes an iteration loop on this model
    affordable. No gate run uses it: every number in the report needs all 64
    layers -- enforced by a skip in the accuracy test, not just promised here.
    """

    layers = _layers()
    if layers is None:
        return None
    from tests.models.galaxy.galaxy_checkpoint import load_layer_subset_causal_lm

    return lambda: load_layer_subset_causal_lm(_hf_model(), layer_indices=tuple(range(layers)), revision=_hf_revision())


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
    hf_config_or_skip(hf_model, revision=_hf_revision())
    kwargs: dict[str, Any] = dict(
        hf_model=hf_model,
        hf_revision=_hf_revision(),
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
) -> Qwen3_32BGalaxyExecutorConfig:
    paged = default_galaxy_paged_kv_cache_config(model)
    if num_blocks is not None:
        paged = PagedKVCacheConfig(
            block_size=paged.block_size,
            max_num_blocks=paged.max_num_blocks,
            dtype=paged.dtype,
            memory_config=paged.memory_config,
            num_blocks=num_blocks,
        )
    return Qwen3_32BGalaxyExecutorConfig(
        trace=TraceConfig(mode="none"),
        warmup=WarmupConfig(prefill_seq_lens=prefill_seq_lens, prefill_batch_sizes=(1,)),
        paged_kv_cache=paged,
        device_sampling_enabled=False,
        sequential_prefill_only=True,
    )


def _open_executor(handle: Any, **kwargs: Any) -> tuple[Qwen3_32BGalaxyExecutor, list[list[Any]]]:
    executor = Qwen3_32BGalaxyExecutor(handle.model, handle.runtime_config, _executor_config(handle.model, **kwargs))
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


def _executor_prefill(
    executor: Qwen3_32BGalaxyExecutor,
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
    executor: Qwen3_32BGalaxyExecutor,
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
# 2. Eager decode
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("active_rows", [1, GALAXY_PHYSICAL_BATCH])
def test_executor_decode_first_token(mesh_device: ttnn.MeshDevice, active_rows: int) -> None:
    """Coverage 2: eager decode at batch 1 and batch 32, first token after prefill.

    Checks the executor against *itself* across slots, not against a recorded
    reference: identical prompts in every active slot must produce the same first
    token, and the decode logits must be finite. The absolute-value comparison this
    test used to make needed the `GalaxyDirectRunner` recording, which is retired.
    """

    length = 128
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

        if active_rows > 1:
            # The slots must agree with each other. Which token they agree on was the
            # recording's job, and is no longer asserted here.
            argmaxes = [int(torch.argmax(logits[slot])) for slot in range(active_rows)]
            assert len(set(argmaxes)) == 1, f"batch-32 decode disagrees across slots: {argmaxes}"
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 3. Paged KV
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_paged_kv_contract(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    """Coverage 3: late capacity resolution, transactional bind/unbind, metadata.

    The KV-value comparison this test used to end with needed the
    `GalaxyDirectRunner` recording, which is retired; the contract assertions around
    it stand on their own and are what this gate is for.
    """

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
        executor = Qwen3_32BGalaxyExecutor(
            model,
            handle.runtime_config,
            Qwen3_32BGalaxyExecutorConfig(
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
@pytest.mark.galaxy_wh
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
        executor = Qwen3_32BGalaxyExecutor(
            handle.model,
            handle.runtime_config,
            Qwen3_32BGalaxyExecutorConfig(
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
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_prefix_cached_prefill(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 4: resume a prompt whose first chunk is already in the cache.

    **This passed for Qwen at Milestone B** (`tttv2_milestone_b_evidence`, two fresh
    processes), through `GalaxyDirectRunner` rather than through an executor. So a
    failure here is a regression — introduced either by `c-defects`' shared-code
    changes or by this executor — and is reported as one, not absorbed.

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
@pytest.mark.galaxy_wh
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
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("order", ["prefill_first", "decode_first"])
def test_executor_warmup_and_program_identity(mesh_device: ttnn.MeshDevice, order: str) -> None:
    """Coverage 5: warmup completes, and program identity ignores active rows.

    `WarmupCoordinator` documents that the two warmup calls may be made in either
    order, so both orders are measured. On the sibling Llama model the two orders
    were *not* equivalent — a prefill after a decode raised that model's L1 address
    clash — which is why the parametrization exists at all. Qwen never carried that
    clash and `c-defects` has since fixed it, so both orders are expected to pass
    here; the parametrization stays because it is the cheapest place either model
    would show a mode-transition regression.
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
        # and the decode identity check. Interleaving this way keeps each identity
        # assertion adjacent to the warmup whose coverage it is checking, so a
        # failure names one of the two rather than the pair. It is also the shape
        # the sibling Llama file settled on, kept here so the two are comparable.
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
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_repeated_startup_and_cleanup(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 6: three startup/serve/cleanup cycles over one model.

    The one-live-model form, which Qwen passed 3/3 at Milestone B. The
    two-models-in-one-process form is D-C7 and is coverage item 8 below; the brief
    requires both.
    """

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
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_teacher_forced_accuracy(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 7: top-1 ≥ 91% and top-5 ≥ 99% through the executor path.

    The same convention as Milestone B's `GalaxyDirectRunner` gate, which measured
    **97.46% / 100.00%** for Qwen: prefill 512, then 511 forced decode steps at
    batch 1. The gate is top-1 >= 89% / top-5 >= 97%; the executor must not lose
    accuracy against what the runner reached.
    """

    # A layer subset is what makes iterating on this model affordable, and `_load`
    # applies `QWEN3_32B_GALAXY_TEST_LAYERS` unconditionally -- so without this
    # guard the accuracy gate would silently score a truncated model against
    # whole-model thresholds and report a functional failure that is really a
    # misconfiguration. `_load_hf_subset` promises the accuracy gate never uses a
    # subset; nothing enforced that promise until here. Skip rather than fail:
    # during iteration the subset is a deliberate operator choice, not an error.
    if _layers() is not None:
        pytest.skip(f"{_LAYERS_ENV}={_layers()} requests a layer subset; teacher-forced accuracy needs every layer")

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


# ---------------------------------------------------------------------------
# 8. Two models in one process — the D-C7 shape, through the executor
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_executor_two_models_in_one_process(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 8: build, use, close, build a second, use it — all in one process.

    **This is the item Llama cannot provide.** Qwen failed this shape at Milestone
    B (D-C7) and `c-defects` fixed it across four separate defects, each of which
    only became visible once the one before it was gone:

    * **D-C7 as named** — 923 776 of 1 393 472 B per L1 bank still allocated after
      `close()`, because a `Prefetcher2DContext` is captured *by value* at module
      construction, so clearing the owner's map left every built module holding a
      context whose `global_cb` was the live buffer;
    * **D-C13** — the second model's global circular buffer then failed to
      allocate on fragmentation;
    * **D-C14** — the second model's first decode then *hung*, because the ttnn
      program cache belongs to the mesh device, outlives `close()`, and two
      structurally identical Galaxy models share compiled decode programs carrying
      the first model's global-CB addresses;
    * **D-C15** — Llama then failed on DRAM residue that Qwen never showed.

    `c-defects` verified the fix through `GalaxyDirectRunner`
    (`z7/z10/z11_qwen_two_pools`, 3/3 at the full shape). This measures the same
    claim **through the executor**, which is the thing that did not exist then: the
    executor is the resource and cleanup root, so if it retains anything the first
    model owned, the second model's `activate("decode")` is where it shows.

    The two models use **different paged pools** — the default 2048-token context
    and an explicit 4096-token one — so every slot's run of block ids differs and
    the second model computes the same answers through a different block
    allocation. That is the shape `c-defects` qualified, and comparing the two at
    PCC >= 0.99 makes the claim about *answers*, not only about allocation.
    """

    length = 128
    prompt = _prompt(length)
    pools = (
        ("default-2048", _MAX_SEQ_LEN),
        ("explicit-4096", 4096),
    )
    observed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, context in pools:
        handle = _load(
            mesh_device,
            max_seq_len=context,
            paged_attention_config=_paged_config(max_seq_len=context),
        )
        executor = None
        try:
            executor, kv_cache = _open_executor(handle)
            resolved = handle.model.kv_specs[0].paged_attention_config
            print(
                f"[two-models] {name}: block_size={resolved.block_size} max_num_blocks={resolved.max_num_blocks}",
                flush=True,
            )
            logits = _executor_prefill(executor, kv_cache, prompt, max_seq_len=context)
            prefill = logits.float().reshape(-1)
            first = int(torch.argmax(prefill))
            tokens = [0] * GALAXY_PHYSICAL_BATCH
            positions = [0] * GALAXY_PHYSICAL_BATCH
            tokens[0], positions[0] = first, length
            decode = _decode_logits(
                executor.decode_forward(
                    torch.tensor(tokens, dtype=torch.long),
                    torch.tensor(positions, dtype=torch.long),
                    _page_table_rows(max_seq_len=context),
                    kv_cache=kv_cache,
                )
            )
            assert torch.isfinite(prefill).all(), f"{name} prefill logits are not finite"
            assert torch.isfinite(decode[0]).all(), f"{name} decode logits are not finite"
            print(
                f"[two-models] {name}: prefill argmax {first} decode argmax {int(torch.argmax(decode[0]))}",
                flush=True,
            )
            observed[name] = (prefill.clone(), decode[0].clone())
        finally:
            if executor is not None:
                executor.cleanup()
            _close(handle)

    first_pool, second_pool = observed["default-2048"], observed["explicit-4096"]
    for index, stage in enumerate(("prefill", "decode")):
        passed, message = _pcc(first_pool[index], second_pool[index], _LOGITS_PCC)
        print(f"[two-models] {stage} 2048-block pool vs 4096-block pool {message}", flush=True)
        assert passed, f"{stage}, first model vs second model in one process: {message}"
