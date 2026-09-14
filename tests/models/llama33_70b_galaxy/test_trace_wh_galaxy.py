# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone C tracing coverage for `Llama33_70BGalaxyExecutor` on WH Galaxy `(8, 4)`.

One test per coverage item of `tttv2_milestone_c_briefs/c3_trace.md`:

1. traced decode at batch 32;
2. traced prefill where eligible (sequential, one row per request);
3. a trace-ineligible request handled by an explicit eager path, with the miss
   surfaced as `TraceCoverageError` rather than absorbed;
4. eager-vs-traced decode logits PCC ≥ 0.999 for the same prepared request;
5. identical deterministic device-sampled tokens between the two paths;
6. trace identity is physical geometry — different active row counts at one
   geometry hit one trace;
7. paged-KV late capacity resolution under tracing;
8. the six mode transitions, each across compile, capture, replay and cleanup;
9. repeated startup / serving / cleanup with tracing active, nothing retained.

**Why this file is not `test_executor_wh_galaxy.py`.** That file constructs
`GalaxyDirectRunner` to compute the reference its claims are checked against. The
runner is frozen and scheduled for removal (`tttv2_milestone_c_briefs/README.md`,
decision of 2026-09-02), and — more to the point — a traced-vs-runner comparison
is a *different, weaker* claim than the one tracing needs. Every parity number
here is eager-vs-traced: **the same graph twice**, through the same executor, in
the same process, which is what the 0.999 threshold is for. Nothing in this file
imports the runner and no expected value here comes from it.

Read every number this file prints as **observed (1 run)**. The one-run rule
replaced the three-run rule on 2026-09-01; no claim here carries repeats.

Run one node id per process, as the house rules require::

    pytest tests/models/llama33_70b_galaxy/test_trace_wh_galaxy.py \
        -v -rA --color=no -p no:cacheprovider -k traced_decode_batch32
"""

from __future__ import annotations

import dataclasses
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
    hf_config_or_skip,
    load_reference_tokens,
)

from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from tt_transformers.llm_runtime.execution import TraceCoverageError
from tt_transformers.models.galaxy.kv_contract import GalaxyPagedAttentionConfig
from tt_transformers.models.llama33_70b_galaxy.executor import (
    Llama33_70BGalaxyExecutor,
    Llama33_70BGalaxyExecutorConfig,
    default_galaxy_paged_kv_cache_config,
)
from tt_transformers.models.llama33_70b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, from_pretrained
from tt_transformers.sampling import SamplingParams

_REFERENCE_NAME = "Llama-3.3-70B-Instruct"
_BLOCK_SIZE = 32
_MAX_SEQ_LEN = 2048
#: The recipes this file's model is built with. 512-token prompts pad to a
#: 1024-token device request, which is the shape items 3 and the failure
#: transition use as their *trace-ineligible* case.
_RECIPE_LENGTHS = (128, 1024)
#: The lane's trace policy, narrower than the registered recipe set on purpose.
#: `from_pretrained` sets `trace_prefill_supported_seq_lens` equal to the
#: registered recipes, so a request that has a recipe but no trace can only be
#: expressed by narrowing the *serving policy* — which is exactly what it is.
_TRACED_LENGTHS = (128,)
_PROMPT_LENGTH = 128
_INELIGIBLE_PROMPT_LENGTH = 512
#: The same graph twice, so the threshold is tighter than a reference comparison.
_TRACE_PCC = 0.999

#: **The mesh has to be opened with a trace region and `GALAXY_DEVICE_PARAMS`
#: carries none**, so every Galaxy device test in this milestone so far ran with
#: `trace_region_size=0` — which tells the runtime to allocate trace buffers
#: dynamically rather than out of a reserved region. This is the published
#: production value for this model on `wh_galaxy_perf`
#: (`models/model_trace_region_sizes.yaml`, key `llama3.3-70b-galaxy`), reserved
#: explicitly because `get_supported_trace_region_size` resolves no SKU for a
#: bare `(8, 4)` mesh parameter on this host ("No trace region size for (8, 4)").
_TRACE_REGION_SIZE = 216580672
GALAXY_TRACE_DEVICE_PARAMS = {**GALAXY_DEVICE_PARAMS, "trace_region_size": _TRACE_REGION_SIZE}

#: The trace policy every test in this file asks for. `"all"` registers the
#: prefill *and* decode coverage, which is what coverage items 2 and 8 need. It
#: cost attempt 1 three device runs to get here: the decode capture body ended in
#: a host barrier the Galaxy collectives perform in-graph (`s3`), and the prefill
#: capture then deadlocked inside `ttnn.end_trace_capture` because the persistent
#: `ttnn.dram_prefetcher` launched at the decode operation boundary had no
#: consumer inside a capture region and stranded itself on the prefetch sender
#: cores, which the prefill subdevice covers (`s4`, and its gdb backtrace).
#: Both are fixed in the model packages; this file asks for the full policy.
_DEFAULT_TRACE_MODE = "all"


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _hf_model() -> str:
    return os.getenv("LLAMA33_70B_HF_MODEL", DEFAULT_HF_MODEL)


def _layers() -> int | None:
    value = os.getenv("LLAMA33_70B_GALAXY_TEST_LAYERS")
    return int(value) if value else None


def _load_hf_subset():
    layers = _layers()
    if layers is None:
        return None
    from tests.models.galaxy.galaxy_checkpoint import load_layer_subset_causal_lm

    return lambda: load_layer_subset_causal_lm(_hf_model(), layer_indices=tuple(range(layers)))


def _blocks_per_user(max_seq_len: int = _MAX_SEQ_LEN) -> int:
    return -(-max_seq_len // _BLOCK_SIZE)


def _paged_config(active_slots: int = GALAXY_PHYSICAL_BATCH) -> GalaxyPagedAttentionConfig:
    per_user = _blocks_per_user()
    sinks = GALAXY_PHYSICAL_BATCH - active_slots
    return GalaxyPagedAttentionConfig(block_size=_BLOCK_SIZE, max_num_blocks=per_user * active_slots + sinks)


def _page_table_rows(active_slots: int = GALAXY_PHYSICAL_BATCH) -> torch.Tensor:
    """Return the `[32, blocks_per_user]` static block-ownership table."""

    per_user = _blocks_per_user()
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
        # Every warmup plan carries one cached (prefix) prefill case per
        # configured length, so the chunked recipe has to be registered for the
        # coordinator — and therefore for `capture_all` — to complete at all.
        chunked_prefill_sequence_lengths=(128,),
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


def _runtime_config(handle: Any) -> Any:
    """Narrow the lane's trace policy so an ineligible request is expressible."""

    return dataclasses.replace(handle.runtime_config, trace_prefill_supported_seq_lens=_TRACED_LENGTHS)


def _executor_config(
    model: Any,
    *,
    mode: str = _DEFAULT_TRACE_MODE,
    num_blocks: int | None = None,
    resolved: bool = True,
    device_sampling: bool = False,
) -> Llama33_70BGalaxyExecutorConfig:
    paged = default_galaxy_paged_kv_cache_config(model)
    paged = PagedKVCacheConfig(
        block_size=paged.block_size,
        max_num_blocks=paged.max_num_blocks,
        dtype=paged.dtype,
        memory_config=paged.memory_config,
        num_blocks=(num_blocks if num_blocks is not None else paged.max_num_blocks) if resolved else None,
    )
    return Llama33_70BGalaxyExecutorConfig(
        trace=TraceConfig(mode=mode),
        warmup=WarmupConfig(prefill_seq_lens=_TRACED_LENGTHS, prefill_batch_sizes=(1,)),
        paged_kv_cache=paged,
        device_sampling_enabled=device_sampling,
        sequential_prefill_only=True,
    )


def _open(handle: Any, **kwargs: Any) -> tuple[Llama33_70BGalaxyExecutor, list[list[Any]]]:
    executor = Llama33_70BGalaxyExecutor(
        handle.model, _runtime_config(handle), _executor_config(handle.model, **kwargs)
    )
    return executor, executor.allocate_kv_cache()


def _warm(
    executor: Any,
    kv_cache: Any,
    *,
    order: tuple[str, ...] = ("prefill", "decode"),
    trace_prefill: bool = True,
    trace_decode: bool = True,
    sample_decode: bool = False,
) -> None:
    """Compile the configured coverage and capture it, in the given order.

    `WarmupCoordinator._maybe_capture` fires `capture_all()` from whichever of
    the two public calls completes the configured plan second, so both orders are
    exercised by the transition matrix rather than assumed equivalent.
    """

    for operation in order:
        if operation == "prefill":
            # `WarmupCoordinator._validate_hints` refuses a prefill trace warmup
            # that exceeds the configured policy, so the hint follows the policy.
            executor.warmup_model_prefill(
                kv_cache=kv_cache,
                can_sample_on_device=False,
                enable_trace=trace_prefill and executor.config.trace.prefill_enabled,
            )
        else:
            executor.warmup_model_decode(
                kv_cache=kv_cache,
                can_sample_on_device=sample_decode,
                enable_trace=trace_decode,
            )


def _prompt(length: int = _PROMPT_LENGTH) -> list[int]:
    reference_tokens, _ = load_reference_tokens(_REFERENCE_NAME)
    source = [int(value) for value in reference_tokens]
    repeats = -(-length // len(source))
    return (source * repeats)[:length]


def _pcc(expected: torch.Tensor, actual: torch.Tensor, threshold: float):
    from tests.support.comparison import comp_pcc

    return comp_pcc(expected.unsqueeze(0).float(), actual.unsqueeze(0).float(), threshold)


def _prefill(executor: Any, kv_cache: Any, prompt: list[int], *, slot: int = 0, rows=None, execution=None):
    rows = _page_table_rows() if rows is None else rows
    return executor.prefill_forward(
        torch.tensor(prompt, dtype=torch.long).reshape(1, -1),
        rows[slot : slot + 1],
        prompt_lens=torch.tensor([len(prompt)], dtype=torch.long),
        empty_slots=[slot],
        kv_cache=kv_cache,
        execution=execution,
    )


def _decode(
    executor: Any,
    kv_cache: Any,
    tokens: list[int],
    positions: list[int],
    *,
    rows=None,
    execution=None,
    sampling_params: Any = None,
):
    rows = _page_table_rows() if rows is None else rows
    return executor.decode_forward(
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
        rows,
        kv_cache=kv_cache,
        execution=execution,
        sampling_params=sampling_params,
    )


def _decode_logits(result: Any) -> torch.Tensor:
    logits, _ = result if isinstance(result, tuple) else (result, None)
    return logits.float().reshape(GALAXY_PHYSICAL_BATCH, -1)


def _decode_tokens(result: Any) -> torch.Tensor:
    tokens, _ = result if isinstance(result, tuple) else (result, None)
    return tokens.reshape(-1)[:GALAXY_PHYSICAL_BATCH].to(torch.int64)


def _greedy_params() -> SamplingParams:
    """Deterministic device sampling: every slot forced onto its own argmax.

    `format_sampling_params` and `Sampling2D._update_call_buffers` both encode
    `temperature == 0` as ``top_k=1, top_p=0``, so there is exactly one candidate
    per row and no draw to make. That is what makes item 5's comparison a claim
    about eager-vs-traced identity rather than about a random stream.
    """

    return SamplingParams(
        temperature=[0.0] * GALAXY_PHYSICAL_BATCH,
        top_k=[1] * GALAXY_PHYSICAL_BATCH,
        top_p=[1.0] * GALAXY_PHYSICAL_BATCH,
    )


def _summary(executor: Any, label: str) -> dict[str, Any]:
    summary = executor.traced_executor.log_runtime_summary(phase=label)
    print(f"[trace] {label} {summary}", flush=True)
    return summary


def _assert_nothing_retained(executor: Any, kv_cache: Any, model: Any, label: str) -> None:
    assert not any(tensor.is_allocated() for pair in kv_cache for tensor in pair), (
        f"{label} retained KV tensors after cleanup"
    )
    assert all(layer.attention.kv_cache_binding is None for layer in model.layers), (
        f"{label} left the model bound to a released cache"
    )
    assert executor.prefill_runtime.transient_orphan_count == 0, f"{label} retained prefill transients"
    assert executor.decode_runtime.transient_orphan_count == 0, f"{label} retained decode transients"
    assert executor.program_compiler.compile_orphan_count == 0, f"{label} retained compile outputs"
    assert not executor.trace_compiler.trace_active, f"{label} left the trace active"
    assert executor.terminal, f"{label} did not become terminal"


# ---------------------------------------------------------------------------
# 1. Traced decode, batch 32
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_traced_decode_batch32(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 1: one captured decode trace serves a full physical batch."""

    handle = _load(mesh_device)
    executor = None
    try:
        executor, kv_cache = _open(handle)
        _warm(executor, kv_cache)
        assert executor.trace_compiler.trace_active, "warmup did not capture"

        prompt = _prompt()
        first_tokens = []
        for slot in range(GALAXY_PHYSICAL_BATCH):
            logits = _prefill(executor, kv_cache, prompt, slot=slot)
            first_tokens.append(int(torch.argmax(logits.float().reshape(-1))))
        assert len(set(first_tokens)) == 1, f"identical prompts produced different first tokens: {first_tokens}"

        before = executor.trace_compiler.replay_counts["decode"]
        logits = _decode_logits(_decode(executor, kv_cache, first_tokens, [_PROMPT_LENGTH] * GALAXY_PHYSICAL_BATCH))
        after = executor.trace_compiler.replay_counts["decode"]
        assert after == before + 1, f"traced decode submitted {after - before} replays, expected 1"
        assert torch.isfinite(logits).all(), "traced decode logits are not finite"
        argmaxes = [int(torch.argmax(logits[slot])) for slot in range(GALAXY_PHYSICAL_BATCH)]
        print(f"[trace] traced decode batch 32 argmaxes {sorted(set(argmaxes))}", flush=True)
        assert len(set(argmaxes)) == 1, f"batch-32 traced decode disagrees across slots: {argmaxes}"
        _summary(executor, "traced_decode_batch32")
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 2. Traced prefill where eligible
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_traced_prefill_where_eligible(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 2: a sequential single-row prefill replays its captured trace."""

    handle = _load(mesh_device)
    executor = None
    try:
        executor, kv_cache = _open(handle)
        _warm(executor, kv_cache)
        prompt = _prompt()
        tokens = torch.tensor(prompt, dtype=torch.long).reshape(1, -1)
        assert executor.can_trace_prefill(
            tokens=tokens,
            prompt_lens=torch.tensor([len(prompt)], dtype=torch.long),
            empty_slots=[0],
        ), "a configured 128-token prefill classified as untraceable"

        before = executor.trace_compiler.replay_counts["prefill"]
        eager_before = executor.eager_executor.eager_prefill_count
        logits = _prefill(executor, kv_cache, prompt, slot=0)
        after = executor.trace_compiler.replay_counts["prefill"]

        assert after > before, "the prefill call submitted no trace replay"
        assert executor.eager_executor.eager_prefill_count == eager_before, (
            "a traced prefill fell back to the eager path"
        )
        evidence = executor.traced_executor.recent_prefill_replay_evidence
        assert len(evidence) == 1, f"expected one prefill replay evidence record, got {evidence}"
        item = evidence[0]
        assert item.execution == "trace_replay", item
        assert item.variant == "regular-single", item
        assert item.active_batch_size == 1 and item.padded_batch_size == 1, item
        assert item.padded_sequence_length == _PROMPT_LENGTH, item
        assert item.trace_key != "unassociated", item
        assert torch.isfinite(logits.float()).all(), "traced prefill logits are not finite"
        print(
            f"[trace] traced prefill argmax {int(torch.argmax(logits.float().reshape(-1)))} "
            f"steps={item.replay_steps} trace_key={item.trace_key}",
            flush=True,
        )
        _summary(executor, "traced_prefill")
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 3. Explicit eager handling for a trace-ineligible request
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_trace_ineligible_prefill_is_an_explicit_eager_path(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    """Coverage 3: the miss is raised at the boundary, never absorbed.

    The ineligible request is a 512-token prompt, which the planner pads to a
    1024-token device request. The model has a 1024 recipe, so the *graph* exists;
    the lane's trace policy does not cover it, so the trace does not. Its eager
    program is compiled before capture, because `ProgramCompiler` refuses new
    compilations once a trace is active — which is precisely why an eager path for
    ineligible traffic has to be an explicit construction-time decision rather
    than a runtime fallback.
    """

    handle = _load(mesh_device)
    executor = None
    try:
        executor, kv_cache = _open(handle)
        long_prompt = _prompt(_INELIGIBLE_PROMPT_LENGTH)
        long_tokens = torch.tensor(long_prompt, dtype=torch.long).reshape(1, -1)
        long_lens = torch.tensor([len(long_prompt)], dtype=torch.long)
        rows = _page_table_rows()

        assert not executor.can_trace_prefill(tokens=long_tokens, prompt_lens=long_lens, empty_slots=[1]), (
            "the ineligible length classified as traceable"
        )

        # The explicit eager path, declared before capture.
        executor.compile_prefill(
            tokens=long_tokens,
            page_table=rows[1:2],
            prompt_lens=long_lens,
            empty_slots=[1],
            kv_cache=kv_cache,
            execution=executor.eager_executor,
        )
        _warm(executor, kv_cache)
        assert executor.trace_compiler.trace_active

        misses_before = executor.traced_executor.coverage_miss_count
        with expect_error(TraceCoverageError, "Required prefill trace is unavailable") as raised:
            _prefill(executor, kv_cache, long_prompt, slot=1, execution=executor.traced_executor)
        message = str(raised.value)
        assert "Required prefill trace is unavailable" in message
        assert "not trace-eligible" in message
        assert "configured_coverage=" in message
        assert executor.traced_executor.coverage_miss_count == misses_before + 1, "the coverage miss was not counted"
        print(f"[trace] coverage miss surfaced: {message[:220]}", flush=True)

        eager_before = executor.eager_executor.eager_prefill_count
        replays_before = executor.trace_compiler.replay_counts["prefill"]
        logits = _prefill(executor, kv_cache, long_prompt, slot=1, execution=executor.eager_executor)
        assert executor.eager_executor.eager_prefill_count == eager_before + 1, (
            "the explicit eager path did not execute"
        )
        assert executor.trace_compiler.replay_counts["prefill"] == replays_before, (
            "the eager path submitted a trace replay"
        )
        assert torch.isfinite(logits.float()).all(), "eager prefill of the ineligible request is not finite"
        assert executor.program_compiler.post_activation_compile_rejections == 0, (
            "the eager path compiled a program after trace activation"
        )
        print(
            f"[trace] explicit eager prefill argmax {int(torch.argmax(logits.float().reshape(-1)))}",
            flush=True,
        )
        _summary(executor, "trace_ineligible_eager")
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 4 and 6. Eager vs traced logits, and trace identity
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_eager_and_traced_decode_logits_agree(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 4: the same prepared request, both paths, PCC ≥ 0.999.

    This is an eager-vs-traced identity measured once in one process. It is **not**
    a determinism claim across processes and must not be written up as one.
    """

    handle = _load(mesh_device)
    executor = None
    try:
        executor, kv_cache = _open(handle)
        _warm(executor, kv_cache)
        prompt = _prompt()
        first = int(torch.argmax(_prefill(executor, kv_cache, prompt, slot=0).float().reshape(-1)))
        tokens = [first] + [0] * (GALAXY_PHYSICAL_BATCH - 1)
        positions = [_PROMPT_LENGTH] + [0] * (GALAXY_PHYSICAL_BATCH - 1)

        eager = _decode_logits(_decode(executor, kv_cache, tokens, positions, execution=executor.eager_executor))
        traced = _decode_logits(_decode(executor, kv_cache, tokens, positions, execution=executor.traced_executor))
        assert torch.isfinite(eager[0]).all() and torch.isfinite(traced[0]).all()

        passed, message = _pcc(eager[0], traced[0], _TRACE_PCC)
        print(f"[trace] eager vs traced decode row 0 {message}", flush=True)
        eager_argmax = int(torch.argmax(eager[0]))
        traced_argmax = int(torch.argmax(traced[0]))
        print(f"[trace] argmax eager {eager_argmax} traced {traced_argmax}", flush=True)
        assert passed, f"eager vs traced decode logits PCC below {_TRACE_PCC}: {message}"
        assert eager_argmax == traced_argmax, f"eager argmax {eager_argmax} != traced {traced_argmax}"
        _summary(executor, "eager_vs_traced_logits")
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_eager_and_traced_decode_logits_agree_with_device_sampling_built(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 4 again, in the configuration coverage 5 actually runs in.

    The node above builds with `enable_device_sampling=False`, so warmup records
    **three** capture bodies and its PCC is a claim about that configuration
    only. Coverage 5 builds with device sampling on, which adds a fourth
    (`sampling_path='topk'`) decode capture body - and on 2026-09-04 that
    difference was measured to matter. The same logits trace that is
    `max_abs_delta=0` on 14 of 14 replays with three bodies was **0 of 10** with
    four (`tttv2_milestone_c_evidence/trace/logs/q8_Q_logits_trace_alone.log`),
    because the decode user gather's fabric-write destination was allocated and
    freed inside a capture body while the two decode bodies recycle L1 addresses
    between them. `c1f30128e13` closed that carrier.

    This node exists so coverage 4 is not a statement about a configuration
    nothing else runs in. It asserts the **same** threshold as the node above -
    nothing is relaxed - and additionally **prints `max_abs_delta`**, because a
    1-3-ulp residual passes a PCC-and-argmax assertion unnoticed and a rounded
    `1.0` in a log is not evidence of exactness. Asked for by the 2026-09-04
    advisory, §7.3.
    """

    handle = _load(mesh_device, enable_device_sampling=True)
    executor = None
    try:
        executor, kv_cache = _open(handle, device_sampling=True)
        _warm(executor, kv_cache, sample_decode=True)
        assert executor.trace_compiler.trace_active, "warmup did not capture"
        bodies = len(executor.trace_compiler._traces)
        print(f"[trace] semantic_trace_count={bodies}", flush=True)

        # Coverage 5's request shape, so this is that configuration end to end and
        # not only its build: every slot decoding at the same position.
        prompt = _prompt()
        first = int(torch.argmax(_prefill(executor, kv_cache, prompt, slot=0).float().reshape(-1)))
        tokens = [first] * GALAXY_PHYSICAL_BATCH
        positions = [_PROMPT_LENGTH] * GALAXY_PHYSICAL_BATCH

        eager = _decode_logits(_decode(executor, kv_cache, tokens, positions, execution=executor.eager_executor))
        replays_before = executor.trace_compiler.replay_counts["decode"]
        traced = _decode_logits(_decode(executor, kv_cache, tokens, positions, execution=executor.traced_executor))
        assert executor.trace_compiler.replay_counts["decode"] == replays_before + 1, (
            "the comparison did not replay a trace, so it is not an eager-vs-traced claim"
        )
        assert torch.isfinite(eager).all() and torch.isfinite(traced).all()

        # Reported, not asserted: the threshold this node gates on is the brief's,
        # and exactness is not one of the brief's gates. It is printed because the
        # residual is the thing the gate could not previously see.
        print(f"[trace] eager vs traced max_abs_delta over all slots {float((traced - eager).abs().max())}", flush=True)
        print(
            "[trace] per-slot max_abs_delta "
            + ", ".join(f"{slot}={float((traced[slot] - eager[slot]).abs().max())}" for slot in range(4)),
            flush=True,
        )
        exact = [
            slot for slot in range(GALAXY_PHYSICAL_BATCH) if float((traced[slot] - eager[slot]).abs().max()) == 0.0
        ]
        print(f"[trace] slots bit-identical to eager: {len(exact)} of {GALAXY_PHYSICAL_BATCH}", flush=True)
        argmax_differing = [
            slot for slot in range(GALAXY_PHYSICAL_BATCH) if int(traced[slot].argmax()) != int(eager[slot].argmax())
        ]
        print(f"[trace] slots whose argmax differs: {argmax_differing}", flush=True)

        passed, message = _pcc(eager[0], traced[0], _TRACE_PCC)
        print(f"[trace] eager vs traced decode row 0 {message}", flush=True)
        eager_argmax = int(torch.argmax(eager[0]))
        traced_argmax = int(torch.argmax(traced[0]))
        print(f"[trace] argmax eager {eager_argmax} traced {traced_argmax}", flush=True)
        assert passed, f"eager vs traced decode logits PCC below {_TRACE_PCC}: {message}"
        assert eager_argmax == traced_argmax, f"eager argmax {eager_argmax} != traced {traced_argmax}"
        _summary(executor, "eager_vs_traced_logits_device_sampling_built")
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_trace_identity_is_physical_geometry(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 6: different active row counts, one physical geometry, one trace.

    Active-row and slot data reach the graph only as the contents of the
    persistent decode inputs, so neither may appear in program or trace identity.
    """

    handle = _load(mesh_device)
    executor = None
    try:
        executor, kv_cache = _open(handle)
        _warm(executor, kv_cache)
        traces_after_capture = executor.trace_compiler.trace_count
        associations = executor.trace_compiler.trace_association_count
        programs = len(executor.program_compiler.compiled_programs)

        prompt = _prompt()
        first = int(torch.argmax(_prefill(executor, kv_cache, prompt, slot=0).float().reshape(-1)))

        rows = _page_table_rows()
        one_active = (
            [first] + [0] * (GALAXY_PHYSICAL_BATCH - 1),
            [_PROMPT_LENGTH] + [0] * (GALAXY_PHYSICAL_BATCH - 1),
        )
        all_active = ([first] * GALAXY_PHYSICAL_BATCH, [_PROMPT_LENGTH] * GALAXY_PHYSICAL_BATCH)

        keys = []
        for label, (tokens, positions) in (("one_active_row", one_active), ("all_active_rows", all_active)):
            prepared = executor.decode_runtime.prepare(
                tokens=torch.tensor(tokens, dtype=torch.long),
                start_pos=torch.tensor(positions, dtype=torch.long),
                page_table=rows,
            )
            signature = executor.decode_runtime.program_signature(prepared)
            program_key = executor.program_compiler.key_for(signature)
            trace_key = executor.trace_compiler.trace_key_for_program(program_key)
            assert trace_key is not None, f"{label} resolved no trace association"
            keys.append((label, program_key.digest, trace_key.digest))

            before = executor.trace_compiler.replay_counts["decode"]
            logits = _decode_logits(_decode(executor, kv_cache, tokens, positions))
            assert executor.trace_compiler.replay_counts["decode"] == before + 1
            assert torch.isfinite(logits[0]).all(), f"{label} produced non-finite logits"

        print(f"[trace] identity {keys}", flush=True)
        assert keys[0][1] == keys[1][1], f"active rows changed the program key: {keys}"
        assert keys[0][2] == keys[1][2], f"active rows changed the trace key: {keys}"
        assert executor.trace_compiler.trace_count == traces_after_capture, (
            "serving compiled or captured a second trace"
        )
        assert executor.trace_compiler.trace_association_count == associations
        assert len(executor.program_compiler.compiled_programs) == programs
        assert executor.program_compiler.post_activation_compile_rejections == 0
        _summary(executor, "trace_identity")
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 5. Identical deterministic sampled tokens
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_eager_and_traced_sampled_tokens_are_identical(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 5: device-sampled greedy tokens agree between the two paths."""

    handle = _load(mesh_device, enable_device_sampling=True)
    executor = None
    try:
        executor, kv_cache = _open(handle, device_sampling=True)
        _warm(executor, kv_cache, sample_decode=True)
        assert executor.trace_compiler.trace_active

        prompt = _prompt()
        first = int(torch.argmax(_prefill(executor, kv_cache, prompt, slot=0).float().reshape(-1)))
        tokens = [first] * GALAXY_PHYSICAL_BATCH
        positions = [_PROMPT_LENGTH] * GALAXY_PHYSICAL_BATCH

        params = _greedy_params()
        eager = _decode_tokens(
            _decode(executor, kv_cache, tokens, positions, execution=executor.eager_executor, sampling_params=params)
        )
        replays_before = executor.trace_compiler.replay_counts["decode"]
        traced = _decode_tokens(
            _decode(executor, kv_cache, tokens, positions, execution=executor.traced_executor, sampling_params=params)
        )
        assert executor.trace_compiler.replay_counts["decode"] == replays_before + 1, (
            "the sampled decode did not replay a trace"
        )

        print(f"[trace] sampled tokens eager {eager.tolist()}", flush=True)
        print(f"[trace] sampled tokens traced {traced.tolist()}", flush=True)
        mismatches = [slot for slot in range(GALAXY_PHYSICAL_BATCH) if int(eager[slot]) != int(traced[slot])]
        assert not mismatches, f"eager and traced sampled tokens differ in slots {mismatches}"
        _summary(executor, "eager_vs_traced_sampled_tokens")
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 7. Paged-KV late capacity resolution under tracing
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_paged_kv_late_capacity_under_tracing(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    """Coverage 7: capacity resolved after construction, then captured and replayed.

    The resolved pool is smaller than the construction ceiling, which is the branch
    that moves the model's own per-layer metadata and rebuilds the KV manager
    against it. It is deliberately not the *shrink to one active slot* case: that
    is D-C19, an open `Attention2D` paged-KV contract defect with a page-table
    width bound by the pool size, already reported with its log by `c-exec-llama`
    and independently reproduced on Qwen by `c-exec-qwen`. Nothing here relaxes it.
    """

    active_slots = 15
    physical = _blocks_per_user() * active_slots + (GALAXY_PHYSICAL_BATCH - active_slots)
    handle = _load(mesh_device)
    executor = None
    try:
        ceiling = default_galaxy_paged_kv_cache_config(handle.model)
        assert physical < ceiling.max_num_blocks
        executor = Llama33_70BGalaxyExecutor(
            handle.model,
            _runtime_config(handle),
            _executor_config(handle.model, resolved=False),
        )
        assert executor.paged_kv_cache_config.num_blocks is None
        with expect_error(RuntimeError, "capacity must be resolved before allocation"):
            executor.allocate_kv_cache()

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

        _warm(executor, kv_cache)
        assert executor.trace_compiler.trace_active, "capture did not complete on a late-resolved pool"

        rows = _page_table_rows(active_slots=active_slots)
        assert int(rows.max()) < physical
        prompt = _prompt()
        logits = _prefill(executor, kv_cache, prompt, slot=0, rows=rows)
        first = int(torch.argmax(logits.float().reshape(-1)))
        tokens = [first] + [0] * (GALAXY_PHYSICAL_BATCH - 1)
        positions = [_PROMPT_LENGTH] + [0] * (GALAXY_PHYSICAL_BATCH - 1)
        before = executor.trace_compiler.replay_counts["decode"]
        decode = _decode_logits(_decode(executor, kv_cache, tokens, positions, rows=rows))
        assert executor.trace_compiler.replay_counts["decode"] == before + 1
        assert torch.isfinite(decode[0]).all()
        print(
            f"[trace] late capacity {physical} blocks: prefill {first} decode {int(torch.argmax(decode[0]))}",
            flush=True,
        )
        _summary(executor, "late_capacity_tracing")
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


# ---------------------------------------------------------------------------
# 8. Mode transitions
# ---------------------------------------------------------------------------


_TRANSITIONS = (
    "decode_to_prefill",
    "prefill_to_decode",
    "repeated_prefill",
    "repeated_decode",
    "failure_during_transition",
    "cleanup_from_either_mode",
)


def _report_transition(name: str, phases: dict[str, Any]) -> None:
    ordered = " ".join(f"{phase}={phases[phase]}" for phase in ("compile", "capture", "replay", "cleanup"))
    print(f"[transition] {name} {ordered}", flush=True)


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("transition", _TRANSITIONS)
def test_mode_transition(mesh_device: ttnn.MeshDevice, transition: str, expect_error) -> None:
    """Coverage 8: compile, capture, replay and cleanup across one transition.

    The hazard is prefill/decode semaphore or stall-group state leaking across
    capture and replay: `GalaxyExecutionResources.activate` loads a different
    sub-device manager per mode, sets a different stall group, and starts the
    persistent prefetcher for decode only. Milestone A's D3 burned a night on
    exactly that class of fault.
    """

    handle = _load(mesh_device)
    prompt = _prompt()
    long_prompt = _prompt(_INELIGIBLE_PROMPT_LENGTH)
    executor = None
    phases: dict[str, Any] = {
        "compile": "not reached",
        "capture": "not reached",
        "replay": "not reached",
        "cleanup": "not reached",
    }
    try:
        # Warmup order is fixed, and that is a property of the runtime rather
        # than a convenience. `WarmupCoordinator._maybe_capture` fires
        # `capture_all()` as soon as the configured plan is complete, and
        # `ProgramCompiler` refuses every compilation after trace activation. So
        # under `decode_only` the decode warmup has to be last: warming decode
        # first would capture, and the prefill warmup that followed would be
        # counted in `post_activation_compile_rejections` instead of compiling.
        # The transition each case names is therefore the *serving* order, which
        # is the one the mode-transition hazard lives in.
        order = ("prefill", "decode")
        executor, kv_cache = _open(handle)
        _warm(executor, kv_cache, order=order)

        programs = len(executor.program_compiler.compiled_programs)
        assert programs >= 2, f"warmup compiled {programs} programs"
        assert executor.program_compiler.post_activation_compile_rejections == 0
        phases["compile"] = f"{programs}programs/warmup={'-'.join(order)}"

        assert executor.trace_compiler.trace_active, "capture did not complete"
        traces = executor.trace_compiler.trace_count
        expected_traces = 1 + (2 if executor.config.trace.prefill_enabled else 0)
        assert traces == expected_traces, f"captured {traces} traces, expected {expected_traces}"
        phases["capture"] = f"{traces}traces/{executor.trace_compiler.trace_association_count}aliases"

        def prefill_step(slot: int) -> int:
            """Serve one prefill through whichever path the policy selects."""

            logits = _prefill(executor, kv_cache, prompt, slot=slot)
            return int(torch.argmax(logits.float().reshape(-1)))

        def traced_decode(token: int, position: int) -> int:
            tokens = [token] + [0] * (GALAXY_PHYSICAL_BATCH - 1)
            positions = [position] + [0] * (GALAXY_PHYSICAL_BATCH - 1)
            logits = _decode_logits(_decode(executor, kv_cache, tokens, positions))
            assert torch.isfinite(logits[0]).all()
            return int(torch.argmax(logits[0]))

        if transition == "decode_to_prefill":
            traced_decode(0, 0)
            first = prefill_step(0)
            assert executor.active_mode == "prefill"
        elif transition == "prefill_to_decode":
            first = prefill_step(0)
            traced_decode(first, _PROMPT_LENGTH)
            assert executor.active_mode == "decode"
        elif transition == "repeated_prefill":
            first = prefill_step(0)
            second = prefill_step(1)
            assert first == second, f"two identical traced prefills disagreed: {first} vs {second}"
        elif transition == "repeated_decode":
            first = prefill_step(0)
            one = traced_decode(first, _PROMPT_LENGTH)
            two = traced_decode(first, _PROMPT_LENGTH)
            assert one == two, f"two identical traced decodes disagreed: {one} vs {two}"
        elif transition == "failure_during_transition":
            first = prefill_step(0)
            traced_decode(first, _PROMPT_LENGTH)
            # The failure: an ineligible prefill immediately after a traced
            # decode. It is refused at the model boundary before any device work,
            # so nothing is half-committed and the mesh is still usable.
            with expect_error(TraceCoverageError, "Required prefill trace is unavailable"):
                _prefill(executor, kv_cache, long_prompt, slot=2, execution=executor.traced_executor)
            recovered_prefill = prefill_step(1)
            assert recovered_prefill == first, "the mesh did not recover the same prefill answer"
            recovered_decode = traced_decode(first, _PROMPT_LENGTH)
            assert recovered_decode > 0
            assert executor.traced_executor.coverage_miss_count == 1
        elif transition == "cleanup_from_either_mode":
            prefill_step(0)
            assert executor.active_mode == "prefill"
            executor.cleanup()
            _assert_nothing_retained(executor, kv_cache, handle.model, "cleanup_from_prefill")
            first_replays = executor.trace_compiler.replay_counts
            del executor
            gc.collect()

            executor, kv_cache = _open(handle)
            _warm(executor, kv_cache)
            token = prefill_step(0)
            traced_decode(token, _PROMPT_LENGTH)
            assert executor.active_mode == "decode"
            phases["replay"] = f"prefill_then_cleanup={first_replays}/second={executor.trace_compiler.replay_counts}"
        else:  # pragma: no cover - the parametrization is closed
            raise AssertionError(transition)

        if phases["replay"] == "not reached":
            phases["replay"] = str(executor.trace_compiler.replay_counts)
        assert executor.trace_compiler.replay_count > 0, "the transition submitted no replay"

        executor.cleanup()
        _assert_nothing_retained(executor, kv_cache, handle.model, transition)
        phases["cleanup"] = "released/terminal/no-orphans"
        executor_after, executor = executor, None
        assert executor_after.terminal
    finally:
        _report_transition(transition, phases)
        if executor is not None:
            try:
                executor.cleanup()
            except BaseException as error:  # noqa: BLE001 - the failure under test is the result
                print(f"[transition] {transition} cleanup after failure raised {error!r}", flush=True)
        _close(handle)


# ---------------------------------------------------------------------------
# 9. Repeated startup, serving and cleanup with tracing active
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_repeated_startup_serving_and_cleanup_with_tracing(mesh_device: ttnn.MeshDevice) -> None:
    """Coverage 9: three traced startup/serve/cleanup cycles over one model.

    The traced path allocates more than the eager one — persistent inputs, replay
    workspaces and the trace buffers themselves — so this is a stronger test of
    D-C7's fix than the eager cycle was.
    """

    handle = _load(mesh_device)
    prompt = _prompt()
    seen = []
    try:
        for cycle in range(3):
            executor, kv_cache = _open(handle)
            try:
                _warm(executor, kv_cache)
                assert executor.trace_compiler.trace_active, f"cycle {cycle} did not capture"
                first = int(torch.argmax(_prefill(executor, kv_cache, prompt, slot=0).float().reshape(-1)))
                tokens = [first] + [0] * (GALAXY_PHYSICAL_BATCH - 1)
                positions = [_PROMPT_LENGTH] + [0] * (GALAXY_PHYSICAL_BATCH - 1)
                decode = _decode_logits(_decode(executor, kv_cache, tokens, positions))
                counts = executor.trace_compiler.replay_counts
                assert counts["prefill"] >= 1 and counts["decode"] >= 1, counts
                seen.append((first, int(torch.argmax(decode[0]))))
                print(f"[trace] cycle {cycle}: {seen[-1]} replays {counts}", flush=True)
            finally:
                executor.cleanup()
            _assert_nothing_retained(executor, kv_cache, handle.model, f"cycle {cycle}")
            del executor
            gc.collect()
        assert len(set(seen)) == 1, f"three traced cycles disagreed: {seen}"
    finally:
        _close(handle)


# ---------------------------------------------------------------------------
# The capture-phase free hold (F-1's fix) — that it engages on real silicon
# ---------------------------------------------------------------------------


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_TRACE_DEVICE_PARAMS], indirect=True)
def test_capture_body_frees_are_held_for_the_capture_phase(mesh_device: ttnn.MeshDevice) -> None:
    """The F-1 hold engages inside the real capture bodies, and lets go after.

    A DRAM buffer freed inside a capture body hands its address back to the
    allocator while the graph being recorded still writes into it, so a later
    allocation in the same body can be placed on a Galaxy collective's
    fabric-write destination. `src/tt_transformers/models/galaxy/capture_frees.py`
    carries the measurement; this test pins the two facts that make it a fix
    rather than a hypothesis: the hold **fires** on this model's own graphs, and
    it **releases** everything when the capture phase ends, so nothing about it
    is resident.
    """

    handle = _load(mesh_device)
    executor = None
    try:
        executor, kv_cache = _open(handle)
        _warm(executor, kv_cache)
        assert executor.trace_compiler.trace_active, "warmup did not capture"

        hold = executor._capture_frees
        assert hold is not None, "the executor never built a capture free hold"
        print(
            f"[hold] deferred={hold.deferred_count} frees, peak={hold.deferred_bytes / 1e6:.1f} MB/device, "
            f"flushed={hold.flushed_count}, left_with_their_collective={hold.protected_count}",
            flush=True,
        )
        assert hold.deferred_count > 0, "no capture-body DRAM free was held: the hold did not engage"
        assert hold.flushed_count > 0, "the hold engaged and never released"
        # Nothing is retained past the capture phase: `b6` releases the whole
        # hold before its forty replays and is bit-identical to eager on all of
        # them, so a resident hold is not what makes the trace correct.
        assert not hold._deferred, f"{len(hold._deferred)} tensors are still held after the capture phase"
        # The guard that `b5` needed and did not have. A borrowed plan buffer
        # freed inside the body would be a defect of its own, not a hold.
        assert hold.protected_count == 0, (
            f"{hold.protected_count} capture-body frees targeted a buffer the resource owner still "
            "lends to a collective"
        )
    finally:
        if executor is not None:
            executor.cleanup()
        _close(handle)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
