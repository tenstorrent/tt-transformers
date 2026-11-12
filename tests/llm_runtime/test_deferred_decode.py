# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deferred device sampling for vLLM structured outputs, with fake devices.

vLLM fills a decode step's grammar mask only after the step's forward was
submitted, so the forward runs the model body and returns a payload, and
``sample_deferred_decode`` later stages the mask and runs the sampling tail.
These tests drive both call sequences the plugin uses: steps that reload
inputs every time, and steady resident continuations that do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import ttnn

import tt_transformers.llm_runtime.trace_compiler as trace_compiler_module
from tt_transformers.llm_runtime.config import PageTableLayout
from tt_transformers.llm_runtime.decode import (
    DecodeBodySignature,
    DecodeRuntime,
    DecodeRuntimeConfig,
    DeferredDecode,
)
from tt_transformers.llm_runtime.device_grammar import DeviceGrammarExecutorMixin
from tt_transformers.llm_runtime.execution import EagerExecutor, TracedExecutor
from tt_transformers.llm_runtime.output_reader import OutputReader
from tt_transformers.llm_runtime.prefill.runtime import PrefillRuntime
from tt_transformers.llm_runtime.program_compiler import ProgramCompiler
from tt_transformers.llm_runtime.trace_compiler import TraceBodyPlan, TraceCapturePlan, TraceCompiler, TraceKey
from tt_transformers.llm_runtime.warmup import WarmupCase, grammar_warmup_cases
from tt_transformers.sampling.sampling_params import SamplingParams

# ---------------------------------------------------------------------------
# Trace compiler: body-only and tail-only replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Signature:
    kind: str
    variant: int

    @property
    def key_material(self):
        return (("kind", self.kind), ("variant", self.variant))


def _patch_backend(monkeypatch, events):
    next_trace_id = iter(range(100, 200))
    monkeypatch.setattr(
        trace_compiler_module,
        "trace_allocation_tracker",
        SimpleNamespace(acknowledge_corruptible=lambda value: None),
    )
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda mesh, cq_id: next(next_trace_id))
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda mesh, trace_id, cq_id: None)
    monkeypatch.setattr(
        ttnn,
        "execute_trace",
        lambda mesh, trace_id, cq_id, blocking: events.append(("execute", trace_id)),
    )
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, trace_id: None)
    monkeypatch.setattr(trace_compiler_module, "_trim_host_allocator", lambda: None)


def _tail_plan(program, variant, *, body_variant=1):
    return TraceCapturePlan(
        program_key=program.key,
        trace_signature=_Signature("tail", variant),
        operation="decode",
        prepare_inputs=lambda body: ("tail", variant),
        capture=lambda persistent, logits: ("tokens", variant),
        body=TraceBodyPlan(
            signature=_Signature("body", body_variant),
            prepare_inputs=lambda: ("body", body_variant),
            capture=lambda persistent: f"logits-{body_variant}",
        ),
    )


def _captured(monkeypatch, events, bodies=(1, 1)):
    """Two tails; with bodies=(1, 2) each has its own body. Trace ids: body 100, tails 101+."""

    _patch_backend(monkeypatch, events)
    compiler = ProgramCompiler("mesh", lambda: object())
    programs = [compiler.compile(_Signature("program", variant), lambda _: torch.zeros(1)) for variant in (1, 2)]
    trace = TraceCompiler(compiler)
    for variant, (program, body) in enumerate(zip(programs, bodies), start=1):
        trace.register_capture_plan(_tail_plan(program, variant, body_variant=body))
    trace.capture_all()
    events.clear()
    return trace, programs


def _body(trace, *, body=1, reload_inputs=True, refreshes=None):
    return trace.replay_body(
        _Signature("body", body),
        lambda artifact, decision: refreshes.append(("body", decision.full)) if refreshes is not None else None,
        device_feedback_enabled=True,
        feedback_compatible=True,
        reload_inputs=reload_inputs,
    )


def _tail(trace, program, *, reload_inputs=True, refreshes=None):
    return trace.replay_tail(
        program.key,
        lambda artifact: refreshes.append(("tail",)) if refreshes is not None else None,
        reload_inputs=reload_inputs,
    )


@pytest.mark.host
def test_a_deferred_step_replays_its_body_then_its_tail_as_one_step(monkeypatch):
    events, refreshes = [], []
    trace, programs = _captured(monkeypatch, events)

    assert _body(trace, refreshes=refreshes) == "logits-1"
    assert trace.body_pending
    assert events == [("execute", 100)]
    assert trace.replay_counts["decode"] == 0

    assert _tail(trace, programs[0], refreshes=refreshes) == ("tokens", 1)
    assert not trace.body_pending
    assert events == [("execute", 100), ("execute", 101)]
    assert refreshes == [("body", True), ("tail",)]
    assert trace.replay_counts["decode"] == 1
    assert trace.previous_replay_keys == {
        "body": TraceKey.from_signature(_Signature("body", 1)),
        "tail": TraceKey.from_signature(_Signature("tail", 1)),
    }


@pytest.mark.host
def test_a_pending_body_blocks_every_other_replay_until_its_tail_runs(monkeypatch, expect_error):
    """The tail reads the body's retained logits, so nothing may replay in between."""

    events = []
    trace, programs = _captured(monkeypatch, events)
    _body(trace)

    with expect_error(RuntimeError, "waiting for its sampling tail"):
        trace.replay(programs[0].key, lambda artifact, decision: None)
    with expect_error(RuntimeError, "waiting for its sampling tail"):
        _body(trace)
    assert events == [("execute", 100)]

    _tail(trace, programs[1])
    trace.replay(programs[0].key, lambda artifact, decision: None)
    assert events == [("execute", 100), ("execute", 102), ("execute", 100), ("execute", 101)]


@pytest.mark.host
def test_a_tail_without_a_pending_body_is_rejected_without_latching(monkeypatch, expect_error):
    events = []
    trace, programs = _captured(monkeypatch, events)

    with expect_error(RuntimeError, "No decode body is pending"):
        _tail(trace, programs[0])
    trace.replay(programs[0].key, lambda artifact, decision: None)
    assert events == [("execute", 100), ("execute", 101)]


@pytest.mark.host
def test_a_tail_of_another_body_is_terminal(monkeypatch, expect_error):
    events = []
    trace, programs = _captured(monkeypatch, events, bodies=(1, 2))
    _body(trace, body=1)

    with expect_error(RuntimeError, "does not follow the pending body"):
        _tail(trace, programs[1])
    with expect_error(RuntimeError, "previously failed"):
        trace.replay(programs[0].key, lambda artifact, decision: None)


@pytest.mark.host
def test_resident_deferred_steps_follow_the_switch_rule(monkeypatch, expect_error):
    """A resident (reload_inputs=False) step must continue the body and tail that fed
    its inputs back; a first step or a tail switch needs a reload."""

    events, refreshes = [], []
    trace, programs = _captured(monkeypatch, events)
    with expect_error(RuntimeError, "requires reload_inputs=True"):
        _body(trace, reload_inputs=False)

    _body(trace, refreshes=refreshes)
    _tail(trace, programs[0])
    _body(trace, reload_inputs=False, refreshes=refreshes)
    _tail(trace, programs[0], reload_inputs=False)
    assert refreshes == [("body", True), ("body", False)]

    _body(trace, reload_inputs=False)
    with expect_error(RuntimeError, "requires reload_inputs=True"):
        _tail(trace, programs[1], reload_inputs=False)
    with expect_error(RuntimeError, "previously failed"):
        _body(trace)


@pytest.mark.host
def test_a_failed_tail_refresh_latches_the_compiler(monkeypatch, expect_error):
    events = []
    trace, programs = _captured(monkeypatch, events)
    _body(trace)

    def fail(artifact):
        raise RuntimeError("seed refresh failed")

    with expect_error(RuntimeError, "seed refresh failed"):
        trace.replay_tail(programs[0].key, fail, reload_inputs=True)
    with expect_error(RuntimeError, "previously failed"):
        trace.replay(programs[0].key, lambda artifact, decision: None)
    assert events == [("execute", 100)]


# ---------------------------------------------------------------------------
# Decode runtime: deferred preparation, grammar identity and the one payload
# ---------------------------------------------------------------------------


class _FakeSampling:
    def __init__(self):
        self.config = SimpleNamespace(allow_force_argmax=True, max_batch_size=2, max_top_k=32, seeds=None)
        self.calls = []
        self.layout_error = None

    def enable_device_grammar(self):
        if self.layout_error is not None:
            raise ValueError(self.layout_error)
        self.calls.append(("enable",))

    def validate_grammar_bitmask(self, mask):
        if mask == "bad":
            raise ValueError("no allowed token")
        self.calls.append(("validate", mask))

    def update_grammar_bitmask(self, mask):
        self.calls.append(("update", mask))

    def decode_forward(self, logits, **kwargs):
        self.calls.append(("sample", kwargs.get("apply_grammar", False)))
        return logits, None


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_batch_size=2)
        self.sampling = _FakeSampling()
        self.rope_setup = SimpleNamespace()
        self.vocab_size = 8
        self.num_devices = 1

    def iter_executor_named_modules(self):
        return iter(())

    def increment_positions(self, positions, rotary_indices):
        return None


def _runtime(*, sampling=True):
    config = DecodeRuntimeConfig.resolve(
        model=_FakeModel(),
        output_reader=OutputReader(SimpleNamespace(shape=(1, 1))),
        lane_capacity=2,
        page_table_layout=PageTableLayout(block_size=32, raw_capacity_width=8, prefill_width=8, decode_width=8),
        device_sampling_enabled=sampling,
        force_greedy_top_k=False,
        salt_duplicate_seeds=False,
    )
    return DecodeRuntime(config)


def _request(**overrides):
    request = {
        "tokens": torch.zeros(2, dtype=torch.long),
        "start_pos": torch.tensor([3, 4]),
        "page_table": torch.zeros((2, 2), dtype=torch.int32),
        "reload_inputs": True,
    }
    request.update(overrides)
    return request


def _greedy(**fields):
    return SamplingParams(temperature=[0.0, 0.0], top_k=[1, 1], top_p=[1.0, 1.0], **fields)


@pytest.mark.host
def test_a_deferred_forward_prepares_the_body_without_sampling_params(expect_error):
    runtime = _runtime()

    forward = runtime.prepare(**_request(reload_sampling_params=True, reset_sampling_state=True), deferred_forward=True)

    assert forward.sampling_params is None and forward.sampling_path == "logits"
    assert (forward.reload_sampling_params, forward.reset_sampling_state) == (True, True)
    assert runtime.body_signature(forward) == DecodeBodySignature(batch_size=2, page_table_width=8)
    with expect_error(ValueError, "takes its sampling parameters at sample time"):
        runtime.prepare(**_request(), sampling_params=_greedy(), deferred_forward=True)
    with expect_error(ValueError, "previous submitted device-feedback decode"):
        runtime.prepare(**_request(reload_inputs=False), deferred_forward=True)
    with expect_error(ValueError, "requires device sampling"):
        _runtime(sampling=False).prepare(**_request(), deferred_forward=True)


@pytest.mark.host
def test_grammar_is_a_tail_variant_that_shares_the_body(expect_error):
    runtime = _runtime()
    with expect_error(RuntimeError, "enable_grammar"):
        runtime.prepare(**_request(reload_sampling_params=True), sampling_params=_greedy(), grammar=True)

    runtime.enable_grammar()
    plain = runtime.prepare(**_request(reload_sampling_params=True), sampling_params=_greedy())
    grammar = runtime.prepare(**_request(reload_sampling_params=True), sampling_params=_greedy(), grammar=True)

    assert runtime.config.model.sampling.calls == [("enable",)]
    assert runtime.program_signature(grammar).grammar_enabled
    assert ("grammar_enabled", True) in runtime.trace_signature(grammar).key_material()
    assert ("grammar_enabled", True) not in runtime.trace_signature(plain).key_material()
    assert runtime.program_signature(plain) != runtime.program_signature(grammar)
    assert runtime.body_signature(plain) == runtime.body_signature(grammar)


@pytest.mark.host
def test_grammar_rejects_logprobs_before_any_state_change(expect_error):
    runtime = _runtime()
    runtime.enable_grammar()

    with expect_error(ValueError, "does not support logprobs"):
        runtime.prepare(
            **_request(reload_sampling_params=True),
            sampling_params=SamplingParams(
                temperature=[1.0, 1.0], top_k=[4, 4], top_p=[0.9, 0.9], enable_log_probs=[True, False]
            ),
            grammar=True,
        )


@pytest.mark.host
def test_a_layout_the_sampler_cannot_mask_leaves_grammar_off(expect_error):
    runtime = _runtime()
    runtime.config.model.sampling.layout_error = "vocab shard splits a mask word"

    with expect_error(ValueError, "splits a mask word"):
        runtime.enable_grammar()
    assert not runtime.grammar_enabled


@pytest.mark.host
def test_the_grammar_tail_samples_through_the_mask():
    runtime = _runtime()
    runtime.enable_grammar()
    prepared = runtime.prepare(**_request(reload_sampling_params=True), sampling_params=_greedy(), grammar=True)

    runtime._run_tail("logits", SimpleNamespace(), prepared, None, device_feedback=False)

    assert runtime.config.model.sampling.calls[-1] == ("sample", True)


@pytest.mark.host
def test_one_payload_is_pending_and_is_consumed_once(expect_error):
    runtime = _runtime()
    forward = runtime.prepare(**_request(), deferred_forward=True)
    payload = DeferredDecode(owner="executor", request=_request(), forward=forward, logits="logits")
    stale = DeferredDecode(owner="executor", request=_request(), forward=forward, logits="old")

    runtime.begin_deferred(payload)
    with expect_error(RuntimeError, "already pending|is pending"):
        runtime.begin_deferred(stale)
    with expect_error(RuntimeError, "is pending"):
        runtime.require_no_pending_deferred()
    with expect_error(RuntimeError, "stale or was already sampled"):
        runtime.take_deferred(stale)
    assert runtime.deferred_pending

    assert runtime.take_deferred(payload) is payload
    assert payload.consumed and not runtime.deferred_pending
    with expect_error(RuntimeError, "stale or was already sampled"):
        runtime.take_deferred(payload)


# ---------------------------------------------------------------------------
# Traced executor: order of a deferred step, and failures after submission
# ---------------------------------------------------------------------------


def _bare(runtime_type, **methods):
    runtime = object.__new__(runtime_type)
    if runtime_type is DecodeRuntime:
        runtime._transient_orphans = []
        runtime._submission_failed = False
        runtime._external_by_raw_id = {}
        runtime._external_by_host_id = {}
    for name, method in methods.items():
        setattr(runtime, name, method)
    return runtime


def _traced_executor(monkeypatch, events, *, stage_error=None):
    forward = SimpleNamespace(name="forward", reload_inputs=False, reload_page_table=False)
    full = SimpleNamespace(name="full", reload_inputs=False, sampling_params="params")

    def prepare(**kwargs):
        events.append(("prepare", kwargs.get("deferred_forward", False), kwargs.get("grammar", False)))
        return forward if kwargs.get("deferred_forward") else full

    def stage(mask):
        events.append(("stage", mask))
        if stage_error is not None:
            raise stage_error

    decode = _bare(
        DecodeRuntime,
        config=SimpleNamespace(position_feedback_capable=True),
        prepare=prepare,
        ensure_trace_safe=lambda: events.append(("ensure_trace_safe",)),
        body_signature=lambda prepared: _Signature("body", 1),
        program_signature=lambda prepared: _Signature("program", 1),
        refresh_trace_inputs=lambda artifact, prepared, decision: events.append(("refresh_inputs", prepared.name)),
        refresh_trace_sampling=lambda artifact, prepared: events.append(("refresh_sampling", prepared.name)),
        validate_grammar_mask=lambda mask: events.append(("validate_mask", mask)),
        validate_submission=lambda prepared: events.append(("validate_submission", prepared.name)),
        stage_grammar_mask=stage,
        note_submitted=lambda prepared: events.append(("submitted", prepared.name)),
        consume=lambda result, *, read_from_device=True: events.append(("consume", read_from_device)) or result.value,
        invalidate_after_failed_submission=lambda: events.append(("latched",)),
    )
    decode._pending_deferred = None
    program_compiler = ProgramCompiler("mesh", lambda: object())
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    eager = EagerExecutor(prefill=_bare(PrefillRuntime), decode=decode, program_compiler=program_compiler)
    trace = TraceCompiler(program_compiler)
    trace.trace_key_for_program = lambda key: "trace-key"
    trace.get = lambda key: SimpleNamespace(artifact=object())

    def replay_body(signature, refresh, **commands):
        events.append(("replay_body", commands["reload_inputs"]))
        refresh("body-artifact", SimpleNamespace(full=commands["reload_inputs"], page_table=False))
        return "logits"

    def replay_tail(program_key, refresh, *, reload_inputs):
        events.append(("replay_tail", reload_inputs))
        refresh("tail-artifact")
        return "tokens"

    trace.replay_body = replay_body
    trace.replay_tail = replay_tail
    return TracedExecutor(eager=eager, trace_compiler=trace), decode


@pytest.mark.host
def test_a_traced_deferred_step_validates_then_stages_the_mask_then_replays_the_tail(monkeypatch):
    events = []
    traced, decode = _traced_executor(monkeypatch, events)

    payload = traced.defer_decode(**_request(reload_inputs=False))
    assert payload.logits == "logits" and decode.deferred_pending
    assert events == [
        ("prepare", True, False),
        ("ensure_trace_safe",),
        ("replay_body", False),
        ("refresh_inputs", "forward"),
    ]

    events.clear()
    assert traced.sample_deferred_decode(payload, sampling_params="params", grammar_bitmask="mask") == "tokens"
    assert events == [
        ("prepare", False, True),
        ("validate_mask", "mask"),
        ("validate_submission", "full"),
        ("stage", "mask"),
        ("replay_tail", False),
        ("refresh_sampling", "full"),
        ("submitted", "full"),
        ("consume", False),
    ]
    assert not decode.deferred_pending


@pytest.mark.host
def test_a_failure_after_the_body_ran_latches_the_runtime(monkeypatch, expect_error):
    events = []
    traced, decode = _traced_executor(monkeypatch, events, stage_error=RuntimeError("copy failed"))
    payload = traced.defer_decode(**_request())

    with expect_error(RuntimeError, "copy failed"):
        traced.sample_deferred_decode(payload, sampling_params="params", grammar_bitmask="mask")
    assert ("latched",) in events
    assert not any(event[0] == "replay_tail" for event in events)


@pytest.mark.host
def test_a_second_deferred_forward_before_sampling_raises(monkeypatch, expect_error):
    events = []
    traced, _ = _traced_executor(monkeypatch, events)
    traced.defer_decode(**_request())

    with expect_error(RuntimeError, "is pending"):
        traced.defer_decode(**_request())


# ---------------------------------------------------------------------------
# Warmup twins and the executor-facing surface
# ---------------------------------------------------------------------------


@pytest.mark.host
def test_grammar_twins_cover_every_sampled_case_except_logprobs():
    decode = (
        WarmupCase("decode", 32, None, "logits"),
        WarmupCase("decode", 32, None, "argmax"),
        WarmupCase("decode", 32, None, "topk"),
        WarmupCase("decode", 32, None, "argmax", penalties=True),
        WarmupCase("decode", 32, None, "topk", penalties=True),
        WarmupCase("decode", 32, None, "topk", logprobs=True),
    )

    twins = grammar_warmup_cases(decode)

    assert [(case.sampling_path, case.penalties, case.grammar) for case in twins] == [
        ("argmax", False, True),
        ("topk", False, True),
        ("argmax", True, True),
        ("topk", True, True),
    ]
    assert grammar_warmup_cases(twins) == ()


class _GrammarExecutor(DeviceGrammarExecutorMixin):
    def __init__(self, *, device_sampling=True, trace_active=False, layout_error=None):
        self.events = []
        runtime = SimpleNamespace(grammar_enabled=False)

        def enable():
            if layout_error is not None:
                raise ValueError(layout_error)
            runtime.grammar_enabled = True

        runtime.enable_grammar = enable
        runtime.take_deferred = lambda payload: (_ for _ in ()).throw(RuntimeError("stale"))
        self.decode_runtime = runtime
        self.trace_compiler = SimpleNamespace(trace_active=trace_active)
        self.config = SimpleNamespace(device_sampling_enabled=device_sampling)

    def _ensure_active(self):
        self.events.append("active")

    def _ensure_sampling_buffers(self):
        self.events.append("buffers")

    def _ensure_sampling_for(self, sampling_params):
        self.events.append("sampling")


@pytest.mark.host
def test_enable_device_grammar_turns_on_only_where_a_mask_can_apply(expect_error):
    supported = _GrammarExecutor()
    supported.enable_device_grammar()
    assert supported.device_grammar_enabled
    assert supported._grammar_warmup_kwargs() == {"can_sample_device_grammar": True}

    for executor in (_GrammarExecutor(device_sampling=False), _GrammarExecutor(layout_error="splits a word")):
        executor.enable_device_grammar()
        assert not executor.device_grammar_enabled
        assert executor._grammar_warmup_kwargs() == {}

    with expect_error(RuntimeError, "before any trace is captured"):
        _GrammarExecutor(trace_active=True).enable_device_grammar()


@pytest.mark.host
def test_deferred_decode_requires_enabled_grammar_and_no_sampling_now(expect_error):
    executor = _GrammarExecutor()
    execution = SimpleNamespace(defer_decode=lambda **request: ("payload", request))
    with expect_error(RuntimeError, "requires enable_device_grammar"):
        executor._defer_decode(execution, sampling_params=None, read_from_device=False, tokens="t")

    executor.enable_device_grammar()
    with expect_error(ValueError, "takes no sampling_params"):
        executor._defer_decode(execution, sampling_params="params", read_from_device=False, tokens="t")
    with expect_error(ValueError, "read_from_device=False"):
        executor._defer_decode(execution, sampling_params=None, read_from_device=True, tokens="t")
    assert executor._defer_decode(execution, sampling_params=None, read_from_device=False, tokens="t") == (
        "payload",
        {"tokens": "t"},
    )
    with expect_error(RuntimeError, "stale"):
        executor.sample_deferred_decode("not-a-payload", sampling_params="params")
