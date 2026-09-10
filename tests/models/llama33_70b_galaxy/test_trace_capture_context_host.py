# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused host test for `c-trace`'s second common-runtime change.

`TraceCompiler.capture_all` captures every registered trace inside one call, one
`ttnn.begin_trace_capture`/`ttnn.end_trace_capture` region per plan. On a model
whose operations share one device context that is complete. On Galaxy they do
not: `GalaxyExecutionResources.activate(mode)` loads a **different sub-device
manager** per mode, sets a different stall group, and starts the persistent
prefetcher for decode only
(`src/tt_transformers/modules/prefetcher/prefetcher_2d.py::_configure_mode_resources`).
A decode program dispatched under the prefill manager is

    TT_FATAL @ tt_metal/impl/program/program.cpp:2205
    Kernel group cores do not match sub device cores

which is D-C8's own signature, and that abort inside a multi-sub-device program
leaves the mesh un-drainable. `load_sub_device_manager` also cannot be called
inside a capture region.

So `capture_all` has to let each plan establish its operation's context first,
outside the region. The change is one optional field on `TraceCapturePlan`
(`prepare_context`), one call site in `capture_all`, and one optional constructor
argument on `TracedExecutor` (`capture_context`) that binds it to the operation
name the plan already carries. Unset - which is every caller that existed before
- `capture_all` behaves exactly as it did.

These tests live in the model's test directory so that
`pytest tests/llm_runtime` keeps its Milestone B expectation of
1032 passed / 1 skipped exactly. Run::

    pytest tests/models/llama33_70b_galaxy/test_trace_capture_context_host.py \
        -v -rA --color=no -p no:cacheprovider
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import ttnn

import tt_transformers.llm_runtime.trace_compiler as trace_compiler_module
from tt_transformers.llm_runtime.program_compiler import ProgramCompiler
from tt_transformers.llm_runtime.trace_compiler import TraceCapturePlan, TraceCompiler


@dataclass(frozen=True)
class _Signature:
    kind: str
    variant: int

    @property
    def key_material(self):
        return (("kind", self.kind), ("variant", self.variant))


def _patch_backend(monkeypatch, events):
    trace_ids = iter(range(100, 200))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append(("sync",)))
    monkeypatch.setattr(
        ttnn,
        "begin_trace_capture",
        lambda mesh, cq_id: events.append(("begin",)) or next(trace_ids),
    )
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda mesh, trace_id, cq_id: events.append(("end",)))
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, trace_id: events.append(("release",)))
    monkeypatch.setattr(trace_compiler_module, "_trim_host_allocator", lambda: None)


def _compiler(monkeypatch):
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    compiler = ProgramCompiler("mesh", lambda: object())
    programs = {
        operation: compiler.compile(_Signature("program", index), lambda context: torch.zeros(1))
        for index, operation in enumerate(("decode", "prefill"))
    }
    return compiler, programs


def _plan(program, operation, events, *, prepare_context=None):
    return TraceCapturePlan(
        program_key=program.key,
        trace_signature=_Signature("trace", 0 if operation == "decode" else 1),
        operation=operation,
        prepare_inputs=lambda: events.append(("prepare", operation)) or (),
        capture=lambda persistent: events.append(("capture", operation)) or torch.zeros(1),
        prepare_context=prepare_context,
    )


@pytest.mark.host
@pytest.mark.model
def test_each_operation_context_is_established_before_its_own_capture_region(monkeypatch):
    events = []
    compiler, programs = _compiler(monkeypatch)
    _patch_backend(monkeypatch, events)
    trace = TraceCompiler(compiler)
    for operation in ("decode", "prefill"):
        trace.register_capture_plan(
            _plan(
                programs[operation],
                operation,
                events,
                prepare_context=lambda operation=operation: events.append(("context", operation)),
            )
        )

    trace.capture_all()

    # Exactly one context establishment per capture, each of them outside the
    # region it precedes: ("context", op) then "begin" then ("capture", op) then
    # "end", never a context between a begin and its end.
    depth = 0
    contexts = []
    for event in events:
        if event[0] == "begin":
            depth += 1
        elif event[0] == "end":
            depth -= 1
        elif event[0] == "context":
            assert depth == 0, "an operation context was established inside a capture region"
            contexts.append(event[1])
    assert contexts == [
        "decode",
        "prefill",
        "decode",
        "prefill",
    ], f"contexts ran in the wrong order or count: {contexts}"

    # Staging an operation's persistent inputs is that operation's device work
    # too - on the Galaxy models the prefill rotary slice is a program, and
    # `capture_all` stages every plan's inputs before the first capture region.
    # So the context is established before the staging as well as before the
    # capture, and the capture order is decode-then-prefill regardless of the
    # registration order.
    ordered = [event for event in events if event[0] in ("context", "prepare", "begin", "capture", "end")]
    assert ordered == [
        ("context", "decode"),
        ("prepare", "decode"),
        ("context", "prefill"),
        ("prepare", "prefill"),
        ("context", "decode"),
        ("begin",),
        ("capture", "decode"),
        ("end",),
        ("context", "prefill"),
        ("begin",),
        ("capture", "prefill"),
        ("end",),
    ], ordered


@pytest.mark.host
@pytest.mark.model
def test_a_plan_without_a_context_hook_captures_exactly_as_before(monkeypatch):
    events = []
    compiler, programs = _compiler(monkeypatch)
    _patch_backend(monkeypatch, events)
    trace = TraceCompiler(compiler)
    for operation in ("decode", "prefill"):
        trace.register_capture_plan(_plan(programs[operation], operation, events))

    trace.capture_all()

    assert [event for event in events if event[0] in ("context", "begin", "capture", "end")] == [
        ("begin",),
        ("capture", "decode"),
        ("end",),
        ("begin",),
        ("capture", "prefill"),
        ("end",),
    ]
    assert trace.trace_active


@pytest.mark.host
@pytest.mark.model
def test_a_non_callable_context_hook_is_refused_at_construction(monkeypatch, expect_error):
    compiler, programs = _compiler(monkeypatch)
    with expect_error(TypeError, "prepare_context must be callable"):
        _plan(programs["decode"], "decode", [], prepare_context="prefill")


@pytest.mark.host
@pytest.mark.model
def test_a_failing_context_leaves_no_capture_region_open_and_no_trace_active(monkeypatch, expect_error):
    events = []
    compiler, programs = _compiler(monkeypatch)
    _patch_backend(monkeypatch, events)
    trace = TraceCompiler(compiler)

    def boom():
        events.append(("context", "decode"))
        raise RuntimeError("the decode sub-device manager could not be loaded")

    trace.register_capture_plan(_plan(programs["decode"], "decode", events, prepare_context=boom))

    with expect_error(RuntimeError, "sub-device manager"):
        trace.capture_all()
    assert ("begin",) not in events, "a capture region was opened after the context failed"
    assert not trace.trace_active
    assert trace.replay_count == 0


@pytest.mark.host
@pytest.mark.model
def test_the_traced_executor_binds_the_operation_name_it_already_carries():
    """`capture_context` is called with the plan's own operation, nothing else."""

    from tt_transformers.llm_runtime.execution import TracedExecutor

    seen = []
    bound = TracedExecutor._capture_context_for
    executor = object.__new__(TracedExecutor)
    executor.capture_context = lambda operation: seen.append(operation)
    for operation in ("prefill", "decode"):
        hook = bound(executor, operation)
        assert callable(hook)
        hook()
    assert seen == ["prefill", "decode"]

    executor.capture_context = None
    assert bound(executor, "prefill") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
