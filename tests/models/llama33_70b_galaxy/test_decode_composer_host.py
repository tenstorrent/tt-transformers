# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused host test for the one common-runtime change Galaxy tracing needed.

`DecodeRuntime` composed multi-device decode output itself, with a
column-concatenating composition that is correct for a 1D model and wrong on
Galaxy — and, more importantly here, it composed it *behind* the model
body rather than inside it only because the Galaxy model view read the logits to
host **inside** the body. That body is the exact region
`TraceCompiler.capture_all` records between `ttnn.begin_trace_capture` and
`ttnn.end_trace_capture`, so a traced decode cannot contain the read.

The change is one probe on the injected model collaborator, in the same shape the
runtime already uses for `prepare_prefill_rot_mats`
(`llm_runtime/prefill/inputs.py:270`): if the model provides
`compose_decode_logits` / `compose_decode_tokens`, the runtime's own read calls
it; if it does not, the previous composition is unchanged.

These tests are here rather than under `tests/llm_runtime/` so that
`pytest tests/llm_runtime` keeps its expectation of
1032 passed / 1 skipped exactly. Run::

    pytest tests/models/llama33_70b_galaxy/test_decode_composer_host.py \
        -v -rA --color=no -p no:cacheprovider
"""

from __future__ import annotations

import tokenize
from types import SimpleNamespace

import pytest
import torch

import tt_transformers.llm_runtime.decode as decode_module
from tt_transformers.llm_runtime.config import PageTableLayout
from tt_transformers.llm_runtime.decode import DecodeRuntime, DecodeRuntimeConfig
from tt_transformers.llm_runtime.output_reader import OutputReader

_LANE_CAPACITY = 2
_VOCAB = 8


class _FakeMesh:
    shape = (2, 2)


class _FakeRope:
    def get_rot_idxs(self, positions, *, on_host):
        return ("rotary", positions.clone())

    def get_rot_mats(self, rotary_indices):
        return ("cos", "sin")


class _FakeModel:
    """The runtime's model contract, with the composers left off by default."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(max_batch_size=_LANE_CAPACITY)
        self.sampling = None
        self.rope_setup = _FakeRope()
        self.vocab_size = _VOCAB
        self.num_devices = 4

    def iter_executor_named_modules(self):
        return iter(())


def _layout() -> PageTableLayout:
    return PageTableLayout(block_size=32, raw_capacity_width=8, prefill_width=8, decode_width=8)


def _runtime(model: _FakeModel) -> DecodeRuntime:
    return DecodeRuntime(
        DecodeRuntimeConfig.resolve(
            model=model,
            output_reader=OutputReader(_FakeMesh()),
            lane_capacity=_LANE_CAPACITY,
            page_table_layout=_layout(),
            device_sampling_enabled=False,
        )
    )


def _refuse(*_args, **_kwargs):
    raise AssertionError("the runtime's own composition ran while the model provided one")


@pytest.mark.host
@pytest.mark.model
def test_a_model_without_a_logits_composer_keeps_the_runtime_composition(monkeypatch):
    calls = []

    def record(value, cluster_shape):
        calls.append((value, cluster_shape))
        return torch.zeros((1, 1, _LANE_CAPACITY, _VOCAB))

    monkeypatch.setattr(decode_module, "_concat_host_output", record)
    runtime = _runtime(_FakeModel())
    output = runtime._convert_logits(object())
    assert len(calls) == 1, "the runtime must still compose for a model that provides no composer"
    assert calls[0][1] == (2, 2)
    assert tuple(output.shape) == (_LANE_CAPACITY, 1, _VOCAB)


@pytest.mark.host
@pytest.mark.model
def test_a_model_logits_composer_replaces_the_runtime_composition(monkeypatch):
    expected = torch.arange(_LANE_CAPACITY * _VOCAB, dtype=torch.float32).reshape(1, 1, _LANE_CAPACITY, _VOCAB)
    seen = []
    model = _FakeModel()
    model.compose_decode_logits = lambda value: (seen.append(value) or expected)

    monkeypatch.setattr(decode_module, "_concat_host_output", _refuse)
    runtime = _runtime(model)
    sentinel = object()
    output = runtime._convert_logits(sentinel)
    assert seen == [sentinel], "the model's composer must receive the runtime's own host read"
    assert torch.equal(output, expected.reshape(_LANE_CAPACITY, 1, _VOCAB))


@pytest.mark.host
@pytest.mark.model
def test_a_torch_output_still_bypasses_every_composer(monkeypatch):
    model = _FakeModel()
    model.compose_decode_logits = _refuse
    monkeypatch.setattr(decode_module, "_concat_host_output", _refuse)
    runtime = _runtime(model)
    value = torch.arange(_LANE_CAPACITY * _VOCAB, dtype=torch.float32).reshape(1, 1, _LANE_CAPACITY, _VOCAB)
    output = runtime._convert_logits(value)
    assert torch.equal(output, value.reshape(_LANE_CAPACITY, 1, _VOCAB))


@pytest.mark.host
@pytest.mark.model
def test_a_model_without_a_token_composer_keeps_the_runtime_composition(monkeypatch):
    calls = []

    def record(value, batch_size, cluster_shape):
        calls.append((value, batch_size, cluster_shape))
        return torch.zeros(batch_size)

    monkeypatch.setattr(decode_module, "_process_output_tokens", record)
    runtime = _runtime(_FakeModel())
    tokens, log_probs = runtime._normalize_host_output(object(), is_tokens=True)
    assert len(calls) == 1 and calls[0][1] == _LANE_CAPACITY and calls[0][2] == (2, 2)
    assert log_probs is None
    assert tokens.dtype is torch.int64


@pytest.mark.host
@pytest.mark.model
def test_a_model_token_composer_replaces_the_runtime_composition(monkeypatch):
    expected = torch.tensor([11, 22])
    seen = []
    model = _FakeModel()
    model.compose_decode_tokens = lambda value: (seen.append(value) or expected)

    def refuse_tokens(*_args, **_kwargs):
        raise AssertionError("the runtime's own token composition ran while the model provided one")

    monkeypatch.setattr(decode_module, "_process_output_tokens", refuse_tokens)
    runtime = _runtime(model)
    sentinel = object()
    tokens, log_probs = runtime._normalize_host_output(sentinel, is_tokens=True)
    assert seen == [sentinel]
    assert log_probs is None
    assert torch.equal(tokens, expected.to(torch.int64))


@pytest.mark.host
@pytest.mark.model
def test_the_probe_is_a_capability_check_and_not_a_model_branch():
    """No model, family, architecture or mesh identity may decide the branch.

    Executable code only. `decode.py`'s class docstring names `Llama3Executor` as
    the 1D caller that owns this runtime, and always has; a docstring
    is not a branch, and rewriting inherited prose to satisfy an assertion would
    be adjusting the evidence rather than the code.
    """

    with open(decode_module.__file__, "rb") as handle:
        tokens = list(tokenize.tokenize(handle.readline))
    code = " ".join(token.string for token in tokens if token.type not in (tokenize.COMMENT, tokenize.STRING)).lower()
    # `mesh_shape=config.cluster_shape` is a resolved-config argument name that
    # predates this milestone, not a topology branch, so the check names the
    # branch forms the house rules forbid.
    for forbidden in ("galaxy", "llama", "qwen", "is_2d", "is_galaxy", "mesh_shape =="):
        assert forbidden not in code, f"{forbidden!r} entered a common runtime code path"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
