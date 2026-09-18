# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused host test for the prefill hidden-state ownership fix.

`PrefillTraceRuntime.finish` hands `post_process_prefill_output` a hidden state
it names `trace_owned_hidden_output` (`llm_runtime/prefill/trace.py`): the
captured graph's own persistent output buffer, written afresh by **every**
replay. The eager path is the same shape — `sequence_runner` puts the body's
output in `owned` and releases it itself.

`project_prefill_logits` released that tensor the moment the final norm had
consumed it. That is right for a caller who owns it and fatal for one who does
not: the replay that freed it still completes, and the *next* replay writes into
an address the allocator has since handed to something else. It reached silicon
as a segmentation fault inside `rms_norm_pre_all_gather` on the second of
thirty-two prefills.

So ownership became an argument. The default is unchanged for every caller that
had one — including `GalaxyDirectRunner`, which is frozen and passes nothing —
and the executor's view, which is handed the runtime's tensor, passes
`release_hidden=False`.

Run::

    pytest tests/models/llama33_70b_galaxy/test_prefill_hidden_ownership_host.py \
        -v -rA --color=no -p no:cacheprovider
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import tt_transformers.models.llama33_70b_galaxy.executor as executor_module
from tt_transformers.models.llama33_70b_galaxy.model import Llama33_70BGalaxyTransformer2D

_VOCAB = 16
_TOKENS = 8


class _Tensor:
    """The only tensor behaviour these paths use: a shape and a live flag."""

    def __init__(self, shape: tuple[int, ...], name: str) -> None:
        self.shape = shape
        self.name = name
        self.allocated = True

    def is_allocated(self) -> bool:
        return self.allocated

    def deallocate(self, force: bool = True) -> None:
        self.allocated = False

    def __getitem__(self, key: Any) -> _Tensor:
        """One-token slicing, which is all the projection asks of a tensor."""

        return _Tensor((self.shape[0], self.shape[1], 1, self.shape[3]), f"{self.name}[slice]")


class _Norm:
    def prefill_forward(self, hidden: Any) -> Any:
        return _Tensor(hidden.shape, "normed")


class _LMHead:
    def prefill_forward(self, placed: Any) -> Any:
        return _Tensor((1, 1, 1, _VOCAB), "logits")


class _Model:
    """Enough of the 2D transformer for `project_prefill_logits` alone."""

    def __init__(self) -> None:
        self.norm = _Norm()
        self.lm_head = _LMHead()
        self.config = SimpleNamespace(lm_head_config=SimpleNamespace(prefill_input_memcfg="dram"))

    project_prefill_logits = Llama33_70BGalaxyTransformer2D.project_prefill_logits


@pytest.fixture
def _plain_ttnn(monkeypatch):
    """Slicing and relocation reduce to identity; nothing else is exercised."""

    monkeypatch.setattr(executor_module.ttnn, "to_memory_config", lambda tensor, memory_config: tensor, raising=False)
    import tt_transformers.models.llama33_70b_galaxy.model as model_module

    monkeypatch.setattr(model_module, "_relocate", lambda tensor, memcfg: tensor)
    # `deallocate_if_allocated` is left real: it is duck-typed over
    # `is_allocated`/`deallocate`, which is exactly what these fakes provide, and
    # it is the call under test.
    return model_module


def _hidden() -> _Tensor:
    return _Tensor((1, 1, _TOKENS, 64), "hidden")


@pytest.mark.host
@pytest.mark.model
def test_the_default_still_releases_the_hidden_state_it_was_given(_plain_ttnn):
    """Every caller that predates tracing owns its hidden state and frees it."""

    hidden = _hidden()
    _Model().project_prefill_logits(hidden, rows=1, sequence_length=_TOKENS, token_indices=(3,))
    assert not hidden.is_allocated(), "the unchanged default stopped releasing the caller's hidden state"


@pytest.mark.host
@pytest.mark.model
def test_a_borrowed_hidden_state_survives_the_projection(_plain_ttnn):
    """The runtime's tensor - and under tracing the replay buffer - is left alone."""

    hidden = _hidden()
    _Model().project_prefill_logits(hidden, rows=1, sequence_length=_TOKENS, token_indices=(3,), release_hidden=False)
    assert hidden.is_allocated(), "a borrowed hidden state was released by the projection"


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("trace_active", [False, True])
def test_the_executor_view_borrows_only_while_a_trace_is_active(monkeypatch, trace_active):
    """Borrowing is conditional, and both halves of the condition are measured.

    With a trace active the tensor is the replay buffer and freeing it kills the
    next replay. Without one, holding it instead of freeing it moves the peak,
    and on this model moving an allocation is the L1 address-clash class - which
    showed up as a failing executor prefill claim.
    """

    seen: dict[str, Any] = {}

    class _RecordingModel:
        mesh_device = object()
        vocab_size = _VOCAB

        def project_prefill_logits(self, hidden, **kwargs):
            seen.update(kwargs)
            seen["hidden"] = hidden
            return (_Tensor((1, 1, 1, _VOCAB), "logits"),)

    view = executor_module._GalaxyRuntimeModelView.__new__(executor_module._GalaxyRuntimeModelView)
    view._model = _RecordingModel()
    view._prefill_last_row = None
    view._trace_active_probe = lambda: trace_active
    monkeypatch.setattr(executor_module, "compose_galaxy_logits", lambda tensor, **_: torch.zeros((1, _VOCAB)))
    monkeypatch.setattr(executor_module, "deallocate_if_allocated", lambda tensor: None)
    monkeypatch.setattr(
        executor_module._GalaxyRuntimeModelView,
        "_present_logits_block",
        lambda self, row_logits, row: ("block", row),
    )

    hidden = _hidden()
    result = view.post_process_prefill_output(hidden, 3)

    assert result == ("block", 3)
    assert seen["hidden"] is hidden
    assert seen["release_hidden"] is not trace_active, (
        "the view must borrow while a trace is active and release when none is"
    )
    assert hidden.is_allocated()


@pytest.mark.host
@pytest.mark.model
def test_the_executor_call_site_derives_ownership_from_trace_state():
    """The executor, not its caller, decides whether the hidden view is released.

    The tt-metal original also asserted that `GalaxyDirectRunner`'s source
    contained no `release_hidden`, which was how it pinned the frozen runner to
    the argument's default. That runner is not part of this package, so that half
    of the assertion has nothing to pin and was dropped; the half that constrains
    the executor is what carried the actual contract and is kept.
    """

    source = inspect.getsource(executor_module)
    assert "release_hidden=not self._trace_is_active()" in source


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
