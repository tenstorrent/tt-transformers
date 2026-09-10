# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host coverage for `release_device_weights`, the weight-release primitive.

A `LazyWeight` memoizes its device tensor in `_value`, so dropping the Python
reference to a module is **not** what returns device memory - the weight stays
reachable for as long as anything holds the model. Measured on `(8, 4)` with
Galaxy Llama-3.3-70B at the full 80-layer shape, after `close()`, `del` and
`gc.collect()` (`tttv2_milestone_c_evidence/defects/logs/x1_llama_pool4096_only_full.log`):
**398 617 984 B per DRAM bank still allocated, 240 of 240 prefetcher-registered
weights still live**, which is 37 % of the 1 070 773 184 B bank and is what
stopped a second Galaxy model loading in one process.

The two aliasing cases below are not hypothetical. `Attention2D` resolves
`prefill_wqkv = resolved.prefill_wqkv or wqkv`, so its prefill and decode config
fields are the *same object* whenever no separate prefill weight was configured;
and two distinct `LazyWeight`s can memoize one device tensor. Without both
dedupes the release double-frees.

Host only: `ttnn.deallocate` is captured through the module under test, so no
device is opened.
"""

from __future__ import annotations

import pytest

from tt_transformers.modules import lazy_weight as lazy_weight_module
from tt_transformers.modules.lazy_weight import release_device_weights


class _FakeTensor:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_FakeTensor({self.name!r})"


class _FakeWeight:
    """The only part of `LazyWeight` this primitive touches: the `_value` memo."""

    def __init__(self, value: _FakeTensor | None) -> None:
        self._value = value


@pytest.fixture
def deallocations(monkeypatch):
    seen: list[_FakeTensor] = []
    monkeypatch.setattr(lazy_weight_module.ttnn, "deallocate", seen.append)
    return seen


@pytest.mark.host
def test_release_deallocates_each_weight_once_and_clears_the_memo(deallocations):
    first, second = _FakeTensor("wqkv"), _FakeTensor("wo")
    weights = (_FakeWeight(first), _FakeWeight(second))

    release_device_weights(weights)

    assert deallocations == [first, second]
    assert [weight._value for weight in weights] == [None, None]


@pytest.mark.host
def test_release_is_idempotent(deallocations):
    weight = _FakeWeight(_FakeTensor("wqkv"))

    release_device_weights((weight,))
    release_device_weights((weight,))

    assert len(deallocations) == 1
    assert weight._value is None


@pytest.mark.host
def test_one_weight_reached_through_two_fields_is_freed_once(deallocations):
    """`prefill_wqkv is wqkv` when no separate prefill weight was configured."""

    weight = _FakeWeight(_FakeTensor("wqkv"))

    release_device_weights((weight, weight))

    assert len(deallocations) == 1


@pytest.mark.host
def test_two_weights_sharing_one_device_tensor_are_freed_once(deallocations):
    tensor = _FakeTensor("shared")
    first, second = _FakeWeight(tensor), _FakeWeight(tensor)

    release_device_weights((first, second))

    assert deallocations == [tensor]
    assert first._value is None and second._value is None


@pytest.mark.host
def test_unmaterialized_and_absent_weights_are_skipped(deallocations):
    release_device_weights((None, _FakeWeight(None)))

    assert deallocations == []


@pytest.mark.host
def test_a_failing_deallocate_does_not_strand_the_remaining_weights(monkeypatch, expect_error):
    """One bad handle must not leave the rest of a model's weights resident."""

    seen: list[_FakeTensor] = []

    def deallocate(tensor: _FakeTensor) -> None:
        if tensor.name == "bad":
            raise RuntimeError("buffer is not allocated")
        seen.append(tensor)

    monkeypatch.setattr(lazy_weight_module.ttnn, "deallocate", deallocate)
    bad, good = _FakeWeight(_FakeTensor("bad")), _FakeWeight(_FakeTensor("good"))

    with expect_error(RuntimeError, "buffer is not allocated"):
        release_device_weights((bad, good))

    assert [tensor.name for tensor in seen] == ["good"]
    assert bad._value is None and good._value is None
