# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host tests for Sampling1D's greedy tie-break.

ttnn.sampling resolves an exact value tie by candidate position, and the unstable
top-k plus the all-gather don't fix that position, so a greedy user could flip
between two tied tokens. These tests run ``Sampling1D._sample_topk`` against a
torch stand-in for the ttnn ops it issues. The stand-in's ``sampling`` has the
same property as the device op: a k == 1 row takes the first maximal position.
"""

from types import SimpleNamespace

import pytest
import torch

from tt_transformers.modules.sampling import sampling_1d
from tt_transformers.modules.sampling.sampling_1d import Sampling1D


def _dtype(value):
    return value if isinstance(value, torch.dtype) else None


class _TorchTtnn:
    """The subset of ttnn that ``_sample_topk`` and its tie-break issue, on torch tensors."""

    bfloat16 = torch.bfloat16
    int32 = torch.int32
    uint32 = torch.int32
    TILE_LAYOUT = "TILE"
    DRAM_MEMORY_CONFIG = "DRAM"

    def __init__(self):
        self.sampled_values = None

    @staticmethod
    def _binary(a, b, op, dtype=None):
        out = op(a, b)
        target = dtype or (a.dtype if isinstance(a, torch.Tensor) else b.dtype)
        return out.to(target)

    def typecast(self, t, dtype, **_):
        return t.to(dtype)

    def reshape(self, t, shape, **_):
        return t.reshape(shape)

    def to_layout(self, t, layout, **_):
        return t

    def transpose(self, t, dim0, dim1, **_):
        return t.transpose(dim0, dim1).contiguous()

    def slice(self, t, start, end, **_):
        return t[tuple(slice(s, e) for s, e in zip(start, end))]

    def to_memory_config(self, t, memory_config=None, dtype=None, **_):
        return t if dtype is None else t.to(dtype)

    def untilize(self, t, **_):
        return t

    def deallocate(self, t):
        pass

    def manual_seed(self, **_):
        pass

    def max(self, t, dim, keepdim=False, **_):
        return torch.amax(t, dim=dim, keepdim=keepdim)

    def min(self, t, dim, keepdim=False, **_):
        return torch.amin(t, dim=dim, keepdim=keepdim)

    def abs(self, t, **_):
        return t.abs()

    def eq(self, a, b, **_):
        return self._binary(a, b, torch.eq)

    def lt(self, a, b, **_):
        return self._binary(a, b, torch.lt)

    def add(self, a, b, dtype=None, **_):
        return self._binary(a, b, torch.add, _dtype(dtype))

    def multiply(self, a, b, **_):
        return self._binary(a, b, torch.mul)

    def sampling(self, values, indices, k, p, temp, **_):
        self.sampled_values = values.clone()
        rows = values.shape[2]
        picks = []
        for row in range(rows):
            if int(k[row]) != 1:
                raise AssertionError("the stand-in only draws greedy rows")
            picks.append(indices[0, 0, row, int(torch.argmax(values[0, 0, row].float()))])
        return torch.stack(picks).to(torch.int32)


def _sampler(fake, local_values, local_indices):
    rows, width = local_values.shape[2], local_values.shape[3]
    sampler = object.__new__(Sampling1D)
    sampler.config = SimpleNamespace(sub_core_grids=None, sampling_memory_config="DRAM")
    sampler._topk = lambda x: (local_values.clone(), local_indices.clone())
    sampler._prepare_topk_memory = Sampling1D._topk_memory_noop
    sampler._index_offsets = torch.zeros(1, 1, rows, width, dtype=torch.int32)
    sampler._invalid_vocab_mask = None
    sampler._invalid_vocab_tail_mask = None
    sampler._seeds = None
    sampler._user_ids = None
    sampler._sampling_sub_core_grids = None
    sampler._log_probs_calculator = SimpleNamespace(enable_log_probs=False)
    return sampler


def _run(monkeypatch, values, indices, k):
    fake = _TorchTtnn()
    monkeypatch.setattr(sampling_1d, "ttnn", fake)
    sampler = _sampler(fake, values, indices)
    logits = torch.zeros(1, 1, values.shape[2], 64, dtype=torch.bfloat16)
    tokens, _ = sampler._sample_topk(
        logits,
        torch.tensor(k, dtype=torch.int32),
        torch.zeros(len(k)),
        torch.ones(len(k)),
        None,
        None,
    )
    return tokens, fake.sampled_values


def _candidates(rows):
    values = torch.tensor([[v for v, _ in row] for row in rows], dtype=torch.bfloat16).reshape(1, 1, len(rows), -1)
    indices = torch.tensor([[i for _, i in row] for row in rows], dtype=torch.int32).reshape(1, 1, len(rows), -1)
    return values, indices


@pytest.mark.host
def test_greedy_tie_picks_lowest_global_index_whatever_the_candidate_order(monkeypatch):
    # The eval-32 signature: tokens 13 and 2041 tied at the top. Row 0 gets the higher id first,
    # row 1 the lower id first; both must pick 13, as torch.argmax over the full vocab would.
    values, indices = _candidates(
        [
            [(30.0, 2041), (30.0, 13), (29.5, 11), (27.75, 304)],
            [(30.0, 13), (30.0, 2041), (29.5, 11), (27.75, 304)],
        ]
    )
    tokens, _ = _run(monkeypatch, values, indices, k=[1, 1])
    assert tokens.tolist() == [13, 13]


@pytest.mark.host
@pytest.mark.parametrize("tied_value", [0.0, -3.5, 300.0], ids=["zero", "negative", "above-256"])
def test_greedy_tie_break_survives_bfloat16_spacing(monkeypatch, tied_value):
    # At |x| >= 256 the bf16 spacing is 2.0, so a fixed +1 boost would round back into the tie.
    values, indices = _candidates([[(tied_value, 150001), (tied_value, 100003), (tied_value - 64.0, 7)]])
    tokens, _ = _run(monkeypatch, values, indices, k=[1])
    assert tokens.tolist() == [100003]


@pytest.mark.host
def test_untied_greedy_rows_keep_their_token(monkeypatch):
    values, indices = _candidates([[(29.5, 5), (31.0, 9000), (30.75, 3)]])
    tokens, _ = _run(monkeypatch, values, indices, k=[1])
    assert tokens.tolist() == [9000]


@pytest.mark.host
def test_non_greedy_rows_are_bit_identical(monkeypatch):
    values, indices = _candidates(
        [
            [(30.0, 2041), (30.0, 13), (29.5, 11)],
            [(30.0, 2041), (30.0, 13), (29.5, 11)],
        ]
    )
    fake = _TorchTtnn()
    fake.sampling = lambda v, i, k, p, temp, **_: (setattr(fake, "sampled_values", v.clone()), i[0, 0, :, 0])[1]
    monkeypatch.setattr(sampling_1d, "ttnn", fake)
    sampler = _sampler(fake, values, indices)
    sampler._sample_topk(
        torch.zeros(1, 1, 2, 64, dtype=torch.bfloat16),
        torch.tensor([1, 4], dtype=torch.int32),
        torch.zeros(2),
        torch.ones(2),
        None,
        None,
    )
    assert torch.equal(fake.sampled_values[0, 0, 1], values[0, 0, 1])
    assert not torch.equal(fake.sampled_values[0, 0, 0], values[0, 0, 0])
