# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host checks for version-1 sampling state commands."""

from types import SimpleNamespace

import pytest
import torch
import ttnn

from tt_transformers.modules.sampling.params import prepare_sampling_params
from tt_transformers.modules.sampling.sampling_state_1d import SamplingState1D, SamplingState1DState
from tt_transformers.modules.sampling.seed_manager_1d import SeedManager1D
from tt_transformers.sampling.sampling_params import SamplingParams


class Buffer:
    def __init__(self, source):
        self.source = source.clone()
        self.value = source.clone()
        self.updates = []

    def get_device_buffer(self):
        return self

    def cpu(self, *, blocking):
        assert blocking
        return self.value

    def update(self, source):
        self.source = source.clone()
        self.value.copy_(source)
        self.updates.append(source.clone())


def controller_and_state():
    seeds = Buffer(torch.zeros(2, dtype=torch.int64))
    sampling_config = SimpleNamespace(max_batch_size=2, max_top_k=32, vocab_size=4, seeds=seeds)
    controller = object.__new__(SamplingState1D)
    controller.sampling = SimpleNamespace(config=sampling_config)
    controller.seed_manager = SeedManager1D(sampling_config)
    buffers = {
        name: Buffer(torch.arange(8, dtype=torch.int32).reshape(2, 4))
        for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered")
    }
    buffers.update(
        {
            name: Buffer(torch.tensor([[0.25], [0.75]]))
            for name in (
                "presence_penalties",
                "frequency_penalties",
                "repetition_penalties",
                "inverse_repetition_penalties",
            )
        }
    )
    controller.penalties = SimpleNamespace(config=SimpleNamespace(**buffers))
    state = SamplingState1DState(
        seed_state=controller.seed_manager.create_state(),
        penalty_params=None,
        penalty_accumulator=None,
        active_mask=(True, True),
    )
    controller.seed_manager.admit(state.seed_state, (17, 29), (0, 1))
    controller.seed_manager.refresh(state.seed_state, (0, 1), positions=(3, 9))
    return controller, state, buffers


@pytest.mark.host
def test_dormant_sampler_remap_moves_device_history_and_rng_without_replacing_buffers(monkeypatch):
    controller, state, buffers = controller_and_state()
    # Device counters are newer than the host source retained by LazyBuffer.
    buffers["output_counts"].value.add_(100)
    handles = {name: buffer.get_device_buffer() for name, buffer in buffers.items()}
    expected = {name: buffer.value.flip(0).clone() for name, buffer in buffers.items()}
    before = state.seed_state.snapshot()
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda value: [value])
    monkeypatch.setattr(ttnn, "to_torch", lambda value: value)
    controller.remap_slots(state, (1, 0))
    after = state.seed_state.snapshot()
    assert after.request_seeds == tuple(reversed(before.request_seeds))
    assert after.token_counters == tuple(reversed(before.token_counters))
    assert after.unseeded_rng_states == tuple(reversed(before.unseeded_rng_states))
    for name, buffer in buffers.items():
        assert buffer.get_device_buffer() is handles[name]
        assert torch.equal(buffer.value, expected[name])
        assert len(buffer.updates) == 1


@pytest.mark.host
def test_sampler_compaction_clears_vacated_history_after_snapshot(monkeypatch):
    controller, state, buffers = controller_and_state()
    previous = buffers["output_counts"].value.clone()
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda value: [value])
    monkeypatch.setattr(ttnn, "to_torch", lambda value: value)
    controller.remap_slots(state, (1, 1))
    assert state.active_mask == (True, False)
    assert state.seed_state.snapshot().active_slots == (0,)
    assert torch.equal(buffers["output_counts"].value[0], previous[1])
    assert torch.count_nonzero(buffers["output_counts"].value[1]) == 0


@pytest.mark.host
@pytest.mark.parametrize("prior_penalty", [0.0, 0.5])
def test_parameter_only_penalty_enable_rejects_missing_history_before_mutation(prior_penalty):
    controller, state, buffers = controller_and_state()
    previous = prepare_sampling_params(
        SamplingParams(temperature=0.8, top_k=8, top_p=0.9, presence_penalty=prior_penalty),
        2,
        max_device_top_k=32,
        allow_force_argmax=False,
    )
    state.static_identity = controller.static_identity(previous)
    buffers["repetition_penalties"].update(torch.ones(2, 1))
    requested = prepare_sampling_params(
        SamplingParams(temperature=0.8, top_k=8, top_p=0.9, presence_penalty=0.5, repetition_penalty=1.2),
        2,
        max_device_top_k=32,
        allow_force_argmax=False,
    )
    before = state.seed_state.snapshot()
    updates = {name: len(buffer.updates) for name, buffer in buffers.items()}
    with pytest.raises(RuntimeError, match="requires reset_sampling_state=True"):
        controller.validate_decode_history(state, requested, slot_remap=(1, 0))
    assert state.seed_state.snapshot() == before
    assert {name: len(buffer.updates) for name, buffer in buffers.items()} == updates
