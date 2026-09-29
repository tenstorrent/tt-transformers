# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("vllm", reason="vLLM is supplied by the schema-6 serving integration")

from tt_transformers.models.qwen38.qwen36_vllm_dflash import Qwen36DFlashForCausalLM


def _adapter_with_pending_moves(moves):
    adapter = object.__new__(Qwen36DFlashForCausalLM)
    adapter.data_parallel = 1
    adapter._pending_state_slot_moves = moves
    return adapter


@pytest.mark.host
@pytest.mark.model
def test_slot_move_settlement_retires_consumed_decode_remap():
    adapter = _adapter_with_pending_moves({3: 0, 0: 3})

    adapter.note_state_slots_moved({0: 3, 3: 0})

    assert adapter._pending_state_slot_moves is None


@pytest.mark.host
@pytest.mark.model
def test_slot_move_settlement_rejects_a_different_permutation():
    adapter = _adapter_with_pending_moves({3: 0, 0: 3})

    with pytest.raises(RuntimeError, match="does not match the consumed decode remap"):
        adapter.note_state_slots_moved({2: 0, 0: 2})


@pytest.mark.host
@pytest.mark.model
def test_empty_slot_move_notification_is_a_noop():
    adapter = _adapter_with_pending_moves(None)

    adapter.note_state_slots_moved({})

    assert adapter._pending_state_slot_moves is None
