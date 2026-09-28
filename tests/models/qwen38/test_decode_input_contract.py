# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from tt_transformers.models.qwen38.qwen36_vllm import Qwen36ForCausalLM
from tt_transformers.models.qwen38.qwen36_vllm_dflash import Qwen36DFlashForCausalLM
from tt_transformers.models.qwen38.qwen_runtime import Generator


def test_qwen_advertises_explicit_decode_input_contract_without_async():
    assert Qwen36ForCausalLM.decode_input_update_contract == 1
    assert Qwen36DFlashForCausalLM.decode_input_update_contract == 1
    assert Qwen36ForCausalLM.model_capabilities["supports_async_decode"] is False
    assert Qwen36DFlashForCausalLM.model_capabilities["supports_async_decode"] is False


def test_contract_sampling_reset_maps_to_legacy_generator_seed_reset():
    captured = {}

    def fake_decode(self, *args, **kwargs):
        captured.update(kwargs)
        return "decoded"

    with patch.object(Generator, "decode_forward", fake_decode):
        adapter = object.__new__(Qwen36ForCausalLM)
        adapter.data_parallel = 1
        adapter._decode_logged = True
        adapter.model = [SimpleNamespace(args=SimpleNamespace(max_batch_size=1), num_devices=1)]

        result = adapter.decode_forward(
            tokens=None,
            reset_sampling_state=True,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=True,
        )

    assert result == "decoded"
    assert captured["reset_batch"] is True
    assert captured["reload_inputs"] is True
    assert captured["reload_page_table"] is False
    assert captured["reload_sampling_params"] is True
