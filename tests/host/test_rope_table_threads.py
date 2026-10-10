# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host contracts: RoPE cos/sin tables are built on one torch intra-op thread."""

from importlib import import_module

import pytest
import torch

from tt_transformers.tensor_utils import single_threaded_torch

WEIGHT_UTILS_MODELS = (
    "deepseek_r1_distill_qwen_14b",
    "llama32_1b",
    "llama32_3b",
    "llama33_70b",
    "mistral_7b",
    "phi4",
    "qwen25_72b",
    "qwen25_7b",
    "qwen25_coder_32b",
    "qwen2_7b",
    "qwen3_32b",
)


@pytest.fixture
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.host
def test_single_threaded_torch_restores_thread_count(two_threads):
    with single_threaded_torch():
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == 2

    with pytest.raises(RuntimeError, match="boom"):
        with single_threaded_torch():
            raise RuntimeError("boom")
    assert torch.get_num_threads() == 2


@pytest.mark.host
@pytest.mark.parametrize("model", WEIGHT_UTILS_MODELS)
def test_hf_rope_tables_build_on_one_thread(model, two_threads):
    weight_utils = import_module(f"tt_transformers.models.{model}.weight_utils")
    seen = []

    def rotary_emb(x, position_ids):
        seen.append(torch.get_num_threads())
        table = torch.ones(1, x.shape[2], x.shape[3])
        return table, table

    weight_utils.build_rope_cos_sin_torch(rotary_emb, 32, 8, torch.bfloat16)

    assert seen == [1]
    assert torch.get_num_threads() == 2


@pytest.mark.host
def test_llama3_8b_rope_tables_build_on_one_thread(monkeypatch, two_threads):
    hf_generator = import_module("tt_transformers.models.llama3_8b.hf_generator")
    seen = []
    cos = torch.cos

    def recording_cos(x):
        seen.append(torch.get_num_threads())
        return cos(x)

    monkeypatch.setattr(torch, "cos", recording_cos)
    scaling = hf_generator.RopeScaling(
        rope_type="llama3",
        factor=8.0,
        original_max_position_embeddings=8192,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
    )
    hf_generator.compute_gather_cos_sin(dhead=16, end=64, theta=500000.0, rope_scaling=scaling)

    assert seen == [1]
    assert torch.get_num_threads() == 2
