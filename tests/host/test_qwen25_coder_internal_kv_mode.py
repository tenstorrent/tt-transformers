# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tt_transformers.models.qwen25_coder_32b import hf_adaptor
from tt_transformers.models.qwen25_coder_32b import model as qwen_model


@pytest.mark.host
@pytest.mark.model
def test_adaptor_distinguishes_omitted_paged_kv_default_from_explicit_internal_mode():
    default = hf_adaptor._resolve_paged_attention_config(
        None,
        max_batch_size=3,
        max_seq_len=65,
    )
    assert default == qwen_model.Qwen25Coder32BPagedAttentionConfig(block_size=32, max_num_blocks=9)

    assert (
        hf_adaptor._resolve_paged_attention_config(
            hf_adaptor._INTERNAL_KV_CACHE_CONFIG,
            max_batch_size=3,
            max_seq_len=65,
        )
        is None
    )


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("executor_mode", [False, True])
def test_compatibility_constructor_selects_matching_kv_ownership(monkeypatch, executor_mode):
    captured = {}
    expected_model = object()

    def fake_from_pretrained(mesh_device, **kwargs):
        captured.update(mesh_device=mesh_device, **kwargs)
        return SimpleNamespace(model=expected_model)

    monkeypatch.setattr(hf_adaptor, "from_pretrained", fake_from_pretrained)
    mesh_device = object()
    actual_model = qwen_model.Qwen25Coder32B.from_pretrained(
        mesh_device,
        max_batch_size=2,
        max_seq_len=65,
        block_size=32,
        executor_mode=executor_mode,
    )

    assert actual_model is expected_model
    assert captured["mesh_device"] is mesh_device
    paged = captured["paged_attention_config"]
    if executor_mode:
        assert paged == qwen_model.Qwen25Coder32BPagedAttentionConfig(block_size=32, max_num_blocks=6)
    else:
        assert paged is hf_adaptor._INTERNAL_KV_CACHE_CONFIG
        assert (
            hf_adaptor._resolve_paged_attention_config(
                paged,
                max_batch_size=2,
                max_seq_len=65,
            )
            is None
        )
