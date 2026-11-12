# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import ttnn

from tt_transformers.models.qwen3_32b.model import Qwen3_32B, Qwen3_32BDecoderLayer
from tt_transformers.models.qwen3_coder_30b_a3b import model as coder_model
from tt_transformers.models.qwen3_coder_30b_a3b.model import (
    QWEN3_CODER_30B_A3B_ACCURACY,
    QWEN3_CODER_30B_A3B_PERFORMANCE,
    Qwen3Coder30BA3B,
    Qwen3Coder30BA3BDecoderLayer,
    _build_moe,
    _expert_weights,
    _validate_checkpoint_config,
)


def _hf_config(**overrides):
    values = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "hidden_act": "silu",
        "decoder_sparse_step": 1,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "vocab_size": 151936,
        "rope_theta": 10_000_000,
        "norm_topk_prob": True,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "mlp_only_layers": [],
        "use_sliding_window": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.host
@pytest.mark.model
def test_checkpoint_profile_accepts_only_the_supported_geometry():
    _validate_checkpoint_config(_hf_config(), num_devices=4)
    nested_rope_config = _hf_config(rope_parameters={"rope_theta": 10_000_000})
    del nested_rope_config.rope_theta
    _validate_checkpoint_config(nested_rope_config, num_devices=4)

    with pytest.raises(ValueError, match="geometry"):
        _validate_checkpoint_config(_hf_config(num_experts=64), num_devices=4)
    with pytest.raises(ValueError, match="four devices"):
        _validate_checkpoint_config(_hf_config(), num_devices=8)
    with pytest.raises(ValueError, match="normalized top-k"):
        _validate_checkpoint_config(_hf_config(norm_topk_prob=False), num_devices=4)
    with pytest.raises(ValueError, match="rope_theta"):
        _validate_checkpoint_config(_hf_config(rope_theta=1_000_000), num_devices=4)


@pytest.mark.host
@pytest.mark.model
def test_hf_gate_up_tensor_splits_gate_first():
    gate = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    up = gate + 100
    down = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    hf_mlp = SimpleNamespace(
        experts=SimpleNamespace(
            gate_up_proj=torch.nn.Parameter(torch.cat([gate, up], dim=1)),
            down_proj=torch.nn.Parameter(down),
        )
    )

    actual_gate, actual_up, actual_down = _expert_weights(hf_mlp)

    assert torch.equal(actual_gate, gate)
    assert torch.equal(actual_up, up)
    assert torch.equal(actual_down, down)


@pytest.mark.host
@pytest.mark.model
def test_performance_profile_matches_bundle_precision():
    for profile in (QWEN3_CODER_30B_A3B_ACCURACY, QWEN3_CODER_30B_A3B_PERFORMANCE):
        assert profile.expert_gate_up_dtype == profile.expert_down_dtype == ttnn.bfloat4_b
    assert QWEN3_CODER_30B_A3B_PERFORMANCE.kv_cache_dtype == ttnn.bfloat16
    assert QWEN3_CODER_30B_A3B_PERFORMANCE.expert_prefill_gate_up_in0_block_w == 16
    assert QWEN3_CODER_30B_A3B_PERFORMANCE.expert_prefill_down_in0_block_w == 12
    assert QWEN3_CODER_30B_A3B_PERFORMANCE.attn_decode_qkv_kernel is None


@pytest.mark.host
@pytest.mark.model
def test_moe_builder_selects_four_way_expert_parallelism(monkeypatch):
    captured = {}
    weights = SimpleNamespace(
        router=SimpleNamespace(source=object()),
        up_gate=SimpleNamespace(source=object()),
        down=SimpleNamespace(source=object()),
    )

    def fake_weights(*args, **kwargs):
        captured["weight_args"] = args
        captured["weight_kwargs"] = kwargs
        return weights

    def fake_from_config(config):
        captured["config"] = config
        return SimpleNamespace(
            config=config,
            load_device_weights=lambda: captured.__setitem__("loaded", True),
        )

    monkeypatch.setattr(coder_model, "moe_weights_from_hf", fake_weights)
    monkeypatch.setattr(coder_model.MoE1D, "from_config", staticmethod(fake_from_config))
    hf_mlp = SimpleNamespace(
        gate=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(8, 4))),
        experts=SimpleNamespace(
            gate_up_proj=torch.nn.Parameter(torch.zeros(8, 6, 4)),
            down_proj=torch.nn.Parameter(torch.zeros(8, 4, 3)),
        ),
    )
    config = SimpleNamespace(dim=2048, hidden_dim=768, num_experts=128, num_experts_per_token=8, num_devices=4)

    _build_moe(
        hf_mlp=hf_mlp,
        qcfg=config,
        mesh_device=SimpleNamespace(arch=lambda: ttnn.device.Arch.BLACKHOLE),
        tt_ccl=object(),
        topology=ttnn.Topology.Ring,
        precision=QWEN3_CODER_30B_A3B_PERFORMANCE,
        cache_path=None,
        prefix="layer0",
    )

    assert captured["weight_kwargs"]["num_devices"] == 4
    assert captured["weight_kwargs"]["parallelism"] == "expert"
    assert captured["weight_kwargs"]["gate_first"] is True
    assert captured["config"].experts_per_device == 32
    assert captured["config"].k == 8
    assert captured["config"].parallelism == "expert"
    assert captured["config"].num_reduce_scatter_links == 1
    assert captured["config"].prefill_num_reduce_scatter_links == 2
    assert captured["config"].prefill_gate_up_program_config.in0_block_w == 16
    assert captured["config"].prefill_down_program_config.in0_block_w == 12
    assert captured["config"].prefill_gate_up_program_config.compute_with_storage_grid_size == ttnn.CoreCoord(6, 8)
    assert captured["config"].prefill_down_program_config.compute_with_storage_grid_size == ttnn.CoreCoord(8, 8)
    assert captured["config"].prefill_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert captured["config"].prefill_sorted_min_tokens == 128
    assert captured["config"].prefill_sorted_down_in0_block_w == 12
    assert captured["loaded"] is True
    assert all(w.source is None for w in (weights.router, weights.up_gate, weights.down))


@pytest.mark.host
@pytest.mark.model
def test_model_reuses_qwen_runtime_with_a_moe_decoder_layer():
    assert issubclass(Qwen3Coder30BA3B, Qwen3_32B)
    assert issubclass(Qwen3Coder30BA3BDecoderLayer, Qwen3_32BDecoderLayer)
