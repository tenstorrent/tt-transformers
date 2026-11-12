# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-layer Qwen3-Coder-30B-A3B execution smoke on Blackhole TP4."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import ttnn
from tests.integration.cleanup_utils import cleanup_model_case
from tests.integration.run_helpers import make_contiguous_page_table

from tt_transformers.models.qwen3_coder_30b_a3b.hf_generator import (
    EagerQwen3Coder30BA3BExecutor,
    _load_model,
)
from tt_transformers.models.qwen3_coder_30b_a3b.model import (
    DEFAULT_HF_MODEL,
    QWEN3_CODER_30B_A3B_PERFORMANCE,
    SUPPORTED_CLUSTER_TYPES,
)

_BLOCK_SIZE = 32
_PROMPT_LEN = 128
_MAX_SEQ_LEN = 512

pytestmark = [
    pytest.mark.timeout(1800),
    pytest.mark.parametrize(
        "ttnn_mesh_device",
        [
            {
                "mesh_shape": (1, 4),
                "trace_region_size": 0,
                "num_command_queues": 1,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            }
        ],
        indirect=True,
        scope="module",
        ids=["physical-blackhole-tp4-ring"],
    ),
]


def _cache_dir() -> Path:
    if root := os.getenv("TT_CACHE_PATH"):
        return Path(root) / "P150x4"
    return Path("model_cache") / DEFAULT_HF_MODEL / "P150x4"


@pytest.fixture(scope="module")
def production_model(ttnn_mesh_device, require_blackhole_mesh_device):
    assert ttnn.cluster.get_cluster_type() in SUPPORTED_CLUSTER_TYPES
    assert ttnn_mesh_device.get_num_devices() == 4
    ttnn_mesh_device.enable_program_cache()
    llm = None
    try:
        llm = _load_model(
            ttnn_mesh_device,
            hf_model=os.getenv("HF_MODEL", DEFAULT_HF_MODEL),
            max_batch_size=1,
            max_seq_len=_MAX_SEQ_LEN,
            n_layers=1,
            optimizations=QWEN3_CODER_30B_A3B_PERFORMANCE,
            cache_dir=_cache_dir(),
        )
        layer = llm.model.layers[0]
        assert layer.mlp.config.experts_per_device == 32
        assert layer.mlp.config.prefill_num_reduce_scatter_links == 2
        assert layer.mlp.config.num_reduce_scatter_links == 1
        assert layer.self_attn.config.num_reduce_scatter_links == 2
        assert layer.self_attn.config.decode_num_reduce_scatter_links == 1
        assert layer.self_attn.config.kv_cache_dtype == ttnn.bfloat16
        assert len(llm.model.lm_head.config.output_weights) == 1
        yield llm
    finally:
        cleanup_model_case(None if llm is None else llm.model, ttnn_mesh_device)
        ttnn_mesh_device.disable_and_clear_program_cache()
        ttnn.SetDefaultDevice(None)


@pytest.mark.device
@pytest.mark.model
@pytest.mark.blackhole
@pytest.mark.p150x4
def test_qwen3_coder_one_layer_prefill_decode(ttnn_mesh_device, production_model):
    model = production_model.model
    runtime = production_model.runtime_config
    executor = EagerQwen3Coder30BA3BExecutor(model, ttnn_mesh_device)
    try:
        kv_shape = (
            _MAX_SEQ_LEN // _BLOCK_SIZE,
            runtime.n_kv_heads // ttnn_mesh_device.get_num_devices(),
            _BLOCK_SIZE,
            runtime.head_dim,
        )
        kv_cache = executor.allocate_kv_cache(kv_shape, torch.bfloat16, runtime.n_layers)
        page_table = make_contiguous_page_table(1, _MAX_SEQ_LEN, _BLOCK_SIZE)
        tokens = (torch.arange(_PROMPT_LEN, dtype=torch.long).reshape(1, -1) + 17) % 32000
        logits = executor.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([_PROMPT_LEN], dtype=torch.long),
            empty_slots=[0],
            execution=executor.eager_execution,
        )
        assert tuple(logits.shape) == (1, 1, model.vocab_size)
        assert torch.isfinite(logits).all()

        decode_logits, log_probs = executor.decode_forward(
            torch.tensor([int(logits.argmax())], dtype=torch.long),
            torch.tensor([_PROMPT_LEN], dtype=torch.long),
            page_table,
            kv_cache=kv_cache,
            execution=executor.eager_execution,
        )
        assert log_probs is None
        assert tuple(decode_logits.shape) == (1, 1, model.vocab_size)
        assert torch.isfinite(decode_logits).all()
    finally:
        executor.cleanup()
