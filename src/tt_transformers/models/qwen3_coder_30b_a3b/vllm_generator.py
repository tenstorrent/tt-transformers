# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM construction for Qwen3-Coder-30B-A3B: one lane over the four-device mesh, served through Qwen3-32B's
generator facade."""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

from tt_transformers.llm_runtime.config import (
    PagedKVCacheConfig,
    TraceConfig,
    TraceMode,
    WarmupConfig,
    fit_paged_kv_num_blocks,
)
from tt_transformers.models.qwen3_32b.vllm_generator import (
    Qwen3_32BGenerator,
    Qwen3_32BGeneratorConfig,
    _build_vllm_adapter,
    _model_kv_metadata,
)
from tt_transformers.models.qwen3_coder_30b_a3b.hf_generator import (
    Qwen3Coder30BA3BExecutorConfig,
    _load_model,
    build_qwen3_coder_30b_a3b_executor,
)
from tt_transformers.models.qwen3_coder_30b_a3b.model import (
    DEFAULT_HF_REVISION,
    Qwen3Coder30BA3BPagedAttentionConfig,
)

_PROVISIONAL_BLOCK_SIZE = 32


@dataclass(frozen=True)
class Qwen3Coder30BA3BGeneratorConfig(Qwen3_32BGeneratorConfig):
    hf_revision: str | None = DEFAULT_HF_REVISION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tt_data_parallel != 1:
            raise ValueError("Qwen3-Coder-30B-A3B serves one lane over all four devices (tt_data_parallel=1)")


class Qwen3Coder30BA3BGenerator(Qwen3_32BGenerator):
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "accepts_trace_mode": True,
        "supports_chunked_prefill": True,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D},
    }

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations="performance",
        trace_mode: TraceMode = "all",
        device_sampling_enabled: bool = True,
    ):
        hf_model = getattr(hf_config, "_name_or_path", None)
        if not hf_model:
            raise ValueError("hf_config must provide a non-empty _name_or_path")
        return build_qwen3_coder_30b_a3b_generator(
            Qwen3Coder30BA3BGeneratorConfig(
                hf_model=str(hf_model),
                mesh_device=mesh_device,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                tt_data_parallel=tt_data_parallel,
                optimizations="performance" if optimizations is None else optimizations,
                trace_mode=trace_mode,
                device_sampling_enabled=device_sampling_enabled,
            )
        )


def build_qwen3_coder_30b_a3b_generator(config: Qwen3Coder30BA3BGeneratorConfig) -> Qwen3Coder30BA3BGenerator:
    max_num_blocks = fit_paged_kv_num_blocks(
        -(-config.max_seq_len // _PROVISIONAL_BLOCK_SIZE) * config.max_batch_size,
        max_seq_len=config.max_seq_len,
        block_size=_PROVISIONAL_BLOCK_SIZE,
    )
    llm = _load_model(
        mesh_device=config.mesh_device,
        hf_model=config.hf_model,
        hf_revision=config.hf_revision,
        instruct=False,
        max_batch_size=config.max_batch_size,
        max_seq_len=config.max_seq_len,
        optimizations=config.optimizations,
        n_layers=config.n_layers,
        dtype=ttnn.bfloat8_b,
        paged_attention_config=Qwen3Coder30BA3BPagedAttentionConfig(
            block_size=_PROVISIONAL_BLOCK_SIZE, max_num_blocks=max_num_blocks
        ),
    )
    lane = None
    try:
        kv_cache_dtypes, _, _, _ = _model_kv_metadata(llm.model)
        lane = build_qwen3_coder_30b_a3b_executor(
            llm,
            Qwen3Coder30BA3BExecutorConfig(
                trace=TraceConfig(mode=config.trace_mode),
                warmup=WarmupConfig(include_decode_top_k=config.device_sampling_enabled),
                paged_kv_cache=PagedKVCacheConfig(
                    block_size=_PROVISIONAL_BLOCK_SIZE,
                    max_num_blocks=max_num_blocks,
                    dtype=kv_cache_dtypes[0],
                ),
                device_sampling_enabled=config.device_sampling_enabled,
                sequential_prefill=True,
                salt_duplicate_seeds=False,
            ),
        )
        adapter = _build_vllm_adapter(lane)
    except BaseException:
        (lane or llm).cleanup()
        raise
    return Qwen3Coder30BA3BGenerator(lane, adapter)
