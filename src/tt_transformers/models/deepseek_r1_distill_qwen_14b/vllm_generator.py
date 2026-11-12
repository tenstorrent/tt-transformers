# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM construction and compatibility delegation for DeepSeek-R1-Distill-Qwen-14B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ttnn

from tt_transformers.llm_runtime.config import (
    PagedKVCacheConfig,
    TraceConfig,
    TraceMode,
    WarmupConfig,
    fit_paged_kv_num_blocks,
)
from tt_transformers.llm_runtime.lane_group import LaneGroupExecutor
from tt_transformers.llm_runtime.vllm_adapter import VLLMAdapter, VLLMAdapterConfig
from tt_transformers.llm_runtime.vllm_facade import CompilingVLLMGeneratorFacade
from tt_transformers.models.deepseek_r1_distill_qwen_14b.hf_generator import (
    DEFAULT_HF_REVISION,
    DeepSeekR1Qwen14BExecutorConfig,
    _load_model,
    build_deepseek_r1_distill_qwen_14b_executor,
)
from tt_transformers.models.deepseek_r1_distill_qwen_14b.model import DeepSeekR1Qwen14BPagedAttentionConfig

_PROVISIONAL_BLOCK_SIZE = 32


@dataclass(frozen=True)
class DeepSeekR1Qwen14BGeneratorConfig:
    """Validated construction inputs for one vLLM-facing Qwen generator."""

    hf_model: str
    mesh_device: Any
    max_batch_size: int
    max_seq_len: int
    hf_revision: str | None = DEFAULT_HF_REVISION
    n_layers: int | None = None
    tt_data_parallel: int = 1
    optimizations: Any = "performance"
    trace_mode: TraceMode = "all"
    device_sampling_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.hf_model, str) or not self.hf_model:
            raise ValueError("hf_model must be a non-empty string")
        if self.mesh_device is None:
            raise ValueError("mesh_device is required")
        _validate_positive_int("max_batch_size", self.max_batch_size)
        _validate_positive_int("max_seq_len", self.max_seq_len)
        _validate_positive_int("tt_data_parallel", self.tt_data_parallel)
        if self.n_layers is not None:
            _validate_positive_int("n_layers", self.n_layers)
        if self.max_batch_size % self.tt_data_parallel != 0:
            raise ValueError(
                f"max_batch_size={self.max_batch_size} must be divisible by tt_data_parallel={self.tt_data_parallel}"
            )
        if not isinstance(self.device_sampling_enabled, bool):
            raise TypeError("device_sampling_enabled must be bool")
        TraceConfig(mode=self.trace_mode)


class DeepSeekR1Qwen14BGenerator(CompilingVLLMGeneratorFacade):
    """vLLM facade for DeepSeek-R1-Distill-Qwen-14B."""

    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "accepts_trace_mode": True,
        "supports_chunked_prefill": True,
        "supports_device_grammar": True,
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
        hf_revision: str | None = DEFAULT_HF_REVISION,
        device_sampling_enabled: bool = True,
    ):
        """Build the configured single-lane or data-parallel Qwen target."""

        hf_model = getattr(hf_config, "_name_or_path", None)
        if not hf_model:
            raise ValueError("hf_config must provide a non-empty _name_or_path")
        return build_deepseek_r1_distill_qwen_14b_generator(
            DeepSeekR1Qwen14BGeneratorConfig(
                hf_model=str(hf_model),
                hf_revision=hf_revision,
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


def build_deepseek_r1_distill_qwen_14b_generator(
    config: DeepSeekR1Qwen14BGeneratorConfig,
) -> DeepSeekR1Qwen14BGenerator:
    """Construct lane-local models/executors and compose their shared target surface."""

    per_lane_max_batch_size = config.max_batch_size // config.tt_data_parallel
    submeshes = (
        [config.mesh_device]
        if config.tt_data_parallel == 1
        else list(_create_submeshes(config.mesh_device, config.tt_data_parallel))
    )
    if len(submeshes) != config.tt_data_parallel:
        raise ValueError(f"Expected {config.tt_data_parallel} submeshes, got {len(submeshes)}")

    max_num_blocks = fit_paged_kv_num_blocks(
        (config.max_seq_len + _PROVISIONAL_BLOCK_SIZE - 1) // _PROVISIONAL_BLOCK_SIZE + per_lane_max_batch_size,
        max_seq_len=config.max_seq_len,
        block_size=_PROVISIONAL_BLOCK_SIZE,
    )
    lanes = []
    try:
        for submesh in submeshes:
            paged_attention_config = DeepSeekR1Qwen14BPagedAttentionConfig(
                block_size=_PROVISIONAL_BLOCK_SIZE,
                max_num_blocks=max_num_blocks,
            )
            llm = _load_model(
                mesh_device=submesh,
                hf_model=config.hf_model,
                hf_revision=config.hf_revision,
                instruct=True,
                max_batch_size=per_lane_max_batch_size,
                max_seq_len=config.max_seq_len,
                optimizations=config.optimizations,
                n_layers=config.n_layers,
                dtype=ttnn.bfloat8_b,
                paged_attention_config=paged_attention_config,
            )
            model_kv_cache_dtypes, _, _, _ = _model_kv_metadata(llm.model)
            executor_config = DeepSeekR1Qwen14BExecutorConfig(
                trace=TraceConfig(mode=config.trace_mode),
                warmup=WarmupConfig(),
                paged_kv_cache=PagedKVCacheConfig(
                    block_size=_PROVISIONAL_BLOCK_SIZE,
                    max_num_blocks=max_num_blocks,
                    dtype=model_kv_cache_dtypes[0],
                ),
                device_sampling_enabled=config.device_sampling_enabled,
                sequential_prefill=True,
                salt_duplicate_seeds=False,
            )
            lanes.append(build_deepseek_r1_distill_qwen_14b_executor(llm, executor_config))

        adapter = _build_vllm_adapter(lanes[0])
    except BaseException as primary:
        _cleanup_after_construction_failure(lanes, primary)
        raise

    target = lanes[0] if config.tt_data_parallel == 1 else LaneGroupExecutor(lanes, mesh_device=config.mesh_device)
    return DeepSeekR1Qwen14BGenerator(target, adapter)


def _build_vllm_adapter(lane) -> VLLMAdapter:
    model_kv_cache_dtypes, num_layers, kv_heads_per_device, head_dim = _model_kv_metadata(lane.model)
    return VLLMAdapter(
        VLLMAdapterConfig.resolve(
            trace=lane.config.trace,
            paged_kv_cache=lane.config.paged_kv_cache,
            expected_num_layers=num_layers,
            expected_kv_heads_per_device=kv_heads_per_device,
            expected_head_dim=head_dim,
            model_kv_cache_dtype=model_kv_cache_dtypes,
            request_state_fields=lane._request_state_fields,
        )
    )


def _model_kv_metadata(model) -> tuple[tuple[Any, ...], int, int, int]:
    layers = tuple(getattr(model, "layers", ()))
    if not layers:
        raise ValueError("DeepSeek-R1-Distill-Qwen-14B model must contain at least one attention layer")

    attention_configs = tuple(layer.attention.config for layer in layers)
    model_config = model.config
    num_layers = int(model_config.n_layers)
    if len(attention_configs) != num_layers:
        raise ValueError(f"Model config declares {num_layers} layers but exposes {len(attention_configs)}")

    num_devices = int(model_config.num_devices)
    n_kv_heads = int(attention_configs[0].n_kv_heads)
    if n_kv_heads % num_devices != 0:
        raise ValueError(f"n_kv_heads={n_kv_heads} must be divisible by num_devices={num_devices}")

    head_dim = int(attention_configs[0].head_dim)
    if any(
        int(attention_config.n_kv_heads) != n_kv_heads or int(attention_config.head_dim) != head_dim
        for attention_config in attention_configs
    ):
        raise ValueError("Every Qwen layer must expose the same KV head shape")

    return (
        tuple(attention_config.kv_cache_dtype for attention_config in attention_configs),
        num_layers,
        n_kv_heads // num_devices,
        head_dim,
    )


def _create_submeshes(mesh_device, tt_data_parallel):
    from tt_transformers.mesh_utils import create_submeshes

    return create_submeshes(mesh_device, tt_data_parallel)


def _cleanup_after_construction_failure(lanes, primary):
    failures = []
    for lane in lanes:
        try:
            lane.cleanup()
        except BaseException as error:
            failures.append(error)
    if failures:
        setattr(primary, "cleanup_failures", failures)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
