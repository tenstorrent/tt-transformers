# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face loading for Qwen3-Coder-30B-A3B.

The decoder reuses Qwen3-32B's runtime: the causal-LM wrapper, the executor and its configuration. This module owns
the checkpoint, the cache location and the default paged-KV sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import ttnn
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from tt_transformers.cache_environment import (
    environment_flag,
    offline_mode,
    report_model_preflight,
    resolve_model_cache,
)
from tt_transformers.device_utils import cleanup_object_graph
from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig, fit_paged_kv_num_blocks
from tt_transformers.llm_runtime.tensor_resources import attach_cleanup_failures
from tt_transformers.models.qwen3_32b import hf_generator as qwen3
from tt_transformers.models.qwen3_32b.hf_generator import Qwen3_32BExecutor, Qwen3_32BExecutorConfig
from tt_transformers.models.qwen3_coder_30b_a3b.model import (
    DEFAULT_HF_MODEL,
    DEFAULT_HF_REVISION,
    QWEN3_CODER_30B_A3B_ACCURACY,
    QWEN3_CODER_30B_A3B_PERFORMANCE,
    Qwen3Coder30BA3B,
    Qwen3Coder30BA3BPagedAttentionConfig,
    Qwen3Coder30BA3BPrecisionConfig,
    _resolve_sku,
    _validate_checkpoint_config,
)

MAX_PREFILL_CHUNK_SIZE = 4096

Qwen3Coder30BA3BForCausalLM = qwen3.Qwen3_32BForCausalLM
Qwen3Coder30BA3BExecutorConfig = Qwen3_32BExecutorConfig
EagerQwen3Coder30BA3BExecutor = qwen3.EagerQwen3_32BExecutor
TracedQwen3Coder30BA3BExecutor = qwen3.TracedQwen3_32BExecutor


@dataclass(frozen=True)
class Qwen3Coder30BA3BRuntimeConfig(qwen3.Qwen3_32BRuntimeConfig):
    """Qwen3-32B's runtime limits, for the Coder checkpoint."""


def build_qwen3_coder_30b_a3b_executor(
    llm: Qwen3Coder30BA3BForCausalLM, config: Qwen3_32BExecutorConfig
) -> Qwen3_32BExecutor:
    return Qwen3_32BExecutor(llm.model, llm.runtime_config, config)


def _precision(optimizations) -> Qwen3Coder30BA3BPrecisionConfig:
    if isinstance(optimizations, Qwen3Coder30BA3BPrecisionConfig):
        return optimizations
    if optimizations not in ("accuracy", "performance"):
        raise TypeError("optimizations must be 'accuracy', 'performance', or Qwen3Coder30BA3BPrecisionConfig")
    return QWEN3_CODER_30B_A3B_PERFORMANCE if optimizations == "performance" else QWEN3_CODER_30B_A3B_ACCURACY


def load_tokenizer(hf_model: str, hf_revision: str | None = DEFAULT_HF_REVISION):
    tokenizer = AutoTokenizer.from_pretrained(hf_model, revision=hf_revision, local_files_only=offline_mode())
    tokenizer.stop_tokens = list(qwen3._qwen_stop_token_ids(tokenizer))
    return tokenizer


def _cache_path(hf_model: str, mesh_device, cache_dir, *, hf_revision: str | None = None, dtype=None) -> Path:
    resolution = resolve_model_cache(
        hf_model_id=hf_model,
        hf_revision=hf_revision,
        topology="P150x4",
        cache_dir=cache_dir,
        mesh_device=mesh_device,
        dtype=dtype,
    )
    if dtype is not None:
        report_model_preflight(
            hf_model_id=hf_model,
            hf_revision=hf_revision,
            cache_path=resolution.path,
            topology="P150x4",
            mesh_device=mesh_device,
            dtype=dtype,
            cache_dir=cache_dir,
            cache_resolution=resolution,
        )
    return resolution.path


def _load_model(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = DEFAULT_HF_REVISION,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Qwen3Coder30BA3BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Qwen3Coder30BA3BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> Qwen3Coder30BA3BForCausalLM:
    num_devices = mesh_device.get_num_devices()
    _resolve_sku(mesh_device)
    hf_config = AutoConfig.from_pretrained(hf_model, revision=hf_revision, local_files_only=offline_mode())
    _validate_checkpoint_config(hf_config, num_devices=num_devices)
    resolved_layers = hf_config.num_hidden_layers if n_layers is None else n_layers
    if not 1 <= resolved_layers <= hf_config.num_hidden_layers:
        raise ValueError(f"n_layers must be in [1, {hf_config.num_hidden_layers}], got {resolved_layers}")
    precision = _precision(optimizations)
    cache_path = _cache_path(hf_model, mesh_device, cache_dir, hf_revision=hf_revision, dtype=dtype)
    paged = paged_attention_config
    model = Qwen3Coder30BA3B.from_pretrained(
        mesh_device,
        hf_model,
        revision=hf_revision,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=resolved_layers,
        cache_dir=cache_path,
        precision=precision,
        block_size=32 if paged is None else int(paged.block_size),
        max_num_blocks=None if paged is None else int(paged.max_num_blocks),
        executor_mode=True,
    )
    try:
        runtime = Qwen3Coder30BA3BRuntimeConfig(
            model_name=hf_model,
            model_cache_path=cache_path,
            max_prefill_chunk_size=MAX_PREFILL_CHUNK_SIZE,
            max_context_len=int(getattr(hf_config, "max_position_embeddings", max_seq_len)),
            max_seq_len=max_seq_len,
            trace_prefill_supported_seq_lens=qwen3._trace_seq_lens(num_devices, MAX_PREFILL_CHUNK_SIZE, max_seq_len),
            n_layers=resolved_layers,
            n_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            max_batch_size=max_batch_size,
            cluster_shape=list(mesh_device.shape),
            kv_cache_dtype=precision.kv_cache_dtype,
            disable_batched_prefill=True,
            batched_prefill_batched_extract=not environment_flag("DISABLE_BATCHED_EXTRACT"),
        )
        llm = Qwen3Coder30BA3BForCausalLM(
            model=model,
            tokenizer=load_tokenizer(hf_model, hf_revision),
            runtime_config=runtime,
            instruct=instruct,
        )
        llm._hf_model_config = hf_config
        return llm
    except BaseException as primary:
        try:
            cleanup_object_graph(model)
        except BaseException as error:
            attach_cleanup_failures(primary, (error,))
        raise


def from_pretrained(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = DEFAULT_HF_REVISION,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Qwen3Coder30BA3BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Qwen3Coder30BA3BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
    executor_config: Qwen3Coder30BA3BExecutorConfig | None = None,
) -> Qwen3Coder30BA3BForCausalLM:
    """Load the model with one executor ready for ``generate()``.

    Inputs and generated IDs are CPU PyTorch tensors and the mesh stays caller-owned. By default execution is eager
    with host token selection; pass a ``Qwen3Coder30BA3BExecutorConfig`` for trace and sampling policy.
    """
    if executor_config is not None and not isinstance(executor_config, Qwen3Coder30BA3BExecutorConfig):
        raise TypeError("executor_config must be a Qwen3Coder30BA3BExecutorConfig")
    for name, value in (("max_batch_size", max_batch_size), ("max_seq_len", max_seq_len)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if paged_attention_config is None:
        if executor_config is not None:
            kv = executor_config.paged_kv_cache
            block_size, max_num_blocks = kv.block_size, kv.max_num_blocks
        else:
            block_size = 32
            max_num_blocks = fit_paged_kv_num_blocks(
                -(-max_seq_len // block_size) * max_batch_size, max_seq_len=max_seq_len, block_size=block_size
            )
        paged_attention_config = Qwen3Coder30BA3BPagedAttentionConfig(
            block_size=block_size, max_num_blocks=max_num_blocks
        )
    if executor_config is not None:
        kv = executor_config.paged_kv_cache
        if (kv.block_size, kv.max_num_blocks) != (
            paged_attention_config.block_size,
            paged_attention_config.max_num_blocks,
        ):
            raise ValueError("paged_attention_config must match executor_config.paged_kv_cache geometry")
        if kv.num_blocks is None:
            executor_config = replace(executor_config, paged_kv_cache=replace(kv, num_blocks=kv.max_num_blocks))
    llm = _load_model(
        mesh_device,
        hf_model=hf_model,
        hf_revision=hf_revision,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        optimizations=optimizations,
        n_layers=n_layers,
        dtype=dtype,
        paged_attention_config=paged_attention_config,
        cache_dir=cache_dir,
    )
    try:
        if executor_config is None:
            executor_config = Qwen3Coder30BA3BExecutorConfig(
                trace=TraceConfig(mode="none"),
                warmup=WarmupConfig(),
                paged_kv_cache=PagedKVCacheConfig(
                    block_size=paged_attention_config.block_size,
                    max_num_blocks=paged_attention_config.max_num_blocks,
                    num_blocks=paged_attention_config.max_num_blocks,
                    dtype=llm.model.layers[0].attention.config.kv_cache_dtype,
                ),
                device_sampling_enabled=False,
            )
        llm.executor = build_qwen3_coder_30b_a3b_executor(llm, executor_config)
        try:
            llm._hf_generation_config = GenerationConfig.from_pretrained(
                hf_model, revision=hf_revision, local_files_only=offline_mode()
            )
        except OSError:
            llm._hf_generation_config = GenerationConfig.from_model_config(llm._hf_model_config)
        return llm
    except BaseException as primary:
        try:
            llm.cleanup()
        except BaseException as error:
            attach_cleanup_failures(primary, (error,))
        raise
