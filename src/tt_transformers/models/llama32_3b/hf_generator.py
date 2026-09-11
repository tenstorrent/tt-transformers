# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face generation and loading for the TTTv2 Llama-3.2-3B path."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import ttnn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from tt_transformers.cache_environment import (
    environment_flag,
    offline_mode,
    report_model_preflight,
    resolve_model_cache,
)
from tt_transformers.device_utils import cleanup_object_graph
from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from tt_transformers.llm_runtime.tensor_resources import attach_cleanup_failures
from tt_transformers.models.llama3_executor import Llama32_3BExecutor, Llama32_3BExecutorConfig
from tt_transformers.models.llama32_3b import weight_utils
from tt_transformers.models.llama32_3b.model import (
    LLAMA32_3B_ACCURACY,
    LLAMA32_3B_PERFORMANCE,
    Llama32_3BLayerWeights,
    Llama32_3BModelParameters,
    Llama32_3BPagedAttentionConfig,
    Llama32_3BPrecisionConfig,
    Llama32_3BTransformer1D,
    Llama32_3BWeights,
    build_llama32_3b_transformer_1d_config,
)

DEFAULT_HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_HF_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"


def _resolve_hf_revision(hf_model: str, hf_revision: str | None) -> str | None:
    """Pin the default checkpoint without applying its revision to custom IDs."""

    if hf_revision is not None:
        return hf_revision
    return DEFAULT_HF_REVISION if hf_model == DEFAULT_HF_MODEL else None


@dataclass(frozen=True)
class Llama32_3BGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Llama32_3BRuntimeConfig:
    model_name: str
    model_cache_path: Path | None
    max_prefill_chunk_size: int
    max_context_len: int
    max_seq_len: int
    trace_prefill_supported_seq_lens: tuple[int, ...]
    supports_batched_prefill: bool = True
    max_prefill_batch_size: int = 32
    disable_batched_prefill: bool = False
    batched_prefill_batched_extract: bool = True

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int = 0) -> bool:
        return (
            prefill_seq_len in self.trace_prefill_supported_seq_lens
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
        )


def _chat_template_ids(encoded):
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return list(encoded)


def encode_prompt(tokenizer, prompt_text, system_prompt_text=None, *, instruct=True):
    if instruct:
        chat = []
        if isinstance(prompt_text, str):
            if system_prompt_text:
                chat.append({"role": "system", "content": system_prompt_text})
            if prompt_text:
                chat.append({"role": "user", "content": prompt_text})
        else:
            chat = prompt_text
        try:
            return _chat_template_ids(tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True))
        except ValueError:
            pass
    return tokenizer.encode(prompt_text, add_special_tokens=False)


@dataclass
class Llama32_3BForCausalLM:
    model: Llama32_3BTransformer1D
    tokenizer: Any
    runtime_config: Llama32_3BRuntimeConfig
    instruct: bool = True
    generation_config: Llama32_3BGenerationConfig = field(default_factory=Llama32_3BGenerationConfig)
    executor: object | None = field(default=None, repr=False)

    def __post_init__(self):
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            stops = tuple(getattr(self.tokenizer, "stop_tokens", []) or [])
            self.generation_config = Llama32_3BGenerationConfig(
                max_decode_tokens=self.generation_config.max_decode_tokens,
                temperature=self.generation_config.temperature,
                top_k=self.generation_config.top_k,
                top_p=self.generation_config.top_p,
                stop_token_ids=stops,
            )

    @property
    def model_name(self):
        return self.runtime_config.model_name

    @property
    def model_cache_path(self):
        return self.runtime_config.model_cache_path

    @property
    def max_seq_len(self):
        return self.model.config.max_seq_len

    @property
    def max_context_len(self):
        return self.runtime_config.max_context_len

    def encode_prompt(self, prompt_text, system_prompt_text=None, instruct=None):
        return encode_prompt(
            self.tokenizer,
            prompt_text,
            system_prompt_text,
            instruct=self.instruct if instruct is None else instruct,
        )

    def encode_chat(self, messages):
        return self.encode_prompt(messages, instruct=True)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate CPU token IDs using the model's internally owned executor."""
        from tt_transformers.models.hf_generation import generate

        return generate(self, input_ids, attention_mask=attention_mask, **kwargs)

    def cleanup(self):
        """Release execution and model resources while leaving the supplied mesh open."""
        from tt_transformers.models.hf_generation import cleanup

        return cleanup(self)


def load_tokenizer(hf_model: str, hf_revision: str | None = None):
    hf_revision = _resolve_hf_revision(hf_model, hf_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        revision=hf_revision,
        local_files_only=offline_mode(),
    )
    if not hasattr(tokenizer, "stop_tokens") or tokenizer.stop_tokens is None:
        stops = getattr(tokenizer, "eos_token_id", None)
        tokenizer.stop_tokens = [] if stops is None else ([stops] if isinstance(stops, int) else list(stops))
    return tokenizer


def _trace_seq_lens(num_devices: int, max_prefill_chunk_size: int, max_seq_len: int) -> tuple[int, ...]:
    # Llama-3.2-3B has no traced-prefill bucket on N150; decode tracing remains supported.
    allowed = {1: (), 2: (128, 1024), 8: (128, 1024)}.get(num_devices)
    if allowed is None:
        raise ValueError(f"Llama-3.2-3B supports 1, 2, or 8 devices, got {num_devices}")
    return tuple(length for length in allowed if length <= min(max_prefill_chunk_size, max_seq_len))


def _cache_path(
    hf_model: str,
    mesh_device,
    cache_dir: Path | str | None,
    *,
    hf_revision: str | None = None,
    dtype=None,
) -> Path:
    topology = {1: "N150", 2: "N300", 8: "T3K"}[mesh_device.get_num_devices()]
    resolution = resolve_model_cache(
        hf_model_id=hf_model,
        hf_revision=hf_revision,
        topology=topology,
        cache_dir=cache_dir,
        mesh_device=mesh_device,
        dtype=dtype,
    )
    path = resolution.path
    if dtype is not None:
        report_model_preflight(
            hf_model_id=hf_model,
            hf_revision=hf_revision,
            cache_path=path,
            topology=topology,
            mesh_device=mesh_device,
            dtype=dtype,
            cache_dir=cache_dir,
            cache_resolution=resolution,
        )
    return path


def convert_hf_model_weights(
    hf,
    hf_config,
    *,
    n_layers: int,
    num_devices: int,
    rope_table_len: int,
    head_dim: int,
) -> Llama32_3BWeights:
    """Extract and convert all Hugging Face tensors consumed by the TT builder."""
    base = hf.model
    rope_cos, rope_sin = weight_utils.build_rope_cos_sin_torch(
        base.rotary_emb, rope_table_len, head_dim, torch.bfloat16
    )
    layers = []
    for layer in base.layers[:n_layers]:
        wqkv, wo = weight_utils.attention_wqkv_wo_from_hf_layer(layer.self_attn, num_devices)
        w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
        layers.append(
            Llama32_3BLayerWeights(
                wqkv=wqkv,
                wo=wo,
                w1=w1,
                w2=w2,
                w3=w3,
                attention_norm=weight_utils.rms_weight_torch(layer.input_layernorm).to(torch.bfloat16),
                ff_norm=weight_utils.rms_weight_torch(layer.post_attention_layernorm).to(torch.bfloat16),
            )
        )

    lm_head_source = base.embed_tokens.weight if hf_config.tie_word_embeddings else hf.lm_head.weight
    return Llama32_3BWeights(
        embedding=weight_utils.embed_tokens_torch(base.embed_tokens),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        layers=tuple(layers),
        final_norm=weight_utils.rms_weight_torch(base.norm).to(torch.bfloat16),
        lm_head=lm_head_source.detach().to(torch.bfloat16).clone(),
    )


def _load_model(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = None,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Llama32_3BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Llama32_3BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> Llama32_3BForCausalLM:
    cache_dtype = dtype
    hf_revision = _resolve_hf_revision(hf_model, hf_revision)
    hf_config = AutoConfig.from_pretrained(
        hf_model,
        revision=hf_revision,
        local_files_only=offline_mode(),
    )
    hf = AutoModelForCausalLM.from_pretrained(
        hf_model,
        revision=hf_revision,
        torch_dtype=torch.bfloat16,
        local_files_only=offline_mode(),
    )
    hf.eval()
    resolved_layers = hf_config.num_hidden_layers if n_layers is None else n_layers
    precision = (
        optimizations
        if isinstance(optimizations, Llama32_3BPrecisionConfig)
        else (LLAMA32_3B_PERFORMANCE if optimizations == "performance" else LLAMA32_3B_ACCURACY)
    )
    if not isinstance(precision, Llama32_3BPrecisionConfig):
        raise TypeError("optimizations must be 'accuracy', 'performance', or Llama32_3BPrecisionConfig")
    cache_path = _cache_path(hf_model, mesh_device, cache_dir, hf_revision=hf_revision, dtype=cache_dtype)
    if paged_attention_config is None:
        block_size = 32
        paged_attention_config = Llama32_3BPagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=((max_seq_len + block_size - 1) // block_size) * max_batch_size,
        )
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    params = Llama32_3BModelParameters(
        dim=hf_config.hidden_size,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        hidden_dim=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    rope_table_len = ((max(max_seq_len * 2, 8192) + 127) // 128) * 128
    weights = convert_hf_model_weights(
        hf,
        hf_config,
        n_layers=resolved_layers,
        num_devices=mesh_device.get_num_devices(),
        rope_table_len=rope_table_len,
        head_dim=head_dim,
    )
    model_config = build_llama32_3b_transformer_1d_config(
        mesh_device=mesh_device,
        params=params,
        weights=weights,
        n_layers=resolved_layers,
        precision=precision,
        cache_path=cache_path,
        paged_attention_config=paged_attention_config,
    )
    tokenizer = load_tokenizer(hf_model, hf_revision)
    model = Llama32_3BTransformer1D(model_config)
    try:
        max_prefill_chunk_size = 2048
        runtime_config = Llama32_3BRuntimeConfig(
            model_name=Path(hf_model).name,
            model_cache_path=cache_path,
            max_prefill_chunk_size=max_prefill_chunk_size,
            max_context_len=int(hf_config.max_position_embeddings),
            max_seq_len=max_seq_len,
            trace_prefill_supported_seq_lens=_trace_seq_lens(
                mesh_device.get_num_devices(), max_prefill_chunk_size, max_seq_len
            ),
            disable_batched_prefill=environment_flag("DISABLE_BATCHED_PREFILL"),
        )
        del hf
        llm = Llama32_3BForCausalLM(
            model=model,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
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


def build_llama32_3b_executor(llm: Llama32_3BForCausalLM, config: Llama32_3BExecutorConfig) -> Llama32_3BExecutor:
    """Build one executor around an already-loaded Llama 3.2-3B adapter."""

    return Llama32_3BExecutor(llm.model, llm.runtime_config, config)


def from_pretrained(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = None,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Llama32_3BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Llama32_3BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
    executor_config: Llama32_3BExecutorConfig | None = None,
) -> Llama32_3BForCausalLM:
    """Load a text model with one executor ready for ``generate()``.

    Inputs and generated IDs are CPU PyTorch tensors. The supplied mesh remains
    caller-owned. By default execution is eager with host token selection; pass
    the existing ``Llama32_3BExecutorConfig`` for explicit trace/sampling policy.
    """
    if executor_config is not None and not isinstance(executor_config, Llama32_3BExecutorConfig):
        raise TypeError("executor_config must be a Llama32_3BExecutorConfig")
    for name, value in (("max_batch_size", max_batch_size), ("max_seq_len", max_seq_len)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if paged_attention_config is None:
        block_size = executor_config.paged_kv_cache.block_size if executor_config is not None else 32
        max_num_blocks = (
            executor_config.paged_kv_cache.max_num_blocks
            if executor_config is not None
            else ((max_seq_len + block_size - 1) // block_size) * max_batch_size
        )
        paged_attention_config = Llama32_3BPagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
    if executor_config is not None:
        kv_config = executor_config.paged_kv_cache
        if (kv_config.block_size, kv_config.max_num_blocks) != (
            paged_attention_config.block_size,
            paged_attention_config.max_num_blocks,
        ):
            raise ValueError("paged_attention_config must match executor_config.paged_kv_cache geometry")
        if kv_config.num_blocks is None:
            executor_config = replace(
                executor_config,
                paged_kv_cache=replace(kv_config, num_blocks=kv_config.max_num_blocks),
            )
    hf_revision = _resolve_hf_revision(hf_model, hf_revision)
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
            executor_config = Llama32_3BExecutorConfig(
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
        llm.executor = build_llama32_3b_executor(llm, executor_config)
        try:
            llm._hf_generation_config = GenerationConfig.from_pretrained(
                hf_model,
                revision=hf_revision,
                local_files_only=offline_mode(),
            )
        except OSError:
            # Reuse the config already loaded with the weights. Older checkpoints
            # can store generation defaults (including EOS) in config.json.
            hf_config = getattr(llm, "_hf_model_config", None)
            llm._hf_generation_config = GenerationConfig.from_model_config(hf_config) if hf_config is not None else None
        return llm
    except BaseException as primary:
        try:
            llm.cleanup()
        except BaseException as error:
            attach_cleanup_failures(primary, (error,))
        raise
