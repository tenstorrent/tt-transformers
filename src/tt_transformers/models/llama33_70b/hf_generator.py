# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face generation and loading for the TTTv2 Llama 3.3 70B path."""

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
from tt_transformers.models.llama3_executor import Llama33_70BExecutor, Llama33_70BExecutorConfig
from tt_transformers.models.llama33_70b import weight_utils
from tt_transformers.models.llama33_70b.model import (
    LLAMA33_70B_ACCURACY,
    LLAMA33_70B_BH_TP4_CLUSTER_TYPES,
    LLAMA33_70B_PERFORMANCE,
    Llama33_70BLayerWeights,
    Llama33_70BModelParameters,
    Llama33_70BPagedAttentionConfig,
    Llama33_70BPrecisionConfig,
    Llama33_70BTransformer1D,
    Llama33_70BWeights,
    _llama33_70b_ccl_topology,
    build_llama33_70b_transformer_1d_config,
)

DEFAULT_HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_HF_REVISION = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"


def _resolve_hf_revision(hf_model: str, hf_revision: str | None) -> str | None:
    """Pin the default checkpoint without applying its revision to custom IDs."""

    if hf_revision is not None:
        return hf_revision
    return DEFAULT_HF_REVISION if hf_model == DEFAULT_HF_MODEL else None


@dataclass(frozen=True)
class Llama33_70BGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Llama33_70BRuntimeConfig:
    model_name: str
    model_cache_path: Path | None
    max_prefill_chunk_size: int
    max_context_len: int
    max_seq_len: int
    trace_prefill_supported_seq_lens: tuple[int, ...]
    trace_prefill_warmup_seq_lens: tuple[int, ...] = ()
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
class Llama33_70BForCausalLM:
    model: Llama33_70BTransformer1D
    tokenizer: Any
    runtime_config: Llama33_70BRuntimeConfig
    instruct: bool = True
    generation_config: Llama33_70BGenerationConfig = field(default_factory=Llama33_70BGenerationConfig)
    executor: object | None = field(default=None, repr=False)

    def __post_init__(self):
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            self.generation_config = Llama33_70BGenerationConfig(
                max_decode_tokens=self.generation_config.max_decode_tokens,
                temperature=self.generation_config.temperature,
                top_k=self.generation_config.top_k,
                top_p=self.generation_config.top_p,
                stop_token_ids=tuple(getattr(self.tokenizer, "stop_tokens", ()) or ()),
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
    eos = getattr(tokenizer, "eos_token_id", None)
    tokenizer.stop_tokens = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    return tokenizer


def _trace_seq_lens(num_devices: int, max_prefill_chunk_size: int, max_seq_len: int) -> tuple[int, ...]:
    if num_devices not in (4, 8):
        raise ValueError(f"Llama-3.3-70B supports T3K (8 devices) or P150x4 (4 devices), got {num_devices}")
    if num_devices == 4:
        return (128,) if max_seq_len >= 128 else ()
    return tuple(length for length in (128, max_prefill_chunk_size) if length <= max_seq_len)


def _trace_warmup_seq_lens(
    max_prefill_chunk_size: int,
    max_seq_len: int,
    supported_seq_lens: tuple[int, ...],
) -> tuple[int, ...]:
    """Return logical representatives whose first invocation has a trace family."""

    candidates = (128, max_prefill_chunk_size, 2 * max_prefill_chunk_size)
    return tuple(
        dict.fromkeys(
            length
            for length in candidates
            if length <= max_seq_len and min(length, max_prefill_chunk_size) in supported_seq_lens
        )
    )


def _resolve_supported_sku(*, arch, cluster_type, num_devices: int) -> str:
    if arch == ttnn.device.Arch.WORMHOLE_B0 and cluster_type == ttnn.cluster.ClusterType.T3K and num_devices == 8:
        return "T3K"
    if arch == ttnn.device.Arch.BLACKHOLE and cluster_type in LLAMA33_70B_BH_TP4_CLUSTER_TYPES and num_devices == 4:
        return "P150x4"
    raise ValueError(
        "Llama-3.3-70B supports physical Wormhole T3K (8 devices) or "
        "BlackHole P150_X4/P300_X2 as logical P150x4 (4 devices); "
        f"got arch={arch}, cluster_type={cluster_type}, num_devices={num_devices}"
    )


def _cache_path(
    hf_model: str,
    mesh_device,
    cache_dir: Path | str | None,
    *,
    sku: str,
    hf_revision: str | None = None,
    dtype=None,
) -> Path:
    resolution = resolve_model_cache(
        hf_model_id=hf_model,
        hf_revision=hf_revision,
        topology=sku,
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
            topology=sku,
            mesh_device=mesh_device,
            dtype=dtype,
            cache_dir=cache_dir,
            cache_resolution=resolution,
        )
    return path


def _stop_token_ids(hf) -> tuple[int, ...]:
    eos = getattr(getattr(hf, "generation_config", None), "eos_token_id", None)
    if eos is None:
        return ()
    values = (eos,) if isinstance(eos, int) else tuple(eos)
    return tuple(dict.fromkeys(int(value) for value in values))


def convert_hf_model_weights(
    hf,
    hf_config,
    *,
    n_layers: int,
    num_devices: int,
    rope_table_len: int,
    head_dim: int,
) -> Llama33_70BWeights:
    """Extract and convert every HF tensor consumed by the TT graph builder."""

    base = hf.model
    rope_cos, rope_sin = weight_utils.build_rope_cos_sin_torch(
        base.rotary_emb, rope_table_len, head_dim, torch.bfloat16
    )
    layers = []
    for layer in base.layers[:n_layers]:
        wqkv, wo = weight_utils.attention_wqkv_wo_from_hf_layer(layer.self_attn, num_devices)
        w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
        layers.append(
            Llama33_70BLayerWeights(
                wqkv=wqkv,
                wo=wo,
                w1=w1,
                w2=w2,
                w3=w3,
                attention_norm=weight_utils.rms_weight_torch(layer.input_layernorm).to(torch.bfloat16),
                ff_norm=weight_utils.rms_weight_torch(layer.post_attention_layernorm).to(torch.bfloat16),
            )
        )
    return Llama33_70BWeights(
        embedding=weight_utils.embed_tokens_torch(base.embed_tokens),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        layers=tuple(layers),
        final_norm=weight_utils.rms_weight_torch(base.norm).to(torch.bfloat16),
        lm_head=hf.lm_head.weight.detach().to(torch.bfloat16).clone(),
    )


def _load_model(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = None,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Llama33_70BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Llama33_70BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> Llama33_70BForCausalLM:
    cache_dtype = dtype
    hf_revision = _resolve_hf_revision(hf_model, hf_revision)
    num_devices = mesh_device.get_num_devices()
    arch = mesh_device.arch()
    sku = _resolve_supported_sku(
        arch=arch,
        cluster_type=ttnn.cluster.get_cluster_type(),
        num_devices=num_devices,
    )
    _llama33_70b_ccl_topology(mesh_device)
    load_kwargs = {"revision": hf_revision, "local_files_only": offline_mode()}
    hf_config = AutoConfig.from_pretrained(hf_model, **load_kwargs)
    if hf_config.num_attention_heads % num_devices or hf_config.num_key_value_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({hf_config.num_attention_heads}/{hf_config.num_key_value_heads}) "
            f"must be divisible by device count ({num_devices})"
        )
    hf = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.bfloat16, **load_kwargs)
    hf.eval()
    resolved_layers = hf_config.num_hidden_layers if n_layers is None else n_layers
    precision = (
        optimizations
        if isinstance(optimizations, Llama33_70BPrecisionConfig)
        else (LLAMA33_70B_PERFORMANCE if optimizations == "performance" else LLAMA33_70B_ACCURACY)
    )
    if not isinstance(precision, Llama33_70BPrecisionConfig):
        raise TypeError("optimizations must be 'accuracy', 'performance', or Llama33_70BPrecisionConfig")
    cache_path = _cache_path(
        hf_model,
        mesh_device,
        cache_dir,
        sku=sku,
        hf_revision=hf_revision,
        dtype=cache_dtype,
    )
    if paged_attention_config is None:
        block_size = 32
        paged_attention_config = Llama33_70BPagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=((max_seq_len + block_size - 1) // block_size) * max_batch_size,
        )
    head_dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads))
    params = Llama33_70BModelParameters(
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
        num_devices=num_devices,
        rope_table_len=rope_table_len,
        head_dim=head_dim,
    )
    model_config = build_llama33_70b_transformer_1d_config(
        mesh_device=mesh_device,
        params=params,
        weights=weights,
        n_layers=resolved_layers,
        precision=precision,
        cache_path=cache_path,
        paged_attention_config=paged_attention_config,
    )
    tokenizer = load_tokenizer(hf_model, hf_revision)
    stop_token_ids = _stop_token_ids(hf)
    if stop_token_ids:
        tokenizer.stop_tokens = list(stop_token_ids)
    trace_prefill_supported_seq_lens = _trace_seq_lens(num_devices, 2048, max_seq_len)
    runtime_config = Llama33_70BRuntimeConfig(
        model_name=Path(hf_model).name,
        model_cache_path=cache_path,
        max_prefill_chunk_size=2048,
        max_context_len=int(hf_config.max_position_embeddings),
        max_seq_len=max_seq_len,
        trace_prefill_supported_seq_lens=trace_prefill_supported_seq_lens,
        trace_prefill_warmup_seq_lens=_trace_warmup_seq_lens(
            2048,
            max_seq_len,
            trace_prefill_supported_seq_lens,
        ),
        disable_batched_prefill=environment_flag("DISABLE_BATCHED_PREFILL"),
    )
    del hf
    model = Llama33_70BTransformer1D(model_config)
    try:
        llm = Llama33_70BForCausalLM(
            model=model,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            instruct=instruct,
            generation_config=Llama33_70BGenerationConfig(stop_token_ids=stop_token_ids),
        )
        llm._hf_model_config = hf_config
        return llm
    except BaseException as primary:
        try:
            cleanup_object_graph(model)
        except BaseException as error:
            attach_cleanup_failures(primary, (error,))
        raise


def build_llama33_70b_executor(llm: Llama33_70BForCausalLM, config: Llama33_70BExecutorConfig) -> Llama33_70BExecutor:
    """Build one executor around an already-loaded Llama 3.3-70B adapter."""

    return Llama33_70BExecutor(llm.model, llm.runtime_config, config)


def from_pretrained(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = None,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Llama33_70BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Llama33_70BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
    executor_config: Llama33_70BExecutorConfig | None = None,
) -> Llama33_70BForCausalLM:
    """Load a text model with one executor ready for ``generate()``.

    Inputs and generated IDs are CPU PyTorch tensors. The supplied mesh remains
    caller-owned. By default execution is eager with host token selection; pass
    the existing ``Llama33_70BExecutorConfig`` for explicit trace/sampling policy.
    """
    if executor_config is not None and not isinstance(executor_config, Llama33_70BExecutorConfig):
        raise TypeError("executor_config must be a Llama33_70BExecutorConfig")
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
        paged_attention_config = Llama33_70BPagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
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
            executor_config = Llama33_70BExecutorConfig(
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
        llm.executor = build_llama33_70b_executor(llm, executor_config)
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
