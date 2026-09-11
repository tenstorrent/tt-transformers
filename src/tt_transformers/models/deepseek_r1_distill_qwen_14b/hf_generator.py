# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face generation and loading for DeepSeek-R1-Distill-Qwen-14B."""

from __future__ import annotations

import math
from collections.abc import Sequence
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
from tt_transformers.llm_runtime.config import PagedKVCacheConfig, PageTableLayout, TraceConfig, WarmupConfig
from tt_transformers.llm_runtime.decode import DecodeRuntime, DecodeRuntimeConfig
from tt_transformers.llm_runtime.execution import EagerExecutor, TracedExecutor
from tt_transformers.llm_runtime.output_reader import OutputReader
from tt_transformers.llm_runtime.paged_kv_cache import PagedKVCacheManager
from tt_transformers.llm_runtime.prefill.config import PrefillRuntimeConfig
from tt_transformers.llm_runtime.prefill.runtime import PrefillRuntime
from tt_transformers.llm_runtime.program_compiler import ProgramCompiler
from tt_transformers.llm_runtime.tensor_resources import attach_cleanup_failures
from tt_transformers.llm_runtime.trace_compiler import TraceCompiler
from tt_transformers.llm_runtime.warmup import WarmupCoordinator, WarmupCoordinatorConfig
from tt_transformers.models.deepseek_r1_distill_qwen_14b import weight_utils
from tt_transformers.models.deepseek_r1_distill_qwen_14b.model import (
    DEEPSEEK_R1_14B_ACCURACY,
    DEEPSEEK_R1_14B_PERFORMANCE,
    DeepSeekR1Qwen14B,
    DeepSeekR1Qwen14BLayerWeights,
    DeepSeekR1Qwen14BModelParameters,
    DeepSeekR1Qwen14BPagedAttentionConfig,
    DeepSeekR1Qwen14BPrecisionConfig,
    DeepSeekR1Qwen14BWeights,
    build_deepseek_r1_distill_qwen_14b_transformer_config,
)
from tt_transformers.modules.sampling.sampling_1d import Sampling1D
from tt_transformers.sampling.sampling_params import SamplingParams

DEFAULT_HF_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEFAULT_HF_REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"


@dataclass(frozen=True)
class DeepSeekR1Qwen14BGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class DeepSeekR1Qwen14BRuntimeConfig:
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
class DeepSeekR1Qwen14BForCausalLM:
    model: DeepSeekR1Qwen14B
    tokenizer: Any
    runtime_config: DeepSeekR1Qwen14BRuntimeConfig
    instruct: bool = True
    generation_config: DeepSeekR1Qwen14BGenerationConfig = field(default_factory=DeepSeekR1Qwen14BGenerationConfig)
    executor: object | None = field(default=None, repr=False)

    def __post_init__(self):
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            stops = tuple(getattr(self.tokenizer, "stop_tokens", []) or [])
            self.generation_config = DeepSeekR1Qwen14BGenerationConfig(
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
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        revision=hf_revision,
        local_files_only=offline_mode(),
    )
    eos = getattr(tokenizer, "eos_token_id", None)
    stops = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    tokenizer.stop_tokens = stops
    return tokenizer


def _trace_seq_lens(num_devices: int, max_prefill_chunk_size: int, max_seq_len: int) -> tuple[int, ...]:
    if num_devices not in (2, 4, 8):
        raise ValueError(f"DeepSeek-R1-Distill-Qwen-14B supports logical TP2/TP4/TP8, got TP{num_devices}")
    allowed = (128,) if num_devices == 4 else (128, 1024)
    return tuple(length for length in allowed if length <= min(max_prefill_chunk_size, max_seq_len))


def _cache_path(
    hf_model: str,
    mesh_device,
    cache_dir: Path | str | None,
    *,
    hf_revision: str | None = None,
    dtype=None,
) -> Path:
    topology = {2: "N300", 4: "N150x4", 8: "T3K"}[mesh_device.get_num_devices()]
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


def _validate_checkpoint_config(hf_config) -> None:
    expected = {
        "num_hidden_layers": 48,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 13824,
        "vocab_size": 152064,
    }
    actual = {
        name: (
            getattr(hf_config, name, None)
            if name != "head_dim"
            else (getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads)
        )
        for name in expected
    }
    mismatches = {name: (actual[name], value) for name, value in expected.items() if actual[name] != value}
    if mismatches:
        raise ValueError(f"Unexpected DeepSeek-R1-Distill-Qwen-14B geometry: {mismatches}")
    if bool(getattr(hf_config, "tie_word_embeddings", False)):
        raise ValueError("DeepSeek-R1-Distill-Qwen-14B requires an untied LM head")


def convert_hf_model_weights(
    hf,
    hf_config,
    *,
    n_layers: int,
    num_devices: int,
    rope_table_len: int,
    head_dim: int,
) -> DeepSeekR1Qwen14BWeights:
    """Extract and convert all Hugging Face tensors consumed by the TT builder."""
    base = hf.model
    rope_cos, rope_sin = weight_utils.build_rope_cos_sin_torch(
        base.rotary_emb, rope_table_len, head_dim, torch.bfloat16
    )
    layers = []
    for layer in base.layers[:n_layers]:
        wqkv, wo, q_norm, k_norm, wqkv_bias = weight_utils.attention_wqkv_wo_from_hf_layer(layer.self_attn, num_devices)
        if q_norm is not None or k_norm is not None:
            raise ValueError("DeepSeek-R1-Distill-Qwen-14B does not support QK norm weights")
        if wqkv_bias is None:
            raise ValueError("DeepSeek-R1-Distill-Qwen-14B requires QKV projection bias")
        w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
        ff_align = 32 * 32 * num_devices
        ff_padded = math.ceil(w1.shape[-1] / ff_align) * ff_align
        if ff_padded != w1.shape[-1]:
            w1 = torch.nn.functional.pad(w1, (0, ff_padded - w1.shape[-1]))
            w3 = torch.nn.functional.pad(w3, (0, ff_padded - w3.shape[-1]))
            w2 = torch.nn.functional.pad(w2, (0, 0, 0, ff_padded - w2.shape[-2]))
        layers.append(
            DeepSeekR1Qwen14BLayerWeights(
                wqkv=wqkv,
                wo=wo,
                wqkv_bias=wqkv_bias,
                w1=w1,
                w2=w2,
                w3=w3,
                attention_norm=weight_utils.rms_weight_torch(layer.input_layernorm).to(torch.bfloat16),
                ff_norm=weight_utils.rms_weight_torch(layer.post_attention_layernorm).to(torch.bfloat16),
            )
        )

    if hf_config.tie_word_embeddings:
        raise ValueError("DeepSeek-R1-Distill-Qwen-14B requires an untied LM head")
    return DeepSeekR1Qwen14BWeights(
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
    hf_revision: str | None = DEFAULT_HF_REVISION,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | DeepSeekR1Qwen14BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: DeepSeekR1Qwen14BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> DeepSeekR1Qwen14BForCausalLM:
    cache_dtype = dtype
    if mesh_device.get_num_devices() not in (2, 4, 8):
        raise ValueError(
            f"DeepSeek-R1-Distill-Qwen-14B supports logical TP2/TP4/TP8, got TP{mesh_device.get_num_devices()}"
        )
    hf_config = AutoConfig.from_pretrained(
        hf_model,
        revision=hf_revision,
        local_files_only=offline_mode(),
    )
    _validate_checkpoint_config(hf_config)
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
        if isinstance(optimizations, DeepSeekR1Qwen14BPrecisionConfig)
        else (DEEPSEEK_R1_14B_PERFORMANCE if optimizations == "performance" else DEEPSEEK_R1_14B_ACCURACY)
    )
    if not isinstance(precision, DeepSeekR1Qwen14BPrecisionConfig):
        raise TypeError("optimizations must be 'accuracy', 'performance', or DeepSeekR1Qwen14BPrecisionConfig")
    cache_path = _cache_path(hf_model, mesh_device, cache_dir, hf_revision=hf_revision, dtype=cache_dtype)
    if paged_attention_config is None:
        block_size = 32
        paged_attention_config = DeepSeekR1Qwen14BPagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=((max_seq_len + block_size - 1) // block_size) * max_batch_size,
        )
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    params = DeepSeekR1Qwen14BModelParameters(
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
    model_config = build_deepseek_r1_distill_qwen_14b_transformer_config(
        mesh_device=mesh_device,
        params=params,
        weights=weights,
        n_layers=resolved_layers,
        precision=precision,
        cache_path=cache_path,
        paged_attention_config=paged_attention_config,
    )
    tokenizer = load_tokenizer(hf_model, hf_revision)
    model = DeepSeekR1Qwen14B(model_config)
    try:
        max_prefill_chunk_size = 2048
        runtime_config = DeepSeekR1Qwen14BRuntimeConfig(
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
        llm = DeepSeekR1Qwen14BForCausalLM(
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


@dataclass(frozen=True)
class DeepSeekR1Qwen14BExecutorConfig:
    """Immutable aggregate policy paired with one model-owned executor."""

    trace: TraceConfig
    warmup: WarmupConfig
    paged_kv_cache: PagedKVCacheConfig
    device_sampling_enabled: bool

    def __post_init__(self) -> None:
        nested_configs = (
            ("trace", self.trace, TraceConfig),
            ("warmup", self.warmup, WarmupConfig),
            ("paged_kv_cache", self.paged_kv_cache, PagedKVCacheConfig),
        )
        for name, value, expected_type in nested_configs:
            if type(value) is not expected_type:
                raise TypeError(f"{name} must be exactly {expected_type.__name__}")
        if not isinstance(self.device_sampling_enabled, bool):
            raise TypeError("device_sampling_enabled must be bool")


class DeepSeekR1Qwen14BExecutor:
    """Compose every runtime owner for one DeepSeek-R1-Distill-Qwen-14B execution lane.

    Construction wires one `PrefillRuntime`, `DecodeRuntime`,
    `ProgramCompiler`, and `EagerExecutor`. Trace-enabled
    configurations add one `TraceCompiler` and one
    `TracedExecutor` over that exact eager instance.

    ``DeepSeekR1Qwen14BGenerator`` or ``LaneGroupExecutor`` first configures and allocates
    KV cache, then warms or compiles programs, and finally calls
    `prefill_forward` and `decode_forward` with an explicit
    execution target. `cleanup` is the deterministic resource-release
    root and makes the executor terminal.
    """

    requires_prefill_trace_warmup = True

    def __init__(self, model: Any, runtime_config: Any, config: DeepSeekR1Qwen14BExecutorConfig) -> None:
        if not isinstance(config, DeepSeekR1Qwen14BExecutorConfig):
            raise TypeError("config must be a DeepSeekR1Qwen14BExecutorConfig")
        iter_modules = getattr(model, "iter_executor_named_modules", None)
        if not callable(iter_modules):
            raise TypeError("model must provide iter_executor_named_modules()")
        can_enable_trace = getattr(runtime_config, "can_enable_trace", None)
        if not callable(can_enable_trace):
            raise TypeError("runtime_config must provide can_enable_trace()")
        model_config = getattr(model, "config", None)
        mesh_device = getattr(model_config, "mesh_device", None)
        if mesh_device is None:
            raise ValueError("model.config.mesh_device is required")

        self.model = model
        self.runtime_config = runtime_config
        self.model_args = runtime_config
        self.config = config
        self.mesh_device = mesh_device
        self.cache_path = getattr(runtime_config, "model_cache_path", None)
        self._terminal = False
        self._cleaned_up = False
        self._sampling_buffers_loaded = False
        self._runtime_configuration_sealed = False
        self._q128_topk_tile_ends_warmed: set[bool] = set()

        sampling = getattr(model, "sampling", None)
        if config.device_sampling_enabled:
            if not isinstance(sampling, Sampling1D):
                raise TypeError("device sampling requires model.sampling to be a Sampling1D")
            is_resolved = getattr(getattr(sampling, "config", None), "is_resolved", None)
            if not callable(is_resolved) or not is_resolved():
                raise ValueError("model.sampling must have a resolved Sampling1DConfig")

        self.kv_cache_manager = PagedKVCacheManager(model, config.paged_kv_cache)
        self.page_table_layout = self._resolve_page_table_layout()
        self.output_reader = OutputReader(mesh_device)
        self.prefill_runtime = PrefillRuntime(
            PrefillRuntimeConfig.resolve(
                model=model,
                output_reader=self.output_reader,
                page_table_layout=self.page_table_layout,
                max_batch_size=int(model.config.max_batch_size),
                max_prefill_chunk_size=int(runtime_config.max_prefill_chunk_size),
                device_sampling_enabled=config.device_sampling_enabled,
                can_enable_trace=runtime_config.can_enable_trace,
                supports_batched_prefill=bool(runtime_config.supports_batched_prefill),
                disable_batched_prefill=bool(runtime_config.disable_batched_prefill),
                max_prefill_batch_size=int(runtime_config.max_prefill_batch_size),
                batched_prefill_batched_extract=bool(runtime_config.batched_prefill_batched_extract),
            )
        )
        self.decode_runtime = DecodeRuntime(
            DecodeRuntimeConfig.resolve(
                model=model,
                output_reader=self.output_reader,
                lane_capacity=int(model.config.max_batch_size),
                page_table_layout=self.page_table_layout,
                device_sampling_enabled=config.device_sampling_enabled,
                force_greedy_top_k=config.warmup.include_decode_top_k,
            )
        )
        self.program_compiler = ProgramCompiler(mesh_device, lambda: self.kv_cache_manager.bound_context)
        self.eager_executor = EagerExecutor(
            prefill=self.prefill_runtime,
            decode=self.decode_runtime,
            program_compiler=self.program_compiler,
        )
        self.trace_compiler: TraceCompiler | None = None
        self.traced_executor: TracedExecutor | None = None
        if config.trace.mode != "none":
            self.trace_compiler = TraceCompiler(self.program_compiler)
            self.traced_executor = TracedExecutor(
                eager=self.eager_executor,
                trace_compiler=self.trace_compiler,
                trace_mode=config.trace.mode,
            )
        self.eager_execution = self.eager_executor
        self.traced_prefill_execution = (
            self.traced_executor if config.trace.prefill_enabled and self.traced_executor is not None else None
        )
        self.traced_decode_execution = (
            self.traced_executor if config.trace.decode_enabled and self.traced_executor is not None else None
        )
        self._prefill_execution = self.traced_prefill_execution or self.eager_executor
        self._decode_execution = self.traced_decode_execution or self.eager_executor

        prefill_sequence_lengths = getattr(runtime_config, "trace_prefill_supported_seq_lens", (128,))
        self.warmup = WarmupCoordinator(
            config=WarmupCoordinatorConfig.resolve(
                warmup=config.warmup,
                trace=config.trace,
                prefill=self.prefill_runtime.config,
                decode=self.decode_runtime.config,
                prefill_sequence_lengths=prefill_sequence_lengths,
            ),
            execution=self.traced_executor or self.eager_executor,
            ensure_sampling_buffers=self._ensure_sampling_buffers,
            validate_bound_cache=self._validate_bound_cache,
        )

    # Public model execution API

    @property
    def model_config(self):
        return self.model.config

    @property
    def cluster_shape(self) -> list[int]:
        return list(self.mesh_device.shape)

    @property
    def paged_kv_cache_config(self) -> PagedKVCacheConfig:
        return self.kv_cache_manager.config

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def already_warmed_up_prefill(self) -> bool:
        return self.warmup.already_warmed_up_prefill

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("already_warmed_up_prefill compatibility value must be bool")
        # Compatibility writes are intentionally non-authoritative. Warmup
        # coverage is derived from the coordinator's completed case ledger.
        return None

    def configure_paged_kv_cache(self, config: PagedKVCacheConfig) -> None:
        """Resolve vLLM-owned KV geometry before the first allocation."""

        self._ensure_active()
        if self._runtime_configuration_sealed:
            raise RuntimeError("runtime configuration is sealed")
        if not isinstance(config, PagedKVCacheConfig):
            raise TypeError("config must be a PagedKVCacheConfig")
        if self.kv_cache_manager.config.is_resolved():
            raise RuntimeError("paged KV cache configuration is already resolved")
        if config.dtype != self.kv_cache_manager.config.dtype:
            raise ValueError("resolved paged KV cache cannot change dtype")
        if config.memory_config != self.kv_cache_manager.config.memory_config:
            raise ValueError("resolved paged KV cache cannot change memory_config")
        self.model.configure_paged_attention(
            block_size=config.block_size,
            max_num_blocks=config.max_num_blocks,
        )
        self.kv_cache_manager.configure(config)
        self.config = replace(self.config, paged_kv_cache=config)
        self._refresh_page_table_layout()

    def allocate_kv_cache(
        self,
        kv_cache_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        num_layers: int | None = None,
    ) -> list[list[Any]]:
        """Allocate and bind the model-owned paged KV cache."""

        self._ensure_active()
        supplied = (kv_cache_shape is not None, dtype is not None, num_layers is not None)
        if any(supplied):
            if not all(supplied):
                raise TypeError("kv_cache_shape, dtype, and num_layers must be supplied together")
            shape = tuple(int(dimension) for dimension in kv_cache_shape)
            if len(shape) != 4:
                raise ValueError(f"KV cache shape must have rank 4, got {shape}")
            expected_layers = len(self.kv_cache_manager.per_layer_dtypes)
            if int(num_layers) != expected_layers:
                raise ValueError(f"vLLM KV layer count {num_layers} does not match model layer count {expected_layers}")
            self.kv_cache_manager.validate_vllm_cache_spec(
                block_size=shape[2],
                dtype=dtype,
                num_blocks=shape[0],
            )
            if self.kv_cache_manager.config.num_blocks is None:
                self.kv_cache_manager.configure(replace(self.kv_cache_manager.config, num_blocks=shape[0]))
                self._refresh_page_table_layout()
            elif self.kv_cache_manager.config.num_blocks != shape[0]:
                raise ValueError(
                    f"Paged KV cache is resolved to {self.kv_cache_manager.config.num_blocks} blocks, not {shape[0]}"
                )
            if any(tuple(expected) != shape for expected in self.kv_cache_manager.cache_shapes):
                raise ValueError(
                    f"vLLM KV shape {shape} does not match model-derived shapes {self.kv_cache_manager.cache_shapes}"
                )
        if not self.kv_cache_manager.config.is_resolved():
            raise RuntimeError("Paged KV cache capacity must be resolved before allocation")
        self._seal_runtime_configuration()
        return self.kv_cache_manager.allocate()

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> None:
        """Compile prefill on the supplied eager or traced execution target."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return (execution or self._prefill_execution).compile_prefill(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
        )

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> None:
        """Compile decode on the supplied eager or traced execution target."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return (execution or self._decode_execution).compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> Any:
        """Validate ownership and execute one prefill call."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return (execution or self._prefill_execution).prefill_forward(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
        )

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
        read_from_device: bool = True,  # ↓ Output policy
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> Any:
        """Validate ownership and execute one decode call."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        return (execution or self._decode_execution).decode_forward(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            read_from_device=read_from_device,
        )

    def can_trace_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
    ) -> bool:
        """Classify whether the prefill request can use this lane's trace."""

        if self.traced_executor is None or not self.config.trace.prefill_enabled:
            return False
        return self.prefill_runtime.can_trace(
            tokens=tokens,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
        )

    def read_decode_output(self, tt_out: Any, *, async_read: bool = False) -> Any:
        self._ensure_active()
        return self.decode_runtime.read_decode_output(tt_out=tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out: Any, *, is_tokens: bool = False) -> tuple[Any, Any]:
        self._ensure_active()
        return self.decode_runtime.process_decode_output_host(tt_out=tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        self._ensure_active()
        if enable_trace:
            self._warmup_q128_topk_tile_ends(
                kv_cache=kv_cache,
                can_sample_on_device=can_sample_on_device,
                enable_trace=True,
            )
        self.warmup.warmup_prefill(
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )
        if not enable_trace:
            self._warmup_q128_topk_tile_ends(
                kv_cache=kv_cache,
                can_sample_on_device=can_sample_on_device,
                enable_trace=False,
            )

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        max_batch_size: int,  # ↓ Coverage dimensions
        num_blocks: int,
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        self._ensure_active()
        return self.warmup.warmup_decode(
            kv_cache=kv_cache,
            max_batch_size=max_batch_size,
            num_blocks=num_blocks,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )

    def cleanup(self) -> None:
        """Release runtime, trace, program, sampling, and KV resources in order."""

        self._terminal = True
        if self._cleaned_up:
            return

        failures = []
        actions = [
            self.decode_runtime.drain_external_outputs,
            self.output_reader.drain,
            self.prefill_runtime.cleanup,
            self.decode_runtime.cleanup_transients,
        ]
        if self.trace_compiler is not None:
            actions.append(self.trace_compiler.cleanup)
        actions.append(self.program_compiler.cleanup)
        if self.config.device_sampling_enabled:
            actions.append(self.model.sampling.release)
        actions.append(self.kv_cache_manager.release)

        for action in actions:
            try:
                action()
            except BaseException as error:
                failures.append(error)
        if failures:
            _raise_cleanup_failures(failures, "DeepSeekR1Qwen14BExecutor")
        self._cleaned_up = True

    # Private implementation

    def _warmup_q128_topk_tile_ends(
        self,
        *,
        kv_cache: Any,
        can_sample_on_device: bool,
        enable_trace: bool,
    ) -> None:
        """Prime DeepSeek Q128 tile ends in eager and traced execution."""

        if (
            enable_trace in self._q128_topk_tile_ends_warmed
            or not can_sample_on_device
            or (enable_trace and self.traced_executor is None)
            or 128 not in self.warmup.config.prefill_sequence_lengths
            or not self.prefill_runtime.config.static_q128_topk_supported
            or self.warmup.config.prime_q128_tile_ends
        ):
            return
        sampling = SamplingParams(
            temperature=torch.ones(1),
            top_k=torch.full((1,), 32, dtype=torch.int32),
            top_p=torch.full((1,), 0.08),
        )
        execution = self.traced_executor if enable_trace else self.eager_executor
        for sequence_length in (32, 64, 96):
            page_table_width = (
                sequence_length + self.page_table_layout.block_size - 1
            ) // self.page_table_layout.block_size
            self.compile_prefill(
                tokens=torch.zeros((1, sequence_length), dtype=torch.long),
                page_table=torch.zeros((1, page_table_width), dtype=torch.int32),
                prompt_lens=torch.full((1,), sequence_length, dtype=torch.long),
                empty_slots=[0],
                kv_cache=kv_cache,
                sampling_params=sampling,
                execution=execution,
            )
        self._q128_topk_tile_ends_warmed.add(enable_trace)

    def _resolve_page_table_layout(self) -> PageTableLayout:
        kv_config = self.kv_cache_manager.config
        # The direct demo resolves num_blocks=max_num_blocks before constructing
        # this executor, so its geometry is final immediately and it physically
        # allocates that maximum. vLLM constructs against max_num_blocks only as
        # a cheap capacity ceiling: no KV tensor is allocated until vLLM supplies
        # num_blocks, _refresh_page_table_layout installs the final geometry, and
        # allocate_kv_cache seals all runtime configs.
        physical_num_blocks = kv_config.num_blocks or kv_config.max_num_blocks
        return PageTableLayout.resolve(
            block_size=int(kv_config.block_size),
            model_max_sequence_length=int(self.model.config.max_seq_len),
            physical_num_blocks=int(physical_num_blocks),
            max_prefill_chunk_size=min(
                int(self.runtime_config.max_prefill_chunk_size),
                int(self.model.config.max_seq_len),
            ),
        )

    def _refresh_page_table_layout(self) -> None:
        # vLLM reaches this boundary after choosing the physical block count and
        # before PagedKVCacheManager.allocate(), compilation, warmup, or tracing.
        # Re-resolve complete immutable configs so their normal construction
        # checks and capacity ceilings describe vLLM's authoritative geometry.
        # Runtime and resource-owner identities remain unchanged.
        layout = self._resolve_page_table_layout()
        current_prefill = self.prefill_runtime.config
        prefill_config = PrefillRuntimeConfig.resolve(
            model=self.model,
            output_reader=self.output_reader,
            page_table_layout=layout,
            max_batch_size=current_prefill.max_batch_size,
            max_prefill_chunk_size=current_prefill.max_prefill_chunk_size,
            device_sampling_enabled=current_prefill.device_sampling_enabled,
            can_enable_trace=current_prefill.can_enable_trace,
            supports_batched_prefill=current_prefill.supports_batched_prefill,
            disable_batched_prefill=current_prefill.disable_batched_prefill,
            max_prefill_batch_size=current_prefill.max_prefill_batch_size,
            batched_prefill_batched_extract=current_prefill.batched_prefill_batched_extract,
        )
        current_decode = self.decode_runtime.config
        decode_config = DecodeRuntimeConfig.resolve(
            model=self.model,
            output_reader=self.output_reader,
            lane_capacity=current_decode.lane_capacity,
            page_table_layout=layout,
            device_sampling_enabled=current_decode.device_sampling_enabled,
            force_greedy_top_k=current_decode.force_greedy_top_k,
        )
        warmup_config = WarmupCoordinatorConfig.resolve(
            warmup=self.warmup.config.warmup,
            trace=self.config.trace,
            prefill=prefill_config,
            decode=decode_config,
            prefill_sequence_lengths=self.warmup.config.prefill_sequence_lengths,
        )

        self.prefill_runtime.config = prefill_config
        self.decode_runtime.config = decode_config
        self.warmup.config = warmup_config
        self.page_table_layout = layout

    def _seal_runtime_configuration(self) -> None:
        """Seal final geometry immediately before physical KV allocation."""

        self.warmup.seal_configuration()
        self._runtime_configuration_sealed = True

    def _ensure_sampling_for(self, sampling_params: Any) -> None:
        if sampling_params is None:
            return
        if not self.config.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")
        self._ensure_sampling_buffers()

    def _ensure_sampling_buffers(self) -> None:
        if self._sampling_buffers_loaded:
            return
        if self.trace_compiler is not None and self.trace_compiler.trace_active:
            raise RuntimeError("cannot materialize sampling buffers after trace activation")
        self.model.sampling.load_device_buffers()
        self._sampling_buffers_loaded = True

    def _validate_bound_cache(self, kv_cache: Any) -> None:
        if self.kv_cache_manager.bound_context is None:
            raise RuntimeError("Paged KV cache must be allocated and bound before execution")
        if kv_cache is not None:
            self.kv_cache_manager.validate_borrowed_handle(kv_cache)

    def _ensure_active(self) -> None:
        if self._terminal:
            raise RuntimeError("DeepSeekR1Qwen14BExecutor is terminal; construct a new executor")
        if self.prefill_runtime.transient_orphan_count or self.decode_runtime.transient_orphan_count:
            raise RuntimeError("DeepSeekR1Qwen14BExecutor has unreleased transient resources; clean up this executor")


def build_deepseek_r1_distill_qwen_14b_executor(
    llm: DeepSeekR1Qwen14BForCausalLM, config: DeepSeekR1Qwen14BExecutorConfig
) -> DeepSeekR1Qwen14BExecutor:
    """Build one executor around an already-loaded Qwen model adapter."""

    return DeepSeekR1Qwen14BExecutor(llm.model, llm.runtime_config, config)


def _raise_cleanup_failures(failures: list[BaseException], owner: str) -> None:
    primary, *additional = failures
    attach_cleanup_failures(
        primary,
        additional,
        note=f"{owner} cleanup also encountered {{count}} failure(s)",
    )
    raise primary


def from_pretrained(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = DEFAULT_HF_REVISION,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | DeepSeekR1Qwen14BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: DeepSeekR1Qwen14BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
    executor_config: DeepSeekR1Qwen14BExecutorConfig | None = None,
) -> DeepSeekR1Qwen14BForCausalLM:
    """Load a text model with one executor ready for ``generate()``.

    Inputs and generated IDs are CPU PyTorch tensors. The supplied mesh remains
    caller-owned. By default execution is eager with host token selection; pass
    the existing ``DeepSeekR1Qwen14BExecutorConfig`` for explicit trace/sampling policy.
    """
    if executor_config is not None and not isinstance(executor_config, DeepSeekR1Qwen14BExecutorConfig):
        raise TypeError("executor_config must be a DeepSeekR1Qwen14BExecutorConfig")
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
        paged_attention_config = DeepSeekR1Qwen14BPagedAttentionConfig(
            block_size=block_size, max_num_blocks=max_num_blocks
        )
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
            executor_config = DeepSeekR1Qwen14BExecutorConfig(
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
        llm.executor = build_deepseek_r1_distill_qwen_14b_executor(llm, executor_config)
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
