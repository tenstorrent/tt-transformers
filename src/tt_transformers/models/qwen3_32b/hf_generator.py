# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face generation and loading for Qwen3-32B."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any

import torch
import ttnn
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from tt_transformers.cache_environment import (
    environment_flag,
    offline_mode,
    report_model_preflight,
    resolve_model_cache,
)
from tt_transformers.device_utils import cleanup_object_graph
from tt_transformers.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from tt_transformers.llm_runtime.tensor_resources import attach_cleanup_failures
from tt_transformers.models.executor import ModelExecutor, ModelExecutorConfig
from tt_transformers.models.qwen3_32b.model import (
    DEFAULT_HF_REVISION,
    QWEN3_32B_ACCURACY,
    QWEN3_32B_BH_TP4_CLUSTER_TYPES,
    QWEN3_32B_PERFORMANCE,
    Qwen3_32B,
    Qwen3_32BPagedAttentionConfig,
    Qwen3_32BPrecisionConfig,
    _slice_last_token_tile,
)
from tt_transformers.modules.sampling.sampling_1d import Sampling1D
from tt_transformers.modules.sampling.sampling_state_1d import SamplingState1D
from tt_transformers.sampling.sampling_params import SamplingParams

DEFAULT_HF_MODEL = "Qwen/Qwen3-32B"


def _local_files_only() -> bool:
    return offline_mode()


@dataclass(frozen=True)
class Qwen3_32BGenerationConfig:
    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Qwen3_32BRuntimeConfig:
    model_name: str
    model_cache_path: Path | None
    max_prefill_chunk_size: int
    max_context_len: int
    max_seq_len: int
    trace_prefill_supported_seq_lens: tuple[int, ...]
    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_batch_size: int
    cluster_shape: list[int]
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    supports_batched_prefill: bool = True
    max_prefill_batch_size: int = 32
    disable_batched_prefill: bool = False
    batched_prefill_batched_extract: bool = True

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int = 0) -> bool:
        if num_cached_tokens != 0:
            return False
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
class Qwen3_32BForCausalLM:
    model: Qwen3_32B
    tokenizer: Any
    runtime_config: Qwen3_32BRuntimeConfig
    instruct: bool = True
    generation_config: Qwen3_32BGenerationConfig = field(default_factory=Qwen3_32BGenerationConfig)
    executor: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            self.generation_config = Qwen3_32BGenerationConfig(
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


def load_tokenizer(hf_model: str, hf_revision: str | None = DEFAULT_HF_REVISION):
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        revision=hf_revision,
        local_files_only=_local_files_only(),
    )
    tokenizer.stop_tokens = list(_qwen_stop_token_ids(tokenizer))
    return tokenizer


def _qwen_stop_token_ids(tokenizer) -> tuple[int, ...]:
    ids = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.extend([eos] if isinstance(eos, int) else list(eos))
    for token in ("<|im_end|>", "<|im_start|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    return tuple(dict.fromkeys(int(value) for value in ids))


def _trace_seq_lens(num_devices: int, max_prefill_chunk_size: int, max_seq_len: int) -> tuple[int, ...]:
    if num_devices not in (4, 8):
        raise ValueError(f"Qwen3-32B supports T3K (8 devices) or P150x4 (4 devices), got {num_devices}")
    candidates = (128, 1024)
    return tuple(length for length in candidates if length <= min(max_prefill_chunk_size, max_seq_len))


def _resolve_supported_sku(*, arch, cluster_type, num_devices: int) -> str:
    if arch == ttnn.device.Arch.WORMHOLE_B0 and cluster_type == ttnn.cluster.ClusterType.T3K and num_devices == 8:
        return "T3K"
    if arch == ttnn.device.Arch.BLACKHOLE and cluster_type in QWEN3_32B_BH_TP4_CLUSTER_TYPES and num_devices == 4:
        return "P150x4"
    raise ValueError(
        "Qwen3-32B supports physical Wormhole T3K (8 devices) or BlackHole P150_X4/P300_X2 (4 devices); "
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


def _validate_checkpoint_config(hf_config, *, num_devices: int) -> None:
    expected = {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 25600,
        "vocab_size": 151936,
        "head_dim": 128,
    }
    actual = {name: getattr(hf_config, name, None) for name in expected}
    mismatches = {name: (actual[name], value) for name, value in expected.items() if actual[name] != value}
    if mismatches:
        raise ValueError(f"Unexpected Qwen3-32B geometry: {mismatches}")
    if hf_config.num_attention_heads % num_devices or hf_config.num_key_value_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({hf_config.num_attention_heads}/{hf_config.num_key_value_heads}) "
            f"must be divisible by device count ({num_devices})"
        )
    if bool(getattr(hf_config, "attention_bias", False)):
        raise ValueError("Qwen3-32B requires bias-free QKV projections")
    if bool(getattr(hf_config, "tie_word_embeddings", False)):
        raise ValueError("Qwen3-32B requires an untied LM head")


def _load_model(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = DEFAULT_HF_REVISION,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Qwen3_32BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Qwen3_32BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
) -> Qwen3_32BForCausalLM:
    cache_dtype = dtype
    num_devices = mesh_device.get_num_devices()
    sku = _resolve_supported_sku(
        arch=mesh_device.arch(),
        cluster_type=ttnn.cluster.get_cluster_type(),
        num_devices=num_devices,
    )
    hf_config = AutoConfig.from_pretrained(
        hf_model,
        revision=hf_revision,
        local_files_only=_local_files_only(),
    )
    _validate_checkpoint_config(hf_config, num_devices=num_devices)
    resolved_layers = hf_config.num_hidden_layers if n_layers is None else n_layers
    precision = (
        optimizations
        if isinstance(optimizations, Qwen3_32BPrecisionConfig)
        else (QWEN3_32B_PERFORMANCE if optimizations == "performance" else QWEN3_32B_ACCURACY)
    )
    if not isinstance(precision, Qwen3_32BPrecisionConfig):
        raise TypeError("optimizations must be 'accuracy', 'performance', or Qwen3_32BPrecisionConfig")
    if not (1 <= resolved_layers <= hf_config.num_hidden_layers):
        raise ValueError(f"n_layers must be in [1, {hf_config.num_hidden_layers}], got {resolved_layers}")
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
    else:
        block_size = int(paged_attention_config.block_size)
    head_dim = int(getattr(hf_config, "head_dim", 0) or 0)
    if head_dim != 128:
        raise ValueError(f"Qwen3-32B must use explicit HF head_dim=128, got {head_dim}")
    model = Qwen3_32B.from_pretrained(
        mesh_device,
        hf_model,
        revision=hf_revision,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=resolved_layers,
        cache_dir=cache_path,
        precision=precision,
        block_size=block_size,
        executor_mode=True,
    )
    try:
        runtime = Qwen3_32BRuntimeConfig(
            model_name=hf_model,
            model_cache_path=cache_path,
            max_prefill_chunk_size=4096,
            max_context_len=int(getattr(hf_config, "max_position_embeddings", max_seq_len)),
            max_seq_len=max_seq_len,
            trace_prefill_supported_seq_lens=_trace_seq_lens(num_devices, 4096, max_seq_len),
            n_layers=resolved_layers,
            n_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            max_batch_size=max_batch_size,
            cluster_shape=list(mesh_device.shape),
            kv_cache_dtype=precision.kv_cache_dtype,
            # TTTv1 disables batched prefill for Qwen3-32B on P150x4. Keep that
            # architecture policy construction-time and retain the env A/B escape hatch on T3K.
            disable_batched_prefill=sku == "P150x4" or environment_flag("DISABLE_BATCHED_PREFILL"),
            batched_prefill_batched_extract=not environment_flag("DISABLE_BATCHED_EXTRACT"),
        )
        tokenizer = load_tokenizer(hf_model, hf_revision)
        llm = Qwen3_32BForCausalLM(
            model=model,
            tokenizer=tokenizer,
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


@dataclass(frozen=True)
class Qwen3_32BExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Qwen3-32B."""


def _facade_target(facade: Any) -> Any:
    """Return the composed owner, or a test-fabricated facade itself."""

    return getattr(facade, "__dict__", {}).get("_model_executor", facade)


def _delegate_to_model_executor(method_name: str) -> Callable[..., Any]:
    """Create one signature-preserving facade method."""

    model_executor_method = getattr(ModelExecutor, method_name)

    @wraps(model_executor_method)
    def delegated(self, *args, **kwargs):
        return model_executor_method(_facade_target(self), *args, **kwargs)

    return delegated


class _ExecutorFacadeSurface:
    """Descriptors copied onto the public facade without inheritance."""

    @property
    def model_config(self):
        target = _facade_target(self)
        return target.model.config if target is self else target.model_config

    @property
    def cluster_shape(self) -> list[int]:
        target = _facade_target(self)
        return list(target.mesh_device.shape) if target is self else target.cluster_shape

    @property
    def paged_kv_cache_config(self) -> PagedKVCacheConfig:
        target = _facade_target(self)
        return target.kv_cache_manager.config if target is self else target.paged_kv_cache_config

    @property
    def terminal(self) -> bool:
        target = _facade_target(self)
        return bool(target._terminal) if target is self else target.terminal

    @property
    def already_warmed_up_prefill(self) -> bool:
        target = _facade_target(self)
        return target.warmup.already_warmed_up_prefill if target is self else target.already_warmed_up_prefill

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value: bool) -> None:
        target = _facade_target(self)
        if target is self:
            return ModelExecutor.already_warmed_up_prefill.fset(target, value)
        target.already_warmed_up_prefill = value

    configure_paged_kv_cache = _delegate_to_model_executor("configure_paged_kv_cache")
    allocate_kv_cache = _delegate_to_model_executor("allocate_kv_cache")
    compile_prefill = _delegate_to_model_executor("compile_prefill")
    compile_decode = _delegate_to_model_executor("compile_decode")
    prefill_forward = _delegate_to_model_executor("prefill_forward")
    decode_forward = _delegate_to_model_executor("decode_forward")
    can_trace_prefill = _delegate_to_model_executor("can_trace_prefill")
    read_decode_output = _delegate_to_model_executor("read_decode_output")
    process_decode_output_host = _delegate_to_model_executor("process_decode_output_host")
    warmup_model_prefill = _delegate_to_model_executor("warmup_model_prefill")
    warmup_model_decode = _delegate_to_model_executor("warmup_model_decode")
    cleanup = _delegate_to_model_executor("cleanup")

    def __getattr__(self, name: str) -> Any:
        executor = getattr(self, "__dict__", {}).get("_model_executor")
        if executor is None:
            raise AttributeError(name)
        return getattr(executor, name)

    def __setattr__(self, name: str, value: Any) -> None:
        executor = getattr(self, "__dict__", {}).get("_model_executor")
        if name == "_model_executor" or executor is None:
            object.__setattr__(self, name, value)
            return
        setattr(executor, name, value)


_FACADE_SURFACE = (
    "model_config",
    "cluster_shape",
    "paged_kv_cache_config",
    "terminal",
    "already_warmed_up_prefill",
    "configure_paged_kv_cache",
    "allocate_kv_cache",
    "compile_prefill",
    "compile_decode",
    "prefill_forward",
    "decode_forward",
    "can_trace_prefill",
    "read_decode_output",
    "process_decode_output_host",
    "warmup_model_prefill",
    "warmup_model_decode",
    "cleanup",
    "__getattr__",
    "__setattr__",
)


def _with_executor_facade(cls):
    for name in _FACADE_SURFACE:
        if name not in cls.__dict__:
            setattr(cls, name, _ExecutorFacadeSurface.__dict__[name])
    return cls


@_with_executor_facade
class Qwen3_32BExecutor:
    """Qwen3-32B policy facade over one family-neutral executor."""

    requires_prefill_trace_warmup = True
    _owner_name = "Qwen3_32BExecutor"
    request_state_fields = ("prompt_tokens", "output_tokens", "slot_remap")

    def __init__(self, model: Any, runtime_config: Any, config: Qwen3_32BExecutorConfig) -> None:
        if not isinstance(config, Qwen3_32BExecutorConfig):
            raise TypeError("config must be a Qwen3_32BExecutorConfig")

        sampling_state_controller, sampling_state = _create_sampling_state(model, config.device_sampling_enabled)
        trace_capture_prime_sequence_lengths = _resolve_trace_capture_prime_sequence_lengths(
            runtime_config,
            num_devices=int(model.config.mesh_device.get_num_devices()),
        )
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Qwen3_32BExecutor",
            disable_batched_prefill=bool(runtime_config.disable_batched_prefill) or config.device_sampling_enabled,
            trace_capture_prime_sequence_lengths=trace_capture_prime_sequence_lengths,
            sampling_state_controller=sampling_state_controller,
            sampling_state=sampling_state,
            sampling_type=Sampling1D,
            request_state_fields=self.request_state_fields,
            prefill_warmup=_warmup_q128_around_prefill,
        )
        self._model_executor._q128_topk_tile_ends_warmed = set()


def _create_sampling_state(model: Any, enabled: bool) -> tuple[Any, Any]:
    if not enabled:
        return None, None
    sampling = getattr(model, "sampling", None)
    if not isinstance(sampling, Sampling1D):
        raise TypeError("device sampling requires model.sampling to be a Sampling1D")
    is_resolved = getattr(getattr(sampling, "config", None), "is_resolved", None)
    if not callable(is_resolved) or not is_resolved():
        raise ValueError("model.sampling must have a resolved Sampling1DConfig")
    controller = SamplingState1D(sampling)
    return controller, controller.create_state()


def _warmup_q128_around_prefill(
    executor: ModelExecutor,
    default_warmup: Callable[[], None],
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
    """Prime lane-independent Q128 top-k programs before trace activation."""

    if enable_trace:
        _warmup_q128_topk_tile_ends(
            executor,
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=True,
        )
    default_warmup()
    if not enable_trace:
        _warmup_q128_topk_tile_ends(
            executor,
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=False,
        )


def _warmup_q128_topk_tile_ends(
    executor: ModelExecutor,
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
    """Prime Q128 top-k slice programs when sampler capacity exceeds lane capacity."""

    if (
        enable_trace in executor._q128_topk_tile_ends_warmed
        or not can_sample_on_device
        or (enable_trace and executor.traced_executor is None)
        or 128 not in executor.warmup.config.prefill_sequence_lengths
        or not executor.prefill_runtime.config.static_q128_topk_supported
        or executor.warmup.config.prime_q128_tile_ends
    ):
        return
    sampling = SamplingParams(
        temperature=torch.ones(1),
        top_k=torch.full((1,), 32, dtype=torch.int32),
        top_p=torch.full((1,), 0.08),
    )
    # Without a Q128 trace family the trace pass primes these programs eagerly.
    traced = enable_trace and 128 in executor.warmup.config.prefill_trace_sequence_lengths
    execution = executor.traced_executor if traced else executor.eager_executor
    for sequence_length in (32, 64, 96):
        page_table_width = (
            sequence_length + executor.page_table_layout.block_size - 1
        ) // executor.page_table_layout.block_size
        executor.compile_prefill(
            tokens=torch.zeros((1, sequence_length), dtype=torch.long),
            page_table=torch.zeros((1, page_table_width), dtype=torch.int32),
            prompt_lens=torch.full((1,), sequence_length, dtype=torch.long),
            empty_slots=[0],
            kv_cache=kv_cache,
            sampling_params=sampling,
            execution=execution,
        )
    executor._q128_topk_tile_ends_warmed.add(enable_trace)


def build_qwen3_32b_executor(llm: Qwen3_32BForCausalLM, config: Qwen3_32BExecutorConfig) -> Qwen3_32BExecutor:
    return Qwen3_32BExecutor(llm.model, llm.runtime_config, config)


def _resolve_trace_capture_prime_sequence_lengths(runtime_config: Any, *, num_devices: int) -> tuple[int, ...]:
    """Prime every advertised T3K prefill trace body independently of batching policy."""

    if num_devices != 8:
        return ()
    advertised = tuple(getattr(runtime_config, "trace_prefill_supported_seq_lens", (128,)))
    return tuple(length for length in advertised if runtime_config.can_enable_trace(length, 0))


def _compat_executor_config(model, *, trace_mode: str, device_sampling_enabled: bool) -> Qwen3_32BExecutorConfig:
    runtime_config = getattr(model, "model_args", None)
    if runtime_config is None:
        raise ValueError("Qwen3_32B compatibility executor requires model.model_args")
    block_size = 32
    max_seq_len = int(getattr(runtime_config, "max_seq_len", model.config.max_seq_len))
    max_batch_size = int(getattr(runtime_config, "max_batch_size", model.config.max_batch_size))
    max_num_blocks = ((max_seq_len + block_size - 1) // block_size) * max_batch_size
    return Qwen3_32BExecutorConfig(
        trace=TraceConfig(mode=trace_mode),
        warmup=WarmupConfig(
            prefill_seq_lens=tuple(getattr(runtime_config, "trace_prefill_supported_seq_lens", (128, 1024))),
            prefill_batch_sizes=(1,),
            include_decode_top_k=device_sampling_enabled,
        ),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            dtype=getattr(runtime_config, "kv_cache_dtype", ttnn.bfloat8_b),
        ),
        device_sampling_enabled=device_sampling_enabled,
    )


class EagerQwen3_32BExecutor(Qwen3_32BExecutor):
    """Compatibility wrapper over the model-owned eager runtime."""

    def __init__(self, model, mesh_device):
        del mesh_device
        super().__init__(
            model,
            model.model_args,
            _compat_executor_config(model, trace_mode="none", device_sampling_enabled=False),
        )


class TracedQwen3_32BExecutor(Qwen3_32BExecutor):
    """Compatibility wrapper over the model-owned traced runtime."""

    def __init__(
        self,
        model,
        mesh_device,
        ondevice_decode_loop: bool = False,
        fast_prefill_last_token: bool = False,
        trace_mode: str = "all",
    ):
        del mesh_device, fast_prefill_last_token
        super().__init__(
            model,
            model.model_args,
            _compat_executor_config(
                model,
                trace_mode=trace_mode,
                device_sampling_enabled=bool(ondevice_decode_loop),
            ),
        )
        # Transitional compatibility for demo perf helpers that identify
        # trace-capable executors by the legacy trace bookkeeping attributes.
        self.trace_id_prefill = defaultdict(lambda: None)
        self.trace_ids_decode = defaultdict(lambda: None)


def run_prefill(model, token_ids_tt, *, start_pos: int = 0):
    return model.prefill_from_token_ids(token_ids_tt, start_pos=start_pos)


def run_decode(model, token_id_tt, *, current_pos: int):
    return model.decode_from_token_ids(token_id_tt, current_pos=current_pos)


def run_lm_head(model, hidden_tt):
    if len(hidden_tt.shape) == 4 and hidden_tt.shape[2] > 32:
        old = hidden_tt
        hidden_tt = _slice_last_token_tile(old, hidden_tt.shape[2] - 1)
        ttnn.deallocate(old)
    return model.lm_logits(hidden_tt)


def from_pretrained(
    mesh_device,
    *,
    hf_model: str = DEFAULT_HF_MODEL,
    hf_revision: str | None = DEFAULT_HF_REVISION,
    instruct: bool = True,
    max_batch_size: int = 32,
    max_seq_len: int = 4096,
    optimizations: str | Qwen3_32BPrecisionConfig = "accuracy",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config: Qwen3_32BPagedAttentionConfig | None = None,
    cache_dir: Path | str | None = None,
    executor_config: Qwen3_32BExecutorConfig | None = None,
) -> Qwen3_32BForCausalLM:
    """Load a text model with one executor ready for ``generate()``.

    Inputs and generated IDs are CPU PyTorch tensors. The supplied mesh remains
    caller-owned. By default execution is eager with host token selection; pass
    the existing ``Qwen3_32BExecutorConfig`` for explicit trace/sampling policy.
    """
    if executor_config is not None and not isinstance(executor_config, Qwen3_32BExecutorConfig):
        raise TypeError("executor_config must be a Qwen3_32BExecutorConfig")
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
        paged_attention_config = Qwen3_32BPagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
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
            executor_config = Qwen3_32BExecutorConfig(
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
        llm.executor = build_qwen3_32b_executor(llm, executor_config)
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
