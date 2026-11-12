# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared vLLM facade for model-owned execution targets.

Every served model exposes the same vLLM interface: normalize each call with
`VLLMAdapter`, pick the eager or traced execution, and delegate to the model's
executor or ``LaneGroupExecutor``. That delegation lives here once. A model's
``vllm_generator`` module subclasses one of these classes and declares what is
its own: ``model_capabilities`` and ``initialize_vllm_model``. A model that
needs more wraps a shared method and calls ``super()``, so a keyword added
here still reaches it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import torch

from tt_transformers.llm_runtime.vllm_adapter import NormalizedPrefillKwargs, VLLMAdapter


class VLLMGeneratorFacade:
    """Adapt vLLM's model interface to a model-owned execution target.

    vLLM constructs the facade with ``initialize_vllm_model``, resolves KV
    capacity through `allocate_kv_cache`, warms the configured programs, and
    then calls `prefill_forward` and `decode_forward`. Each forward call is
    normalized by `VLLMAdapter`, dispatched to the eager or traced execution,
    and delegated to the target.

    The facade owns dispatch policy but no TT resources. `cleanup` delegates
    to the target that owns them.
    """

    decode_input_update_contract = 1
    requires_prefill_trace_warmup = True
    model_capabilities: ClassVar[dict[str, Any]]

    def __init__(self, target: Any, adapter: VLLMAdapter):
        self.target = target
        self._adapter = adapter

    # Public vLLM API

    @property
    def model(self):
        return self.target.model

    @property
    def model_args(self):
        return self.target.model_args

    @property
    def mesh_device(self):
        return self.target.mesh_device

    @property
    def cache_path(self):
        return self.target.cache_path

    @property
    def already_warmed_up_prefill(self):
        return self.target.already_warmed_up_prefill

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value):
        self.target.already_warmed_up_prefill = value

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int = 0,
        max_num_seqs: int = 1,
    ) -> int:
        """Return the unpadded per-submesh KV token budget for vLLM sizing."""

        return int(max_model_len)

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        """Resolve the late vLLM capacity, then allocate a borrowed cache handle."""

        supplied = (kv_cache_shape is not None, dtype is not None, num_layers is not None)
        if not any(supplied):
            return self.target.allocate_kv_cache()
        if not all(supplied):
            raise TypeError("kv_cache_shape, dtype, and num_layers must be supplied together")

        resolved = self._adapter.resolve_legacy_kv_cache_config(kv_cache_shape, dtype, num_layers)
        self.target.configure_paged_kv_cache(resolved)
        return self.target.allocate_kv_cache()

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,  # ↓ Required policy
        prompt_lens: Sequence[int] | torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        **compatibility_kwargs: Any,  # ↓ Compatibility
    ) -> Any:
        """Normalize and dispatch one vLLM prefill call."""

        normalized, trace_requested = self._adapter.normalize_prefill(
            tokens,
            page_table,
            enable_trace=enable_trace,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            compatibility_kwargs=compatibility_kwargs,
        )
        execution = self._select_prefill_execution(normalized, trace_requested)
        return self.target.prefill_forward(execution=execution, **normalized)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,  # ↓ Required policy
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reload_inputs: bool = True,  # ↓ Decode update commands
        reload_page_table: bool = False,
        reload_sampling_params: bool = False,
        reset_sampling_state: bool = False,
        read_from_device: bool = True,  # ↓ Output policy
        defer_device_sampling: bool = False,  # ↓ Deferred device sampling
        **compatibility_kwargs: Any,  # ↓ Compatibility
    ) -> Any:
        """Normalize and dispatch one vLLM decode call.

        With ``defer_device_sampling=True`` the forward runs now and returns an
        opaque payload for `sample_deferred_decode`.
        """

        normalized, trace_requested = self._adapter.normalize_decode(
            tokens,
            start_pos,
            page_table,
            enable_trace=enable_trace,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
            defer_device_sampling=defer_device_sampling,
            read_from_device=read_from_device,
            compatibility_kwargs=compatibility_kwargs,
        )
        execution = self._select_execution("decode", trace_requested)
        return self.target.decode_forward(
            execution=execution,
            read_from_device=read_from_device,
            **normalized,
        )

    def sample_deferred_decode(
        self,
        payload: Any,
        sampling_params: Any = None,
        *,
        grammar_bitmask: Any = None,
    ) -> Any:
        """Sample a deferred decode through the step's grammar mask (vLLM structured outputs)."""

        return self.target.sample_deferred_decode(
            payload,
            sampling_params=sampling_params,
            grammar_bitmask=grammar_bitmask,
        )

    def enable_device_grammar(self) -> None:
        """Allocate grammar mask buffers before warmup, when the target supports them."""

        enable = getattr(self.target, "enable_device_grammar", None)
        if callable(enable):
            enable()

    @property
    def device_grammar_enabled(self) -> bool:
        """True only when every lane of the target can apply a grammar mask."""

        return bool(getattr(self.target, "device_grammar_enabled", False))

    def read_decode_output(
        self,
        tt_out: Any,
        *,
        async_read: bool = False,
    ) -> Any:
        """Delegate vLLM's raw decode-output read."""

        return self.target.read_decode_output(tt_out=tt_out, async_read=async_read)

    def process_decode_output_host(
        self,
        tt_out: Any,
        *,
        is_tokens: bool = False,
    ) -> tuple[Any, Any]:
        """Delegate vLLM's asynchronous host-output completion."""

        return self.target.process_decode_output_host(tt_out=tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        result = self.target.warmup_model_prefill(
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )
        self._after_warmup()
        return result

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        max_batch_size: int,  # ↓ Coverage dimensions
        num_blocks: int,
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
        can_sample_device_grammar: bool = False,
    ) -> None:
        grammar = {"can_sample_device_grammar": True} if can_sample_device_grammar else {}
        result = self.target.warmup_model_decode(
            kv_cache=kv_cache,
            max_batch_size=max_batch_size,
            num_blocks=num_blocks,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
            **grammar,
        )
        self._after_warmup()
        return result

    def cleanup(self):
        """Release every resource owned by the concrete target."""

        return self.target.cleanup()

    # Private implementation

    def _after_warmup(self) -> None:
        """Run after each successful warmup call; a model may override it."""

    def _select_prefill_execution(
        self,
        normalized: NormalizedPrefillKwargs,
        trace_requested: bool,
    ):
        # Guarded eager degrade. Static trace intent still gates whether a trace
        # is attempted, but a request whose padded bucket was never captured has no
        # required trace, so it runs eager (against a program pre-compiled at warmup)
        # rather than raising. A trace-eligible request still selects the traced target,
        # which hard-fails on a missing artifact, so a required trace miss is never turned
        # into eager KV writes.
        if not trace_requested:
            return self.target.eager_execution
        if self.target.can_trace_prefill(
            tokens=normalized["tokens"],
            prompt_lens=normalized.get("prompt_lens"),
            start_pos=normalized.get("start_pos"),
            empty_slots=normalized.get("empty_slots"),
        ):
            return self._select_execution("prefill", True)
        self.target.note_eager_prefill_degrade(
            tokens=normalized["tokens"],
            prompt_lens=normalized.get("prompt_lens"),
            start_pos=normalized.get("start_pos"),
            empty_slots=normalized.get("empty_slots"),
        )
        return self.target.eager_execution

    def _select_execution(self, operation: str, enable_trace: bool):
        if not enable_trace:
            return self.target.eager_execution
        execution = getattr(self.target, f"traced_{operation}_execution")
        if execution is None:
            raise RuntimeError(f"vLLM requested unavailable traced {operation} execution")
        return execution


class CompilingVLLMGeneratorFacade(VLLMGeneratorFacade):
    """A `VLLMGeneratorFacade` that also exposes compile-only entry points.

    `compile_prefill` and `compile_decode` normalize a call exactly as the
    matching forward does, then ask the target to compile it without running.
    """

    def compile_prefill(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,  # ↓ Required policy
        prompt_lens: Sequence[int] | torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
    ) -> None:
        """Normalize a vLLM prefill call and compile its selected target."""

        normalized, trace_requested = self._adapter.normalize_prefill(
            tokens,
            page_table,
            enable_trace=enable_trace,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )
        execution = self._select_prefill_execution(normalized, trace_requested)
        return self.target.compile_prefill(execution=execution, **normalized)

    def compile_decode(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,  # ↓ Required policy
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reload_inputs: bool = True,  # ↓ Decode update commands
        reload_page_table: bool = False,
        reload_sampling_params: bool = False,
        reset_sampling_state: bool = False,
    ) -> None:
        """Normalize a vLLM decode call and compile its selected target."""

        normalized, trace_requested = self._adapter.normalize_decode(
            tokens,
            start_pos,
            page_table,
            enable_trace=enable_trace,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
        )
        execution = self._select_execution("decode", trace_requested)
        return self.target.compile_decode(execution=execution, **normalized)
