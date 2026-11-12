# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM device grammar surface shared by every lane executor.

vLLM structured outputs constrain decode with a per-step allowed-token mask
that vLLM fills only after the step's forward is submitted. A lane executor
therefore runs a deferred decode's model body first and samples it later,
through the mask, with `sample_deferred_decode`. The executor owns the policy
here; the decode runtime and the executors in ``llm_runtime.execution`` do the
work.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from tt_transformers.llm_runtime.decode import DeferredDecode


class DeviceGrammarExecutorMixin:
    """Grammar members for an executor with ``decode_runtime``, ``trace_compiler`` and ``config``."""

    decode_runtime: Any
    trace_compiler: Any
    config: Any

    @property
    def device_grammar_enabled(self) -> bool:
        """True once the sampler holds a grammar mask buffer."""

        return bool(self.decode_runtime.grammar_enabled)

    def enable_device_grammar(self) -> None:
        """Allocate grammar mask buffers before warmup; leave the feature off if unsupported.

        The vLLM plugin calls this once after load. A sampler whose vocab layout
        cannot apply a mask, or a lane without device sampling, keeps host
        grammar: `device_grammar_enabled` stays False and nothing raises.
        """

        self._ensure_active()
        if self.device_grammar_enabled:
            return
        if self.trace_compiler is not None and self.trace_compiler.trace_active:
            raise RuntimeError("enable_device_grammar() must run before any trace is captured")
        if not self.config.device_sampling_enabled:
            logger.warning("Device grammar sampling stays off: device sampling is disabled")
            return
        self._ensure_sampling_buffers()
        try:
            self.decode_runtime.enable_grammar()
        except ValueError as error:
            logger.warning(f"Device grammar sampling stays off: {error}")

    def sample_deferred_decode(self, payload: Any, *, sampling_params: Any, grammar_bitmask: Any = None) -> Any:
        """Sample a deferred decode through ``grammar_bitmask``; return the device output.

        Returns what ``decode_forward(..., read_from_device=False)`` returns for a
        device-sampled step. Any failure after validation leaves the executor
        unusable.
        """

        self._ensure_active()
        if sampling_params is None:
            raise ValueError("sample_deferred_decode requires sampling_params")
        self._ensure_sampling_for(sampling_params)
        owner = payload.owner if isinstance(payload, DeferredDecode) else None
        if owner is None:
            self.decode_runtime.take_deferred(payload)  # raises: not the pending payload
        return owner.sample_deferred_decode(payload, sampling_params=sampling_params, grammar_bitmask=grammar_bitmask)

    def _defer_decode(self, execution: Any, *, sampling_params: Any, read_from_device: bool, **request: Any) -> Any:
        if not self.device_grammar_enabled:
            raise RuntimeError("deferred device sampling requires enable_device_grammar()")
        if sampling_params is not None or read_from_device:
            raise ValueError("a deferred decode takes no sampling_params and requires read_from_device=False")
        return execution.defer_decode(**request)

    def _grammar_warmup_kwargs(self) -> dict[str, Any]:
        """Warmup keywords for a caller that asked for grammar twins: only when grammar is enabled."""

        return {"can_sample_device_grammar": True} if self.device_grammar_enabled else {}

    # Provided by the executor.
    def _ensure_active(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _ensure_sampling_buffers(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _ensure_sampling_for(self, sampling_params: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError
