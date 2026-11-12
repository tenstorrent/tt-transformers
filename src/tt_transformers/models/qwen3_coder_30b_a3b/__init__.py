# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Qwen3Coder30BA3B": ("tt_transformers.models.qwen3_coder_30b_a3b.model", "Qwen3Coder30BA3B"),
    "Qwen3Coder30BA3BConfig": ("tt_transformers.models.qwen3_coder_30b_a3b.model", "Qwen3Coder30BA3BConfig"),
    "Qwen3Coder30BA3BExecutorRuntimeConfig": (
        "tt_transformers.models.qwen3_coder_30b_a3b.model",
        "Qwen3Coder30BA3BExecutorRuntimeConfig",
    ),
    "Qwen3Coder30BA3BPagedAttentionConfig": (
        "tt_transformers.models.qwen3_coder_30b_a3b.model",
        "Qwen3Coder30BA3BPagedAttentionConfig",
    ),
    "Qwen3Coder30BA3BPrecisionConfig": (
        "tt_transformers.models.qwen3_coder_30b_a3b.model",
        "Qwen3Coder30BA3BPrecisionConfig",
    ),
    "QWEN3_CODER_30B_A3B_ACCURACY": (
        "tt_transformers.models.qwen3_coder_30b_a3b.model",
        "QWEN3_CODER_30B_A3B_ACCURACY",
    ),
    "QWEN3_CODER_30B_A3B_PERFORMANCE": (
        "tt_transformers.models.qwen3_coder_30b_a3b.model",
        "QWEN3_CODER_30B_A3B_PERFORMANCE",
    ),
    "Qwen3Coder30BA3BGenerator": (
        "tt_transformers.models.qwen3_coder_30b_a3b.vllm_generator",
        "Qwen3Coder30BA3BGenerator",
    ),
    "Qwen3Coder30BA3BGeneratorConfig": (
        "tt_transformers.models.qwen3_coder_30b_a3b.vllm_generator",
        "Qwen3Coder30BA3BGeneratorConfig",
    ),
    "EagerQwen3Coder30BA3BExecutor": (
        "tt_transformers.models.qwen3_coder_30b_a3b.hf_generator",
        "EagerQwen3Coder30BA3BExecutor",
    ),
    "TracedQwen3Coder30BA3BExecutor": (
        "tt_transformers.models.qwen3_coder_30b_a3b.hf_generator",
        "TracedQwen3Coder30BA3BExecutor",
    ),
}

__all__ = [
    "Qwen3Coder30BA3B",
    "Qwen3Coder30BA3BConfig",
    "Qwen3Coder30BA3BExecutorRuntimeConfig",
    "Qwen3Coder30BA3BPagedAttentionConfig",
    "Qwen3Coder30BA3BPrecisionConfig",
    "QWEN3_CODER_30B_A3B_ACCURACY",
    "QWEN3_CODER_30B_A3B_PERFORMANCE",
    "Qwen3Coder30BA3BGenerator",
    "Qwen3Coder30BA3BGeneratorConfig",
    "EagerQwen3Coder30BA3BExecutor",
    "TracedQwen3Coder30BA3BExecutor",
]


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
