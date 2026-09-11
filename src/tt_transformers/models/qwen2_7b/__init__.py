# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "QWEN2_7B_ACCURACY": ("tt_transformers.models.qwen2_7b.model", "QWEN2_7B_ACCURACY"),
    "QWEN2_7B_PERFORMANCE": ("tt_transformers.models.qwen2_7b.model", "QWEN2_7B_PERFORMANCE"),
    "Qwen2_7B": ("tt_transformers.models.qwen2_7b.model", "Qwen2_7B"),
    "Qwen2_7BPrecisionConfig": ("tt_transformers.models.qwen2_7b.model", "Qwen2_7BPrecisionConfig"),
    "Qwen2_7BTransformerConfig": ("tt_transformers.models.qwen2_7b.model", "Qwen2_7BTransformerConfig"),
    "Qwen2Executor": ("tt_transformers.models.qwen2_7b.hf_generator", "Qwen2Executor"),
    "Qwen2ExecutorConfig": ("tt_transformers.models.qwen2_7b.hf_generator", "Qwen2ExecutorConfig"),
    "Qwen2ForCausalLM": ("tt_transformers.models.qwen2_7b.hf_generator", "Qwen2ForCausalLM"),
    "Qwen2PagedAttentionConfig": ("tt_transformers.models.qwen2_7b.model", "Qwen2PagedAttentionConfig"),
    "Qwen2Generator": ("tt_transformers.models.qwen2_7b.vllm_generator", "Qwen2Generator"),
    "Qwen2GeneratorConfig": ("tt_transformers.models.qwen2_7b.vllm_generator", "Qwen2GeneratorConfig"),
    "Qwen2RuntimeConfig": ("tt_transformers.models.qwen2_7b.hf_generator", "Qwen2RuntimeConfig"),
}

__all__ = [
    "QWEN2_7B_ACCURACY",
    "QWEN2_7B_PERFORMANCE",
    "Qwen2_7B",
    "Qwen2_7BPrecisionConfig",
    "Qwen2_7BTransformerConfig",
    "Qwen2Executor",
    "Qwen2ExecutorConfig",
    "Qwen2ForCausalLM",
    "Qwen2PagedAttentionConfig",
    "Qwen2Generator",
    "Qwen2GeneratorConfig",
    "Qwen2RuntimeConfig",
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
