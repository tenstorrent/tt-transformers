# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Qwen25Coder32B": ("tt_transformers.models.qwen25_coder_32b.model", "Qwen25Coder32B"),
    "Qwen25Coder32BConfig": ("tt_transformers.models.qwen25_coder_32b.model", "Qwen25Coder32BConfig"),
    "Qwen25Coder32BPagedAttentionConfig": (
        "tt_transformers.models.qwen25_coder_32b.model",
        "Qwen25Coder32BPagedAttentionConfig",
    ),
    "Qwen25Coder32BPrecisionConfig": ("tt_transformers.models.qwen25_coder_32b.model", "Qwen25Coder32BPrecisionConfig"),
    "Qwen25Coder32BLayerWeights": ("tt_transformers.models.qwen25_coder_32b.model", "Qwen25Coder32BLayerWeights"),
    "Qwen25Coder32BWeights": ("tt_transformers.models.qwen25_coder_32b.model", "Qwen25Coder32BWeights"),
    "build_qwen25_coder_32b_model": ("tt_transformers.models.qwen25_coder_32b.model", "build_qwen25_coder_32b_model"),
    "QWEN25_CODER_32B_ACCURACY": ("tt_transformers.models.qwen25_coder_32b.model", "QWEN25_CODER_32B_ACCURACY"),
    "QWEN25_CODER_32B_PERFORMANCE": ("tt_transformers.models.qwen25_coder_32b.model", "QWEN25_CODER_32B_PERFORMANCE"),
    "Qwen25Coder32BGenerator": ("tt_transformers.models.qwen25_coder_32b.vllm_generator", "Qwen25Coder32BGenerator"),
    "Qwen25Coder32BGeneratorConfig": (
        "tt_transformers.models.qwen25_coder_32b.vllm_generator",
        "Qwen25Coder32BGeneratorConfig",
    ),
    "build_qwen25_coder_32b_generator": (
        "tt_transformers.models.qwen25_coder_32b.vllm_generator",
        "build_qwen25_coder_32b_generator",
    ),
    "Qwen25Coder32BExecutor": ("tt_transformers.models.qwen25_coder_32b.hf_generator", "Qwen25Coder32BExecutor"),
    "Qwen25Coder32BExecutorConfig": (
        "tt_transformers.models.qwen25_coder_32b.hf_generator",
        "Qwen25Coder32BExecutorConfig",
    ),
    "build_qwen25_coder_32b_executor": (
        "tt_transformers.models.qwen25_coder_32b.hf_generator",
        "build_qwen25_coder_32b_executor",
    ),
    "EagerQwen25Coder32BExecutor": (
        "tt_transformers.models.qwen25_coder_32b.hf_generator",
        "EagerQwen25Coder32BExecutor",
    ),
    "TracedQwen25Coder32BExecutor": (
        "tt_transformers.models.qwen25_coder_32b.hf_generator",
        "TracedQwen25Coder32BExecutor",
    ),
    "Qwen25Coder32BForCausalLM": ("tt_transformers.models.qwen25_coder_32b.hf_generator", "Qwen25Coder32BForCausalLM"),
    "from_pretrained": ("tt_transformers.models.qwen25_coder_32b.hf_generator", "from_pretrained"),
}

__all__ = [
    "Qwen25Coder32B",
    "Qwen25Coder32BConfig",
    "Qwen25Coder32BPagedAttentionConfig",
    "Qwen25Coder32BPrecisionConfig",
    "Qwen25Coder32BLayerWeights",
    "Qwen25Coder32BWeights",
    "build_qwen25_coder_32b_model",
    "QWEN25_CODER_32B_ACCURACY",
    "QWEN25_CODER_32B_PERFORMANCE",
    "Qwen25Coder32BGenerator",
    "Qwen25Coder32BGeneratorConfig",
    "build_qwen25_coder_32b_generator",
    "Qwen25Coder32BExecutor",
    "Qwen25Coder32BExecutorConfig",
    "build_qwen25_coder_32b_executor",
    "EagerQwen25Coder32BExecutor",
    "TracedQwen25Coder32BExecutor",
    "Qwen25Coder32BForCausalLM",
    "from_pretrained",
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
