# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Qwen25_72B": ("tt_transformers.models.qwen25_72b.model", "Qwen25_72B"),
    "Qwen25_72BConfig": ("tt_transformers.models.qwen25_72b.model", "Qwen25_72BConfig"),
    "Qwen25_72BExecutorRuntimeConfig": ("tt_transformers.models.qwen25_72b.model", "Qwen25_72BExecutorRuntimeConfig"),
    "Qwen25_72BPagedAttentionConfig": ("tt_transformers.models.qwen25_72b.model", "Qwen25_72BPagedAttentionConfig"),
    "Qwen25_72BPrecisionConfig": ("tt_transformers.models.qwen25_72b.model", "Qwen25_72BPrecisionConfig"),
    "QWEN25_72B_ACCURACY": ("tt_transformers.models.qwen25_72b.model", "QWEN25_72B_ACCURACY"),
    "QWEN25_72B_PERFORMANCE": ("tt_transformers.models.qwen25_72b.model", "QWEN25_72B_PERFORMANCE"),
    "Qwen25_72BGenerator": ("tt_transformers.models.qwen25_72b.vllm_generator", "Qwen25_72BGenerator"),
    "Qwen25_72BGeneratorConfig": ("tt_transformers.models.qwen25_72b.vllm_generator", "Qwen25_72BGeneratorConfig"),
    "Qwen25_72BExecutor": ("tt_transformers.models.qwen25_72b.hf_generator", "Qwen25_72BExecutor"),
    "Qwen25_72BExecutorConfig": ("tt_transformers.models.qwen25_72b.hf_generator", "Qwen25_72BExecutorConfig"),
    "Qwen25_72BForCausalLM": ("tt_transformers.models.qwen25_72b.hf_generator", "Qwen25_72BForCausalLM"),
    "build_qwen25_72b_executor": ("tt_transformers.models.qwen25_72b.hf_generator", "build_qwen25_72b_executor"),
    "build_qwen25_72b_generator": ("tt_transformers.models.qwen25_72b.vllm_generator", "build_qwen25_72b_generator"),
    "from_pretrained": ("tt_transformers.models.qwen25_72b.hf_generator", "from_pretrained"),
}

__all__ = [
    "Qwen25_72B",
    "Qwen25_72BConfig",
    "Qwen25_72BExecutorRuntimeConfig",
    "Qwen25_72BPagedAttentionConfig",
    "Qwen25_72BPrecisionConfig",
    "QWEN25_72B_ACCURACY",
    "QWEN25_72B_PERFORMANCE",
    "Qwen25_72BGenerator",
    "Qwen25_72BGeneratorConfig",
    "Qwen25_72BExecutor",
    "Qwen25_72BExecutorConfig",
    "Qwen25_72BForCausalLM",
    "build_qwen25_72b_executor",
    "build_qwen25_72b_generator",
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
