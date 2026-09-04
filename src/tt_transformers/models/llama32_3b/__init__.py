# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "LLAMA32_3B_ACCURACY": ("tt_transformers.models.llama32_3b.model", "LLAMA32_3B_ACCURACY"),
    "LLAMA32_3B_PERFORMANCE": ("tt_transformers.models.llama32_3b.model", "LLAMA32_3B_PERFORMANCE"),
    "Llama32_3BExecutor": ("tt_transformers.models.llama32_3b.executor", "Llama32_3BExecutor"),
    "Llama32_3BExecutorConfig": ("tt_transformers.models.llama32_3b.executor", "Llama32_3BExecutorConfig"),
    "Llama32_3BForCausalLM": ("tt_transformers.models.llama32_3b.hf_adaptor", "Llama32_3BForCausalLM"),
    "Llama32_3BPrecisionConfig": ("tt_transformers.models.llama32_3b.model", "Llama32_3BPrecisionConfig"),
    "Llama32_3BRuntimeConfig": ("tt_transformers.models.llama32_3b.hf_adaptor", "Llama32_3BRuntimeConfig"),
    "Llama32_3BTransformer1D": ("tt_transformers.models.llama32_3b.model", "Llama32_3BTransformer1D"),
    "Llama32_3BTransformer1DConfig": ("tt_transformers.models.llama32_3b.model", "Llama32_3BTransformer1DConfig"),
}

__all__ = [
    "LLAMA32_3B_ACCURACY",
    "LLAMA32_3B_PERFORMANCE",
    "Llama32_3BExecutor",
    "Llama32_3BExecutorConfig",
    "Llama32_3BForCausalLM",
    "Llama32_3BPrecisionConfig",
    "Llama32_3BRuntimeConfig",
    "Llama32_3BTransformer1D",
    "Llama32_3BTransformer1DConfig",
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
