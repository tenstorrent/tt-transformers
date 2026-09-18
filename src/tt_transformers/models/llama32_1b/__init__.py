# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "LLAMA32_1B_ACCURACY": ("tt_transformers.models.llama32_1b.model", "LLAMA32_1B_ACCURACY"),
    "LLAMA32_1B_PERFORMANCE": ("tt_transformers.models.llama32_1b.model", "LLAMA32_1B_PERFORMANCE"),
    "Llama32_1BExecutor": ("tt_transformers.models.llama32_1b.hf_generator", "Llama32_1BExecutor"),
    "Llama32_1BExecutorConfig": ("tt_transformers.models.llama32_1b.hf_generator", "Llama32_1BExecutorConfig"),
    "Llama32_1BForCausalLM": ("tt_transformers.models.llama32_1b.hf_generator", "Llama32_1BForCausalLM"),
    "Llama32_1BPrecisionConfig": ("tt_transformers.models.llama32_1b.model", "Llama32_1BPrecisionConfig"),
    "Llama32_1BRuntimeConfig": ("tt_transformers.models.llama32_1b.hf_generator", "Llama32_1BRuntimeConfig"),
    "Llama32_1BTransformer1D": ("tt_transformers.models.llama32_1b.model", "Llama32_1BTransformer1D"),
    "Llama32_1BTransformer1DConfig": ("tt_transformers.models.llama32_1b.model", "Llama32_1BTransformer1DConfig"),
}

__all__ = [
    "LLAMA32_1B_ACCURACY",
    "LLAMA32_1B_PERFORMANCE",
    "Llama32_1BExecutor",
    "Llama32_1BExecutorConfig",
    "Llama32_1BForCausalLM",
    "Llama32_1BPrecisionConfig",
    "Llama32_1BRuntimeConfig",
    "Llama32_1BTransformer1D",
    "Llama32_1BTransformer1DConfig",
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
