# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "MISTRAL_ACCURACY": ("tt_transformers.models.mistral_7b.model", "MISTRAL_ACCURACY"),
    "MISTRAL_PERFORMANCE": ("tt_transformers.models.mistral_7b.model", "MISTRAL_PERFORMANCE"),
    "Mistral7B": ("tt_transformers.models.mistral_7b.model", "Mistral7B"),
    "Mistral7BExecutor": ("tt_transformers.models.mistral_7b.executor", "Mistral7BExecutor"),
    "Mistral7BExecutorConfig": ("tt_transformers.models.mistral_7b.executor", "Mistral7BExecutorConfig"),
    "Mistral7BForCausalLM": ("tt_transformers.models.mistral_7b.hf_adaptor", "Mistral7BForCausalLM"),
    "Mistral7BGenerator": ("tt_transformers.models.mistral_7b.generator", "Mistral7BGenerator"),
    "Mistral7BGeneratorConfig": ("tt_transformers.models.mistral_7b.generator", "Mistral7BGeneratorConfig"),
    "Mistral7BPagedAttentionConfig": ("tt_transformers.models.mistral_7b.model", "Mistral7BPagedAttentionConfig"),
    "Mistral7BPrecisionConfig": ("tt_transformers.models.mistral_7b.model", "Mistral7BPrecisionConfig"),
    "Mistral7BRuntimeConfig": ("tt_transformers.models.mistral_7b.hf_adaptor", "Mistral7BRuntimeConfig"),
    "Mistral7BTransformerConfig": ("tt_transformers.models.mistral_7b.model", "Mistral7BTransformerConfig"),
    "build_mistral_7b_executor": ("tt_transformers.models.mistral_7b.executor", "build_mistral_7b_executor"),
    "build_mistral_7b_generator": ("tt_transformers.models.mistral_7b.generator", "build_mistral_7b_generator"),
    "build_mistral_7b_transformer_config": (
        "tt_transformers.models.mistral_7b.model",
        "build_mistral_7b_transformer_config",
    ),
}

__all__ = [
    "MISTRAL_ACCURACY",
    "MISTRAL_PERFORMANCE",
    "Mistral7B",
    "Mistral7BExecutor",
    "Mistral7BExecutorConfig",
    "Mistral7BForCausalLM",
    "Mistral7BGenerator",
    "Mistral7BGeneratorConfig",
    "Mistral7BPagedAttentionConfig",
    "Mistral7BPrecisionConfig",
    "Mistral7BRuntimeConfig",
    "Mistral7BTransformerConfig",
    "build_mistral_7b_executor",
    "build_mistral_7b_generator",
    "build_mistral_7b_transformer_config",
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
