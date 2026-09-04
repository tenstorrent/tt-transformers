# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DEFAULT_HF_MODEL": ("tt_transformers.models.llama33_70b.hf_adaptor", "DEFAULT_HF_MODEL"),
    "LLAMA33_70B_ACCURACY": ("tt_transformers.models.llama33_70b.model", "LLAMA33_70B_ACCURACY"),
    "LLAMA33_70B_PERFORMANCE": ("tt_transformers.models.llama33_70b.model", "LLAMA33_70B_PERFORMANCE"),
    "Llama33_70BForCausalLM": ("tt_transformers.models.llama33_70b.hf_adaptor", "Llama33_70BForCausalLM"),
    "Llama33_70BGenerationConfig": ("tt_transformers.models.llama33_70b.hf_adaptor", "Llama33_70BGenerationConfig"),
    "Llama33_70BExecutor": ("tt_transformers.models.llama33_70b.executor", "Llama33_70BExecutor"),
    "Llama33_70BExecutorConfig": ("tt_transformers.models.llama33_70b.executor", "Llama33_70BExecutorConfig"),
    "Llama33_70BGenerator": ("tt_transformers.models.llama33_70b.generator", "Llama33_70BGenerator"),
    "Llama33_70BGeneratorConfig": ("tt_transformers.models.llama33_70b.generator", "Llama33_70BGeneratorConfig"),
    "Llama33_70BLayerWeights": ("tt_transformers.models.llama33_70b.model", "Llama33_70BLayerWeights"),
    "Llama33_70BModelParameters": ("tt_transformers.models.llama33_70b.model", "Llama33_70BModelParameters"),
    "Llama33_70BPagedAttentionConfig": ("tt_transformers.models.llama33_70b.model", "Llama33_70BPagedAttentionConfig"),
    "Llama33_70BPrecisionConfig": ("tt_transformers.models.llama33_70b.model", "Llama33_70BPrecisionConfig"),
    "Llama33_70BRuntimeConfig": ("tt_transformers.models.llama33_70b.hf_adaptor", "Llama33_70BRuntimeConfig"),
    "Llama33_70BTransformer1D": ("tt_transformers.models.llama33_70b.model", "Llama33_70BTransformer1D"),
    "Llama33_70BTransformer1DConfig": ("tt_transformers.models.llama33_70b.model", "Llama33_70BTransformer1DConfig"),
    "Llama33_70BWeights": ("tt_transformers.models.llama33_70b.model", "Llama33_70BWeights"),
    "build_llama33_70b_transformer_1d_config": (
        "tt_transformers.models.llama33_70b.model",
        "build_llama33_70b_transformer_1d_config",
    ),
    "build_llama33_70b_executor": ("tt_transformers.models.llama33_70b.executor", "build_llama33_70b_executor"),
    "build_llama33_70b_generator": ("tt_transformers.models.llama33_70b.generator", "build_llama33_70b_generator"),
    "from_pretrained": ("tt_transformers.models.llama33_70b.hf_adaptor", "from_pretrained"),
}

__all__ = [
    "DEFAULT_HF_MODEL",
    "LLAMA33_70B_ACCURACY",
    "LLAMA33_70B_PERFORMANCE",
    "Llama33_70BForCausalLM",
    "Llama33_70BGenerationConfig",
    "Llama33_70BExecutor",
    "Llama33_70BExecutorConfig",
    "Llama33_70BGenerator",
    "Llama33_70BGeneratorConfig",
    "Llama33_70BLayerWeights",
    "Llama33_70BModelParameters",
    "Llama33_70BPagedAttentionConfig",
    "Llama33_70BPrecisionConfig",
    "Llama33_70BRuntimeConfig",
    "Llama33_70BTransformer1D",
    "Llama33_70BTransformer1DConfig",
    "Llama33_70BWeights",
    "build_llama33_70b_transformer_1d_config",
    "build_llama33_70b_executor",
    "build_llama33_70b_generator",
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
