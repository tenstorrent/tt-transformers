# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DEFAULT_HF_MODEL": ("tt_transformers.models.llama33_70b_galaxy.hf_adaptor", "DEFAULT_HF_MODEL"),
    "LLAMA33_70B_CHECKPOINT_CONTRACT": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "LLAMA33_70B_CHECKPOINT_CONTRACT",
    ),
    "LLAMA33_70B_GALAXY_ACCURACY": ("tt_transformers.models.llama33_70b_galaxy.model", "LLAMA33_70B_GALAXY_ACCURACY"),
    "LLAMA33_70B_GALAXY_HF_MODEL": ("tt_transformers.models.llama33_70b_galaxy.model", "LLAMA33_70B_GALAXY_HF_MODEL"),
    "LLAMA33_70B_GALAXY_PERFORMANCE": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "LLAMA33_70B_GALAXY_PERFORMANCE",
    ),
    "LLAMA33_70B_PREFETCHED_WEIGHT_NAMES": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "LLAMA33_70B_PREFETCHED_WEIGHT_NAMES",
    ),
    "Llama33_70BGalaxyBlockConfig": ("tt_transformers.models.llama33_70b_galaxy.model", "Llama33_70BGalaxyBlockConfig"),
    "Llama33_70BGalaxyForCausalLM": (
        "tt_transformers.models.llama33_70b_galaxy.hf_adaptor",
        "Llama33_70BGalaxyForCausalLM",
    ),
    "Llama33_70BGalaxyGenerationConfig": (
        "tt_transformers.models.llama33_70b_galaxy.hf_adaptor",
        "Llama33_70BGalaxyGenerationConfig",
    ),
    "Llama33_70BGalaxyLayerWeights": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "Llama33_70BGalaxyLayerWeights",
    ),
    "Llama33_70BGalaxyLazyLayerWeights": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "Llama33_70BGalaxyLazyLayerWeights",
    ),
    "Llama33_70BGalaxyLazyWeights": ("tt_transformers.models.llama33_70b_galaxy.model", "Llama33_70BGalaxyLazyWeights"),
    "Llama33_70BGalaxyModelParameters": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "Llama33_70BGalaxyModelParameters",
    ),
    "Llama33_70BGalaxyPrecision": ("tt_transformers.models.llama33_70b_galaxy.model", "Llama33_70BGalaxyPrecision"),
    "Llama33_70BGalaxyRuntimeConfig": (
        "tt_transformers.models.llama33_70b_galaxy.hf_adaptor",
        "Llama33_70BGalaxyRuntimeConfig",
    ),
    "Llama33_70BGalaxyTransformer2D": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "Llama33_70BGalaxyTransformer2D",
    ),
    "Llama33_70BGalaxyTransformer2DConfig": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "Llama33_70BGalaxyTransformer2DConfig",
    ),
    "Llama33_70BGalaxyWeights": ("tt_transformers.models.llama33_70b_galaxy.model", "Llama33_70BGalaxyWeights"),
    "Llama33_70BTransformerBlock2D": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "Llama33_70BTransformerBlock2D",
    ),
    "build_llama33_70b_galaxy_lazy_weights": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "build_llama33_70b_galaxy_lazy_weights",
    ),
    "build_llama33_70b_galaxy_model": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "build_llama33_70b_galaxy_model",
    ),
    "build_llama33_70b_galaxy_transformer_2d_config": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "build_llama33_70b_galaxy_transformer_2d_config",
    ),
    "convert_hf_model_weights": ("tt_transformers.models.llama33_70b_galaxy.hf_adaptor", "convert_hf_model_weights"),
    "default_paged_attention_config": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "default_paged_attention_config",
    ),
    "from_pretrained": ("tt_transformers.models.llama33_70b_galaxy.hf_adaptor", "from_pretrained"),
    "load_tokenizer": ("tt_transformers.models.llama33_70b_galaxy.hf_adaptor", "load_tokenizer"),
    "parameters_from_hf_config": ("tt_transformers.models.llama33_70b_galaxy.model", "parameters_from_hf_config"),
    "validate_llama33_70b_checkpoint": (
        "tt_transformers.models.llama33_70b_galaxy.model",
        "validate_llama33_70b_checkpoint",
    ),
}

__all__ = [
    "DEFAULT_HF_MODEL",
    "LLAMA33_70B_CHECKPOINT_CONTRACT",
    "LLAMA33_70B_GALAXY_ACCURACY",
    "LLAMA33_70B_GALAXY_HF_MODEL",
    "LLAMA33_70B_GALAXY_PERFORMANCE",
    "LLAMA33_70B_PREFETCHED_WEIGHT_NAMES",
    "Llama33_70BGalaxyBlockConfig",
    "Llama33_70BGalaxyForCausalLM",
    "Llama33_70BGalaxyGenerationConfig",
    "Llama33_70BGalaxyLayerWeights",
    "Llama33_70BGalaxyLazyLayerWeights",
    "Llama33_70BGalaxyLazyWeights",
    "Llama33_70BGalaxyModelParameters",
    "Llama33_70BGalaxyPrecision",
    "Llama33_70BGalaxyRuntimeConfig",
    "Llama33_70BGalaxyTransformer2D",
    "Llama33_70BGalaxyTransformer2DConfig",
    "Llama33_70BGalaxyWeights",
    "Llama33_70BTransformerBlock2D",
    "build_llama33_70b_galaxy_lazy_weights",
    "build_llama33_70b_galaxy_model",
    "build_llama33_70b_galaxy_transformer_2d_config",
    "convert_hf_model_weights",
    "default_paged_attention_config",
    "from_pretrained",
    "load_tokenizer",
    "parameters_from_hf_config",
    "validate_llama33_70b_checkpoint",
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
