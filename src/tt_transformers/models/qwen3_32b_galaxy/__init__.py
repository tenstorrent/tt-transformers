# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DEFAULT_HF_MODEL": ("tt_transformers.models.qwen3_32b_galaxy.hf_adaptor", "DEFAULT_HF_MODEL"),
    "DEFAULT_HF_REVISION": ("tt_transformers.models.qwen3_32b_galaxy.model", "DEFAULT_HF_REVISION"),
    "QWEN3_32B_CHECKPOINT_CONTRACT": ("tt_transformers.models.qwen3_32b_galaxy.model", "QWEN3_32B_CHECKPOINT_CONTRACT"),
    "QWEN3_32B_GALAXY_ACCURACY": ("tt_transformers.models.qwen3_32b_galaxy.model", "QWEN3_32B_GALAXY_ACCURACY"),
    "QWEN3_32B_GALAXY_HF_MODEL": ("tt_transformers.models.qwen3_32b_galaxy.model", "QWEN3_32B_GALAXY_HF_MODEL"),
    "QWEN3_32B_GALAXY_PERFORMANCE": ("tt_transformers.models.qwen3_32b_galaxy.model", "QWEN3_32B_GALAXY_PERFORMANCE"),
    "QWEN3_32B_PREFETCHED_WEIGHT_NAMES": (
        "tt_transformers.models.qwen3_32b_galaxy.model",
        "QWEN3_32B_PREFETCHED_WEIGHT_NAMES",
    ),
    "Qwen3_32BGalaxyBlockConfig": ("tt_transformers.models.qwen3_32b_galaxy.model", "Qwen3_32BGalaxyBlockConfig"),
    "Qwen3_32BGalaxyForCausalLM": ("tt_transformers.models.qwen3_32b_galaxy.hf_adaptor", "Qwen3_32BGalaxyForCausalLM"),
    "Qwen3_32BGalaxyGenerationConfig": (
        "tt_transformers.models.qwen3_32b_galaxy.hf_adaptor",
        "Qwen3_32BGalaxyGenerationConfig",
    ),
    "Qwen3_32BGalaxyLayerWeights": ("tt_transformers.models.qwen3_32b_galaxy.model", "Qwen3_32BGalaxyLayerWeights"),
    "Qwen3_32BGalaxyLazyLayerWeights": (
        "tt_transformers.models.qwen3_32b_galaxy.model",
        "Qwen3_32BGalaxyLazyLayerWeights",
    ),
    "Qwen3_32BGalaxyLazyWeights": ("tt_transformers.models.qwen3_32b_galaxy.model", "Qwen3_32BGalaxyLazyWeights"),
    "Qwen3_32BGalaxyModelParameters": (
        "tt_transformers.models.qwen3_32b_galaxy.model",
        "Qwen3_32BGalaxyModelParameters",
    ),
    "Qwen3_32BGalaxyPrecision": ("tt_transformers.models.qwen3_32b_galaxy.model", "Qwen3_32BGalaxyPrecision"),
    "Qwen3_32BGalaxyRuntimeConfig": (
        "tt_transformers.models.qwen3_32b_galaxy.hf_adaptor",
        "Qwen3_32BGalaxyRuntimeConfig",
    ),
    "Qwen3_32BGalaxyTransformer2D": ("tt_transformers.models.qwen3_32b_galaxy.model", "Qwen3_32BGalaxyTransformer2D"),
    "Qwen3_32BGalaxyTransformer2DConfig": (
        "tt_transformers.models.qwen3_32b_galaxy.model",
        "Qwen3_32BGalaxyTransformer2DConfig",
    ),
    "Qwen3_32BGalaxyWeights": ("tt_transformers.models.qwen3_32b_galaxy.model", "Qwen3_32BGalaxyWeights"),
    "Qwen3_32BTransformerBlock2D": ("tt_transformers.models.qwen3_32b_galaxy.model", "Qwen3_32BTransformerBlock2D"),
    "build_qwen3_32b_galaxy_lazy_weights": (
        "tt_transformers.models.qwen3_32b_galaxy.model",
        "build_qwen3_32b_galaxy_lazy_weights",
    ),
    "build_qwen3_32b_galaxy_model": ("tt_transformers.models.qwen3_32b_galaxy.model", "build_qwen3_32b_galaxy_model"),
    "build_qwen3_32b_galaxy_transformer_2d_config": (
        "tt_transformers.models.qwen3_32b_galaxy.model",
        "build_qwen3_32b_galaxy_transformer_2d_config",
    ),
    "convert_hf_model_weights": ("tt_transformers.models.qwen3_32b_galaxy.hf_adaptor", "convert_hf_model_weights"),
    "default_paged_attention_config": (
        "tt_transformers.models.qwen3_32b_galaxy.model",
        "default_paged_attention_config",
    ),
    "from_pretrained": ("tt_transformers.models.qwen3_32b_galaxy.hf_adaptor", "from_pretrained"),
    "load_tokenizer": ("tt_transformers.models.qwen3_32b_galaxy.hf_adaptor", "load_tokenizer"),
    "parameters_from_hf_config": ("tt_transformers.models.qwen3_32b_galaxy.model", "parameters_from_hf_config"),
    "validate_qwen3_32b_checkpoint": ("tt_transformers.models.qwen3_32b_galaxy.model", "validate_qwen3_32b_checkpoint"),
}

__all__ = [
    "DEFAULT_HF_MODEL",
    "DEFAULT_HF_REVISION",
    "QWEN3_32B_CHECKPOINT_CONTRACT",
    "QWEN3_32B_GALAXY_ACCURACY",
    "QWEN3_32B_GALAXY_HF_MODEL",
    "QWEN3_32B_GALAXY_PERFORMANCE",
    "QWEN3_32B_PREFETCHED_WEIGHT_NAMES",
    "Qwen3_32BGalaxyBlockConfig",
    "Qwen3_32BGalaxyForCausalLM",
    "Qwen3_32BGalaxyGenerationConfig",
    "Qwen3_32BGalaxyLayerWeights",
    "Qwen3_32BGalaxyLazyLayerWeights",
    "Qwen3_32BGalaxyLazyWeights",
    "Qwen3_32BGalaxyModelParameters",
    "Qwen3_32BGalaxyPrecision",
    "Qwen3_32BGalaxyRuntimeConfig",
    "Qwen3_32BGalaxyTransformer2D",
    "Qwen3_32BGalaxyTransformer2DConfig",
    "Qwen3_32BGalaxyWeights",
    "Qwen3_32BTransformerBlock2D",
    "build_qwen3_32b_galaxy_lazy_weights",
    "build_qwen3_32b_galaxy_model",
    "build_qwen3_32b_galaxy_transformer_2d_config",
    "convert_hf_model_weights",
    "default_paged_attention_config",
    "from_pretrained",
    "load_tokenizer",
    "parameters_from_hf_config",
    "validate_qwen3_32b_checkpoint",
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
