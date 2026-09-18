# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DEFAULT_HF_MODEL": ("tt_transformers.models.phi4.hf_generator", "DEFAULT_HF_MODEL"),
    "DEFAULT_HF_REVISION": ("tt_transformers.models.phi4.hf_generator", "DEFAULT_HF_REVISION"),
    "PHI4_ACCURACY": ("tt_transformers.models.phi4.model", "PHI4_ACCURACY"),
    "PHI4_PERFORMANCE": ("tt_transformers.models.phi4.model", "PHI4_PERFORMANCE"),
    "Phi4Transformer": ("tt_transformers.models.phi4.model", "Phi4Transformer"),
    "Phi4TransformerConfig": ("tt_transformers.models.phi4.model", "Phi4TransformerConfig"),
    "Phi4PagedAttentionConfig": ("tt_transformers.models.phi4.model", "Phi4PagedAttentionConfig"),
    "Phi4PrecisionConfig": ("tt_transformers.models.phi4.model", "Phi4PrecisionConfig"),
    "Phi4ForCausalLM": ("tt_transformers.models.phi4.hf_generator", "Phi4ForCausalLM"),
    "Phi4RuntimeConfig": ("tt_transformers.models.phi4.hf_generator", "Phi4RuntimeConfig"),
    "Phi4Executor": ("tt_transformers.models.phi4.hf_generator", "Phi4Executor"),
    "Phi4ExecutorConfig": ("tt_transformers.models.phi4.hf_generator", "Phi4ExecutorConfig"),
    "Phi4Generator": ("tt_transformers.models.phi4.vllm_generator", "Phi4Generator"),
    "Phi4GeneratorConfig": ("tt_transformers.models.phi4.vllm_generator", "Phi4GeneratorConfig"),
    "build_phi4_executor": ("tt_transformers.models.phi4.hf_generator", "build_phi4_executor"),
    "build_phi4_generator": ("tt_transformers.models.phi4.vllm_generator", "build_phi4_generator"),
    "build_phi4_transformer_config": ("tt_transformers.models.phi4.model", "build_phi4_transformer_config"),
}

__all__ = [
    "DEFAULT_HF_MODEL",
    "DEFAULT_HF_REVISION",
    "PHI4_ACCURACY",
    "PHI4_PERFORMANCE",
    "Phi4Transformer",
    "Phi4TransformerConfig",
    "Phi4PagedAttentionConfig",
    "Phi4PrecisionConfig",
    "Phi4ForCausalLM",
    "Phi4RuntimeConfig",
    "Phi4Executor",
    "Phi4ExecutorConfig",
    "Phi4Generator",
    "Phi4GeneratorConfig",
    "build_phi4_executor",
    "build_phi4_generator",
    "build_phi4_transformer_config",
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
