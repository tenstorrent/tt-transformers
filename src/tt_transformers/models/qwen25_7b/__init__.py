# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    'QWEN25_7B_ACCURACY': ('tt_transformers.models.qwen25_7b.model', 'QWEN25_7B_ACCURACY'),
    'QWEN25_7B_PERFORMANCE': ('tt_transformers.models.qwen25_7b.model', 'QWEN25_7B_PERFORMANCE'),
    'Qwen25_7B': ('tt_transformers.models.qwen25_7b.model', 'Qwen25_7B'),
    'Qwen25_7BPrecisionConfig': ('tt_transformers.models.qwen25_7b.model', 'Qwen25_7BPrecisionConfig'),
    'Qwen25_7BTransformerConfig': ('tt_transformers.models.qwen25_7b.model', 'Qwen25_7BTransformerConfig'),
    'Qwen25Executor': ('tt_transformers.models.qwen25_7b.executor', 'Qwen25Executor'),
    'Qwen25ExecutorConfig': ('tt_transformers.models.qwen25_7b.executor', 'Qwen25ExecutorConfig'),
    'Qwen25ForCausalLM': ('tt_transformers.models.qwen25_7b.hf_adaptor', 'Qwen25ForCausalLM'),
    'Qwen25PagedAttentionConfig': ('tt_transformers.models.qwen25_7b.model', 'Qwen25PagedAttentionConfig'),
    'Qwen25Generator': ('tt_transformers.models.qwen25_7b.generator', 'Qwen25Generator'),
    'Qwen25GeneratorConfig': ('tt_transformers.models.qwen25_7b.generator', 'Qwen25GeneratorConfig'),
    'Qwen25RuntimeConfig': ('tt_transformers.models.qwen25_7b.hf_adaptor', 'Qwen25RuntimeConfig'),
}

__all__ = [
    'QWEN25_7B_ACCURACY',
    'QWEN25_7B_PERFORMANCE',
    'Qwen25_7B',
    'Qwen25_7BPrecisionConfig',
    'Qwen25_7BTransformerConfig',
    'Qwen25Executor',
    'Qwen25ExecutorConfig',
    'Qwen25ForCausalLM',
    'Qwen25PagedAttentionConfig',
    'Qwen25Generator',
    'Qwen25GeneratorConfig',
    'Qwen25RuntimeConfig',
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
