# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    'Qwen3_32B': ('tt_transformers.models.qwen3_32b.model', 'Qwen3_32B'),
    'Qwen3_32BConfig': ('tt_transformers.models.qwen3_32b.model', 'Qwen3_32BConfig'),
    'Qwen3_32BExecutorRuntimeConfig': ('tt_transformers.models.qwen3_32b.model', 'Qwen3_32BExecutorRuntimeConfig'),
    'Qwen3_32BPagedAttentionConfig': ('tt_transformers.models.qwen3_32b.model', 'Qwen3_32BPagedAttentionConfig'),
    'Qwen3_32BPrecisionConfig': ('tt_transformers.models.qwen3_32b.model', 'Qwen3_32BPrecisionConfig'),
    'QWEN3_32B_ACCURACY': ('tt_transformers.models.qwen3_32b.model', 'QWEN3_32B_ACCURACY'),
    'QWEN3_32B_PERFORMANCE': ('tt_transformers.models.qwen3_32b.model', 'QWEN3_32B_PERFORMANCE'),
    'Qwen3_32BGenerator': ('tt_transformers.models.qwen3_32b.generator', 'Qwen3_32BGenerator'),
    'Qwen3_32BGeneratorConfig': ('tt_transformers.models.qwen3_32b.generator', 'Qwen3_32BGeneratorConfig'),
    'EagerQwen3_32BExecutor': ('tt_transformers.models.qwen3_32b.executor', 'EagerQwen3_32BExecutor'),
    'TracedQwen3_32BExecutor': ('tt_transformers.models.qwen3_32b.executor', 'TracedQwen3_32BExecutor'),
}

__all__ = [
    'Qwen3_32B',
    'Qwen3_32BConfig',
    'Qwen3_32BExecutorRuntimeConfig',
    'Qwen3_32BPagedAttentionConfig',
    'Qwen3_32BPrecisionConfig',
    'QWEN3_32B_ACCURACY',
    'QWEN3_32B_PERFORMANCE',
    'Qwen3_32BGenerator',
    'Qwen3_32BGeneratorConfig',
    'EagerQwen3_32BExecutor',
    'TracedQwen3_32BExecutor',
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
