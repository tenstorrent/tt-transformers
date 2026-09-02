# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    'DEEPSEEK_R1_14B_ACCURACY': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.model', 'DEEPSEEK_R1_14B_ACCURACY'),
    'DEEPSEEK_R1_14B_PERFORMANCE': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.model', 'DEEPSEEK_R1_14B_PERFORMANCE'),
    'DeepSeekR1Qwen14B': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.model', 'DeepSeekR1Qwen14B'),
    'DeepSeekR1Qwen14BPrecisionConfig': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.model', 'DeepSeekR1Qwen14BPrecisionConfig'),
    'DeepSeekR1Qwen14BTransformerConfig': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.model', 'DeepSeekR1Qwen14BTransformerConfig'),
    'DeepSeekR1Qwen14BExecutor': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.executor', 'DeepSeekR1Qwen14BExecutor'),
    'DeepSeekR1Qwen14BExecutorConfig': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.executor', 'DeepSeekR1Qwen14BExecutorConfig'),
    'DeepSeekR1Qwen14BForCausalLM': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.hf_adaptor', 'DeepSeekR1Qwen14BForCausalLM'),
    'DeepSeekR1Qwen14BPagedAttentionConfig': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.model', 'DeepSeekR1Qwen14BPagedAttentionConfig'),
    'DeepSeekR1Qwen14BGenerator': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.generator', 'DeepSeekR1Qwen14BGenerator'),
    'DeepSeekR1Qwen14BGeneratorConfig': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.generator', 'DeepSeekR1Qwen14BGeneratorConfig'),
    'DeepSeekR1Qwen14BRuntimeConfig': ('tt_transformers.models.deepseek_r1_distill_qwen_14b.hf_adaptor', 'DeepSeekR1Qwen14BRuntimeConfig'),
}

__all__ = [
    'DEEPSEEK_R1_14B_ACCURACY',
    'DEEPSEEK_R1_14B_PERFORMANCE',
    'DeepSeekR1Qwen14B',
    'DeepSeekR1Qwen14BPrecisionConfig',
    'DeepSeekR1Qwen14BTransformerConfig',
    'DeepSeekR1Qwen14BExecutor',
    'DeepSeekR1Qwen14BExecutorConfig',
    'DeepSeekR1Qwen14BForCausalLM',
    'DeepSeekR1Qwen14BPagedAttentionConfig',
    'DeepSeekR1Qwen14BGenerator',
    'DeepSeekR1Qwen14BGeneratorConfig',
    'DeepSeekR1Qwen14BRuntimeConfig',
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
