# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The focused test behind the one common-runtime change this job makes.

`WarmupConfig.cached_prefill_tokens` exists because the runtime's warmup plan
fixed the prefix of its cached-prefill coverage case at one page-table block,
and a model whose attention refuses a chunk start that is not a multiple of a
larger alignment cannot serve that case at all (D-C16: `chunk_start must be
non-negative and aligned to chunk_alignment`, `attention_2d.py:908`, three fresh
processes in `c-defects`' `logs/y1_exec_warmup_df_r1.log`).

The field is a token count. It names no model, no architecture, no mesh shape
and no topology, and leaving it unset reproduces the previous plan byte for
byte — which is what the first test here asserts.

Host only: no device, no checkpoint.
"""

from __future__ import annotations

import pytest

from tt_transformers.llm_runtime.config import PageTableLayout, WarmupConfig
from tt_transformers.llm_runtime.warmup import _build_plan

_LAYOUT = PageTableLayout.resolve(
    block_size=32,
    model_max_sequence_length=2048,
    physical_num_blocks=2048,
    max_prefill_chunk_size=2048,
)


def _plan(warmup: WarmupConfig):
    return _build_plan(
        warmup=warmup,
        layout=_LAYOUT,
        prefill_sequence_lengths=(128,),
        lane_batch_size=32,
        allow_force_argmax=False,
        can_sample_on_device=False,
    )


def _cached_cases(plan):
    return [case for case in plan.prefill if case.cached_tokens]


@pytest.mark.host
@pytest.mark.model
def test_unset_cached_prefill_tokens_preserves_the_block_sized_prefix() -> None:
    """The default must be exactly what every caller resolved to before."""

    plan = _plan(WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)))
    cases = _cached_cases(plan)
    assert cases, "the warmup plan lost its cached-prefill coverage case"
    assert {case.cached_tokens for case in cases} == {_LAYOUT.block_size}


@pytest.mark.host
@pytest.mark.model
def test_a_model_can_ask_for_a_prefix_its_attention_can_actually_resume_from() -> None:
    """A chunk-aligned prefix is expressible without any runtime branch."""

    alignment = 128
    plan = _plan(WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,), cached_prefill_tokens=alignment))
    cases = _cached_cases(plan)
    assert cases, "the warmup plan lost its cached-prefill coverage case"
    assert {case.cached_tokens for case in cases} == {alignment}
    assert all(case.cached_tokens % alignment == 0 for case in cases)


@pytest.mark.host
@pytest.mark.model
def test_the_plan_is_otherwise_untouched_by_the_new_field() -> None:
    """Only the cached case moves: every uncached case is identical."""

    default = _plan(WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)))
    aligned = _plan(WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,), cached_prefill_tokens=128))
    assert [case for case in default.prefill if not case.cached_tokens] == [
        case for case in aligned.prefill if not case.cached_tokens
    ]
    assert default.decode == aligned.decode


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_cached_prefill_tokens_rejects_a_value_that_is_not_a_positive_count(value, expect_error) -> None:
    with expect_error(ValueError, "cached_prefill_tokens"):
        WarmupConfig(cached_prefill_tokens=value)
