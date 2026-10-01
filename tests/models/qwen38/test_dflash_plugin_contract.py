# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host proofs for Qwen's standard vLLM speculative-decode boundary."""

from types import SimpleNamespace

import pytest
import torch
from vllm_tt_plugin.spec_decode import DraftOutput, SpecPlan, SpecReject, VerifyOutput

from tt_transformers.models.qwen38 import dflash2_serving as serving
from tt_transformers.models.qwen38 import qwen36_vllm_dflash as adapter
from tt_transformers.models.qwen38.dflash2_serving import Bucket


class _Target:
    def __init__(self):
        self.calls = []
        self.hidden = object()

    def verify_traced(self, tokens, positions, mi_prev, **kwargs):
        self.calls.append((tokens, positions, mi_prev, kwargs))
        return [12, 13, 99, 100], self.hidden, None


class _Drafter:
    def __init__(self):
        self.extends = []
        self.drafts = []

    def extend_context(self, slot0, nrows, **kwargs):
        self.extends.append((list(slot0), list(nrows), kwargs))

    def draft(self, anchors, contexts, **kwargs):
        self.drafts.append((list(anchors), list(contexts), kwargs))
        return [[31, 32, 33]]


def _contract_decoder():
    dec = object.__new__(serving.DFlash2DualBucketDecoder)
    dec._captured = True
    dec._contract_pending = None
    dec.cur = Bucket(
        id="1x4",
        B=1,
        T=4,
        K=3,
        identity=True,
        rows=[0],
        tables=torch.zeros(1, 1, dtype=torch.int32),
    )
    dec.model = _Target()
    dec.drafter = _Drafter()
    dec.vision_context = [None]
    dec.active = [True]
    dec.fresh = [False]
    dec.p = [9]
    dec.pending = [11]
    dec.mi = [0]
    dec.ctx_len = [10]
    dec.iters = [0]
    dec.accepted = [0]
    dec.drafted = [0]
    dec.committed = [0]
    dec.hist = [[0] * 4]
    dec.total_steps = 0
    dec._armed = True
    dec._planned = True
    dec._switched_since_step = False
    return dec


@pytest.mark.host
@pytest.mark.model
def test_verify_then_commit_and_propose_preserves_the_state_boundary():
    dec = _contract_decoder()
    tokens = torch.tensor([[11, 12, 13, 14]], dtype=torch.int32)
    positions = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)

    argmax, hidden = dec.verify_contract(tokens, positions, torch.tensor([3], dtype=torch.int32), [0])

    assert hidden is dec.model.hidden
    assert argmax.tolist() == [[12, 13, 99, 100]]
    # Verify alone never advances authoritative host state.
    assert dec.p == [9]
    assert dec.pending == [11]

    committed = torch.tensor([[12, 13, 99, -1, -1, -1, -1, -1]], dtype=torch.int32)
    drafts, num_valid = dec.commit_and_draft_contract(
        committed, torch.tensor([3], dtype=torch.int32), requested_k=7, output_rows=1
    )

    assert dec.mi == [2]
    assert dec.p == [12]
    assert dec.ctx_len == [13]
    assert dec.pending == [99]
    assert dec.drafter.extends[0][:2] == ([10], [3])
    assert dec.drafter.drafts[0][:2] == ([99], [13])
    assert drafts.tolist() == [[31, 32, 33, 0, 0, 0, 0]]
    assert num_valid.tolist() == [3]
    assert dec._contract_pending is None


class _ContractDecoder:
    def __init__(self, slots):
        self.active = [False] * slots
        self.mi = [0] * slots
        self.calls = []
        self.hidden = object()

    def plan(self, live_after):
        self.calls.append(("plan", frozenset(live_after)))
        return "1x4"

    def begin(self, phys, first, prompt_len, row, seed_slot=None, contract_seed=False):
        self.calls.append(("begin", phys, first, prompt_len, seed_slot, contract_seed))
        self.active[phys] = True

    def set_table(self, phys, row):
        self.calls.append(("set_table", phys))

    def verify_contract(self, tokens, positions, valid, phys_by_row):
        self.calls.append(("verify", list(phys_by_row)))
        out = torch.zeros(tokens.shape, dtype=torch.int32)
        out[0, :4] = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
        return out, self.hidden

    def commit_and_draft_contract(self, committed, counts, requested_k, output_rows):
        self.calls.append(("propose", requested_k, output_rows))
        drafts = torch.zeros(output_rows, requested_k, dtype=torch.int32)
        drafts[0, :3] = torch.tensor([21, 22, 23], dtype=torch.int32)
        valid = torch.zeros(output_rows, dtype=torch.int32)
        valid[0] = 3
        return drafts, valid


def _adapter(decoder, slots=4):
    obj = adapter.Qwen36DFlashForCausalLM.__new__(adapter.Qwen36DFlashForCausalLM)
    obj._dflash = adapter.DFlashRuntimeConfig.from_env()
    obj._spec, obj._spec_pre, obj._in_warmup = decoder, None, False
    obj.data_parallel = 1
    obj.model = [SimpleNamespace(vocab_size=248320)]
    obj._eos, obj._eos_fill = {151645}, 151645
    obj._B = slots
    obj._phys = list(range(slots))
    obj._pending = [None] * slots
    obj._ordinary_state_slots = set()
    obj._carry = [[] for _ in range(slots)]
    obj._stopped = [False] * slots
    obj._prev_tail = [None] * slots
    obj._pending_state_slot_moves = None
    obj._anchor_warned = obj._oov_warned = obj._nosession_warned = False
    obj._buckets, obj._multi_bucket, obj._last_bucket = ((1, 4), (slots, 4)), True, None
    return obj


@pytest.mark.host
@pytest.mark.model
def test_adapter_returns_standard_verify_and_draft_outputs():
    dec = _ContractDecoder(4)
    obj = _adapter(dec)
    obj._pending[0] = (10, torch.zeros(8, dtype=torch.int32), 0)
    tokens = torch.zeros(4, 8, dtype=torch.int32)
    positions = torch.full((4, 8), -1, dtype=torch.int32)
    tokens[0, 0] = 6
    positions[0] = torch.arange(10, 18, dtype=torch.int32)

    verified = obj.decode_forward(
        tokens=tokens,
        start_pos=positions,
        page_table=torch.zeros(4, 8, dtype=torch.int32),
        num_valid_drafts=torch.zeros(4, dtype=torch.int32),
        accepted_counts=torch.ones(4, dtype=torch.int32),
        spec_mode="argmax_ids",
    )

    assert isinstance(verified, VerifyOutput)
    assert verified.argmax_ids.shape == (4, 8)
    assert verified.argmax_ids[0, 4:].tolist() == [0, 0, 0, 0]
    assert verified.hidden is dec.hidden
    assert dec.calls[:3] == [
        ("plan", frozenset({0})),
        ("begin", 0, 6, 10, 0, True),
        ("verify", [0, None, None, None]),
    ]

    drafted = obj.propose_draft_tokens(
        7,
        torch.tensor([[7] + [-1] * 7] + [[-1] * 8] * 3, dtype=torch.int32),
        torch.zeros(4, 8, dtype=torch.int32),
        torch.ones(4, dtype=torch.int32),
    )
    assert isinstance(drafted, DraftOutput)
    assert drafted.draft_token_ids.shape == (4, 7)
    assert drafted.num_valid.tolist() == [3, 0, 0, 0]


@pytest.mark.host
@pytest.mark.model
def test_spec_plan_publishes_the_widest_exact_bucket(monkeypatch):
    monkeypatch.setenv("QWEN36_DRAFTER", "dflash2")
    monkeypatch.setenv("QWEN36_DFLASH_SERVE_BLOCK", "32")
    monkeypatch.setenv("QWEN36_DFLASH_BUCKETS", "1x8,4x8,8x4")

    plan = adapter.Qwen36DFlashForCausalLM.spec_plan(None, 8, 7)
    assert isinstance(plan, SpecPlan)
    assert plan.effective_k == 7
    assert plan.supports_narrow_decode is True
    assert plan.k_by_rows == ((4, 7), (8, 3))

    rejected = adapter.Qwen36DFlashForCausalLM.spec_plan(None, 8, 3)
    assert isinstance(rejected, SpecReject)
    assert rejected.supported_k == (7,)
