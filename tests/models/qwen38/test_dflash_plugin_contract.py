# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host proofs for Qwen's standard vLLM speculative-decode boundary."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm", reason="vLLM is supplied by the schema-6 serving integration")
pytest.importorskip(
    "vllm_tt_plugin.spec_decode",
    reason="vLLM TT plugin is supplied by the schema-6 serving integration",
)

from vllm_tt_plugin.spec_decode import DraftOutput, SpecPlan, SpecReject, VerifyOutput

from tt_transformers.models.qwen38 import dflash2_serving as serving
from tt_transformers.models.qwen38 import qwen36_vllm_dflash as adapter
from tt_transformers.models.qwen38.dflash2_serving import Bucket

_VOCAB = 4


class _Target:
    def __init__(self):
        self.calls = []
        self.hidden = object()
        self.logits = None

    def verify_traced(self, tokens, positions, mi_prev, **kwargs):
        self.calls.append((tokens, positions, mi_prev, kwargs))
        rows = sum(len(row) for row in tokens)
        ids = [12, 13, 99, 100] * (rows // 4)
        if not kwargs.get("read_logits"):
            return ids, self.hidden, None
        # Every value names its bucket row and vocabulary column, and bf16 holds these small integers exactly.
        self.logits = torch.arange(rows * _VOCAB, dtype=torch.bfloat16).reshape(rows, _VOCAB)
        return ids, self.hidden, self.logits


class _Drafter:
    def __init__(self):
        self.extends = []
        self.drafts = []

    def extend_context(self, slot0, nrows, **kwargs):
        self.extends.append((list(slot0), list(nrows), kwargs))

    def draft(self, anchors, contexts, **kwargs):
        self.drafts.append((list(anchors), list(contexts), kwargs))
        return [[31, 32, 33]]


def _contract_decoder(rows=(0,)):
    """A decoder whose bucket row r seats physical slot ``rows[r]``; every slot is active at position 10."""
    n = len(rows)
    dec = object.__new__(serving.DFlash2DualBucketDecoder)
    dec._captured = True
    dec._contract_pending = None
    dec.cur = Bucket(
        id=f"{n}x4",
        B=n,
        T=4,
        K=3,
        identity=list(rows) == list(range(n)),
        rows=list(rows),
        tables=torch.zeros(n, 1, dtype=torch.int32),
    )
    dec.model = _Target()
    dec.drafter = _Drafter()
    dec.vision_context = [None] * n
    dec.active = [True] * n
    dec.fresh = [False] * n
    dec.p = [9] * n
    dec.pending = [11] * n
    dec.mi = [0] * n
    dec.ctx_len = [10] * n
    dec.iters = [0] * n
    dec.accepted = [0] * n
    dec.drafted = [0] * n
    dec.committed = [0] * n
    dec.hist = [[0] * 4 for _ in range(n)]
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

    argmax, hidden, logits = dec.verify_contract(tokens, positions, torch.tensor([3], dtype=torch.int32), [0])

    assert hidden is dec.model.hidden
    assert argmax.tolist() == [[12, 13, 99, 100]]
    assert logits is None
    assert dec.model.calls[0][3]["read_logits"] is False
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


@pytest.mark.host
@pytest.mark.model
def test_logits_verify_puts_each_bucket_row_at_its_logical_row():
    # Logical row 0 is slot 0, which sits on bucket row 1, so a row mix-up swaps the two rows' logits.
    dec = _contract_decoder(rows=(1, 0))
    tokens = torch.tensor([[11, 12, 13, 14, -1, -1, -1, -1]] * 2, dtype=torch.int32)
    positions = torch.tensor([[10, 11, 12, 13, -1, -1, -1, -1]] * 2, dtype=torch.int32)

    argmax, hidden, logits = dec.verify_contract(
        tokens, positions, torch.tensor([3, 3], dtype=torch.int32), [0, 1], read_logits=True
    )

    bucket = dec.model.logits
    assert dec.model.calls[0][3]["read_logits"] is True
    assert hidden is dec.model.hidden
    assert argmax.shape == (2, 8)
    assert logits.shape == (2, 8, _VOCAB)
    assert logits.dtype == bucket.dtype
    assert torch.equal(logits[0, :4], bucket[4:8])
    assert torch.equal(logits[1, :4], bucket[0:4])
    # Columns past the bucket's T are padding the accept walk never reads; zeros keep a whole-block cast finite.
    assert not logits[:, 4:].any()
    # A verify never advances authoritative state, whichever mode it reads back.
    assert dec.p == [9, 9]
    assert dec.pending == [11, 11]


class _ContractDecoder:
    def __init__(self, slots):
        self.active = [False] * slots
        self.mi = [0] * slots
        self.calls = []
        self.hidden = object()
        self.logits = torch.ones(slots, 8, 16)

    def plan(self, live_after):
        self.calls.append(("plan", frozenset(live_after)))
        return "1x4"

    def begin(self, phys, first, prompt_len, row, seed_slot=None, contract_seed=False):
        self.calls.append(("begin", phys, first, prompt_len, seed_slot, contract_seed))
        self.active[phys] = True

    def set_table(self, phys, row):
        self.calls.append(("set_table", phys))

    def verify_contract(self, tokens, positions, valid, phys_by_row, read_logits=False):
        self.calls.append(("verify", list(phys_by_row), read_logits))
        out = torch.zeros(tokens.shape, dtype=torch.int32)
        out[0, :4] = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
        return out, self.hidden, self.logits if read_logits else None

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
        ("verify", [0, None, None, None], False),
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
    assert plan.accept_modes == ("argmax_ids", "logits")
    assert plan.supports_narrow_decode is False
    assert plan.k_by_rows == ((4, 7), (8, 3))

    rejected = adapter.Qwen36DFlashForCausalLM.spec_plan(None, 8, 3)
    assert isinstance(rejected, SpecReject)
    assert rejected.supported_k == (7,)


def _first_verify_inputs(slots=4):
    tokens = torch.zeros(slots, 8, dtype=torch.int32)
    positions = torch.full((slots, 8), -1, dtype=torch.int32)
    tokens[0, 0] = 6
    positions[0] = torch.arange(10, 18, dtype=torch.int32)
    return dict(
        tokens=tokens,
        start_pos=positions,
        page_table=torch.zeros(slots, 8, dtype=torch.int32),
        num_valid_drafts=torch.zeros(slots, dtype=torch.int32),
        accepted_counts=torch.ones(slots, dtype=torch.int32),
    )


@pytest.mark.host
@pytest.mark.model
def test_adapter_returns_the_verify_logits_in_logits_mode():
    dec = _ContractDecoder(4)
    obj = _adapter(dec)
    obj._pending[0] = (10, torch.zeros(8, dtype=torch.int32), 0)

    verified = obj.decode_forward(**_first_verify_inputs(), spec_mode="logits")

    assert isinstance(verified, VerifyOutput)
    assert verified.spec_mode == "logits"
    assert verified.logits is dec.logits
    assert verified.argmax_ids is None
    assert verified.hidden is dec.hidden
    assert dec.calls[2] == ("verify", [0, None, None, None], True)


@pytest.mark.host
@pytest.mark.model
def test_adapter_refuses_a_decode_without_spec_mode(expect_error):
    dec = _ContractDecoder(4)
    obj = _adapter(dec)
    obj._pending[0] = (10, torch.zeros(8, dtype=torch.int32), 0)
    inputs = _first_verify_inputs()

    with expect_error(RuntimeError, "without spec_mode"):
        obj.decode_forward(**inputs)
    with expect_error(RuntimeError, "got 'fused_sample'"):
        obj.decode_forward(**inputs, spec_mode="fused_sample")
    # Neither refusal seeded the request or reached the decoder.
    assert dec.calls == []
    assert obj._pending[0] is not None
