# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.x on Blackhole served by vLLM with the DFlash2 speculative drafter.

The default integration uses the TT plugin's speculative-decoding contract,
not adaptive block output. The launch supplies vLLM ``speculative_config``
with the model-owned ``custom_class`` drafter. The plugin owns acceptance,
committed output, request history, and KV lookahead; this adapter owns exact
target verification plus the drafter's target/GDN/ring state:

  * prefill emits one sampled anchor and records request-local DFlash state;
  * every decode step is a verify of one candidate block, also for a row with
    no drafts (``supports_narrow_decode=False``). A plain decode cannot
    continue an active request, because its GDN state lives only in the
    speculative ring;
  * ``spec_mode="argmax_ids"`` returns the target argmax ids, which certify
    only greedy rows. ``spec_mode="logits"`` returns the target logits of
    every candidate column, and the plugin samples each row with its own
    sampling controls. Neither mode advances authoritative state;
  * after the plugin accepts a prefix, ``propose_draft_tokens`` commits the
    matching candidate state, extends the drafter, and returns ``DraftOutput``.

The older block-output methods remain temporarily for compatibility with the
banked v5.1 artifact, but the v6 launch does not advertise or select that rail.

Greedy: each request's tokens are the target's greedy trajectory from its sampled anchor (lossless,
tested per request in tests/test_dflash2_serving.py). B_max x (K+1) verify rows must fit one 32-row
decode tile (K = 7 for DFlash2 up to 4 slots, K = 3 at 8). QWEN36_DRAFTER=mtp (or
QWEN36_DFLASH_SERVE_BLOCK=1) turns speculation off and this class serves exactly like
Qwen36ForCausalLM (plain batched decode), so one bundle covers both profiles.

VERIFY BUCKETS (QWEN36_DFLASH_BUCKETS, e.g. '8x4,4x8'; see profiles/dual_bucket_spec.json): one server keeps
several (B_cfg x T_cfg) verify geometries captured and verifies in the smallest bucket that seats the live
requests -- K = 7 while <= 4 requests are live, K = 3 at 5..8 -- switching at runtime with NO recapture.
The decoder moves every live session's GDN state between buckets bit-exactly. ``spec_plan`` advertises
the maximum K=7 plus ``k_by_rows=((4, 7), (8, 3))``. The plugin truncates any outstanding proposal
before a busy-bucket verify, and the adapter pads its exact four-column result to the launch width;
the accept walk never reads the suffix. ``decode_forward`` plans the bucket before any begin/set-table/
verify operation. Unset selects one ``max_num_seqs x (K+1)`` bucket.

Slots: the plugin keys a request's device state by a "state slot" and may permute decode rows
(``slot_remap``: row i reads slot remap[i]); the base class moves its GDN state accordingly. The
speculative session state (GDN ring, drafter ring, tables) is NEVER moved: this class keeps a
row -> physical-session indirection and composes the plugin's remaps into it.

Warmup (server start): after the plain traces are captured, every spec buffer is allocated and every
spec program compiled (eager tap-capturing prefill for each mask bucket into a slot, the chunked spec
prefill shapes, every per-bucket seed / bucket-switch variant, the draft and extend of every bucket
layout), then the verify trace of EVERY bucket is captured ONCE and dummy sessions in every bucket
capture the drafter's draft/extend traces. Two guards then run before the first request: the program
cache must not grow over a repeat of the warm sweep + one traced step per bucket (a program compiling
under a parked trace clobbers it -- a device hang at request time), and the TRACE region must stay
under _TRACE_REGION_MAX_FRAC (an overflow only surfaces as a TT_FATAL at the next capture).
"""

import json
import os
import time
from dataclasses import dataclass

import torch
import ttnn  # noqa: E402
from loguru import logger
from vllm.model_executor.models.qwen3_5 import Qwen3VLDummyInputsBuilder  # noqa: E402
from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa: E402

from tt_transformers.models.qwen38.dflash2_decode import default_draft_len  # noqa: E402
from tt_transformers.models.qwen38.dflash2_serving import (  # noqa: E402
    DFlash2ServingDecoder,
    dummy_prompt,
    serve_block_size,
)
from tt_transformers.models.qwen38.qwen36_vllm import (  # noqa: E402
    Qwen36ForCausalLM,
    TT_Qwen3_5ProcessingInfo,
    TTQwen3VLMultiModalProcessor,
)
from tt_transformers.models.qwen38.spec_decode import MAX_SPEC_ROWS  # noqa: E402

try:  # the multi-bucket decoder (profiles/dual_bucket_spec.json); a single bucket falls back to today's class without it
    from tt_transformers.models.qwen38.dflash2_serving import BucketPlanner, DFlash2DualBucketDecoder  # noqa: E402
except ImportError:  # pragma: no cover - a tree without the dual-bucket decoder
    DFlash2DualBucketDecoder = None
    BucketPlanner = None

# Verify buckets (module doc): 'BxT[,BxT...]' -- B_cfg users x T_cfg verify rows per user (K = T-1 drafts). Unset:
# ONE bucket (max_num_seqs, default_draft_len()+1), today's single geometry.
_BUCKETS_ENV = "QWEN36_DFLASH_BUCKETS"
# Startup fails (multi-bucket; a warning on the single-bucket profiles) when the device TRACE region is fuller than
# this after every spec capture.
# Make the post-capture program-cache growth guard and the trace-region guard FATAL for ANY bucket count
# (the shipped default only fails startup for a multi-bucket profile; single-bucket profiles warn). A program
# compiling under a parked trace is a device-hang risk regardless of bucket count, so CI / the device tests set
# this to catch a regression on the batch4 / single-user profiles too.
# Stop handling inside a block. "1" (default, serving): a block that reaches a stop token is emitted at
# once, EOS-filled past the stop (the request ends here in normal use, so no time is spent decoding
# beyond it). "0" (benchmarks): stops are ordinary tokens, every block is W real committed tokens; vLLM
# still ends a normal request at its first stop, and an ignore_eos client is charged the real decode
# cost of every token it counts, exactly like the plain server. Never coast on free fills either way.
# Ragged block widths (see module doc): ONE dec.step() per decode step, rows padded with _PAD. Needs the
# plugin side of the contract (tt_adaptive_block_ragged); default off so the fixed-width plugin keeps working.
_PAD = -1  # ragged rows: padding id past a row's real tokens (the plugin counts non-negative ids)
# Prefill logits readout: the eager prefill returns a tile-padded [32, vocab] bf16 tensor replicated on
# every device; "1" (default) untilizes it ON DEVICE (drops the 31 padding rows) and reads one replica
# (~0.5 MB), "0" pulls every replica through ConcatMeshToTensor (~64 MB) as before. Same row either way.
# Prompts up to one chunk minus one token take the single eager MASKED-BUCKET prefill; longer prompts
# the eager CHUNKED spec prefill. Both go through prefill_for_spec and both capture the drafter taps.
_PREFILL_CHUNK = 2048
# Every DFlash checkpoint drafts at most block-1 <= 15 tokens; the KV lookahead only has to bound the
# verify's reach past the block (W committed positions + K+1 candidate rows).
_MAX_DRAFT = 15


@dataclass(frozen=True)
class DFlashRuntimeConfig:
    """Per-model execution policy, read when the server constructs the model.

    Keeping this out of module globals means two model instances do not inherit stale decisions
    from whichever profile imported this module first. The platform calls ``capabilities()`` at
    startup through ``get_model_capabilities``; the instance reads the same launch environment.
    """

    spec_on: bool
    output_width: int
    trace_region_max_frac: float
    guard_strict: bool
    stop_fill: bool
    ragged: bool
    fast_logits: bool
    debug: bool

    @classmethod
    def from_env(cls):
        spec_on = os.environ.get("QWEN36_DRAFTER", "dflash2") == "dflash2"
        output_width = serve_block_size() if spec_on else 1
        spec_on = spec_on and output_width > 1
        return cls(
            spec_on=spec_on,
            output_width=output_width if spec_on else 1,
            trace_region_max_frac=float(os.environ.get("QWEN36_DFLASH_TRACE_REGION_MAX_FRAC", "0.85")),
            guard_strict=os.environ.get("QWEN36_DFLASH_GUARD_STRICT", "0") == "1",
            stop_fill=os.environ.get("QWEN36_DFLASH_STOP_FILL", "1") != "0",
            ragged=os.environ.get("QWEN36_DFLASH_RAGGED", "0") == "1",
            fast_logits=os.environ.get("QWEN36_DFLASH_LOGITS_FAST", "1") != "0",
            debug=os.environ.get("QWEN36_DFLASH_DEBUG", "0") == "1",
        )

    def capabilities(self, base):
        if not self.spec_on:
            return dict(base)
        return {
            **base,
            "supports_spec_decode": True,
            "supports_async_spec_decode": False,
            "spec_requirements": ["device_propose"],
        }


def parse_buckets(spec):
    """``QWEN36_DFLASH_BUCKETS`` text ('8x4,4x8') -> ((8, 4), (4, 8)): (B_cfg, T_cfg) pairs in the order given;
    unset / empty -> (). Host-only, no model needed (the checks are ``check_buckets``)."""
    buckets = []
    for tok in (spec or "").replace(";", ",").split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        parts = [p.strip() for p in tok.split("x")]
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise RuntimeError(
                f"{_BUCKETS_ENV}: bucket {tok!r} is not of the form BxT (users x verify rows per user, e.g. 8x4,4x8)"
            )
        buckets.append((int(parts[0]), int(parts[1])))
    return tuple(buckets)


def check_buckets(buckets, max_num_seqs, k_default, ragged):
    """The per-bucket startup checks (RuntimeError with the launch fix). Every bucket: B_cfg x T_cfg verify rows in
    one 32-row decode tile, B_cfg <= max_num_seqs, K = T_cfg-1 in [1, min(_MAX_DRAFT, k_default)] (k_default =
    default_draft_len(): the drafter checkpoint's block and QWEN36_DFLASH_BLOCK bound every bucket's K). Across
    buckets: distinct B_cfg (plan() picks the smallest bucket that seats the live requests) and the largest B_cfg ==
    max_num_seqs (every slot can be seated). More than one bucket needs the RAGGED contract: one speculative
    iteration per decode step keeps a switch inside one step. Returns the buckets as given."""
    buckets = tuple((int(b), int(t)) for b, t in buckets)
    if not buckets:
        raise RuntimeError(f"{_BUCKETS_ENV}: no verify bucket configured")
    for b, t in buckets:
        name = f"bucket {b}x{t}"
        if b < 1 or t < 2:
            raise RuntimeError(f"{_BUCKETS_ENV}: {name} needs B >= 1 users and T >= 2 rows (K = T-1 >= 1 draft)")
        if b * t > MAX_SPEC_ROWS:
            raise RuntimeError(
                f"Qwen36DFlash speculative serving: {name}: {b} users x T={t} verify rows = {b * t} exceed one "
                f"{MAX_SPEC_ROWS}-row decode tile; use B <= {MAX_SPEC_ROWS // t} for T={t} (or QWEN36_DRAFTER=mtp)"
            )
        if b > max_num_seqs:
            raise RuntimeError(
                f"{_BUCKETS_ENV}: {name} seats more users than --max-num-seqs {max_num_seqs}; every bucket's B must be "
                f"<= max_num_seqs (and the largest == max_num_seqs)"
            )
        if t - 1 > _MAX_DRAFT:
            raise RuntimeError(f"{name}: K={t - 1} exceeds the KV lookahead bound {_MAX_DRAFT}")
        if t - 1 > k_default:
            raise RuntimeError(
                f"{_BUCKETS_ENV}: {name} drafts K={t - 1} but the drafter drafts at most K={k_default} "
                f"(the checkpoint's block, or QWEN36_DFLASH_BLOCK={os.environ.get('QWEN36_DFLASH_BLOCK', 'unset')}): "
                f"set QWEN36_DFLASH_BLOCK >= {t}"
            )
    bs = [b for b, _ in buckets]
    if len(set(bs)) != len(bs):
        raise RuntimeError(f"{_BUCKETS_ENV}: buckets must have distinct B (got {bs}): the planner picks by live users")
    if max(bs) != max_num_seqs:
        raise RuntimeError(
            f"{_BUCKETS_ENV}: the largest bucket seats {max(bs)} users but --max-num-seqs is {max_num_seqs}; one bucket "
            f"must seat every slot"
        )
    if len(buckets) > 1 and not ragged:
        raise RuntimeError(
            f"{_BUCKETS_ENV}: {len(buckets)} buckets need QWEN36_DFLASH_RAGGED=1 (one speculative iteration per decode "
            f"step keeps a bucket switch inside one step)"
        )
    return buckets


def bucket_id(bucket):
    """The decoder's name for a (B_cfg, T_cfg) bucket ('8x4'), used when it exposes no registry of its own."""
    return f"{int(bucket[0])}x{int(bucket[1])}"


@MULTIMODAL_REGISTRY.register_processor(
    TTQwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36DFlashForCausalLM(Qwen36ForCausalLM):
    """Qwen36 plus contract-native multi-user DFlash2 speculation."""

    # Legacy readers see the safe width-one contract. The TT platform prefers the dynamic method
    # below, which reads launch policy when it builds the engine rather than when Python imports us.
    model_capabilities = {
        **Qwen36ForCausalLM.model_capabilities,
        "supports_spec_decode": True,
        "supports_async_spec_decode": False,
        "spec_requirements": ["device_propose"],
    }

    @classmethod
    def get_model_capabilities(cls):
        return DFlashRuntimeConfig.from_env().capabilities(Qwen36ForCausalLM.model_capabilities)

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Resolve the plugin-visible maximum K across all exact buckets."""
        from vllm_tt_plugin.spec_decode import (
            ACCEPT_MODE_ARGMAX_IDS,
            ACCEPT_MODE_LOGITS,
            DRAFTER_STATE_INTERNAL,
            SpecPlan,
            SpecReject,
        )

        runtime = DFlashRuntimeConfig.from_env()
        if not runtime.spec_on:
            return SpecReject("QWEN36_DRAFTER does not select dflash2")
        max_num_seqs = int(max_num_seqs)
        requested_k = int(requested_k)
        try:
            buckets = parse_buckets(os.environ.get(_BUCKETS_ENV))
            # Capability planning happens before the model and auxiliary
            # checkpoint are loaded.  Never call default_draft_len() here:
            # it reads the drafter config through snapshot_download and made
            # a host-only vLLM configuration query depend on network/cache
            # state.  Explicit buckets fully describe the required capacity.
            # For the legacy single-bucket launch, QWEN36_DFLASH_BLOCK is the
            # author-supplied checkpoint bound; absent that, the requested K
            # is the only information available until model construction.
            k_capacity = (
                max(t - 1 for _b, t in buckets)
                if buckets
                else int(os.environ.get("QWEN36_DFLASH_BLOCK", requested_k + 1)) - 1
            )
            buckets = buckets or ((max_num_seqs, k_capacity + 1),)
            # The standard plugin contract is intrinsically ragged: a verify
            # returns the exact current bucket width and vLLM commits each row's
            # accepted prefix. It does not depend on the legacy block-output
            # RAGGED switch.
            buckets = check_buckets(buckets, max_num_seqs, k_capacity, True)
        except (RuntimeError, ValueError) as exc:
            return SpecReject(str(exc))
        effective_k = max(t - 1 for _b, t in buckets)
        if requested_k < effective_k:
            return SpecReject(
                f"the configured buckets require K={effective_k}, but the launch requested K={requested_k}",
                supported_k=(effective_k,),
            )
        k_by_rows = []
        for max_rows, width in sorted(buckets):
            k = width - 1
            if k_by_rows and k_by_rows[-1][1] == k:
                k_by_rows[-1] = (max_rows, k)
            else:
                k_by_rows.append((max_rows, k))
        return SpecPlan(
            effective_k=effective_k,
            lanes_per_request=max(t for _b, t in buckets),
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS),
            drafter_state=DRAFTER_STATE_INTERNAL,
            # A plain decode reads the durable GDN rows, which an active
            # request stopped updating at its first verify.
            supports_narrow_decode=False,
            k_by_rows=tuple(k_by_rows),
        )

    def __init__(self, *args, **kwargs):
        self._dflash = DFlashRuntimeConfig.from_env()
        super().__init__(*args, **kwargs)
        self._spec = None  # DFlash2ServingDecoder, armed at warmup (phase 2)
        self._spec_pre = None  # prepared (allocated + compiled) decoder between warmup phases
        self._in_warmup = False
        model = self.model[0]
        self._eos = self._load_eos_ids(model)
        self._eos_fill = min(self._eos)
        B = int(model.args.max_batch_size)
        self._B = B
        # row -> physical session slot (the plugin's slot_remap is composed into it; nothing moves on device)
        self._phys = list(range(B))
        # Physical-session slot -> (prompt_len, page_table_row, base_state_slot)
        # from prefill until the first decode.  The drafter/spec session is
        # keyed by the physical slot, while the target model's temporary GDN
        # seed must live in the request's unique logical base row.  Keeping
        # those two identities separate prevents a concurrently-prefilled
        # request from overwriting another request's not-yet-seeded target
        # state.
        self._pending = [None] * B
        self._carry = [[] for _ in range(B)]  # committed-but-unemitted tokens per physical slot
        self._stopped = [False] * B  # a stop token was committed; the row is EOS-filled from there
        self._prev_tail = [None] * B
        # The plugin settles an old-slot -> new-slot permutation only after an
        # accepted decode step.  Qwen consumes the equivalent new-row ->
        # old-row ``slot_remap`` as part of that decode (and therefore has
        # already composed it into ``_phys``); retain the expected settlement
        # so the explicit lifecycle callback can verify, rather than apply, the
        # move a second time.
        self._pending_state_slot_moves = None
        self._anchor_warned = False
        self._oov_warned = False
        self._nosession_warned = False
        self._buckets = ()  # (B_cfg, T_cfg) verify geometries, largest B == max_num_seqs (module doc)
        self._multi_bucket = False
        self._last_bucket = None  # bucket id plan() last put in force (logged on change)
        if self._dflash.spec_on:
            # Prompt tap clones are Python-side and cannot be produced by a masked-bucket replay.
            # Make this an instance decision after construction instead of mutating os.environ at import.
            for target in self.model:
                target._mb_trace_buckets = ()
            if model.num_devices <= 1:
                raise RuntimeError("Qwen36DFlash speculative serving needs the TP mesh (MESH_DEVICE=P150x4)")
            K = default_draft_len()
            buckets = parse_buckets(os.environ.get(_BUCKETS_ENV)) or ((B, K + 1),)
            self._buckets = check_buckets(buckets, B, K, self._dflash.ragged)
            self._multi_bucket = len(self._buckets) > 1
            if self._multi_bucket and DFlash2DualBucketDecoder is None:
                raise RuntimeError(
                    f"{_BUCKETS_ENV}={os.environ.get(_BUCKETS_ENV)!r} needs the multi-bucket decoder "
                    "(dflash2_serving.DFlash2DualBucketDecoder), which this tree does not have"
                )
            logger.info(
                f"Qwen36DFlash serving: slots={B} block W={self._dflash.output_width} tokens/step"
                f"{' (ragged: one iteration/step)' if self._dflash.ragged else ''}, "
                f"buckets={','.join(bucket_id(bt) for bt in self._buckets)} "
                f"(K={','.join(str(t - 1) for _, t in self._buckets)}), eos={sorted(self._eos)}, "
                f"drafter={os.environ.get('DFLASH_WEIGHTS')}"
            )
        else:
            logger.info("Qwen36DFlash serving: speculation OFF (plain Qwen36ForCausalLM behaviour)")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_eos_ids(model):
        ids = set()
        ckpt = model.args.CKPT_DIR
        for name in ("generation_config.json", "config.json"):
            try:
                with open(os.path.join(ckpt, name)) as f:
                    v = json.load(f).get("eos_token_id")
            except Exception:
                v = None
            if isinstance(v, int):
                ids.add(int(v))
            elif isinstance(v, (list, tuple)):
                ids.update(int(x) for x in v)
        hf_eos = getattr(getattr(model.args, "hf_config", None), "eos_token_id", None)
        if isinstance(hf_eos, int):
            ids.add(int(hf_eos))
        elif isinstance(hf_eos, (list, tuple)):
            ids.update(int(x) for x in hf_eos)
        if not ids:
            raise RuntimeError("no eos_token_id found for the served checkpoint")
        return ids

    def _clean_ids(self, block):
        """At most W committed ids, every out-of-vocab id replaced by EOS (never happens on the exact path;
        a corrupted id must not reach the detokenizer)."""
        block = [int(t) for t in block][: self._dflash.output_width]
        vocab = self.model[0].vocab_size
        oov = [t for t in block if not 0 <= t < vocab]
        if oov:
            if not self._oov_warned:
                self._oov_warned = True
                logger.warning(
                    f"Qwen36DFlash: {len(oov)} out-of-vocab committed id(s) (first={oov[0]}); substituting EOS"
                )
            block = [t if 0 <= t < vocab else self._eos_fill for t in block]
        return block

    def _fill_block(self, block):
        """Exactly W ids: a short block happens only at a genuine stop -> EOS-fill (the scheduler trims at
        the first stop token)."""
        block = self._clean_ids(block)
        out = torch.full((self._dflash.output_width,), self._eos_fill, dtype=torch.int32)
        if block:
            out[: len(block)] = torch.tensor(block, dtype=torch.int32)
        return out

    @staticmethod
    def _spec_pref_pt(model, pt):
        """A page-table row padded/trimmed to the width the (batched, B=1-scratch) prefill programs were
        compiled with: the chunk buffer's width when the chunk-prefill trace exists, else as given."""
        pt = torch.as_tensor(pt).reshape(1, -1).to(torch.int32)
        buf = getattr(model, "_chunk_full_page_table_buf", None)
        if buf is None:
            return pt
        nb = int(buf.shape[-1])
        if pt.shape[1] >= nb:
            return pt[:, :nb].contiguous()
        return torch.cat([pt, torch.zeros(1, nb - pt.shape[1], dtype=torch.int32)], dim=1)

    def _spec_prefill(
        self,
        model,
        dec,
        phys,
        prompt,
        T,
        pt_row,
        vision_context=None,
        state_slot=None,
    ):
        """Eager tap-capturing prefill of ONE request into physical slot ``phys`` (masked bucket for a
        short prompt, 2048-token chunks + masked tail for a long one), each chunk's taps ingested into the
        drafter's ring for that slot. ``state_slot`` is the target model's
        temporary base-state row; it is intentionally independent from the
        physical speculative-session slot. Returns host logits [1, vocab]
        (float)."""

        state_slot = int(phys if state_slot is None else state_slot)

        def on_chunk(hidden, chunk_start, valid_len):
            taps = model.take_dflash_eager_taps()
            if taps is None:
                raise RuntimeError("Qwen36DFlash: eager prefill captured no drafter taps (bucket trace gate on?)")
            dec.ingest_prompt(phys, taps, chunk_start + valid_len, chunk_start=chunk_start)

        dec.ctx_len[phys] = 0
        set_vision_context = getattr(dec, "set_vision_context", None)
        if callable(set_vision_context):
            set_vision_context(phys, vision_context)
        model._dflash_tap = True
        try:
            logits_dev = model.prefill_for_spec(
                prompt,
                self._spec_pref_pt(model, pt_row),
                T,
                on_chunk,
                slot=state_slot,
                vision_context=vision_context,
            )
        except Exception:
            if callable(set_vision_context):
                set_vision_context(phys, None)
            raise
        finally:
            model._dflash_tap = False
        if self._dflash.fast_logits:
            # Replicated, tile-padded to 32 rows: untilize on device (31 padding rows dropped, 16 MB -> 0.5 MB per
            # device) and read device 0 only -- the QWEN36_PREFILL_LOGITS_FAST path of prefill_paged_slots. The
            # untilize program compiles here during phase-1 warm-up (every _spec_prepare prefill takes this path),
            # i.e. before any trace is parked.
            lg_rm = ttnn.to_layout(logits_dev, ttnn.ROW_MAJOR_LAYOUT)
            lt = ttnn.to_torch(ttnn.get_device_tensors(lg_rm)[0])
            ttnn.deallocate(lg_rm)
        else:
            lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
        ttnn.deallocate(logits_dev)
        if dec.ctx_len[phys] != T:
            raise RuntimeError(f"Qwen36DFlash: spec prefill covered {dec.ctx_len[phys]} of {T} positions (slot {phys})")
        return lt.reshape(-1, model.vocab_size)[:1].float()

    # ------------------------------------------------------------------ warmup: arm the session once
    def warmup_model_decode(self, *args, **kwargs):
        """Two-phase like the runner's own warmup: phase 1 (enable_trace=False) ALLOCATES every spec
        buffer and COMPILES every spec program while no trace is parked yet; phase 2 (enable_trace=True)
        only CAPTURES the spec traces, after the plain decode traces. Buffers the parked traces bake must
        exist before any capture, and nothing spec-related may compile once one is parked."""
        self._in_warmup = True
        try:
            out = super().warmup_model_decode(*args, **kwargs)
            if not self._dflash.spec_on:
                return out
            self._spec_warmup_phases(kwargs)
        finally:
            self._in_warmup = False
        return out

    def warmup_model_prefill(self, *args, **kwargs):
        self._in_warmup = True
        try:
            return super().warmup_model_prefill(*args, **kwargs)
        finally:
            self._in_warmup = False

    def _spec_warmup_phases(self, kwargs):
        num_blocks = kwargs.get("num_blocks")
        if not num_blocks:
            kv = kwargs.get("kv_cache")
            num_blocks = int(kv[0][0].shape[0]) if kv else 4096
        if not kwargs.get("enable_trace"):
            if self._spec_pre is None:
                self._spec_pre = self._spec_prepare(int(num_blocks), kwargs.get("kv_cache"))
        elif self._spec is None:
            if self._spec_pre is None:  # a runner that skipped phase 1 (no compile warmup)
                self._spec_pre = self._spec_prepare(int(num_blocks), kwargs.get("kv_cache"))
            self._spec_capture()

    def _spec_prepare(self, num_blocks, kv_cache):
        """Phase 1: allocate + compile (no capture). Returns the prepared decoder."""
        model = self.model[0]
        t0 = time.perf_counter()
        # Spec verify runs the fused GDN op, so decode must use the same math (model-scoped, explicit).
        model.set_gdn_fused_decode(True)
        # The verify trace's table width is vLLM's per-request row width, which the runner pads to the
        # whole pool: use the pool block count.
        dec = self._make_decoder(model, num_blocks)
        k_max = int(getattr(dec, "K_max", dec.K))
        if k_max > _MAX_DRAFT:
            raise RuntimeError(f"drafter K={k_max} exceeds the KV lookahead bound {_MAX_DRAFT}")
        dec.alloc()
        # Eager: every bucket's verify pass (its throwaway KV writes land at S0+1, the position the capture pass
        # uses too), every seed / switch variant, every layout's draft + extend.
        S0 = min(model._PREFILL_MASK_BUCKETS[0], _PREFILL_CHUNK - 1)
        dec.warm(warm_position=S0 + 1)
        B = self._B
        # Dummy per-slot page-table rows for the warm-up prefills: disjoint block ranges of the pool.
        nbu = max(1, num_blocks // B)
        rows = [torch.tensor([u * nbu + (i % nbu) for i in range(num_blocks)], dtype=torch.int32) for u in range(B)]
        # 1) The eager tap-capturing prefill + drafter context fill for EVERY mask bucket a prompt can take
        #    (each bucket is its own program set), into slot 0; the largest bucket with a (chunk-1)-token
        #    prompt. Then the chunked shapes (one full chunk + a masked tail, and an exact chunk multiple).
        top = model._mask_bucket_for(_PREFILL_CHUNK - 1)
        buckets = [b for b in model._PREFILL_MASK_BUCKETS if b <= top] or [top]
        lens = [min(S, _PREFILL_CHUNK - 1) for S in buckets] + [_PREFILL_CHUNK + buckets[0], _PREFILL_CHUNK]
        for T in lens:
            self._spec_prefill(model, dec, 0, dummy_prompt(T, seed=T), T, rows[0])
        # 2) Every OTHER slot once (the GDN slot-write and the drafter's per-slot fill are per-slot programs).
        S0 = min(buckets[0], _PREFILL_CHUNK - 1)
        for u in range(1, B):
            self._spec_prefill(model, dec, u, dummy_prompt(S0, seed=100 + u), S0, rows[u])
        ttnn.synchronize_device(model.mesh_device)
        logger.info(f"Qwen36DFlash phase-1 warmup (alloc + compile) done in {time.perf_counter() - t0:.1f}s")
        self._warm_rows = rows
        return dec

    def _make_decoder(self, model, num_blocks):
        """The serving decoder for the configured buckets: the multi-bucket DFlash2DualBucketDecoder (a single
        bucket degenerates to today's DFlash2ServingDecoder behaviour, row == slot, no switch path); today's class
        itself when the dual decoder is absent from the tree and one bucket is configured."""
        if DFlash2DualBucketDecoder is not None:
            return DFlash2DualBucketDecoder(model, num_blocks, buckets=self._buckets, stop_tokens=self._eos)
        assert not self._multi_bucket, "__init__ rejects several buckets without the dual-bucket decoder"
        dec = DFlash2ServingDecoder(model, num_blocks, stop_tokens=self._eos)
        ((_, t_cfg),) = self._buckets
        if dec.K + 1 != t_cfg:
            raise RuntimeError(f"Qwen36DFlash: decoder T={dec.K + 1} != configured bucket T={t_cfg}")
        return dec

    def _bucket_ids(self, dec):
        """(B_cfg, T_cfg) -> the decoder's bucket id: from its own registry when it has one (``dec.buckets``: id ->
        object with .B/.T), else the 'BxT' convention."""
        ids = {}
        reg = getattr(dec, "buckets", None)
        if isinstance(reg, dict):
            for bid, bk in reg.items():
                try:
                    ids[(int(bk.B), int(bk.T))] = bid
                except (AttributeError, TypeError, ValueError):
                    pass
        for bt in self._buckets:
            ids.setdefault(bt, bucket_id(bt))
        return ids

    @staticmethod
    def _plan_if_available(dec, live_after):
        """dec.plan(live_after) when the decoder has a bucket planner (host-only; the single-bucket planner is a no-op)."""
        plan = getattr(dec, "plan", None)
        return plan(set(live_after)) if plan is not None else None

    @staticmethod
    def _expect_bucket(dec, want, when):
        cur = getattr(dec, "cur_id", None)
        if cur != want:
            raise RuntimeError(f"Qwen36DFlash warmup: verify bucket in force is {cur!r}, expected {want!r} {when}")

    @staticmethod
    def _timed_steps(dev, dec, n_steps, tag):
        """n_steps traced steps over the live slots; returns the 'ms/step, tok/step' summary for the warm-up log."""
        ttnn.synchronize_device(dev)
        t1 = time.perf_counter()
        n = 0
        for _ in range(n_steps):
            n += sum(len(v) for v in dec.step().values())
        ttnn.synchronize_device(dev)
        t2 = time.perf_counter()
        n_live = sum(1 for a in dec.active if a)
        return f"{tag}: {(t2 - t1) / n_steps * 1e3:.1f} ms/step, {n / n_steps:.1f} tok/step over {n_live} slot(s)"

    # ---- trace-region accounting (the region is fixed and shared by the chunk / decode / sampling / spec traces)
    @staticmethod
    def _trace_region_total(dev):
        try:
            mv = ttnn.get_memory_view(dev, ttnn.BufferType.TRACE)
            return int(mv.total_bytes_per_bank) * int(mv.num_banks)
        except Exception:
            return None

    def _log_trace_region(self, model, tag, since=None):
        """Log the TRACE region occupancy (and the delta since ``since`` bytes); returns bytes used or None."""
        dev = model.mesh_device
        used = model._trace_region_bytes(dev)
        if used is None:
            return None
        total = self._trace_region_total(dev)
        share = f" ({used / total:.0%} of {total / 2**20:.0f} MiB)" if total else ""
        delta = f", +{(used - since) / 2**20:.1f} MiB" if since is not None else ""
        logger.info(f"Qwen36DFlash TRACE region {tag}: {used / 2**20:.1f} MiB{share}{delta}")
        return used

    def _check_trace_region(self, model):
        """Startup guard: the region must stay under _TRACE_REGION_MAX_FRAC once every spec trace is parked (an
        overflow would only surface as a TT_FATAL at the next capture -- or at request time)."""
        dev = model.mesh_device
        used = model._trace_region_bytes(dev)
        total = self._trace_region_total(dev)
        if used is None or not total:
            return
        limit = self._dflash.trace_region_max_frac
        if used > limit * total:
            msg = (
                f"Qwen36DFlash: TRACE region {used / 2**20:.0f} MiB of {total / 2**20:.0f} MiB ({used / total:.0%}) "
                f"after the spec captures exceeds {limit:.0%}: raise additional_config.tt."
                f"trace_region_size or park fewer traces (TT_DECODE_BUCKETING=0, fewer prefill mask buckets)"
            )
            if self._multi_bucket or self._dflash.guard_strict:
                raise RuntimeError(msg)
            logger.warning(msg)

    def _spec_dummy_sessions_multi(self, model, dec, rows, S0):
        """Dummy sessions that capture the drafter's draft/extend traces in EVERY bucket and exercise a switch in
        each direction with the traces parked: every slot joins in the largest bucket (4 steps); then for each
        smaller bucket the surplus slots leave and the decoder is switched to it EXPLICITLY (the hysteresis would
        wait; 4 steps); then the ended slots re-join -- prefills first, ONE plan() that must force the largest bucket
        back, then the begins, decode_forward's order -- and step once. Returns the per-bucket timing summary."""
        dev = model.mesh_device
        B = self._B
        ids = self._bucket_ids(dec)
        order = sorted(self._buckets, key=lambda bt: -bt[0])  # largest B first
        big_id = ids[order[0]]
        live = set()
        for u in range(B):
            lt = self._spec_prefill(model, dec, u, dummy_prompt(S0, seed=7 + u), S0, rows[u])
            live.add(u)
            dec.plan(set(live))
            dec.begin(u, int(lt.argmax()), S0, rows[u])
        ttnn.synchronize_device(dev)
        self._expect_bucket(dec, big_id, f"with all {B} slots live")
        stats = [self._timed_steps(dev, dec, 4, big_id)]
        for bt in order[1:]:
            bid = ids[bt]
            for u in range(bt[0], B):
                dec.end(u)
                live.discard(u)
            t = time.perf_counter()
            dec.set_bucket(bid)
            ttnn.synchronize_device(dev)
            self._expect_bucket(dec, bid, f"after set_bucket({bid!r})")
            logger.info(
                f"Qwen36DFlash warmup: switch -> {bid} with {len(live)} live slot(s) in {(time.perf_counter() - t) * 1e3:.1f} ms"
            )
            stats.append(self._timed_steps(dev, dec, 4, bid))
        rejoin = [u for u in range(B) if u not in live]
        firsts = {
            u: int(self._spec_prefill(model, dec, u, dummy_prompt(S0, seed=7 + u), S0, rows[u]).argmax())
            for u in rejoin
        }
        t = time.perf_counter()
        dec.plan(set(range(B)))
        ttnn.synchronize_device(dev)
        self._expect_bucket(dec, big_id, f"after plan() with all {B} slots live (forced up-switch)")
        logger.info(
            f"Qwen36DFlash warmup: plan()-forced switch -> {big_id} with {len(live)} live slot(s) in "
            f"{(time.perf_counter() - t) * 1e3:.1f} ms"
        )
        for u in rejoin:
            dec.begin(u, firsts[u], S0, rows[u])
        dec.step()
        for u in range(B):
            dec.end(u)
        ttnn.synchronize_device(dev)
        return "; ".join(stats)

    def _spec_post_capture_guard(self, model, dec, rows, S0):
        """Nothing spec-related may compile once a trace is parked (it clobbers the trace: a device hang at request
        time). With every trace captured: count the program cache, run the decoder's warm sweep again (every seed
        and switch realization) plus one traced step in each bucket, and fail startup if the count changed
        (multi-bucket; a warning on the single-bucket profiles)."""
        dev = model.mesh_device
        count = getattr(dev, "num_program_cache_entries", None)
        if count is None:
            logger.warning("Qwen36DFlash: mesh device has no num_program_cache_entries(); post-capture guard skipped")
            return
        ttnn.synchronize_device(dev)
        n0 = int(count())
        sweep = getattr(dec, "warm_sweep", None) or getattr(dec, "_warm_sweep", None)
        if sweep is not None:
            sweep()  # every seed (bucket, row, slot) and switch (row, mi) realization again, with the traces parked
            ttnn.synchronize_device(dev)
        elif self._multi_bucket:
            logger.warning(
                "Qwen36DFlash: decoder has no warm_sweep(); the post-capture guard covers one traced step per bucket only"
            )
        ids = self._bucket_ids(dec)
        lt = self._spec_prefill(model, dec, 0, dummy_prompt(S0, seed=7), S0, rows[0])
        self._plan_if_available(dec, {0})
        dec.begin(0, int(lt.argmax()), S0, rows[0])
        dec.step()
        if self._multi_bucket:
            # ...one traced step in every other bucket too, ending in the smallest: the bucket a lightly loaded
            # server should start serving in (plan() forces the larger one the moment it is needed).
            for bt in sorted(self._buckets, key=lambda bt: -bt[0]):
                if ids[bt] != dec.cur_id:
                    dec.set_bucket(ids[bt])
                    dec.step()
        dec.end(0)
        ttnn.synchronize_device(dev)
        n1 = int(count())
        if n1 != n0:
            msg = (
                f"Qwen36DFlash: {n1 - n0} program(s) compiled AFTER the spec traces were captured (program cache "
                f"{n0} -> {n1} entries over the warm sweep + one traced step per bucket): a seed / switch / draft "
                f"variant is missing from the phase-1 warm-up, and at request time it would clobber a parked trace"
            )
            if self._multi_bucket or self._dflash.guard_strict:
                raise RuntimeError(msg)
            logger.warning(msg)
        else:
            logger.info(
                f"Qwen36DFlash post-capture guard: program cache unchanged at {n0} entries over the warm sweep + one "
                f"traced step per bucket ({','.join(ids[bt] for bt in self._buckets)})"
            )

    def _spec_capture(self):
        """Phase 2: capture the verify trace of every bucket, then dummy sessions (the drafter's draft/extend traces
        are captured by the first traced step per bucket layout), then the two startup guards: the program cache
        must not grow over a repeat of the warm sweep + one traced step per bucket, and the TRACE region must stay
        under _TRACE_REGION_MAX_FRAC."""
        model = self.model[0]
        dec = self._spec_pre
        B = self._B
        t0 = time.perf_counter()
        S0 = min(model._PREFILL_MASK_BUCKETS[0], _PREFILL_CHUNK - 1)
        used0 = self._log_trace_region(model, "before the spec captures")
        dec.capture(warm_position=S0 + 1)
        used1 = self._log_trace_region(model, "after the verify capture(s)", used0)
        rows = self._warm_rows
        if self._multi_bucket:
            stats = self._spec_dummy_sessions_multi(model, dec, rows, S0)
        else:
            live = set()
            for u in range(B):
                lt = self._spec_prefill(model, dec, u, dummy_prompt(S0, seed=7 + u), S0, rows[u])
                live.add(u)
                self._plan_if_available(dec, live)
                dec.begin(u, int(lt.argmax()), S0, rows[u])
            ttnn.synchronize_device(model.mesh_device)
            t1 = time.perf_counter()
            n = 0
            for _ in range(4):
                n += sum(len(v) for v in dec.step().values())
            ttnn.synchronize_device(model.mesh_device)
            t2 = time.perf_counter()
            for u in range(B):
                dec.end(u)
            stats = f"{(t2 - t1) / 4 * 1e3:.1f} ms/step, {n / 4:.1f} tok/step over {B} slot(s) (K={dec.K})"
        self._log_trace_region(model, "after the drafter captures", used1)
        self._spec_post_capture_guard(model, dec, rows, S0)
        self._check_trace_region(model)
        self._reset_bucket_policy(dec)
        self._spec = dec
        self._spec_pre = None
        logger.info(
            f"Qwen36DFlash phase-2 warmup (captures) done in {time.perf_counter() - t0:.1f}s: "
            f"{stats} (W={self._dflash.output_width})"
        )

    def _reset_bucket_policy(self, dec):
        """The warm-up dummy sessions and the post-capture guard drive real bucket switches (down/up with slots
        live), so by the first request the planner has an inflated cooldown (churn back-off) and the switch
        counters / per-bucket stats carry the synthetic warm-up traffic. Clear both so the server starts with a
        clean hysteresis (no lingering 32-step down-cooldown for its life) and the shutdown log reports only real
        request switches. Host-only; a no-op for a single bucket (no switch path)."""
        if not self._multi_bucket:
            return
        reset = getattr(dec, "reset_bucket_policy", None)  # lane B may add a dedicated hook later
        if callable(reset):
            reset()
            return
        reg = getattr(dec, "buckets", None)
        if BucketPlanner is not None and isinstance(reg, dict) and reg:
            try:
                dec._planner = BucketPlanner({b.id: b.B for b in reg.values()}, start=dec.cur_id)
            except Exception as e:  # pragma: no cover - defensive; the planner shape is fixed
                logger.warning(f"Qwen36DFlash: could not reset the bucket planner after warm-up: {e!r}")
        if hasattr(dec, "switch_count"):
            dec.switch_count = 0
        sm = getattr(dec, "switch_ms", None)
        if sm is not None:
            try:
                sm.clear()
            except AttributeError:
                dec.switch_ms = []
        if hasattr(dec, "_last_switch_step"):
            dec._last_switch_step = None
        for b in (reg or {}).values():
            st = getattr(b, "stats", None)
            if isinstance(st, dict):
                for k in list(st):
                    st[k] = 0
        logger.info(
            "Qwen36DFlash: bucket planner and switch stats reset after warm-up (clean hysteresis at request time)"
        )

    # ------------------------------------------------------------------ prefill: eager, taps -> drafter rings
    def _spec_ready(self):
        """True once the session is armed; False for the base warmup's own forwards (they must pass
        through to the plain path); raises if a request arrives without a session, since the declared
        block-output contract could not be honoured (a width mismatch would kill the engine later)."""
        if self._spec is not None:
            return True
        if self._in_warmup:
            return False
        raise RuntimeError(
            "Qwen36DFlash: speculative session not armed -- the plugin's decode warmup (enable_model_warmup "
            "with decode traces) did not run, but the block-output contract is declared; serve with "
            "warmup on, or QWEN36_DRAFTER=mtp for plain decode"
        )

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        if not self._dflash.spec_on or not self._spec_ready():
            return super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        model = self.model[0]
        dec = self._spec
        N = int(tokens.shape[0])
        has_images = self._has_visual(kwargs, "pixel_values")
        has_videos = self._has_visual(kwargs, "pixel_values_videos")
        if has_videos:
            raise RuntimeError("Qwen36DFlash video input is not enabled yet; use an image or text request")
        vision_contexts = [
            self._compute_vision_context(model, kwargs, user=u) if has_images else None for u in range(N)
        ]
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [int(tokens.shape[1])] * N
        empty_slots = kwargs.get("empty_slots")
        logical = [int(s) for s in empty_slots] if empty_slots is not None else list(range(N))
        pt = torch.as_tensor(page_table)
        out = []
        for u in range(N):
            state_slot = logical[u]
            phys = self._phys[state_slot]
            if dec.active[phys]:
                # vLLM released the previous occupant before reusing its slot; close its session if not.
                dec.end(phys)
            self._carry[phys], self._stopped[phys], self._prev_tail[phys] = [], False, None
            T = plens[u]
            prompt = torch.as_tensor(tokens)[u : u + 1, :T].to(torch.int32)
            row = pt[u].reshape(-1).clone()
            logger.info(f"Prefilling slot {phys} up to {T} tokens (TP eager spec prefill)")
            lt = self._spec_prefill(
                model,
                dec,
                phys,
                prompt,
                T,
                row,
                vision_context=vision_contexts[u],
                state_slot=state_slot,
            )
            out.append(lt.view(1, 1, -1))
            self._pending[phys] = (T, row, state_slot)
        logger.info(f"Finished prefill of {N} request(s), starting decode...")
        return torch.cat(out, dim=0), torch.zeros(N, dtype=torch.long)

    # ------------------------------------------------------------------ decode: one block per step, all live slots
    def decode_forward(self, *args, **kwargs):
        if not self._dflash.spec_on or not self._spec_ready():
            return super().decode_forward(*args, **kwargs)
        spec_mode = kwargs.get("spec_mode")
        if spec_mode is None and hasattr(self._spec, "verify_contract"):
            raise RuntimeError(
                "Qwen36DFlash verifies every decode step (supports_narrow_decode=False) but received a decode "
                "without spec_mode; a plain decode would read GDN state that an active request stopped "
                "updating at its first verify"
            )
        if spec_mode not in (None, "argmax_ids", "logits"):
            raise RuntimeError(f"Qwen36DFlash supports spec_mode 'argmax_ids' or 'logits', got {spec_mode!r}")

        def _read(name, pos):
            if name in kwargs:
                return kwargs[name]
            return args[pos] if pos < len(args) else None

        # Spec state stays in its fixed physical slots, so a row remap only
        # changes this indirection. Nothing gathers the target GDN rows either:
        # a request that has not seeded yet still seeds from the row its
        # prefill wrote.
        slot_remap = _read("slot_remap", 10)
        if slot_remap is not None:
            remap = [int(r) for r in torch.as_tensor(slot_remap).reshape(-1).tolist()]
            if sorted(remap) != list(range(self._B)):
                raise RuntimeError(f"Qwen36DFlash: slot_remap {remap} is not a permutation of {self._B} slots")
            self._phys = [self._phys[r] for r in remap]
            expected_moves = {old_slot: new_slot for new_slot, old_slot in enumerate(remap) if old_slot != new_slot}
            if expected_moves:
                pending_moves = self._pending_state_slot_moves
                if pending_moves is not None and pending_moves != expected_moves:
                    raise RuntimeError(
                        "Qwen36DFlash received conflicting unsettled state-slot moves: "
                        f"pending={pending_moves}, current={expected_moves}"
                    )
                self._pending_state_slot_moves = expected_moves

        if spec_mode is not None:
            return self._decode_spec_contract(
                tokens=_read("tokens", 0),
                start_pos=_read("start_pos", 1),
                page_table=_read("page_table", 2),
                num_valid_drafts=kwargs.get("num_valid_drafts"),
                accepted_counts=kwargs.get("accepted_counts"),
                spec_mode=spec_mode,
            )

        tokens = _read("tokens", 0)
        start_pos = _read("start_pos", 1)
        page_table = _read("page_table", 2)
        if tokens is None:
            raise RuntimeError("Qwen36DFlash decode expects token input")
        dec = self._spec
        B = self._B
        Bp = int(tokens.shape[0])
        if Bp > B:
            raise RuntimeError(f"Qwen36DFlash: {Bp} decode rows for {B} slots")
        toks = torch.as_tensor(tokens).reshape(Bp, -1)[:, 0].tolist()
        poss = torch.as_tensor(start_pos).reshape(-1).tolist() if start_pos is not None else [0] * Bp
        pts = torch.as_tensor(page_table) if page_table is not None else None
        # The verify bucket of THIS step is decided BEFORE any begin()/set_table()/step() of the step, from the
        # sessions that are live after the row loop: the active ones plus the prefilled slots that begin below.
        # plan() reseeds every live session into the bucket it picks (and only the larger bucket can seat a 5th
        # user), so the begins below land in the geometry in force; a mis-ordering is a decoder assert, never a
        # wrong-geometry seed. Single bucket: a host no-op.
        plan = getattr(dec, "plan", None)
        if plan is not None:
            live_after = {phys for phys in range(B) if dec.active[phys]}
            for i in range(Bp):
                if int(poss[i]) >= 0 and self._pending[self._phys[i]] is not None:
                    live_after.add(self._phys[i])
            bucket = plan(live_after)
            if bucket != self._last_bucket:
                logger.info(f"Qwen36DFlash: verify bucket {bucket} in force ({len(live_after)} live slot(s))")
                self._last_bucket = bucket
        live_rows = []
        nosession_rows = []  # live rows without a session: EOS so the request ends (ragged: [EOS, -1, ...])
        for i in range(Bp):
            if int(poss[i]) < 0:
                continue  # padding row
            phys = self._phys[i]
            anchor = int(toks[i])
            row = pts[i].reshape(-1) if pts is not None else None
            if self._pending[phys] is not None:
                pending = self._pending[phys]
                if len(pending) == 2:  # Backward-compatible with old in-process test fixtures.
                    T, pt_row = pending
                    state_slot = phys
                else:
                    T, pt_row, state_slot = pending
                self._pending[phys] = None
                if int(poss[i]) != T:
                    logger.warning(f"Qwen36DFlash: slot {phys} first decode start_pos {int(poss[i])} != prompt_len {T}")
                dec.begin(
                    phys,
                    anchor,
                    T,
                    row if row is not None else pt_row,
                    seed_slot=state_slot,
                )
                self._carry[phys], self._stopped[phys], self._prev_tail[phys] = [], False, None
                self._anchor_warned = False
            elif dec.active[phys]:
                if row is not None:
                    dec.set_table(phys, row)  # vLLM allocates a new block every block_size tokens
                if self._prev_tail[phys] is not None and anchor != self._prev_tail[phys] and not self._anchor_warned:
                    self._anchor_warned = True
                    logger.warning(
                        f"Qwen36DFlash: runner anchor {anchor} != last emitted token {self._prev_tail[phys]} (slot {phys}); "
                        "the session owns the trajectory (expected only after an EOS-filled block under ignore_eos)"
                    )
            else:
                if not self._nosession_warned:
                    self._nosession_warned = True
                    logger.warning(f"Qwen36DFlash: live row {i} (slot {phys}) has no speculative session; EOS-filling")
                nosession_rows.append(i)
                continue
            live_rows.append((i, phys))
        t0 = time.perf_counter() if self._dflash.debug else 0.0
        if self._dflash.ragged:
            out = self._decode_ragged(dec, live_rows, Bp, t0)
            for i in nosession_rows:
                out[i, 0] = self._eos_fill
            return out
        # Step until every live row can fill its block. A row whose carry already holds a stop token
        # needs nothing: it emits through the stop this step (latency for a normal request); the
        # session itself is NOT stopped, so an ignore_eos client keeps getting real tokens after it.
        iters = 0

        def _needs(phys):
            return (
                dec.active[phys]
                and not (self._dflash.stop_fill and self._stopped[phys])
                and len(self._carry[phys]) < self._dflash.output_width
            )

        while True:
            need = [phys for _, phys in live_rows if _needs(phys)]
            if not need:
                break
            # Only the slots still short of a block step; the others hold, so a fast slot never runs
            # more than one block + one iteration ahead of what vLLM has consumed (its KV write reach
            # stays inside the declared lookahead).
            com = dec.step(only=need)
            iters += 1
            if not com:
                break
            for phys, ids in com.items():
                self._carry[phys].extend(int(t) for t in ids)
                if any(t in self._eos for t in ids):
                    self._stopped[phys] = True
        w = self._dflash.output_width
        out = torch.full((Bp, w), self._eos_fill, dtype=torch.int32)
        for i, phys in live_rows:
            carry = self._carry[phys]
            stop_i = next((k for k, t in enumerate(carry) if t in self._eos), None) if self._stopped[phys] else None
            if self._dflash.stop_fill and stop_i is not None and stop_i < w:
                # Emit through the first stop token and EOS-fill the rest of the block. The scheduler
                # normally releases the request next. An ignore_eos client (benchmarks) does not: the
                # session stays live and the tokens it already committed past the stop stay in the carry,
                # so the following blocks carry the model's real continuation at the real decode cost
                # (the runner's anchor then differs from the session's tail once: the session owns the
                # trajectory). Never leave a request coasting on free EOS fills.
                block, self._carry[phys] = carry[: stop_i + 1], carry[stop_i + 1 :]
                self._stopped[phys] = any(t in self._eos for t in self._carry[phys])
            else:
                block, self._carry[phys] = carry[:w], carry[w:]
                # A stop still sitting in the carry keeps the row marked; it is emitted next step.
                self._stopped[phys] = stop_i is not None
            out[i] = self._fill_block(block)
            self._prev_tail[phys] = int(out[i, -1])
        if self._dflash.debug:
            logger.info(
                f"[dflash2-serve] step: rows={[i for i, _ in live_rows]} iters={iters} "
                f"{(time.perf_counter() - t0) * 1e3:.1f} ms carry={[len(self._carry[p]) for _, p in live_rows]}"
            )
        return out

    def _decode_spec_contract(
        self,
        *,
        tokens,
        start_pos,
        page_table,
        num_valid_drafts,
        accepted_counts,
        spec_mode,
    ):
        """Verify one plugin-owned candidate block and return the target argmax ids or logits (``spec_mode``)."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        if tokens is None or start_pos is None:
            raise RuntimeError("Qwen36DFlash speculative verify needs tokens and positions")
        if num_valid_drafts is None or accepted_counts is None:
            raise RuntimeError("Qwen36DFlash speculative verify needs both side tensors")
        tokens = torch.as_tensor(tokens)
        positions = torch.as_tensor(start_pos)
        valid = torch.as_tensor(num_valid_drafts).reshape(-1)
        counts = torch.as_tensor(accepted_counts).reshape(-1)
        if tokens.ndim != 2 or positions.shape != tokens.shape:
            raise RuntimeError(
                f"Qwen36DFlash expected matching [B, 1+K] candidates, got "
                f"{tuple(tokens.shape)} and {tuple(positions.shape)}"
            )
        Bp = int(tokens.shape[0])
        if Bp > self._B or valid.numel() != Bp or counts.numel() != Bp:
            raise RuntimeError("Qwen36DFlash speculative side tensors have inconsistent rows")
        pts = torch.as_tensor(page_table) if page_table is not None else None
        dec = self._spec

        # Validate the previous commit before a bucket switch resets mi to 0.
        for logical_row in range(Bp):
            if int(positions[logical_row, 0]) < 0:
                continue
            phys = self._phys[logical_row]
            if dec.active[phys] and self._pending[phys] is None:
                expected = int(dec.mi[phys]) + 1
                if int(counts[logical_row]) != expected:
                    raise RuntimeError(
                        f"Qwen36DFlash slot {phys} accepted_counts={int(counts[logical_row])}, "
                        f"but its target state expects {expected}"
                    )

        live_after = {phys for phys in range(self._B) if dec.active[phys]}
        for logical_row in range(Bp):
            if int(positions[logical_row, 0]) >= 0 and self._pending[self._phys[logical_row]] is not None:
                live_after.add(self._phys[logical_row])
        plan = getattr(dec, "plan", None)
        if plan is not None:
            bucket = plan(live_after)
            if bucket != self._last_bucket:
                logger.info(f"Qwen36DFlash: verify bucket {bucket} in force ({len(live_after)} live slot(s))")
                self._last_bucket = bucket

        phys_by_row = [None] * Bp
        for logical_row in range(Bp):
            start = int(positions[logical_row, 0])
            if start < 0:
                continue
            phys = self._phys[logical_row]
            row = pts[logical_row].reshape(-1) if pts is not None else None
            if self._pending[phys] is not None:
                pending = self._pending[phys]
                if len(pending) == 2:
                    prompt_len, saved_row = pending
                    state_slot = phys
                else:
                    prompt_len, saved_row, state_slot = pending
                self._pending[phys] = None
                if start != int(prompt_len):
                    raise RuntimeError(
                        f"Qwen36DFlash slot {phys} first verify starts at {start}, expected prompt length {prompt_len}"
                    )
                dec.begin(
                    phys,
                    int(tokens[logical_row, 0]),
                    int(prompt_len),
                    row if row is not None else saved_row,
                    seed_slot=state_slot,
                    contract_seed=True,
                )
            elif dec.active[phys]:
                if row is not None:
                    dec.set_table(phys, row)
            else:
                raise RuntimeError(f"Qwen36DFlash live row {logical_row} (slot {phys}) has no speculative session")
            phys_by_row[logical_row] = phys

        read_logits = spec_mode == "logits"
        argmax_ids, hidden, logits = dec.verify_contract(tokens, positions, valid, phys_by_row, read_logits=read_logits)
        if read_logits:
            return VerifyOutput(spec_mode="logits", logits=logits, hidden=hidden)
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=argmax_ids, hidden=hidden)

    def propose_draft_tokens(
        self,
        num_speculative_tokens,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ):
        """Commit the preceding exact verify and return the next device drafts."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        del committed_positions, hidden
        rows = int(torch.as_tensor(committed_tokens).shape[0])
        drafts, num_valid = self._spec.commit_and_draft_contract(
            committed_tokens,
            accepted_counts,
            int(num_speculative_tokens),
            rows,
        )
        return DraftOutput(draft_token_ids=drafts, num_valid=num_valid)

    def _absorb(self, com):
        """Committed ids of one dec.step() -> the slots' carries (+ stop bookkeeping)."""
        for phys, ids in com.items():
            self._carry[phys].extend(int(t) for t in ids)
            if any(t in self._eos for t in ids):
                self._stopped[phys] = True

    def _decode_ragged(self, dec, live_rows, Bp, t0):
        """RAGGED contract (tt_adaptive_block_ragged): ONE speculative iteration over the live rows, then a
        [Bp, W] int32 block whose row i holds n_i = min(len(carry_i), W) real ids followed by _PAD (-1).

          * rows that step: every live row whose carry holds < W tokens (and, under _STOP_FILL, no stop
            token yet); a row whose carry already holds >= W tokens HOLDS (its KV write reach stays inside
            the declared lookahead) and emits W ids from the carry;
          * n_i >= 1 for every live row: a step commits >= 1 token (its pending token) per stepping slot. The
            single exception is a request whose folded seed token is itself a stop token (com == [first],
            nothing new -- only reachable under ignore_eos, since vLLM ends a request whose first token is a
            stop before it ever decodes): that row is stepped once more, which commits >= 1;
          * _STOP_FILL semantics: with "1" a row that reaches a stop emits through the FIRST stop token and
            pads the rest with -1 (the session stays live; tokens committed past the stop stay in the carry
            for an ignore_eos client); with "0" stops are ordinary tokens and n_i = min(len(carry), W);
          * padding rows (start_pos < 0) are all -1 (0 tokens; the plugin never reads them); a live row
            WITHOUT a speculative session gets [EOS, -1, ...] so the request ends (unchanged in spirit
            from the fixed-width path's EOS fill).
        """
        iters = 0

        def _needs(phys):
            return (
                dec.active[phys]
                and not (self._dflash.stop_fill and self._stopped[phys])
                and len(self._carry[phys]) < self._dflash.output_width
            )

        need = [phys for _, phys in live_rows if _needs(phys)]
        if need:
            self._absorb(dec.step(only=need))
            iters += 1
        # Folded-seed-is-a-stop corner (see docstring): an active row with nothing to emit steps again.
        empty = [phys for _, phys in live_rows if dec.active[phys] and not self._carry[phys]]
        if empty:
            self._absorb(dec.step(only=empty))
            iters += 1
        w = self._dflash.output_width
        out = torch.full((Bp, w), _PAD, dtype=torch.int32)
        for i, phys in live_rows:
            carry = self._carry[phys]
            if not carry:
                if dec.active[phys]:
                    raise RuntimeError(f"Qwen36DFlash: live slot {phys} committed no token in a ragged step")
                block = [self._eos_fill]  # no session (unreachable: such rows are not live_rows)
            else:
                stop_i = next((k for k, t in enumerate(carry) if t in self._eos), None) if self._stopped[phys] else None
                if self._dflash.stop_fill and stop_i is not None and stop_i < w:
                    block, self._carry[phys] = carry[: stop_i + 1], carry[stop_i + 1 :]
                    self._stopped[phys] = any(t in self._eos for t in self._carry[phys])
                else:
                    block, self._carry[phys] = carry[:w], carry[w:]
                    self._stopped[phys] = stop_i is not None
            block = self._clean_ids(block)
            out[i, : len(block)] = torch.tensor(block, dtype=torch.int32)
            self._prev_tail[phys] = int(block[-1])
        if self._dflash.debug:
            logger.info(
                f"[dflash2-serve] ragged step: rows={[i for i, _ in live_rows]} iters={iters} "
                f"{(time.perf_counter() - t0) * 1e3:.1f} ms widths={[int((out[i] >= 0).sum()) for i, _ in live_rows]} "
                f"carry={[len(self._carry[p]) for _, p in live_rows]}"
            )
        return out

    def read_decode_output(self, tt_out, async_read=False, *args, **kwargs):
        # A spec block step returns committed HOST tokens: nothing to read.
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        return super().read_decode_output(tt_out, async_read, *args, **kwargs)

    # ------------------------------------------------------------------ plugin lifecycle hooks
    def note_state_slots_moved(self, moves: dict[int, int]) -> None:
        """Settle the slot permutation already consumed by ``decode_forward``.

        The TT runner reports moves as ``old_slot -> new_slot`` after the model
        accepted the step.  This adapter receives the inverse representation
        (``new_row -> old_slot``) in the decode inputs and must consume it there
        so target GDN state and speculative session lookup use the same rows
        during that step.  Reapplying the callback would move a session twice;
        instead require the two views to agree exactly and retire the pending
        settlement.
        """
        settled = {int(old): int(new) for old, new in dict(moves).items()}
        expected = self._pending_state_slot_moves
        if not settled:
            return
        if expected != settled:
            raise RuntimeError(
                "Qwen36DFlash state-slot settlement does not match the consumed "
                f"decode remap: expected={expected}, settled={settled}"
            )
        self._pending_state_slot_moves = None

    def release_request(self, row: int) -> None:
        if self._spec is None:
            return
        phys = self._phys[int(row)] if 0 <= int(row) < self._B else None
        if phys is None:
            return
        self._spec.end(phys)
        self._pending[phys] = None
        self._carry[phys], self._stopped[phys], self._prev_tail[phys] = [], False, None

    def release_persistent_capture(self) -> None:
        if self._spec is None:
            return
        self._spec.release()
        self._spec = None
