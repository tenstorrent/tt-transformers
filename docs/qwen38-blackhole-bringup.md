# Qwen3.8-27B Blackhole bring-up: measured implementation lessons

This note contributes implementation experience from bitpartner/tt-stack to
reviewers and developers of the Qwen3.8 stack (#41, #42 and its follow-ups).
The measurements below come from a separate serving implementation on two
cabled P150a devices (TP=2, mesh 1x2). They do not qualify this package, its
pending Qwen implementation, another checkpoint, or another TTNN release.

The source master was checked on 2026-11-11: `782434b31721d1232bda854c9848c5ff7b88caf5`.
The measurement tree is `a8e3549107a4502fc6a3d031081c5eb873db2c29`.
A compact, allowlisted extraction of the measured record is checked in as
[evidence/qwen38-p150x2-cold-prefill.json](evidence/qwen38-p150x2-cold-prefill.json).
It includes the original record SHA256 and exact source URL. Full source
records may require access to the source Forgejo instance.

## Relation to the pending upstream model

Upstream #41 provides `qkv_causal_conv1d_silu`, `gdn_decode_step`, and
`gdn_spec_step`; #42 provides the hybrid model foundation. Their declared
runtime contract is `ttnn>=0.79.1rc1,<0.80`, including the required
`KernelDescriptor.BuildOptLevel` binding. Main currently pins TTNN 0.79.0.
Our measurements use a different pinned tt-metal plus custom overlays.
An operation with the same name or shape is not proof of numerical or
performance equivalence. Audit native/generic operations and compute fidelity
before moving kernels or declaring these figures reproducible upstream.

Qwen3-32B's conventional attention model is useful scaffolding, but this
27B model has 48 GDN layers and 16 full-attention layers. GDN recurrence,
causal convolution carry, and hybrid cache ownership are essential contracts.

## Cold-prompt measurement, including the 7k+ prefill result

ThinkingCap G2 is a Qwen3.8-27B fine-tune, packed bfp4 weights (GPTQ-snapped),
with K bfp8 / V bfp4 and MTP. It is not the untouched FP8 checkpoint.
The tested arm is `s12phv-1m-onestep-pack-l1234`; records identify it as the
gated twin of the production arm. Measurement date: 2026-11-03, one repetition
per cell, cold salted prompts, 1024 output tokens, EOS ignored,
T=1.0 / top_p=0.95 / top_k=20, exact speculative acceptance.

| Concurrent requests | Requested prompt tokens each | PP tok/s | TG tok/s/user | Worst TTFT s |
|---|---|---|---|---|
| 1 | 4096 | 5,313 | 68.1 | 0.771 |
| 1 | 32768 | 7,375 | 63.7 | 4.443 |
| 4 | 4096 | 5,923 | 75.1 | 2.766 |
| 4 | 32768 | 7,218 | 63.0 | 18.158 |
| 8 | 4096 | 5,561 | 60.8 | 5.892 |
| 8 | 32768 | 7,164 | 53.4 | 36.591 |

PP is **B × requested prompt tokens / worst TTFT**, so B4/B8 are aggregate
prompt throughput, not a per-user prefill rate. Actual tokenized lengths were
calibrated near the requested lengths. TG comes from the record's decode
window; it includes the speculative system's emitted tokens. These are not
warm-prefix-hit figures or isolated kernel timings. Single repetitions do
not establish variance or a complete performance envelope.

The associated gate record reports GSM8K 59/60, retrieval, agent-session and
equivalence passes. This gate is evidence for that exact serving arm; it does
not establish bitwise equivalence to upstream or to the original FP8 model.

## What produced the latest prefill improvement

The measured comparison against the earlier 1M-pool arm improved T=1.0 PP
by 7.4–10.3% at 32k and 19.1–34.8% at 4k. The important mechanism was host
orchestration and state checkpoint traffic, in addition to device kernels:

- `QWEN36_PREFIX_ONESTEP`: run an align-mode prompt in one engine step while
  taking GDN checkpoints inside that step.
- `QWEN36_PREFIX_SNAP_PACK`: pack eight GDN layers into a checkpoint read.
- `QWEN36_PREFIX_CKPT_DEFER`: stage checkpoints in a bounded, four-entry
  device pool and read them back at the next engine step. The cited pool
  costs 151.5 MiB per card.
- `QWEN36_PREFIX_HEAD_DEFER` and `QWEN36_KV_TIER_STORE_DEFER`: defer related
  host work rather than dispatching it synchronously in the prefill path.

These changes shift work as well as remove overhead: TG/user fell 0.5–2.7%
in the comparison because a roughly 0.4 s checkpoint readback landed in the
first decode step. Measure TTFT, first decode gap, full generation makespan,
and steady decode together. Moving a cost beyond TTFT does not erase it.

## Contracts to preserve during the port

### Recurrent state, padding and slot ownership

GDN needs explicit recurrence and convolution carry for every request.
Use separate B=1 prefill scratch and persistent batched decode slots. A
prefill reset must not clear another request's live state. Copy final scratch
state into its destination slot before the next request resets the scratch.

A padded token is not automatically inert in a recurrence. Mask invalid rows
in the scan, state update and convolution finalization. Cover partial tails,
exact chunk multiples, slot reuse, and new arrivals during ongoing decode.
Test each batch width: the source implementation has observed greedy output
changes when batch width changes, even without a cache change.

### Traces and chunk boundaries

The source uses a 2048-token default prefill chunk and warmed masked buckets
128/256/512/1024/2048. Persistent trace input buffers must be refreshed on
every replay, including tokens, positions/RoPE, masks, and page tables.
Do not compile or allocate aliasing intermediates after parking a trace.
Restoring a prefix at a nonzero boundary must preserve recurrence rather than
zeroing it as if this were a fresh prompt.

### Hybrid prefix caching

Attention pages alone cannot resume this model: restore GDN recurrence and
conv carry at the same logical token boundary. Source align mode uses
2048-token blocks and GDN snapshots at matching boundaries. Its attention
block identifiers and GDN state identifiers have distinct capacity needs;
separate ID pools avoid wasting recurrent-state slots. Account for host
snapshot RAM separately from device KV and trace storage.

### Precision and topology traps worth regression tests

The following are source observations at its pinned runtime; re-establish
them at the selected upstream release rather than assuming they still apply:

- `k_chunk_size=32` silently corrupted non-causal masked SDPA. The source
  keeps its smallest masked bucket at 128 and requires safe SDPA chunking.
- LoFi plus `fp32_dest_acc_en=True` returned incorrect SDPA values, without
  necessarily producing NaNs. Test the combination, not just each knob.
- Fused prefill matmul/reduce-scatter hung on the 1x2 mesh configurations
  tested there. Collective worker placement is part of the execution contract.
- Freshly prefilled slots must be excluded from a decode reset that would
  otherwise overwrite the state just installed by prefill.
- Weight caches must include checkpoint identity and relevant quantization
  options. Same shapes and names can otherwise load a different model's cache.

## Suggested integration sequence

1. Establish geometry and mapping against the exact pinned checkpoint.
   Compare one attention layer and one GDN layer with CPU references, then
   compose the full model. Include planted wrong-axis/head/position cases.
2. Validate ordinary prefill then single-token decode, including recurrent
   state continuity and repeated requests. Keep trace and precision changes
   independently switchable.
3. Add masked tails, long-prompt chunks, TP=2 collectives and multiple slots.
   Run actual P150x2 hardware checks; simulator correctness is not timing or
   proof that the hardware cannot hang.
4. Add low-bit weights/KV behind paired accuracy gates, then prefix caching,
   trace capture and MTP. Compare at the same batch/context/sampling settings.
5. Profile request wall time as well as device time. Screen checkpoint
   packing/defer choices with end-to-end records before adopting them.

## Source map (pinned implementation and evidence)

The reference root is
[tt-stack at 782434b3](https://git.bitp.cz/bitpartner/tt-stack/src/commit/782434b31721d1232bda854c9848c5ff7b88caf5).
Paths below are relative to that source root:

| Area | Source paths |
|---|---|
| Measurement and scope | `docs/STATUS.md`, `docs/LEDGER.md`, `bench/runs/ttx-sweep32k-s12phv-1m-onestep-pack-l1234-ab20261103T194006Z-r0.jsonl` |
| Associated gate | `bench/runs/cold-pp-gate-s12-prefix-host-vsplit-sf-20261103T221004Z.jsonl` |
| Hybrid model, prefill scratch, checkpoint pack/defer | `patches/prod-local-metal/models/demos/blackhole/qwen36/tt/model.py` |
| GDN scan, decode and multi-token verification | Same directory: `gdn/tp.py`, `gdn/fused_chunk.py`, `gdn/state_once.py`, `gdn/conv_multi.py` |
| Trace input and model integration | Same directory: `masked_bucket_trace.py`, `qwen36_vllm.py`, `generator_interface.py` |
| Exact speculative sampling and state commit | Same directory: `mtp_batch.py`, `mtp_batch_host.py`, `gdn/state_slot.py` |
| Cache/runtime policy | `docs/SERVING.md`, `docs/MTP-BATCH.md`, `docs/KV-HOST-TIER.md` |

This contribution adds porting guidance and external evidence only. It adds
no production imports, support-manifest entry, or new hardware qualification.
