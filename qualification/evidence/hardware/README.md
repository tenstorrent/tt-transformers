# Hardware evidence

This directory stores immutable runner JSON/stdout pairs and their canonical
indexes. Evidence is grouped by the exact code candidate it qualifies; later
report-only commits do not rewrite the recorded candidate SHA.

## Current candidate: `2883a949860d749adc2ed1af5525b27a9a547505`

- 34 of 42 matrix nodes executed on reserved `wh-lb-42` and `bh-qb-05` hosts.
- All 34 executed nodes passed: 23 modules, one runtime trace/order gate, three
  smokes, and seven end-to-end/token-accuracy nodes.
- The official strict `wh-t3k-runtime-trace-order` node passed all four W6
  capture/sampling orders at the unchanged default thresholds. The accuracy
  profile kept folded QKV/W2 prefill on `ttnn.linear`; no environment or
  threshold override was used.
- Eight single-P150 nodes were not run because their required `bh-lb-11`
  physical host was unavailable; P150_X4 results are not substituted.
- No functional, hardware-lifecycle, pre-device, or missing-acceptance result
  and no reset occurred.

Each host directory contains the unmodified runner JSON and matching stdout
log. Absolute remote paths inside raw JSON are retained as provenance. The
top-level `index.json` normalizes repository-relative locations and binds every
pair with SHA-256 hashes.

Validate the current bundle from the repository root:

```bash
python -B qualification/tools/validate_hardware_evidence.py \
  --candidate-sha 2883a949860d749adc2ed1af5525b27a9a547505 \
  --evidence-root qualification/evidence/hardware/2883a949860d749adc2ed1af5525b27a9a547505 \
  --output qualification/evidence/hardware/2883a949860d749adc2ed1af5525b27a9a547505/index.json
```

### Noncanonical accuracy-TTFT diagnostic

The separate
[`accuracy-ttft/summary.json`](../diagnostics/2883a949860d749adc2ed1af5525b27a9a547505/accuracy-ttft/summary.json)
records four Llama-3.3-70B accuracy-profile performance cells on `wh-lb-42`.
TTFT passed 4/4 at 86.7–87.2 ms against the tolerance-adjusted 105 ms ceiling;
throughput failed its performance floor in all 4/4 cells. These pytest exits
are `performance_floor_failure` diagnostics, not correctness or lifecycle
failures. The summary is noncanonical, is excluded from the 34 hardware pass
count, and does not alter the canonical index.

## Superseded candidate: `b24eabe35c8f2c73f45493da40e5a6351eb0ec2d`

- 34 of 42 matrix nodes executed on the same two reserved host roles.
- 33 passed: 23 modules, three smokes, and seven end-to-end/token-accuracy
  nodes. The strict W6 runtime node was the sole functional failure.
- Eight single-P150 nodes were deferred for the same unavailable `bh-lb-11`
  role. No hardware-lifecycle failure or reset occurred.

Its immutable raw pairs and index remain under the candidate-named directory;
all ledger rows are now ineligible superseded history.

### Historical non-qualifying W6 diagnostic

At code candidate `d7677f822356e839f707a6447fd0abc89e620d56`, one
explicitly diagnostic W6 order passed the later trace, KV, replay, sampling,
resume, chunk, and cache invariants in 176.12 seconds after relaxing only the
known cross-geometry logits oracle. That run is not in a canonical index, is
not a qualification pass, and does not weaken or supersede the official strict
W6 functional failure recorded by the superseded `b24eabe...` bundle.

## Historical candidate: `ba7abefba4484689c953ac53fe8810322db1d184`

- 34 of 42 matrix nodes executed on reserved `wh-lb-42` and `bh-qb-05` hosts.
- 33 passed: 23 modules, three smokes, and seven end-to-end/token-accuracy
  nodes.
- `wh-t3k-runtime-trace-order` is the sole functional failure.
- Eight single-P150 nodes were not run because their required `bh-lb-11`
  physical host was unavailable; P150_X4 results are not substituted.
- No hardware-lifecycle failure or reset occurred.

Each host directory contains the unmodified runner JSON and matching stdout
log. Absolute remote paths inside raw JSON are retained as provenance. The
top-level `index.json` normalizes repository-relative locations and binds every
pair with SHA-256 hashes.

Validate the historical bundle from the repository root:

```bash
python -B qualification/tools/validate_hardware_evidence.py \
  --candidate-sha ba7abefba4484689c953ac53fe8810322db1d184 \
  --evidence-root qualification/evidence/hardware/ba7abefba4484689c953ac53fe8810322db1d184 \
  --output qualification/evidence/hardware/ba7abefba4484689c953ac53fe8810322db1d184/index.json
```

Compiled cache trees and model weights are intentionally excluded.
