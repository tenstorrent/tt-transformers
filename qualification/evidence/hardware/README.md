# Hardware evidence

This directory stores immutable runner JSON/stdout pairs and their canonical
indexes. Evidence is grouped by the exact code candidate it qualifies; later
report-only commits do not rewrite the recorded candidate SHA.

## Current candidate: `b24eabe35c8f2c73f45493da40e5a6351eb0ec2d`

- 34 of 42 matrix nodes executed on reserved `wh-lb-42` and `bh-qb-05` hosts.
- 33 passed: 23 modules, three smokes, and seven end-to-end/token-accuracy
  nodes.
- The official strict `wh-t3k-runtime-trace-order` node is the sole functional
  failure: all four W6 order cases reported row-0 max-abs 1.5 against the 1.0
  limit and top-5 overlap 3 against the required 4.
- Eight single-P150 nodes were not run because their required `bh-lb-11`
  physical host was unavailable; P150_X4 results are not substituted.
- No hardware-lifecycle failure or reset occurred.

Each host directory contains the unmodified runner JSON and matching stdout
log. Absolute remote paths inside raw JSON are retained as provenance. The
top-level `index.json` normalizes repository-relative locations and binds every
pair with SHA-256 hashes.

Validate the current bundle from the repository root:

```bash
python -B qualification/tools/validate_hardware_evidence.py \
  --candidate-sha b24eabe35c8f2c73f45493da40e5a6351eb0ec2d \
  --evidence-root qualification/evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d \
  --output qualification/evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json
```

### Non-qualifying W6 diagnostic

At code candidate `d7677f822356e839f707a6447fd0abc89e620d56`, one
explicitly diagnostic W6 order passed the later trace, KV, replay, sampling,
resume, chunk, and cache invariants in 176.12 seconds after relaxing only the
known cross-geometry logits oracle. That run is not in a canonical index, is
not a qualification pass, and does not weaken or supersede the official strict
W6 functional failure at `b24eabe35c8f2c73f45493da40e5a6351eb0ec2d`.

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
