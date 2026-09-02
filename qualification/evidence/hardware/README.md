# Hardware evidence

This directory stores immutable runner JSON/stdout pairs and their canonical
indexes. Evidence is grouped by the exact code candidate it qualifies; later
report-only commits do not rewrite the recorded candidate SHA.

## `ba7abefba4484689c953ac53fe8810322db1d184`

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

Validate the bundle from the repository root:

```bash
python -B qualification/tools/validate_hardware_evidence.py \
  --candidate-sha ba7abefba4484689c953ac53fe8810322db1d184 \
  --evidence-root qualification/evidence/hardware/ba7abefba4484689c953ac53fe8810322db1d184 \
  --output qualification/evidence/hardware/ba7abefba4484689c953ac53fe8810322db1d184/index.json
```

Compiled cache trees and model weights are intentionally excluded.
