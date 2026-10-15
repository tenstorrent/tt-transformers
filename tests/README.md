# Tests

The test suites check standalone model execution, shared runtime behavior,
and package policy. Hardware qualification runs ordinary target execution
before speculative execution so that target or environment failures remain
distinguishable from verification failures.

## Layout

- `host/`: package, configuration, policy, and pure host behavior
- `modules/`: reusable module correctness and configuration
- `llm_runtime/`: model-neutral runtime behavior
- `models/`: model-owned configuration and execution contracts
- `integration/`: cross-layer behavior
- `hardware/`: exact device/model wrappers and the hardware matrix
- `qualification/`: readiness-client and capability validation
- `support/`: reusable test-only helpers
- `assets/`: small reference inputs, outputs, and model configuration used by
  tests

## Host tests

```bash
python -m pytest -m host
```

Host tests must not open a Tenstorrent device. Python 3.10 and 3.12 are the
supported host-test interpreters.

## Markers

Tests use `host`, `device`, `slow`, `model`, `wormhole`, `blackhole`, `n150`,
`n300`, `t3k`, `p150`, `p300`, and `p150x4`. Select slow tests through pytest
markers, for example `-m slow` or `-m 'host and not slow'`.

## Hardware tests

Validate or list the checked-in matrix without opening a device:

```bash
python qualification/tools/run_hardware_matrix.py --validate
python qualification/tools/run_hardware_matrix.py --list
```

The active matrix contains 51 nodes: N150 14, N300 6, T3K 8, P150 12, and
P150x4 11. The Llama HF generation gates cover single prompts, repeated calls,
left-padded batches, and traced device-greedy generation compared with an
explicit eager host-argmax reference on N150 and P150. The cached HF
teacher-forcing gates run the existing demo accuracy thresholds on both
architectures. Historical evidence in [validation](../docs/validation.md)
applies only to its recorded source revision and selectors.

Every hardware process must set `MESH_DEVICE` explicitly and must run through
the exact matrix selector or a documented qualification workflow. Run at most one TT process per physical host. Two
different physical hosts may run independently. Never use an automatic reset;
reset only after a confirmed lifecycle failure and after the failed process
has exited.

Full-model tests require their documented checkpoint, tokenizer, offline/cache
configuration, and writable TT cache. A collected, skipped, or xfailed test is
not hardware evidence.

The [Llama CPU n-gram qualification workflow](../docs/llama32-ngram.md)
owns the 18 cases in `hardware/models/llama32_1b/test_speculation.py`.
Run `.github/scripts/qualify_llama32_ngram.py` for that suite. The qualification
script runs ordinary serving first, requires every device case to pass without
skips, then checks n-gram serving and request lifecycle behavior. The Llama
speculation suite is separate from the public-demo hardware matrix selectors.
