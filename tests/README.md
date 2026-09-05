# Tests

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

Every hardware process must set `MESH_DEVICE` explicitly and must run through
the exact matrix selector. Run at most one TT process per physical host. Two
different physical hosts may run independently. Never use an automatic reset;
reset only after a confirmed lifecycle failure and after the failed process
has exited.

Full-model tests require their documented checkpoint, tokenizer, offline/cache
configuration, and writable TT cache. A collected, skipped, or xfailed test is
not hardware evidence.
