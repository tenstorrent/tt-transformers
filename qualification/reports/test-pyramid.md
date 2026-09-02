# TTTv2 standalone test pyramid and marker policy

## Status

Authority inputs are the pinned Phase 0 `marker_inventory.csv` (23 tokens),
`test_inventory.csv` (1,368 source definitions), and
`hardware_coverage.csv` (53 declared coverage rows). The audit checks those
input counts before grading the expanded standalone tree.

This report defines selection policy; it is not hardware evidence. The static
taxonomy audit currently covers 1,452 source-level test functions:

- 1,198 explicitly `host`;
- 254 explicitly `device`;
- 390 concrete `model` surfaces;
- 27 explicitly `slow` source definitions;
- exact static topology marks where a node name/path proves a SKU, plus
  collection-time topology marks when `MESH_DEVICE` proves the selected
  architecture and SKU.

Parameterized collection expands those source definitions into more nodes.
The marker application did not change function names, argument names,
parameter values/IDs, assertions, skip/xfail behavior, or hardware commands.

## Required taxonomy

Every test function has exactly one execution-lane mark:

- `host`: does not open a Tenstorrent device;
- `device`: may open one or more Tenstorrent devices.

Orthogonal marks are:

- `slow`: long-running gate;
- `model`: concrete model/executor/demo surface;
- `wormhole` / `blackhole`: architecture selection;
- `n150`, `n300`, `t3k`, `p150`, `p300`, `p150x4`: SKU/logical-product
  selection.

An SKU mark implies `device` and its architecture. Host tests never receive
architecture/SKU marks merely because a test name discusses a topology.
Generic cross-architecture device tests remain architecture-unclaimed until a
scheduler explicitly sets `MESH_DEVICE`.

`tests.support.marker_policy.selected_topology_marks` maps only explicit
`MESH_DEVICE` values. The collection hook runs first and applies those marks to
already-declared device nodes without querying hardware. Blank or unknown
values add nothing; path inference cannot manufacture support.

## Per-PR host gates

The single consolidated `.github/workflows/host.yml` declares exact CPython
3.10.19 and 3.12.13 jobs. Each leg installs its validated target-specific
version constraints, runs repository policy/manifest/API/compatibility
validators, builds and audits the wheel and sdist, installs the wheel
non-editably, runs the complete selector below without `PYTHONPATH`, and probes
a second base-only installed environment:

```bash
python -m pytest -q -m host
```

The separate `device-policy` job performs static taxonomy and hardware-matrix
validation only; it does not execute device tests. The consolidated workflow is
checked in but has not yet produced an observed required CI result. Ruff,
formatting, and mypy are also deferred pending a reviewed extracted-code
baseline; see `qualification/reports/ci-and-constraints.md`.

## Focused hardware selection

Run these only in a TTNN environment on reserved hardware. These are selection
examples, not recorded results:

```bash
# Wormhole one-chip reusable modules
MESH_DEVICE=N150 pytest -q -m 'device and wormhole and n150 and not model' tests/modules

# Wormhole T3K model gates
MESH_DEVICE=T3K pytest -q -m 'device and model and wormhole and t3k' tests/hardware/models

# Blackhole one-die model/module gates
MESH_DEVICE=P150 pytest -q -m 'device and blackhole and p150' tests/modules tests/hardware/models

# Blackhole four-die model/module gates
MESH_DEVICE=P150x4 pytest -q -m 'device and blackhole and p150x4' tests/modules tests/hardware/models
```

The explicit environment lets the collection hook prove and attach topology
marks. A generic device node can then participate in the selected SKU lane;
tests retain their existing geometry guards and skips.

## Scheduled and release gates

Scheduled model characterization selects slow device/model nodes for one
declared topology at a time:

```bash
MESH_DEVICE=T3K pytest -q -m 'device and slow and model and wormhole and t3k' tests/hardware/models
MESH_DEVICE=P150x4 pytest -q -m 'device and slow and model and blackhole and p150x4' tests/hardware/models
```

Release qualification must use an reviewed exact-node manifest and invoke one
node per pytest process. Do not add `pytest-xdist`, shell backgrounding, or
parallel matrix processes on the same cards. For each node:

1. verify the synchronized full source/package SHA;
2. launch one pytest node in one process;
3. wait for teardown and record exit status/log/evidence;
4. proceed to the next node only after resources close;
5. use `tt-smi -r` only after a confirmed lifecycle/device fault, never for a
   PCC, accuracy, assertion, or ordinary functional failure.

The hardware instruction manual remains authoritative for reservation, SSH,
physical-SKU provenance, cache, and evidence fields.

## Serialized hardware runner coverage

`qualification/manifests/hardware-matrix.json` and
`qualification/tools/run_hardware_matrix.py` define 42 individually executable
one-process nodes:

- 30 reusable-module nodes;
- one runtime trace/order node;
- three model-smoke nodes;
- eight end-to-end/token-accuracy nodes;
- 23 Wormhole and 19 Blackhole nodes across N150, N300, T3K, P150, and P150x4.

The runner is host-validated by ten test definitions in
`tests/host/test_hardware_matrix_runner.py`. It validates and lists the matrix,
supports a no-device dry run, requires external synchronization and physical
inventory attestation before execution, owns a single-process lock, rejects
parallel pytest options, records acceptance/teardown evidence, and never
reserves, connects to, or resets hardware itself.

The runner accepts exit code zero as `passed` only when pytest's final terminal
summary contains at least one passing test. All-skipped/all-xfailed selections,
empty selections, and missing terminal summaries are `no_passing_tests` and
cannot become hardware evidence.

Host-safe runner checks are:

```bash
python3 qualification/tools/run_hardware_matrix.py --validate
python3 qualification/tools/run_hardware_matrix.py --list
PYTHONPATH=src:. pytest -q --confcutdir=tests/host \
  tests/host/test_hardware_matrix_runner.py
```

The exact execution and physical-inventory contract is documented in
`qualification/reports/hardware-readiness.md`. Matrix presence and dry-run
success are readiness evidence only, never a device-support claim.

## Evidence rule

A collected node is only a candidate. A skipped or xfailed device node is not
evidence that the architecture, SKU, mesh, TP, DP, trace, cache, sampling, or
model path works. Support requires a non-skipped passing run at the exact
recorded full SHA with physical machine/SKU, environment, node ID, metrics,
teardown status, and log path.

The pinned extraction SHA currently has no attributable hardware evidence;
see `qualification/analysis/support/hardware_evidence.csv`.

## Audit

Static audit:

```bash
python -B qualification/tools/audit_test_taxonomy.py --json
```

Host collection audit:

```bash
python -B qualification/tools/audit_test_taxonomy.py --collect-host
```

The audit fails on missing/contradictory host/device lanes, missing `model` or
`slow` marks in hardware model wrappers, invalid SKU implications, missing
marker registration, or wrappers lacking explicit `MESH_DEVICE` topology
augmentation.
