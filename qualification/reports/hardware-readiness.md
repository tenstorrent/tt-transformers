# Serialized hardware-qualification readiness

Status: **ready for externally synchronized, serialized execution; no hardware was contacted or run**

Artifacts:

- `qualification/manifests/hardware-matrix.json`
- `qualification/tools/run_hardware_matrix.py`
- `tests/host/test_hardware_matrix_runner.py`

## Matrix summary

The matrix contains 42 individually executable pytest nodes/processes, ordered by priority:

| Surface | Nodes |
|---|---:|
| Focused reusable modules | 30 |
| Runtime trace/order correctness | 1 |
| One-layer/model smoke | 3 |
| Token-accuracy/e2e | 8 |

| `MESH_DEVICE` | Nodes | Physical meaning |
|---|---:|---|
| `N150` | 9 | Logical one-chip regression submesh on the physical T3K host; not standalone-N150 product evidence. |
| `N300` | 6 | A recorded physical N300 left/right board pair within the T3K host. |
| `T3K` | 8 | Full eight-device Wormhole T3K. |
| `P150` | 8 | One P150 die selected on the eight-P150 development loudbox. |
| `P150x4` | 11 | Logical 1x4 Ring requiring physical `P150_X4` or `P300_X2` provenance; arbitrary four-device submeshes are rejected. |

Each node records an exact test target, optional `-k` selector, timeout, environment, cache rule, allowed machine pool, physical-SKU constraint, acceptance types, source basis, and the complete evidence schema.

All support manifests remain `experimental`; appearing in this readiness matrix is not a support or pass claim.

## Authoritative inputs

The matrix reconciles:

- the authoritative `TTTV2_HARDWARE_INSTRUCTION_MANUAL.md` lifecycle, machine, topology, cache, timeout, and evidence rules;
- all twelve `examples/*/support.json` hardware declarations;
- `qualification/analysis/support/hardware_coverage.csv` physical/logical geometry inventory;
- `qualification/analysis/support/marker_inventory.csv` and current explicit pytest markers/parameters.

No reservation file was read. No SSH, `tt-smi`, pytest collection, device test, or hardware process was run.

## External gates remain mandatory

The runner deliberately does not reserve machines, read reservations, SSH, fetch, pull, push, copy, switch branches, stash, commit, reset, or clean.

Before `--execute`, an operator must independently complete the manual's reservation and participating-checkout synchronization procedure. The runner then requires:

- `--sync-gate-passed` as an explicit caller attestation;
- `--common-sha` containing the same 40-character lowercase full SHA;
- `--branch` and `--machine-identity`;
- a caller-captured physical-inventory JSON from the external `tt-smi -s` step.

Actual execution rechecks the local checkout branch/SHA against the attestation. A mismatch is a preflight refusal, not a test result.

## Physical inventory format

Example only—the operator supplies real values from the selected machine:

```json
{
  "captured_utc": "2026-09-02T00:00:00Z",
  "machine_identity": "bh-qb-05.yyz2.tenstorrent.com",
  "architecture": "blackhole",
  "physical_sku": "four physical P150B boards",
  "device_count": 4,
  "board_types": ["p150b"],
  "cluster_type": "P150_X4",
  "system_mesh": "2x2",
  "tt_visible_devices": null,
  "source_command": "tt-smi -s"
}
```

`bh-lb-11` P150x4 additionally requires a revalidated `selected_bdfs` list matching the four documented Ring BDFs. Quietboxes require `TT_VISIBLE_DEVICES` to remain unset. P150_X4 and physical P300_X2 results retain distinct provenance.

## Serialization and reset policy

The runner starts exactly one selected node with one `python -m pytest` process.

- Matrix validation rejects xdist/parallel arguments.
- Runtime refuses `PYTEST_XDIST_WORKER` or parallel `PYTEST_ADDOPTS`.
- An atomic cooperative lock defaults to `/tmp/tt-transformers-hardware.lock`; a second runner refuses before process creation.
- External reservation/synchronization must also ensure no non-runner TT process shares the cards.
- The runner never performs hardware reset.
- Timeouts terminate the pytest process group and classify the result as `hardware_lifecycle_failure`; an operator decides whether a later external reset is justified.
- Functional PCC/assertion/accuracy failures are not reset reasons.

## Cache and environment rules

Every process sets `MESH_DEVICE` exactly. Module nodes receive a writable node-local `TT_CACHE_PATH`. Model nodes require the shared offline HF cache and an established warm model/topology cache.

The matrix records model-specific cache semantics:

- Llama 3 8B uses a model root and appends the topology exactly once.
- Llama 3.3 70B and Qwen3 32B use the topology directory itself.
- A duplicated topology component is a preflight/cache error, not permission to rematerialize another full cache.

## Commands

Host-safe validation/listing:

```bash
python3 qualification/tools/run_hardware_matrix.py --validate
python3 qualification/tools/run_hardware_matrix.py --list
```

Dry-run after supplying a real external inventory file:

```bash
python3 qualification/tools/run_hardware_matrix.py --dry-run \
  --node wh-n150-rmsnorm-prefill \
  --sync-gate-passed \
  --common-sha <40_HEX_COMMON_SHA> \
  --branch <COMMON_BRANCH> \
  --machine-identity wh-lb-42.yyz2.tenstorrent.com \
  --physical-inventory /path/to/inventory.json \
  --checkout /path/to/tt_transformers \
  --output-dir /path/to/hardware-results
```

Execution uses the same arguments with `--execute`. It is intentionally not demonstrated or run in this readiness task.

Standalone failure classification for an existing log:

```bash
python3 qualification/tools/run_hardware_matrix.py \
  --classify-log /path/to/node.log --exit-code 1
```

## Evidence output

An executed node writes one complete stdout log and one JSON evidence record containing:

- UTC start/finish;
- branch and common full SHA;
- caller identity and actual FQDN;
- architecture, physical inventory, mesh, and visible devices;
- exact selector, argv, environment, and cache path;
- exit code and separate failure classification;
- extracted PCC/accuracy/cache/timing/throughput metric lines;
- acceptance types;
- teardown status;
- reset record, always `performed=false` and `automatic=false`;
- stdout and evidence paths.

Result classifications are: `passed`, `functional_failure`, `hardware_lifecycle_failure`, `missing_acceptance_data`, `unimplemented_gate`, `different_hardware_deferred`, `preflight_refusal`, and `not_executed_dry_run`.

An e2e process that exits zero but emits no recognizable acceptance metric is classified `missing_acceptance_data`, not passed.

## Host validation

```bash
python3 qualification/tools/run_hardware_matrix.py --validate
PYTHONPATH=src:. pytest -q --confcutdir=tests/host \
  tests/host/test_hardware_matrix_runner.py
env PYTHONPYCACHEPREFIX=/tmp/gwang/tttv2_hw_matrix_pycache \
  python3 -m compileall -q \
  qualification/tools/run_hardware_matrix.py \
  tests/host/test_hardware_matrix_runner.py
```

The tests cover schema validation, five-mesh coverage, dry-run serialization, synchronization/SHA/machine refusal, lock ownership, parallel-option refusal, physical P150x4 provenance refusal, failure classification, evidence completeness, and the no-auto-reset policy.

## Evidence boundary

This deliverable proves readiness logic only. It contains no reservation state and no hardware result. Every future run must still pass the external synchronization and physical-inventory gates and must be reported against its actual common SHA and physical machine.
