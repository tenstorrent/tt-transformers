# Repository examples

Examples are runnable from a repository checkout and are not installed by the
wheel. Every model is currently experimental.

## Set up

```bash
git clone https://github.com/tenstorrent/tt_transformers.git
cd tt_transformers
python -m pip install -e '.[examples,test]'
```

Each model README documents its `HF_MODEL`, pinned revision, `MESH_DEVICE`,
cache requirements, supported arguments, and exact commands.

| Example | Current validated subset |
|---|---|
| [DeepSeek R1 Distill Qwen 14B](deepseek_r1_distill_qwen_14b/README.md) | no model-family cell in the current 42-node regression matrix |
| [Llama 3.2 1B](llama32_1b/README.md) | N150 and N300 token accuracy |
| [Llama 3.2 3B](llama32_3b/README.md) | no model-family cell in the current matrix |
| [Llama 3.3 70B](llama33_70b/README.md) | T3K token accuracy; P150_X4 smoke and token accuracy |
| [Llama 3.1 8B](llama3_8b/README.md) | logical P150 token accuracy |
| [Mistral 7B](mistral_7b/README.md) | no model-family cell in the current matrix |
| [Phi-4](phi4/README.md) | no model-family cell in the current matrix |
| [Qwen2.5 72B](qwen25_72b/README.md) | no model-family cell in the current matrix |
| [Qwen2.5 7B](qwen25_7b/README.md) | N300 token accuracy |
| [Qwen2.5 Coder 32B](qwen25_coder_32b/README.md) | T3K one-layer smoke |
| [Qwen2 7B](qwen2_7b/README.md) | no model-family cell in the current matrix |
| [Qwen3 32B](qwen3_32b/README.md) | T3K token accuracy; P150_X4 smoke and token accuracy |

## Validation boundary

The current regression record is 42/42 passing at code SHA
`9134c399334240e3e4d35dfa93013c6c7293a3d1`: modules 30/30, runtime 1/1,
smoke 3/3, and end-to-end 8/8. See
[`docs/validation.md`](../docs/validation.md) for topology qualifications and
evidence hashes.

A passing subset does not qualify a complete model contract. The canonical
status and declared candidate geometries are summarized in
[`SUPPORT.md`](../SUPPORT.md); machine-readable details remain in each
`support.json`.

## Test wrappers

Hardware pytest wrappers live under `tests/hardware/models`. Use the checked-in
matrix runner rather than inventing selectors.

### Inspect the matrix without opening a device

```bash
python qualification/tools/run_hardware_matrix.py --validate
python qualification/tools/run_hardware_matrix.py --list
```

`--validate` checks the matrix, selectors, supported meshes, machine pools,
and evidence contracts. `--list` prints nodes in execution order. Both are
host-safe and do not import or open a TT device.

### Prepare a hardware run

Before using `--dry-run` or `--execute`:

1. Reserve the physical host and ensure no other TT process is using it.
2. Use a clean checkout at the exact commit to be tested. Verify its branch,
   full SHA, upstream, and `0 0` upstream divergence.
3. Install this checkout into an environment containing `ttnn==0.77.0` and
   the test dependencies.
4. Run `tt-smi -s` and save a physical-inventory JSON document outside the
   repository. It must include at least:

   ```json
   {
     "captured_utc": "2026-01-01T00:00:00Z",
     "machine_identity": "<allowed identity from the matrix>",
     "architecture": "wormhole",
     "physical_sku": "<physical system description>",
     "device_count": 8,
     "board_types": ["n300 L", "n300 R"],
     "cluster_type": "T3K",
     "system_mesh": "1x8",
     "tt_visible_devices": null,
     "source_command": "tt-smi -s"
   }
   ```

The inventory must describe the physical host, not merely the logical
`MESH_DEVICE` requested by one node. `--sync-gate-passed` is an attestation
that these checkout checks were actually performed; it is not a bypass.

### Preview one exact node

Set the run identity once, then use `--dry-run` to verify the resolved command,
environment, cache path, machine, and physical inventory without opening a
device:

```bash
CHECKOUT="$(pwd)"
BRANCH="$(git branch --show-current)"
CANDIDATE_SHA="$(git rev-parse HEAD)"
PYTHON="$(command -v python)"
MACHINE_IDENTITY="<allowed identity from tests/hardware/hardware-matrix.json>"
PHYSICAL_INVENTORY="<absolute path to physical-inventory.json>"
RESULTS_ROOT="<absolute path outside the repository>"

"${PYTHON}" qualification/tools/run_hardware_matrix.py \
  --dry-run \
  --node wh-n150-rmsnorm-prefill \
  --common-sha "${CANDIDATE_SHA}" \
  --branch "${BRANCH}" \
  --machine-identity "${MACHINE_IDENTITY}" \
  --sync-gate-passed \
  --physical-inventory "${PHYSICAL_INVENTORY}" \
  --checkout "${CHECKOUT}" \
  --output-dir "${RESULTS_ROOT}" \
  --python "${PYTHON}"
```

The runner rejects a checkout whose actual branch or SHA differs from the
attested values.

### Execute one node

After reviewing the dry-run output, replace `--dry-run` with `--execute` and
provide a host-scoped lock:

```bash
"${PYTHON}" qualification/tools/run_hardware_matrix.py \
  --execute \
  --node wh-n150-rmsnorm-prefill \
  --common-sha "${CANDIDATE_SHA}" \
  --branch "${BRANCH}" \
  --machine-identity "${MACHINE_IDENTITY}" \
  --sync-gate-passed \
  --physical-inventory "${PHYSICAL_INVENTORY}" \
  --checkout "${CHECKOUT}" \
  --output-dir "${RESULTS_ROOT}" \
  --lock-file /tmp/tt-transformers-hardware.lock \
  --python "${PYTHON}"
```

Each invocation runs exactly one pytest process and writes a paired JSON record
and complete stdout log. Exit zero means the node produced at least one passing
test and met its acceptance-data requirements. On a nonzero exit, inspect the
record's `failure_classification` before doing anything else.

### Execute one model node

For example, the following runs the Llama 3.1 8B token-accuracy gate as a
logical single-P150 workload on a compatible Blackhole P150_X4 host. Use that
host's allowed matrix identity and physical-inventory file in the variables
defined above:

```bash
"${PYTHON}" qualification/tools/run_hardware_matrix.py \
  --execute \
  --node bh-p150-llama3-8b-token-accuracy \
  --common-sha "${CANDIDATE_SHA}" \
  --branch "${BRANCH}" \
  --machine-identity "${MACHINE_IDENTITY}" \
  --sync-gate-passed \
  --physical-inventory "${PHYSICAL_INVENTORY}" \
  --checkout "${CHECKOUT}" \
  --output-dir "${RESULTS_ROOT}" \
  --lock-file /tmp/tt-transformers-hardware.lock \
  --python "${PYTHON}"
```

The checked-in node supplies the exact pytest selector, timeout,
`HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`, offline Hugging Face settings,
writable model-cache root, and `MESH_DEVICE=P150`. On a P150_X4 quietbox it
also requires `TT_VISIBLE_DEVICES` to remain unset. Review these resolved
values with the same command using `--dry-run` before executing it.

### Run every node assigned to one architecture

One host may process its assigned nodes sequentially with this Bash loop:

```bash
ARCHITECTURE=wormhole  # use blackhole on the Blackhole host

mapfile -t NODES < <("${PYTHON}" - "${ARCHITECTURE}" <<'PY'
import json
import sys

architecture = sys.argv[1]
with open("tests/hardware/hardware-matrix.json", encoding="utf-8") as stream:
    matrix = json.load(stream)
for node in sorted(matrix["nodes"], key=lambda item: item["priority"]):
    if node["architecture"] == architecture:
        print(node["id"])
PY
)

for node in "${NODES[@]}"; do
  "${PYTHON}" qualification/tools/run_hardware_matrix.py \
    --execute \
    --node "${node}" \
    --common-sha "${CANDIDATE_SHA}" \
    --branch "${BRANCH}" \
    --machine-identity "${MACHINE_IDENTITY}" \
    --sync-gate-passed \
    --physical-inventory "${PHYSICAL_INVENTORY}" \
    --checkout "${CHECKOUT}" \
    --output-dir "${RESULTS_ROOT}" \
    --lock-file /tmp/tt-transformers-hardware.lock \
    --python "${PYTHON}" || exit $?
done
```

Run at most one TT process per physical host. Independent physical hosts may
run their sequential loops concurrently. For a `P150` node on a compatible
P150_X4 quietbox, the checked-in matrix sets `MESH_DEVICE=P150` and unsets
`TT_VISIBLE_DEVICES`; do not add a competing topology selector.

### Validate and retain evidence

After copying each host's JSON/log pairs into a shared external evidence root,
build the canonical index:

```bash
python qualification/tools/validate_hardware_evidence.py \
  --candidate-sha "${CANDIDATE_SHA}" \
  --evidence-root "${EVIDENCE_ROOT}" \
  --output "${EVIDENCE_ROOT}/index.json"
sha256sum "${EVIDENCE_ROOT}/index.json"
```

Keep raw logs, caches, inventories, and archives outside Git. Publish only the
approved immutable artifact location, tested SHA, hashes, topology boundary,
and compact result summary.

Never reset hardware as routine cleanup. Use `tt-smi -r` only after the failed
pytest process has exited and its evidence confirms a hardware/lifecycle
failure. Functional PCC, accuracy, geometry, or assertion failures are not
reset conditions. A collected, skipped, or xfailed test is not validation
evidence.
