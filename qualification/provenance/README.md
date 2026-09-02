# TTTv2 source provenance inventory

This directory freezes the source-to-destination classification for the standalone extraction described by `TTTV2_MIGRATION_PLAN.md`.

- Source repository: `/localdev/gwang/tt-metal`
- Source revision: `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`
- Inventory rows: **365**
- Rows assigned a destination: **339**
- Explicitly excluded rows: **26**
- Boundary-cleanup rows: **86**
- Generated: 2026-09-02 (UTC)

`source_inventory.csv` is the machine-readable authority. Its columns are:

| Column | Meaning |
|---|---|
| `source_path` | Path relative to the pinned `tt-metal` tree. |
| `destination_path` | Planned standalone path. A semicolon separates destinations when a file must be split. Empty only for an exclusion. |
| `git_blob_sha` | Git blob object ID at the pinned revision, not a worktree checksum. |
| `disposition` | One of `copied`, `split`, `renamed`, `replaced`, or `excluded`. |
| `reason` | Why the file moves, splits, is structurally replaced, or is excluded. |
| `category` | Stable grouping used by coverage checks. |
| `boundary_cleanup` | `true` when Phase 3 must remove an old namespace, fixture, test-helper, broad utility, or duplicate compatibility surface. |

## Coverage summary

Disposition counts:

| Disposition | Count |
|---|---:|
| `excluded` | 26 |
| `renamed` | 292 |
| `replaced` | 5 |
| `split` | 42 |

Category counts:

| Category | Count |
|---|---:|
| `boundary_source` | 4 |
| `boundary_support` | 5 |
| `example_support` | 6 |
| `excluded_moe` | 26 |
| `model_documentation` | 12 |
| `model_hybrid_demo` | 2 |
| `production_documentation` | 5 |
| `production_model` | 74 |
| `production_modules` | 16 |
| `production_runtime` | 21 |
| `production_sampling` | 8 |
| `production_support` | 3 |
| `qualification_asset` | 21 |
| `qualification_documentation` | 1 |
| `qualification_manifest` | 5 |
| `qualification_schema` | 1 |
| `qualification_tool` | 29 |
| `source_ci_manifest` | 7 |
| `test` | 101 |
| `test_config` | 2 |
| `test_fixture` | 3 |
| `test_hybrid_demo` | 12 |
| `test_support` | 1 |

The explicit production roots are complete at the pinned tree:

- `models/common/modules/`: 17 retained non-MoE files and 23 excluded MoE files.
- `models/common/llm_runtime/`: 24 files.
- Shared executors plus all twelve concrete model directories: 88 files.
- `models/common/tests/`: 124 retained files and 3 MoE tests excluded with MoE.
- `models/common/sampling/`: 9 files.
- `models/common/readiness_check/`: 15 qualification tools/assets and 6 readiness tests.
- Narrow production/validation support, fixtures, qualification manifests/tools, model reference artifacts, prompt/corpus/model-parameter assets, and source-CI evidence: 77 files.

Per-model production and directly owned test/demo source counts:

| Model | Production/docs | Tests/demos |
|---|---:|---:|
| `deepseek_r1_distill_qwen_14b` | 7 | 6 |
| `llama32_1b` | 7 | 5 |
| `llama32_3b` | 7 | 5 |
| `llama33_70b` | 7 | 9 |
| `llama3_8b` | 5 | 6 |
| `mistral_7b` | 7 | 4 |
| `phi4` | 7 | 4 |
| `qwen25_72b` | 8 | 5 |
| `qwen25_7b` | 7 | 3 |
| `qwen25_coder_32b` | 8 | 4 |
| `qwen2_7b` | 7 | 5 |
| `qwen3_32b` | 8 | 6 |

The inventory includes every file under each explicit production/runtime/model/test root. MoE's 26 rows are present rather than silently omitted, and every one is classified `excluded`. No other row is classified `excluded`.

## Scope boundary

The plan's phrase “MoE is the only planned directory exclusion” applies to the TTTv2 source set. It does not make every unrelated `tt-metal/models` product part of TTTv2. Multimodal/Mixtral TTTv1 code, the experimental Quasar fork, and unrelated model suites are outside this product definition rather than exclusions from it. The generic readiness-check suite is included because it is qualification infrastructure retained by the migration plan.

Four TTTv1 source files are nevertheless inventoried because current TTTv2 files import a narrow surface from them:

- `models/tt_transformers/tt/common.py`: the model-neutral Mode value used by reusable dispatch;
- `models/tt_transformers/tt/generator.py`: `create_submeshes`;
- `models/tt_transformers/tt/model_config.py`: `OpGroup`/`TensorGroup` used only by `from_model_args` bridges;
- `models/tt_transformers/tt/rope.py`: `compute_gather_cos_sin` used only by the RoPE bridge.

These are `split` or `replaced`, never copied wholesale. The CSV also flags all ten reusable-module `from_model_args` surfaces, twelve generator submesh imports, hybrid pytest demos, production imports from test helpers, the broad `utility_functions.py` dependency, duplicate `SamplingParams` ownership, and tt-metal fixture/qualification path coupling.

A transitive owner audit also includes `models/common/utils.py`. TTTv2 imports
only its `LogProbsCalculator` compatibility export, so that row is split to the
narrow `tt_transformers.sampling.logprobs` owner; the unrelated legacy
top-k/top-p filtering function is not part of the reusable foundation.

## Mapping rules

- Reusable non-MoE modules move to `src/tt_transformers/modules/`.
- Runtime code moves to `src/tt_transformers/llm_runtime/`.
- Shared and concrete model implementation moves to `src/tt_transformers/models/`.
- Model READMEs move to `examples/<model>/README.md`.
- Hybrid pytest demos are split between `examples/<model>/` and `tests/hardware/models/<model>/`.
- Module, runtime, and model tests preserve their relative ownership under `tests/`.
- Validation, reference-generation, and hardware-gate tooling moves under `qualification/tools/`; schemas and manifests move under `qualification/schemas/` and `qualification/manifests/`.
- Readiness contracts, runners, prompts, and references move under `qualification/readiness/`; their tests move under `tests/qualification/readiness/`.
- Broad tt-metal helpers and fixtures are marked `split` so only the imported TTTv2 surface is internalized.
- A renamed row means a mechanical content-preserving path/namespace move. A later import rewrite does not by itself make it a behavior-changing replacement.

## Ambiguities to resolve during extraction

1. The final helper filenames for lifecycle cleanup, page-table construction, validation metrics, and benchmark support may change. The CSV assigns an owner and provisional destination now; any Phase 2 change must update provenance before code moves.
2. The plan requires deliberate removal of TTTv1-only `from_model_args` bridges, but the exact compatibility release is not decided. Their source files and external helper owners are explicitly marked `boundary_cleanup=true`.
3. Current demo modules combine CLI/example logic, pytest fixtures, correctness gates, and performance gates. Their `split` destinations express ownership, not a behavior redesign.
4. `models/tttv2_vllm_hardware_gate_runner.sh` names expectation JSON files that do not exist at the pinned commit. The inventory does not invent provenance rows for absent files; Phase 0 qualification should record that gap.
5. `models/model_targets.yaml`, `models/model_trace_region_sizes.yaml`, and source-CI YAML files cover many unrelated models. Their disposition is `split`: only TTTv2 entries are retained.
6. The committed reference-output list contains one selected artifact for each of the twelve current models. Other TTTv1/multimodal/MoE artifacts in the same source directory are not TTTv2 qualification assets.

## Exact enumeration command

This is the command used to enumerate candidate paths and blob IDs before applying the mapping rules above:

```bash
SOURCE_REPO=/localdev/gwang/tt-metal
SOURCE_COMMIT=00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0

git -C "$SOURCE_REPO" ls-tree -r "$SOURCE_COMMIT" -- \
  'models/common/modules' \
  'models/common/llm_runtime' \
  'models/common/models/executor.py' \
  'models/common/models/llama3_executor.py' \
  'models/common/models/qwen2_executor.py' \
  'models/common/models/deepseek_r1_distill_qwen_14b' \
  'models/common/models/llama32_1b' \
  'models/common/models/llama32_3b' \
  'models/common/models/llama33_70b' \
  'models/common/models/llama3_8b' \
  'models/common/models/mistral_7b' \
  'models/common/models/phi4' \
  'models/common/models/qwen25_72b' \
  'models/common/models/qwen25_7b' \
  'models/common/models/qwen25_coder_32b' \
  'models/common/models/qwen2_7b' \
  'models/common/models/qwen3_32b' \
  'models/common/readiness_check' \
  'models/common/tests' \
  'models/common/sampling' \
  'models/common/README.md' \
  'models/common/auto_compose.py' \
  'models/common/device_utils.py' \
  'models/common/distribute_as.py' \
  'models/common/lightweightmodule.py' \
  'models/common/metrics.py' \
  'models/common/tensor_utils.py' \
  'models/common/utility_functions.py' \
  'models/common/utils.py' \
  'models/common/validation_tools.py' \
  'models/demos/utils/llm_demo_utils.py' \
  'models/demos/utils/model_targets.py' \
  'models/demos/utils/trace_region_sizes.py' \
  'models/perf/benchmarking_utils.py' \
  'conftest.py' \
  'models/conftest.py' \
  'models/model_targets.yaml' \
  'models/model_trace_region_sizes.yaml' \
  'models/test_tttv2_validate_vllm_matrix.py' \
  'models/tttv2_bh_required_capabilities.schema.json' \
  'models/tttv2_llama33_70b_bh_required_capabilities.json' \
  'models/tttv2_llama3_8b_bh_required_capabilities.json' \
  'models/tttv2_qwen3_32b_bh_required_capabilities.json' \
  'models/tttv2_validate_bh_required_capabilities.py' \
  'models/tttv2_validate_vllm_matrix.py' \
  'models/tttv2_vllm_hardware_gate_runner.sh' \
  '.github/workflows/t3000-unit-tests.yaml' \
  '.github/workflows/silencer.lock.yml' \
  '.github/workflows/test-command.lock.yml' \
  'tests/pipeline_reorg/models_e2e_tests.yaml' \
  'tests/pipeline_reorg/models_sweep_tests.yaml' \
  'tests/pipeline_reorg/models_unit_tests.yaml' \
  'tests/scripts/t3000/run_t3000_unit_tests.sh' \
  'models/tt_transformers/tt/common.py' \
  'models/tt_transformers/tt/generator.py' \
  'models/tt_transformers/tt/model_config.py' \
  'models/tt_transformers/tt/rope.py' \
  'models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json' \
  'models/tt_transformers/demo/sample_prompts/eval_repeat_prompts_batch32.json' \
  'models/tt_transformers/model_params/Qwen3-32B/config.json' \
  'models/tt_transformers/tests/generate_reference_hf.py' \
  'models/tt_transformers/tests/generate_reference_outputs.py' \
  'models/tt_transformers/tests/generate_reference_outputs.sh' \
  'models/tt_transformers/tests/tale-of-two-cities.txt.bz2' \
  'models/tt_transformers/tests/reference_outputs/DeepSeek-R1-Distill-Qwen-14B.refpt' \
  'models/tt_transformers/tests/reference_outputs/Llama-3.2-1B-Instruct.refpt' \
  'models/tt_transformers/tests/reference_outputs/Llama-3.2-3B-Instruct.refpt' \
  'models/tt_transformers/tests/reference_outputs/Llama-3.3-70B-Instruct.refpt' \
  'models/tt_transformers/tests/reference_outputs/Llama-3.1-8B-Instruct.refpt' \
  'models/tt_transformers/tests/reference_outputs/Mistral-7B-Instruct-v0.3.refpt' \
  'models/tt_transformers/tests/reference_outputs/phi-4.refpt' \
  'models/tt_transformers/tests/reference_outputs/Qwen2.5-72B-Instruct.refpt' \
  'models/tt_transformers/tests/reference_outputs/Qwen2.5-7B-Instruct.refpt' \
  'models/tt_transformers/tests/reference_outputs/Qwen2.5-Coder-32B-Instruct.refpt' \
  'models/tt_transformers/tests/reference_outputs/Qwen2-7B-Instruct.refpt' \
  'models/tt_transformers/tests/reference_outputs/Qwen3-32B.refpt'
```

It must print 365 rows. The pathspec deliberately includes the entire module, readiness, and common-test roots so the MoE exclusion and qualification coverage are auditable, while model products and old TTTv1 assets are selected narrowly.

## Verification commands

Run from the `tt_transformers` repository root. This checks header/schema, row uniqueness, every blob ID, dispositions, exclusion policy, explicit-root completeness, all twelve model directories, and summary counts:

```bash
python3 - <<'PY'
import csv
import subprocess
from collections import Counter
from pathlib import Path

SOURCE_REPO = Path("/localdev/gwang/tt-metal")
REV = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
CSV = Path("qualification/provenance/source_inventory.csv")
MODELS = {
    "deepseek_r1_distill_qwen_14b", "llama32_1b", "llama32_3b",
    "llama33_70b", "llama3_8b", "mistral_7b", "phi4", "qwen25_72b",
    "qwen25_7b", "qwen25_coder_32b", "qwen2_7b", "qwen3_32b",
}

with CSV.open(newline="") as stream:
    rows = list(csv.DictReader(stream))

assert list(rows[0]) == [
    "source_path", "destination_path", "git_blob_sha", "disposition",
    "reason", "category", "boundary_cleanup",
]
assert len(rows) == 365
paths = [row["source_path"] for row in rows]
assert len(paths) == len(set(paths))
assert Counter(row["disposition"] for row in rows) == {"excluded":26,"renamed":292,"replaced":5,"split":42}
assert sum(row["boundary_cleanup"] == "true" for row in rows) == 86

for row in rows:
    actual = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "rev-parse", f"{REV}:{row['source_path']}"],
        text=True,
    ).strip()
    assert actual == row["git_blob_sha"], row["source_path"]
    assert row["disposition"] in {"copied", "split", "renamed", "replaced", "excluded"}
    assert bool(row["destination_path"]) == (row["disposition"] != "excluded")

excluded = [row for row in rows if row["disposition"] == "excluded"]
assert len(excluded) == 26
assert all(
    row["source_path"].startswith("models/common/modules/moe/")
    or row["source_path"].startswith("models/common/tests/modules/moe/")
    for row in excluded
)

inventory = set(paths)
def source_paths(prefix):
    output = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "ls-tree", "-r", "--name-only", REV, prefix],
        text=True,
    )
    return {line for line in output.splitlines() if line}

for prefix in (
    "models/common/modules",
    "models/common/llm_runtime",
    "models/common/readiness_check",
    "models/common/tests",
):
    missing = source_paths(prefix) - inventory
    assert not missing, (prefix, sorted(missing))

for model in MODELS:
    prefix = f"models/common/models/{model}"
    missing = source_paths(prefix) - inventory
    assert not missing, (prefix, sorted(missing))

assert len(MODELS) == 12
print(f"verified {len(rows)} rows; {len(excluded)} explicit MoE exclusions")
PY
```

For a quick CSV-only policy audit:

```bash
python3 - <<'PY'
import csv
from collections import Counter

with open("qualification/provenance/source_inventory.csv", newline="") as stream:
    rows = list(csv.DictReader(stream))
print("rows", len(rows))
print("dispositions", dict(sorted(Counter(r["disposition"] for r in rows).items())))
print("categories", dict(sorted(Counter(r["category"] for r in rows).items())))
print("boundary_cleanup", sum(r["boundary_cleanup"] == "true" for r in rows))
PY
```
