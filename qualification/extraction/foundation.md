# Phase 2 reusable-foundation extraction

Source repository: `/localdev/gwang/tt-metal`

Source revision: `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`

Inventory authority: `qualification/provenance/source_inventory.csv`

Status: **mechanical foundation extraction complete; Phase 3 boundary closure pending**

## Result

This lane extracted the reusable package foundation only:

- 27 direct production inventory rows: 16 `production_modules`, 8 `production_sampling`, and 3 `production_support`;
- five narrow split surfaces needed by this layer;
- 28 newly authored Python files plus the scaffold-owned sampling `__init__.py` merge;
- no MoE, runtime, concrete model, test, or example extraction.

The existing `src/tt_transformers/__init__.py` and
`src/tt_transformers/modules/__init__.py` were left untouched. The existing
sampling package docstring was retained and merged with the pinned lazy export
policy rather than overwritten.

## Direct snapshot rows

All 27 blob IDs below were verified with `git rev-parse
<revision>:<source_path>`. For 25 rows, destination content is byte-for-byte
the pinned blob after the import substitutions documented below. The two
intentional merges are `device_utils.py` and `sampling/__init__.py`.

| Source | Blob SHA | Inventory destination | Extraction treatment |
|---|---|---|---|
| `models/common/device_utils.py` | `0066f8fe3123d24a1d14a25bd9345a6690b6470e` | `src/tt_transformers/device_utils.py` | Copied after namespace rewrite; merged with lifecycle-cleanup split. |
| `models/common/lightweightmodule.py` | `6743db9026f0547737bda649db0b3ab8c78289c0` | `src/tt_transformers/modules/lightweightmodule.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/attention/attention_1d.py` | `0170f2cd382eb48267c26aa66f023b80648699a2` | `src/tt_transformers/modules/attention/attention_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/embedding/embedding_1d.py` | `37940e756c1edc7e896a66289641bcdab44632db` | `src/tt_transformers/modules/embedding/embedding_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/lazy_buffer.py` | `b01f4aec40973087ef4148e83edef5e68bd173d1` | `src/tt_transformers/modules/lazy_buffer.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/lazy_weight.py` | `5bf526f7debedac8e8c5541441d73cdc6a20ffa1` | `src/tt_transformers/modules/lazy_weight.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/lm_head/lm_head_1d.py` | `235a584bd257eb2a67a9173b0d4f19555bf429c1` | `src/tt_transformers/modules/lm_head/lm_head_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/mlp/mlp_1d.py` | `87570219df81d10f956e957dea1323fae867ba24` | `src/tt_transformers/modules/mlp/mlp_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/mlp/mlp_2d.py` | `edc156730ff7bf918b2c0d56f4590f66218adba1` | `src/tt_transformers/modules/mlp/mlp_2d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/rmsnorm/rmsnorm_1d.py` | `32e482f531dc50ad4e3210c70ac27ab0be7027a2` | `src/tt_transformers/modules/rmsnorm/rmsnorm_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/rmsnorm/rmsnorm_2d.py` | `2079d7266b8027d1858b34195787302f11d65b4d` | `src/tt_transformers/modules/rmsnorm/rmsnorm_2d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/rope/rope_1d.py` | `9384325d1876968081375229e6f13e66af385eeb` | `src/tt_transformers/modules/rope/rope_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/sampling/params.py` | `f120ff1973fba8a8a643a56a2e4a2da3dd6ce35b` | `src/tt_transformers/modules/sampling/params.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/sampling/penalties_1d.py` | `8d2e25560373d9f32305d2e0d6da68744b7bb285` | `src/tt_transformers/modules/sampling/penalties_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/sampling/sampling_1d.py` | `6c4178d7a270d7fac6eeb26a81938afecc7ee903` | `src/tt_transformers/modules/sampling/sampling_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/sampling/sampling_state_1d.py` | `cc9fc31d6045c2dc3973e8cdc6d318e36c320a48` | `src/tt_transformers/modules/sampling/sampling_state_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/sampling/seed_manager_1d.py` | `ce5002a859711e127bdd2d9ac9a8f4bc925ffa32` | `src/tt_transformers/modules/sampling/seed_manager_1d.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/modules/tt_ccl.py` | `6883e714a97d1badd3bb02f95ea33ef0f3cd002b` | `src/tt_transformers/modules/tt_ccl.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/sampling/__init__.py` | `39a917fa491b8f6a662a6dc6485b3f6bb9d5051a` | `src/tt_transformers/sampling/__init__.py` | Merged scaffold docstring with all pinned lazy exports and `__dir__` behavior. |
| `models/common/sampling/_utils.py` | `2acc7a26f615118c8aa2f9375f02ecd4643d4461` | `src/tt_transformers/sampling/_utils.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/sampling/generator.py` | `8ced3dee267b629cb777e72ff95869fd86e2efc4` | `src/tt_transformers/sampling/generator.py;src/tt_transformers/sampling/params.py` | Copied to `sampling/generator.py`; secondary `sampling/params.py` canonicalization deferred. |
| `models/common/sampling/sampling_params.py` | `772d8aae382dd3e0b673f85e4d1e5ef0d90fb55a` | `src/tt_transformers/sampling/sampling_params.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/sampling/tt_log_probs.py` | `9e71381cbd8318460a1e0a3494540e41d37a5609` | `src/tt_transformers/sampling/tt_log_probs.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/sampling/tt_penalties.py` | `b5398e2816286fc89f83550a91ba72e20e4f0f5a` | `src/tt_transformers/sampling/tt_penalties.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/sampling/tt_sampling.py` | `b75700fef5619d893bf7ea61a15c5d651f3b188b` | `src/tt_transformers/sampling/tt_sampling.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/sampling/vocab_padding.py` | `cffe2c2d3f598cf8ce94748587cba402b2bc36ac` | `src/tt_transformers/sampling/vocab_padding.py` | Mechanical copy with only canonical internal-import rewrites. |
| `models/common/tensor_utils.py` | `382e92e35d38fb80d60b80e2cd9206538b8fe417` | `src/tt_transformers/tensor_utils.py` | Mechanical copy with only canonical internal-import rewrites. |

## Narrow split and deferred rows

| Source | Blob SHA | Inventory destination | Status | Exact treatment |
|---|---|---|---|---|
| `models/common/tests/demos/cleanup_utils.py` | `80df955f818ab054d69d9e003d52665f0dcb369d` | `src/tt_transformers/device_utils.py;tests/integration/cleanup_utils.py` | split extracted | Extracted all lifecycle-cleanup functions into `device_utils.py`; rewrote `LazyWeight` import. |
| `models/common/tests/demos/run_helpers.py` | `6f346e4df73e79fd4fe30b91e4bd1b6c1a87c201` | `src/tt_transformers/mesh_utils.py;tests/integration/run_helpers.py` | split extracted | Extracted only `make_contiguous_page_table` into `mesh_utils.py`. |
| `models/common/utility_functions.py` | `ebfea6474e67f4fd79363ddf81b85bedfa66961b` | `src/tt_transformers/device_utils.py;tests/support/comparison.py` | split extracted | Required `is_blackhole` semantics coalesced with the identical narrow helper already in pinned `device_utils.py`; no broad copy. |
| `models/common/utils.py` | `a999b20aab2de3e0f38635af0d5f162a665cf563` | `src/tt_transformers/sampling/logprobs.py` | split extracted | Extracted only the `LogProbsCalculator` compatibility export into `sampling/logprobs.py`; legacy top-k/top-p helper deferred. |
| `models/tt_transformers/tt/common.py` | `4a52c41b1c9ef0dcfcddc3e5f499c03df056d762` | `src/tt_transformers/models/mode.py;src/tt_transformers/modules/rope/rope_scaling.py` | deferred | Deferred. Mode and RoPE-scaling dependencies remain visible as Phase 3 boundary failures; no TTTv1 common-module copy. |
| `models/tt_transformers/tt/generator.py` | `76b014c9623b391d3622062fffb41d1942187b30` | `src/tt_transformers/mesh_utils.py` | split extracted | Extracted `_mesh_shape_tuple`, `_galaxy_data_parallel_submesh_shape`, and `create_submeshes` into `mesh_utils.py`. |
| `models/tt_transformers/tt/model_config.py` | `d0bc8fe7a4c5753b2671c8abc7d27a4d720fd6c2` | `src/tt_transformers/modules/attention/attention_1d.py;src/tt_transformers/modules/mlp/mlp_1d.py;src/tt_transformers/modules/mlp/mlp_2d.py` | deferred | Deferred. `from_model_args` bridges still import `OpGroup`/`TensorGroup`; no TTTv1 model-config copy. |
| `models/tt_transformers/tt/rope.py` | `ea12c22c3934ea4af583be655eedd8e0bb856abd` | `src/tt_transformers/modules/rope/rope_1d.py` | deferred | Deferred. The RoPE `from_model_args` bridge remains; no TTTv1 RoPE-module copy. |

The secondary `src/tt_transformers/sampling/params.py` destination assigned to
`models/common/sampling/generator.py` is also deferred. Creating it now would
choose canonical `SamplingParams` ownership and remove the duplicate legacy
surface, which the task explicitly reserves for Phase 3.

## Provenance correction and owner audit

The extraction exposed one missing transitive provenance owner:
`models/common/modules/sampling/sampling_1d.py` lazily imported
`models.common.utils.LogProbsCalculator`. The source module is a compatibility
re-export of `models.common.sampling.tt_log_probs.LogProbsCalculator`.

A new pinned row was added:

| Source | Blob | Disposition | Destination |
|---|---|---|---|
| `models/common/utils.py` | `a999b20aab2de3e0f38635af0d5f162a665cf563` | split | `src/tt_transformers/sampling/logprobs.py` |

After that correction, every module named by
`qualification/analysis/external_models_dependencies.csv` resolves to either
a `source_inventory.csv` file row or an inventoried package `__init__.py`.
The inventory now has 343 unique rows: 317 assigned destinations and 26
explicit MoE-only exclusions. Its disposition counts are 270 renamed, 42
split, 5 replaced, and 26 excluded; 74 rows are marked for boundary cleanup.

## Mechanical import rewrites

These substitutions were applied to extracted source text:

| Old prefix/import | New owner |
|---|---|
| `models.common.lightweightmodule` | `tt_transformers.modules.lightweightmodule` |
| `models.common.modules` | `tt_transformers.modules` |
| `models.common.sampling` | `tt_transformers.sampling` |
| `models.common.tensor_utils` | `tt_transformers.tensor_utils` |
| `models.common.device_utils` | `tt_transformers.device_utils` |
| `models.common.utility_functions.is_blackhole` | `tt_transformers.device_utils.is_blackhole` |
| `models.common.utils.LogProbsCalculator` | `tt_transformers.sampling.logprobs.LogProbsCalculator` |

Relative imports inside the legacy sampling package were preserved. Imports
from `models.tt_transformers.tt.*` were intentionally not rewritten because
that would remove or redesign the TTTv1 bridges before Phase 3.

## Exact verification commands and results

### Pinned blob and content-fidelity check

```bash
python3 - <<'PY'
import csv
import subprocess
from pathlib import Path

REV = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
SOURCE = "/localdev/gwang/tt-metal"

def rewrite(text):
    return (
        text
        .replace("models.common.lightweightmodule", "tt_transformers.modules.lightweightmodule")
        .replace("models.common.modules", "tt_transformers.modules")
        .replace("models.common.sampling", "tt_transformers.sampling")
        .replace("models.common.tensor_utils", "tt_transformers.tensor_utils")
        .replace("models.common.device_utils", "tt_transformers.device_utils")
        .replace("models.common.utility_functions", "tt_transformers.device_utils")
        .replace("models.common.utils", "tt_transformers.sampling.logprobs")
    )

with open("qualification/provenance/source_inventory.csv", newline="") as stream:
    rows = [
        row for row in csv.DictReader(stream)
        if row["category"] in {
            "production_modules", "production_sampling", "production_support"
        } and row["disposition"] != "excluded"
    ]

assert len(rows) == 27
merged = {
    "models/common/device_utils.py",
    "models/common/sampling/__init__.py",
}
verified = 0
for row in rows:
    source = row["source_path"]
    blob = subprocess.check_output(
        ["git", "-C", SOURCE, "rev-parse", f"{REV}:{source}"], text=True
    ).strip()
    assert blob == row["git_blob_sha"]
    if source in merged:
        continue
    destination = row["destination_path"].split(";")[0]
    expected = rewrite(subprocess.check_output(
        ["git", "-C", SOURCE, "show", f"{REV}:{source}"], text=True
    ))
    assert Path(destination).read_text() == expected
    verified += 1

print(f"blob-verified {len(rows)} direct rows; byte-for-byte after import rewrite {verified}; two intentional merges")
PY
```

Result:

```text
blob-verified 27 direct rows; byte-for-byte after import rewrite 25; two intentional merges
```

### Syntax compilation

```bash
env PYTHONPYCACHEPREFIX=/tmp/gwang/tttv2_foundation_pycache \
  python3 -m compileall -q \
  src/tt_transformers/modules \
  src/tt_transformers/sampling \
  src/tt_transformers/device_utils.py \
  src/tt_transformers/tensor_utils.py \
  src/tt_transformers/mesh_utils.py
```

Result: exit code 0 with no output.

### Static import boundary check

```bash
python3 tools/check_import_boundaries.py src/tt_transformers
```

Result: exit code 1 with exactly the six expected Phase 3 failures:

```text
src/tt_transformers/modules/attention/attention_1d.py:1392: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.model_config
src/tt_transformers/modules/mlp/mlp_1d.py:36: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.common
src/tt_transformers/modules/mlp/mlp_1d.py:514: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.model_config
src/tt_transformers/modules/mlp/mlp_2d.py:507: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.model_config
src/tt_transformers/modules/rope/rope_1d.py:291: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.common
src/tt_transformers/modules/rope/rope_1d.py:300: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.rope
```

No foundation file imports `tests`, `examples`, or `pytest`. The reusable
foundation otherwise imports the declared base dependencies `ttnn`, `torch`,
and `loguru`. `tensor_utils.py` has a guarded, function-local `yaml`
import used only when YAML serialization is requested; PyYAML is not currently
declared and remains an optional-path packaging decision.

## Remaining Phase 3 work

1. Replace the top-level TTTv1 `Mode` dependency with a TTTv2-owned neutral
   mode contract.
2. Remove the `Attention1D`, `MLP1D`, and `MLP2D` `from_model_args`
   bridges and their `OpGroup`/`TensorGroup` imports.
3. Remove `RotarySetup1D.from_model_args` and its TTTv1 common/RoPE imports.
4. Choose one canonical `SamplingParams`, then replace the legacy lazy export
   and deferred `sampling/params.py` split.
5. Decide whether guarded YAML serialization is supported and, if so, declare
   PyYAML in the appropriate optional dependency group.
