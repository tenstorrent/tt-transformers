# Phase 2 runtime and shared-executor extraction

Status: mechanically extracted and verified from the pinned source revision.

Source repository: `/localdev/gwang/tt-metal`

Source revision: `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`

Extraction scope: 21 Python files under `models/common/llm_runtime` and exactly three shared executors: `executor.py`, `llama3_executor.py`, and `qwen2_executor.py`.

Out of scope for this checkpoint: concrete model packages, reusable/foundation modules, sampling/support helpers, tests, examples, runtime Markdown documents, and hardware execution.

## Result

The 24 pinned Python files are present at their authoritative `src/tt_transformers` destinations. The checked-in extractor verifies the exact source commit and every Git blob SHA, reads source through Git objects, and performs 95 import-statement namespace rewrites. Phase 3 subsequently added three exact active-caller migrations in `decode.py` after the foundation removed legacy RoPE adapters; those substitutions are asserted by `apply_boundary_closures()` and detailed in `runtime_boundary.md`.

The Phase 2 snapshot changed no function/class signature, symbol, statement order, expression, default, annotation, comment, or docstring. The later Phase 3 caller migrations are deliberately separated in the extractor. The existing package initializers were retained rather than overwritten:

- `src/tt_transformers/llm_runtime/__init__.py`: SHA-256 `3f173507955bd26a1b44f0f7009166853a3ce9872bbee34ca7618cc42845b182`;
- `src/tt_transformers/models/__init__.py`: SHA-256 `c261020690a4de3a0ff5164d856601b8744ea01c8a6db96162bcd606e1e888ff`.

The pinned source has no `models/common/llm_runtime/__init__.py` or `prefill/__init__.py`, so no source initializer was available to merge. The scaffold's model-lazy-import policy remains intact. Setuptools namespace discovery finds both `tt_transformers.llm_runtime` and `tt_transformers.llm_runtime.prefill` without synthesizing another initializer.

## Source, destination, and blob checks

| Source path | Destination path | Pinned Git blob SHA |
|---|---|---|
| `models/common/llm_runtime/config.py` | `src/tt_transformers/llm_runtime/config.py` | `34e767bf49743e4d1df23139121d31ad5e4aa0ae` |
| `models/common/llm_runtime/decode.py` | `src/tt_transformers/llm_runtime/decode.py` | `8d1685b68ddb1c41bbf7c31df0259c22cd46e825` |
| `models/common/llm_runtime/execution.py` | `src/tt_transformers/llm_runtime/execution.py` | `5be975228764c390d23d4b856a29cbf6bf3d5738` |
| `models/common/llm_runtime/lane_group.py` | `src/tt_transformers/llm_runtime/lane_group.py` | `4cffe71d197231578ff5d2214bc3946ca6cd0c23` |
| `models/common/llm_runtime/output_reader.py` | `src/tt_transformers/llm_runtime/output_reader.py` | `2635a5b5117ecff139570624830ecf93c3e30df3` |
| `models/common/llm_runtime/paged_kv_cache.py` | `src/tt_transformers/llm_runtime/paged_kv_cache.py` | `ad4e774390d0ee09ee76f56133f070aca176870b` |
| `models/common/llm_runtime/prefill/config.py` | `src/tt_transformers/llm_runtime/prefill/config.py` | `ffe15b61d0446a0712cf6a744a4fa7a43f784027` |
| `models/common/llm_runtime/prefill/inputs.py` | `src/tt_transformers/llm_runtime/prefill/inputs.py` | `edf86ca70500bd543f5d6878ac72fee51f61170d` |
| `models/common/llm_runtime/prefill/plan.py` | `src/tt_transformers/llm_runtime/prefill/plan.py` | `fa0c05779aae434f288ace93415cb2f3b71474cf` |
| `models/common/llm_runtime/prefill/postprocess.py` | `src/tt_transformers/llm_runtime/prefill/postprocess.py` | `c7ab39d4803fc0be6b13356094a32970aa80afb1` |
| `models/common/llm_runtime/prefill/result_collector.py` | `src/tt_transformers/llm_runtime/prefill/result_collector.py` | `03df6a1828f4678ed43f2a0820ed38eb537c35fb` |
| `models/common/llm_runtime/prefill/runtime.py` | `src/tt_transformers/llm_runtime/prefill/runtime.py` | `1bd3b99ff81774af505e66cc8da8bca84d28d939` |
| `models/common/llm_runtime/prefill/sampling_helpers.py` | `src/tt_transformers/llm_runtime/prefill/sampling_helpers.py` | `7ecdb1c92baf703b59259df69d7bffd1cb6928f3` |
| `models/common/llm_runtime/prefill/sequence_runner.py` | `src/tt_transformers/llm_runtime/prefill/sequence_runner.py` | `1f7523f79faaa9f18208cf3c035341fe444887ee` |
| `models/common/llm_runtime/prefill/signatures.py` | `src/tt_transformers/llm_runtime/prefill/signatures.py` | `6f5449e9a2dfeb9f45831e40dc12c1ec1d949a0b` |
| `models/common/llm_runtime/prefill/trace.py` | `src/tt_transformers/llm_runtime/prefill/trace.py` | `499343b812a421953b163f4d995d6b3e96340b82` |
| `models/common/llm_runtime/program_compiler.py` | `src/tt_transformers/llm_runtime/program_compiler.py` | `3ac6ef89614f269861505fe2f5789273a108aa02` |
| `models/common/llm_runtime/tensor_resources.py` | `src/tt_transformers/llm_runtime/tensor_resources.py` | `16d96d7b6e42d66951307947450181e46a3168dd` |
| `models/common/llm_runtime/trace_compiler.py` | `src/tt_transformers/llm_runtime/trace_compiler.py` | `fb48777d15eef1595da21029a377b7b46cb361fd` |
| `models/common/llm_runtime/vllm_adapter.py` | `src/tt_transformers/llm_runtime/vllm_adapter.py` | `5ce2eafa479b73734766445971babac25d870845` |
| `models/common/llm_runtime/warmup.py` | `src/tt_transformers/llm_runtime/warmup.py` | `47b1792045c2152271fe0f74646948b04ef4c1c7` |
| `models/common/models/executor.py` | `src/tt_transformers/models/executor.py` | `bd1d4a8607aa27a75ad63109b046bcc41cd47d11` |
| `models/common/models/llama3_executor.py` | `src/tt_transformers/models/llama3_executor.py` | `9c84008b6b551a97ed2a156aa1d2a4d8994a457a` |
| `models/common/models/qwen2_executor.py` | `src/tt_transformers/models/qwen2_executor.py` | `321d10bd5c1165eecac5ec30d9185d5e1b71321a` |

The extractor manifest and `qualification/provenance/source_inventory.csv` agree on all 24 source paths, destination paths, and blob SHAs.

## Mechanical rewrite rules

The following longest-prefix-first rules apply only to modules named by `from ... import ...` or `import ...` statements:

| Pinned namespace | Standalone namespace |
|---|---|
| `models.common.sampling.sampling_params` | `tt_transformers.sampling.sampling_params` |
| `models.common.llm_runtime` | `tt_transformers.llm_runtime` |
| `models.common.models.executor` | `tt_transformers.models.executor` |
| `models.common.modules` | `tt_transformers.modules` |
| `models.common.sampling` | `tt_transformers.sampling.sampling_params` for the selected runtime/shared-executor `SamplingParams` imports |

The extractor fails if any imported `models.*` module lacks a rule. It parses source before and after the rewrite and compares the rewritten AST to an AST produced by mutating only `Import`/`ImportFrom` module fields. It also checks that the number of changed physical import statements equals the number of legacy import statements and that no legacy import remains.

Phase 3 selected `tt_transformers.sampling.sampling_params.SamplingParams` as the single runtime/model import owner. The source-level aggregate import in the two family executors is now reproducibly redirected to that module without changing its fields or defaults.

## Reproduction and drift check

Write the exact extraction:

```bash
cd /localdev/gwang/tt_transformers
python tools/extract_runtime.py \
  --source-repo /localdev/gwang/tt-metal \
  --destination-root /localdev/gwang/tt_transformers
```

Verify every destination without writing:

```bash
cd /localdev/gwang/tt_transformers
python tools/extract_runtime.py \
  --source-repo /localdev/gwang/tt-metal \
  --destination-root /localdev/gwang/tt_transformers \
  --check
```

Expected summary:

```text
mode: check
runtime_files: 21
shared_executor_files: 3
namespace_rewrite_statements: 95
```

## Syntax validation

The required compile ran successfully with bytecode directed outside the source tree:

```bash
cd /localdev/gwang/tt_transformers
runtime_cache=$(mktemp -d /tmp/gwang/tttv2-runtime-pyc.XXXXXX)
PYTHONPYCACHEPREFIX="$runtime_cache" python -m compileall -q -f \
  src/tt_transformers/llm_runtime \
  src/tt_transformers/models/executor.py \
  src/tt_transformers/models/llama3_executor.py \
  src/tt_transformers/models/qwen2_executor.py
```

Result: exit code 0. The validation cache used `/tmp/gwang/tttv2-runtime-pyc.Z91e8u`; no bytecode from this compile was directed into the extracted source directories.

## Static import validation

The 24 extracted files have:

- zero old `models.*` imports;
- zero `tests`, `examples`, or `pytest` imports;
- zero layer-policy violations from `tools/check_import_boundaries.py`;
- 25 distinct canonical `tt_transformers.*` imported modules, all present at this checkpoint;
- only declared third-party roots: `ttnn` at 11 statement sites, `torch` at 17, and `loguru` at 4.

The whole package checker was also run exactly as required:

```bash
cd /localdev/gwang/tt_transformers
python tools/check_import_boundaries.py src/tt_transformers
```

At the original Phase 2 checkpoint it exited 1 with six expected Phase 3 edges, all in separately extracted foundation modules and none in the 24 files owned here:

```text
src/tt_transformers/modules/attention/attention_1d.py:1392: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.model_config
src/tt_transformers/modules/mlp/mlp_1d.py:36: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.common
src/tt_transformers/modules/mlp/mlp_1d.py:514: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.model_config
src/tt_transformers/modules/mlp/mlp_2d.py:507: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.model_config
src/tt_transformers/modules/rope/rope_1d.py:291: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.common
src/tt_transformers/modules/rope/rope_1d.py:300: legacy tt-metal namespace is forbidden: models.tt_transformers.tt.rope
```

Those historical failures were not masked. They were closed during Phase 3; see `runtime_boundary.md` for the current passing result.

## Unresolved Phase 3 boundary work

The Phase 2 items above were transferred to and closed or reclassified by the Phase 3 boundary checkpoint in `runtime_boundary.md`.

No concrete models, foundation helpers, tests, examples, main migration work log, pinned source files, or hardware state were changed by this runtime extraction checkpoint.
