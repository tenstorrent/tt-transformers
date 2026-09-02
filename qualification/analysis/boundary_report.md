# TTTv2 Phase 0 production boundary report

Source repository: `/localdev/gwang/tt-metal`

Source revision: `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`

Scope: the Python production implementation named in `TTTV2_MIGRATION_PLAN.md`: non-MoE `models/common/modules`, `models/common/llm_runtime`, the three shared executor files, and all twelve model packages under `models/common/models`.

Status: reproducible characterization complete; no implementation was extracted or edited and no hardware was used.

## Executive verdict

The pinned production slice is internally well layered but not repository-independent. It contains exactly 113 Python files and 52,645 lines: 16 module files, 21 runtime files, and 76 shared/concrete model files. MoE is excluded and no other Python production directory is omitted by the analyzer's scope rule.

The internal import direction already matches the proposed architecture. Across 1,031 imported-symbol rows that resolve within the slice, modules depend only on modules, runtime depends only on runtime/modules, and models depend on models/runtime/modules. There are no reverse layer violations. A mechanical namespace rewrite is therefore viable without redesigning the dependency direction.

The boundary is not yet closable as a wheel:

- 119 imported-symbol rows at 87 statement sites escape to 14 external `models.*` modules. All 119 rows have a proposed migration owner and treatment; none are unassigned.
- There are 60 uses of unstable TTNN APIs: 59 experimental call sites across 13 APIs and one private `ttnn._ttnn.tensor.dump_tensor_flatbuffer` call.
- Two Qwen demo modules in the production tree import `pytest` and two `models.common.tests.*` helpers. These produce six explicit production/test-boundary import rows; both modules also import the test-oriented `comp_pcc` helper through `models.common.utility_functions`.
- Ten `from_model_args` factories, three explicitly legacy adapters/dispatchers, and 21 imported-symbol rows from `models.tt_transformers.*` preserve TTTv1 surfaces or namespace dependencies.
- Runtime construction mutates the TTNN global default device 38 times and falls back to it 10 times.
- All twelve HF adaptors can resolve a default relative `model_cache/...` path against the current working directory. Cache fingerprints omit several release-critical inputs.
- Production has 108 environment-variable access occurrences resolving to 21 known keys, with inconsistent boolean/offline semantics.

These findings make Phase 2 mechanical extraction feasible, but Phase 3 boundary closure and Phase 4 TTNN qualification must remain explicit gates.

## Reproduction

The analyzer reads only Git objects at the pinned revision; it does not import or modify `tt-metal`.

```bash
cd /localdev/gwang/tt_transformers
git -C /localdev/gwang/tt-metal rev-parse --verify 00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0^{commit}
python qualification/analysis/analyze_boundary.py \
  --source-repo /localdev/gwang/tt-metal \
  --revision 00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0 \
  --output-dir qualification/analysis
```

To reproduce into an isolated directory and compare deterministic outputs:

```bash
cd /localdev/gwang/tt_transformers
boundary_output=$(mktemp -d /tmp/tttv2-boundary.XXXXXX)
python qualification/analysis/analyze_boundary.py \
  --source-repo /localdev/gwang/tt-metal \
  --revision 00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0 \
  --output-dir "$boundary_output"
for artifact in analysis_summary.json production_files.csv public_symbols.csv imports.csv import_edges.csv dependency_roots.csv external_models_dependencies.csv ttnn_unstable_uses.csv environment_variables.csv runtime_assumptions.csv tttv1_bridges.csv production_test_imports.csv; do
  cmp "qualification/analysis/$artifact" "$boundary_output/$artifact"
done
```

The scope invariant is executable: the analyzer fails unless the pinned tree resolves to 16 module, 21 runtime, and 76 model Python files. It also verifies that the supplied revision resolves to the exact 40-character commit.

A defensive scan found no production use of `importlib`, `__import__`, `sys.path`, `PYTHONPATH`, or `TT_METAL_HOME`; static AST import classification therefore covers the executable import mechanisms present in this slice. Reproduce that check against the Git object with:

```bash
git -C /localdev/gwang/tt-metal grep -n -E 'importlib|__import__|sys\.path|PYTHONPATH|TT_METAL_HOME' \
  00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0 -- \
  models/common/modules models/common/llm_runtime models/common/models
```

The command also sees prose in non-Python documentation if it is extended to path strings such as `/localdev`; the executable inventory intentionally parses the 113 Python files only.

## Artifact index

| Artifact | Purpose |
|---|---|
| `analysis_summary.json` | Machine-readable headline counts and exact revision/scope |
| `production_files.csv` | Every in-scope Python file, source/destination module candidate, Git blob SHA, layer, and line counts |
| `public_symbols.csv` | Public-name candidates and signatures, including public methods/protocol methods and export status |
| `imports.csv` | Every imported symbol with source, line, lexical scope, import timing, classification, and internal resolution |
| `import_edges.csv` | Aggregated source-module to target-module dependency graph |
| `dependency_roots.csv` | Every stdlib/third-party/root namespace with site/file counts and dependency policy |
| `external_models_dependencies.csv` | Every external `models.*` imported symbol occurrence with owner, treatment, and priority |
| `ttnn_unstable_uses.csv` | Every maximal `ttnn.experimental.*` and `ttnn._ttnn.*` attribute use |
| `environment_variables.csv` | Every `os.getenv`, `os.environ.get`, and `os.environ[...]` access, including resolved generator-expression keys |
| `runtime_assumptions.csv` | Per-site global-device, relative-path, filesystem, checkpoint, and cache-I/O assumptions |
| `tttv1_bridges.csv` | TTTv1 namespace imports, all `from_model_args` factories, and explicit legacy adapters |
| `production_test_imports.csv` | Production imports from `pytest` or `models.common.tests.*` |

CSV counts refer to imported symbols unless a table explicitly says statement sites. A single `from x import A, B` is two imported-symbol rows but one statement site.

## Public symbol and signature boundary

`public_symbols.csv` contains 1,760 rows:

| Layer | Top-level class/function/value candidates | Explicit re-exports not otherwise defined in-file | Public class members | Total |
|---|---:|---:|---:|---:|
| modules | 61 | 0 | 137 | 198 |
| llm_runtime | 96 | 0 | 187 | 283 |
| models | 453 | 156 | 670 | 1,279 |
| total | 610 | 156 | 994 | 1,760 |

This is a characterization inventory, not a recommendation to export all 1,760 names. Eighteen source files define a literal `__all__`; the 156 `explicit_export` rows retain package/model re-exports (and explicitly exported underscore names) that are not defined as top-level symbols in the same file. Everywhere without `__all__`, a non-underscore top-level name is marked `candidate_no___all__`. Public protocol and construction signatures such as `__init__`, `__call__`, iteration, and context-manager methods are retained because consumers can depend on them even though their names begin with underscores.

Before import rewrites, review the 610 top-level candidates and 156 explicit re-exports into supported public API, model-local public surface, compatibility-only, or private-by-policy. Phase 2 must preserve every signature unless the associated row is explicitly approved for Phase 3 bridge removal. Package `__init__` exports should change only after that review so extraction does not accidentally expand or contract the API.

## Import and dependency graph

The inventory contains 1,891 imported-symbol rows and 1,113 aggregated source/target/classification edges. The internal layer totals are:

| Source layer | Target layer | Imported-symbol rows |
|---|---|---:|
| modules | modules | 57 |
| llm_runtime | modules | 11 |
| llm_runtime | llm_runtime | 120 |
| models | modules | 284 |
| models | llm_runtime | 175 |
| models | models | 384 |

No module→runtime/model or runtime→model edge exists in the in-scope graph.

Direct third-party imports are exactly:

| Root | Statement sites | Files | Policy |
|---|---:|---:|---|
| `ttnn` | 74 | 74 | Base dependency; pin and qualify 0.77.0 |
| `torch` | 82 | 76 | Hard base dependency; choose a qualified bound |
| `loguru` | 17 | 17 | Base dependency; choose a qualified minimum |
| `transformers` | 15 | 15 | Optional examples/model-provider extra; load lazily |
| `tqdm` | 9 | 9 | Optional examples extra |
| `pytest` | 2 | 2 | Test-only; invalid in production |

There are 28 standard-library roots. `dependency_roots.csv` records all of them with counts. No unclassified third-party root exists at the pinned revision.

## External `models.*` closure ownership

The following table groups all 119 per-symbol rows in `external_models_dependencies.csv`. Site counts are import statements, not symbols.

| Existing module / symbols | Sites | Proposed owner | Treatment |
|---|---:|---|---|
| `models.common.lightweightmodule.LightweightModule` | 22 | package module core | Internalize the class with its construction/call contract |
| `models.common.tensor_utils` (12 imported helpers/constants) | 22 | `tt_transformers.tensor_utils` | Extract only the required helpers and transitive minimum |
| `models.common.device_utils.get_device_name` | 3 | `tt_transformers.device_utils` | Internalize narrow architecture/SKU logic |
| `models.common.sampling.SamplingParams` | 5 | `tt_transformers.sampling` | Converge on one canonical value without changing fields/defaults |
| `models.common.sampling.sampling_params.SamplingParams` | 8 | `tt_transformers.sampling` | Same canonical owner; eliminate duplicate import paths |
| `models.common.sampling.vocab_padding` (3 helpers) | 1 | `tt_transformers.sampling.vocab_padding` | Internalize the three required helpers |
| `models.common.utils.LogProbsCalculator` | 1 | `tt_transformers.sampling.logprobs` | Internalize the narrow log-probability helper |
| `models.common.utility_functions.is_blackhole` | 1 | `tt_transformers.device_utils` | Replace broad utility import with narrow predicate |
| `models.common.utility_functions.comp_pcc` | 2 | `tests/qualification` | Move correctness comparison out of production/example package code |
| `models.common.tests.demos.cleanup_utils.cleanup_model_case` | 2 | examples plus tests | Remove production→tests edge; retain orchestration beside examples |
| `models.common.tests.demos.run_helpers.make_contiguous_page_table` | 2 | examples or neutral runtime helper | Remove production→tests edge; place according to actual non-test reuse |
| `models.tt_transformers.tt.common.Mode` | 1 | neutral package mode contract | Use a TTTv2 enum/Literal or the existing string-only contract |
| `models.tt_transformers.tt.common.rope_scaling_model_factory` | 1 | TTTv1 bridge removal | Remove with `RotarySetup1D.from_model_args` |
| `models.tt_transformers.tt.generator.create_submeshes` | 12 | `tt_transformers.mesh_utils` | Internalize narrow submesh creation and characterize cleanup ownership |
| `models.tt_transformers.tt.model_config.OpGroup/TensorGroup` | 3 | TTTv1 bridge removal | Remove the three consuming `from_model_args` factories; do not port legacy config |
| `models.tt_transformers.tt.rope.compute_gather_cos_sin` | 1 | TTTv1 bridge removal | Remove the bridge and use TTTv2-owned RoPE table construction |

The full CSV is per occurrence and is the extraction checklist. Its proposed-owner column contains no `unassigned` value.

## Production/test boundary violations

Both `models/common/models/qwen25_coder_32b/demo.py` and `models/common/models/qwen3_32b/demo.py` import:

- `pytest` at line 30;
- `cleanup_model_case` from `models.common.tests.demos.cleanup_utils` at line 43;
- `make_contiguous_page_table` from `models.common.tests.demos.run_helpers` at line 44;
- `comp_pcc` from the broad `models.common.utility_functions` at line 45.

Move these runnable/qualification files out of the installed model implementation and split orchestration from assertions. The base wheel must import without `pytest`, and installed production code must have no `tests` namespace edge.

## Unstable TTNN API surface

| API | Occurrences |
|---|---:|
| `ttnn.experimental.all_gather_async` | 31 |
| `ttnn.experimental.reduce_scatter_minimal_async` | 6 |
| `ttnn.experimental.paged_fill_cache` | 6 |
| `ttnn.experimental.rotary_embedding_llama` | 4 |
| `ttnn.experimental.minimal_matmul` | 3 |
| `ttnn.experimental.paged_update_cache` | 2 |
| `ttnn.experimental.all_gather_matmul_async` | 1 |
| `ttnn.experimental.nlp_concat_heads` | 1 |
| `ttnn.experimental.nlp_concat_heads_decode` | 1 |
| `ttnn.experimental.nlp_create_qkv_heads` | 1 |
| `ttnn.experimental.nlp_create_qkv_heads_decode` | 1 |
| `ttnn.experimental.paged_fused_update_cache` | 1 |
| `ttnn.experimental.rotary_embedding_llama_fused_qk` | 1 |
| `ttnn._ttnn.tensor.dump_tensor_flatbuffer` | 1 |

The private use is `models/common/modules/lazy_weight.py:340`. Do not vendor the implementation. The 0.77 qualification probe should first test symbol/signature availability for all 14 APIs, then exercise each semantic surface by category: collectives, QKV/head transforms, RoPE, paged cache, minimal matmul, and tensor serialization. Every failure should be classified using the plan's TTNN-gap taxonomy.

## Environment, filesystem, cache, and process-state assumptions

`environment_variables.csv` resolves these 21 keys:

- provider/offline selection: `CI`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_MODEL`;
- cache/chunk policy: `TT_CACHE_PATH`, `TT_CACHE_FALLBACK_PATH`, `MAX_PREFILL_CHUNK_SIZE`;
- runtime tuning: `DISABLE_BATCHED_PREFILL`, `DISABLE_BATCHED_EXTRACT`, `DISABLE_MINIMAL_MATMUL`, `DISABLE_PREFILL_AG_BF8`, `DISABLE_PREFILL_REDUCE_BF8`;
- topology: `MESH_DEVICE`;
- embedded Qwen demo/number-divergence controls: eight `QWEN25_CODER_32B_*` or `QWEN3_32B_*` keys.

Boolean parsing is inconsistent. Most adaptors treat only `CI == "true"` as offline, four Qwen construction paths accept lower-cased `1/true/yes` across `CI`, `HF_HUB_OFFLINE`, and `TRANSFORMERS_OFFLINE`, while many disable flags use `bool(os.getenv(...))`, for which even `"0"` is true. Normalize this policy and document every retained key.

Filesystem/cache findings:

- Every HF adaptor has a default relative `Path("model_cache")/...` path, so invocation CWD controls cache placement. Eleven create the selected cache directory eagerly; Llama 3.1-8B creates parents later on cache write and only uses `/tmp/tttv2_model_cache` when an explicitly configured `TT_CACHE_PATH` fails with a permission/read-only error.
- There are 47 `AutoConfig`/`AutoModelForCausalLM`/`AutoTokenizer.from_pretrained` calls in 15 production files. They may access the network or HF caches and reinforce that Transformers must be optional/lazy and checkpoint revision/offline policy explicit.
- `LazyWeight` reads `.tensorbin` files via `ttnn.load_tensor` and writes them through the private flatbuffer API. Its fingerprint includes source shape, dtype, layout, selected memory/mapper representation, device ID, and non-default pad value, but not source contents, HF model ID/revision, `tt-transformers` version, TTNN version, conversion schema version, architecture, firmware, or full topology.
- `PagedKVCacheManager` passes a `cache_file_name` derived from K/V role and tensor shape to TTNN; the filename itself does not identify the package/TTNN/checkpoint/conversion release tuple.
- The two embedded Qwen qualification modules write CSV output to an environment-selected path, defaulting under `/tmp`; keep this side effect in examples/qualification, not installed model implementation.

Global-device findings:

- 38 calls mutate `ttnn.SetDefaultDevice`; most HF/provider construction paths do not restore the previous value. The two embedded demo modules clear it to `None` rather than restore an observed prior value.
- 10 reusable module construction/config paths call `ttnn.GetDefaultDevice` when explicit config/device fields are absent.

Require and propagate `mesh_device` in the standalone core. If compatibility temporarily needs the global default, scope it, restore the exact previous value, and add repeated/concurrent construction characterization tests.

## TTTv1 bridges

All ten `from_model_args` factories are inventoried with exact signatures:

- `Attention1D.from_model_args`;
- `Embedding1D.from_model_args`;
- `LMHead1D.from_model_args`;
- `MLP1D.from_model_args`;
- `MLP2D.from_model_args`;
- `RMSNorm1D.from_model_args`;
- `RMSNorm2D.from_model_args`;
- `RotarySetup1D.from_model_args`;
- `Penalties1D.from_model_args`;
- `Sampling1D.from_model_args`.

Three additional methods are explicitly legacy adapters: `RotarySetup1D.get_rot_idxs`, `RotarySetup1D.get_rot_mats`, and `Llama3Transformer1D.forward`. The first two wrap the TTTv2 preparation/decode path; the last is documented as a backward-compatible dispatcher over prefill/decode.

Not every `models.tt_transformers.*` dependency is itself a removable compatibility entry point. The twelve `create_submeshes` imports support the current generator facade and need a new `mesh_utils` owner. `Mode` is used both by an active MLP dispatcher and a legacy factory, so replace its type/value contract before removing the old import. The model-config and RoPE imports are confined to removable `from_model_args` bridges.

## Prioritized closure list

1. **P0 — approve the public API characterization.** Review all 610 top-level candidates, 156 explicit re-exports, and 994 public members; assign supported/model-local/compatibility/private status. Freeze signatures before namespace rewrites.
2. **P0 — use the pinned file/edge inventories for mechanical extraction.** Copy only rows in `production_files.csv`, preserve blob provenance, rewrite the 1,031 internal imported-symbol rows to `tt_transformers.*`, and assert the same layer direction after rewriting.
3. **P0 — close all 119 external `models.*` rows.** Internalize only the narrow owned helpers in the ownership table. Acceptance: `external_models_dependencies.csv` regenerated against extracted production is empty and no old `models.*` import remains.
4. **P0 — separate production, examples, and tests.** Relocate the two embedded Qwen demo/qualification files and their helpers. Acceptance: installed production has zero `pytest`, `tests.*`, `models.common.tests.*`, or test-assertion utility imports.
5. **P0 — remove TTTv1-only bridges deliberately.** Delete the ten characterized `from_model_args` factories and three approved legacy adapters only after caller checks; replace `Mode`, internalize `create_submeshes`, and do not port legacy `OpGroup`/`TensorGroup` or RoPE factories. Acceptance: zero `models.tt_transformers.*` imports with characterization tests covering retained TTTv2 constructors.
6. **P1 — run TTNN 0.77 compatibility probes before claiming support.** Probe all 14 unstable/private APIs, then host/hardware semantics in the plan's order. Acceptance: each occurrence maps to a passing supported API, an explicit TTTv2 adaptation, or a documented minimal TTNN gap.
7. **P1 — make dependency optionality enforceable.** Keep `ttnn`, `torch`, and `loguru` as direct base dependencies; isolate `transformers`/`tqdm`; keep `pytest` test-only. Acceptance: base wheel imports without Transformers, tqdm, pytest, or a `tt-metal` checkout.
8. **P1 — replace process-global device dependence.** Thread explicit device ownership through the ten fallback sites and provider constructors. Acceptance: repeated and interleaved construction does not leak/change another owner's default device.
9. **P1 — define one explicit, versioned cache policy.** Eliminate CWD-relative defaults, report the resolved root, normalize permissions/fallback behavior, and include package/TTNN/checkpoint/conversion/architecture/topology inputs. Acceptance: cold/warm/invalidation tests pass in isolated writable, read-only, and offline environments.
10. **P1 — normalize environment policy.** Specify types/defaults for the 21 keys, remove demo-only keys from installed production, and stop deriving online/offline mode solely from `CI`. Acceptance: a machine-readable preflight report shows effective values without exposing secrets.

## Phase boundary

This report is sufficient to drive extraction and boundary closure, but it is not hardware evidence and makes no TTNN 0.77 compatibility or model-support claim. Hardware is intentionally untouched during this static Phase 0 subtask; subsequent hardware work should follow the repository's remote-machine instructions and serialize access/reset as directed by the migration task.
