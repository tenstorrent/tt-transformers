# Phase 3 runtime/model boundary closure

Status: runtime/model production boundary closed and statically enforced.

Source baseline: `/localdev/gwang/tt-metal` revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`.

Scope: the extracted `src/tt_transformers/llm_runtime` and `src/tt_transformers/models` trees, including the three shared executors and twelve concrete model packages. Reusable foundation implementation under `src/tt_transformers/modules`, `src/tt_transformers/sampling`, and standalone support helpers was audited as a dependency but not edited here.

No test, example, support, main work-log, source-repository, or hardware file/state was changed by this checkpoint.

## Outcome

The audited runtime/model tree contains 97 Python files: 22 under `llm_runtime` (21 pinned implementation files plus the scaffold initializer) and 75 under `models` (three shared executors, 71 concrete-model files, and the scaffold initializer). Static AST audit now reports:

| Boundary | Result |
|---|---:|
| old `models.*` imports | 0 |
| production imports from `tests`, `examples`, or `pytest` | 0 |
| runtime/model `from_model_args` definitions | 0 |
| `Llama3Transformer1D.forward` definitions | 0 |
| aggregate/generator `SamplingParams` imports | 0 |
| canonical `tt_transformers.sampling.sampling_params.SamplingParams` imports | 12 |
| static import-boundary violations | 0 |

TTTv1 references that remain in comments/docstrings describe precision/tuning provenance; they are not executable imports or adapters. Two runtime identifiers containing `legacy` are intentionally retained and are not TTTv1 repository bridges:

- `VLLMAdapter.resolve_legacy_kv_cache_config` validates the current external vLLM cache-spec compatibility input;
- `_legacy_batched_prefill_size` preserves an active request-shape fallback when the newer capability value is absent.

## Closed changes

### Canonical `SamplingParams`

The foundation extracted both pinned definitions but deferred canonical ownership. Runtime and reusable module preparation already used the narrow immutable value at:

```text
tt_transformers.sampling.sampling_params.SamplingParams
```

This checkpoint adopts that same owner for all runtime/model production imports. Five aggregate imports were redirected without changing the dataclass implementation, fields, defaults, or construction call sites:

- `models/deepseek_r1_distill_qwen_14b/executor.py`;
- `models/llama3_executor.py`;
- `models/mistral_7b/executor.py`;
- `models/phi4/executor.py`;
- `models/qwen2_executor.py`.

`models/qwen3_32b/executor.py` and six runtime files already used the selected owner. The aggregate lazy export in the foundation package remains outside this lane, but runtime/model production can no longer import `SamplingParams` through it: the checker rejects both `tt_transformers.sampling.SamplingParams` and `tt_transformers.sampling.generator.SamplingParams`.

Both pinned extractors reproduce the closed result:

- `tools/extract_runtime.py` maps its two source aggregate imports directly to `tt_transformers.sampling.sampling_params`;
- `tools/extract_model_packages.py` applies the specific submodule rewrite before the aggregate rewrite, preserving existing submodule imports and canonicalizing aggregate imports.

### Removed `Llama3Transformer1D.forward`

The characterized method was a TTTv1-only mode dispatcher over the already-public `prefill_forward` and `decode_forward` methods. It was removed only after the following caller audit:

1. Direct symbol/reference search across `src`, `tests`, `examples`, and `qualification` found no call to `Llama3Transformer1D.forward`; the only non-source mentions were the Phase 0 characterization artifacts.
2. AST enumeration of every `.forward(...)` call across those trees found module/layer forward calls but no explicit transformer-dispatcher call.
3. `DecodeRuntime` invokes `model.decode_forward(...)` directly.
4. `PrefillRuntime` invokes `config.model.prefill_forward(...)` directly for chunked, hidden-only, and batched paths.
5. `Llama3ForCausalLM.from_pretrained` only constructs `Llama3Transformer1D(model_config)`.
6. Llama generators target the executor facade, whose prefill/decode methods delegate to `ModelExecutor`, not the model's generic dispatcher.
7. Focused model-contract tests call `Llama3Transformer1D.prefill_forward`, postprocessing helpers, and lifecycle helpers explicitly; none calls the removed method.

There was therefore no active caller to rewrite. The underlying prefill/decode methods and their signatures remain unchanged. `tools/extract_model_packages.py` now removes exactly one method named `forward` from exactly the pinned `Llama3Transformer1D` class, checks its characterized backward-compatibility docstring, removes the adjacent blank line, and reparses the result. It refuses drift or a differently purposed method.

### Preserved concurrent RoPE caller closure

The concurrent foundation boundary work removed `RotarySetup1D.get_rot_idxs` and `get_rot_mats`. Its two retained runtime callers were deliberately migrated in `llm_runtime/decode.py`:

- host rotation-index preparation now calls `prepare_rot_idxs(model.rope_setup.config, ..., on_host=True)`;
- decode rotation lookup now calls `model.rope_setup.decode_forward(inputs.rotary_indices)`.

`tools/extract_runtime.py::apply_boundary_closures` asserts and reproduces exactly those three edits (one import and two calls) after namespace rewriting. This lane detected the concurrent drift and retained it rather than overwriting foundation work.

## Strengthened static policy

`tools/check_import_boundaries.py` now enforces the following in addition to the original layer rules:

- runtime/model third-party roots must belong to declared base roots (`ttnn`, `torch`, `loguru`) or declared model-optional roots (`transformers`, `tqdm`);
- model-optional roots are forbidden in `llm_runtime`;
- unknown third-party roots fail the runtime/model product check;
- `tt_transformers/__init__.py` cannot eagerly import model packages;
- `tt_transformers/models/__init__.py` cannot eagerly import concrete model modules or optional dependencies;
- runtime/model `from_model_args` methods fail structurally;
- the removed `Llama3Transformer1D.forward` shape fails structurally if reintroduced;
- `SamplingParams` imported from the aggregate package or `sampling.generator` fails in favor of the canonical owner.

Dependency-root enforcement is intentionally scoped to `llm_runtime` and `models` in this lane. The foundation's guarded `yaml` path remains a separately recorded packaging decision and was not silently declared or edited here.

The existing host boundary test invokes this checker as a subprocess, so it now exercises the stronger rules without a test-file modification. The existing base-package surface test also verifies the package root remains importable.

## Dependency and laziness audit

Direct third-party imports in runtime/models are exactly:

| Root | Import statement sites | Files | Policy |
|---|---:|---:|---|
| `ttnn` | 59 | 59 | declared base |
| `torch` | 64 | 64 | declared base |
| `loguru` | 15 | 15 | declared base |
| `transformers` | 13 | 13 | declared model optional |
| `tqdm` | 9 | 9 | declared model optional |

A clean source-layout probe imported `tt_transformers` and `tt_transformers.models` while asserting:

- `transformers` was absent from `sys.modules`;
- no `tt_transformers.models.*` concrete module was loaded;
- no HF adaptor was loaded.

Thus optional dependencies remain isolated behind explicit concrete-model imports rather than package-root imports.

## Exact audit commands

Static boundary:

```bash
cd /localdev/gwang/tt_transformers
PYTHONDONTWRITEBYTECODE=1 python tools/check_import_boundaries.py src/tt_transformers
```

Result: exit code 0, no output.

Extractor drift checks:

```bash
cd /localdev/gwang/tt_transformers
PYTHONDONTWRITEBYTECODE=1 python tools/extract_runtime.py --check
PYTHONDONTWRITEBYTECODE=1 python tools/extract_model_packages.py
```

Results: both exit 0; 24 runtime/shared-executor and 71 concrete-model destinations match their pinned-plus-characterized transformations.

Residual edge/adapter audit:

```bash
rg -n '^\s*(from|import) models(\.|\s|$)|^\s*(from|import) (tests|examples|pytest)(\.|\s|$)' \
  src/tt_transformers/llm_runtime src/tt_transformers/models
rg -n '^\s*def from_model_args\b|from tt_transformers\.sampling import SamplingParams|Llama3Transformer1D\.forward' \
  src/tt_transformers/llm_runtime src/tt_transformers/models
```

Result: no matches.

Optional-laziness probe:

```bash
cd /localdev/gwang/tt_transformers
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/localdev/gwang/tt_transformers/src python - <<'PY'
import sys
import tt_transformers
import tt_transformers.models

assert "transformers" not in sys.modules
assert not any(name.startswith("tt_transformers.models.") for name in sys.modules)
PY
```

Result: exit code 0.

## Syntax and host validation

Syntax validation uses an external bytecode cache:

```bash
cd /localdev/gwang/tt_transformers
boundary_cache=$(mktemp -d /tmp/gwang/tttv2-runtime-boundary-pyc.XXXXXX)
PYTHONPYCACHEPREFIX="$boundary_cache" python -m compileall -q -f \
  tools/check_import_boundaries.py \
  tools/extract_runtime.py \
  tools/extract_model_packages.py \
  src/tt_transformers/llm_runtime \
  src/tt_transformers/models
```

Result: exit code 0.

Host policy/package tests can run without the hardware conftest by bounding conftest discovery and using the source layout:

```bash
cd /localdev/gwang/tt_transformers
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/localdev/gwang/tt_transformers/src \
  python -m pytest -q --confcutdir=tests/host \
  tests/host/test_import_boundaries.py \
  tests/host/test_package_surface.py
```

Result: `3 passed`.

The broader focused runtime/model pytest command is explicitly blocked in this environment before collection:

```text
ImportError while loading conftest ...
ModuleNotFoundError: No module named 'ttnn'
```

The only available source-only TTNN package also lacks its compiled `ttnn._ttnn` extension, so substituting it would not be a valid host qualification environment. No hardware test or reset was attempted.

## Acceptance verdict

- Static boundary across runtime/models: pass.
- Whole-tree static checker at the integrated checkpoint: pass.
- Runtime/model syntax: pass.
- Existing host policy/package tests: pass (`3 passed`).
- Broader focused runtime/model tests: environment-blocked by absent installed TTNN; exact blocker recorded.
- Optional HF dependencies remain lazy at package/model root: pass.
- Pinned extraction reproducibility after Phase 3 transformations: pass.
