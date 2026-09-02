# Phase 3 support-boundary closure

## Result

The extracted test, example, documentation, readiness, and qualification
surface is closed over standalone owners. This work used pinned source revision
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0` and changed only `tests/`,
`examples/`, `qualification/`, and the support extraction scripts/manifests.
It did not modify `src/tt_transformers`, use a remote machine, run TT hardware,
reset devices, or edit the main migration work log.

## Provenance and readiness reconciliation

The corrected provenance authority contains 366 source rows. Phase 2 now
extracts 206 support-side source rows through 223 source/destination
assignments into 220 distinct destinations. This includes the pinned
cache-entry counter required by the retained device fixtures, now owned by
`tests/support/cache_entries_counter.py`.

All 21 files formerly listed as unassigned readiness assets now have explicit
provenance destinations:

- 15 tools, package files, and text assets under `qualification/readiness/`;
- 6 test modules under `tests/qualification/readiness/`.

Their pinned blob IDs are verified by `extract_support_surface.py`, their
internal imports use `qualification.readiness`, and
`support_unassigned.csv` has been removed after its empty-state check. The
readiness suite collects and passes without `openai` installed; the live-server
path raises a clear runtime error only when that optional client is actually
needed.

## Hybrid example/test separation

Fourteen duplicated source modules were split by ownership:

- twelve end-to-end model demos;
- Qwen2.5-Coder-32B package smoke;
- Qwen3-32B package smoke.

Each `examples/<model>/demo.py` or `smoke.py` is now a public runner module:

- no `pytest` import, decorator, fixture, `pytestmark`, or `test_*` function;
- no `assert` AST node;
- no import from `tests.*`;
- original gate conditions preserved as explicit `AssertionError` failures;
- original inapplicable configurations preserved as
  `UnsupportedConfiguration`;
- one CLI `main()` with model cases, optimization profile, and mesh lifecycle;
- device locking, fabric ownership, full-parent/submesh creation, teardown,
  and temporary cache ownership supplied by `examples.common.runtime`.

Every corresponding `tests/hardware/models/<model>/test_demo.py` or
`test_smoke.py` preserves the original pytest function names, argument names,
parameter values, parameter IDs, markers, and skip semantics. The wrappers
import the public example module, invoke its `run_*` helper, and translate
`UnsupportedConfiguration` back to `pytest.skip`. Assertions raised by the
public runner remain test failures.

Shared example ownership is one-way:

- cleanup uses public `tt_transformers.device_utils`;
- benchmark/teacher-forcing helpers live in
  `examples.common.run_helpers` without test assertions;
- prompt compatibility helpers live in `examples.common.prompting`;
- the smoke PCC helper is narrowed to `examples.common.comparison`;
- Llama example helpers are public under `examples.llama3_8b.demo_utils`, with
  the test-side path reduced to a compatibility re-export.

## Deferred split closure

All 53 Phase 2 destinations marked for a semantic split or filter have a
recorded resolution in `support_boundary_manifest.json`:

- 28 example/hardware destinations became public runner plus pytest wrapper;
- model targets retain exactly 12 TTTv2 HF product entries;
- trace-region sizes retain the 9 products with exact matching HF aliases;
- source-CI lists retain 6 e2e, 12 unit, and 7 sweep TTTv2 entries with
  standalone paths;
- large workflow locks became compact TTTv2 evidence-line manifests;
- the T3K dispatch script became a valid standalone test-path dispatcher;
- qualification tools have standalone imports and assets;
- broad test helpers now have distinct example, test, and package owners;
- split prompt/reference assets remain byte-identical in their assigned roles.

The three merged fixture sources were reduced enough for host-only collection:
the unrelated `models/conftest.py` image/performance segment was replaced by
GC hygiene, tt-metal helper imports were internalized in
`tests.support.fixture_policy`, TTNN is lazy for host collection, and device
fixtures retain their original runtime ownership behavior.

## Imports, paths, manifests, and assets

There are zero executable imports through `models.*` or `tests.scripts` in
active `tests/`, `examples/`, `qualification/tools`, or
`qualification/readiness` Python files. Examples also have zero imports from
`tests.*`.

Unambiguous executable asset paths now point to:

- `qualification/assets/sample_prompts/`;
- `qualification/assets/reference_inputs/`;
- `qualification/assets/reference_outputs/<model>/`.

All reference tensors copied into the neutral qualification root remain exact
binary copies. Old capability-manifest node IDs now point to hardware wrappers,
while `model.demo_entry_point` points to the public example. Schema references
resolve to `qualification/schemas/bh_required_capabilities.schema.json`.
Non-executable historical prose may still name the source tree where the text
is explaining provenance or the pre-migration design.

Each of the twelve examples now has an `experimental` `support.json` validated
against the Draft 2020-12 support schema. No manifest claims qualification:
validation date/SHA are null, evidence is empty, unpinned checkpoints say
`UNPINNED`, and the known-gaps list records the absence of pinned-revision
hardware evidence.

## Example runtime policy closure

All executable example Python files now have zero:

- `trust_remote_code=True` call sites;
- CWD-relative `Path("model_cache")` defaults;
- direct `SetDefaultDevice` or `GetDefaultDevice` calls.

The eleven demos that own a cache helper call the public
`tt_transformers.cache_environment.resolve_model_cache_path` policy with the
model revision, mesh, and topology. Explicit `TT_CACHE_PATH` behavior remains
`<TT_CACHE_PATH>/<topology>` exactly once; implicit paths use
`TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache and never the
invocation directory. Llama 3.1-8B already delegates cache policy through its
adaptor.

The six runnable functions in each Qwen smoke example use
`default_device_scope`. Its process lock and `finally` restoration preserve
the exact prior device on normal return and exceptions; no function restores
blindly to `None`. `functools.wraps` plus `inspect.signature` preserves CLI
argument discovery after scoping.

`example_runtime_policy_manifest.json` records deterministic hashes and zero
violation lists for 34 example Python files, plus the twelve scoped smoke
functions. Support manifests and generated READMEs record the same cache and
remote-code defaults.

## Verification

Completed host checks:

- `PYTHONPATH=src pytest -q tests/host` — **47 passed**;
- `PYTHONPATH=src pytest -q tests/qualification/readiness` — **23 passed**;
- combined host/readiness run — **70 passed**;
- static import-boundary checker — passed;
- all 12 `support.json` files validated with `Draft202012Validator`;
- 200 Python files under the support roots parsed successfully;
- all qualification JSON and YAML parsed successfully;
- all extracted shell tools passed `bash -n`;
- all 53 deferred destinations are present in the Phase 3 resolution ledger;
- active legacy executable-import count is zero;
- example pytest/test-boundary AST violations are zero.
- example forced-remote/CWD-cache/unscoped-default-device violations are zero;
- the focused example runtime-policy host suite passes 8 tests.

Hardware collection was attempted only as `--collect-only`, with no device
opened. It stops because this coordinator environment does not have the
`ttnn` package installed: all 14 wrapper imports fail at their public example's
`import ttnn`. This is an environment collection blocker, not a functional or
hardware result. Hardware collection and execution must occur in the supported
TTNN environment and must follow the serialized hardware procedure.

## Reproduction and audit

Phase 2 pinned extraction:

```text
python -B qualification/extraction/extract_support_surface.py
```

On an already-closed tree this command refuses to overwrite Phase 3 output.
An intentional from-scratch rebuild must remove the boundary manifest or set
`TT_TRANSFORMERS_REEXTRACT_PHASE2=1`, then immediately rerun Phase 3.

Phase 3 closure (runs once after Phase 2 and records a guard manifest):

```text
python -B qualification/extraction/close_support_boundary.py
```

Idempotent example-policy and documentation regeneration:

```text
python -B qualification/extraction/apply_example_runtime_policy.py
python -B qualification/tools/generate_support_docs.py
python -B qualification/tools/validate_support_docs.py
```

`support_copy_manifest.csv` remains the pinned source/copy ledger.
`support_boundary_manifest.json` records the final support-tree hashes and all
53 semantic resolutions. Neither script imports a model or executes tests.
