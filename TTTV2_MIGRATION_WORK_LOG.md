# TTTv2 Migration Work Log

This log records milestones and evidence for the standalone migration defined in
`TTTV2_MIGRATION_PLAN.md`. Times are UTC.

## 2026-09-02T03:35:21Z — Milestone 0: plan intake and goal established

- Read the complete 706-line migration plan and the complete 704-line remote
  hardware instruction manual.
- Created the tracked `/goal` for the full standalone migration, including
  extraction, dependency-boundary closure, packaging, qualification, support
  documentation, and release/consumer-readiness evidence.
- Confirmed the deferred HF-like direct-generation/vLLM facade redesign remains
  out of scope for this migration.
- Confirmed the destination starts at `main` commit
  `48959bd9167e6faa4466fc7b8f5c711a601b7615` with only `README.md` tracked;
  `TTTV2_MIGRATION_PLAN.md` is a pre-existing untracked user file and will be
  preserved.
- Confirmed the pinned source commit
  `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0` exists in
  `/localdev/gwang/tt-metal`. The source worktree has only untracked
  `models/docs/tttv2/` content, which does not alter the pinned Git snapshot.
- Hardware policy: all TT jobs will be serialized. Before the first remote
  connection, the reservation table will be read in a standalone command;
  participating checkouts must then pass the mandatory synchronized-checkout
  gate. `tt-smi -r` will be used only after a confirmed device/lifecycle fault.

## 2026-09-02T03:40:14Z — Milestone 1A: minimal standalone project scaffold

- Added PEP 621 metadata for the working distribution name `tt-transformers`,
  a `src/tt_transformers` package, lazy top-level model ownership, separate
  `examples`, `test`, and `dev` extras, pytest hardware markers, formatting,
  linting, typing, and coverage configuration.
- Added Apache-2.0 `LICENSE` and source `NOTICE`, changelog, contribution and
  security policies, initial example-support documentation, TTNN 0.77 candidate
  constraints, and a draft JSON Schema for model support manifests.
- Verified from PyPI that `ttnn==0.77.0` was published for CPython 3.10 and 3.12
  as `manylinux_2_34_x86_64` wheels. Candidate Torch, Transformers, Loguru,
  tqdm, and pytest pins currently mirror the `v0.77.0` tt-metal development
  environment and remain provisional pending isolated qualification.
- `python -m compileall -q src`, JSON parsing of the support schema, and a
  direct `PYTHONPATH=src` import all passed on Python 3.10.19.
- The first no-isolation wheel build exposed an unnecessarily exact build-time
  Setuptools pin (`80.10.2`) against the available `80.0.0`; changing the build
  requirement to the compatible floor `setuptools>=80.0` made the build pass.
- Built `dist/tt_transformers-0.1.0.dev0-py3-none-any.whl`, inspected its file
  list, installed it with `--no-deps` into a fresh venv outside the repository,
  and imported it successfully as version `0.1.0.dev0`.
- Python 3.12 is not installed in the coordinating environment, so the 3.12
  clean-install half of the Phase 1 matrix remains pending on an appropriate
  host or CI runner. Full dependency installation was deliberately not claimed
  by this `--no-deps` smoke.

## 2026-09-02T03:41:47Z — Milestone 1B: initial host and CI policy gates

- Added an AST-based import-boundary checker enforcing the legacy `models.*`,
  tests/examples/pytest, modules-to-runtime/model, and runtime-to-model rules.
- Added base-package, support-schema, and import-boundary host tests.
- Added pre-commit configuration and a Python 3.10/3.12 GitHub Actions matrix
  that builds the wheel, installs it without runtime dependencies, runs the
  host gates, and prints wheel contents.
- `python tools/check_import_boundaries.py`, `PYTHONPATH=src pytest -q
  tests/host` (3 passed), and `git diff --check` passed.
- Ruff is not installed in the coordinating environment, so the configured
  Ruff gate was not run locally; the command failed before analysis with exit
  127 and remains pending in a provisioned dev environment.

## 2026-09-02T03:43:31Z — Checkpoint 0A: pinned-source static baseline

- Confirmed `/localdev/gwang/tt-metal` is exactly at the migration source SHA
  `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`.
- The hardware manual's scans for forbidden caller-facing architecture config
  wrappers and execution-path `is_blackhole()`/`get_arch_name()` calls produced
  no matches. `git diff --check` passed.
- Syntax compilation of reusable modules and the three BH-enabled model
  packages passed after redirecting bytecode to
  `/tmp/gwang/tttv2-source-pyc`; the read-only source checkout correctly
  rejected the initial attempt to create local `__pycache__` files.
- The manual's historical WH zero-diff comparison against
  `6de73d1ec382279f640f4b01c52076ca5737e06c` does **not** pass at the pinned
  migration SHA: the nine WH-only product packages contain later runtime and
  documentation changes. This is recorded as source characterization, not
  relabeled as a WH regression result.
- Source pytest collection is currently blocked in the coordinating Python
  environment: there is no source `python_env`, and `/opt/venv` exposes an
  incomplete `ttnn` namespace lacking `ttnn.device`. Collection stopped in the
  root `conftest.py` with `ModuleNotFoundError` before collecting or opening any
  device. No host-test pass is claimed.

## 2026-09-02T03:52:07Z — Milestone 0 complete: frozen characterization baseline

- Final provenance inventory: 342 unique pinned-tree rows, 316 assigned
  destinations, 26 explicit MoE-only exclusions, and 73 boundary-cleanup rows.
  Every Git blob SHA was verified at source revision `00748e6...`; the CSV
  SHA256 is `468b8fc147d759297b53ebdbfea71024ec5b04a201e82ab30d28dee6e0ec6849`.
- Production census: 113 Python files / 52,645 lines (16 module, 21 runtime,
  76 model), with 1,760 public-symbol/signature/re-export characterization
  rows and 1,891 imported-symbol rows.
- The existing internal graph already respects the target direction: modules
  do not depend on runtime/models and runtime does not depend on concrete
  models. Mechanical namespace extraction can preserve the layering.
- Boundary debt is fully assigned: 119 external `models.*` symbol rows at 87
  sites across 14 modules; 60 unstable TTNN uses across 13 experimental APIs
  and one private API; 6 production-to-test import rows; 10 `from_model_args`
  factories, 3 legacy adapters, and 21 TTTv1 namespace rows.
- Runtime assumptions include 108 accesses to 21 environment keys, 38 global
  TTNN default-device mutations, 10 default-device fallbacks, CWD-relative
  cache roots in all twelve HF adaptors, and incomplete cache fingerprints.
- Test/support census: 111 files, 1,368 source-level test definitions, 63
  fixtures, 23 marker tokens, 71 support assets, and 53 normalized hardware
  coverage rows, all pinned by blob SHA.
- All twelve model packages are concrete runnable source products, but only
  five pin HF revisions and only three have BH pre-acceptance contracts.
  Presence, collected cases, contracts, and skips are not support evidence.
- No hardware evidence in the tree is attributable to the pinned source SHA.
  Manual evidence at `b1a75d...` is ineligible and that SHA is not an ancestor
  of `00748e6...`; no historical hardware pass was transferred.
- Deterministic regeneration and clean-directory comparisons passed for all
  Phase 0 CSV/JSON reports. The three agents completed their dedicated goals
  and then received fresh dedicated extraction goals for disjoint paths.

## 2026-09-02T03:55:09Z — Milestone 2A: twelve concrete model packages extracted

- Added a deterministic, blob-verifying extractor and mechanically moved 71
  files across all twelve concrete model packages into
  `src/tt_transformers/models/`.
- The inventory's other three `production_model` rows are shared executors and
  remain owned by the parallel runtime lane. Model READMEs, hybrid demos,
  tests, and reference assets remain owned by the parallel support lane.
- Rewrote only the fixed namespace prefixes for modules, runtime, models,
  sampling, narrow helpers, and `mesh_utils`; signatures, tuning, cache and
  environment policy, checkpoint revisions, and execution behavior were not
  changed.
- Deterministic destination verification, syntax compilation, and
  `git diff --check` passed. The twelve model packages contain no residual
  `from/import models.*` statements.
- Repository-wide boundary checking currently fails only on deliberately
  deferred module-foundation TTTv1/helper edges being handled in Phase 3; it
  reports no violation in the concrete model packages.

## 2026-09-02T04:55:06Z — Milestone 2B: foundation and support surfaces extracted

- The transitive boundary audit caught and repaired the sole missing
  provenance source: `models/common/utils.py`, which owns the required
  `LogProbsCalculator`. The corrected inventory contains 343 rows, 317 assigned
  destinations, 26 MoE-only exclusions, 42 split rows, and 74 boundary-cleanup
  rows; all external dependency owners now have inventoried sources. Corrected
  inventory SHA256:
  `2778c886e012044f4a07dfe39d231a6821419b5de822c8e1a93ba1115d7a41e6`.
- Extracted the reusable foundation: 27 direct rows (16 modules, 8 sampling,
  3 support) plus narrow lifecycle, mesh, architecture, and log-probability
  splits. Syntax compilation passed. The boundary checker now reports exactly
  six deliberately retained TTTv1 bridge imports for Phase 3 removal.
- Extracted 183 support/provenance rows into 197 destinations: 128 tests, 35
  examples, and 34 qualification paths. Binary/reference assets are exact and
  three shell entry points retain executable mode.
- Fourteen hybrid demos were copied to both assigned example/hardware-test
  destinations for behavior preservation; semantic separation from pytest and
  test helpers remains an explicit Phase 3 task. Three source fixture snapshots
  were merged into `tests/conftest.py` without deleting fixture intent.
- Applied 656 unambiguous support namespace rewrites. Deterministic copy
  verification, 153 Python AST parses, JSON parsing, and shell syntax/mode
  checks passed. Twenty-one readiness assets without provenance destinations
  are recorded in `support_unassigned.csv` rather than silently invented.

## 2026-09-02T04:55:06Z — Checkpoint 4A: TTNN 0.77 Python 3.10 symbol probe

- Installed the published `ttnn==0.77.0` wheel and its declared dependencies
  into a fresh Python 3.10.19 environment outside both repositories.
- All 14 unstable/private API paths used by the pinned implementation resolve:
  13 `ttnn.experimental` APIs and
  `ttnn._ttnn.tensor.dump_tensor_flatbuffer`.
- Most operations expose only a generic callable signature and the private
  nanobind function has no inspectable signature, so this is symbol-availability
  evidence only. Semantic, full dependency, Python 3.12, and device
  qualification remain pending; no hardware claim is made.

## 2026-09-02T05:00:02Z — Milestone 2C: runtime and shared executors extracted

- Mechanically extracted all 21 runtime Python files plus the three shared
  executor files into their standalone owners.
- The reproducible extractor verifies the pinned commit, all 24 blob SHAs and
  provenance rows, applies 95 import-module-only rewrites, and proves normalized
  AST equivalence. Existing package initializers were preserved.
- Extraction drift checking, syntax compilation, and selected boundary checks
  passed. The 24 runtime/shared-executor files contain no legacy/test import
  violations; the only whole-tree violations remain the six known foundation
  bridge edges assigned to Phase 3.
- Duplicate aggregate-versus-submodule `SamplingParams` ownership remains
  explicitly documented for canonicalization rather than being silently
  conflated during the mechanical move.

## 2026-09-02T05:00:02Z — Checkpoint 4B: Python 3.10 base dependencies resolved

- Added the candidate `torch==2.11.0` CPU wheel and exact `loguru==0.6.0` pin to
  the isolated TTNN 0.77 environment. Imports report TTNN 0.77.0, Torch
  `2.11.0+cpu`, and Loguru 0.6.0.
- This establishes that the candidate base dependency tuple resolves on the
  coordinating Python 3.10/glibc environment. The extracted package wheel and
  its module surfaces still need to be built/installed against this environment
  after concurrent Phase 3 edits settle.

## 2026-09-02T05:06:03Z — Milestone 1 complete: two-Python minimal wheel matrix

- Provisioned a temporary managed CPython 3.12.13 runtime outside the
  repository because the coordinating image did not provide Python 3.12.
- Installed the already-built pure-Python wheel with `--no-deps` into a fresh
  Python 3.12 venv and imported both its distribution metadata and
  `tt_transformers` namespace successfully as version `0.1.0.dev0`.
- Together with the earlier fresh Python 3.10.19 wheel install/import, the
  Phase 1 minimal-wheel exit matrix now passes for both intended interpreters.
  Full extracted code and runtime dependencies remain subject to later phase
  gates; this milestone does not broaden compatibility claims.

## 2026-09-02T05:08:23Z — Milestone 3A: reusable foundation boundary closed

- Removed all ten characterized module `from_model_args` TTTv1 factories and
  both legacy RoPE adapters after extracted caller scans.
- Updated the two proven retained RoPE call sites to use
  `prepare_rot_idxs`/`decode_forward`; concrete models already used direct or
  `from_config` construction.
- Replaced the legacy Mode import with model-neutral
  `tt_transformers.modules.mode` ownership while preserving the exact
  `decode`/`prefill` values and dispatch behavior.
- Removed 20 legacy-factory-only tests and retained direct-constructor
  characterization. Whole-source static boundary checking is clean, syntax
  compilation passed, and the focused host boundary suite reports 16 passed.
- The readiness completeness repair added all 21 pinned
  `models/common/readiness_check` assets. The authoritative provenance now has
  364 rows, 338 assigned destinations, 26 MoE-only exclusions, and 86 boundary
  cleanup rows; every blob verifies and every support inventory path is
  assigned. Inventory SHA256:
  `210710759200fc8c3ca0f9ab78204c6aa4700849486bf2869004538609376595`.

## 2026-09-02T05:11:21Z — Milestone 3B: runtime/model import boundary closed

- Removed the unused `Llama3Transformer1D.forward` legacy dispatcher after an
  exhaustive source/test/example/qualification caller audit; retained runtime
  paths already call explicit `prefill_forward` and `decode_forward` methods.
- Canonicalized the last five aggregate sampling imports. Runtime/models now
  have twelve imports solely from
  `tt_transformers.sampling.sampling_params` and no aggregate duplicate owner.
- Strengthened the import checker for declared layer-specific third-party
  roots, optional dependency placement/root laziness, canonical sampling,
  factory/dispatcher regressions, and old namespace/test imports.
- Updated pinned extractors so deterministic drift checks reproduce the
  approved bridge removals, canonical imports, and concurrent RoPE caller
  migrations.
- Whole-tree static checking, external-cache compilation, both extractor
  checks, optional-dependency laziness, and the existing host policy suite
  passed (3 tests). Runtime/models have zero old namespace, test, pytest, or
  TTTv1 edges.
- Broader runtime/model pytest collection in the coordinating environment is
  still blocked before collection because the source-only TTNN namespace lacks
  the installed `_ttnn` extension; this is not recorded as a test pass/fail.

## 2026-09-02T05:13:06Z — Checkpoint 4C: TTNN 0.77 Python 3.12 symbol parity

- In a fresh managed CPython 3.12.13 venv, installed TTNN 0.77.0, Torch
  `2.11.0+cpu`, and Loguru 0.6.0; `pip check` passed.
- Final probes ran from `/tmp/gwang` with `PYTHONPATH` removed. TTNN, Torch, and
  Loguru resolved only inside the venv, with neither the `tt-metal` checkout nor
  destination `src` importable.
- All 14 unstable/private paths covering 60 source occurrences resolve, matching
  the Python 3.10 symbol result. The report explicitly excludes API invocation,
  semantic, hardware, firmware, SKU, accuracy, and performance claims.

## 2026-09-02T05:22:27Z — Milestone 3C: public API policy frozen

- Classified all 1,760 characterized source API/signature/re-export rows exactly
  once: 367 `supported_public`, 950 `model_local_public`, 13
  `compatibility_only`, and 430 `private_by_policy`.
- Preserved all 228 explicit source `__all__` rows as model-local public API;
  the root package remains intentionally limited to `__version__` and does not
  gain eager model/HF exports.
- Signature accounting is complete: 1,576 exact destinations, 153 namespace-only
  re-export rewrites, and 31 approved ledger rows with no unexplained absence or
  mismatch.
- The removal ledger records 13 approved compatibility removals (ten factories,
  two RoPE adapters, one Llama dispatcher) and 18 hybrid-demo symbols relocated
  to non-package example/test surfaces.
- Flagged 34 genuine cross-family exposure inconsistencies for human review,
  covering generator/config classes, default HF constants, and build factories;
  no arbitrary unification was performed during extraction.
- Deterministic regeneration and the API policy checker pass. Policy CSV SHA256:
  `44676ca2551d83e1ff49a23175ddc741a0fa49be7f719ebea6a3c5d820007346`.

## 2026-09-02T05:28:07Z — Milestone 2 complete / 3D: support extraction and boundary closure

- Reconciled the corrected 364-row provenance inventory. Support extraction now
  covers 204 source rows / 221 assignments / 218 destinations, including all
  21 readiness assets; the temporary unassigned ledger was removed.
- Resolved all 53 Phase 2 semantic split/filter destinations with role-specific
  content and recorded their final hashes in
  `support_boundary_manifest.json`.
- Split all 14 hybrid source modules into pytest-free runnable examples and 14
  delegating hardware wrappers. Wrappers preserve source test names, arguments,
  parameter IDs, markers, decorators, and unsupported-configuration skip
  semantics.
- Runnable examples contain zero pytest imports/decorators/fixtures/test
  functions/assert AST nodes and zero imports from tests. Active
  tests/examples/qualification contain zero executable old `models.*` or
  `tests.scripts` imports.
- Added 12 schema-valid `experimental` support manifests and neutralized
  prompt/reference asset paths. None claims pinned-revision hardware evidence.
- Made the readiness live-vLLM OpenAI dependency collect-safe and failure-
  explicit when the optional client is actually requested.
- Passed 47 host tests and 23 readiness tests; the static boundary checker,
  200-file Python parse, JSON/YAML/shell validation, provenance/blob/binary
  verification, and 319 final support-file hash checks all passed.
- Hardware-wrapper collection remains environment-blocked on this coordinator
  because TTNN is not installed in its default interpreter: 14 example import
  errors occur before any test collection/device open. No hardware result is
  inferred.

## 2026-09-02T05:32:49Z — Milestone 3E/7A: explicit cache and environment policy

- Added a typed environment/cache/preflight policy and integrated it across all
  twelve HF adaptors without changing their public construction APIs or
  model-specific topology suffixes.
- Implicit cache roots now resolve through `TT_TRANSFORMERS_CACHE`, then
  `XDG_CACHE_HOME/tt-transformers`, then `~/.cache/tt-transformers`; no adaptor
  defaults to a CWD-relative `model_cache` path.
- Explicit `cache_dir` remains exact. `TT_CACHE_PATH` remains exact for eleven
  adaptors, while Llama 3.1 8B preserves its device suffix and permission
  fallback semantics.
- `CI` no longer implicitly forces offline checkpoint behavior. Explicit
  `TT_TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE`, and `TRANSFORMERS_OFFLINE`
  controls use strict normalized booleans.
- Secret-redacted preflight reports include a complete version identity over
  package/TTNN/checkpoint/conversion/architecture/topology/dtype/layout/sharding
  inputs. The identity is intentionally report-only for this compatibility
  phase and is not injected into established warm-cache paths, avoiding forced
  rematerialization before hardware qualification.
- No model/checkpoint/cache construction ran. Static audits show zero direct
  adaptor environment access or CWD defaults and twelve preflight call sites.
  Compilation, checker/extractor drift policy, 35 focused tests, and the full
  host suite (87 passed) succeeded.

## 2026-09-02T05:33:44Z — Milestone 3F: canonical sampling and lazy optional imports

- Removed the duplicate sampling dataclass. Generator helpers, runtime/models,
  and the lazy aggregate export now resolve the same frozen
  `sampling.sampling_params.SamplingParams` identity with all nine fields and
  defaults unchanged; representative broadcast/slice/chunk behavior preserves
  that identity.
- Converted all eleven existing concrete model initializers to cached lazy
  export maps without changing ordered `__all__`; the source-init-less Llama
  3.1 8B package has an intentionally empty `__all__`.
- Moved Qwen3's Transformers import into `from_pretrained`. All twelve model
  packages and tensor-core modules now import with Transformers/tqdm explicitly
  blocked, while optional HF/generator symbols load only when requested.
- Updated deterministic foundation/model extractors and static/public API
  policy checks for the approved transforms. Public classifications are
  unchanged; signature accounting now records 1,576 exact, 152 lazy-export,
  one namespace-only, and 31 approved ledger rows.
- Foundation/model extractor checks, the whole static checker, API checker,
  compilation, and 28 focused host cases passed.

## 2026-09-02T05:35:37Z — Milestone 6A: experimental support contracts documented

- Added a root `SUPPORT.md`, a complete examples matrix, standardized support
  sections in all twelve active example READMEs, and enriched machine-readable
  manifests.
- Every model remains explicitly `experimental`; concrete/runnable source is
  distinguished from qualification. Five exact HF revisions are recorded and
  seven models carry prominent `UNPINNED` blockers.
- Each contract records the candidate 0.1.0.dev0 / TTNN 0.77.0 / Python
  3.10+3.12 / Torch 2.11 / Transformers 5.12 tuple, source-proven
  architecture/SKU/mesh/TP/DP and limits, trace/sampling caveats,
  cache/offline/assets, exact install/run/collect commands, correctness
  criteria, null validation date/SHA/evidence, unsupported gaps, and links.
- No model is marked qualified and no missing hardware evidence is inferred.
- Added deterministic docs generation and validation. Regeneration is
  byte-stable, all twelve manifests/READMEs validate, the static checker passes,
  and the host suite reports 88 passed.
- Added direct `jsonschema` test dependency and an isolated `qualification`
  extra for the OpenAI/requests packages used only by live vLLM readiness work.

## 2026-09-02T05:43:00Z — Milestone 3G: process-global TTNN device ownership scoped

- Added a single process-global compatibility owner with an exact-restore,
  re-entrant lock scope. Nested, repeated, body-failure, partial-set-failure,
  and cross-thread construction cannot leak or observe another temporary
  default device.
- Removed fourteen installed-production default-device mutations: eleven HF
  adaptors and the Qwen2.5-72B, Qwen2.5-Coder-32B, and Qwen3 builders. Llama
  3.1 8B already had no write.
- Direct `SetDefaultDevice` and `GetDefaultDevice` calls outside the owner module
  are both zero. The owner contains only three scoped set sites and two read
  sites.
- Retained exactly ten compatibility fallback calls for simple constructors;
  each matches one explicit code ledger entry and raises for an unregistered
  owner or absent scoped default.
- Compilation, whole-tree checker, deterministic extractors, AST ledger audit,
  nine focused fake-TTNN ownership tests, and the full host suite (98 passed)
  all succeeded. No device/checkpoint/cache operation ran.

## 2026-09-02T05:44:24Z — Milestone 5A: explicit test and CI taxonomy

- Applied behavior-neutral explicit taxonomy to 1,403 test functions: 1,150
  host and 253 device, including 390 model and 27 slow definitions. Every test
  has exactly one host/device classification; SKU marks imply the proven
  architecture/device requirements.
- Added exact static topology marks and collection-safe runtime `MESH_DEVICE`
  augmentation for all fourteen hardware wrappers without probing hardware.
- Preserved test names, parameters/IDs, assertions, skip/xfail semantics, and
  commands; marker application is byte-idempotent.
- Added deterministic taxonomy/audit tools, Python 3.10/3.12 host-only CI, and
  a static device-policy job. CI does not execute device suites.
- The pyramid report provides focused/scheduled/release selectors, mandatory
  one-node/one-process TT serialization, teardown/reset rules, and the rule
  that a skip is never support evidence.
- Taxonomy audit, 121-node host collection, 121 host executions, CI YAML and
  137 test-file parses, and the import boundary all passed. No device opened.

## 2026-09-02T11:41:36Z — Milestone 8A: downstream cutover ledger prepared

- Read-only consumer analysis inventoried 209 current sites/gaps: 64 downstream
  Python imports, 107 CI/test-path references, 9 qualification duplicates, 12
  vLLM generator registrations, 15 selector/error references, one missing vLLM
  registry-test gate, and one missing plugin dependency contract.
- All vLLM TTTv2 registrations still target `models.common.*`; the plugin still
  defaults to legacy products, declares only `tblib`, and has no TTTv2 registry
  tests. One experimental Quasar consumer lacks a symbol-compatible replacement.
- One test under TTNN must relocate rather than introduce a forbidden reverse
  dependency. The audit found no current production `ttnn/ttnn` or `ttnn/cpp`
  dependency on TTTv2 and proposes none.
- The report records consumer owner, replacement package path/version contract,
  shim need/duration, test gate, ordering, and nine-step one-way cutover with
  split-brain checks. Every row remains `not_cut_over`; no external repository,
  vLLM test/server, remote, or hardware was changed or run.

## 2026-09-02T11:41:36Z — Checkpoint 4D: first full isolated host collection

- The Python 3.10 TTNN 0.77 environment now resolves the declared base,
  examples, and test packages with `pip check` clean.
- The first whole-suite `-m host` command omitted the required `PYTHONPATH=src`,
  so package-import errors from that attempt are environment invocation errors
  and not product failures.
- Collection nevertheless exposed three real support defects before execution:
  one taxonomy marker inserted before a `from __future__` import, one validator
  test using an obsolete sibling path, and eager `requests` import in optional
  live-vLLM readiness code. A dedicated agent goal is correcting generator
  reproducibility and rerunning the exact host suite with the package path and
  qualification dependencies. No device opened.

## 2026-09-02T11:50:14Z — Milestone 4E: dependency groups resolve on both Pythons

- Fresh CPython 3.10.19 and 3.12.13 environments both resolve the declared
  base/examples/test tuple and pass `pip check` plus all nine direct imports
  with `PYTHONPATH` unset, no checkout on `sys.path`, and no package install.
- The exact direct tuple is TTNN 0.77.0, Torch `2.11.0+cpu`, Loguru 0.6.0,
  Transformers 5.12.1, tqdm 4.66.3, pytest 9.0.3, pytest-cov 7.0.0,
  pytest-timeout 2.4.0, and huggingface-hub 1.29.0.
- Recorded complete transitive freezes and per-distribution wheel metadata: 60
  distributions on Python 3.10 and 56 on Python 3.12. Interpreter-specific
  dependency differences are documented.
- An initial discarded Python 3.10 attempt used the CPU index as primary and
  encountered dependency metadata/flit-core fallback failure. The passing
  configuration uses PyPI as primary and the PyTorch CPU index as supplemental;
  final conflict lists are empty on both interpreters.

## 2026-09-02T11:52:43Z — Checkpoint 4F: full Python 3.10 host baseline executes

- After correcting the three initial collection defects and invoking with
  `PYTHONPATH=src`, the full TTNN 0.77 host-marked suite collected and executed:
  1,940 passed, 117 failed, 14 skipped, 6,790 deselected, and 81 subtests passed
  in 18.44 seconds. No device opened.
- Failures are not hidden or relabeled. The largest class is mechanical test
  characterization still naming old generator modules, source-tree paths,
  unsplit demo text, or pre-migration capability hashes. A support agent owns
  those updates under the approved relocation ledger.
- Smaller potentially substantive groups remain separated for triage: TTNN
  0.77 model-profile/config differences, sampling seed-state behavior,
  tensor-utils program-config serialization, and any true runtime/model
  contract failures. These will not be bulk-updated as path rewrites.

## 2026-09-02T11:59:52Z — Milestone 7B: full wheel and sdist audit

- Built and audited the full extracted distribution. Final audited wheel:
  594,085 bytes / 138 files / SHA256
  `a4ff402c4016f5cd97ddfdc0b2ad3cc7c1684aec1a2316cf8b885ce7ed301ca4`;
  sdist: 506,595 bytes / 171 members / SHA256
  `e4cb7744a65d27bb0d2ee6d471a65f6fc0e485787b7c0b6732a7fe7c0af2b5da`.
- Fixed two discovered packaging defects: restored the pinned LM-head
  `_nearest_32` alias consumed by eleven model modules, and declared guarded
  PyYAML functionality as a non-base `yaml` extra with dependency-root checks.
- Archive/source-byte, Core Metadata, PKG-INFO, WHEEL, RECORD, dependency, path,
  and forbidden-payload policies pass. The wheel has 132 source-identical
  Python files plus six metadata/license files and no tests/examples/
  qualification, weights/reference tensors/caches, pyc, old namespace, absolute
  path, or embedded workspace path.
- Fresh non-editable CPython 3.10 and 3.12 installs use the exact base tuple.
  With Transformers/tqdm/pytest absent and actively blocked, both import 76
  surfaces including reusable modules, runtime/shared executors, all twelve
  model packages, and all twelve core model modules from site-packages only.
  Static boundary, cache/device preflight, cleanup ownership, and `pip check`
  pass on both.
- Local Twine 4.0.2 itself crashes before reading artifacts due its own metadata
  adapter `KeyError('license')`. Direct `pkginfo` and complete metadata/archive
  validation passed and are recorded; no Twine pass is claimed.

## 2026-09-02T12:12:42Z — Milestone 4G: complete TTNN API availability audit

- Inventoried 160 unique maximal TTNN-root paths across 4,393 current production
  occurrences: 146 public, 13 experimental, and one private.
- Isolated Python 3.10 and 3.12 TTNN 0.77 wheels resolve all 160/160 paths with
  no source checkout on `sys.path` and without invoking constructors, operations,
  device queries, or hardware.
- All 14 previously identified unstable/private paths remain present. Signature
  evidence is categorized as informative, generic wrapper, or uninspectable;
  35 introspection-only differences between interpreters are recorded rather
  than treated as semantic differences.
- Typed dynamic audit covers 204 sites / 36 identities. Twenty-nine descriptors
  are present; five are complex annotations; two abstract compute-config
  descriptors are absent on placeholders but proven on concrete Wormhole
  configs in the official local v0.77.0 tag. Those remain semantic/signature
  ambiguities, not missing APIs.
- Added the sole missing test-referenced asset,
  `models/tt_transformers/model_params/Qwen3-32B/config.json`, to its
  provenance-backed qualification destination. Inventory now has 365 verified
  rows.

## 2026-09-02T12:12:42Z — Milestone 4H: TTNN 0.77 full host gate clean

- Final isolated Python 3.10 host result: **2,041 passed, 29 intentional skips,
  6,790 device/other deselections, 81 subtests passed, and 0 failures**.
- Mechanical old-path/demo/capability/factory expectations were closed under the
  approved relocation/removal ledgers rather than hidden as implementation
  changes.
- The one substantive TTNN 0.77 difference is that `SDPAProgramConfig.repr`
  raises in its binding while supported public attributes remain readable.
  `program_config_to_dict` now recursively serializes public non-callable
  attributes/to-JSON values and uses a guarded opaque fallback.
- A test-only `num_workers_per_dram_bank` expectation was removed because the
  field is absent from the 0.77 matmul config constructor and unused by
  production; no missing feature was invented or vendored.
- Tensor utility tests pass 15/15, the substantive target set passes 65 with 7
  expected deselections, and compilation/checker/extractor gates pass. No TT
  device opened.

## 2026-09-02T12:18:29Z — Milestone 4I: host taxonomy corrected and rerun

- Corrected the final taxonomy error: the 8x4 MLP2D topology-bug probe opens a
  mesh and is now `device` + `wormhole`, not host. The deterministic audit now
  classifies 1,403 functions as 1,149 host and 254 device with zero errors.
- Final exact isolated host rerun: **2,041 passed, 28 intentional skips, 6,791
  deselected, 81 subtests passed, and 0 failures/errors** in 20.02 seconds.
  No PCI/device query remains in host selection.
- The 28 skips are explicitly accounted for in the host report: deselected
  wrapper collection guards, one T3K-only case, and retired TTTv1 factory,
  generator, and legacy vLLM characterizations. None is hardware evidence.

## 2026-09-02T12:18:29Z — Milestone 7C: final post-triage artifacts published

- Rebuilt after the TTNN 0.77 serializer adaptation and latest metadata, safely
  replacing only the known 8,980-byte scaffold wheel in `dist/`.
- Final wheel: 594,387 bytes, SHA256
  `a0d5c0d9d5d1b01271402867008c7541e46028c94700060c8a78f0f3d2d99a92`.
  Final sdist: 506,875 bytes, SHA256
  `ab18a603880a5a3ba9db7d0735fac8336f69e8f7f12211693b45632d3b172ce0`.
- Wheel audit again proves all 132 package Python files are byte-identical to
  source plus six dist-info files, with zero forbidden payloads/paths. The
  source digest is
  `a44d58249fd8d59968ca344f3caa0215e3ec64d6bc4bb05e34a59c8c30354116`.
- Fresh non-editable Python 3.10/3.12 reinstalls pass all 76 imports, all twelve
  model/core packages, optional-package blocking, installed static boundary,
  cache/device/cleanup preflight, the TTNN 0.77 SDPA serializer probe, and
  `pip check`.

## 2026-09-02T12:33:39Z — Milestone 5B: serialized hardware runner ready

- Added a 42-node machine-readable matrix: 30 module, one runtime trace, three
  smoke, and eight end-to-end gates. Mesh counts are N150=9, N300=6, T3K=8,
  P150=8, and P150x4=11 across 23 Wormhole and 19 Blackhole nodes.
- Every node records an exact selector/keyword, environment and cache policy,
  timeout, allowed machine pool, physical-SKU provenance, acceptance types, and
  the complete evidence schema.
- The runner validates/lists/dry-runs or executes exactly one selected node. It
  requires caller-supplied external synchronization attestation, common full
  SHA/branch/machine/physical inventory, rechecks the local checkout, owns an
  atomic single-process lock, rejects xdist/parallel pytest options, separates
  failure classification, and never auto-resets hardware.
- P150x4 validation rejects arbitrary cluster identities and missing physical
  BDF provenance, preserving P150_X4 versus P300_X2 claims.
- Matrix validation and dry-run safety pass; the runner has no reservation,
  SSH, or `tt-smi -r` execution path. Fifteen parametrized runner cases pass.
  No reservation was read and no remote/device process ran.

## 2026-09-02T12:33:39Z — Milestone 4J: final two-Python host matrix

- After adding explicit host marks to the ten hardware-runner test definitions,
  taxonomy now covers 1,413 functions (1,159 host, 254 device) with zero errors.
- Exact Python 3.10 and Python 3.12 TTNN 0.77 host runs both report **2,056
  passed, 28 intentional skips, 6,791 deselected, 81 subtests passed, and zero
  failures/errors**.
- Python 3.12 emits nanobind reference-leak diagnostics during interpreter
  shutdown despite pytest exit success. This is recorded as TTNN binding
  feedback, not hidden and not converted to a TTTv2 test failure.
- Python 3.12 collection exposed `pytz` as a direct qualification/test utility
  dependency rather than host-transitive on both matrices; it is now declared
  alongside JSON Schema validation and the dependency reports are being
  refreshed.

## 2026-09-02T13:04:29Z — Milestone 7D: remote-code defaults secured

- Removed every forced Qwen `trust_remote_code=True` use: three tokenizer call
  sites and the Qwen2.5-72B shared config/model kwargs entry. A full audit of 51
  production `from_pretrained` calls now finds zero forced-true sites.
- Preserved only Llama3's existing explicit caller controls, both defaulting to
  false. Transformers 5.12 built-in registrations are used by default.
- Deterministic extractor/static gates, Python 3.10/3.12 compilation, seven
  focused mocked-HF tests, and the import boundary pass. No model/network/
  checkpoint execution occurred.

## 2026-09-02T13:04:29Z — Milestone 7E: all default checkpoints pinned

- Pinned the seven formerly floating defaults to immutable shared-cache
  snapshots and verified each snapshot contains config, tokenizer, and
  safetensor assets:
  - Llama 3.2 1B `9213176726f574b556790deb65791e0c5aa438b6`;
  - Llama 3.2 3B `0cb88a4f764b7a12671c53f0838cd831a0843b95`;
  - Llama 3.3 70B `6f6073b423013f6a7d4d9f39144961bfbfbc386b`;
  - Llama 3.1 8B `0e9e39f249a16976918f6564b8830bc894c89659`;
  - Mistral 7B `c170c708c41dac9275d15a8fff4eca08d52bab71`;
  - Qwen2.5 7B `a09a35458c702b33eeacc393d103063234e8bc28`;
  - Qwen2 7B `f2826a00ceef68f0f2b946d945ecc0477ce4450c`.
- Default IDs consistently pass the resolved revision to config/model/tokenizer,
  cache identity, and preflight. Custom IDs remain unpinned unless callers
  explicitly supply a revision; no cross-model revision inheritance occurs.
- Llama 3.1 8B now has an explicit default ID/revision. Manifests, all seven
  READMEs, both support matrices, extractors, and public API policy were updated.
- Mocked-HF/adaptor/docs validation reports 123 passed; support docs, model
  extractor, 1,760-row API policy, and static boundary checks pass. No weight
  load, network request, or hardware ran.

## 2026-09-02T13:17:21Z — Milestone 7F: implicit cache namespace versioned

- Implicit cache paths now have the stable form
  `<root>/<HF owner>/<model>/identity-1-<CacheIdentity SHA256>/<topology>`;
  topology remains the final component expected by model code.
- The digest covers package and TTNN versions, HF ID and resolved revision,
  conversion schema, architecture/topology, dtype, layout, and sharding inputs.
- Explicit `cache_dir` remains exact. `TT_CACHE_PATH` remains exact, including
  Llama 3.1 8B's device suffix, for established hardware warm-cache
  compatibility. Preflight labels these overrides
  `identity_applied_to_path=false` with a release warning rather than falsely
  describing them as versioned.
- Sidecars remain deliberately deferred; no established cache was read,
  modified, migrated, or rematerialized.
- All twelve adaptors pass one `CacheResolution` into preflight. The cache suite
  passes 38 tests; the combined focused host set passes 157; model extraction,
  static boundary, and compilation checks pass.

## 2026-09-02T13:53:29Z — Milestone 4/5/7 host-side freeze complete

- Executable examples now have zero forced `trust_remote_code=True`, zero
  CWD-relative cache defaults, and zero unscoped TTNN default-device calls.
  Eleven demo cache helpers use the standalone resolver and twelve Qwen smoke
  runners use exact-restore device scopes; the 34-file policy audit is
  idempotent.
- Final taxonomy covers 1,430 test functions: 1,176 host and 254 device, with
  zero classification errors.
- Installed the final wheel non-editably in both qualified test environments
  and ran the complete repository host selector **without `PYTHONPATH`**.
  CPython 3.10.19 and 3.12.13 each report **2,115 passed, 28 intentional
  skips, 6,791 deselected, 5 warnings, 81 subtests passed, and zero
  failures/errors**. Python 3.12 retains the documented non-failing nanobind
  shutdown diagnostic.
- Rebuilt final artifacts after checkpoint, cache, remote-code, serializer, and
  metadata changes. Wheel: 596,432 bytes, SHA256
  `7f9eec81ed4bb6d7ad0b038fa75a96255f1a300cdd9b30e9ac90182367014862`;
  sdist: 508,964 bytes, SHA256
  `d5a27e5c513f9d8590ba3d084bb3e4bc667cf717afddde1d38234809e2d3300c`.
  Archive/source identity audit, both 76-surface installed probes, and both
  `pip check` runs pass.
- Added exact host version constraints: Python 3.10 has 63 packages (SHA256
  `b18a42fa787a81df5cca8bc4a6a56953016f14d4a654cb2d56e079d120f1ec44`)
  and Python 3.12 has 60 (SHA256
  `f1aeb029b428de852ff3e10fef19f75e9c9b8adf76ea21fb2bc7c843ee7fe7b5`).
  Artifact-hash locks remain an explicit release gate.
- Consolidated the overlapping CI files into one exact two-Python workflow that
  installs constrained dependencies, validates all policy/evidence, builds and
  audits artifacts, installs the wheel non-editably, runs the full host selector,
  and probes a second base-only environment. Device CI remains static-only.
- Final deterministic audits pass: 365 unique source provenance rows, 26
  MoE-only exclusions, zero missing destinations, all source blobs exact; 1,760
  public API rows; 160 TTNN paths/14 unstable paths; twelve support manifests;
  42 serialized hardware nodes; and all extraction/import/compatibility checks.
- Release-readiness currently validates 79 criteria as 43 pass, 11 partial,
  and 25 blocked. All twelve models remain experimental until same-SHA hardware
  evidence exists. Remaining material gates are the release commit/tag and
  synchronized hardware runs, clean-device example flow, external consumer
  cutover, explicit-cache override evidence, artifact-hash locks, quality-tool
  baseline, and observed CI execution.

## 2026-09-02T13:54:29Z — Mandatory synchronized-checkout gate: paused

- Local destination identity is branch `main`, SHA
  `48959bd9167e6faa4466fc7b8f5c711a601b7615`, upstream `origin/main`, divergence
  `0 0` before committing the migration.
- The checkout is not a hardware-testable release candidate: tracked
  `README.md` is modified and 531 non-ignored files are untracked. The
  pre-existing user-owned `TTTV2_MIGRATION_PLAN.md` is among those untracked
  files; all migration implementation/evidence must receive an intentional
  commit decision rather than being copied ad hoc.
- Testing is stopped under the hardware manual's mandatory participating-
  checkout gate. No reservation table was read, no remote checkout was
  inspected or changed, no code was copied/pulled/reset, and no TT hardware
  process or reset was run.
- Required decision: choose whether to create and push a reviewed migration
  branch/commit (and whether the pre-existing plan document belongs in it).
  Only after that decision can the repository be cloned on participating WH/BH
  hosts, the complete same-branch/SHA/clean/upstream gate be run, and the
  42-node hardware matrix begin serially.

## 2026-09-02T14:35:06Z — Milestone 7G: strict locks, reproducible artifacts, and quality baseline

- Closed the base/host artifact-lock gap with four wheel-only, hash-complete
  locks. Strict fresh installs and `pip check` pass for Python 3.10 base (31
  packages, lock SHA256 `224ed4d1...00f08`), Python 3.10 host (64,
  `7808a399...d635e`), Python 3.12 base (29, `74654b92...a4a8`), and Python
  3.12 host (61, `ce2e5593...e3af`). Every wheel has HTTPS URL, selected-index,
  filename, and SHA256 evidence; Torch is exactly `2.11.0+cpu`.
- Consolidated host CI now consumes the hash locks, validates dependency and
  static-quality evidence, builds/audits in an isolated tool environment,
  runs Twine strict checking, installs the wheel non-editably, executes the
  complete host selector, and probes a separate hash-locked base environment.
  Workflow YAML and all seven embedded shell blocks parse successfully.
- A clean Twine 7.0.0 / readme-renderer 46.0 environment passes `twine check
  --strict` for both artifacts, superseding the image's broken Twine 4.0.2.
- Established the explicit pre-Phase9 quality debt at the current 132-file
  source digest: Ruff 0.11.0 finds 2,148 issues (626 fixable), Ruff format would
  change 163 files, and mypy 1.15.0 with the qualified dependency interpreter
  finds 489 errors across 99 files. A bounded Black 25.1.0 package scan did not
  finish within 30 seconds. No formatter/refactor was applied before hardware
  parity, and none of these gates is mislabeled green.
- Made release artifacts byte-reproducible. Builds use
  `SOURCE_DATE_EPOCH=1788213269`; a narrow sdist normalizer fixes staging-tree
  ownership/order/mtime and gzip metadata. Two independent builds match:
  wheel SHA256 `a8837a3803930b97d527ef569190ec69b9dddc7b0c7537d183c1ad1268a0ae51`
  (596,432 bytes) and sdist SHA256
  `4f62cb8df64da31af6076d86bbb4fb7528de4ae8ff5eddeab0f14170e40dc407`
  (497,211 bytes). Archive and Twine checks pass after normalization.
- Final readiness improves to 45 pass, 10 partial, and 24 blocked criteria.
  Remaining blocks are genuinely external/post-parity: release commit/tag and
  observed CI, same-SHA hardware/model evidence and clean-device flow, explicit
  cache-override evidence, auxiliary-extra/build locks, Phase9 quality cleanup,
  and 209 downstream cutover rows.

## 2026-09-02T14:36:41Z — Synchronized-checkout gate re-audit: still paused

- After exhausting remaining host-safe work, the local identity remains
  `main@48959bd9167e6faa4466fc7b8f5c711a601b7615`, tracking `origin/main` at
  divergence `0 0` before the migration commit.
- `README.md` is modified and 545 non-ignored files remain untracked, including
  the pre-existing user-owned migration plan. The new count reflects the
  completed lock, CI, static-quality, and reproducible-build evidence.
- The mandatory hardware gate still cannot establish a common full destination
  SHA. Reservation reads, SSH, remote clones/copies, device tests, and resets
  remain intentionally unperformed pending the user's commit/push decision.

## 2026-09-02T15:14:00Z — Final host-safe checkpoint: external decision required

- Expanded strict dependency evidence to eight hash-complete locks: base,
  host, qualification, and build/dev for both Python 3.10 and 3.12. All eight
  pass clean `--require-hashes` installation, exact-freeze comparison, and
  `pip check`; base/host select Torch `2.11.0+cpu`, auxiliary locks prove Torch
  absent, and every environment proves `tt-transformers` absent before wheel
  installation. Both base locks supply hashed PyYAML 6.0.3.
- CI now consumes the matching host/base/build-dev locks. A clean Twine 7.0.0
  strict check passes both release artifacts.
- Set `SOURCE_DATE_EPOCH=1788213269` and added safe sdist normalization. Two
  independent builds are byte-identical: wheel SHA256
  `a8837a3803930b97d527ef569190ec69b9dddc7b0c7537d183c1ad1268a0ae51`
  and sdist SHA256
  `4f62cb8df64da31af6076d86bbb4fb7528de4ae8ff5eddeab0f14170e40dc407`.
- Every final local validator passes. Readiness remains 45 pass, 10 partial,
  and 24 blocked because all twelve models deliberately remain experimental
  and no destination-SHA hardware evidence exists.
- The same synchronization blocker has now persisted through the original turn
  and two goal continuations. The checkout remains
  `main@48959bd9167e6faa4466fc7b8f5c711a601b7615`, upstream divergence `0 0`,
  with modified `README.md` and 549 non-ignored untracked files including the
  pre-existing user migration plan.
- All remaining progress requires external state/authority: approve the
  release-candidate branch/commit/push, then observe CI and synchronize remote
  WH/BH checkouts; only afterward can device/cache/model evidence and downstream
  consumer cutover proceed. No reservation, SSH, hardware, reset, or external
  repository mutation has occurred.

## 2026-09-02T15:35:19Z — Release-candidate commit authorized

- The user authorized creating and pushing the migration branch while
  explicitly excluding the pre-existing `TTTV2_MIGRATION_PLAN.md` file.
- Created branch `tttv2-standalone-migration` from
  `48959bd9167e6faa4466fc7b8f5c711a601b7615`.
- Staged 549 migration files. `git diff --cached --check` passed and the plan
  file has no staged entry; it remains preserved as an untracked user file.
- Next: commit and push this exact staged candidate, then begin the reservation
  and participating-checkout same-SHA gate before any TT process.

## 2026-09-02T15:39:06Z — Release-candidate branch published

- Committed 549 migration files as
  `02290fb` (`Migrate TTTv2 to standalone package`); the migration plan remained
  untracked and was not included.
- The initial HTTPS push failed without remote mutation because the unavailable
  VS Code credential socket could not authenticate. Verified GitHub CLI access,
  repository identity, write permission, and SSH authentication, then changed
  this checkout's `origin` URL to the authenticated SSH form.
- Pushed `tttv2-standalone-migration` to
  `git@github.com:tenstorrent/tt_transformers.git` and established upstream
  tracking. Next: commit this log checkpoint, push it, then use the resulting
  common full SHA for all local/remote synchronization and hardware evidence.

## 2026-09-02T22:58:00Z — Final-SHA hardware, host, and artifact qualification

- Published frozen hardware code SHA
  `ba7abefba4484689c953ac53fe8810322db1d184` after closing four
  hardware-discovered migration defects: stale Llama helper/selector wiring,
  cross-model cache-root collision, omitted cache-counter test support, and
  Qwen2.5-Coder internal-KV ownership. The Llama 3.3 executor's intended
  batched-prefill policy and all fixes are reproduced by pinned extractors.
- Executed all 34 matrix nodes supported by the reserved WH T3K and BH
  P150_X4 hosts. Exact-SHA results are 33 pass and one functional blocker:
  reusable modules 23/23, model smoke 3/3, end-to-end/token accuracy 7/7, and
  W6 runtime trace/order 0/1. Eight single-P150 nodes remain explicitly
  deferred because their required `bh-lb-11` host was unavailable. There were
  zero hardware-lifecycle failures and zero resets.
- Preserved the W6 failure rather than weakening its unvalidated thresholds:
  all four orders report row-0 max-abs 1.5 against 1.0 and top-5 overlap 3
  against 4. The runner now correctly distinguishes this functional result
  from lifecycle failures and rejects skip-only pytest runs as passes.
- Added and validated a canonical hardware-evidence schema/index with one
  SHA-256-bound JSON/log pair per executed node under
  `qualification/evidence/hardware/ba7abefba4484689c953ac53fe8810322db1d184`.
- Rebuilt the final package twice with `SOURCE_DATE_EPOCH=1788213269`; both
  runs are byte-identical. Wheel SHA256 is
  `18adf91d873ec27501909ab4fc26f1efb6058b9d4a1cd6c3e120f27e1285813c`
  (596,738 bytes), and normalized sdist SHA256 is
  `e26ca18f49d54caeff87f430e4ade3ef7bca8b5f1b0984b46082a80d8d74965f`
  (497,470 bytes). Archive policy, two base-only 76-surface probes, dependency
  checks, and Twine strict validation pass.
- Final non-editable-wheel host suites pass identically on Python 3.10.19 and
  3.12.13: 2,153 passed, 28 intentional skips, 6,791 deselected, 5 warnings,
  and 81 passing subtests, with the known Python 3.12 nanobind shutdown
  diagnostic retained. Taxonomy covers 1,452 functions: 1,198 host and 254
  device, with zero audit errors.
- Static debt remains explicit and non-green after the new qualification
  code: Ruff 2,158 findings (629 fixable), Ruff format 166 files, mypy 502
  errors across 99 package files, and bounded Black timeout.
- Release readiness is now 45 pass, 18 partial, and 16 blocked. Five model
  rows have partial exact-SHA evidence, while every support manifest remains
  experimental; W6, single-P150 coverage, observed CI, release tagging,
  complete model contracts, and downstream cutover remain open.
