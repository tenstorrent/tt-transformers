# Release-readiness audit

Audit date: 2026-09-02 (UTC)

Source baseline: `tt-metal` `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`
Destination candidate: `tt-transformers==0.1.0.dev0`

## Verdict

**Not release ready.** The standalone package builds, installs non-editably, imports, and passes its complete host suite on both supported Pythons against `ttnn==0.77.0`, but no model or hardware geometry is qualified. External consumers still use the in-tree product, and release identity, clean-device execution, CI observation, explicit-cache evidence, and non-green quality cleanup remain open.

The machine-readable checklist contains 79 criteria:

| Status | Count | Meaning in this audit |
|---|---:|---|
| pass | 45 | Exact current criterion is supported by pinned static, isolated-host, or artifact evidence |
| partial | 10 | A useful subset passes, but the completion criterion is not fully met |
| blocked | 24 | Hardware, external-consumer, release-identity, or prerequisite work prevents closure |

Machine-readable authority: [`release-readiness-checklist.csv`](release-readiness-checklist.csv). Validate it with:

```bash
python -B qualification/tools/validate_release_readiness.py
```

Current validator result:

```text
Validated 79 release-readiness criteria: 45 pass, 10 partial, 24 blocked; 12/12 models remain experimental and not release-qualified.
```

This is a repository and host-evidence audit. It did not contact remote machines, query or reset TT hardware, load checkpoints, materialize caches, start vLLM, or execute device tests. A source declaration, collected node, skip, symbol-presence probe, or result from another SHA is not hardware evidence.

## What is complete on the host

- The provenance inventory classifies 365 pinned rows. All 26 exclusions are under `models/common/modules/moe`; there is no second excluded subsystem.
- All twelve concrete model packages, reusable Python modules, runtime components, and shared executors are installed under `tt_transformers`.
- Source and installed-wheel import-boundary checks pass. The base wheel imports without `transformers`, `tqdm`, `pytest`, a legacy `models.*` namespace, or a `tt-metal` checkout.
- The current audited wheel is SHA-256 `a8837a3803930b97d527ef569190ec69b9dddc7b0c7537d183c1ad1268a0ae51`; the normalized sdist is `4f62cb8df64da31af6076d86bbb4fb7528de4ae8ff5eddeab0f14170e40dc407`. Two `SOURCE_DATE_EPOCH` builds are byte-identical, and their recorded pyproject and 132-file source identities match the current tree.
- Fresh non-editable CPython 3.10 and 3.12 wheel installs pass the 76-surface import probe and `pip check` with `ttnn==0.77.0`, Torch 2.11.0, and Loguru 0.6.0.
- All 160 currently referenced TTNN module paths, including 13 experimental and one private path, exist in the published 0.77 wheels for both interpreters.
- The exact full host command passes from the non-editable wheel, without `PYTHONPATH`, on both CPython 3.10 and 3.12: 2,115 passed, 28 intentional skips, 6,791 deselected, 5 warnings, 81 subtests, zero failures/errors on each interpreter.
- CPython 3.12 emits a shutdown-only TTNN/nanobind diagnostic after pytest succeeds: 8 leaked instances, 36 leaked types, and 330 leaked functions. It remains actionable TTNN binding feedback, not a test failure and not a cleanup qualification pass.
- All 1,430 test functions have an explicit lane: 1,176 host and 254 device. The current taxonomy audit has zero errors.
- Five retained runtime/module/sampling documents now have verified pinned-source destinations and standalone path rewrites; every non-excluded provenance assignment resolves.
- The Qwen3 configuration required by host characterization is retained as qualification-only pinned blob `12ea4a36c6ac093af8d8dbc3bd435ae8b67067d6`; production does not load the repository asset.
- All twelve example READMEs and `support.json` files validate. Every model is honestly marked `experimental`, with null validation date/SHA and an empty evidence array.
- All twelve examples pin immutable 40-character HF revisions. Executable examples have zero forced-remote-code, CWD-relative cache, or unscoped default-device sites.
- Typed cache/offline policy and scoped default-device ownership are implemented and host-tested across all twelve adaptors.
- Implicit cache defaults are identity-versioned; explicit `cache_dir` and `TT_CACHE_PATH` overrides retain their established unversioned paths and emit release warnings.
- Eight binary dependency locks are hash-complete and pass clean strict installs plus `pip check`: base 31/29, host 64/61, qualification 20/19, and build-dev 44/40 packages on Python 3.10/3.12. Torch is exactly `2.11.0+cpu` in base/host, absent from auxiliary locks, and `tt-transformers` is excluded. Both base locks prove hashed PyYAML 6.0.3.
- Clean `twine==7.0.0` with `readme-renderer==46.0` passes `twine check --strict` for both final artifacts with exit zero.
- Static-quality debt is measured rather than called green: 2,148 Ruff findings, 163 files needing Ruff format, 489 mypy errors across 99 package files, and a bounded Black timeout.
- The deterministic TTNN verdict correctly says **host-compatible; hardware and model correctness unqualified**.

## Release blockers

| Blocker | Exact current evidence | Owner | Next gate |
|---|---|---|---|
| No attributable hardware evidence | The readiness matrix has 42 unexecuted nodes: 23 Wormhole and 19 Blackhole; 30 module, 1 runtime, 3 smoke, and 8 end-to-end. All twelve manifests have empty evidence. | Hardware qualification | Execute one process/node at a time at the exact release-candidate SHA; record physical SKU, software, metrics, teardown, and logs. |
| Python 3.12 TTNN binding shutdown leak | The full suite exits zero, then nanobind reports 8 instances, 36 types, and 330 functions leaked. | TTNN binding owner | Reduce the retained diagnostic to a minimal TTNN import/config-object reproducer and resolve reference ownership. |
| Consolidated CI is unobserved and static-quality cleanup is deferred | One `host.yml` consumes strict host, base, and build-dev hash locks, validates dependency and quality baselines, builds/audits/Twine-checks, non-editably installs, runs full host, and probes base-only imports. It has no observed required result; the explicit Ruff/format/mypy/Black baseline remains non-green. | CI and static-quality owners | Observe and require the workflow, then reduce the measured quality baseline during Phase 9 without masking debt. |
| Hardware CI is policy only | Marker selectors, reservation policy, and serialization rules exist, but no reservation-aware hardware workflow/result exists. | Hardware CI owner | Implement scheduled/release execution without parallel TT processes and ingest non-skipped evidence. |
| Explicit cache overrides remain unversioned | Implicit defaults include the complete identity digest; compatibility `cache_dir` and `TT_CACHE_PATH` overrides preserve their established paths, report `identity_applied_to_path=false`, and warn. | Cache owner | Qualify explicit cold/warm caches and bind every override path to its exact preflight identity in evidence. |
| Clean-machine release flow is incomplete | Isolated installation/import passes, but no standalone wheel has loaded a pinned checkpoint, produced validated device output, and cleaned all TT resources. | Release qualification | Complete the plan's install → checkpoint → qualified example → validated output → cleanup flow. |
| No final release identity | The package is `0.1.0.dev0`; support validation SHAs are null; the migration tree is not yet represented by an approved destination commit/tag. | Release owner | Commit/review the tree, choose the release version, rebuild, and regenerate all hashes/evidence against that identity. |
| Consumer cutover has not started | All 209 mapped downstream rows remain `not_cut_over`: 64 imports, 107 CI/test paths, 9 duplicated qualification assets, 12 vLLM registrations, 15 selector/error sites, and 2 missing plugin contracts. | Consumer migration owners | Follow the staged vLLM → tt-metal/tt-train/Emule → CI/qualification → selector → shim/removal sequence. |
| Single-source and split-brain gates fail | vLLM registrations and tt-metal consumers still use `models.common`; no common released package version or source/destination hardware comparison exists. | Release and consumer owners | Pass registry/import-origin/version equality/zero-reference checks before disabling the in-tree implementation. |

## Phase assessment

| Phase | Status | Current conclusion | Next gate |
|---|---|---|---|
| 0 — freeze and characterize | pass | Scope, symbols, dependencies, tests, hardware intent, assumptions, and exclusions are characterized at the pinned source SHA. | Re-run deterministic provenance checks at release freeze. |
| 1 — scaffold | partial | Metadata, package layout, all eight hash-complete locks, current wheel matrix, and one consolidated CI workflow exist; host/base/build-dev locks are consumed, but the workflow is unobserved and static-quality cleanup is deferred. | Observe the workflow and reduce the explicit quality baseline in Phase 9. |
| 2 — extraction | pass | Python and support products are extracted and syntax-valid; all 365 provenance rows resolve through a destination or MoE-only exclusion. | Re-run all extractors and the five-document verifier at release freeze. |
| 3 — boundary closure | partial | Import, optional-dependency, cache, device ownership, and TTTv1 bridge policies pass statically and on host. Hardware execution parity is unknown. | Run module/runtime/model device characterization. |
| 4 — TTNN 0.77 qualification | partial | Both wheel matrices, all symbols, and full host semantics pass on both Pythons. The 3.12 shutdown leak is recorded; all device semantics remain open. | Triage the binding leak, execute serialized WH/BH matrix, and extend the verdict. |
| 5 — test/CI pyramid | partial | Taxonomy and local two-Python non-editable-wheel host evidence pass; consolidated CI and scheduled/release hardware execution remain unobserved. | Require a passing host workflow and implement serialized hardware jobs. |
| 6 — docs/examples | partial | All twelve experimental contracts validate with pinned revisions and closed example policy; there are zero qualified examples. | Promote only after exact hardware evidence. |
| 7 — packaging/release | partial | Current artifacts, identities, all eight hash locks, isolated installs, and strict Twine checks pass. Explicit-cache evidence, release identity, tool-quality cleanup, unobserved CI, and clean-device flow remain. | Resolve remaining gates, tag an RC, rebuild from it, and run clean-machine qualification. |
| 8 — consumer cutover | blocked | The inventory and one-way design are ready, but every external row remains not cut over. | Add downstream gates and migrate consumers in documented order. |
| 9 — post-extraction cleanup | partial/deferred | A precise failing static-quality baseline exists, but cleanup is correctly deferred until hardware parity: Ruff 2,148, Ruff format 163, mypy 489/99, and bounded Black timeout. | After parity, reduce reviewed categories incrementally and make them zero-regression gates. |

## Twelve-model readiness

Every package is concrete, imports from the installed wheel, participates in the clean host suite, and has a runnable CLI. None has destination-SHA hardware evidence, so none is qualified.

| Model | HF revision | Candidate geometry summary | Release blockers |
|---|---|---|---|
| DeepSeek R1 Distill Qwen 14B | `1df8507178afcc1bef68cd8c393f61a886323761` | WH N300 TP2; T3K TP8/TP4/TP2 with declared DP | Complete N300/T3K correctness, cache, and teardown matrix |
| Llama 3.2 1B | `9213176726f574b556790deb65791e0c5aa438b6` | WH N150/N300/T3K; TP1/2/8 and declared DP | Execute all declared geometries |
| Llama 3.2 3B | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | WH N150/N300/T3K; TP1/2/8 and declared DP | Execute all declared geometries |
| Llama 3.3 70B | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | WH T3K TP8; BH logical P150_X4 TP4 | Qualify T3K and physical P150_X4/P300_X2 provenance |
| Llama 3.1 8B | `0e9e39f249a16976918f6564b8830bc894c89659` | Broad WH N150/N300/T3K and BH P150/P150_X4/P300 declarations | Execute the broad TP/DP, accuracy, performance, cache, and cleanup matrix |
| Mistral 7B | `c170c708c41dac9275d15a8fff4eca08d52bab71` | WH N150/N300/T3K with single-device lanes | Execute N150/N300/T3K gates |
| Phi-4 | `187ef0342fff0eb3333be9f00389385e95ef0b61` | WH N300 TP2; T3K TP2/DP4 | Execute correctness and cleanup gates |
| Qwen2.5 72B | `495f39366efef23836d0cfae4fbe635880d2be31` | WH T3K TP8 | Execute T3K gate |
| Qwen2.5 7B | `a09a35458c702b33eeacc393d103063234e8bc28` | WH N300 TP2; T3K TP2/DP4 | Execute N300/T3K gates |
| Qwen2.5 Coder 32B | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` | WH T3K TP8 | Execute T3K gate |
| Qwen2 7B | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | WH N300 TP2; T3K TP2/DP4 | Execute N300/T3K gates |
| Qwen3 32B | `9216db5781bf21249d130ec9da846c4624c16137` | WH T3K TP8; BH logical P150_X4 TP4 | Qualify T3K and physical BH gates |

The three Blackhole capability contracts are pre-acceptance requirements, not results. Their presence does not promote Llama 3.1 8B, Llama 3.3 70B, or Qwen3 32B.

## Overall completion criteria

| Plan criterion | Status | Reason |
|---|---|---|
| All agreed code/tests except MoE | pass | All 365 rows resolve through an existing destination or one of 26 MoE-only exclusions. |
| Wheel install without tt-metal | pass | Proven on CPython 3.10 and 3.12 from site-packages. |
| Production import boundary | pass | Source and installed wheel checks pass. |
| Reviewed TTNN 0.77 verdict | partial | Deterministic host verdict exists; hardware/model semantics remain unqualified. |
| Distinct reproducible base/example/test dependency sets | pass | Eight target-specific hash-complete base/host/qualification/build-dev locks pass strict clean installs and `pip check`; YAML is already proven in base. |
| Models installed under `tt_transformers.models` | pass | All twelve package/core imports pass from the wheel. |
| Runnable examples documented with manifests | pass | All twelve experimental docs/manifests validate. |
| Qualified examples pass declared host/hardware gates | blocked | There are zero qualified examples and zero attributable hardware results. |
| Legacy examples preserved/versioned | pass | The pinned-scope audit found twelve active examples and no obsolete legacy candidate; `examples/legacy/README.md` defines the future freeze policy. |
| Current direct/vLLM behavior preserved | partial | Host characterization passes; real direct generation, server, and device behavior are unrun. |
| Downstream consumers use one source of truth | blocked | All 209 external rows remain `not_cut_over`. |
| HF/vLLM redesign remains separate | pass | The deferred redesign was not mixed into extraction. |

## Legacy and security policy

TTTv1-only factories, adapters, and generator-state characterizations are explicitly recorded in the public-removal and support-boundary ledgers; active production does not ship a compatibility `models.*` namespace. The pinned-scope audit in `examples/legacy/README.md` found twelve active TTTv2 examples and no obsolete/legacy example candidate. It also defines the required future freeze under `examples/legacy/<model>/` with last-supported constraints and manifests. Silent deletion remains prohibited.

The three Qwen adaptors no longer force `trust_remote_code`; the repository scan now finds zero default-true sites. This closes the Phase 7 remote-code default gate, subject to retaining the scan in release CI.

## Recommended gate order

1. Observe and require the consolidated two-Python workflow while preserving the explicit non-green quality baseline.
2. Create an approved destination release-candidate commit/version, rebuild artifacts from it, and regenerate hashes and manifest identities.
3. Run the 42-node hardware matrix serially. Include explicit-cache cold/warm evidence, preflight, correctness, accuracy/performance where declared, repeated construction, and teardown.
4. Promote only the models/geometries that have complete evidence, then execute the clean-machine example flow from the exact wheel.
5. Add vLLM registry/dependency/server gates, migrate tt-metal and other consumers, atomically switch CI/qualification, and pass all split-brain checks before removing the duplicate implementation.

Until those gates close, the accurate release statement is: **host-compatible with TTNN 0.77.0; all models and hardware support remain experimental and unqualified; downstream cutover has not occurred.**
