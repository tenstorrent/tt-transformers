# Release-readiness audit

Audit date: 2026-09-03 (UTC)

Source baseline: `tt-metal` `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`
Destination candidate: `tt-transformers==0.1.0.dev0` at pushed hardware commit `73d414f8b826a7da982df8c8229d4ac41ed8ba33`

## Verdict

**Not release ready.** The standalone package builds, installs non-editably, imports, and passes its complete host suite on both supported Pythons against `ttnn==0.77.0`. At the pushed hardware candidate SHA, all 42 matrix nodes pass, including the strict T3K W6 runtime trace/order node and all eight logical single-P150 nodes on `bh-qb-05`. A separate noncanonical four-cell T3K diagnostic recorded against earlier candidate `2883a949860d749adc2ed1af5525b27a9a547505` passes every TTFT ceiling but misses every throughput floor; it was not rerun at the current hardware SHA. All model manifests remain experimental, while broader model contracts, external consumer cutover, final release tagging, hardware CI, explicit-cache qualification, and non-green quality cleanup also remain open.

The machine-readable checklist contains 79 criteria:

| Status | Count | Meaning in this audit |
|---|---:|---|
| pass | 45 | Exact current criterion is supported by pinned static, isolated-host, or artifact evidence |
| partial | 19 | A useful subset, including exact-SHA hardware coverage, passes, but the completion criterion is not fully met |
| blocked | 15 | Hardware CI, external-consumer, model-contract, or prerequisite work prevents closure |

Machine-readable authority: [`release-readiness-checklist.csv`](release-readiness-checklist.csv). Validate it with:

```bash
python -B qualification/tools/validate_release_readiness.py
```

Current validator result:

```text
Validated 79 release-readiness criteria: 45 pass, 19 partial, 15 blocked; 12/12 models remain experimental and not release-qualified.
```

This audit includes ingested evidence from remote Wormhole and Blackhole execution at the exact pushed candidate SHA. Each retained result is attributable to its physical inventory, selector, log, metrics, exit classification, and teardown record. No run reported a hardware-lifecycle failure or reset. A source declaration, collected node, skip, symbol-presence probe, or result from another SHA is still not hardware evidence.

## What is complete

- The provenance inventory classifies 366 pinned rows. All 26 exclusions are under `models/common/modules/moe`; there is no second excluded subsystem.
- All twelve concrete model packages, reusable Python modules, runtime components, and shared executors are installed under `tt_transformers`.
- Source and installed-wheel import-boundary checks pass. The base wheel imports without `transformers`, `tqdm`, `pytest`, a legacy `models.*` namespace, or a `tt-metal` checkout.
- The audited wheel rebuilt from package-source candidate `2883a949860d749adc2ed1af5525b27a9a547505` is 597,778 bytes with SHA-256 `8c55fac0a764fb9ae4d6ca514062ef2cb6cfe3097306a2f877bcad41269c50c2`; the normalized 498,499-byte sdist is `f161e13dedc5ce076d9553b677f0a1a4785996f932316f2325de9217376da644`. Two `SOURCE_DATE_EPOCH` builds are byte-identical, and the 132-file source digest is `7b03baf498e2a2252759d89813fcb898dd88daf573fb46f0e77e5c2cc6abad97`. The later hardware candidate changes qualification authority, not package source.
- Fresh non-editable CPython 3.10 and 3.12 wheel installs pass the 76-surface import probe and `pip check` with `ttnn==0.77.0`, Torch 2.11.0, and Loguru 0.6.0.
- All 160 currently referenced TTNN module paths, including 13 experimental and one private path, exist in the published 0.77 wheels for both interpreters.
- The exact full host command passes from the non-editable wheel, without `PYTHONPATH`, on both CPython 3.10 and 3.12: 2,170 passed, 28 intentional skips, 6,791 deselected, 5 warnings, 81 subtests, zero failures/errors on each interpreter.
- CPython 3.12 emits a shutdown-only TTNN/nanobind diagnostic after pytest succeeds: 10 leaked instances, 36 leaked types, and 330 leaked functions. It remains actionable TTNN binding feedback, not a test failure and not a cleanup qualification pass.
- All 1,465 test functions have an explicit lane: 1,211 host and 254 device; 393 are concrete model surfaces. The current taxonomy audit has zero errors.
- Five retained runtime/module/sampling documents now have verified pinned-source destinations and standalone path rewrites; every non-excluded provenance assignment resolves.
- The Qwen3 configuration required by host characterization is retained as qualification-only pinned blob `12ea4a36c6ac093af8d8dbc3bd435ae8b67067d6`; production does not load the repository asset.
- All twelve example READMEs and `support.json` files validate. Every model is honestly marked `experimental`, with null validation date/SHA and an empty evidence array.
- All twelve examples pin immutable 40-character HF revisions. Executable examples have zero forced-remote-code, CWD-relative cache, or unscoped default-device sites.
- Typed cache/offline policy and scoped default-device ownership are implemented and host-tested across all twelve adaptors.
- Implicit cache defaults are identity-versioned; explicit `cache_dir` and `TT_CACHE_PATH` overrides retain their established unversioned paths and emit release warnings.
- Eight binary dependency locks are hash-complete and pass clean strict installs plus `pip check`: base 31/29, host 64/61, qualification 20/19, and build-dev 44/40 packages on Python 3.10/3.12. Torch is exactly `2.11.0+cpu` in base/host, absent from auxiliary locks, and `tt-transformers` is excluded. Both base locks prove hashed PyYAML 6.0.3.
- Clean `twine==7.0.0` with `readme-renderer==46.0` passes `twine check --strict` for both final artifacts with exit zero.
- Static-quality debt is measured rather than called green: 2,158 Ruff findings (628 legacy CLI-comparable fixable; JSON classifies 505 safe and 154 unsafe fix applications), 167 files needing Ruff format with 204 already formatted, 489 mypy errors across 99 package files, and a bounded Black timeout.
- Final-SHA device execution covers the complete 42-node matrix, and all 42 pass: modules 30/30, runtime 1/1, smoke 3/3, and end-to-end 8/8. The exact split is 23 Wormhole and 19 Blackhole results, including P150 8/8 and P150x4 11/11. The canonical index SHA-256 is `4e98b62c34fb9f8744e00624c091dc9de18b3f32c5cd74f2ad2be1ad26274d7a`.
- A separate noncanonical four-cell T3K accuracy/TTFT diagnostic at earlier candidate `2883a949860d749adc2ed1af5525b27a9a547505` passes latency 4/4 but reports throughput performance-floor failures 4/4. It was not rerun at hardware candidate `73d414f8b826a7da982df8c8229d4ac41ed8ba33` and is not counted among the 42 canonical hardware nodes.
- Passing coverage spans Wormhole N150, N300, and T3K plus logical single-P150 and P150x4 execution on a physical four-board P150_X4 Blackhole quietbox. Six model rows now have partial hardware evidence, without changing any experimental manifest.
- All retained processes exited; no result is classified as a hardware-lifecycle failure and no hardware reset was performed.
- The deterministic TTNN verdict remains partial: host compatibility and the complete canonical hardware matrix pass, but the earlier-candidate throughput-floor failures and incomplete broader model contracts prevent a complete release hardware/performance verdict.

## Release blockers

| Blocker | Exact current evidence | Owner | Next gate |
|---|---|---|---|
| T3K throughput floors fail | The noncanonical accuracy/TTFT diagnostic recorded at earlier candidate `2883a949860d749adc2ed1af5525b27a9a547505` passes latency in all four cells (`86.7`–`87.2` ms against a `105.0` ms ceiling), but all four miss their adjusted per-user throughput floors; measured aggregate throughput is `245.4`–`389.2` tok/s. It was not rerun at the current hardware candidate. | Runtime performance owner | Profile the host and on-device-topk paths, resolve or formally rebaseline the throughput targets, then rerun the four cells. |
| Python 3.12 TTNN binding shutdown leak | The full suite exits zero, then nanobind reports 10 instances, 36 types, and 330 functions leaked. | TTNN binding owner | Reduce the retained diagnostic to a minimal TTNN import/config-object reproducer and resolve reference ownership. |
| Consolidated CI is unobserved and static-quality cleanup is deferred | One `host.yml` consumes strict host, base, and build-dev hash locks, validates dependency and quality baselines, builds/audits/Twine-checks, non-editably installs, runs full host, and probes base-only imports. It has no observed required result; the explicit Ruff/format/mypy/Black baseline remains non-green. | CI and static-quality owners | Observe and require the workflow, then reduce the measured quality baseline during Phase 9 without masking debt. |
| Hardware CI is policy only | Marker selectors, reservation policy, host-scoped serialization, and manual final-SHA results exist, but no reservation-aware hardware workflow or observed scheduled result exists. | Hardware CI owner | Implement scheduled/release execution and ingest exact-SHA non-skipped evidence. |
| Explicit cache overrides remain unversioned | Implicit defaults include the complete identity digest; compatibility `cache_dir` and `TT_CACHE_PATH` overrides preserve their established paths, report `identity_applied_to_path=false`, and warn. | Cache owner | Qualify explicit cold/warm caches and bind every override path to its exact preflight identity in evidence. |
| Clean-machine release flow is incomplete | Isolated installation/import and final-SHA device output are proven, but no example is promoted and the complete flow has not run from an approved non-development release artifact through validation and cleanup. | Release qualification | Complete the plan's release artifact → checkpoint → qualified example → validated output → cleanup flow. |
| No final release identity | The hardware candidate is committed and pushed at `73d414f8b826a7da982df8c8229d4ac41ed8ba33`, with exact-SHA canonical device evidence; audited artifacts remain tied to unchanged package source at `2883a949860d749adc2ed1af5525b27a9a547505`. The package remains `0.1.0.dev0`; no approved release tag exists; support identities remain null. | Release owner | Choose an approved non-development version/tag, rebuild, and regenerate artifact and support identities against it. |
| Consumer cutover has not started | All 209 mapped downstream rows remain `not_cut_over`: 64 imports, 107 CI/test paths, 9 duplicated qualification assets, 12 vLLM registrations, 15 selector/error sites, and 2 missing plugin contracts. | Consumer migration owners | Follow the staged vLLM → tt-metal/tt-train/Emule → CI/qualification → selector → shim/removal sequence. |
| Single-source and split-brain gates fail | vLLM registrations and tt-metal consumers still use `models.common`; no common released package version or source/destination hardware comparison exists. | Release and consumer owners | Pass registry/import-origin/version equality/zero-reference checks before disabling the in-tree implementation. |

## Phase assessment

| Phase | Status | Current conclusion | Next gate |
|---|---|---|---|
| 0 — freeze and characterize | pass | Scope, symbols, dependencies, tests, hardware intent, assumptions, and exclusions are characterized at the pinned source SHA. | Re-run deterministic provenance checks at release freeze. |
| 1 — scaffold | partial | Metadata, package layout, all eight hash-complete locks, current wheel matrix, and one consolidated CI workflow exist; host/base/build-dev locks are consumed, but the workflow is unobserved and static-quality cleanup is deferred. | Observe the workflow and reduce the explicit quality baseline in Phase 9. |
| 2 — extraction | pass | Python and support products are extracted and syntax-valid; all 366 provenance rows resolve through a destination or MoE-only exclusion. | Re-run all extractors and the five-document verifier at release freeze. |
| 3 — boundary closure | partial | Import, optional-dependency, cache, device ownership, and TTTv1 bridge policies pass statically and on host; all 42 final-SHA device nodes pass, including strict W6 and all eight logical single-P150 nodes, while the earlier-candidate performance floors remain open. | Resolve the throughput floor and complete the broader model contracts. |
| 4 — TTNN 0.77 qualification | partial | Both wheel matrices, all symbols, and full host semantics pass on both Pythons. Final-SHA modules are 30/30, runtime 1/1, smoke 3/3, and end-to-end 8/8; the four diagnostic throughput floors from the earlier candidate still fail. | Triage the binding leak and throughput gap, then extend the verdict across complete model contracts. |
| 5 — test/CI pyramid | partial | Taxonomy, two-Python host evidence, and the complete 42-node manual WH/BH matrix pass; earlier-candidate throughput diagnostics fail 4/4 and consolidated CI plus scheduled/release hardware workflows remain unobserved. | Resolve throughput, require the host workflow, and implement reservation-aware hardware CI. |
| 6 — docs/examples | partial | All twelve experimental contracts validate; six model rows have partial exact-SHA hardware evidence, but there are still zero qualified examples. | Complete each intended support contract before evidence-backed manifest promotion. |
| 7 — packaging/release | partial | The development candidate is committed, pushed, and hardware-evidenced at one SHA; current artifacts, locks, installs, and Twine checks pass. A non-development tag, regenerated artifact/support identities, explicit-cache evidence, observed CI, and clean release flow remain. | Choose an approved release identity, rebuild from it, and run the complete clean-machine qualification flow. |
| 8 — consumer cutover | blocked | The inventory and one-way design are ready, but every external row remains not cut over. | Add downstream gates and migrate consumers in documented order. |
| 9 — post-extraction cleanup | partial/deferred | Hardware matrix parity now passes and a precise failing static-quality baseline exists, but cleanup remains unexecuted while release qualification is incomplete: Ruff 2,158, Ruff format 167, mypy 489/99, and bounded Black timeout. | Reduce reviewed categories incrementally and make them zero-regression gates. |

## Twelve-model readiness

Every package is concrete, imports from the installed wheel, participates in the clean host suite, and has a runnable CLI. Six have partial destination-SHA model evidence, but all twelve manifests remain `experimental` with empty evidence arrays, so none is release-qualified.

| Model | HF revision | Candidate geometry summary | Release blockers |
|---|---|---|---|
| DeepSeek R1 Distill Qwen 14B | `1df8507178afcc1bef68cd8c393f61a886323761` | WH N300 TP2; T3K TP8/TP4/TP2 with declared DP | Complete N300/T3K correctness, cache, and teardown matrix |
| Llama 3.2 1B | `9213176726f574b556790deb65791e0c5aa438b6` | N150 and N300 token accuracy pass; T3K remains | Complete T3K and remaining cache/cleanup gates |
| Llama 3.2 3B | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | WH N150/N300/T3K; TP1/2/8 and declared DP | Execute all declared geometries |
| Llama 3.3 70B | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | T3K token accuracy and physical P150_X4 smoke/token accuracy pass | Review complete contract, cache/cleanup, and release identity before promotion |
| Llama 3.1 8B | `0e9e39f249a16976918f6564b8830bc894c89659` | Logical single-P150 token accuracy passes; broader WH N150/N300/T3K and BH P150/P150_X4/P300 declarations remain | Execute the broader TP/DP, accuracy, performance, cache, and cleanup matrix |
| Mistral 7B | `c170c708c41dac9275d15a8fff4eca08d52bab71` | WH N150/N300/T3K with single-device lanes | Execute N150/N300/T3K gates |
| Phi-4 | `187ef0342fff0eb3333be9f00389385e95ef0b61` | WH N300 TP2; T3K TP2/DP4 | Execute correctness and cleanup gates |
| Qwen2.5 72B | `495f39366efef23836d0cfae4fbe635880d2be31` | WH T3K TP8 | Execute T3K gate |
| Qwen2.5 7B | `a09a35458c702b33eeacc393d103063234e8bc28` | N300 token accuracy passes; T3K remains | Complete T3K and remaining cache/cleanup gates |
| Qwen2.5 Coder 32B | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` | T3K one-layer smoke passes | Add and pass token-accuracy/end-to-end plus cache/teardown gates |
| Qwen2 7B | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | WH N300 TP2; T3K TP2/DP4 | Execute N300/T3K gates |
| Qwen3 32B | `9216db5781bf21249d130ec9da846c4624c16137` | T3K token accuracy and physical P150_X4 smoke/token accuracy pass | Review complete contract, cache/cleanup, and release identity before promotion |

The three Blackhole capability contracts remain pre-acceptance requirements rather than results by themselves. Separate final-SHA physical P150_X4 evidence supports partial Llama 3.3 70B and Qwen3 32B rows, while final-SHA logical single-P150 token accuracy supports a partial Llama 3.1 8B row. These results do not promote any experimental manifest or establish a complete model support contract.

## Overall completion criteria

| Plan criterion | Status | Reason |
|---|---|---|
| All agreed code/tests except MoE | pass | All 366 rows resolve through an existing destination or one of 26 MoE-only exclusions. |
| Wheel install without tt-metal | pass | Proven on CPython 3.10 and 3.12 from site-packages. |
| Production import boundary | pass | Source and installed wheel checks pass. |
| Reviewed TTNN 0.77 verdict | partial | Deterministic host evidence and 42/42 final-SHA device passes exist; four throughput-floor failures from the separate earlier-candidate diagnostic and incomplete broader model contracts keep the release verdict incomplete. |
| Distinct reproducible base/example/test dependency sets | pass | Eight target-specific hash-complete base/host/qualification/build-dev locks pass strict clean installs and `pip check`; YAML is already proven in base. |
| Models installed under `tt_transformers.models` | pass | All twelve package/core imports pass from the wheel. |
| Runnable examples documented with manifests | pass | All twelve experimental docs/manifests validate. |
| Qualified examples pass declared host/hardware gates | blocked | Six models have partial attributable evidence, but all twelve manifests remain experimental and none passes its complete promoted contract. |
| Legacy examples preserved/versioned | pass | The pinned-scope audit found twelve active examples and no obsolete legacy candidate; `examples/legacy/README.md` defines the future freeze policy. |
| Current direct/vLLM behavior preserved | partial | Host characterization, strict W6, and eight end-to-end device nodes pass; vLLM server behavior, earlier-candidate throughput floors, and broader unexecuted model geometries remain open. |
| Downstream consumers use one source of truth | blocked | All 209 external rows remain `not_cut_over`. |
| HF/vLLM redesign remains separate | pass | The deferred redesign was not mixed into extraction. |

## Legacy and security policy

TTTv1-only factories, adapters, and generator-state characterizations are explicitly recorded in the public-removal and support-boundary ledgers; active production does not ship a compatibility `models.*` namespace. The pinned-scope audit in `examples/legacy/README.md` found twelve active TTTv2 examples and no obsolete/legacy example candidate. It also defines the required future freeze under `examples/legacy/<model>/` with last-supported constraints and manifests. Silent deletion remains prohibited.

The three Qwen adaptors no longer force `trust_remote_code`; the repository scan now finds zero default-true sites. This closes the Phase 7 remote-code default gate, subject to retaining the scan in release CI.

## Recommended gate order

1. Observe and require the consolidated two-Python workflow while preserving the explicit non-green quality baseline.
2. Resolve or formally rebaseline the four T3K throughput-floor failures, then rerun the noncanonical diagnostic against the approved candidate.
3. Choose an approved non-development release version/tag, rebuild artifacts from it, and regenerate artifact and support identities.
4. Promote only models/geometries with complete evidence, then execute the clean-machine example flow from the exact release wheel.
5. Add vLLM registry/dependency/server gates, migrate tt-metal and other consumers, atomically switch CI/qualification, and pass all split-brain checks before removing the duplicate implementation.

Until those gates close, the accurate release statement is: **host-compatible with TTNN 0.77.0, with the complete 42-node final-SHA hardware matrix passing and six partially evidenced models; four earlier-candidate throughput floors and broader model contracts remain open, all manifests and hardware support remain experimental and unqualified, and downstream cutover has not occurred.**
