# TTNN 0.77.0 compatibility verdict

Status: **host-compatible for the declared standalone surfaces; hardware and model correctness unqualified**

Validation date: 2026-09-02 (UTC)

Machine verdict: [`ttnn-0.77.0-compatibility-matrix.json`](ttnn-0.77.0-compatibility-matrix.json)

## Verdict

`tt-transformers==0.1.0.dev0` may retain `ttnn==0.77.0` as its host-qualified dependency on CPython 3.10 and 3.12. The final non-editable wheel imports from site-packages with the exact base tuple, all 160 currently referenced TTNN module paths exist on both interpreters, and the complete marked host suite passes with the same exact totals on both interpreters.

This verdict does **not** qualify a Tenstorrent device, firmware/driver stack, architecture, SKU, mesh, TP/DP geometry, context/batch bucket, TT kernel result, model output, accuracy, or performance. All twelve model manifests remain `experimental`, and their hardware arrays are source-declared targets rather than validated support claims. [Package evidence](package-wheel.md), [host evidence](host-ttnn-0.77.md), [API evidence](ttnn-0.77.0-api-availability.md).

## Exact software matrix

| Python | Dependency-group environment | Installed standalone base-wheel environment | Result | Evidence |
|---|---|---|---|---|
| CPython 3.10.19 | `ttnn==0.77.0`; `torch==2.11.0+cpu`; `loguru==0.6.0`; optional qualification set includes `transformers==5.12.1`, `tqdm==4.66.3`, `pytest==9.0.3` | `tt-transformers==0.1.0.dev0`; `ttnn==0.77.0`; `torch==2.11.0`; `loguru==0.6.0`; Transformers/tqdm/pytest absent and blocked | dependency resolution/import, wheel import, and `pip check` pass | [3.10 dependencies](dependencies-py310.md), [wheel audit](package-wheel.md) |
| CPython 3.12.13 | `ttnn==0.77.0`; `torch==2.11.0+cpu`; `loguru==0.6.0`; optional qualification set includes `transformers==5.12.1`, `tqdm==4.66.3`, `pytest==9.0.3` | `tt-transformers==0.1.0.dev0`; `ttnn==0.77.0`; `torch==2.11.0`; `loguru==0.6.0`; Transformers/tqdm/pytest absent and blocked | dependency resolution/import, wheel import, and `pip check` pass | [3.12 dependencies](dependencies-py312.md), [wheel audit](package-wheel.md) |
| CPython 3.10.19 host semantics | direct host-test set additionally includes `jsonschema==4.26.0` and `pytz==2026.3.post1` | final wheel installed non-editably; no `PYTHONPATH` | 2,115 passed; 28 intentional skips; 6,791 deselected; 5 warnings; 81 subtests passed; zero failures/errors; exit 0 | [full host report](host-ttnn-0.77.md), [substantive triage](ttnn-0.77.0-host-triage.md) |
| CPython 3.12.13 host semantics | direct host-test set additionally includes `jsonschema==4.26.0` and `pytz==2026.3.post1` | final wheel installed non-editably; no `PYTHONPATH`; offline controls enabled | 2,115 passed; 28 intentional skips; 6,791 deselected; 5 warnings; 81 subtests passed; zero failures/errors; exit 0; shutdown-only binding diagnostics retained below | [full host report](host-ttnn-0.77.md), [substantive triage](ttnn-0.77.0-host-triage.md) |

The `+cpu` Torch build is used by dependency and host-test qualification; the published package metadata requests the version-equivalent `torch==2.11.0`, and the base-wheel isolation probes resolved that exact distribution version. Neither distinction is hardware evidence.

The CPython 3.12 command returned exit 0. After pytest's successful summary, retained output `/tmp/gwang/tttv2-py312-host-final.log` reported `nanobind: leaked 8 instances!`, `leaked 36 types!`, and `leaked 330 functions!`, ending with a likely binding reference-counting issue. This is open TTNN 0.77 binding-lifecycle feedback, not a test failure and not a TTTv2 correctness claim for clean binding teardown. [Exact command and diagnostic context](host-ttnn-0.77.md).

Final package/source identity:

- wheel: `dist/tt_transformers-0.1.0.dev0-py3-none-any.whl`, SHA-256 `a0d5c0d9d5d1b01271402867008c7541e46028c94700060c8a78f0f3d2d99a92`;
- 132 wheel Python files are byte-identical to the source tree;
- source-tree SHA-256: `3814705a56bcf5bdd3ed88578e203d6d9e428f329289facb9bc2215366fb4e5f`.

[Artifact hashes](package-artifact-hashes.json), [per-file wheel manifest](package-wheel-manifest.json).

## TTNN API availability

| Inventory | CPython 3.10.19 | CPython 3.12.13 | Interpretation | Evidence |
|---|---:|---:|---|---|
| All current maximal module-rooted paths | 160/160 present | 160/160 present | zero missing module-level TTNN paths across 4,393 source occurrences | [API report](ttnn-0.77.0-api-availability.md), [machine inventory](ttnn-0.77.0-api-availability.json) |
| Public paths | 146/146 | 146/146 | presence only unless exercised by host tests below | [API report](ttnn-0.77.0-api-availability.md) |
| Experimental paths | 13/13 | 13/13 | symbol presence; device semantics unqualified | [API report](ttnn-0.77.0-api-availability.md), [unstable source inventory](../analysis/ttnn_unstable_uses.csv) |
| Private paths | 1/1 | 1/1 | `ttnn._ttnn.tensor.dump_tensor_flatbuffer` exists; private status and device/cache semantics remain a release risk | [API report](ttnn-0.77.0-api-availability.md), [unstable source inventory](../analysis/ttnn_unstable_uses.csv) |
| Typed dynamic members | 29 identities have descriptors; 5 complex identities un-attributed; 2 abstract descriptors resolved through official source | same | class surface/source interpretation, not instance/device behavior | [API report](ttnn-0.77.0-api-availability.md) |
| Signatures | 15 informative, 72 generic, 73 not inspectable | 15 informative, 103 generic, 42 not inspectable | 145/160 per interpreter lack complete signature evidence; 35 introspection representations differ | [API report](ttnn-0.77.0-api-availability.md) |

All 14 previously tracked unstable paths are present on both interpreters: 13 `ttnn.experimental.*` operations and the one private flatbuffer writer. This closes availability only. In particular, it does not prove collective synchronization, numerical behavior, memory layout, paged-cache mutation, traceability, device lifetime, or serialization compatibility on hardware. [Unstable cross-reference](ttnn-0.77.0-api-availability.json).

## Surface verdicts

| Surface | Host verdict | Hardware verdict | What the verdict covers | Evidence |
|---|---|---|---|---|
| Package artifact and metadata | pass on 3.10/3.12 | not applicable | wheel/sdist build, exclusions, metadata, RECORD, source byte identity, non-editable install, `pip check` | [wheel audit](package-wheel.md) |
| Root package | pass on 3.10/3.12 | not applicable | site-packages import with no checkout and optional HF/test packages absent/blocked | [wheel audit](package-wheel.md) |
| Foundation helpers | imports and host semantics pass on 3.10/3.12 | not qualified | tensor/device/mesh helpers, cache/environment preflight, program-config serialization, scoped ownership | [wheel audit](package-wheel.md), [triage](ttnn-0.77.0-host-triage.md) |
| Reusable modules | imports and host contracts pass on 3.10/3.12 | not qualified | attention, embedding, lazy owners, LM head, MLP, RMSNorm, RoPE, CCL configuration/validation | [host report](host-ttnn-0.77.md), [wheel audit](package-wheel.md) |
| Sampling foundation | imports and host contracts pass on 3.10/3.12 | not qualified | canonical params, preparation, penalties, seed/state, log-probability and cleanup ownership | [host report](host-ttnn-0.77.md), [wheel audit](package-wheel.md) |
| LLM runtime | imports and host contracts pass on 3.10/3.12 | not qualified | prefill/decode planning, program/trace identities, page/cache geometry, adapter and cleanup behavior via fakes/mocks | [host report](host-ttnn-0.77.md), [wheel audit](package-wheel.md) |
| Shared executors | imports and host contracts pass on 3.10/3.12 | not qualified | family-neutral/Llama/Qwen configuration, delegation, request and ownership contracts | [host report](host-ttnn-0.77.md), [wheel audit](package-wheel.md) |
| Twelve concrete model cores | package/core imports and profile/config contracts pass on 3.10/3.12 | not qualified | importability and host-only model geometry/profile/config construction; no weights or outputs | [host report](host-ttnn-0.77.md), [wheel audit](package-wheel.md), [triage](ttnn-0.77.0-host-triage.md) |

## Failure classification and fixes

| Primary class | Finding | Resolution/verdict | Evidence |
|---|---|---|---|
| Missing public TTNN API | none among 146 current public module paths | 0 missing on both Pythons | [API report](ttnn-0.77.0-api-availability.md) |
| Missing experimental/private TTNN API | none among 14 tracked unstable paths | 0 missing on both Pythons; semantics remain unqualified | [API report](ttnn-0.77.0-api-availability.md) |
| Semantic/signature difference | `num_workers_per_dram_bank` is absent from the 0.77 DRAM-sharded matmul config constructor | obsolete test expectation removed; production never used the later parameter | [host triage](ttnn-0.77.0-host-triage.md) |
| Semantic/signature evidence gap | nanobind introspection is generic/unavailable for 145 paths; 35 cross-Python representations differ; two abstract descriptors map to concrete config fields | recorded, not treated as missing and not hidden by vendoring | [API report](ttnn-0.77.0-api-availability.md) |
| TTTv2 packaging defect | missing pinned `_nearest_32` alias used by eleven model cores | alias restored; cross-module audit and both wheel imports pass | [wheel audit](package-wheel.md) |
| TTTv2 packaging defect | guarded YAML serializer lacked a direct optional dependency declaration | `yaml` extra added; base dependency set unchanged | [wheel audit](package-wheel.md) |
| TTTv2 implementation defect | repr-only fallback failed for 0.77 `SDPAProgramConfig` | serialize supported public fields recursively; focused tests and installed probes pass | [host triage](ttnn-0.77.0-host-triage.md), [wheel audit](package-wheel.md) |
| Obsolete characterization | old namespaces/asset paths and retired TTTv1 factories/generators | paths migrated or tests intentionally retired; final host suite clean | [host report](host-ttnn-0.77.md) |
| TTNN binding-lifecycle feedback | CPython 3.12 shutdown reports 8 leaked instances, 36 leaked types and 330 leaked functions | open upstream binding feedback; pytest completed with exit 0, so this is not hidden and not classified as a test failure | [host report](host-ttnn-0.77.md), [host triage](ttnn-0.77.0-host-triage.md) |
| Hardware/firmware/driver mismatch | not evaluated | no classification possible without a recorded hardware run | [host report](host-ttnn-0.77.md) |
| Unsupported model geometry | not evaluated on hardware | manifest constraints remain source-declared gaps, not measured support | [support baseline](../analysis/support/support_baseline.md) |

## Host semantics exercised

| Exercised on host | Bound | Evidence |
|---|---|---|
| Dependency resolution, metadata, wheel installation, import isolation and optional laziness | both Pythons | [dependency reports](dependencies-py310.md), [wheel audit](package-wheel.md) |
| TTNN module/class/enum/descriptor presence | both Pythons | [API report](ttnn-0.77.0-api-availability.md) |
| Supported host-safe TTNN config constructors and program-config serialization | both Pythons | [host triage](ttnn-0.77.0-host-triage.md), [wheel audit](package-wheel.md) |
| Pure configuration, profile, topology, planning, signature and cache-geometry logic | both Pythons | [host report](host-ttnn-0.77.md) |
| Runtime ownership/cleanup, trace/program identity, adapters and execution selection through fakes/mocks | both Pythons | [host report](host-ttnn-0.77.md), [device ownership](../extraction/device_ownership.md) |
| Cache/environment resolution, boolean policy, redaction, version identity and topology suffixes | both Pythons | [cache policy](../extraction/cache_environment.md), [host report](host-ttnn-0.77.md) |
| All twelve model package/core imports and host model configuration/contracts | both Pythons | [wheel audit](package-wheel.md), [host report](host-ttnn-0.77.md) |

## Host semantics not exercised

| Not exercised | Consequence | Evidence |
|---|---|---|
| TT tensor kernels or numerical tensor results on a Tenstorrent device | no module numerical/correctness claim | [host report](host-ttnn-0.77.md) |
| The 13 experimental operations and private flatbuffer writer on device | availability is not semantic compatibility | [API report](ttnn-0.77.0-api-availability.md) |
| Real collective/fabric ordering, trace capture/replay, command queues, asynchronous reads | runtime hardware behavior remains open | [runtime boundary](../extraction/runtime_boundary.md), [host report](host-ttnn-0.77.md) |
| Device sampling, paged KV mutation, tensor-cache cold/warm behavior, real TT cleanup | sampling/cache/lifecycle hardware claims remain open | [cache policy](../extraction/cache_environment.md), [device ownership](../extraction/device_ownership.md) |
| HF checkpoint loading, conversion/materialization, token/text outputs, accuracy or performance | no end-to-end model claim | [wheel audit](package-wheel.md), [support baseline](../analysis/support/support_baseline.md) |
| Firmware/driver, architecture, SKU, mesh, TP/DP, batch/context and model geometry | every hardware target remains unqualified | [support baseline](../analysis/support/support_baseline.md) |

## Per-model verdicts

Every row below means: package/core imports and host-only configuration/contracts pass on CPython 3.10 and 3.12; hardware is **not qualified**. The target count describes manifest intent only.

| Model | Lifecycle / host verdict | HF revision | Source-declared targets (not validated) | Remaining gaps | Evidence |
|---|---|---|---|---|---|
| DeepSeek R1 Distill Qwen 14B | experimental; host pass | `1df8507178afcc1bef68cd8c393f61a886323761` | 4 WH targets: N300 TP2 plus T3K TP8/TP4/TP2 lanes | no pinned-revision hardware evidence; N150/TP1; N300 accuracy eval-32/batch-32-ci; undeclared geometry | [manifest](../../examples/deepseek_r1_distill_qwen_14b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.2 1B | experimental; host pass | unpinned | 6 WH targets across N150/N300/T3K, TP1/2/8 and declared DP | no pinned-revision hardware evidence; revision; TP4; N150 batch-32-ci 2048; undeclared geometry | [manifest](../../examples/llama32_1b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.2 3B | experimental; host pass | unpinned | 6 WH targets across N150/N300/T3K, TP1/2/8 and declared DP | no pinned-revision hardware evidence; revision; TP4; N150 traced prefill; undeclared geometry | [manifest](../../examples/llama32_3b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.3 70B | experimental; host pass | unpinned | 2 targets: WH T3K TP8 and BH logical P150_X4 TP4 | no pinned-revision hardware evidence; revision; DP>1; P150x4 orientation; 32K context; undeclared geometry | [manifest](../../examples/llama33_70b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.1 8B | experimental; host pass | unpinned | 11 WH/BH targets across N150/N300/T3K/P150/P150_X4/physical-P300 declarations | no pinned-revision hardware evidence; revision; removed TTTv1 prefetcher; P100/128K P300; stand-in provenance; undeclared geometry | [manifest](../../examples/llama3_8b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Mistral 7B | experimental; host pass | unpinned | 5 WH targets across N150/N300/T3K and one-device DP lanes | no pinned-revision hardware evidence; revision; other DP layouts; undeclared geometry | [manifest](../../examples/mistral_7b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Phi-4 | experimental; host pass | `187ef0342fff0eb3333be9f00389385e95ef0b61` | 2 WH targets: N300 TP2 and T3K DP4/TP2 | no pinned-revision hardware evidence; N150; ordinary T3K/TG TP; other DP; undeclared geometry | [manifest](../../examples/phi4/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen 2.5 72B | experimental; host pass | `495f39366efef23836d0cfae4fbe635880d2be31` | 1 WH target: T3K TP8 | no pinned-revision hardware evidence; non-T3K; DP>1; undeclared geometry | [manifest](../../examples/qwen25_72b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen 2.5 7B | experimental; host pass | unpinned | 2 WH targets: N300 TP2 and T3K DP4/TP2 | no pinned-revision hardware evidence; revision; N150; ordinary T3K/TG TP; other DP; undeclared geometry | [manifest](../../examples/qwen25_7b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen 2.5 Coder 32B | experimental; host pass | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` | 1 WH target: T3K TP8 | no pinned-revision hardware evidence; non-T3K; DP>1; undeclared geometry | [manifest](../../examples/qwen25_coder_32b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen2 7B | experimental; host pass | unpinned | 2 WH targets: N300 TP2 and T3K DP4/TP2 | no pinned-revision hardware evidence; revision; N150; ordinary T3K/TG TP; other DP; undeclared geometry | [manifest](../../examples/qwen2_7b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen3 32B | experimental; host pass | `9216db5781bf21249d130ec9da846c4624c16137` | 2 targets: WH T3K TP8 and BH logical P150_X4 TP4 | no pinned-revision hardware evidence; P100/P150/P300 single/two-die; DP>1; orientation; undeclared geometry | [manifest](../../examples/qwen3_32b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |

All manifest `validation.evidence` arrays are empty and validation dates/SHAs are null. No row may be promoted to `qualified` from this host result.

## Remaining hardware, firmware and geometry gates

| Gap | Required closure evidence | Current verdict | Evidence |
|---|---|---|---|
| Reusable module kernels | serialized device tests on representative Wormhole and Blackhole with numerical criteria | not evaluated | [support baseline](../analysis/support/support_baseline.md) |
| Runtime eager/trace/cache/sampling/cleanup | real-device compile/capture/replay/cache mutation/sampling/cleanup, including repeated construction | not evaluated | [runtime boundary](../extraction/runtime_boundary.md), [device ownership](../extraction/device_ownership.md) |
| Fourteen unstable/private APIs | per-operation device call/signature and semantic evidence on supported architectures | present, semantics unqualified | [API report](ttnn-0.77.0-api-availability.md) |
| Firmware/driver compatibility | versions and health/preflight recorded with each device result | not evaluated | [support baseline](../analysis/support/support_baseline.md) |
| Model geometry | correctness for every claimed SKU/mesh/TP/DP/batch/context bucket; explicit unsupported outcomes elsewhere | source-declared only | [hardware inventory](../analysis/support/hardware_coverage.csv), [host report](host-ttnn-0.77.md) |
| End-to-end model correctness/performance | pinned HF revision, cache identity, deterministic token/text or numerical thresholds, and performance measurements | not evaluated | [support baseline](../analysis/support/support_baseline.md), [wheel audit](package-wheel.md) |
| CPython 3.12 TTNN binding teardown | minimize the exit-0 nanobind diagnostic and correct reference ownership in TTNN bindings | open non-failing binding feedback | [host report](host-ttnn-0.77.md), [host triage](ttnn-0.77.0-host-triage.md) |

Hardware tests must run one TT process at a time and a skip is not evidence. Until these gates are recorded, the correct release statement is “host-compatible with `ttnn==0.77.0`; all model/hardware support experimental and unqualified.”

## Deterministic coverage gate

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python qualification/tools/validate_ttnn_077_compatibility.py
```

The validator regenerates the matrix projection in memory and requires byte-stable JSON. It verifies:

- exactly the eight base/module/runtime/model aggregate surfaces above have verdicts and existing evidence;
- source model directories, all current `examples/*/support.json` manifests, and matrix model rows are the same twelve-package set;
- every model remains experimental/hardware-unqualified with non-empty gaps and empty hardware validation evidence;
- 160 total TTNN paths and all 14 unstable paths are present on both Pythons;
- both exact host-suite records contain 2,115 passes, 28 skips, 6,791 deselections, 5 warnings, 81 passing subtests, zero failures/errors and exit 0;
- Python 3.12's 8-instance/36-type/330-function nanobind shutdown diagnostic is retained and classified as non-failing TTNN binding feedback;
- every local evidence path exists.

This validator and report do not execute TTNN callables, query devices, or make hardware claims.
