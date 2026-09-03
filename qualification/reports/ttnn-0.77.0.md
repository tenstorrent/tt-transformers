# TTNN 0.77.0 compatibility verdict

Status: **host-compatible for the declared standalone surfaces; partial exact-SHA hardware qualification with W6 and P150 blockers**

Validation date: 2026-09-03 (UTC)

Machine verdict: [`ttnn-0.77.0-compatibility-matrix.json`](ttnn-0.77.0-compatibility-matrix.json)

## Verdict

`tt-transformers==0.1.0.dev0` may retain `ttnn==0.77.0` as its host-qualified dependency on CPython 3.10 and 3.12. The final non-editable wheel imports from site-packages with the exact base tuple, all 160 currently referenced TTNN module paths exist on both interpreters, and the complete marked host suite passes with the same exact totals on both interpreters.

At exact candidate SHA `b24eabe35c8f2c73f45493da40e5a6351eb0ec2d`, 33/34 executed nodes passed. The executed reusable-module, smoke, and token-accuracy/e2e stages are green; the W6 trace-order node is a functional blocker, and eight single-P150 nodes remain deferred. This evidence qualifies only the recorded selectors, machines, physical provenance, software stacks, and acceptance criteria. It does not promote every declared geometry or an entire model contract. All twelve model manifests remain `experimental`, with empty manifest validation evidence and null validation SHAs/dates. [Package evidence](package-wheel.md), [host evidence](host-ttnn-0.77.md), [API evidence](ttnn-0.77.0-api-availability.md), [canonical hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json).

## Exact-SHA hardware qualification

The canonical index contains 34 execution records against the 42-node matrix:

| Stage | Matrix | Executed | Passed | Functional failure | Deferred |
|---|---:|---:|---:|---:|---:|
| Reusable modules | 30 | 23 | 23 | 0 | 7 |
| Runtime trace/order | 1 | 1 | 0 | 1 | 0 |
| Model smoke | 3 | 3 | 3 | 0 | 0 |
| Token-accuracy/e2e | 8 | 7 | 7 | 0 | 1 |
| **Total** | **42** | **34** | **33** | **1** | **8** |

The positive executed-stage verdict is therefore modules 23/23, smoke 3/3, and e2e 7/7. By mesh, N150 is 9/9 passed, N300 is 6/6 passed, T3K has 7 passed and one functional failure, and physical P150x4 is 11/11 passed. The N150 rows are logical one-chip submeshes on the physical T3K host; they are not standalone-N150 product evidence.

Priority 17, `wh-t3k-runtime-trace-order`, ran all four W6 capture/sampling order cases. Each failed strict logits parity: row-0 maximum absolute difference was 1.5 against a 1.0 limit, and top-5 overlap was 3 against a minimum of 4. This is classified `functional_failure`, not a hardware-lifecycle failure. [W6 evidence](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/wh-lb-42/20260903T001610.842135Z-wh-t3k-runtime-trace-order.json).

Priorities 24–30 and 38 are the eight deferred `MESH_DEVICE=P150` nodes. They require `bh-lb-11`, an eight-P150 development loudbox. The available `bh-qb-05` is a physical P150_X4 quietbox and cannot substitute for that single-P150 physical provenance. These absences are neither passes nor failures. [Hardware matrix](../manifests/hardware-matrix.json).

There were zero hardware-lifecycle failures and zero resets. All 34 records report that the process exited, while fixture teardown was not independently hardware-verified; this verdict preserves that limitation rather than claiming independent clean-device verification.

## Exact software matrix

| Python | Dependency-group environment | Installed standalone base-wheel environment | Result | Evidence |
|---|---|---|---|---|
| CPython 3.10.19 | `ttnn==0.77.0`; `torch==2.11.0+cpu`; `loguru==0.6.0`; optional qualification set includes `transformers==5.12.1`, `tqdm==4.66.3`, `pytest==9.0.3` | `tt-transformers==0.1.0.dev0`; `ttnn==0.77.0`; `torch==2.11.0`; `loguru==0.6.0`; Transformers/tqdm/pytest absent and blocked | dependency resolution/import, wheel import, and `pip check` pass | [3.10 dependencies](dependencies-py310.md), [wheel audit](package-wheel.md) |
| CPython 3.12.13 | `ttnn==0.77.0`; `torch==2.11.0+cpu`; `loguru==0.6.0`; optional qualification set includes `transformers==5.12.1`, `tqdm==4.66.3`, `pytest==9.0.3` | `tt-transformers==0.1.0.dev0`; `ttnn==0.77.0`; `torch==2.11.0`; `loguru==0.6.0`; Transformers/tqdm/pytest absent and blocked | dependency resolution/import, wheel import, and `pip check` pass | [3.12 dependencies](dependencies-py312.md), [wheel audit](package-wheel.md) |
| CPython 3.10.19 host semantics | direct host-test set additionally includes `jsonschema==4.26.0` and `pytz==2026.3.post1` | final rebuilt wheel installed non-editably; no `PYTHONPATH` | 2,162 passed; 28 intentional skips; 6,791 deselected; 5 warnings; 81 subtests passed; zero failures/errors; exit 0 | [full host report](host-ttnn-0.77.md), [substantive triage](ttnn-0.77.0-host-triage.md) |
| CPython 3.12.13 host semantics | direct host-test set additionally includes `jsonschema==4.26.0` and `pytz==2026.3.post1` | final rebuilt wheel installed non-editably; no `PYTHONPATH`; offline controls enabled | 2,162 passed; 28 intentional skips; 6,791 deselected; 5 warnings; 81 subtests passed; zero failures/errors; exit 0; shutdown-only binding diagnostics retained below | [full host report](host-ttnn-0.77.md), [substantive triage](ttnn-0.77.0-host-triage.md) |

The `+cpu` Torch build is used by dependency and host-test qualification; the published package metadata requests the version-equivalent `torch==2.11.0`, and the base-wheel isolation probes resolved that exact distribution version. Neither distinction is hardware evidence.

The CPython 3.12 command returned exit 0. After pytest's successful summary, retained output `/tmp/gwang/tttv2-py312-host-final.log` reported `nanobind: leaked 8 instances!`, `leaked 36 types!`, and `leaked 330 functions!`, ending with a likely binding reference-counting issue. This is open TTNN 0.77 binding-lifecycle feedback, not a test failure and not a TTTv2 correctness claim for clean binding teardown. [Exact command and diagnostic context](host-ttnn-0.77.md).

The final static taxonomy contains 1,461 source-level test functions: 1,207 host and 254 device. The hardware records below are drawn from explicit device nodes; they are not inferred from the host suites. [Taxonomy](test-pyramid.md).

Final package/source identity:

- wheel: `dist/tt_transformers-0.1.0.dev0-py3-none-any.whl`, SHA-256 `012b58b9c8b773eb4a2a4a2d247d752572652ede81944ac91608aaa3b6e4ff1b`;
- sdist: `dist/tt_transformers-0.1.0.dev0.tar.gz`, SHA-256 `abd1bcc097596c5d992750c3ef10a668799f3342390f8179f1c39ec39344693b`;
- 132 wheel Python files are byte-identical to the source tree;
- source-tree SHA-256: `11f65588f13055117316d808fece794759929196e82031f24a6add1b0a51149f`.

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
| Foundation helpers | imports and host semantics pass on 3.10/3.12 | partial exact-SHA matrix evidence; not surface-complete | tensor/device/mesh helpers, cache/environment preflight, program-config serialization, scoped ownership | [wheel audit](package-wheel.md), [triage](ttnn-0.77.0-host-triage.md), [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| Reusable modules | imports and host contracts pass on 3.10/3.12 | executed scope passes 23/23; seven P150 nodes deferred | attention, LM head, MLP, RMSNorm and paged-transition selectors on recorded N150/N300/T3K/P150x4 geometry | [host report](host-ttnn-0.77.md), [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| Sampling foundation | imports and host contracts pass on 3.10/3.12 | partial exact-SHA matrix evidence; not surface-complete | canonical params, preparation, penalties, seed/state, log-probability and cleanup ownership | [host report](host-ttnn-0.77.md), [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| LLM runtime | imports and host contracts pass on 3.10/3.12 | blocked by W6 functional failure | prefill/decode planning and exact recorded execution; trace-order parity remains failing | [host report](host-ttnn-0.77.md), [W6 evidence](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/wh-lb-42/20260903T001610.842135Z-wh-t3k-runtime-trace-order.json) |
| Shared executors | imports and host contracts pass on 3.10/3.12 | partial; executed smoke/e2e pass, with W6 blocker | family-neutral/Llama/Qwen configuration, delegation, request and ownership contracts plus recorded selectors | [host report](host-ttnn-0.77.md), [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| Twelve concrete model cores | package/core imports and profile/config contracts pass on 3.10/3.12 | executed smoke 3/3 and e2e 7/7; manifests not qualified | exact recorded model selectors only; incomplete per-manifest geometry, context and lifecycle coverage | [host report](host-ttnn-0.77.md), [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |

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
| T3K runtime functional failure | W6 capture/sampling order parity | all four cases fail the same strict 1.5/1.0 max-abs and 3/4 top-5 criteria; remains a release blocker | [W6 evidence](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/wh-lb-42/20260903T001610.842135Z-wh-t3k-runtime-trace-order.json) |
| Hardware lifecycle/reset | no lifecycle-classified record and no reset | zero lifecycle failures and zero resets across 34 records; independent post-fixture verification was not recorded | [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| Unsupported/unavailable model geometry | eight single-P150 nodes require unavailable `bh-lb-11` | deferred, not transferable from the physical P150_X4 quietbox | [hardware matrix](../manifests/hardware-matrix.json) |

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

## Hardware evidence limits

| Not established beyond the exact records | Consequence | Evidence |
|---|---|---|
| Single-P150 behavior on the required `bh-lb-11` development loudbox | seven module nodes and one Llama 3.1 8B e2e node remain deferred | [hardware matrix](../manifests/hardware-matrix.json) |
| Experimental/private operations not selected by the 34 executed nodes | availability remains broader than hardware semantic evidence | [API report](ttnn-0.77.0-api-availability.md), [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| Runtime trace-order correctness | W6 is measured and failing, not untested or qualified | [W6 evidence](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/wh-lb-42/20260903T001610.842135Z-wh-t3k-runtime-trace-order.json) |
| Hardware behavior outside the exact N150, N300, T3K and P150x4 selectors | no transfer to other SKU/mesh/TP/DP/batch/context combinations | [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| Performance qualification beyond recorded correctness and token-accuracy criteria | passing e2e correctness does not establish a performance envelope | [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| Independent post-fixture device teardown verification | records prove process exit but explicitly do not claim independent hardware verification | [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |

## Per-model verdicts

Every row below means: package/core imports and host-only configuration/contracts pass on CPython 3.10 and 3.12. Some models have passing exact-SHA smoke/e2e records, but no complete manifest contract is qualified and every lifecycle remains `experimental`. The target count describes manifest intent, not a promoted support claim.

| Model | Lifecycle / host verdict | HF revision | Source-declared targets (not fully validated) | Remaining gaps | Evidence |
|---|---|---|---|---|---|
| DeepSeek R1 Distill Qwen 14B | experimental; host pass | `1df8507178afcc1bef68cd8c393f61a886323761` | 4 WH targets: N300 TP2 plus T3K TP8/TP4/TP2 lanes | partial matrix evidence does not qualify full model; N150/TP1; N300 accuracy eval-32/batch-32-ci; undeclared geometry | [manifest](../../examples/deepseek_r1_distill_qwen_14b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.2 1B | experimental; host pass | `9213176726f574b556790deb65791e0c5aa438b6` | 6 WH targets across N150/N300/T3K, TP1/2/8 and declared DP | partial matrix evidence does not qualify full model; TP4; N150 batch-32-ci 2048; undeclared geometry | [manifest](../../examples/llama32_1b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.2 3B | experimental; host pass | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | 6 WH targets across N150/N300/T3K, TP1/2/8 and declared DP | partial matrix evidence does not qualify full model; TP4; N150 traced prefill; undeclared geometry | [manifest](../../examples/llama32_3b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.3 70B | experimental; host pass | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | 2 targets: WH T3K TP8 and BH logical P150_X4 TP4 | partial matrix evidence does not qualify full model; DP>1; P150x4 orientation; 32K context; undeclared geometry | [manifest](../../examples/llama33_70b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.1 8B | experimental; host pass | `0e9e39f249a16976918f6564b8830bc894c89659` | 11 WH/BH targets across N150/N300/T3K/P150/P150_X4/physical-P300 declarations | partial matrix evidence does not qualify full model; removed TTTv1 prefetcher; P100/128K P300; stand-in provenance; undeclared geometry | [manifest](../../examples/llama3_8b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Mistral 7B | experimental; host pass | `c170c708c41dac9275d15a8fff4eca08d52bab71` | 5 WH targets across N150/N300/T3K and one-device DP lanes | partial matrix evidence does not qualify full model; other DP layouts; undeclared geometry | [manifest](../../examples/mistral_7b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Phi-4 | experimental; host pass | `187ef0342fff0eb3333be9f00389385e95ef0b61` | 2 WH targets: N300 TP2 and T3K DP4/TP2 | partial matrix evidence does not qualify full model; N150; ordinary T3K/TG TP; other DP; undeclared geometry | [manifest](../../examples/phi4/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen 2.5 72B | experimental; host pass | `495f39366efef23836d0cfae4fbe635880d2be31` | 1 WH target: T3K TP8 | partial matrix evidence does not qualify full model; non-T3K; DP>1; undeclared geometry | [manifest](../../examples/qwen25_72b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen 2.5 7B | experimental; host pass | `a09a35458c702b33eeacc393d103063234e8bc28` | 2 WH targets: N300 TP2 and T3K DP4/TP2 | partial matrix evidence does not qualify full model; N150; ordinary T3K/TG TP; other DP; undeclared geometry | [manifest](../../examples/qwen25_7b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen 2.5 Coder 32B | experimental; host pass | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` | 1 WH target: T3K TP8 | partial matrix evidence does not qualify full model; non-T3K meshes; DP>1; undeclared geometry | [manifest](../../examples/qwen25_coder_32b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen2 7B | experimental; host pass | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | 2 WH targets: N300 TP2 and T3K DP4/TP2 | partial matrix evidence does not qualify full model; N150; ordinary T3K/TG TP; other DP; undeclared geometry | [manifest](../../examples/qwen2_7b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen3 32B | experimental; host pass | `9216db5781bf21249d130ec9da846c4624c16137` | 2 targets: WH T3K TP8 and BH logical P150_X4 TP4 | partial matrix evidence does not qualify full model; P100/P150/P300 single/two-die; DP>1; orientation; undeclared geometry | [manifest](../../examples/qwen3_32b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |

All manifest `validation.evidence` arrays are empty and validation dates/SHAs are null. The global exact-SHA evidence does not silently populate those model-owned fields, so no row is promoted to `qualified`.

## Remaining hardware, firmware and geometry gates

| Gap | Required closure evidence | Current verdict | Evidence |
|---|---|---|---|
| Reusable module kernels | run priorities 24–30 on the required `bh-lb-11` host | 23/23 executed pass; seven single-P150 nodes deferred | [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json), [hardware matrix](../manifests/hardware-matrix.json) |
| Runtime eager/trace/cache/sampling/cleanup | resolve and rerun W6, then extend beyond the selected matrix paths | measured functional failure on W6; other executed module/model paths retain their own passes | [W6 evidence](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/wh-lb-42/20260903T001610.842135Z-wh-t3k-runtime-trace-order.json) |
| Fourteen unstable/private APIs | per-operation device call/signature and semantic evidence on intended architectures | all present; only matrix-selected behavior has device evidence | [API report](ttnn-0.77.0-api-availability.md), [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| Firmware/driver compatibility | preserve exact stack identity per future execution and test additional intended stacks explicitly | exact stacks recorded for `wh-lb-42` and `bh-qb-05`; no transfer beyond those records | [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| Model geometry | correctness for every claimed SKU/mesh/TP/DP/batch/context bucket; explicit unsupported outcomes elsewhere | partial exact-SHA evidence; manifests remain experimental | [hardware inventory](../analysis/support/hardware_coverage.csv), [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) |
| End-to-end model correctness/performance | run priority 38, expand incomplete model contracts, and add separate performance criteria | 7/7 executed e2e pass; one P150 e2e deferred; performance not generalized | [hardware index](../evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json), [wheel audit](package-wheel.md) |
| CPython 3.12 TTNN binding teardown | minimize the exit-0 nanobind diagnostic and correct reference ownership in TTNN bindings | open non-failing binding feedback | [host report](host-ttnn-0.77.md), [host triage](ttnn-0.77.0-host-triage.md) |

Hardware tests remain serialized within each physical host; independent Wormhole and Blackhole hosts may each run one process concurrently. A skip is not evidence. The correct release statement is: “host-compatible with `ttnn==0.77.0`; 33/34 exact-SHA hardware executions pass, with W6 failing functionally and eight P150 nodes deferred; all model manifests remain experimental.”

## Deterministic coverage gate

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python qualification/tools/validate_ttnn_077_compatibility.py
```

The validator regenerates the matrix projection in memory and requires byte-stable JSON. It verifies:

- exactly the eight base/module/runtime/model aggregate surfaces above have verdicts and existing evidence;
- source model directories, all current `examples/*/support.json` manifests, and matrix model rows are the same twelve-package set;
- every model remains experimental/not model-qualified with non-empty gaps and empty manifest-owned hardware validation evidence;
- 160 total TTNN paths and all 14 unstable paths are present on both Pythons;
- both exact rebuilt-wheel host-suite records contain 2,162 passes, 28 skips, 6,791 deselections, 5 warnings, 81 passing subtests, zero failures/errors and exit 0;
- the static taxonomy remains 1,461 source-level functions: 1,207 host and 254 device;
- Python 3.12's 8-instance/36-type/330-function nanobind shutdown diagnostic is retained and classified as non-failing TTNN binding feedback;
- the canonical final-SHA index and all referenced JSON/log sizes and SHA-256 digests match, all 34 records reconcile to the matrix, and no mixed-SHA record is accepted;
- exact hardware totals remain 33 passes and one W6 functional failure, with modules 23/23, smoke 3/3, e2e 7/7, eight P150 nodes deferred, and no lifecycle failure/reset;
- every local evidence path exists.

This validator and report do not execute TTNN callables or query devices. They validate and project the already-recorded exact-SHA hardware evidence without broadening it into unmeasured support claims.
