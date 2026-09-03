# TTNN 0.77.0 compatibility verdict

Status: **host-compatible for the declared standalone surfaces; all 42 exact-SHA hardware matrix nodes pass; model-contract and performance qualification remain partial**

Validation date: 2026-09-03 (UTC)

Machine verdict: [`ttnn-0.77.0-compatibility-matrix.json`](ttnn-0.77.0-compatibility-matrix.json)

## Verdict

`tt-transformers==0.1.0.dev0` may retain `ttnn==0.77.0` as its host-qualified dependency on CPython 3.10 and 3.12. The final non-editable wheel imports from site-packages with the exact base tuple, all 160 currently referenced TTNN module paths exist on both interpreters, and the complete marked host suite passes with the same exact totals on both interpreters.

At exact candidate SHA `73d414f8b826a7da982df8c8229d4ac41ed8ba33`, 42/42 matrix nodes passed. The reusable-module, runtime trace/order, smoke, and token-accuracy/e2e stages are all green. This evidence qualifies only the recorded selectors, machines, physical provenance, software stacks, and acceptance criteria. It does not promote every declared geometry or an entire model contract. All twelve model manifests remain `experimental`, with empty manifest validation evidence and null validation SHAs/dates. [Package evidence](package-wheel.md), [host evidence](host-ttnn-0.77.md), [API evidence](ttnn-0.77.0-api-availability.md), [canonical hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json).

## Exact-SHA hardware qualification

The canonical index contains all 42 execution records against the 42-node matrix. Its SHA-256 is `4e98b62c34fb9f8744e00624c091dc9de18b3f32c5cd74f2ad2be1ad26274d7a`; the matrix SHA-256 is `1d04716dd79cf3c5ab6a7a2ac251224fe326ad64ef19fa440c06187118b9d7de`:

| Stage | Matrix | Executed | Passed | Functional failure | Deferred |
|---|---:|---:|---:|---:|---:|
| Reusable modules | 30 | 30 | 30 | 0 | 0 |
| Runtime trace/order | 1 | 1 | 1 | 0 | 0 |
| Model smoke | 3 | 3 | 3 | 0 | 0 |
| Token-accuracy/e2e | 8 | 8 | 8 | 0 | 0 |
| **Total** | **42** | **42** | **42** | **0** | **0** |

The positive executed-stage verdict is therefore modules 30/30, runtime 1/1, smoke 3/3, and e2e 8/8. By mesh, N150 is 9/9 passed, N300 is 6/6 passed, T3K is 8/8 passed, P150 is 8/8 passed, and P150x4 is 11/11 passed. The N150 rows are logical one-chip submeshes on the physical T3K host; they are not standalone-N150 product evidence.

Priority 17, `wh-t3k-runtime-trace-order`, ran the unchanged strict W6 oracle across all four capture/sampling order cases. W6 passes all four after the accuracy profile made folded QKV/W2 prefill use the same linear operator family as its batch-one oracle. The four cases completed in 490.54 seconds. That slower timing is an observation, not a configured performance-floor failure; the node has correctness and cross-order acceptance criteria. This closes the prior numerical blocker without relaxing its rowwise logits thresholds. [W6 evidence](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/wh-lb-42/20260903T124923.309445Z-wh-t3k-runtime-trace-order.json).

Priorities 24–30 and 38 passed on `bh-qb-05` with `MESH_DEVICE=P150` as the only device-selection environment setting. These records establish logical 1x1 execution on the physical P150_X4 2x2 quietbox, using the default TTNN-selected board without a visibility mask. They are not standalone-product or full-host P150_X4 evidence. [Hardware matrix](../manifests/hardware-matrix.json).

There were zero hardware-lifecycle failures and zero resets. All 42 records report that the process exited, while fixture teardown was not independently hardware-verified; this verdict preserves that limitation rather than claiming independent clean-device verification.

## Noncanonical Llama 3.3 performance feedback

Four T3K accuracy-profile diagnostics at the older code SHA
`2883a949860d749adc2ed1af5525b27a9a547505` rechecked the source-declared
batch-32 and batch-32-ci performance targets after the W6 precision-policy
change. TTFT passed 4/4 against the 100 ms target plus 5% tolerance, while
throughput failed 4/4 against the existing profile- and sampling-specific
floors. All four pytest processes therefore exited 1 from the throughput
assertion, despite the TTFT sub-target passing. They completed clean hardware
teardown with zero lifecycle failures and zero resets.

| Case | Sampling | TTFT / adjusted max | tok/s/u / adjusted minimum | Verdict |
|---|---|---:|---:|---|
| accuracy batch-32 | host | 87.1 / 105.0 ms | 7.9 / 8.835 | TTFT pass; throughput and overall target fail |
| accuracy batch-32 | on-device top-k | 86.7 / 105.0 ms | 12.2 / 13.68 | TTFT pass; throughput and overall target fail |
| accuracy batch-32-ci | host | 86.9 / 105.0 ms | 7.7 / 8.455 | TTFT pass; throughput and overall target fail |
| accuracy batch-32-ci | on-device top-k | 87.2 / 105.0 ms | 11.9 / 13.49 | TTFT pass; throughput and overall target fail |

These runs are explicitly `non-canonical performance diagnostic` feedback and
were not rerun at hardware candidate `73d414f8b826a7da982df8c8229d4ac41ed8ba33`.
They add zero records to the canonical 42-node index, do not change its 42/42
pass count, do not establish a performance envelope, and do not promote the
Llama manifest. [Diagnostic summary](../evidence/diagnostics/2883a949860d749adc2ed1af5525b27a9a547505/wh-lb-42/accuracy-ttft/summary.json).

## Exact software matrix

| Python | Dependency-group environment | Installed standalone base-wheel environment | Result | Evidence |
|---|---|---|---|---|
| CPython 3.10.19 | `ttnn==0.77.0`; `torch==2.11.0+cpu`; `loguru==0.6.0`; optional qualification set includes `transformers==5.12.1`, `tqdm==4.66.3`, `pytest==9.0.3` | `tt-transformers==0.1.0.dev0`; `ttnn==0.77.0`; `torch==2.11.0`; `loguru==0.6.0`; Transformers/tqdm/pytest absent and blocked | dependency resolution/import, wheel import, and `pip check` pass | [3.10 dependencies](dependencies-py310.md), [wheel audit](package-wheel.md) |
| CPython 3.12.13 | `ttnn==0.77.0`; `torch==2.11.0+cpu`; `loguru==0.6.0`; optional qualification set includes `transformers==5.12.1`, `tqdm==4.66.3`, `pytest==9.0.3` | `tt-transformers==0.1.0.dev0`; `ttnn==0.77.0`; `torch==2.11.0`; `loguru==0.6.0`; Transformers/tqdm/pytest absent and blocked | dependency resolution/import, wheel import, and `pip check` pass | [3.12 dependencies](dependencies-py312.md), [wheel audit](package-wheel.md) |
| CPython 3.10.19 host semantics | direct host-test set additionally includes `jsonschema==4.26.0` and `pytz==2026.3.post1` | final rebuilt wheel installed non-editably; no `PYTHONPATH` | 2,170 passed; 28 intentional skips; 6,791 deselected; 5 warnings; 81 subtests passed; zero failures/errors; exit 0 | [full host report](host-ttnn-0.77.md), [substantive triage](ttnn-0.77.0-host-triage.md) |
| CPython 3.12.13 host semantics | direct host-test set additionally includes `jsonschema==4.26.0` and `pytz==2026.3.post1` | final rebuilt wheel installed non-editably; no `PYTHONPATH`; offline controls enabled | 2,170 passed; 28 intentional skips; 6,791 deselected; 5 warnings; 81 subtests passed; zero failures/errors; exit 0; shutdown-only binding diagnostics retained below | [full host report](host-ttnn-0.77.md), [substantive triage](ttnn-0.77.0-host-triage.md) |

The `+cpu` Torch build is used by dependency and host-test qualification; the published package metadata requests the version-equivalent `torch==2.11.0`, and the base-wheel isolation probes resolved that exact distribution version. Neither distinction is hardware evidence.

The CPython 3.12 command returned exit 0. After pytest's successful summary, retained output `/tmp/gwang/tttv2-73d-py312-host-final.log` reported `nanobind: leaked 10 instances!`, `leaked 36 types!`, and `leaked 330 functions!`, ending with a likely binding reference-counting issue. This is open TTNN 0.77 binding-lifecycle feedback, not a test failure and not a TTTv2 correctness claim for clean binding teardown. [Exact command and diagnostic context](host-ttnn-0.77.md).

The final static taxonomy contains 1,465 source-level test functions: 1,211 host, 254 device, and 393 model-marked surfaces. The hardware records below are drawn from explicit device nodes; they are not inferred from the host suites. [Taxonomy](test-pyramid.md).

Final package/source identity:

- wheel: `dist/tt_transformers-0.1.0.dev0-py3-none-any.whl`, SHA-256 `8c55fac0a764fb9ae4d6ca514062ef2cb6cfe3097306a2f877bcad41269c50c2`;
- sdist: `dist/tt_transformers-0.1.0.dev0.tar.gz`, SHA-256 `f161e13dedc5ce076d9553b677f0a1a4785996f932316f2325de9217376da644`;
- 132 wheel Python files are byte-identical to the source tree;
- source-tree SHA-256: `7b03baf498e2a2252759d89813fcb898dd88daf573fb46f0e77e5c2cc6abad97`.

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
| Foundation helpers | imports and host semantics pass on 3.10/3.12 | full exact-SHA matrix exercised; not surface-complete | tensor/device/mesh helpers, cache/environment preflight, program-config serialization, scoped ownership | [wheel audit](package-wheel.md), [triage](ttnn-0.77.0-host-triage.md), [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| Reusable modules | imports and host contracts pass on 3.10/3.12 | executed scope passes 30/30 | attention, LM head, MLP, RMSNorm and paged-transition selectors on recorded N150/N300/T3K/P150/P150x4 geometry | [host report](host-ttnn-0.77.md), [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| Sampling foundation | imports and host contracts pass on 3.10/3.12 | full exact-SHA matrix exercised; not surface-complete | canonical params, preparation, penalties, seed/state, log-probability and cleanup ownership | [host report](host-ttnn-0.77.md), [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| LLM runtime | imports and host contracts pass on 3.10/3.12 | executed W6 trace/order scope passes; runtime scope remains partial | prefill/decode planning and exact recorded execution; all four strict W6 order cases pass | [host report](host-ttnn-0.77.md), [W6 evidence](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/wh-lb-42/20260903T124923.309445Z-wh-t3k-runtime-trace-order.json) |
| Shared executors | imports and host contracts pass on 3.10/3.12 | partial; executed runtime, smoke and e2e scopes pass | family-neutral/Llama/Qwen configuration, delegation, request and ownership contracts plus recorded selectors | [host report](host-ttnn-0.77.md), [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| Twelve concrete model cores | package/core imports and profile/config contracts pass on 3.10/3.12 | executed smoke 3/3 and e2e 8/8; manifests not qualified | exact recorded model selectors only; incomplete per-manifest geometry, context and lifecycle coverage | [host report](host-ttnn-0.77.md), [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |

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
| TTNN binding-lifecycle feedback | CPython 3.12 shutdown reports 10 leaked instances, 36 leaked types and 330 leaked functions | open upstream binding feedback; pytest completed with exit 0, so this is not hidden and not classified as a test failure | [host report](host-ttnn-0.77.md), [host triage](ttnn-0.77.0-host-triage.md) |
| TTTv2 implementation defect | accuracy-profile folded QKV/W2 used a different operator family from the W6 batch-one oracle | explicit accuracy-linear/performance-minimal policy; W6 passes all four strict order cases without threshold relaxation | [W6 evidence](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/wh-lb-42/20260903T124923.309445Z-wh-t3k-runtime-trace-order.json) |
| Hardware lifecycle/reset | no lifecycle-classified record and no reset | zero lifecycle failures and zero resets across 42 records; independent post-fixture verification was not recorded | [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| Logical single-P150 provenance | eight `MESH_DEVICE=P150` nodes run on `bh-qb-05` | 8/8 pass as logical 1x1 execution on physical P150_X4; no standalone-product or full-host claim | [hardware matrix](../manifests/hardware-matrix.json) |

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
| Experimental/private operations not selected by the 42 executed nodes | availability remains broader than hardware semantic evidence | [API report](ttnn-0.77.0-api-availability.md), [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| Runtime behavior beyond the recorded W6 selector | the strict four-case W6 trace/order scope passes, but that does not qualify every runtime path or geometry | [W6 evidence](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/wh-lb-42/20260903T124923.309445Z-wh-t3k-runtime-trace-order.json) |
| Hardware behavior outside the exact N150, N300, T3K, P150 and P150x4 selectors | no transfer to other SKU/mesh/TP/DP/batch/context combinations; logical P150 evidence is not full-host P150_X4 evidence | [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| Performance qualification beyond recorded correctness and token-accuracy criteria | four noncanonical diagnostics pass TTFT but fail throughput; they do not establish a performance envelope | [diagnostic summary](../evidence/diagnostics/2883a949860d749adc2ed1af5525b27a9a547505/wh-lb-42/accuracy-ttft/summary.json) |
| Independent post-fixture device teardown verification | canonical records prove process exit but explicitly do not claim independent hardware verification | [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |

## Per-model verdicts

Every row below means: package/core imports and host-only configuration/contracts pass on CPython 3.10 and 3.12. Some models have passing exact-SHA smoke/e2e records, but no complete manifest contract is qualified and every lifecycle remains `experimental`. The target count describes manifest intent, not a promoted support claim.

| Model | Lifecycle / host verdict | HF revision | Source-declared targets (not fully validated) | Remaining gaps | Evidence |
|---|---|---|---|---|---|
| DeepSeek R1 Distill Qwen 14B | experimental; host pass | `1df8507178afcc1bef68cd8c393f61a886323761` | 4 WH targets: N300 TP2 plus T3K TP8/TP4/TP2 lanes | matrix-selected evidence does not qualify full model; N150/TP1; N300 accuracy eval-32/batch-32-ci; undeclared geometry | [manifest](../../examples/deepseek_r1_distill_qwen_14b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.2 1B | experimental; host pass | `9213176726f574b556790deb65791e0c5aa438b6` | 6 WH targets across N150/N300/T3K, TP1/2/8 and declared DP | matrix-selected evidence does not qualify full model; TP4; N150 batch-32-ci 2048; undeclared geometry | [manifest](../../examples/llama32_1b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.2 3B | experimental; host pass | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | 6 WH targets across N150/N300/T3K, TP1/2/8 and declared DP | matrix-selected evidence does not qualify full model; TP4; N150 traced prefill; undeclared geometry | [manifest](../../examples/llama32_3b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.3 70B | experimental; host pass | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | 2 targets: WH T3K TP8 and BH logical P150_X4 TP4 | matrix-selected evidence does not qualify full model; DP>1; P150x4 orientation; 32K context; undeclared geometry | [manifest](../../examples/llama33_70b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Llama 3.1 8B | experimental; host pass | `0e9e39f249a16976918f6564b8830bc894c89659` | 11 WH/BH targets across N150/N300/T3K/P150/P150_X4/physical-P300 declarations | matrix-selected evidence does not qualify full model; removed TTTv1 prefetcher; P100/128K P300; stand-in provenance; undeclared geometry | [manifest](../../examples/llama3_8b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Mistral 7B | experimental; host pass | `c170c708c41dac9275d15a8fff4eca08d52bab71` | 5 WH targets across N150/N300/T3K and one-device DP lanes | matrix-selected evidence does not qualify full model; other DP layouts; undeclared geometry | [manifest](../../examples/mistral_7b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Phi-4 | experimental; host pass | `187ef0342fff0eb3333be9f00389385e95ef0b61` | 2 WH targets: N300 TP2 and T3K DP4/TP2 | matrix-selected evidence does not qualify full model; N150; ordinary T3K/TG TP; other DP; undeclared geometry | [manifest](../../examples/phi4/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen 2.5 72B | experimental; host pass | `495f39366efef23836d0cfae4fbe635880d2be31` | 1 WH target: T3K TP8 | matrix-selected evidence does not qualify full model; non-T3K; DP>1; undeclared geometry | [manifest](../../examples/qwen25_72b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen 2.5 7B | experimental; host pass | `a09a35458c702b33eeacc393d103063234e8bc28` | 2 WH targets: N300 TP2 and T3K DP4/TP2 | matrix-selected evidence does not qualify full model; N150; ordinary T3K/TG TP; other DP; undeclared geometry | [manifest](../../examples/qwen25_7b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen 2.5 Coder 32B | experimental; host pass | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` | 1 WH target: T3K TP8 | matrix-selected evidence does not qualify full model; non-T3K meshes; DP>1; undeclared geometry | [manifest](../../examples/qwen25_coder_32b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen2 7B | experimental; host pass | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | 2 WH targets: N300 TP2 and T3K DP4/TP2 | matrix-selected evidence does not qualify full model; N150; ordinary T3K/TG TP; other DP; undeclared geometry | [manifest](../../examples/qwen2_7b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |
| Qwen3 32B | experimental; host pass | `9216db5781bf21249d130ec9da846c4624c16137` | 2 targets: WH T3K TP8 and BH logical P150_X4 TP4 | matrix-selected evidence does not qualify full model; P100/P150/P300 single/two-die; DP>1; orientation; undeclared geometry | [manifest](../../examples/qwen3_32b/support.json), [host](host-ttnn-0.77.md), [wheel](package-wheel.md) |

All manifest `validation.evidence` arrays are empty and validation dates/SHAs are null. The global exact-SHA evidence does not silently populate those model-owned fields, so no row is promoted to `qualified`.

## Remaining hardware, firmware and geometry gates

| Gap | Required closure evidence | Current verdict | Evidence |
|---|---|---|---|
| Reusable module kernels | retain exact-SHA regression coverage when implementation or stack changes | 30/30 matrix module nodes pass | [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json), [hardware matrix](../manifests/hardware-matrix.json) |
| Runtime eager/trace/cache/sampling/cleanup | extend beyond the selected W6 matrix path | W6 passes all four strict cases; broader runtime scope remains partial | [W6 evidence](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/wh-lb-42/20260903T124923.309445Z-wh-t3k-runtime-trace-order.json) |
| Fourteen unstable/private APIs | per-operation device call/signature and semantic evidence on intended architectures | all present; only matrix-selected behavior has device evidence | [API report](ttnn-0.77.0-api-availability.md), [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| Firmware/driver compatibility | preserve exact stack identity per future execution and test additional intended stacks explicitly | exact stacks recorded for `wh-lb-42` and `bh-qb-05`; no transfer beyond those records | [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| Model geometry | correctness for every claimed SKU/mesh/TP/DP/batch/context bucket; explicit unsupported outcomes elsewhere | full 42-node matrix evidence exists; manifests remain experimental | [hardware inventory](../analysis/support/hardware_coverage.csv), [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json) |
| End-to-end model correctness/performance | expand incomplete model contracts and close the measured throughput regressions | 8/8 canonical e2e executions pass; all four older noncanonical diagnostics fail throughput and were not rerun at the hardware SHA | [hardware index](../evidence/hardware/73d414f8b826a7da982df8c8229d4ac41ed8ba33/index.json), [diagnostic summary](../evidence/diagnostics/2883a949860d749adc2ed1af5525b27a9a547505/wh-lb-42/accuracy-ttft/summary.json) |
| CPython 3.12 TTNN binding teardown | minimize the exit-0 nanobind diagnostic and correct reference ownership in TTNN bindings | open non-failing binding feedback | [host report](host-ttnn-0.77.md), [host triage](ttnn-0.77.0-host-triage.md) |

Hardware tests remain serialized within each physical host; independent Wormhole and Blackhole hosts may each run one process concurrently. A skip is not evidence. The correct release statement is: “host-compatible with `ttnn==0.77.0`; all 42 exact-SHA hardware matrix executions pass; performance and complete model contracts remain partial; all model manifests remain experimental.”

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
- both exact rebuilt-wheel host-suite records contain 2,170 passes, 28 skips, 6,791 deselections, 5 warnings, 81 passing subtests, zero failures/errors and exit 0;
- the static taxonomy remains 1,465 source-level functions: 1,211 host, 254 device, and 393 model-marked surfaces;
- Python 3.12's 8-instance/36-type/330-function nanobind shutdown diagnostic is retained and classified as non-failing TTNN binding feedback;
- the canonical final-SHA index and all referenced JSON/log sizes and SHA-256 digests match, all 42 records reconcile to the matrix, and no mixed-SHA record is accepted;
- exact hardware totals remain 42 passes and zero functional failures, with modules 30/30, runtime 1/1, smoke 3/3, e2e 8/8, and no lifecycle failure/reset;
- all four noncanonical Llama 3.3 performance diagnostics at older SHA `2883a949860d749adc2ed1af5525b27a9a547505` pass TTFT and fail throughput without changing the canonical 42-record index;
- every local evidence path exists.

This validator and report do not execute TTNN callables or query devices. They validate and project the already-recorded exact-SHA hardware evidence without broadening it into unmeasured support claims.
