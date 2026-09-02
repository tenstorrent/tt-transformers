# Qwen3-32B with TTTv2

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Run the concrete Qwen3 model on T3K TP8 or declared Blackhole P150x4 TP4, including accuracy, trace, determinism, seeded batching, and focused smoke diagnostics.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `Qwen/Qwen3-32B`.
- Hugging Face revision: 9216db5781bf21249d130ec9da846c4624c16137.

### Candidate software tuple

This is the declared qualification candidate, not a passing verdict:

- `tt-transformers==0.1.0.dev0`
- `ttnn==0.77.0`
- Python `3.10, 3.12`
- `torch==2.11.0`
- `transformers==5.12.1`

### Declared hardware geometry

| Architecture | Physical SKU/system declaration | Mesh | TP | DP |
| --- | --- | --- | ---: | ---: |
| wormhole | T3K | 1x8 | 8 | 1 |
| blackhole | P150_X4 required logical target; code accepts physical P300_X2 alias with distinct provenance | 1x4 | 4 | 1 |

Only these source-declared rows are candidates. No row has passing hardware evidence at the pinned extraction revision.

### Proven limits and features

- Demo cases cover active batch 1 or 32.
- P150x4 buckets are 4096 for token accuracy, 1024 for eval, and 2048 for batch-32-ci.
- model lanes require at least TP4; all DP cases capacity-skip.
- P150x4 batched prefill remains disabled unless exact-token cross-cardinality invariance is proven.
- Device sampling: Host and on-device top-k paths exist; P150x4 includes a seeded exact-token cross-cardinality diagnostic.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `qualification/assets/reference_outputs/qwen3_32b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative run using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3-32B python -m examples.qwen3_32b.demo --case token-accuracy --optimizations performance
```

Focused smoke:

```bash
MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3-32B python -m examples.qwen3_32b.smoke --case qwen3-32b-prefill-smoke
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=T3K pytest --collect-only -q tests/hardware/models/qwen3_32b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/qwen3_32b/test_demo.py)
- [Pinned support baseline](../../qualification/analysis/support/support_baseline.md)
- [Phase 3 boundary evidence](../../qualification/extraction/support_boundary.md)
- [Historical evidence ledger](../../qualification/analysis/support/hardware_evidence.csv)

### Known and unsupported gaps

- P100/P150/P300 single/two-die model paths.
- DP greater than 1.
- noncanonical P150x4 orientations.
- any geometry not declared in support.json.
- No hardware evidence is attributable to the pinned extraction revision.

<!-- END GENERATED SUPPORT -->

This directory contains the model-owned Qwen3-32B TTTv2 product path.
Qwen3-32B is currently the only Qwen3 model in `models/common/models`, so its
sampling and trace policy remains here rather than in a speculative Qwen3
family module.

## Product path

```text
Qwen/Qwen3-32B
  -> hf_adaptor.py: provider metadata, tokenizer, and weight conversion
  -> model.py: Qwen3 tensor graph composed from TTTv2 modules
  -> executor.py: Qwen3 policy over the shared model-layer ModelExecutor
  -> generator.py: vLLM construction, DP composition, and dispatch
```

## Files

| File | Responsibility |
| --- | --- |
| `hf_adaptor.py` | Resolve HF configuration/tokenizer/weights and construct runtime metadata |
| `weight_utils.py` | Convert and map provider weights |
| `model.py` | Build and execute the Qwen3-32B tensor graph |
| `executor.py` | Supply Qwen3-native sampling, trace-prime, and compatibility policy |
| `generator.py` | Build lanes, compose DP, configure `VLLMAdapter`, and dispatch calls |
| `demo.py` | Direct model demonstration entry point |

## Tensor-module composition

The model graph uses reusable TTTv2 modules for embedding, rotary setup,
RMSNorm, attention, MLP, LM head, and optional `Sampling1D`, together with
common TT collective helpers. Qwen3-specific QK normalization, dimensions,
precision, and device tuning remain in `model.py` and its configuration.

## Executor construction

`Qwen3_32BExecutor` is a composition facade over
`models/common/models/executor.py::ModelExecutor`. It does not subclass the
shared executor and does not duplicate its resource lifecycle.

The shared owner composes:

```text
Qwen3_32B model
├── PagedKVCacheManager
├── OutputReader
├── PrefillRuntime
├── DecodeRuntime
├── ProgramCompiler
├── EagerExecutor
├── optional TraceCompiler
├── optional TracedExecutor
└── WarmupCoordinator
```

The model-owned facade supplies only Qwen3-32B policy:

- caller-owned `SamplingState1D` state when device sampling is enabled;
- prompt history, output history, and slot-remap forwarding;
- sequential prefill while stateful device sampling is active;
- T3K trace-capture prime sequence lengths; and
- legacy eager/traced wrappers, trace bookkeeping, and direct-run helpers.

There is intentionally no `models/common/models/qwen3_executor.py`. A Qwen3
family layer should be introduced only after another Qwen3 model demonstrates
the same policy.

## vLLM and data parallelism

`Qwen3_32BGenerator` constructs one model/executor per lane. DP1 uses the
executor facade directly. DP greater than one uses `LaneGroupExecutor`, which
slices request state per lane and restores global output/logprob order.

`VLLMAdapter` normalizes server calls and validates the vLLM-selected paged-KV
shape. The executor lanes own TT resources; the generator and adapter own no TT
tensors.

## Ownership and cleanup

The composed `ModelExecutor` owns KV tensors, compile/trace registries, output
leases, sampling state, and deterministic cleanup. Cleanup terminalizes the
lane, drains external reads, releases prefill/decode transients and traces,
then releases sampling and KV resources with retryable failure reporting.

## Tests

Relevant entry points include:

- `models/common/tests/models/qwen3_32b/test_hf_adaptor.py`
- `models/common/tests/models/qwen3_32b/test_model_runtime_surface.py`
- `models/common/tests/models/qwen3_32b/test_module_profiles.py`
- `models/common/tests/models/qwen3_32b/test_demo_contract.py`
- `models/common/tests/models/qwen3_32b/test_p150x4_smoke.py`
- `models/common/tests/demos/qwen3_32b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
