# Llama 3.3 70B with TTTv2

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Run the concrete 70B Llama model on its full tensor-parallel lane for token accuracy, repeat-batch determinism, trace, and throughput characterization.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `meta-llama/Llama-3.3-70B-Instruct`.
- Hugging Face revision: 6f6073b423013f6a7d4d9f39144961bfbfbc386b.

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
- P150x4 contract buckets are 4096 for token accuracy and 1024 for eval/performance cases.
- one model replica requires the full declared lane; every collected DP>1 case skips.
- P150x4 long-prefill coverage is not established.
- Device sampling: Host and on-device top-k paths exist; capability contracts distinguish trace none/decode_only/all.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `qualification/assets/reference_outputs/llama33_70b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative run using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=T3K HF_MODEL=meta-llama/Llama-3.3-70B-Instruct python -m examples.llama33_70b.demo --case token-accuracy --optimizations performance
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=T3K pytest --collect-only -q tests/hardware/models/llama33_70b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/llama33_70b/test_demo.py)
- [Pinned support baseline](../../qualification/analysis/support/support_baseline.md)
- [Phase 3 boundary evidence](../../qualification/extraction/support_boundary.md)
- [Historical evidence ledger](../../qualification/analysis/support/hardware_evidence.csv)

### Known and unsupported gaps

- DP greater than 1.
- noncanonical P150x4 orientations.
- unproven 32K context.
- any geometry not declared in support.json.
- No hardware evidence is attributable to the pinned extraction revision.

<!-- END GENERATED SUPPORT -->

This directory contains the Llama 3.3 70B TTTv2 product path.

## Construction path

```text
meta-llama/Llama-3.3-70B-Instruct
  -> hf_adaptor.py: provider metadata, tokenizer, and weights
  -> model.py: multi-device TTTv2 Llama graph
  -> executor.py: thin typed entry point into llama3_executor.py
  -> generator.py: vLLM construction, DP lanes, and dispatch
```

`model.py` composes reusable embedding, rotary, RMSNorm, attention, MLP,
LM-head, sampling, and TT collective modules. The 70B topology, precision, and
program policy remains model-owned.

## Executor composition

The local file retains `Llama33_70BExecutor`, its config, and its builder as
stable imports. `models/common/models/llama3_executor.py` supplies the 70B
family policy and composes `models/common/models/executor.py::ModelExecutor`.

The family-neutral owner composes paged-KV management, output reading,
prefill/decode runtimes, eager and traced execution, warmup coordination, and
ordered cleanup. The Llama family layer adds only:

- `SamplingState1D` and complete request history/remap state;
- device-sampling prefill serialization;
- the 70B trace-warmup sequence preference; and
- Q128 top-k priming before traced warmup and after eager warmup.

## vLLM, DP, and ownership

The generator builds one model/executor per submesh. DP1 uses one executor;
larger DP uses `LaneGroupExecutor`. `VLLMAdapter` normalizes server calls and
validates the physical KV cache. Each lane owns its TT resources and sampling
state; generator/adapter objects own no TT tensors.

## Tests

- `models/common/tests/models/llama33_70b/test_hf_adaptor.py`
- `models/common/tests/models/llama33_70b/test_model_profile.py`
- `models/common/tests/models/llama33_70b/test_demo_contract.py`
- `models/common/tests/models/llama33_70b/test_logits_oracle.py`
- `models/common/tests/models/llama33_70b/test_t3k_batched_prefill_correctness.py`
- `models/common/tests/models/llama33_70b/test_p150x4_smoke.py`
- `models/common/tests/demos/llama33_70b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
