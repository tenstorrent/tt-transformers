# Mistral-7B with TTTv2

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Run the concrete Mistral tensor model through direct executor accuracy, throughput, determinism, and DP smoke paths.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `mistralai/Mistral-7B-Instruct-v0.3`.
- Hugging Face revision: c170c708c41dac9275d15a8fff4eca08d52bab71.

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
| wormhole | N150 | 1x1 | 1 | 1 |
| wormhole | N300 | 1x2 | 2 | 1 |
| wormhole | N300 | 1x2 | 1 | 2 |
| wormhole | T3K | 1x8 | 8 | 1 |
| wormhole | T3K | 1x8 | 1 | 8 |

Only these source-declared rows are candidates. No row has passing hardware evidence at the pinned extraction revision.

### Proven limits and features

- Demo cases cover active batch 1 or 32.
- standard/CI budgets are 1024/2048; retained DP smokes reach 4096.
- N300 DP2 and T3K DP8 use one-device lanes; other retained DP factors skip.
- Device sampling: Host and on-device top-k paths exist; the demo default is host sampling.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `qualification/assets/reference_outputs/mistral_7b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative run using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=N150 HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 python -m examples.mistral_7b.demo --case token-accuracy --optimizations performance
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=N150 pytest --collect-only -q tests/hardware/models/mistral_7b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/mistral_7b/test_demo.py)
- [Pinned support baseline](../../qualification/analysis/support/support_baseline.md)
- [Phase 3 boundary evidence](../../qualification/extraction/support_boundary.md)
- [Historical evidence ledger](../../qualification/analysis/support/hardware_evidence.csv)

### Known and unsupported gaps

- DP layouts other than the declared one-device lanes.
- any geometry not declared in support.json.
- No hardware evidence is attributable to the pinned extraction revision.

<!-- END GENERATED SUPPORT -->

This directory is the model-owned TTTv2 path for the Mistral-7B family.

It intentionally demonstrates direct executor construction from
`models/common/llm_runtime`. It is not part of the Llama/Qwen executor
consolidation and does not use `models/common/models/executor.py`.

## Product path

```text
Hugging Face checkpoint
  -> hf_adaptor.py: provider metadata, tokenizer, and weight conversion
  -> model.py: Mistral tensor graph composed from TTTv2 modules
  -> executor.py: direct composition of common runtime owners for one lane
  -> generator.py: vLLM construction, DP composition, and dispatch
```

## Files

| File | Responsibility |
| --- | --- |
| `hf_adaptor.py` | Resolve provider configuration/tokenizer and construct the product model |
| `weight_utils.py` | Convert and map provider weights |
| `model.py` | Build and execute the TTTv2 Mistral transformer graph |
| `executor.py` | Directly compose one execution lane and own its resources |
| `generator.py` | Build lanes, configure the vLLM boundary, and select eager/traced execution |

## Tensor-module composition

`model.py` composes:

- `Embedding1D`
- `RotarySetup1D`
- `RMSNorm1D`
- `Attention1D`
- `MLP1D`
- `LMHead1D`
- optional `Sampling1D`
- common TT collective helpers

Mistral-specific attention, RoPE, precision, and device-tuning policy remains
model-owned.

## Direct executor composition

`Mistral7BExecutor` directly constructs:

```text
Mistral7B model
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

This is a supported alternative to the shared model-layer `ModelExecutor`.
Models with distinct orchestration may compose the focused `llm_runtime`
modules directly without subclassing or modifying a universal executor.

The lane executor owns paged KV, compile/trace registries, output leases,
sampling buffers, and deterministic cleanup. The generator owns orchestration
only and does not own TT tensors.

## vLLM and data parallelism

`Mistral7BGenerator` builds one model/executor per lane and uses
`LaneGroupExecutor` when `tt_data_parallel > 1`. `VLLMAdapter` normalizes the
server boundary and validates the vLLM-selected KV-cache specification.

## Tests

Relevant entry points include:

- `models/common/tests/models/mistral_7b/test_hf_adaptor.py`
- `models/common/tests/models/mistral_7b/test_demo_contract.py`
- `models/common/tests/models/mistral_7b/test_prefill_last_token_contract.py`
- `models/common/tests/demos/mistral_7b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
