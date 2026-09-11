# Phi-4 with TTTv2

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Run the concrete Phi-4 tensor model through direct executor accuracy, performance, determinism, and TP2 lane smoke paths.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `microsoft/phi-4`.
- Hugging Face revision: 187ef0342fff0eb3333be9f00389385e95ef0b61.

### Candidate software tuple

This is the declared qualification candidate, not a passing verdict:

- `tt-transformers==2.0.0.dev0`
- `ttnn==0.77.0`
- Python `3.10, 3.12`
- `torch==2.11.0`
- `transformers==5.12.1`

### Declared hardware geometry

| Architecture | Physical SKU/system declaration | Mesh | TP | DP |
| --- | --- | --- | ---: | ---: |
| wormhole | N300 | 1x2 | 2 | 1 |
| wormhole | T3K | 1x8 | 2 | 4 |

Only these source-declared rows are candidates. No row has passing hardware evidence at the pinned extraction revision.

### Proven limits and features

- Demo cases cover active batch 1 or 32.
- standard/CI budgets are 1024/2048; DP smoke metadata reaches 4096.
- ordinary execution is N300 TP2; physical T3K is admitted only as four TP2 lanes.
- N150 exceeds the source L1 capacity guard.
- Device sampling: Host and on-device top-k paths exist.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `tests/assets/reference_outputs/phi4`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative run using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=N300 HF_MODEL=microsoft/phi-4 python -m examples.phi4.demo --case token-accuracy --optimizations performance
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=N300 pytest --collect-only -q tests/hardware/models/phi4/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/phi4/test_demo.py)
- [Support matrix](../../SUPPORT.md)
- [Validation summary](../../docs/validation.md)

### Known and unsupported gaps

- N150.
- ordinary T3K/TG tensor parallelism.
- DP layouts other than T3K DP4/TP2.
- any geometry not declared in support.json.
- Current regression evidence does not qualify the complete declared model contract.

<!-- END GENERATED SUPPORT -->

This directory is the model-owned TTTv2 path for Microsoft Phi-4.

It intentionally demonstrates direct executor construction from
`src/tt_transformers/llm_runtime`. It is not part of the Llama/Qwen executor
consolidation and does not use `src/tt_transformers/models/executor.py`.

## Product path

```text
Hugging Face checkpoint
  -> hf_generator.py: checkpoint/tokenizer loading, executor construction, and HF generation
  -> model.py: Phi-4 tensor graph composed from TTTv2 modules
  -> vllm_generator.py: vLLM construction, DP composition, and dispatch
```

## Files

| File | Responsibility |
| --- | --- |
| `hf_generator.py` | Resolve the Phi-4 provider configuration/tokenizer and build runtime metadata; construct the executor and expose `.generate()` |
| `weight_utils.py` | Convert and map HF weights into the model-owned layout |
| `model.py` | Build and execute the TTTv2 Phi-4 transformer graph |
| `vllm_generator.py` | Build lanes, configure `VLLMAdapter`, and expose serving dispatch |

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

Phi-specific provider mapping, architecture values, precision, and tuning
remain model-owned.

## Direct executor composition

`Phi4Executor` directly constructs:

```text
Phi4Transformer
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

This direct construction path is intentionally retained as an example for new
models whose orchestration does not belong in the shared Llama or Qwen family
composition. It reuses focused runtime modules without adding a protocol,
profile, or executor subclass hierarchy.

The executor owns the physical KV cache, compile/trace artifacts, output
leases, sampling resources, and cleanup. Cleanup is terminal, ordered,
retryable, and idempotent.

## vLLM and data parallelism

`Phi4Generator` builds one executor per lane. DP1 uses the executor directly;
larger DP configurations use `LaneGroupExecutor`. `VLLMAdapter` performs
server-boundary normalization and KV-cache validation.

## Tests

Relevant entry points include:

- `tests/models/phi4/test_hf_adaptor.py`
- `tests/models/phi4/test_demo_contract.py`
- `tests/hardware/models/phi4/test_demo.py`
- `tests/llm_runtime/test_executor_integration.py`
