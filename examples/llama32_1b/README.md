# Llama 3.2 1B with TTTv2

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Exercise the concrete small Llama tensor model, shared Llama executor, accuracy, performance, and DP lane routing.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `meta-llama/Llama-3.2-1B-Instruct`.
- Hugging Face revision: 9213176726f574b556790deb65791e0c5aa438b6.

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
| wormhole | T3K | 1x8 | 2 | 4 |
| wormhole | T3K | 1x8 | 1 | 8 |

Only these source-declared rows are candidates. No row has passing hardware evidence at the pinned extraction revision.

### Proven limits and features

- Demo cases cover active batch 1 or 32.
- standard/CI sequence budgets are 1024/2048; DP smokes reach 4096.
- TP1, TP2, and TP8 model paths are declared; TP4 DP lanes are deliberately rejected.
- N150 batch-32-ci at sequence 2048 is not enabled.
- traced prefill is Q128 on N150 and Q128/Q1024 on N300/T3K.
- Device sampling: Host and on-device paths exist; the demo default is host sampling.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `qualification/assets/reference_outputs/llama32_1b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative run using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.2-1B-Instruct python -m examples.llama32_1b.demo --case token-accuracy --optimizations performance
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=N150 pytest --collect-only -q tests/hardware/models/llama32_1b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/llama32_1b/test_demo.py)
- [Pinned support baseline](../../qualification/analysis/support/support_baseline.md)
- [Phase 3 boundary evidence](../../qualification/extraction/support_boundary.md)
- [Historical evidence ledger](../../qualification/analysis/support/hardware_evidence.csv)

### Known and unsupported gaps

- TP4 lanes.
- N150 batch-32-ci sequence 2048.
- any geometry not declared in support.json.
- No hardware evidence is attributable to the pinned extraction revision.

<!-- END GENERATED SUPPORT -->

This directory contains the Llama 3.2 1B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_adaptor.py: provider configuration, tokenizer, and weights
  -> model.py: TTTv2 Llama tensor graph
  -> executor.py: thin typed entry point into llama3_executor.py
  -> generator.py: vLLM boundary, lane construction, and dispatch
```

`model.py` composes reusable embedding, rotary, RMSNorm, attention, MLP,
LM-head, and optional sampling modules. Llama 3.2 1B architecture and tuning
remain model-owned.

## Executor composition

The model-local `executor.py` exports the historical
`Llama32_1BExecutor`/config/builder names. The implementation lives in
`models/common/models/llama3_executor.py`, which supplies Llama-family Q128
warmup policy and composes `models/common/models/executor.py::ModelExecutor`.

The shared executor owns:

- `PagedKVCacheManager` and `PageTableLayout`;
- `OutputReader`;
- `PrefillRuntime` and `DecodeRuntime`;
- `ProgramCompiler` and `EagerExecutor`;
- optional `TraceCompiler`/`TracedExecutor`;
- `WarmupCoordinator`; and
- terminal, ordered cleanup.

Llama 3.2 1B preserves its narrow pre-history request signatures, current
sampling behavior, and Q128 priming-before-prefill order. It does not silently
adopt the newer Llama-8B/70B `SamplingState1D` lifecycle in this structural
refactor.

## vLLM, DP, and ownership

`generator.py` builds one model/executor per lane and configures
`VLLMAdapter`. DP1 uses the executor directly; larger DP uses
`LaneGroupExecutor`. Executors own TT resources; the generator/adapter own
dispatch policy only.

## Tests

- `models/common/tests/models/llama32_1b/test_hf_adaptor.py`
- `models/common/tests/models/llama32_1b/test_batched_prefill_postprocess.py`
- `models/common/tests/models/llama32_1b/test_demo_warmup.py`
- `models/common/tests/demos/llama32_1b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
