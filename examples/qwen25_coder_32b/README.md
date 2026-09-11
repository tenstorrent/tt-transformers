# Qwen2.5-Coder-32B with TTTv2

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Run the concrete 32B coder model on T3K TP8, with end-to-end and focused smoke/PCC diagnostic CLIs.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `Qwen/Qwen2.5-Coder-32B-Instruct`.
- Hugging Face revision: 381fc969f78efac66bc87ff7ddeadb7e73c218a7.

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
| wormhole | T3K | 1x8 | 8 | 1 |

Only these source-declared rows are candidates. No row has passing hardware evidence at the pinned extraction revision.

### Proven limits and features

- End-to-end cases cover active batch 1 or 32.
- standard/CI sequence budgets are 1024/2048.
- weights and KV require T3K TP8.
- all retained DP cases capacity-skip.
- focused smoke prefill uses sequence 128 by default.
- Device sampling: Host and on-device top-k paths exist; the performance path defaults to on_device_topk.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `tests/assets/reference_outputs/qwen25_coder_32b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative run using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct python -m examples.qwen25_coder_32b.demo --case token-accuracy --optimizations performance
```

Focused smoke:

```bash
MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct python -m examples.qwen25_coder_32b.smoke --case qwen25-coder-32b-prefill-smoke
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=T3K pytest --collect-only -q tests/hardware/models/qwen25_coder_32b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/qwen25_coder_32b/test_demo.py)
- [Support matrix](../../SUPPORT.md)
- [Validation summary](../../docs/validation.md)

### Known and unsupported gaps

- non-T3K meshes.
- DP greater than 1.
- any geometry not declared in support.json.
- Current regression evidence does not qualify the complete declared model contract.

<!-- END GENERATED SUPPORT -->

This directory contains the Qwen2.5-Coder-32B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_generator.py: checkpoint/tokenizer loading, executor construction, and HF generation
  -> model.py: Qwen2.5-Coder tensor graph
  -> vllm_generator.py: vLLM construction, DP composition, and dispatch
```

The model uses reusable embedding, rotary, RMSNorm, attention, MLP, LM-head,
optional sampling, and collective modules. Coder architecture, dimensions,
precision, and tuning remain model-owned.

## Executor composition

The lifecycle is shared through `src/tt_transformers/models/qwen2_executor.py`, which
configures `src/tt_transformers/models/executor.py::ModelExecutor`. The common owner
constructs paged KV, output reading, prefill/decode runtimes, eager/traced
execution, warmup, and ordered cleanup.

The model-local file retains its historical executor/config/builder names,
eager/traced compatibility wrappers, compatibility config construction,
direct-run helpers, concrete last-token slicing, and TT deallocation behavior.
It uses plain coordinator prefill warmup and does not adopt Qwen3 native
sampling state during this structural extraction.

## vLLM, DP, and ownership

The generator owns vLLM normalization/dispatch and constructs one executor per
lane. `LaneGroupExecutor` provides DP fanout. Lane executors exclusively own TT
resources and cleanup.

## Tests

- `tests/models/qwen25_coder_32b/test_hf_adaptor.py`
- `tests/models/qwen25_coder_32b/test_model_runtime_surface.py`
- `tests/models/qwen25_coder_32b/test_demo_contract.py`
- `tests/hardware/models/qwen25_coder_32b/test_demo.py`
- `tests/models/test_qwen2_executor_family.py`
- `tests/llm_runtime/test_executor_integration.py`
- `tests/llm_runtime/test_model_executor.py`
