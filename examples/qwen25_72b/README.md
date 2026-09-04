# Qwen2.5-72B with TTTv2

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Run the concrete 72B Qwen2.5 model on its full T3K TP8 lane for accuracy, performance, and determinism characterization.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `Qwen/Qwen2.5-72B-Instruct`.
- Hugging Face revision: 495f39366efef23836d0cfae4fbe635880d2be31.

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

- Demo cases cover active batch 1 or 32.
- standard/CI sequence budgets are 1024/2048.
- weights and KV require T3K TP8.
- all retained DP cases capacity-skip.
- Device sampling: Host and on-device top-k paths exist; the performance path defaults to on_device_topk.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `tests/assets/reference_outputs/qwen25_72b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative run using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen2.5-72B-Instruct python -m examples.qwen25_72b.demo --case token-accuracy --optimizations performance
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=T3K pytest --collect-only -q tests/hardware/models/qwen25_72b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/qwen25_72b/test_demo.py)
- [Support matrix](../../SUPPORT.md)
- [Validation summary](../../docs/validation.md)

### Known and unsupported gaps

- non-T3K meshes.
- DP greater than 1.
- any geometry not declared in support.json.
- Current regression evidence does not qualify the complete declared model contract.

<!-- END GENERATED SUPPORT -->

This directory contains the large Qwen2.5-72B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_adaptor.py: provider metadata, tokenizer, and weights
  -> model.py: multi-device Qwen2.5 tensor graph
  -> executor.py: thin family builder plus concrete direct-run helpers
  -> generator.py: vLLM construction, DP composition, and dispatch
```

The tensor model composes reusable embedding, rotary, RMSNorm, attention, MLP,
LM-head, optional sampling, and collective modules while retaining 72B
topology, precision, and program policy locally.

## Executor composition

The public 72B executor/config/builder remains importable from this directory.
Its lifecycle is supplied by `src/tt_transformers/models/qwen2_executor.py` over
`src/tt_transformers/models/executor.py::ModelExecutor`.

The shared owner composes paged-KV management, prefill/decode runtimes,
eager/trace compilation, output handling, warmup, and cleanup. Qwen2.5-72B uses
plain coordinator prefill warmup rather than the 7B Q128 priming hook.

Concrete `run_prefill`, `run_decode`, `run_lm_head`, last-token slicing, and
deallocation helpers remain model-local because they invoke the concrete
tensor model.

## vLLM, DP, and ownership

The generator builds one executor per lane, configures `VLLMAdapter`, and uses
`LaneGroupExecutor` for DP. Each lane owns its KV tensors, compile/trace
artifacts, output leases, sampling buffers, and terminal cleanup.

## Tests

- `tests/models/qwen25_72b/test_hf_adaptor.py`
- `tests/models/qwen25_72b/test_model_runtime_surface.py`
- `tests/models/qwen25_72b/test_demo_contract.py`
- `tests/hardware/models/qwen25_72b/test_demo.py`
- `tests/models/test_qwen2_executor_family.py`
- `tests/llm_runtime/test_executor_integration.py`
- `tests/llm_runtime/test_model_executor.py`
