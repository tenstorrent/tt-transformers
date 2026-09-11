# Qwen2-7B with TTTv2

## Generate text

[`demo.py`](demo.py) loads the public model with `hf_generator.from_pretrained`,
formats a chat with `model.tokenizer.apply_chat_template`, calls
`model.generate`, prints the decoded continuation, and cleans up the model.

With the checkpoint cached locally, run:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MESH_DEVICE=N300 python -m examples.qwen2_7b.demo \
  --hf-model "Qwen/Qwen2-7B-Instruct" \
  --prompt "Explain paged attention briefly." \
  --max-new-tokens 40 --max-seq-len 2048
```

The inputs stay as CPU PyTorch tensors; the model handles TT transfers and KV
storage. `MESH_DEVICE` selects the caller-owned mesh. Configure `TT_CACHE_PATH`
when using an existing writable TT model cache, as described below.

[`benchmark.py`](benchmark.py) contains the accuracy, performance, tracing, and
DP workloads. Run its `--case` / `--optimizations` commands for those checks.

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Run the concrete Qwen2-7B model on TP2 lanes for accuracy, throughput, determinism, and T3K DP4 smoke coverage.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `Qwen/Qwen2-7B-Instruct`.
- Hugging Face revision: f2826a00ceef68f0f2b946d945ecc0477ce4450c.

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

- Benchmark cases cover active batch 1 or 32.
- standard/CI budgets are 1024/2048.
- ordinary execution is N300 TP2; T3K DP4 partitions into four TP2 lanes.
- N150 and ordinary TP8 fail source capacity/head-divisibility guards.
- Device sampling: Host and on-device top-k paths exist; the benchmark default is host sampling.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `tests/assets/reference_outputs/qwen2_7b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative benchmark using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen2-7B-Instruct python -m examples.qwen2_7b.benchmark --case token-accuracy --optimizations performance
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=N300 pytest --collect-only -q tests/hardware/models/qwen2_7b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/qwen2_7b/test_demo.py)
- [Support matrix](../../SUPPORT.md)
- [Validation summary](../../docs/validation.md)

### Known and unsupported gaps

- N150.
- ordinary T3K/TG TP.
- DP factors other than T3K DP4.
- any geometry not declared in support.json.
- Current regression evidence does not qualify the complete declared model contract.

<!-- END GENERATED SUPPORT -->

This directory contains the Qwen2-7B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_generator.py: checkpoint/tokenizer loading, executor construction, and HF generation
  -> model.py: Qwen2 tensor graph composed from TTTv2 modules
  -> vllm_generator.py: vLLM boundary, DP composition, and dispatch
```

The model composes reusable embedding, rotary, RMSNorm, attention, MLP,
LM-head, optional sampling, and TT collective modules. Qwen2 architecture and
device tuning remain model-owned.

## Executor composition

The local `hf_generator.py` retains the public Qwen2 class/config/builder names.
`src/tt_transformers/models/qwen2_executor.py` supplies Qwen2-family policy and
composes `src/tt_transformers/models/executor.py::ModelExecutor`.

The shared owner constructs paged-KV management, output reading,
prefill/decode runtimes, eager/trace execution, warmup, and ordered cleanup.
The family layer preserves Qwen2-7B's narrow request signatures and Q128
top-k tile-end priming before traced warmup and after eager warmup. This
structural refactor does not add Qwen3 native sampling state.

## vLLM, DP, and ownership

The generator builds one executor per lane and configures `VLLMAdapter`. DP1
uses one lane; larger DP uses `LaneGroupExecutor`. Executors own TT resources;
the generator and adapter own dispatch policy only.

## Tests

- `tests/models/qwen2_7b/test_hf_adaptor.py`
- `tests/models/qwen2_7b/test_demo_contract.py`
- `tests/hardware/models/qwen2_7b/test_demo.py`
- `tests/models/test_qwen2_executor_family.py`
- `tests/llm_runtime/test_executor_integration.py`
- `tests/llm_runtime/test_model_executor.py`
