# DeepSeek-R1-Distill-Qwen-14B with TTTv2

## Generate text

[`demo.py`](demo.py) loads the public model with `hf_generator.from_pretrained`,
formats a chat with `model.tokenizer.apply_chat_template`, calls
`model.generate`, prints the decoded continuation, and cleans up the model.

With the checkpoint cached locally, run:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MESH_DEVICE=N300 python -m examples.deepseek_r1_distill_qwen_14b.demo \
  --hf-model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
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

Run the concrete 14B distilled-Qwen tensor model through direct eager/traced execution, teacher-forced accuracy, throughput, and DP smoke paths.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.
- Hugging Face revision: 1df8507178afcc1bef68cd8c393f61a886323761.

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
| wormhole | T3K | 1x8 | 8 | 1 |
| wormhole | T3K | 1x8 | 4 | 2 |
| wormhole | T3K | 1x8 | 2 | 4 |

Only these source-declared rows are candidates. No row has passing hardware evidence at the pinned extraction revision.

### Proven limits and features

- Benchmark cases cover active batch 1 or 32.
- standard/CI sequence budgets are 1024/2048; retained DP smokes reach 4096.
- TP1 is rejected; supported model lanes are TP2, TP4, or TP8.
- N300 accuracy eval-32 and batch-32-ci are explicitly DRAM-infeasible.
- Device sampling: Host and on-device top-k paths exist; the benchmark default is on_device_topk.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `tests/assets/reference_outputs/deepseek_r1_distill_qwen_14b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative benchmark using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=N300 HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B python -m examples.deepseek_r1_distill_qwen_14b.benchmark --case token-accuracy --optimizations performance
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=N300 pytest --collect-only -q tests/hardware/models/deepseek_r1_distill_qwen_14b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/deepseek_r1_distill_qwen_14b/test_demo.py)
- [Support matrix](../../SUPPORT.md)
- [Validation summary](../../docs/validation.md)

### Known and unsupported gaps

- N150/TP1.
- N300 accuracy eval-32 and batch-32-ci.
- any geometry not declared in support.json.
- Current regression evidence does not qualify the complete declared model contract.

<!-- END GENERATED SUPPORT -->

This directory is a model-owned TTTv2 product path for
`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.

It intentionally demonstrates direct executor construction from the reusable
common LLM runtime. It is not part of the shared Llama/Qwen executor
consolidation and does not use `src/tt_transformers/models/executor.py`.

## Product path

```text
Hugging Face checkpoint
  -> hf_generator.py: checkpoint/tokenizer loading, executor construction, and HF generation
  -> model.py: DeepSeek-Qwen tensor graph composed from TTTv2 modules
  -> vllm_generator.py: vLLM construction, DP composition, and dispatch
```

## Files

| File | Responsibility |
| --- | --- |
| `hf_generator.py` | Resolve the HF checkpoint/tokenizer and construct model/runtime configuration; construct the executor and expose `.generate()` |
| `weight_utils.py` | Convert and map provider weights into the model-owned layout |
| `model.py` | Build and execute the DeepSeek-Qwen transformer graph |
| `vllm_generator.py` | Build one or more lanes, configure `VLLMAdapter`, and expose the vLLM-facing API |

## Tensor-module composition

`model.py` composes the model from reusable TTTv2 modules, including:

- `Embedding1D`
- `RotarySetup1D`
- `RMSNorm1D`
- `Attention1D`
- `MLP1D`
- `LMHead1D`
- optional `Sampling1D`
- common TT collective helpers

The model owns DeepSeek/Qwen architecture and tuning policy. The reusable
modules own their tensor programs and lazy weights.

## Direct executor composition

`DeepSeekR1Qwen14BExecutor` directly constructs:

```text
DeepSeekR1Qwen14B model
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

This direct pattern is supported when a model has genuinely distinct
orchestration or has not been deliberately migrated to a shared model-layer
executor. It still reuses the same `src/tt_transformers/llm_runtime` mechanics; it
does not copy those runtime implementations.

The executor owns paged-KV tensors, eager/trace registries, output leases,
sampling buffers, and deterministic cleanup. Cleanup terminalizes the lane,
drains external decode outputs, releases runtime transients and traces, then
releases sampling and KV resources with retryable failure reporting.

## vLLM and data parallelism

`DeepSeekR1Qwen14BGenerator` builds one executor per lane. DP1 uses the lane
directly; DP greater than one wraps lanes in `LaneGroupExecutor`.

`VLLMAdapter` normalizes vLLM calls and validates the physical KV-cache shape.
The generator owns dispatch policy but no TT tensor resources.

## Tests

Relevant entry points include:

- `tests/models/deepseek_r1_distill_qwen_14b/test_hf_adaptor.py`
- `tests/models/deepseek_r1_distill_qwen_14b/test_demo_contract.py`
- `tests/models/deepseek_r1_distill_qwen_14b/test_prefill_last_token_contract.py`
- `tests/hardware/models/deepseek_r1_distill_qwen_14b/test_demo.py`
- `tests/llm_runtime/test_executor_integration.py`
