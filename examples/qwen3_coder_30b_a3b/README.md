# Qwen3-Coder-30B-A3B with TTTv2

## Generate text

[`demo.py`](demo.py) loads the public model with `hf_generator.from_pretrained`,
formats a chat with `model.tokenizer.apply_chat_template`, calls
`model.generate`, prints the decoded continuation, and cleans up the model. The
checkpoint's generation config sets `repetition_penalty=1.05`, which HF
`generate()` does not implement and refuses; see the known gaps below.

With the checkpoint cached locally, run:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MESH_DEVICE=P150x4 python -m examples.qwen3_coder_30b_a3b.demo \
  --prompt "Write a Python function that checks whether a string is a palindrome." \
  --max-new-tokens 64 --max-seq-len 2048
```

The inputs stay as CPU PyTorch tensors; the model handles TT transfers and KV
storage. `MESH_DEVICE` selects the caller-owned mesh. Configure `TT_CACHE_PATH`
when using an existing writable TT model cache, as described below.

[`benchmark.py`](benchmark.py) contains the token-accuracy case, the one- and
32-user performance cases, and TTFT at 128, 1024 and 4096 prompt tokens.

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Run Qwen3-Coder-30B-A3B (48 layers, 128 experts, top-8) on a declared Blackhole P150x4 mesh, with attention tensor-parallel and the routed experts expert-parallel across the four devices, including token accuracy, trace and one- and 32-user performance.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `Qwen/Qwen3-Coder-30B-A3B-Instruct`.
- Hugging Face revision: b2cff646eb4bb1d68355c01b18ae02e7cf42d120.

### Candidate software tuple

This is the declared qualification candidate, not a passing verdict:

- `tt-transformers==2.0.0.dev0`
- `ttnn==0.79.0`
- Python `3.10, 3.12`
- `torch==2.11.0`
- `transformers==5.12.1`

### Declared hardware geometry

| Architecture | Physical SKU/system declaration | Mesh | TP | DP |
| --- | --- | --- | ---: | ---: |
| blackhole | P150_X4 required logical target; code accepts physical P300_X2 alias with distinct provenance | 1x4 | 4 | 1 |

Only these source-declared rows are candidates. No row has passing hardware evidence at the pinned extraction revision.

### Proven limits and features

- Attention is tensor-parallel over the four devices; the 128 routed experts are expert-parallel, 32 per device.
- One-user decode runs the fused MoE decode; more users run the routed sparse MoE path.
- Prefill of 128 tokens or more runs the expert-sorted MoE prefill, in slices of at most 4096 tokens.
- Benchmark cases cover token accuracy and 1- and 32-user performance at max_seq_len 2048.
- Device sampling: Host and on-device top-k paths exist.
- Trace support is case/topology-specific; use the exact hardware test parameters rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `tests/assets/reference_outputs/qwen3_coder_30b_a3b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative benchmark using the declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=P150x4 python -m examples.qwen3_coder_30b_a3b.benchmark --case token-accuracy --optimizations performance
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=P150x4 pytest --collect-only -q tests/hardware/models/qwen3_coder_30b_a3b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` (generated from the pinned checkpoint in bf16) and enforces the source-declared top-1/top-5 floor. Performance cases are observational: they report TTFT and decode throughput without a floor.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to the pinned source revision**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/qwen3_coder_30b_a3b/test_demo.py)
- [Support matrix](../../SUPPORT.md)
- [Validation summary](../../docs/validation.md)

### Known and unsupported gaps

- Decode for more than one user runs the routed sparse MoE path, not the fused MoE decode.
- HF generate() refuses the checkpoint's repetition_penalty=1.05, which it does not implement (as for Qwen2.5 Coder 32B and Qwen2.5 7B); vLLM serving applies it.
- P150 and P300 single/two-die model paths.
- DP greater than 1.
- any geometry not declared in support.json.
- Current regression evidence does not qualify the complete declared model contract.

<!-- END GENERATED SUPPORT -->

## Product path

```text
Qwen/Qwen3-Coder-30B-A3B-Instruct
  -> hf_generator.py: checkpoint/tokenizer loading, executor construction, and HF generation
  -> model.py: the Qwen3 decoder with an MoE1D layer in place of the dense MLP
  -> vllm_generator.py: vLLM construction and dispatch through the Qwen3-32B generator facade
```

The decoder, executor and vLLM facade are Qwen3-32B's; this package owns the
checkpoint geometry, the MoE layer and the four-device expert-parallel layout.
