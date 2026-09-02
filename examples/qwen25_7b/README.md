# Qwen2.5-7B with TTTv2

<!-- BEGIN GENERATED SUPPORT -->

## Standalone support contract

### Purpose

Run the concrete Qwen2.5-7B model on TP2 lanes for accuracy, throughput, determinism, and T3K DP4 smoke coverage.

### Status and checkpoint

- Status: **experimental**.
- Implementation: **concrete and runnable**; this is not evidence of qualification.
- Hugging Face ID: `Qwen/Qwen2.5-7B-Instruct`.
- Hugging Face revision: a09a35458c702b33eeacc393d103063234e8bc28.

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
| wormhole | N300 | 1x2 | 2 | 1 |
| wormhole | T3K | 1x8 | 2 | 4 |

Only these source-declared rows are candidates. No row has passing hardware evidence at the pinned extraction revision.

### Proven limits and features

- Demo cases cover active batch 1 or 32.
- standard/CI budgets are 1024/2048.
- ordinary execution is N300 TP2; T3K DP4 partitions into four TP2 lanes.
- N150 overflows the source L1 capacity guard.
- Device sampling: Host and on-device top-k paths exist; the demo default is host sampling.
- Trace support is case/topology-specific; use the exact hardware test parameters and capability manifest rather than extrapolating a generic trace claim.

### Checkpoint, cache, and offline requirements

- Obtain the authorized HF config, tokenizer/chat template, and complete checkpoint separately; weights are not shipped.
- Set a writable `TT_CACHE_PATH` (the example appends topology exactly once), or use the versioned standalone root selected by `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.
- Examples use built-in Transformers loading behavior; `trust_remote_code` is not enabled by default.
- For offline runs set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after populating the exact snapshot.
- Reference asset root: `qualification/assets/reference_outputs/qwen25_7b`.

### Install, run, and collect

```bash
python -m pip install -e '.[examples,test]'
```

Representative run using the first declared geometry:

```bash
HF_HOME=/path/to/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen2.5-7B-Instruct python -m examples.qwen25_7b.demo --case token-accuracy --optimizations performance
```

Collect the equivalent hardware gate without running it:

```bash
PYTHONPATH=src MESH_DEVICE=N300 pytest --collect-only -q tests/hardware/models/qwen25_7b/test_demo.py
```

### Correctness criterion

`token-accuracy` compares teacher-forced predictions with the committed `.refpt` and enforces the source-declared top-1/top-5 floor when one exists. Performance cases enforce a floor only when a complete source target exists; otherwise measurements are observational. Eval cases require cross-batch consistency, DP smokes require every lane to complete without invalid special-token output, and any explicit source capacity guard remains a skip rather than support evidence.

### Validation and evidence

- Last standalone hardware validation date: **none**.
- Validated git SHA: **none**.
- Evidence: **none attributable to pinned source revision `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`**.
- [Machine-readable manifest](support.json)
- [Hardware gate](../../tests/hardware/models/qwen25_7b/test_demo.py)
- [Pinned support baseline](../../qualification/analysis/support/support_baseline.md)
- [Phase 3 boundary evidence](../../qualification/extraction/support_boundary.md)
- [Historical evidence ledger](../../qualification/analysis/support/hardware_evidence.csv)

### Known and unsupported gaps

- N150.
- ordinary T3K/TG TP.
- DP factors other than T3K DP4.
- any geometry not declared in support.json.
- No hardware evidence is attributable to the pinned extraction revision.

<!-- END GENERATED SUPPORT -->

This directory contains the Qwen2.5-7B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_adaptor.py: provider configuration, tokenizer, and weights
  -> model.py: Qwen2.5 tensor graph composed from TTTv2 modules
  -> executor.py: thin typed entry point into qwen2_executor.py
  -> generator.py: vLLM boundary, DP composition, and dispatch
```

`model.py` owns Qwen2.5 architecture/tuning while composing reusable embedding,
rotary, RMSNorm, attention, MLP, LM-head, optional sampling, and collective
modules.

## Executor composition

The model-local file preserves the Qwen2.5-7B executor/config/builder imports.
The implementation is shared with Qwen2 through
`models/common/models/qwen2_executor.py`, which configures the family-neutral
`ModelExecutor`.

The common owner composes paged KV, output reading, prefill/decode runtimes,
eager/trace execution, warmup, and cleanup. The family layer retains the
7B Q128 top-k warmup order and narrow pre-history request contract. It does not
silently adopt Qwen3 stateful sampling behavior.

## vLLM, DP, and ownership

`generator.py` builds one executor per lane, uses `VLLMAdapter` for external
normalization/cache validation, and composes multiple lanes with
`LaneGroupExecutor`. TT resources remain lane-owned.

## Tests

- `models/common/tests/models/qwen25_7b/test_hf_adaptor.py`
- `models/common/tests/models/qwen25_7b/test_demo_contract.py`
- `models/common/tests/demos/qwen25_7b/demo.py`
- `models/common/tests/models/test_qwen2_executor_family.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
