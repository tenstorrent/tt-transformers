# Examples and model support

All twelve examples are concrete and runnable, but **experimental**. No example is qualified, every `support.json` retains null validation metadata and an empty evidence list, and no hardware pass is attributable to the pinned source revision.

Install the candidate environment with:

```bash
python -m pip install -e '.[examples,test]'
```

| Example | Status | Implementation | HF ID | Revision | Declared candidate geometry | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| [deepseek_r1_distill_qwen_14b](deepseek_r1_distill_qwen_14b/README.md) | experimental | concrete; runnable CLI | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `1df8507178afcc1bef68cd8c393f61a886323761` | wormhole N300 mesh=1x2 TP2/DP1; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP4/DP2; wormhole T3K mesh=1x8 TP2/DP4 | none at pinned SHA |
| [llama32_1b](llama32_1b/README.md) | experimental | concrete; runnable CLI | `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | wormhole N150 mesh=1x1 TP1/DP1; wormhole N300 mesh=1x2 TP2/DP1; wormhole N300 mesh=1x2 TP1/DP2; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP2/DP4; wormhole T3K mesh=1x8 TP1/DP8 | none at pinned SHA |
| [llama32_3b](llama32_3b/README.md) | experimental | concrete; runnable CLI | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | wormhole N150 mesh=1x1 TP1/DP1; wormhole N300 mesh=1x2 TP2/DP1; wormhole N300 mesh=1x2 TP1/DP2; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP2/DP4; wormhole T3K mesh=1x8 TP1/DP8 | none at pinned SHA |
| [llama33_70b](llama33_70b/README.md) | experimental | concrete; runnable CLI | `meta-llama/Llama-3.3-70B-Instruct` | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | wormhole T3K mesh=1x8 TP8/DP1; blackhole P150_X4 required logical target; code accepts physical P300_X2 alias with distinct provenance mesh=1x4 TP4/DP1 | none at pinned SHA |
| [llama3_8b](llama3_8b/README.md) | experimental | concrete; runnable CLI | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | wormhole N150 mesh=1x1 TP1/DP1; wormhole N300 mesh=1x2 TP2/DP1; wormhole N300 mesh=1x2 TP1/DP2; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP4/DP2; wormhole T3K mesh=1x8 TP2/DP4; wormhole T3K mesh=1x8 TP1/DP8; blackhole P150 mesh=1x1 TP1/DP1; blackhole P150_X4 mesh=1x4 TP4/DP1; blackhole P150_X4 mesh=1x4 TP1/DP4; blackhole physical_P300 mesh=1x2 TP1/DP2 | none at pinned SHA |
| [mistral_7b](mistral_7b/README.md) | experimental | concrete; runnable CLI | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | wormhole N150 mesh=1x1 TP1/DP1; wormhole N300 mesh=1x2 TP2/DP1; wormhole N300 mesh=1x2 TP1/DP2; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP1/DP8 | none at pinned SHA |
| [phi4](phi4/README.md) | experimental | concrete; runnable CLI | `microsoft/phi-4` | `187ef0342fff0eb3333be9f00389385e95ef0b61` | wormhole N300 mesh=1x2 TP2/DP1; wormhole T3K mesh=1x8 TP2/DP4 | none at pinned SHA |
| [qwen25_72b](qwen25_72b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen2.5-72B-Instruct` | `495f39366efef23836d0cfae4fbe635880d2be31` | wormhole T3K mesh=1x8 TP8/DP1 | none at pinned SHA |
| [qwen25_7b](qwen25_7b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | wormhole N300 mesh=1x2 TP2/DP1; wormhole T3K mesh=1x8 TP2/DP4 | none at pinned SHA |
| [qwen25_coder_32b](qwen25_coder_32b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen2.5-Coder-32B-Instruct` | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` | wormhole T3K mesh=1x8 TP8/DP1 | none at pinned SHA |
| [qwen2_7b](qwen2_7b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen2-7B-Instruct` | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | wormhole N300 mesh=1x2 TP2/DP1; wormhole T3K mesh=1x8 TP2/DP4 | none at pinned SHA |
| [qwen3_32b](qwen3_32b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | wormhole T3K mesh=1x8 TP8/DP1; blackhole P150_X4 required logical target; code accepts physical P300_X2 alias with distinct provenance mesh=1x4 TP4/DP1 | none at pinned SHA |

## Current final-SHA subset evidence

The [centralized hardware ledger](../qualification/evidence/hardware/2883a949860d749adc2ed1af5525b27a9a547505/index.json) records 34 passing nodes at final tested standalone SHA `2883a949860d749adc2ed1af5525b27a9a547505`. All 23 executed module nodes, the official strict W6 runtime node, all three smoke nodes, and all seven end-to-end/token-accuracy nodes passed. It covers reusable modules and narrow subsets of five model families:

- Llama-3.2-1B: Wormhole N150 TP1/DP1 and N300 TP2/DP1 token accuracy passed; its N300 DP and T3K variants were not exercised.
- Qwen2.5-7B: Wormhole N300 TP2/DP1 token accuracy passed; its T3K variant was not exercised.
- Qwen2.5-Coder-32B: a Wormhole T3K TP8/DP1 prefill smoke passed; no end-to-end accuracy node was run.
- Qwen3-32B: Wormhole T3K TP8/DP1 and physical Blackhole P150_X4 TP4/DP1 token accuracy passed, as did the physical P150_X4 one-layer smoke; the accepted P300_X2 code alias was not exercised.
- Llama-3.3-70B: Wormhole T3K TP8/DP1 and physical Blackhole P150_X4 TP4/DP1 token accuracy passed, as did the physical P150_X4 one-layer smoke; the strict T3K W6 trace-order node passed all four orders at unchanged thresholds. The accepted P300_X2 code alias was not exercised.

This is hash-bound diagnostic migration evidence, not qualification of an example or its whole declared geometry. The seven single-card P150 module nodes and Llama-3.1-8B P150 token-accuracy node were not run, and P150_X4 evidence does not cover P150. The other seven model families have no final-SHA model-family node; every geometry, parallelism variant, hardware alias, and workload absent from the ledger remains uncovered and unqualified.

The separate [accuracy-TTFT diagnostic](../qualification/evidence/diagnostics/2883a949860d749adc2ed1af5525b27a9a547505/accuracy-ttft/summary.json) is noncanonical: TTFT passed 4/4 at 86.7–87.2 ms, while throughput produced `performance_floor_failure` in all 4/4 cells. It is excluded from the canonical pass count and does not change the correctness ledger or any example's experimental status.

The superseded [`b24eabe35c8f2c73f45493da40e5a6351eb0ec2d` bundle](../qualification/evidence/hardware/b24eabe35c8f2c73f45493da40e5a6351eb0ec2d/index.json) and earlier [`ba7abefba4484689c953ac53fe8810322db1d184` bundle](../qualification/evidence/hardware/ba7abefba4484689c953ac53fe8810322db1d184/index.json) remain preserved as historical evidence and do not transfer to the current candidate. The relaxed one-order W6 run at `d7677f822356e839f707a6447fd0abc89e620d56` remains non-qualifying history associated with the superseded strict failure.

Each model README contains exact run/collection commands, checkpoint/cache requirements, proven source limits, unsupported configurations, and evidence links. Machine-readable truth lives in each `support.json`; validate all documentation with:

```bash
python -B qualification/tools/validate_support_docs.py
```
