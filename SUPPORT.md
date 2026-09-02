# TT Transformers support matrix

Concrete implementation and a runnable CLI mean the source is present and callable. They do **not** mean qualified. Every current row is experimental because no hardware result is attributable to the pinned extraction SHA.

Candidate software tuple: `tt-transformers==0.1.0.dev0`, `ttnn==0.77.0`, Python 3.10/3.12, `torch==2.11.0`, `transformers==5.12.1`.

| Example | Status | Implementation | HF ID | Revision | Declared candidate geometry | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| [deepseek_r1_distill_qwen_14b](examples/deepseek_r1_distill_qwen_14b/README.md) | experimental | concrete; runnable CLI | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `1df8507178afcc1bef68cd8c393f61a886323761` | wormhole N300 mesh=1x2 TP2/DP1; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP4/DP2; wormhole T3K mesh=1x8 TP2/DP4 | none at pinned SHA |
| [llama32_1b](examples/llama32_1b/README.md) | experimental | concrete; runnable CLI | `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | wormhole N150 mesh=1x1 TP1/DP1; wormhole N300 mesh=1x2 TP2/DP1; wormhole N300 mesh=1x2 TP1/DP2; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP2/DP4; wormhole T3K mesh=1x8 TP1/DP8 | none at pinned SHA |
| [llama32_3b](examples/llama32_3b/README.md) | experimental | concrete; runnable CLI | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | wormhole N150 mesh=1x1 TP1/DP1; wormhole N300 mesh=1x2 TP2/DP1; wormhole N300 mesh=1x2 TP1/DP2; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP2/DP4; wormhole T3K mesh=1x8 TP1/DP8 | none at pinned SHA |
| [llama33_70b](examples/llama33_70b/README.md) | experimental | concrete; runnable CLI | `meta-llama/Llama-3.3-70B-Instruct` | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | wormhole T3K mesh=1x8 TP8/DP1; blackhole P150_X4 required logical target; code accepts physical P300_X2 alias with distinct provenance mesh=1x4 TP4/DP1 | none at pinned SHA |
| [llama3_8b](examples/llama3_8b/README.md) | experimental | concrete; runnable CLI | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | wormhole N150 mesh=1x1 TP1/DP1; wormhole N300 mesh=1x2 TP2/DP1; wormhole N300 mesh=1x2 TP1/DP2; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP4/DP2; wormhole T3K mesh=1x8 TP2/DP4; wormhole T3K mesh=1x8 TP1/DP8; blackhole P150 mesh=1x1 TP1/DP1; blackhole P150_X4 mesh=1x4 TP4/DP1; blackhole P150_X4 mesh=1x4 TP1/DP4; blackhole physical_P300 mesh=1x2 TP1/DP2 | none at pinned SHA |
| [mistral_7b](examples/mistral_7b/README.md) | experimental | concrete; runnable CLI | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | wormhole N150 mesh=1x1 TP1/DP1; wormhole N300 mesh=1x2 TP2/DP1; wormhole N300 mesh=1x2 TP1/DP2; wormhole T3K mesh=1x8 TP8/DP1; wormhole T3K mesh=1x8 TP1/DP8 | none at pinned SHA |
| [phi4](examples/phi4/README.md) | experimental | concrete; runnable CLI | `microsoft/phi-4` | `187ef0342fff0eb3333be9f00389385e95ef0b61` | wormhole N300 mesh=1x2 TP2/DP1; wormhole T3K mesh=1x8 TP2/DP4 | none at pinned SHA |
| [qwen25_72b](examples/qwen25_72b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen2.5-72B-Instruct` | `495f39366efef23836d0cfae4fbe635880d2be31` | wormhole T3K mesh=1x8 TP8/DP1 | none at pinned SHA |
| [qwen25_7b](examples/qwen25_7b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | wormhole N300 mesh=1x2 TP2/DP1; wormhole T3K mesh=1x8 TP2/DP4 | none at pinned SHA |
| [qwen25_coder_32b](examples/qwen25_coder_32b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen2.5-Coder-32B-Instruct` | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` | wormhole T3K mesh=1x8 TP8/DP1 | none at pinned SHA |
| [qwen2_7b](examples/qwen2_7b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen2-7B-Instruct` | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | wormhole N300 mesh=1x2 TP2/DP1; wormhole T3K mesh=1x8 TP2/DP4 | none at pinned SHA |
| [qwen3_32b](examples/qwen3_32b/README.md) | experimental | concrete; runnable CLI | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | wormhole T3K mesh=1x8 TP8/DP1; blackhole P150_X4 required logical target; code accepts physical P300_X2 alias with distinct provenance mesh=1x4 TP4/DP1 | none at pinned SHA |

Evidence policy and details:

- [Pinned support baseline](qualification/analysis/support/support_baseline.md)
- [Hardware evidence ledger](qualification/analysis/support/hardware_evidence.csv)
- [Phase 3 support-boundary report](qualification/extraction/support_boundary.md)
- [Example overview](examples/README.md)

An empty evidence cell would be ambiguous, so every row explicitly says that pinned-SHA evidence is absent. Historical observations at other SHAs are context only.
