# TT Transformers support matrix

All current model packages are **experimental**. A concrete implementation,
runnable repository example, or passing regression cell is not a promise that
the model's complete geometry and workload matrix is qualified.

Software target: `tt-transformers==2.0.0.dev0`, `ttnn==0.77.0`, Python 3.10
or 3.12, and Linux on x86-64.

| Model | Status | Pinned Hugging Face revision | Declared candidate geometry |
|---|---|---|---|
| DeepSeek R1 Distill Qwen 14B | experimental | `1df8507178afcc1bef68cd8c393f61a886323761` | N300 TP2; T3K TP8, TP4/DP2, TP2/DP4 |
| Llama 3.2 1B | experimental | `9213176726f574b556790deb65791e0c5aa438b6` | N150, N300, T3K with declared TP/DP variants |
| Llama 3.2 3B | experimental | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | N150, N300, T3K with declared TP/DP variants |
| Llama 3.3 70B | experimental | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | T3K TP8; Blackhole P150_X4 TP4 |
| Llama 3.1 8B | experimental | `0e9e39f249a16976918f6564b8830bc894c89659` | N150, N300, T3K, P150, P150_X4, and declared TP/DP variants |
| Mistral 7B | experimental | `c170c708c41dac9275d15a8fff4eca08d52bab71` | N150, N300, T3K |
| Phi-4 | experimental | `187ef0342fff0eb3333be9f00389385e95ef0b61` | N300 TP2; T3K TP2/DP4 |
| Qwen2.5 72B | experimental | `495f39366efef23836d0cfae4fbe635880d2be31` | T3K TP8 |
| Qwen2.5 7B | experimental | `a09a35458c702b33eeacc393d103063234e8bc28` | N300 TP2; T3K TP2/DP4 |
| Qwen2.5 Coder 32B | experimental | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` | T3K TP8 |
| Qwen2 7B | experimental | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | N300 TP2; T3K TP2/DP4 |
| Qwen3 32B | experimental | `9216db5781bf21249d130ec9da846c4624c16137` | T3K TP8; Blackhole P150_X4 TP4 |

Machine-readable details live in each `examples/<model>/support.json`. Those
manifests are the authority for checkpoint, cache, topology, context, and
known-limit declarations.

## Current validation snapshot

At tested code SHA `9134c399334240e3e4d35dfa93013c6c7293a3d1`, the
42-node regression matrix passed in full:

| Scope | Passed |
|---|---:|
| Reusable modules | 30/30 |
| Runtime trace/order | 1/1 |
| Model smoke | 3/3 |
| End-to-end/token accuracy | 8/8 |
| Wormhole | 23/23 |
| Blackhole | 19/19 |

Selected model evidence exists for Llama 3.2 1B, Llama 3.3 70B, Llama 3.1
8B, Qwen2.5 7B, Qwen2.5 Coder 32B, and Qwen3 32B. It covers only the exact
recorded cells and does not promote any model from `experimental`.

The eight P150 cells were logical 1x1 selections on a physical P150_X4 host
with `MESH_DEVICE=P150` and `TT_VISIBLE_DEVICES` unset. They are not
standalone-P150 product evidence.

See [docs/validation.md](docs/validation.md) for exact counts, topology
qualifications, current evidence hashes, and the historical audit tag.

## Unsupported or unqualified by default

- model/topology/TP/DP/batch/context combinations absent from a manifest;
- a complete performance envelope for any model;
- physical SKU substitution inferred only from a logical `MESH_DEVICE` name;
- checkpoints or revisions other than the manifest's pinned revision;
- training, arbitrary Hugging Face generation semantics, or full vLLM server
  compatibility;
- clean teardown claims beyond the exact recorded regression cells.

Support is expanded only through a reviewed manifest change and attributable
non-skipped evidence at the exact tested code revision.
