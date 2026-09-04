# Repository examples

Examples are runnable from a repository checkout and are not installed by the
wheel. Every model is currently experimental.

## Set up

```bash
git clone https://github.com/tenstorrent/tt_transformers.git
cd tt_transformers
python -m pip install -e '.[examples,test]'
```

Each model README documents its `HF_MODEL`, pinned revision, `MESH_DEVICE`,
cache requirements, supported arguments, and exact commands.

| Example | Current validated subset |
|---|---|
| [DeepSeek R1 Distill Qwen 14B](deepseek_r1_distill_qwen_14b/README.md) | no model-family cell in the current 42-node regression matrix |
| [Llama 3.2 1B](llama32_1b/README.md) | N150 and N300 token accuracy |
| [Llama 3.2 3B](llama32_3b/README.md) | no model-family cell in the current matrix |
| [Llama 3.3 70B](llama33_70b/README.md) | T3K token accuracy; P150_X4 smoke and token accuracy |
| [Llama 3.1 8B](llama3_8b/README.md) | logical P150 token accuracy |
| [Mistral 7B](mistral_7b/README.md) | no model-family cell in the current matrix |
| [Phi-4](phi4/README.md) | no model-family cell in the current matrix |
| [Qwen2.5 72B](qwen25_72b/README.md) | no model-family cell in the current matrix |
| [Qwen2.5 7B](qwen25_7b/README.md) | N300 token accuracy |
| [Qwen2.5 Coder 32B](qwen25_coder_32b/README.md) | T3K one-layer smoke |
| [Qwen2 7B](qwen2_7b/README.md) | no model-family cell in the current matrix |
| [Qwen3 32B](qwen3_32b/README.md) | T3K token accuracy; P150_X4 smoke and token accuracy |

## Validation boundary

The current pre-cleanup regression record is 42/42 passing at code SHA
`73d414f8b826a7da982df8c8229d4ac41ed8ba33`: modules 30/30, runtime 1/1,
smoke 3/3, and end-to-end 8/8. See
[`docs/validation.md`](../docs/validation.md) for topology qualifications and
the archived raw-evidence location.

A passing subset does not qualify a complete model contract. The canonical
status and declared candidate geometries are summarized in
[`SUPPORT.md`](../SUPPORT.md); machine-readable details remain in each
`support.json`.

## Test wrappers

Hardware pytest wrappers live under `tests/hardware/models`. Use the checked-in
matrix runner rather than inventing selectors:

```bash
python qualification/tools/run_hardware_matrix.py --list
```

Hardware processes must be serialized per physical host. A skipped or xfailed
test is not validation evidence.
