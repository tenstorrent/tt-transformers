# Model implementations

`tt_transformers.models` contains concrete model implementations and shared
family executor policy. Models combine the reusable tensor modules, sampling
utilities, and inference runtime described in the
[package guide](../README.md) and [software architecture](../../../docs/architecture.md).

All current models are **experimental**. The [support matrix](../../../SUPPORT.md)
summarizes their status; each model's `examples/<model>/support.json` records
its checkpoint revision, candidate topologies, cache requirements, and known
limitations. An implementation or passing test does not imply support for
every hardware and workload combination.

## Model directory

| Model | Implementation | Runnable example and configuration |
|---|---|---|
| DeepSeek R1 Distill Qwen 14B | [`deepseek_r1_distill_qwen_14b`](deepseek_r1_distill_qwen_14b/) | [Example](../../../examples/deepseek_r1_distill_qwen_14b/README.md) |
| Llama 3.2 1B | [`llama32_1b`](llama32_1b/) | [Example](../../../examples/llama32_1b/README.md) |
| Llama 3.2 3B | [`llama32_3b`](llama32_3b/) | [Example](../../../examples/llama32_3b/README.md) |
| Llama 3.3 70B | [`llama33_70b`](llama33_70b/) | [Example](../../../examples/llama33_70b/README.md) |
| Llama 3.1 8B | [`llama3_8b`](llama3_8b/) | [Example](../../../examples/llama3_8b/README.md) |
| Mistral 7B | [`mistral_7b`](mistral_7b/) | [Example](../../../examples/mistral_7b/README.md) |
| Phi-4 | [`phi4`](phi4/) | [Example](../../../examples/phi4/README.md) |
| Qwen2.5 72B | [`qwen25_72b`](qwen25_72b/) | [Example](../../../examples/qwen25_72b/README.md) |
| Qwen2.5 7B | [`qwen25_7b`](qwen25_7b/) | [Example](../../../examples/qwen25_7b/README.md) |
| Qwen2.5 Coder 32B | [`qwen25_coder_32b`](qwen25_coder_32b/) | [Example](../../../examples/qwen25_coder_32b/README.md) |
| Qwen2 7B | [`qwen2_7b`](qwen2_7b/) | [Example](../../../examples/qwen2_7b/README.md) |
| Qwen3 32B | [`qwen3_32b`](qwen3_32b/) | [Example](../../../examples/qwen3_32b/README.md) |

The `llama3_8b` directory implements Llama **3.1** 8B. Examples run from a
repository checkout and are not installed by the wheel.

## Model package layout

| File | Responsibility |
|---|---|
| `hf_adaptor.py` | Adapt Hugging Face configuration, checkpoints, and tokenizers to the model. |
| `model.py` | Compose reusable modules into the model's TTNN tensor computation and layouts. |
| `executor.py` | Construct or specialize model execution using shared runtime owners. |
| `generator.py` | Expose direct or vLLM-facing generation calls and delegate execution. |
| `weight_utils.py` | Convert model-specific weights where a separate helper is needed. |
| `__init__.py` | Declare the model package's intended import surface. |

Some models reuse another family implementation instead of defining every
component independently. Shared policy lives in [`executor.py`](executor.py),
[`llama3_executor.py`](llama3_executor.py), and
[`qwen2_executor.py`](qwen2_executor.py) at this directory's root. Generic
prefill/decode, cache, trace, and cleanup machinery belongs in
[`llm_runtime`](../llm_runtime/README.md).

For installation and TTNN version selection, see
[release compatibility](../README.md#release-compatibility). Each
tt-transformers release uses a pinned, publicly available TTNN package.

## Model lifecycle

Experimental models are available for development with the limitations in
their support manifests. Experimental status does not mean a model is
deprecated; none of the current model implementations has been moved to a
legacy directory.

When a model is deprecated, its implementation will move to the `legacy/`
subdirectory here: `src/tt_transformers/models/legacy/<model>/`. Its repository
example will move to [`examples/legacy/<model>/`](../../../examples/legacy/README.md).
The archived code is retained for reference and reproduction against its last
supported environment.

A deprecation change must record the reason and any replacement, the last
supported tt-transformers release or tag, TTNN and other dependency versions,
checkpoint revision, hardware topology, and final validation evidence. Retain
the example's README, support manifest, and constraints as described in the
[legacy examples policy](../../../examples/legacy/README.md), and update the
model directory and support matrix to identify the archive location.

Legacy models are excluded from current support claims and default CI when
archived. Their old import paths and compatibility with newer TTNN releases
are not guaranteed; use the recorded release or tag to reproduce the original
environment.
