# vLLM generator interface

Every model in `tt_transformers.models` exposes a `vllm_generator.py`. This file
is the model's only integration point with vLLM: the
[TT vLLM plugin](https://github.com/tenstorrent/vllm-tt-plugin) loads the
generator class, constructs it, and drives every serving call through it.

Because the plugin calls the same methods on every model, a model that
implements this interface can be served without plugin changes. Models that
invent their own serving surface cannot. This page describes what the
generator must provide.

## Where it lives

```text
src/tt_transformers/models/<model>/vllm_generator.py
```

The plugin registers the generator by module path and class name, for example
`tt_transformers.models.llama3_8b.vllm_generator:Llama3Generator`.

## Registering with the plugin

The plugin resolves a checkpoint's Hugging Face architecture to a generator
class through vLLM's model registry. It renames every architecture with a `TT`
prefix before lookup, so `LlamaForCausalLM` is looked up as
`TTLlamaForCausalLM`. Point that architecture at a tt-transformers generator in
one of two ways:

- **`TT_MODEL_CLASS_OVERRIDES`**: comma-separated `Arch=module.path:Class`
  entries. Overrides are registered ahead of every other source and win for the
  whole process.

  ```bash
  export TT_MODEL_CLASS_OVERRIDES="TTLlamaForCausalLM=tt_transformers.models.llama3_8b.vllm_generator:Llama3Generator"
  ```

- **`EXTRA_MODELS_DIR`**: a directory of bundle folders. Each folder holds a
  `vllm_metadata.json` with `arch` (the HF architecture name) and `main_class`
  (`"module:Class"`). Bundles register only when no override already claims
  the architecture.

  ```json
  {
    "arch": "LlamaForCausalLM",
    "main_class": "tt_transformers.models.llama3_8b.vllm_generator:Llama3Generator"
  }
  ```

Several HF architectures can map to one generator. An override applies only to
the architecture it names, so register every alias your checkpoints resolve to.
The plugin's built-in map still points at tt-metal generators. Set
`TT_VLLM_BUILTIN_MODELS=0` to serve only overrides and bundles.

## Lifecycle

The plugin calls the generator in this order:

1. `get_max_tokens_all_users(...)` to size the scheduler's KV budget.
2. `initialize_vllm_model(...)` to build the model and its executor.
3. `allocate_kv_cache(...)` once vLLM has chosen the physical cache size.
4. `warmup_model_prefill(...)` and `warmup_model_decode(...)`.
5. Repeated `prefill_forward(...)` and `decode_forward(...)` calls, with
   `read_decode_output(...)` and `process_decode_output_host(...)` when decode
   is asynchronous.
6. `cleanup()`.

The model is built before the cache size is known. Construct it against a
maximum capacity, and apply the real capacity in step 3 before allocating
anything on the device.

## Class attributes

| Attribute | Meaning |
|---|---|
| `model_capabilities` | Dict of flags the plugin reads: `supports_prefix_caching`, `supports_async_decode`, `supports_sample_on_device`. Declare only what the model actually supports; each flag changes how the plugin calls the methods below. |
| `requires_prefill_trace_warmup` | `True` when prefill traces must be captured during warmup. Every current model sets it. |

Current generators also declare `max_device_top_k` and `accepts_trace_mode` in
`model_capabilities`. Multi-device models declare `fabric_config`, which the
plugin reads before it opens the mesh; for example,
`{"config": ttnn.FabricConfig.FABRIC_1D_RING}` for T3K and P150x4 geometries.
The full key inventory, with defaults and validation rules, is the plugin's
[`MODEL_CAPABILITIES.md`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/docs/MODEL_CAPABILITIES.md).

## Class methods

### `initialize_vllm_model`

```python
@classmethod
def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len,
                          n_layers=None, tt_data_parallel=1, optimizations="performance",
                          **kwargs) -> "Generator"
```

Build and return a generator instance: load weights, build the tensor model,
and build one executor per data-parallel lane. Read the checkpoint name from
`hf_config._name_or_path`. `max_batch_size` is global and must divide evenly
by `tt_data_parallel`. Do not allocate the KV cache here.

### `get_max_tokens_all_users`

```python
@classmethod
def get_max_tokens_all_users(cls, model_name="", num_devices=1, tt_data_parallel=1,
                             max_model_len=0, max_num_seqs=1, **kwargs) -> int
```

Return the KV token budget for one data-parallel submesh. The plugin calls it
before the model exists, so it must not touch the device. Raise `ValueError`
for configurations the model does not support.

## Instance methods

### `allocate_kv_cache`

```python
def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers) -> list[list[ttnn.Tensor]]
```

`kv_cache_shape` is `(num_blocks, local_kv_heads, block_size, head_dim)`.
Check that the block size, head count, head dimension, dtype, and layer count
match the model, and that `num_blocks` does not exceed the construction
maximum. Then allocate the cache and return one `[K, V]` pair per layer.
vLLM passes this exact object back as `kv_cache` on every later call.
Reject it if it is a different object.

### `warmup_model_prefill` / `warmup_model_decode`

```python
def warmup_model_prefill(self, *, kv_cache, can_sample_on_device, enable_trace, **kwargs) -> None
def warmup_model_decode(self, *, kv_cache, max_batch_size, num_blocks,
                        can_sample_on_device, enable_trace, **kwargs) -> None
```

Compile every program that serving will use and, when `enable_trace` is set,
capture the traces. Warmup must exercise the same paths as serving. A request
that warmup did not cover cannot be traced later.

The plugin warms up in two passes: both methods with `enable_trace=False`, then
prefill and decode again with `enable_trace=True` for the operations its trace
mode selects. Between the passes it sets `already_warmed_up_prefill = False` on
the generator when that attribute exists, so expose it if prefill warmup is
guarded by such a flag.

### `prefill_forward`

```python
def prefill_forward(self, tokens, page_table, *, kv_cache, enable_trace,
                    prompt_lens=None, start_pos=None, empty_slots=None,
                    sampling_params=None, **kwargs)
```

| Argument | Meaning |
|---|---|
| `tokens` | `[batch, padded_len]` prompt tokens. |
| `page_table` | `[batch, blocks]` block ids per row. |
| `prompt_lens` | Real prompt length per row. |
| `start_pos` | Cached prefix length per row (prefix caching). |
| `empty_slots` | Decode slot per row; selects the data-parallel lane. |
| `sampling_params` | Present only when sampling on device. |

Returns, in input row order:

- host sampling: logits as a `[batch, 1, vocab]` float tensor;
- device sampling: `(tokens, log_probs)`, where `tokens` is a `[batch]` `int64` tensor.

### `decode_forward`

```python
def decode_forward(self, tokens, start_pos, page_table, *, kv_cache, enable_trace,
                   read_from_device=True, sampling_params=None, reset_batch=False,
                   **kwargs)
```

`tokens` and `start_pos` cover the full fixed-size decode batch. With
`read_from_device=True`, return `(output, log_probs)` on the host, where
`output` is sampled tokens or logits. With `read_from_device=False`, return the
device output without reading it. The plugin then completes it through the two
methods below.

When sampling on device, the plugin may also pass `prompt_tokens`,
`output_tokens`, and `slot_remap` (sampler state) and `reset_batch` (reset that
state).

### `read_decode_output` / `process_decode_output_host`

```python
def read_decode_output(self, tt_out, *, async_read=False)
def process_decode_output_host(self, tt_out, *, is_tokens=False) -> tuple
```

These are required when `supports_async_decode` is true.

- `read_decode_output(async_read=True)` starts the device-to-host copy and
  returns `(host_value, events)`.
- `process_decode_output_host` waits for those events and returns
  `(output, log_probs)`. It must not submit new device work.

### `cleanup`

```python
def cleanup(self) -> None
```

Release every device resource. Cleanup must be safe to call twice.

## Rules for all methods

- **Accept `**kwargs` on forward and warmup calls.** The plugin passes fields
  that only some models use. Ignore the ones you don't need, and raise on the
  ones you can't honour.
- **Honour `enable_trace` exactly.** If a traced call is requested and not
  available, raise. Never fall back to eager silently.
- **Keep the generator thin.** It adapts arguments, picks eager or traced
  execution, and delegates. Device tensors, traces, and the KV cache are owned
  by the executor, and `cleanup()` delegates to it.

## Executors

The generator delegates to an executor built on
[`llm_runtime`](../src/tt_transformers/llm_runtime/README.md).
Models typically use one of these:

- the shared [`ModelExecutor`](../src/tt_transformers/models/executor.py),
  directly or through a family facade such as
  [`llama3_executor.py`](../src/tt_transformers/models/llama3_executor.py)
  or [`qwen2_executor.py`](../src/tt_transformers/models/qwen2_executor.py);
- a model-owned executor composed from `llm_runtime` modules, when the shared
  one doesn't fit (see [Phi-4](../examples/phi4/README.md)).

Two `llm_runtime` helpers cover most of the generator's own work:

- [`VLLMAdapter`](../src/tt_transformers/llm_runtime/vllm_adapter.py)
  normalizes forward-call arguments and validates the KV cache shape.
- [`LaneGroupExecutor`](../src/tt_transformers/llm_runtime/lane_group.py)
  presents several data-parallel lanes as one executor.

## Reference implementation

[`llama3_8b/vllm_generator.py`](../src/tt_transformers/models/llama3_8b/vllm_generator.py)
implements this interface in full, and the
[Llama 3.1 8B guide](../examples/llama3_8b/README.md#vllm-lifecycle)
walks through its lifecycle.
