# vLLM decode and trace lifecycle

The model `vllm_generator.py` classes declare
`decode_input_update_contract = 1`. They implement the TT plugin's
[decode reload contract](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/docs/DECODE_RELOAD_CONTRACT.md).
Use a plugin that supports this contract. The vLLM boundary rejects the old
`reset_batch` keyword. The standalone executor keeps its existing interface.

## Decode commands

The plugin sends four Boolean commands on every decode call:

| Command | Effect |
| --- | --- |
| `reload_inputs` | Copy tokens, positions, RoPE inputs, and page tables. |
| `reload_page_table` | Copy page tables without changing resident tokens or positions. |
| `reload_sampling_params` | Upload sampling configuration. |
| `reset_sampling_state` | Rebuild mutable penalty and RNG state. |

`reload_inputs` and `reload_page_table` cannot both be true.
`reset_sampling_state` requires `reload_inputs`. Direct calls default to
`reload_inputs=True` with the other commands false.
The first device-sampled decode must also set `reload_sampling_params=True`
and `reset_sampling_state=True`. The plugin sends these transition commands.
The runtime rejects a sampled call that would reuse synthetic warmup state.
Model families without a native penalty controller declare
`supports_device_penalties=False`. The plugin uses host sampling for their
penalty requests. Direct device-penalty calls fail before state changes.
Enabling penalties after a step without penalties requires a state reset and
token history. Enabling repetition penalty also requires a reset with prompt
history. A parameter upload alone cannot recover history that was not tracked.

When `reload_inputs=False`, host tokens and positions are stale. The runtime
keeps the resident request layout and advances its seed counters once per
sample. A page-table update uses the scheduler's current mapping. It does not
use a stale host position to select the blocks to copy.

`slot_remap[i] = j` moves continuing request state from slot `j` to slot `i`.
The runtime applies it once, before a requested reset. This also applies to
the dormant device sampler when sampling runs on the host. Lane execution
validates the merged remap and converts its indices to each lane's slot space.

A step with resident inputs must use the trace that owns those inputs.
Switching to a different trace requires authoritative inputs. The runtime
rejects an unsupported resident transition instead of copying stale host data.

## Output lifetime

For split decode, submit the read before another operation can reuse the
trace's output buffer:

1. Call `decode_forward(..., read_from_device=False)`.
2. Call `read_decode_output(output, async_read=True)`.
3. Complete the read before host processing.
4. Call `process_decode_output_host(...)`.

The device-to-host copy and trace replay use command queue 0. This ordering
allows the next replay after the copy has been queued. The runtime retains
the host destination until read completion. It rejects reuse while a raw
trace output still needs a read. An eager output that owns device allocations
must be retired before trace replay.

## Trace preparation

Prefill and decode warmup must both finish before capture, including when
only decode uses traces. Capture preparation allocates the persistent inputs
and workspaces first. It then runs the exact capture paths and postprocessing
before the first trace capture. This includes padded prefill rows and sampled
decode variants covered by the warmup configuration.

Each sampled capture binds its own static sampling identity. The runtime
checks the device program-cache count when the backend exposes it. A change
after trace capture starts is a failure, not permission to compile more work.
Only known trace outputs are acknowledged as corruptible. A broad allocation
scope must not hide late program-cache buffers.

For device validation, enable allocation tracking before Python starts:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 python qualification/tools/run_hardware_matrix.py ...
```

Use an exact selector from `tests/hardware/hardware-matrix.json`. Follow the
device policy in [tests/README.md](../tests/README.md). The host CI checks
ordering and command handling with fake devices. It does not prove device
accuracy or allocator safety. See the upstream
[TraceCorrectness guide](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md)
for the device trace requirements.
