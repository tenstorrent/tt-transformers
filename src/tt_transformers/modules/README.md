# TTTv2 Modules User's Guide

This document is for **users** of TTTv2 modules (`src/tt_transformers/modules`). We identify two categories of users:
- **Quick Start Users** (90% of users) who want to get started quickly with the simple positional-weight API.
- **Power Users** (10% of users) who need fine-grained control and customization of the TTNN ops within the modules.

Every module in TTTv2 follows the **same contract** (see [The Universal Module Contract](#the-universal-module-contract)), so once you learn one (e.g. `MLP1D` below) you know them all.

---

## Zen of TTTv2

TTTv2 is built on a few core principles that guide its design and usage.

### 1. Library, not Framework
Users control the execution flow. TTTv2 provides high-performance building blocks, not an opinionated orchestration layer.

### 2. No if-else on static conditions in `forward()`
Avoid runtime branching on static configuration to keep execution paths predictable and fast.
- **Keep hot paths simple**: Implementation should be a straight line of compute.
- **Decouple config and implementation**: Strategy decisions happen during construction, not in the inner loop. (See how `Sampling1D` binds a topology-specific strategy at construction time instead of branching in `forward()`.)

### 3. Lazy and Transparent is better than Proactive and Opaque in weight loading and API design
Efficiency and predictability through lazy initialization and explicit APIs.
- **Lazy weight loading**: Weights (`LazyWeight`) and mutable state buffers (`LazyBuffer`) load/allocate on first use, not at construction, saving memory and time during model setup.
- **Transparent API**: TTNN interfaces are used directly where possible so the underlying operations are clear.
- **Sensible defaults**: Each module ships with a config known to work for a select set of ML models.
- **Full override capability**: Every default can be customized for your specific model architecture.

### 4. More unit tests than end-to-end tests
We prioritize fast, focused, and debuggable unit tests of modules. This ensures faster iteration, easier debugging, and left-shifted CI testing.
- Initially, we parameterize unit tests with real use cases by all the models TTTv1 supports.
- As we add more models to TTTv2, we will continue to add more parameterizations to the unit tests to cover those models.
- We also added code coverage tests to ensure that we are covering most of the code paths in the TTTv2 modules.

---

## The Universal Module Contract

Every TTTv2 module is a `LightweightModule` subclass that exposes the same surface:

- a **`<Name>Config`** dataclass — the single source of truth; every field is optional except the weights, and unset fields are filled with sensible defaults at construction;
- a **simple constructor** (90% path) that takes only weights + essential dimensions and derives everything else;
- a **`from_config(cfg)`** classmethod (10% path) for full customization;
- a **`forward(...)`** that is a straight line of compute (no static if-else — see Zen #2);
- selected **1D modules** retain a `from_model_args(...)` bridge for the retiring TTTv1 stack. New 2D modules intentionally do not expose this bridge; product models construct their configs explicitly.

Current module inventory:

| Module | Class (file) | Simple constructor | `forward` signature |
|--------|--------------|--------------------|---------------------|
| MLP (1D) | `MLP1D` (`mlp/mlp_1d.py`) | `MLP1D(w1, w2, w3)` | `forward(x, mode)` |
| MLP (2D) | `MLP2D` (`mlp/mlp_2d.py`) | `MLP2D(w1, w2, w3)` | `forward(x, mode)` |
| Attention | `Attention1D` (`attention/attention_1d.py`) | `Attention1D(wqkv, wo, n_heads, n_kv_heads, head_dim, max_batch_size, max_seq_len)` | `forward(..., mode)` |
| Attention (2D) | `Attention2D` (`attention/attention_2d.py`) | `Attention2D(wqkv, wo, ...)` | `forward(..., mode)` |
| RMSNorm (1D) | `RMSNorm1D` (`rmsnorm/rmsnorm_1d.py`) | `RMSNorm1D(weight)` | `forward(x, mode)` |
| RMSNorm (2D) | `RMSNorm2D` (`rmsnorm/rmsnorm_2d.py`) | `RMSNorm2D(weight)` | `forward(x, mode)` |
| RoPE | `RotarySetup1D` (`rope/rope_1d.py`) | `RotarySetup1D(cos_matrix, sin_matrix, max_batch_size)` | `forward(mode, **kwargs)` |
| RoPE (2D) | `RotarySetup2D` (`rope/rope_2d.py`) | `RotarySetup2D(cos_matrix, sin_matrix, max_batch_size)` | `forward(mode, **kwargs)` |
| Embedding | `Embedding1D` (`embedding/embedding_1d.py`) | `Embedding1D(weights, embed_scale=1.0)` | `forward(x)` |
| Embedding (2D) | `Embedding2D` (`embedding/embedding_2d.py`) | `Embedding2D(weights, embed_scale=1.0)` | `forward(x, mode)` |
| LM Head | `LMHead1D` (`lm_head/lm_head_1d.py`) | `LMHead1D(output_weights)` | `forward(x)` |
| LM Head (2D) | `LMHead2D` (`lm_head/lm_head_2d.py`) | `LMHead2D(output_weights, vocab_size)` | `forward(x, mode)` |
| Sampling | `Sampling1D` (`sampling/sampling_1d.py`) | `Sampling1D(vocab_size, mesh_device)` | `forward(logits, **kwargs)` |
| Sampling (2D) | `Sampling2D` (`sampling/sampling_2d.py`) | `Sampling2D(vocab_size, mesh_device)` | `forward(logits, **kwargs)` |
| Penalties | `Penalties1D` (`sampling/penalties_1d.py`) | `Penalties1D(vocab_size, mesh_device)` | `forward(logits, params, accum)` |

Notes:
- `forward(x, mode)` modules take `mode="prefill"` or `mode="decode"` (a `str` or the `Mode` enum). Modules without a `mode` argument (`Embedding1D`, `LMHead1D`) are called the same way regardless of phase.
- `Sampling1D`/`Penalties1D` are **stateful** ops driven by runtime token data, so they take `vocab_size` + `mesh_device` (not `LazyWeight`s) and manage device state through `LazyBuffer` (see [Supporting Infrastructure](#supporting-infrastructure)).

---

## Quick Start (90% of Users)

For most use cases, the simple positional-weight API is all you need. You wrap your PyTorch tensors in `LazyWeight` and pass them to the constructor:

```python
import ttnn
from tt_transformers.modules.mlp.mlp_1d import MLP1D
from tt_transformers.modules.lazy_weight import LazyWeight

# 1. Prepare weights (from PyTorch)
# LazyWeight doesn't load to device until needed
w1 = LazyWeight(source=torch_w1, dtype=ttnn.bfloat4_b)
w2 = LazyWeight(source=torch_w2, dtype=ttnn.bfloat8_b)
w3 = LazyWeight(source=torch_w3, dtype=ttnn.bfloat4_b)

# 2. Construct the module
# Sensible defaults are automatically resolved (device, topology, etc.)
mlp = MLP1D(w1, w2, w3)

# 3. Forward pass
# First forward pass will trigger weight loading to device
x = ttnn.from_torch(torch_x, device=mesh_device, dtype=ttnn.bfloat16)
y = mlp.forward(x, mode="prefill") # or mode="decode"
```

The same three steps apply to every module — only the constructor args change (see the inventory table above).

See the test at `tests/modules/mlp/test_mlp_1d.py::test_mlp_1d_vs_reference` for a complete working example that shows:
- How to create `LazyWeight` instances with disk caching
- How to run prefill/decode forward passes
- How to validate outputs against a HuggingFace reference model

---

## Power Users (10% of Users)

When you need fine-grained control, use the module's config dataclass (here `MLP1DConfig`):

```python
from tt_transformers.modules.mlp.mlp_1d import MLP1D, MLP1DConfig

# Create config with any overrides you need
config = MLP1DConfig(
    w1=w1, w2=w2, w3=w3,
    mesh_device=mesh_device,
    topology=ttnn.Topology.Ring,
    max_batch_size=64,
    # ... any other overrides
)

mlp = MLP1D.from_config(config)
y = mlp.forward(x, mode="prefill")
```

### Why Use the Config Dataclass?

The config dataclass is the single source of truth for the module.

| Benefit | Description |
|---------|-------------|
| **Composable** | Pass around, modify, inherit |
| **Self-documenting** | All options in one dataclass |
| **Gradual customization** | Override just what you need |


### Example: Overriding Program Configs

See `tests/modules/mlp/test_mlp_1d.py::test_mlp_1d_config_prefill_override` for a complete example that demonstrates:
- Creating a custom `prefill_w2_prg_config` function
- Overriding it on an existing `MLP1D` instance
- Validating the custom config works correctly

```python
# After constructing the model, override specific configs
tt_model = MLP1D.from_config(MLP1DConfig(w1=lazy_w1, w2=lazy_w2, w3=lazy_w3))

@lru_cache
def custom_prefill_w2_prg_config(seq_len: int):
    # Your custom config logic here
    return _matmul_config(...)

tt_model.config.prefill_w2_prg_config = custom_prefill_w2_prg_config
```

### The Continuous Customization Path

TTTv2 provides a **continuous experience** from simple to advanced usage:

```
Simple Usage                       Advanced Customization
      │                                     │
      │ (pass weights)                      │ (pass config)
      ▼                                     ▼
MLP1D(w1, w2, w3)                   MLP1D.from_config(cfg)
      │                                     │
      │  + sensible defaults                │  + your overrides
      └──────────────────┬──────────────────┘
                         │
                         ▼
                  Resolved Config
             (Fully defined at runtime)
```

---

## Supporting Infrastructure

These shared building blocks back every module. You rarely construct them directly (modules wire them up for you), but understanding them explains the design.

### Weights vs. Buffers: `LazyWeight` and `LazyBuffer`

Both defer device allocation until first use and accept any tensor `ttnn.from_torch()` can handle (no hard `torch` dependency).

| | `LazyWeight` (`lazy_weight.py`) | `LazyBuffer` (`lazy_buffer.py`) |
|---|---|---|
| For | Immutable model weights | Mutable state tensors (e.g. token counts, penalty masks) |
| Disk cache | Yes — fingerprinted for cache invalidation | No — caching a mutable buffer would corrupt state |
| Materialize | `get_device_weight()` | `get_device_buffer()` |
| Update | n/a (immutable) | `update(new_source)` writes device in-place, same handle |

`Sampling1D` and `Penalties1D` are the only modules that use `LazyBuffer` today, because they mutate device state across decode steps.

### Collectives: `TT_CCL`

Multi-device modules need collective ops (reduce-scatter, all-gather). `TT_CCL` (`tt_ccl.py`) owns the hardware semaphores those ops require. There is **one instance per `mesh_device`**, created and cached for you via `get_tt_ccl(mesh_device)` — so modules sharing a device share semaphores. Pass your own through the config's `tt_ccl` field only if you need to. CCL tuning constants (`CCL_CHUNKS_PER_SYNC`, `CCL_NUM_WORKERS_PER_LINK`, `CCL_NUM_BUFFERS_PER_CHANNEL`) live in the same file and are shared across all modules.

### `Mode`

`mode`-aware modules accept either the string `"prefill"`/`"decode"` or the `Mode` enum from `models.tt_transformers.tt.common`.

---

## 1D vs 2D Modules

`*1D` modules target 1D-topology devices: N150 (1×1), N300 (1×2), and T3K (1×8). Production `*2D` configs target Wormhole Galaxy with the canonical logical mesh shape `(8, 4)` and exactly 32 devices. Resolution fails closed for another architecture, mesh orientation, device count, incompatible collaborator, or indivisible partition.

The 2D tensor-placement contract is explicit in each config: source and padded weight shapes, row/column shard dimensions, decode and prefill input/output placement, and transient ownership are resolved before execution. Static strategy decisions do not branch in a module hot path.

Galaxy collectives are injected from `src/tt_transformers/models/galaxy`; 2D modules do not extend or specialize the 1D `TT_CCL` owner. The target ownership contract makes `Prefetcher2D` the model-owned resource root for subdevice managers, the global circular buffer, and sealed weight-address registration. Modules borrow immutable prefill/decode contexts, and the executor activates a context at operation boundaries and owns cleanup. Integrated Prefetcher2D/Galaxy-resource ownership is qualified on hardware for the MLP2D consumer shape: the prefill↔decode transition matrix, the failed-transition rollback, and cleanup from either active mode all run on a `(8, 4)` 6U Galaxy with PCC asserted at every step.

Two ownership limits are known and documented. **Cleanup of the global circular buffer is not deterministic from the owner's side:** ttnn exposes no free for one, so its L1 is reclaimed when the last handle dies, and every module holding a `Prefetcher2DContext` holds one — tear consumers down before, or together with, the owner. Dropping the last Python reference *does* return the L1: a second production-size global CB is refused while the first is alive and is created immediately after the first is dropped, so the C++ destructor frees on the last reference. A `Prefetcher2DContext` is captured **by value** at module construction, so `Prefetcher2D.cleanup()` clearing its own `_contexts` map used to leave every already-built module holding a context whose `global_cb` was still the live buffer; `cleanup()` now nulls the field on every context it handed out. **The ordering rule above is therefore sufficient:** tear the consumers down with the owner and the buffer goes. One caveat remains. A second model in the same process can still fail to create its global circular buffer even when the total free L1 exceeds `GALAXY_GLOBAL_CB_SIZE`, because the bank is **fragmented rather than full** and the largest contiguous block falls short. Two details make allocator messages about this readable: `validate_circular_buffer_region` computes one device-wide lowest-occupied L1 address *before* it loops over circular-buffer core ranges, so the core range in the message names the **circular buffers**, not the clashing buffer; and the allocator's block table prints addresses without `offset_bytes_` while every allocator message adds it.

**The global circular buffer has a placement contract.** Including its config page the buffer lands *below* the address the prefill embedding's static circular buffers reach, and the L1 above that line is smaller than the resident model plus the buffer, so **the buffer and a prefill program of this shape cannot coexist on any allocation order.** `defer_global_cb` covers only the first prefill; `release_global_cb_on_prefill` is therefore **required** for a prefill after a decode, which is what serving does by construction, and it is the default for both Galaxy models. Releasing alone is not sufficient: L1 is allocated top-down and an address never moves, so the first decode forward strands a **32-byte** buffer beneath the resident CB and the release does not move it. `global_cb_headroom` reserves L1 above the buffer while it is created so nothing is stranded under it, and because a *fixed* headroom lets the recreated buffer land lower — after which the next decode hangs on the stale address held by cached programs — `Prefetcher2D._allocate_global_cb` **reproduces the free-region top the first creation saw**, reserving in a loop since `FreeListOpt` prefers a small exact-size free block and one reservation lands in the gap the previous headroom left, and **raises** if the buffer does not come back on the recorded floor.

Two refinements are part of that contract. First, the buffer is **two** allocations — a `GALAXY_GLOBAL_CB_SIZE` data buffer and a 192-byte config page — and a cached program holds the address of **both** (`CircularBufferImpl::set_global_circular_buffer` captures `buffer_address()` and `config_address()` once; `dispatch.cpp` re-sends the captured pair on every launch), so the owner pins both and reports the *blocks* rather than a lowest-address proxy. Reproducing only the free top is not enough: `FreeListOpt::allocate` takes the smallest free block that fits, and a 192-byte hole anywhere in L1 will take the config page and separate it from its data buffer. The owner therefore holds every free gap above the low region before creating the buffer, so the low region is the only free block and the pair is forced to be adjacent. Second — and this is the **scope** of the guarantee — the ttnn program cache belongs to the *mesh device*, outlives any one model, and keys on op-and-config hashes, so two structurally identical Galaxy models in one process share compiled decode programs and therefore share the first model's buffer addresses. The invariant is per **process and mesh**, not per owner: `GlobalCBPlacement` is the record, `src/tt_transformers/models/galaxy/prefetch.py` holds one per mesh device, and an owner given none keeps its own. Holding the record per owner instead lets a second model's first decode hang in `FDMeshCommandQueue::wait_for_outstanding_reads` with both models' weights resident. So the guarantee the owner makes is: *the global circular buffer is created on the same L1 blocks every time, for every model in the process, or an exception is raised;* a silently moved buffer is a wrong-address read and is never returned.

**The cleanup contract covers layer weights as well as the buffer.** `close()` returns the global circular buffer, but returning the layer **weights** needs its own release: a `LazyWeight` memoizes its device tensor, so dropping the last Python reference is *not* what returns device memory — the weight is reachable for as long as any caller holds the model, and a serving system holds a closed executor by construction. Without that release a closed model leaves its full layer weights resident in DRAM, which exhausts the bank when a second model of the same size is built in the same process. `Embedding2D`, `LMHead2D`, `RotarySetup2D` and `Sampling2D` already had a `release`; `RMSNorm2D`, `MLP2D` and `Attention2D` now do too, over the shared `lazy_weight.release_device_weights` primitive, which dedupes on the `LazyWeight` **and** on its `_value` because `Attention2D` resolves `prefill_wqkv = resolved.prefill_wqkv or wqkv`. **The model owns the ordering, not the module:** the release runs *after* `Prefetcher2D.cleanup()`, because the attention decode weights are registered with the prefetcher and freeing one while a prefetch can still read it is a use-after-free — and a `TT_FATAL` inside a multi-subdevice program leaves the mesh un-drainable. For the same reason a model that *borrows* shared resources does not release; it does not control when the prefetch stops, so its weights stay resident and that is recorded rather than assumed away. **So the guarantee the owner makes is: after `close()` returns, the model holds no device weight and no global circular buffer, whether or not the caller still holds the model** — asserted per model by `test_<model>_a_closed_model_returns_its_device_weights`, which deletes nothing and collects nothing before it measures.

**`Attention2D`'s decode projection grid is compatible with the prefetch subdevice partition, at a named cost.** A `(7, 1)` QKV grid straddles the sender and worker subdevices, so both attention decode matmuls are confined to the largest worker rectangle anchored at the worker envelope's origin (`dense_matmul_worker_rectangle`, three columns wide on `(8, 4)`), and with `in0_block_w = gcd(k_tiles, 4)` their circular buffers fit. **Two costs, recorded rather than absorbed:** three worker columns instead of seven, and — because the global circular buffer is received only by the 24 ring cores — the attention weights cannot be prefetched and must be read from DRAM. Registering them with the prefetcher anyway is a *correctness* defect, not a performance one, because the unconsumed entries shift every later consumer. Moving the two matmuls to the 24-core `gather_in0` ring recovers both at once, and is open performance work.

The reusable 2D set consists of `Embedding2D`, `RotarySetup2D`, `RMSNorm2D`, `Attention2D`, `MLP2D`, `LMHead2D`, and `Sampling2D`. There is no `Penalties2D`.

Everything Galaxy-specific but model-neutral lives in `src/tt_transformers/models/galaxy`: the `(8, 4)` geometry and placement recipes, the collective-resource plans, the `Attention2D`/`LMHead2D` collective adapters, the prefetch construction policy, the paged-KV metadata view, and a direct prefill/decode runner used by tests and demos. No transformer graph lives there. Each product package (`src/tt_transformers/models/llama33_70b_galaxy`, `src/tt_transformers/models/qwen3_32b_galaxy`) owns its own graph — checkpoint contract, precision recipe, provider weight conversion, every 2D module config, its decoder layer, its tensor model, and its construction order — and borrows only that topology-neutral machinery. Neither package imports another model-named package.

`Attention2D` takes two different paged page-table layouts, because the two modes address the cache differently. Prefill fills a named user's blocks with `paged_fill_cache(..., batch_idx=u)`, so its device-local table needs one row for every user the request fills; concatenated physical-batch-32 prefill therefore carries all 32 rows. Decode attends to one mesh column's users, so `paged_update_cache` and the paged decode SDPA both require the device-local table to carry exactly `users_per_column` rows — or that batch repeated once per core when the table is L1-sharded.

> **Known gap — a prefill-shaped table handed to decode is *not* rejected.** `_validate_decode_page_table` discriminates on row count alone and accepts any positive multiple of `users_per_column`. That modulo is deliberate — an L1-sharded decode table legitimately repeats the device-local batch once per core — but the replicated prefill table's device-local view is 32 rows and `32 == 4 * 8`, so it passes; the width check passes too, because the prefill table is stick-aligned and therefore *wider*, never narrower. **Shape cannot separate the two cases**; the discriminator that would work is `memory_config()`, which the validator never consults. The reverse direction *does* fail closed. Do not rely on decode rejecting the prefill layout until this is fixed.
>
> The validator's docstring states this limitation rather than claiming the rejection. Closing the gap is a behaviour change and is tracked separately.

All seven modules plus Galaxy CCL/resources and `Prefetcher2D` are qualified on real WH `(8, 4)` hardware: 37 device cases pass in one sweep with clean teardown and no reset.

### Module contract amendments

Three contract changes landed **after** the module contracts above were first recorded. They are listed here so a reader of the module contracts sees them rather than finding them inside a model diff. Two of the three have since been exercised on silicon; the third is dead code by construction and cannot be.

- **`Attention2D` `wo` source shape** is now `(n_heads * head_dim, dim)` rather than `(dim, dim)`. That is the only way to express Qwen3-32B's decoupled 64-head geometry (`attention_dim = 8192 != dim = 5120`); the two spellings coincide for every square case, so no recorded numerical result changes. Both the decoupled and the square case are pinned, plus the rejection message for a wrong shape. **Qualified on silicon** by the Qwen3-32B bring-up, which runs the real decoupled geometry.
- **`LMHead2D` activation width** now also accepts a column-local width (`dim / 4`) alongside the full `dim` (`lm_head_2d.py:507-511`), because a device activation taken off the column-sharded residual stream carries its column shard. A strict superset, so no existing test changes behaviour. **Exercised on silicon**: it is the width both Galaxy models actually present at decode — 2048 for Llama, 1280 for Qwen — so every decode logit produced on the Galaxy goes through the widened branch.
- **`Attention2D` lazy weight resolution**: `wqkv` and `wo` were each being handed the other's placement field. Corrected; **host-tested only, and it cannot be otherwise.** The swap was **unreachable** — `_require_exact_weight_policy` runs first and rejects any weight whose `memory_config` is not already equal to its own config field — so it was dead code rather than a live defect, and the gate that makes it unreachable is now itself pinned.

### The Galaxy model packages

Two product models are built on this inventory — `src/tt_transformers/models/llama33_70b_galaxy` (Llama-3.3-70B, 80 layers) and `src/tt_transformers/models/qwen3_32b_galaxy` (Qwen3-32B, 64 layers, 64 decoupled attention heads) — plus the model-neutral Galaxy plumbing in `src/tt_transformers/models/galaxy` described above.

> **Both models run on real `(8, 4)` silicon.** Teacher-forced accuracy, the one-layer block comparisons against a Hugging Face reference, batch-32 demos, prefix-cached prefill and the 4K/32K/128K functional smokes all exercise them there. **Performance is not yet characterised.**
>
> Two capabilities have no reachable case on either model, and both are limits of the shared recipe rather than of one geometry: concat-32 physical prefill does not fit in L1 at any supported length, and on-device sampling is blocked, so every sampled-token readback taken so far is one mesh column repeated four times. Treat both as unavailable.

---

## Running the Tests

Each module has its own test directory under `tests/modules/`:

```bash
# Run all MLP1D tests (fast subset)
pytest tests/modules/mlp/test_mlp_1d.py -v

# Include slow tests for full coverage
pytest tests/modules/mlp/test_mlp_1d.py -v --slow

# Run a specific test
pytest tests/modules/mlp/test_mlp_1d.py::test_mlp_1d_config_creation -v

# Run the whole module test suite
pytest tests/modules/ -v
```

Test files follow the module layout and use `test_<name>_1d.py` / `test_<name>_2d.py` where both variants exist. `prefetcher/test_prefetcher_2d.py` covers explicit registration, sealing, activation, and cleanup. Shared infrastructure has its own tests too (`test_lazy_buffer.py`, `test_tensor_utils.py`).

`test_<name>_2d.py` files are **host-only** and run against a mock mesh; the real `(8, 4)` Galaxy suites are `test_<name>_2d_wh_galaxy.py`, plus `sampling/test_sampling_2d_wh_galaxy_stochastic.py` and `prefetcher/test_prefetcher_2d_wh_galaxy.py`. Select against them deliberately — passing a module directory to pytest picks up its device suite as well, which needs 32 devices. Filter with `--ignore-glob="*_wh_galaxy*.py"` for a host-only run.

Both 1D and 2D numerical suites qualify against the same HuggingFace references, shared through `tests/modules/_hf_reference.py`; the Galaxy MLP geometry that several 2D device suites reuse lives in `tests/modules/_mlp_2d_galaxy.py`, and the hardware plumbing in `tests/modules/_wh_galaxy_hardware.py`.

### Device Topologies Tested

| Mesh Shape | Device Type | Module |
|------------|-------------|--------|
| `(1, 1)` | N150 (single device) | `*1D` |
| `(1, 2)` | N300 | `*1D` |
| `(1, 8)` | T3K | `*1D` |
| `(8, 4)` | Galaxy | `*2D` |
