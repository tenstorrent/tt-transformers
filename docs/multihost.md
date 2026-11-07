# Multi-host (cross-box) mesh support

TTTv2 is single-host-mesh today: one process owns the whole mesh. This page
documents the opt-in primitives in
[`tt_transformers.distributed`](../src/tt_transformers/distributed.py) for
running a mesh that spans more than one host (for example two Blackhole boxes
driven as one `1x4` FABRIC_1D mesh, 2+2), and the correctness rules that make
such a run safe.

These primitives are additive and opt-in. Nothing in the existing generators or
models calls them yet — call-site adoption is a deliberate follow-up — so
single-host behaviour is unchanged.

> **Status:** the cross-host behaviour below was validated out-of-tree during a
> pure-ttnn model bring-up across two Blackhole hosts (a 1x4 FABRIC_1D mesh,
> 2+2; 20/20 multi-size soak). The primitives here are a port of that discipline
> and need code-owner validation on real multi-host hardware. The single-host
> no-op behaviour is covered by host tests in this repository.

## Branch on the host rank count, never on the mesh shape

The single most important rule:

> Single-host vs. multi-host is a function of the host **rank count**, not the
> mesh shape.

A `1x4` mesh can be one host owning four devices, or two hosts owning two
devices each. The shape alone cannot tell them apart; only the
distributed-context / MPI world size can. Selecting a cross-host path from
`mesh_device.shape` will silently do the wrong thing on one of those topologies.

Use the rank-count helpers to make the decision:

```python
from tt_transformers.distributed import num_host_ranks, host_rank, is_multi_rank

if is_multi_rank():
    ...  # cross-host path
else:
    ...  # single-host path (unchanged)
```

`num_host_ranks()` resolves, in order, the ttnn distributed-context world size
(when a context is initialized), then `OMPI_COMM_WORLD_SIZE`, then `1`. In a
plain single process it returns `1` and `is_multi_rank()` returns `False`.

## The readback-DMA-drain invariant

The cross-host correctness rule:

> `ttnn.execute_trace(..., blocking=True)` fences **compute**, but it does **not**
> fence the device->host readback DMA that `ttnn.to_torch` / `Tensor.cpu` starts.

If a rank consumes the host buffer, or lets the next `execute_trace` overwrite
the source tensor, before that readback DMA has drained, it corrupts its host
heap. In a traced per-step decode loop this manifests as:

- **rank-asymmetric** corruption — the send-first rank trips; a recv-first rank
  is incidentally safe, so a single-host run or one rank can look fine while a
  peer crashes;
- **memory-checker-blind** — the overwrite is a device DMA, not a CPU
  instruction, so CPU memory checkers (valgrind/memcheck) never see it;
- **deterministic under a hardened allocator** — a hardened/guarded malloc turns
  the latent corruption into a reliable, reproducible crash.

### How to stay safe

1. **Drain after every readback.** Immediately after reading a step's output(s)
   back to host, call `ttnn.synchronize_device(mesh)` before touching the host
   buffers or launching the next `execute_trace`. The canonical readback path,
   [`OutputReader.read_synchronized`](../src/tt_transformers/llm_runtime/output_reader.py),
   already does this: it submits every host copy non-blocking, retains each
   destination, and issues a single `synchronize_device` that drains them all
   before returning. Route per-step host round-trips through it (or an equivalent
   drain) rather than a bare non-blocking `cpu()`.

2. **Prefer gather-once on-device.** Where the model allows it, keep state
   on-device across steps and gather to host once at the end instead of reading
   back every step. This sidesteps the per-step race entirely.

3. **Rendezvous across hosts when multi-rank.** On a cross-host run, add an MPI
   barrier after the drain so no rank races ahead into the next step before every
   peer has drained its own readback.
   [`cross_host_rendezvous(mesh_device)`](../src/tt_transformers/distributed.py)
   does both: it drains with `synchronize_device`, then — only when
   `is_multi_rank()` — issues `ttnn.distributed_context_barrier()`. In a
   single-host run it is exactly one `synchronize_device` and no barrier.

```python
from tt_transformers.distributed import cross_host_rendezvous

# ... per traced step, right after reading this step's output back to host:
cross_host_rendezvous(mesh_device)
```

## Hardening mesh-open

The cross-host fabric link intermittently fails to finish training the first
time the mesh is opened, surfacing as a fabric-router-sync timeout that clears on
a retry. `open_mesh_device_with_retry(open_callable)` retries only errors that
match that signature and re-raises anything else immediately. It is a no-op on
the common path — a successful first open returns right away and never sleeps —
so single-host behaviour is unchanged:

```python
from tt_transformers.distributed import open_mesh_device_with_retry

parent = open_mesh_device_with_retry(
    lambda: ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(parent_shape), **params)
)
```

`open_callable` must fully open and, on failure, fully tear down the mesh so each
attempt starts from a clean state.

## Operational runbook

For the launch procedure (fabric-clean reset, launching under `tt-run`, path and
user requirements, link count), see the multi-host runbook in
[the TTNN development guide](ttnn-development.md#multi-host-cross-box-runs).

## Deferred: call-site adoption

This change introduces the primitives only. Wiring the generators, the mesh/
submesh construction in [`mesh_utils`](../src/tt_transformers/mesh_utils.py), and
the example runtimes to select paths via `is_multi_rank()` and rendezvous via
`cross_host_rendezvous()` is an explicit follow-up to be done under code-owner
review on multi-host hardware.
