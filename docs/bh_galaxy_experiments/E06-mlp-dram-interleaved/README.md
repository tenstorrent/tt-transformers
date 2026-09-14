# E06 — Does DRAM-interleaved MLP decode hold PCC on Blackhole?

**Hardware: one 32-device Blackhole Galaxy. Specification, not a patch.**
**Depends on [E04](../E04-module-capability-gates/) and follows [E05](../E05-multilayer-residual/).**

---

## Why it exists

Two separate failures make the Wormhole ring memory configs unusable on Blackhole, and **one of
them is silent**:

**Loud.** The ring memcfg's shard grid uses core column `x=6`, outside the auto-selected 1D
matmul compute grid: *"Tensor shard spec grid ... must lie within compute grid"*, *"output shard
grid ... must lie within extent"*. This one is safe — it fails to compile.

**Silent.** Feeding a ring-sharded all-gather output into the interleaved W2 matmul mismatches
per-device channel order and yields **MLP decode PCC ≈ 0** with nothing raised.

The reference's answer to both is DRAM interleaved with `program_config=None`.
`GalaxyCapabilities.has_ring_matmul` is already `False` on the Blackhole descriptor, and
`ring_core_coords` is `None` rather than Wormhole's coordinates, so a consumer that still reaches
for a ring fails loudly instead of silently receiving another chip's geometry. What is untested
is whether the DRAM-interleaved path is numerically right.

## Hypothesis

`MLP2D` decode on Blackhole, DRAM-interleaved with `program_config=None`, holds PCC ≥ 0.99
against a CPU reference.

## What to build

A Blackhole MLP2D decode suite modelled on `tests/modules/mlp/test_mlp_2d_wh_galaxy.py`, with:

1. **DRAM-interleaved weights and `program_config=None`.** Not a ring memcfg.
2. **A CPU reference**, never another TT path.
3. **Per-user and per-row correlation of the output**, not only an aggregate PCC. This is the
   specific instrumentation this failure needs — see below.
4. Byte-identical across three fresh processes, as everywhere in this vault.

## The instrumentation that actually finds this class of bug

Aggregate PCC is close to useless here, and the Wormhole evidence is explicit about why:
**column-local sharding bugs on a 2D mesh do not fail loudly — they give PCC in the 0–0.01 range
and need per-user/per-row correlation to localize.** A single number tells you something is
wrong, not which column.

So: correlate output per mesh **row** and per mesh **column** separately. A channel-order
mismatch shows up as a permutation — every column individually well-correlated with *some*
reference column, just not its own. That signature distinguishes it immediately from a genuine
numerics problem, where the correlation is bad everywhere.

Two further practices from the Wormhole bring-up, both of which paid for themselves:

- **Locate the closest known-passing reference for the exact op/shape/axis before inventing a
  resource plan.** Repeatedly, several ttnn-legal spellings of the same math existed and exactly
  one worked. Two named reusable ones:
  `tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py` and
  `.../ccl/test_new_all_reduce.py`.
- **Staged PCC instrumentation beats final-output PCC.** Note that `comp_pcc` logs nothing on
  success, so a passing stage tells you nothing unless you print it.

## Run

```bash
MESH_DEVICE=BHGLX pytest tests/modules/mlp/test_mlp_2d_bh_galaxy.py -sv
```

Debug any CCL hang with **worker-sub-device-scoped** `synchronize()` fences, never a whole-device
sync.

## What each outcome means

**PCC ≥ 0.99, byte-identical across processes.** The DRAM-interleaved path is right. Record the
resolved memory configs — they are the Blackhole MLP recipe, and the next module should start
from them rather than from Wormhole's.

**PCC ≈ 0, per-column correlation shows a permutation.** The channel-order failure, exactly as
predicted. The all-gather output ordering does not match what the W2 matmul expects. Do not
"fix" it by re-sorting the output — find which collective produces the order, because the same
mismatch will appear in attention.

**PCC ≈ 0, correlation bad everywhere.** Not the channel-order bug. Suspect the dtype recipe or
the reduce-scatter width arithmetic. The decode shard height is `tile_padded_batch_rows` rather
than a fixed 32 on this path.

**It hangs.** Check `num_links` before anything else. The ring and line CCLs index ethernet
channels by link, and Blackhole has two: an over-requested link count deadlocks with no
traceback, and a `TT_FATAL` out of `enqueue_mesh_workload` leaves the mesh **un-drainable** —
SIGTERM cannot service it and SIGKILL plus reset is the only recovery.

---

## Result

*Not run.*
