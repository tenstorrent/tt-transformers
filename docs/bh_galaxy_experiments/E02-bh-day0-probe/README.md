# E02 — The Blackhole Galaxy day-0 probe

**Hardware: one 32-device Blackhole Galaxy. No patch.** The test is already in the tree.

**Run this first on the first Blackhole window.** It answers four of the port plan's five open
hardware questions, it is read-only, and it takes about as long as opening the mesh.

---

## Why it exists

Every number the port's geometry rests on came from `tt-metal`'s descriptors or from reading a
third-party reference port. None of it was measured here. This is where that changes.

It allocates no tensor and runs no collective, so it cannot leave the mesh in a state the next
run has to recover from — which matters when the recovery unit is a reset cycle.

## Hypotheses

| # | Hypothesis | Source |
| --- | --- | --- |
| 1 | The mesh is `(8, 4)`, 32 devices, `Arch.BLACKHOLE` | `single_bh_galaxy_mesh_graph_descriptor.textproto` declares `arch: BLACKHOLE`, `device_topology { dims: [8, 4] }` |
| 2 | `get_cluster_type()` returns `BLACKHOLE_GALAXY` | the detection route the SKU and marker plumbing keys on |
| 3 | The compute grid is `12 x 10`, DRAM views 8 | the reference marks `12x10 tensix grid (x:0-11, y:0-9), 8 DRAM banks` as *measured* |
| 4 | **Harvesting is uniform across all 32 devices** | nobody has checked. This is the genuinely open one |
| 5 | Both link tables report 2 | `channels { count: 2 }` vs `tt_ccl`'s `("BHGLX", (2, 2))` — two independent derivations |
| 6 | The mesh opens with an L1 small region of ≥ 32768 B | sized against Wormhole's bank; Blackhole's is unmeasured |
| 7 | The worker envelope excludes the dispatch column | excluded empirically, not derivably |

## Run

```bash
# One node id per process. The probe's five tests are independent; run them
# separately so a crash in one does not cost the others.
MESH_DEVICE=BHGLX pytest tests/models/galaxy/test_topology_bh_galaxy.py -sv
```

`-s` matters: every probe prints its readings on `[bhglx]` lines, and **the readings are the
point** — the assertions only catch the cases that are already understood.

## What each outcome means

**Hypothesis 2 fails (cluster type is not `BLACKHOLE_GALAXY`).** The detection route in
`fixture_policy._is_blackhole_cluster` falls back to `ttnn.device.is_blackhole()`, so nothing
breaks loudly — but `get_logical_sku` would then depend on that fallback, and the marker policy
keys on `MESH_DEVICE` rather than the cluster, so it is unaffected. Record what it *does* return
and adjust the named enum members. Do not add enum members speculatively: one that does not exist
raises `AttributeError` inside the probe's `try` and **silently disables the whole check**.

**Hypothesis 3 fails (grid is not `12 x 10`).** Not a failure — the resolver derives the envelope
from whatever is reported, and `11 x 10` and `13 x 10` are both handled. It means the chassis is
not the one the reference characterized, so treat every other reference-sourced number as
correspondingly weaker.

**Hypothesis 4 fails (mixed harvesting).** *This is the one that costs design work.* The
descriptor resolves one grid for the whole mesh. A mixed chassis needs a per-mesh **minimum**,
which no current code takes, and the failure mode is a placement on a core that exists on 31
boards and not on the 32nd — silent on almost everything. Fix in `resolve_galaxy_chip_topology`,
by taking the element-wise minimum across `get_devices()` rather than the mesh-level grid, and
add the per-device enumeration to the validation.

**Hypothesis 5 fails (the two link tables disagree).** One of them is wrong and both are used.
`GALAXY_FABRIC_LINKS` feeds `plans.py`'s clamp and the reserved worker-core count;
`get_num_links` feeds anything calling it directly. An incorrect Galaxy `num_links` **deadlocks**
rather than failing, so resolve this before running any collective. The mesh graph descriptor is
the tiebreaker.

**Hypothesis 6 fails (no L1 small region, or a different size).** `GALAXY_L1_SMALL_SIZE` is
32768 B chosen against Wormhole's 1 393 472 B bank, and main L1 shrinks by exactly that much for
*every* Galaxy op. If Blackhole's bank differs materially, set `l1_small_size` on the Blackhole
descriptor — it is already a per-architecture field for this reason. If the region is absent
entirely, generic collectives will place their program-compile-time semaphores mid-bank in main
L1, where nothing can be placed across them afterwards, and the `prefill -> decode` transition
fails.

**Hypothesis 7 fails.** The descriptor is wrong about the dispatch column. Read the reported
grid and fix `reserved_columns`; the validator will then reject any placement that reaches it.

---

## Result

*Not run.*

Record here: the `[bhglx]` readings verbatim, and which hypotheses held.
