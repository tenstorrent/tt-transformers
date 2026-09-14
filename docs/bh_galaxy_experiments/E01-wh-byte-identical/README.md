# E01 — Is Wormhole decode byte-identical after the topology refactor?

**Hardware: one 32-device Wormhole Galaxy (`wh-glx6u-05`). No patch.**

**The highest-value single experiment here.** It is the only thing that can confirm phase 2 did
not move a hardware-qualified path, and it is on a different machine and a different schedule
from everything else in this vault.

---

## Why it exists

Phase 2 replaced every intra-chip core coordinate in `recipes.py` and `prefetch.py` with a
`GalaxyChipTopology` descriptor. That is the geometry source of a path that **was** qualified on
silicon — 56 of 57 cheap module ids green, roughly 50 minutes of device time.

The port plan's own exit criterion for that phase is *"one WH Galaxy run with **bit-identical**
decode output, not merely passing PCC"*, and it says the criterion is not optional.

**It has not been run, and there is no Wormhole Galaxy access either.** So phase 2 currently
rests on two things:

1. `test_topology.py`'s golden tables, which pin every core sequence element-wise and in order
   against the literals as they stood before the refactor;
2. `test_wormhole_link_counts_are_unchanged_by_the_clamp`, which pins the full ordered
   `num_links` tuple for both models.

Both are real, and both check the **resolver**. Neither checks the **callers**. That gap is what
this experiment closes.

## Hypothesis

Every Wormhole Galaxy module suite produces output byte-identical to the pre-refactor commit.

Not "passes PCC". Byte-identical. The distinction is the whole point: the
`CoreRangeSet`-ordering defect this refactor is most at risk of reintroducing produced
**bit-for-bit identical wrong output across runs** — stable, not noisy, which is exactly why PCC
thresholds did not catch it and why it took weeks to find.

## Run

```bash
# The commit before the descriptor existed. `0a3e045` is phase 1, which changed
# no geometry; `b0f0024` is the refactor itself.
BASE=0a3e045
HEAD=$(git rev-parse HEAD)

for SHA in $BASE $HEAD; do
  git checkout $SHA
  # One node id per process. The ttnn program cache belongs to the mesh device
  # and the process, and the weight cache is keyed on MeshDevice.id().
  MESH_DEVICE=TG pytest tests/modules/mlp/test_mlp_2d_wh_galaxy.py -sv \
      2>&1 | tee wh-$SHA-mlp.log
  MESH_DEVICE=TG pytest tests/modules/rmsnorm/test_rmsnorm_2d_wh_galaxy.py -sv \
      2>&1 | tee wh-$SHA-rmsnorm.log
  # ... and the rest of the 57 cheap module ids.
done
```

Then compare. **Comparing the logs is not enough** — they carry timings and addresses. The
comparison has to be on tensor contents, so the suites need to dump them; add a
`torch.save` of each decode output under a path keyed by SHA, and compare with
`torch.equal`, not `allclose`.

### Two things that will otherwise waste the window

**`attention_decode_with_active_prefetch` is a known-red baseline.** It passes setup, call, PCC
and its own cleanup, and still exits non-zero because `ttnn.close_mesh_device` hangs in the
fixture afterwards: the test starts the DRAM producer, but attention decode is precisely the
module that must *not* consume the global CB, so nothing drains the ring. Treat it as red on
both sides of the comparison, not as a regression this refactor introduced.

**Three fresh processes, not one.** The fused-RMSNorm stats CB binds to the first core of the
norm input shard grid and aliases whatever the allocator left there, which has produced PCC of
0.0977 / 0.1394 / 0.1701 / 0.9999 across processes on an unchanged test. A single run that
matches proves less than you think.

## What each outcome means

**Byte-identical across the board.** Phase 2 is confirmed and the descriptor may keep the
qualified path. Record it in `docs/validation.md` and in the port log — this is the evidence the
whole refactor is missing, and it should be findable later.

**PCC passes but output differs.** *This is the dangerous outcome and the one to look hardest
at.* Something in the resolved geometry moved in a way the golden tables did not catch, which
means the tables are checking the wrong thing. First suspect: **iteration order** somewhere a
`CoreRangeSet` is rebuilt from the descriptor. `core_points` preserves the tuple order it is
given, so check that the *tuple* is what you expect, then that nothing downstream reconstructs
the set from an unordered collection.

**A suite fails outright.** Compare against the base SHA before concluding anything. The base run
is not ceremony — 56 of 57 was the state before this work, so a red suite may predate it.

**It cannot be run at all.** Then say so, plainly, wherever phase 2 is described. Carrying an
unverified refactor of a qualified path is a legitimate decision; carrying it *silently* is not.

---

## Result

*Not run. No Wormhole Galaxy access.*
