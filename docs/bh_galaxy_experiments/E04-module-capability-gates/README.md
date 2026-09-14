# E04 — Let the 2D modules accept a Blackhole mesh

**Hardware: one 32-device Blackhole Galaxy. One patch.**
**Base SHA: `746bb77`.**

**This is phase 5's first task, and it unblocks every module suite after it.** Nothing in
[E05](../E05-multilayer-residual/), [E06](../E06-mlp-dram-interleaved/) or
[E07](../E07-fused-ccl-noop/) can run until this lands.

---

## Why it exists

Phases 1–4 built the per-architecture topology and capability infrastructure, but the 2D modules
never got wired to it. `MLP2D` and `RMSNorm2D` still gate on
`mesh_device.arch() != ttnn.device.Arch.WORMHOLE_B0` and reject a Blackhole mesh by name.

That was deliberate, for a layering reason: `GalaxyCapabilities` lives under `models/`, and
`modules/` never imports `models/` — the dependency runs the other way at four sites and never
back. Widening the gates needed a decision about where a module-layer Galaxy fact belongs, and
that decision wanted a hardware run to mean anything.

## What the patch does

It adds `src/tt_transformers/modules/galaxy_mesh.py` holding the two facts a module's own mesh
gate needs, and routes both gates through it:

```python
GALAXY_MESH_SHAPE = (8, 4)          # identical on both architectures
GALAXY_ARCHITECTURES = frozenset({Arch.WORMHOLE_B0, Arch.BLACKHOLE})

def require_galaxy_mesh(name, mesh_device) -> tuple[int, int]: ...
```

Three things about the shape of that, because each was a choice:

**It lives in `modules/`, not `models/`.** `modules/tt_ccl.py` already knows `BHGLX`, so a
module-layer fact about Galaxy hardware is not a model-layer concern. No import inversion.

**It is weaker than `recipes.validate_galaxy_mesh`, on purpose.** This one asks *"is this a
Galaxy mesh of an architecture the modules run on"*. The stronger question — *"does this
architecture have a qualified intra-chip geometry"* — stays with the model layer, keyed on a
topology descriptor existing. Two gates, two different claims, and the strong one is not
weakened.

**It also deduplicates `WH_GALAXY_MESH_SHAPE`**, which was defined identically in both modules
and used arithmetically in `mlp_2d.py`, not only for validation. The name is kept as a re-export
so the arithmetic reads unchanged.

The patch also updates the two tests that asserted the old messages. The `requires Wormhole` row
is **deleted** rather than adjusted — it asserted the opposite of the new contract — and replaced
with a test that a Blackhole Galaxy mesh now resolves.

## Hypothesis

With the gates widened, `_resolve_2d_config` and `_resolve_mlp2d_config` **complete** on a
Blackhole mesh.

They will not be *correct*. The resolved memory configs, dtypes and program configs are still the
Wormhole recipe. This experiment asks only whether resolution gets far enough to start
measuring — the acceptance tests in the patch say so explicitly.

## Run

```bash
git apply docs/bh_galaxy_experiments/E04-module-capability-gates/patch.diff

# Host first: this part needs no device, and it is where a resolution error shows up.
pytest -m host tests/modules/mlp/test_mlp_2d.py tests/modules/rmsnorm/test_rmsnorm_2d.py -v
pytest -m host                       # full host sweep; expect no new failures
python qualification/tools/audit_test_taxonomy.py

# Then the device probe, to confirm nothing about the mesh gate regressed.
MESH_DEVICE=BHGLX pytest tests/models/galaxy/test_topology_bh_galaxy.py -sv
```

**Run the host half even without a Blackhole machine.** It is the part that most likely fails,
and it costs nothing.

## What each outcome means

**Host resolution passes on both modules.** The gate was the only thing in the way. Land it and
proceed to E05. Expect the *device* suites to fail afterwards — that is E05/E06/E07's job, not a
regression.

**Resolution raises inside a memory-config or program-config helper.** The expected and
informative outcome. Record *which* helper and the exact error: that is the list of what the
Blackhole recipe actually has to change, and it is worth more than the experiment's nominal
result. Likely candidates, in order — the ring memcfgs (`has_ring_matmul` is False on Blackhole,
so any path still reaching for one is a real bug in phase 3's wiring), the DRAM-sharded weight
placement (8 views rather than 12), and the decode shard-height arithmetic.

**Resolution passes but a Wormhole test changes behaviour.** Stop. The patch touches a
hardware-qualified path, and `test_wormhole_link_counts_are_unchanged_by_the_clamp` plus the
golden tables in `test_topology.py` are the tripwires. Any Wormhole movement is a defect in the
patch, not a finding.

**`check_import_boundaries` fails.** The new module imported something it should not. It must
depend on `ttnn` and nothing else in this repository.

---

## After this lands

The gate is widened but the modules are not yet *capability-driven* — they still do not consult
`GalaxyCapabilities.has_fused_residual_norm`, `has_fused_ccl` or `has_ring_matmul`. Wiring those
through the module configs is the natural next step, and E05–E07 are precisely the measurements
that say which ones matter. Do that with evidence rather than ahead of it.

---

## Result

*Not run.*
