# Blackhole Galaxy port — findings and decisions log

Working log for the deviceless phases of
[blackhole_galaxy_port_plan.md](blackhole_galaxy_port_plan.md). **Important findings and
non-trivial decisions only** — routine work belongs in commit messages, and anything needing
silicon belongs in [bh_galaxy_experiments/README.md](bh_galaxy_experiments/README.md).

Tags follow [bh_galaxy_deferred_work.md](bh_galaxy_deferred_work.md) §0: **[repo]** verified in
this working tree, **[measure]** needs silicon, **[inference]** reasoning rather than evidence.

---

## 0. The verification envelope — better than expected

**Finding (2026-09-14): four of the seven host gates, and 305 host tests, run on a Mac with no
`ttnn`.** **[repo]**

The working assumption was that nothing could be executed locally, because `ttnn` publishes only
`manylinux_2_34_x86_64` wheels and this machine has no Linux container. That is true of `ttnn`, but
it over-generalises: most host gates never import it.

| Gate | Local? | Notes |
| --- | --- | --- |
| `audit_test_taxonomy` | **yes** | pure `ast` + `pathlib` |
| `check_import_boundaries` | **yes** | needs Python ≥ 3.10 for `sys.stdlib_module_names` |
| `run_hardware_matrix --validate` | **yes** | JSON only |
| `validate_bh_required_capabilities` | **yes** | JSON + `ast` |
| `ruff` | **yes** | standalone binary, see the version caveat below |
| `pytest -m host` | **partly** | 305 of ~2612 pass; 149 files fail collection on `import ttnn` |
| `mypy` | untested | targets six `src/` files |

Recipe, reproducible from scratch:

```bash
uv python install 3.12
uv venv --python 3.12 hostvenv
uv pip install --python ./hostvenv/bin/python pytest pyyaml loguru numpy torch
PYTHONPATH=src ./hostvenv/bin/python -m pytest -m host tests/ -q --continue-on-collection-errors
```

`tests/conftest.py` imports `numpy` and `torch` at module scope but **not** `ttnn`, which is what
makes this work at all; `ttnn` is imported lazily inside fixtures and helpers.

**How to use it: as a differential, never as an absolute.** 195 failures (46 failed + 149
collection errors) are baseline in this venv, from absent `ttnn`, `transformers` and
`huggingface_hub`. A run is "green" when `comm` against a stashed-HEAD run shows no new entries,
not when the count is zero:

```bash
git stash -u && <run> | grep -E '^(FAILED|ERROR)' | sort > before.txt && git stash pop
<run> | grep -E '^(FAILED|ERROR)' | sort > after.txt && comm -13 before.txt after.txt
```

This does **not** retire [E00](bh_galaxy_experiments/README.md): the real `pytest -m host` covers
~2612 tests against the real `ttnn`, and every Galaxy geometry suite is in the 149 that cannot
collect here. It does mean a deviceless phase can be wrong in fewer ways before it gets there.

**Caveat — `ruff` version skew.** The repo pins `ruff>=0.11.0`; `uv tool run ruff` resolves 0.16.7,
whose `I001` disagrees with the pinned version about whether `qualification` and `tests` are
first-party. It reports 11 pre-existing files as un-sorted. Treat `I001` as advisory locally and
keep matching the surrounding files' convention; every other rule is trustworthy.

---

## 1. Phase 1 — taxonomy, SKU and link budget

### 1.1 `ClusterType.BLACKHOLE_GALAXY` already exists in the pinned `ttnn==0.77.0` **[repo]**

[tests/conftest.py:76-80](../tests/conftest.py#L76-L80)'s `is_galaxy()` already tests against it.
So the plan's §3 detection route is available *now*, not only in a future `tt-metal`, and needs no
version bump. `GALAXY` and `TG` are likewise present.

**The constraint this puts on the code:** name only enum members the repo already depends on. A
`ClusterType` member that does not exist raises `AttributeError`, and inside the `try` that every
cluster probe needs, that is swallowed — silently disabling the detection rather than failing. The
first draft of `_is_blackhole_cluster` named `P100`/`P150`/`P150_X2`/`P150_X4` on no evidence and
would have degraded exactly that way. It now names only the three Galaxy members and defers
everything else to `ttnn.device.is_blackhole()`.

### 1.2 `get_logical_sku` has never been invoked, and would crash if it were **[repo]**

B0 item 3 describes it as arch-blind. It is, but two further facts change what the fix must be.

**Its argument is not a `MeshDevice`.** Both call sites pass `request.param` — an `int` or a
`(rows, columns)` tuple — because the function runs *before* `open_mesh_device`, to choose
`trace_region_size`, which is a device-open parameter. The body called `.get_num_devices()` on it,
which raises `AttributeError` for both shapes.

**That never fired because both paths are dead.** Nothing in the tree sets `TRACE_MODEL_KEY_PARAM`
(`"_trace_model_key"`) or `trace_model_key` in a `device_params` dict, so neither branch at
[tests/conftest.py:641](../tests/conftest.py#L641) is reachable.

Two consequences. B0 item 3 is a **forward-looking** fix — it starts mattering when a Blackhole
Galaxy trace node is registered in phase 4, not today — and the fix **cannot** be the plan's
suggested delegation to `device_utils.get_device_name`, which needs an open mesh. Cluster type is
the right probe precisely because it answers without one.

**Decision:** keep the signature, make the count extraction accept what callers actually pass, and
return `None` rather than a wrong SKU for an unrecognised parametrization.

**Accepted behaviour change:** a four-device Wormhole mesh now reports `N150x4` instead of
`P150x4`. `P150x4` was a Blackhole SKU being returned for Wormhole hardware. `N150x4` has no
`_SKU_ALIASES` entry, so it resolves no trace region and the runtime allocates dynamically —
correct, where the old answer silently resolved Blackhole's numbers. Dead path, so zero live risk.

### 1.3 `BHGLX` already normalizes to its own targets entry **[repo]**

`_SKU_ALIASES` in [examples/common/model_targets.py:32](../examples/common/model_targets.py#L32)
carries `bh_galaxy_perf: {bh_galaxy_perf, bh_galaxy, bhglx}`. With
[device_utils.py](../src/tt_transformers/device_utils.py)'s `32: "BHGLX"` and
`benchmarking_utils.py`'s `"BHGLX": "galaxy_bh"`, the repo already had three quarters of the
convention. Phase 1 extends it rather than inventing one; nothing needed a new token.

### 1.4 `num_links` — clamp, do not centralize **[inference]**

The plan says route `plans.py`'s hardcoded 4 / 3 / 1 through `get_num_links()`. Taken literally
that erases information: §4.3 of the plan itself warns that a collective's `num_links` and the
mesh's link budget are different quantities. Only the `4`s mean "every link". The `3` on the fused
QKV collective and the `1`s on the user gather and norm stats are tuned values measured below the
budget.

**Decision: `num_links=min(requested, budget)`.** On Wormhole `min(4,4)`, `min(3,4)`, `min(1,4)`
reproduce every existing literal exactly — pinned by
`test_wormhole_link_counts_are_unchanged_by_the_clamp`, which asserts the full ordered tuple for
both models. On Blackhole the 4s and the 3 become 2 and the 1s stay 1.

This matters because neither error mode raises: the ring and line CCLs index ethernet channels by
link, so over-requesting deadlocks the host with no traceback, and an under-specified
`num_links=1` on axis 1 has separately caused a real stall.

### 1.5 The link budget is keyed on `arch()`, not resolved through `get_num_links` **[repo]**

`tt_ccl.get_num_links` is the plan's preferred source and already returns `(2, 2)` for `BHGLX`.
But it reaches the architecture via `get_device_name` → `get_device_ids()` and
`ttnn.device.is_blackhole(mesh_device)` — a pybind call. Every Galaxy geometry test in
`tests/models/galaxy/` drives a `MagicMock(spec=ttnn.MeshDevice)`, which cannot answer either, and
[test_recipes.py:46-48](../tests/models/galaxy/test_recipes.py#L46-L48) already records that these
pybind bindings reject duck-typed stand-ins.

**Decision:** `GALAXY_FABRIC_LINKS` is keyed on `mesh_device.arch()`, which the existing host mocks
already provide and which `validate_galaxy_mesh` already reads. It fails closed on an unknown
architecture.

**The cost, stated rather than hidden:** two independent link tables now exist, and nothing on the
host can prove they agree. That cross-check is
[E02](bh_galaxy_experiments/README.md) and it is a one-liner on hardware.

### 1.6 `GALAXY_CCL_RESERVED_WORKER_CORES` moves with the links

It is documented as "one per fabric link, four links", so it is 2 on Blackhole and derived from the
same table. The name is kept, now defined as the Wormhole entry rather than a bare `4`, because the
qualified Wormhole recipes were written against it. New code should call
`galaxy_ccl_reserved_worker_cores(mesh_device)`. Getting this wrong warns and then segmentation
faults, so it is §8 Q5 and it is in the vault.

### 1.7 The taxonomy gate caught a real defect in this phase's own test

`audit_test_taxonomy` rejected the new `test_link_budget_follows_the_architecture_and_fails_closed`
for lacking the `model` mark under `tests/models/`. Worth recording because it is the concrete
argument for running these gates locally rather than banking them: the finding cost one command,
and would otherwise have surfaced on an allocated node.

---

## 2. Phase 2 — the topology descriptor, Wormhole only

### 2.1 The descriptor holds plain tuples, not `ttnn.CoreRangeSet` **[decision]**

The plan sketches `GalaxyChipTopology` with `worker_cores: ttnn.CoreRangeSet` and
`ring_cores: tuple[ttnn.CoreCoord, ...]`. It holds `(x, y)` tuples and `(x0, y0, x1, y1)`
rectangles instead.

Three reasons, and the third is the one that decided it:

1. A frozen dataclass wants hashable, comparable fields. Tuples are; pybind objects are not
   reliably so.
2. The golden-table test compares **order**, and tuple comparison is unambiguous about it.
   Asserting order through `CoreRangeSet` would depend on pybind equality semantics that this
   repo has never leaned on — it uses `num_cores()`, `bounding_box()` and `subtract()`, never
   `ranges()`.
3. **It makes the descriptor testable without a device.** `topology.py` touches `ttnn` only for
   enum values, so its entire validation surface is pure Python set arithmetic. All 19
   validation paths were verified on this Mac against a 12-line stand-in for those enums —
   including every rejection case. That would have been impossible with `CoreRangeSet` fields.

The recipes build `CoreRangeSet`s at the point of use, freshly, exactly as before. Nothing is
cached and shared, so no caller can mutate another's core set.

### 2.2 Callers were deliberately not moved

The plan expects `recipes.py`'s module-level functions to "become methods or take this
descriptor … mechanical but wide". They instead take an **optional** descriptor defaulting to
Wormhole:

```python
def worker_cores(topology: GalaxyChipTopology | None = None) -> ttnn.CoreRangeSet:
    return core_ranges(*_topology(topology).worker_core_ranges)
```

Zero call sites move, the Wormhole answer is byte-identical by construction, and phase 3 threads
a Blackhole descriptor through the same parameter. This matters more than usual here: phase 2
rewrites the geometry source of a hardware-qualified path whose bit-identical exit criterion
**cannot be run** — there is no Wormhole Galaxy access either. Minimising the diff is the only
mitigation available, and the golden test in `test_topology.py` is the other half.

### 2.3 The descriptor validates; it does not derive **[finding]**

§6.1 says "derive, never name". Applied to Wormhole it mostly cannot be: `_RING_CORE_COORDS` is
a physical walk of the chip, not a comprehension, which is why the reference port's Wormhole
tables are hand-written literals while its Blackhole ones are generated. Deriving them would
mean inventing an order, and order is the load-bearing part.

So the descriptor **names** the geometry and **derives the checks**. That is where the new value
is, and the checks are not hypothetical — each is a failure the reference or this project has
already paid for:

| Check | The failure it front-runs |
| --- | --- |
| every named core inside `compute_with_storage_grid_size()` | *"Tensor shard spec grid … must lie within compute grid"* |
| top-k / ring / norm / sampling inside the worker envelope | *"Kernel group cores do not match sub device cores"* |
| senders disjoint from workers | silent misplacement |
| reserved columns absent from workers | regressed prefill warmup, nothing raised |
| receiver mapping covers every worker core | *"Specified cores are not contained in associated GlobalCircularBuffer"* |
| senders ∪ receivers == the whole grid | the property that makes the 8 dummy entries load-bearing |
| sender count == receiver-group count | the lists are `zip`ped, so a mismatch silently truncates |

### 2.4 `WH_GALAXY_MESH_SHAPE` was **not** deduplicated, on purpose **[decision]**

§2.1 of the plan wants the duplicate constant in `mlp_2d.py` and `rmsnorm_2d.py` to become one
shared value. It stays duplicated, because the only shared home is `models/galaxy/topology.py`
and **`modules/` never imports `models/`** — verified: the dependency runs the other way at four
sites (`collectives.py`, `kv_contract.py`, `prefetch.py`, `resources.py` all import
`modules.*`), and never back. Deduplicating through `models` would invert the layering.

The right fix is for the capability record to be reachable from `modules/`, which is a phase-3
question about where `GalaxyCapabilities` lives. Deferred rather than solved by a layering
violation.

### 2.5 The two bare asserts are now raises (B9)

`mlp_2d.py` and `rmsnorm_2d.py` gated the mesh with bare `assert`, which `python -O` strips.
Both are now `TypeError`/`ValueError`. The regression test requires `ValueError` specifically:
an `assert` can only raise `AssertionError`, so demanding a different type is a direct proof
that the gate is not compiled away — and it needs no `-O` subprocess, which could not run here
anyway.

### 2.6 `validate_galaxy_mesh` generalizes on geometry, not on architecture

It now admits any architecture that **has a topology descriptor**, rather than a widened
allowlist. On Wormhole-only that is behaviourally identical to before; in phase 3 Blackhole
becomes admissible the moment its geometry is written down, and not one commit earlier. The mesh
shape stays a hard equality, because `(8, 4)` is architecture-invariant.
