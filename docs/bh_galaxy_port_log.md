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

**Caveat — `ruff` version skew.** The repo's `pyproject.toml` pins `ruff>=0.11.0`; `uv tool run
ruff` resolves 0.16.7, whose `I001` disagrees with the pinned version about whether `qualification`
and `tests` are first-party. It reports 11 pre-existing files as un-sorted.

**Correction (2026-09-15): do not treat `I001` as advisory — pin the version instead.** This entry
originally said to treat `I001` as advisory locally and match the surrounding convention. That
advice cost a real defect. The version of record is not `>=0.11.0` at all: CI installs the
hash-locked **`ruff==0.11.0`** from `constraints/locks/build-dev-py310.txt`, which is also the rev
pinned in `.pre-commit-config.yaml`, and it scopes both ruff commands to
`src tests examples tools qualification` rather than `.`.

Run that way, the tree has exactly **one** `I001` — in `tests/host/test_galaxy_arch_taxonomy.py`,
a file this work added, while `main` is clean — and `ruff format --check` reports **431 files
already formatted**. Run the other way, 12 errors and 18 reformats, 11 and 16 of which `main`
"fails" too. So the noise did not merely obscure the one real finding; it made the entire gate look
like a known version artifact, which is how the finding survived to be caught on a Linux host
instead of here. Two commands are enough to remove the ambiguity:

```bash
uvx ruff@0.11.0 check src tests examples tools qualification
uvx ruff@0.11.0 format --check src tests examples tools qualification
```

Pinning also makes the Mac gate *stronger* than this section claimed, not weaker: with the right
version, `ruff` locally is exactly what CI will say.

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

---

## 3. Phase 3 — the Blackhole resolver

### 3.1 Blackhole's geometry is derived; Wormhole's is refused if it moves **[decision]**

The two architectures get opposite treatment, and that is deliberate.

Wormhole's resolver **refuses** any grid but `7 x 10`. Its core tables were hand-measured
against exactly that shape, and reinterpreting hand-measured coordinates against a grid they
were never checked on is how a silent misplacement happens.

Blackhole's resolver **derives** the envelope from whatever the device reports, because
harvesting is per-part: `12 x 10` is the reference's chassis (the 1x-harvested key), and `13 x 10`
unharvested and `11 x 10` are shapes a real board can present. All three resolve, with the worker
columns and the dispatch column expressed against the reported width.

That is what makes the plan's phase-3 exit criterion meaningful rather than three copies of a
literal.

### 3.2 The dispatch column is the one thing not derived

It sits *inside* `compute_with_storage_grid_size()`, so a purely derived envelope folds it into
the workers — and the reference records that doing so *"regresses prefill warmup"*, with nothing
raising. `reserved_columns` carries it, and `validate_against_device` rejects any descriptor
whose workers reach into it.

This is the concrete case §6.1's rule has to bend for: "derive, never name" applies to geometry
that follows from the grid, and cannot apply to a fact only measurement knows.

### 3.3 §8 Q2 is one named constant with one switch point **[measure]**

The largest genuinely unanswered geometry question — worth ~22 worker cores — is two linked
unknowns, and both are now single constants in `topology.py` with their reasoning attached:

- `BLACKHOLE_FIRST_WORKER_COLUMN = 1`. The reference excludes column 0 *because its prefetcher
  senders live there*. Milestone 1 has no senders, so `cols 0..10` is plausible — but that is an
  inference, and the reference has no prefetcher-free Blackhole worker range to copy.
- `BLACKHOLE_SUB_DEVICE_MAX_Y = None` (rows 0-9). The reference's `sub_core_max_y = 7` exists
  *"to match the 24-core ring geometry"*; with no ring there is no extent to match. It is written
  unconditionally rather than gated on `use_prefetcher`, and the reference's own Llama path does
  not apply it at all.

**Conservative on the first, optimistic on the second**, because the failures are not
symmetrical. Too few worker cores costs throughput and nothing else. Too many puts tensors on a
column reserved for a reason nobody wrote down — and every failure in that class is silent. The
row cap is different: it has a *recorded* rationale that demonstrably does not apply here, so
inheriting it would be cargo-culting a constraint, not being careful.

Settling either is a one-line change. `E05` in the vault is the experiment.

### 3.4 The link table is now one table **[correction]**

Phase 1 put `GALAXY_FABRIC_LINKS` in `recipes.py`; phase 3 would have made it a second source
beside `GalaxyChipTopology.fabric_links`. It now lives in `topology.py`, both resolvers read it,
and `recipes` re-exports it. `ccl_reserved_worker_cores` is a property over the same number, so
"one per fabric link" cannot drift from the link count.

`lm_head_reduce_core_count` grew a `reserved_worker_cores` parameter defaulting to the Wormhole
value. It subtracted the Wormhole `4` unconditionally, which on Blackhole would have reserved
twice what the fabric needs — not a failure, but wrong for a reason that would have been hard to
find later, since starving that collective segmentation-faults rather than raising.

### 3.5 `topology.py` deliberately imports nothing but `ttnn` **[decision]**

`l1_small_size` was briefly an import of `device_utils.GALAXY_L1_SMALL_SIZE`. That pulled
`lazy_weight`, `loguru` and `torch` into a geometry module — and, concretely, **broke the ability
to exercise the descriptor without a device**, which is the single most valuable verification
tool this session found.

It is a literal again, with a host test asserting equality with the shared constant. The trade is
explicit: the drift risk is a one-line host-testable equality, while the testability it buys is
not replaceable. Keep `topology.py`'s import surface at `ttnn` alone.

### 3.6 `NullPrefetcher2D` honours the protocol rather than bypassing it

Prefetcher-free is a *design task* here, not a config change: every prefetched decode weight is
registered with `Prefetcher2D` and the module configs receive its resolved contexts. So the
absence is an object implementing register → seal → activate → cleanup, returning contexts with
`global_cb=None` and `sub_device_manager_id=None`, leaving weights DRAM-interleaved.

Two checks are kept rather than dropped as "not applicable", because neither is about the
prefetcher:

- duplicate and count checks on registration, which catch model-side wiring mistakes;
- `borrow_context`'s sub-device policy check. A module that disagrees about the partition places
  tensors on cores the loaded sub-device manager does not own and aborts with *"Kernel group
  cores do not match sub device cores"* — a failure that does not care whether a prefetcher
  exists.

`sub_device_id` still resolves, so confined matmuls are told their sub-device instead of silently
defaulting to sub-device zero — the prefetch senders. The entire global-CB apparatus is untouched
and merely unused, so the deferred work is additive.

---

## 4. Phase 4 — matrix and capability registration

### 4.1 `ALLOWED_MESH_DEVICES` is what forces a node to exist **[finding]**

`run_hardware_matrix.validate_matrix` requires **every** mesh name in
`ALLOWED_MESH_DEVICES` to be covered by at least one node. So adding `BHGLX` is not a
formality — it makes a Blackhole Galaxy node mandatory, and the node's selector must resolve to
a real test function in a real file. There is no way to register the mesh name and defer the
node.

That is a good constraint and it decided what phase 4 ships.

### 4.2 The node selects a topology probe, not a ported module **[decision]**

The obvious mirror of the TG proposal node would be a Blackhole RMSNorm2D suite. That would be
a large piece of unverifiable code, and it would fail for an uninteresting reason:
**the 2D module gates still accept Wormhole only** (§4.4).

`tests/models/galaxy/test_topology_bh_galaxy.py` is the better first artifact. It opens the
mesh, reads what the device reports, and resolves the descriptor against it — answering four of
the plan's five §8 open questions in one cheap, read-only run that allocates no tensor and runs
no collective, so it cannot leave the mesh needing a reset:

| Probe | Question |
| --- | --- |
| grid and DRAM views, descriptor resolves | §8 Q1 shape, and every containment check at once |
| per-device grid enumeration | §8 Q1 uniformity — the part that is genuinely unknown |
| `fabric_links` vs `tt_ccl.get_num_links` | §8 Q5, and the cross-check §1.5 could not do on the host |
| `has_l1_small_region` + bank sizes | §8 Q4 |
| worker envelope vs dispatch column | the §4.2 rule, plus a report of what `BLACKHOLE_FIRST_WORKER_COLUMN` costs |

Opening the mesh *at all*, with `FABRIC_2D_TORUS_XY`, is itself the §3 measurement.

### 4.3 The capability geometry is `additive_functional`, and Qwen-only

Declaring the geometry makes it visible to the contract validator. Declaring it **required**
would assert a coverage obligation no measurement supports. `role: additive_functional` is the
honest encoding, and a host test pins it there — promoting it means deleting that test.

It is declared on **Qwen3-32B alone**. Qwen is tier-1 CI on both architectures upstream, so
"the reference runs on Blackhole Galaxy" means *Qwen* runs on it. Llama-3.3-70B Galaxy has never
run on Blackhole anywhere — its demo hardcodes `fabric_config: True` with no 2D-torus variant,
its CI lists only the Wormhole Galaxy SKU, its arg bag has no Blackhole opt-in, and its
model-level test is explicitly skipped for Blackhole. A geometry entry there would be fiction.

### 4.4 The 2D module gates still accept Wormhole only — deliberately, and this is phase 5's first task

§6.2 wants the module gates to become capability checks. They have not, and the reason is §2.4's
layering: `GalaxyCapabilities` lives under `models/`, and `modules/` never imports `models/`.

The clean fix is **injection, not import** — the capability record reaching `MLP2D` and
`RMSNorm2D` through their configs, the way `tt_ccl` and the prefetch contexts already do. That
is a module-surface change that wants a hardware run to mean anything, so it belongs at the top
of phase 5 rather than bolted onto a deviceless phase.

Until then a Blackhole mesh is rejected by the modules by design, with a clear message. The
matrix node is `enabled: false` and its probe deliberately needs no module, so nothing here
claims otherwise.

### 4.5 Caveat — do not run `ruff format` with the unpinned version **[correction]**

`uv tool run ruff` resolves 0.16.7 against the repo's pinned `ruff>=0.11.0`, and the two
**formatters** disagree, not only the `I001` lint. A single `ruff format` over `src/ tests/
qualification/` rewrote 16 files, 14 of them unrelated to this work; they were reverted.

Two rules follow. Never run `ruff format` over a directory here — name the files. And treat
`ruff format --check` under the pinned version as **unverified** for the files this work
touched: they are internally consistent under 0.16.7, and the first real gate run may still
reformat them. That is cosmetic, but it will show up as a diff.

---

## 5. The experiment vault

[bh_galaxy_experiments/](bh_galaxy_experiments/) banks the eight questions the deviceless work
could not answer. Three decisions about its shape are worth recording, because they are what
make it usable rather than decorative.

**Every folder states what each *outcome* means for the code, not just the hypothesis.** An
experiment whose result nobody can act on is not worth a hardware window. The outcome tables are
the largest part of each folder, deliberately, and several of them say "keep the current
behaviour and write down why" — a negative result that gets recorded is worth as much as a
positive one here, because the alternative is that the next person redoes the same inference.

**Patches only where the change is already known.** E03's three arms are one-line constant
flips and E04's is a design change that phase 3 deliberately deferred; both are real, generated
against a pinned SHA, and verified to apply. E05–E07 ship a **specification of the test** and no
patch, because the shape of their fix depends on what E04 reveals. A plausible-looking patch for
an unmeasured failure is worse than no patch — it invites someone to apply it and believe the
result.

**E02 is a committed test, not a patch.** `tests/models/galaxy/test_topology_bh_galaxy.py`
answers four of the plan's five open hardware questions in one read-only run, and it is the
selector the disabled matrix node points at. Landing it in the tree rather than in the vault
means it is subject to the taxonomy and matrix gates like anything else.

### One thing found while writing it

The exabox partition survey lists a **`cpu_only`** partition, which
[the harness README](../../my-tt-dev-tools/exabox/README.md) §6 does not mention. E00 — the real
host gates — needs Linux and `ttnn`, not silicon. If `cpu_only` can host a venv, the host gates
come off the critical path without spending a Galaxy window, which is precisely the problem that
section says it has no good answer for.

---

## 6. Where this leaves the port

**Phases 1–4 are complete and committed.** As of 2026-09-15 the Wormhole window has closed the
larger half of the verification gap; what remains open is Blackhole, which has still been measured
zero times.

| | State |
| --- | --- |
| Seven host gates | **green** on Linux + `ttnn`, ttnn 0.77.0 — `wh-glx6u-05`, 2026-09-15 |
| `ruff check` / `ruff format` | **both green** under the locked `ruff==0.11.0` at CI's scope |
| `mypy` | **green**, 6 source files, mypy 1.15.0 |
| Host tests | **2641 passed**, 87 skipped, 81 subtests; 1 failed, a doc-link gate held red by a docs-location decision |
| Galaxy geometry suites | **executed for the first time**; found 3 stale assertions, all fixed |
| Phase 2, host-resolved geometry | **byte-identical** to `0a3e045`: 395 fields, same sha256 ([E01](bh_galaxy_experiments/E01-wh-byte-identical/)) |
| Wormhole Galaxy module suites | **56 / 57 green** on device — exactly the pre-refactor baseline; the one red prints `PASSED` then hangs in teardown |
| Numerical stability, 3 fresh processes | MLP and fused-residual RMSNorm decode PCC **identical to 16 digits** |
| Topology descriptor, all validation paths | exercised directly, including every rejection case |
| Blackhole resolver, 11x10 / 12x10 / 13x10 | exercised directly, **on host only** |
| Anything on Blackhole silicon | **never run** |

Where the two risks now stand:

1. **Closed. Phase 2's exit criterion is met on both halves.** Every host-resolved memory config,
   program config, core range set, sub-device partition, ordered `num_links` tuple and collective
   spec is byte-identical to the pre-descriptor commit, for both models in both modes; the device
   run reproduces the pre-refactor 56/57 exactly; and the two highest-signal suites give PCC
   identical to 16 digits across three fresh processes. It is no longer true that "the golden
   tables check the resolver and nothing checks the callers" — **the topology descriptor may keep
   the qualified path**, and this is the evidence the refactor was missing.
2. **The 2D module gates still accept Wormhole only.** Unchanged, and worth being precise about
   now that the two gates have diverged: the *mesh* contract, `recipes.validate_galaxy_mesh`,
   generalised in phase 3 and admits any architecture with a topology descriptor — which is what
   three of E00's failures were about. The *module* gates did not: `embedding_2d`, `rmsnorm_2d`,
   `rope_2d`, `lm_head_2d`, `sampling_2d` and `prefetcher_2d` each still test
   `mesh_device.arch() != ttnn.device.Arch.WORMHOLE_B0` directly. So a Blackhole mesh now passes
   the mesh gate and is then refused by every module. That is phase 5's first task and it is
   specified in [E04](bh_galaxy_experiments/E04-module-capability-gates/), patch included.

A third item, new and cheap to state: **the tooling advice in this log was wrong in a way that hid
a defect**, corrected in §0 above. An unpinned `ruff` reports 12 errors where the locked one
reports 1, and the one real error was inside the noise.
