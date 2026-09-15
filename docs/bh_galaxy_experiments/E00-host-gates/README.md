# E00 — Run the real host gates

**Hardware: none.** Linux x86_64 with `ttnn` installed. No Tenstorrent device is opened.

**Run this before anything else.** It is the only experiment that can tell you the deviceless
work is sound, and it costs no silicon.

---

## Why it exists

Phases 1–4 were written on a Mac, where `ttnn` cannot be installed — it publishes only
`manylinux_2_34_x86_64` wheels. Four gates and 307 host tests *were* run there, differentially,
in a venv holding everything except `ttnn` (the recipe is in
[bh_galaxy_port_log.md](../../bh_galaxy_port_log.md) §0). That is real coverage and it caught
real defects, but it leaves a specific, known hole:

**149 test files fail collection on `import ttnn`, and every Galaxy geometry suite is among
them.** So the tests that most directly exercise the new code — `test_topology.py`,
`test_recipes.py`, `test_plans.py`, `test_mlp_2d.py`, `test_rmsnorm_2d.py` — have never been
executed.

## Hypothesis

The seven host gates pass, and `pytest -m host` reports 2612–2613 passed.

## Run

```bash
# On a Linux x86_64 host; no device needed.
uv venv --python 3.12 && uv pip install -e '.[test,examples]'

pytest -m host                                          # expect 2612-2613 passed
python qualification/tools/audit_test_taxonomy.py
python tools/check_import_boundaries.py
python qualification/tools/run_hardware_matrix.py --validate
python qualification/tools/validate_bh_required_capabilities.py
ruff check . && ruff format --check .
mypy
```

The exabox partition survey lists a **`cpu_only`** partition. Try there first: this needs Linux
and `ttnn`, not silicon, and it should not cost a Galaxy window.

## What each outcome means

| Outcome | Meaning |
| --- | --- |
| All green, 2612–2613 passed | Phases 1–4 are sound. Proceed to E02. |
| `test_topology.py` fails | The most likely place. See below. |
| `ruff format --check` rewrites files | **Expected, and cosmetic.** See below. |
| Taxonomy/matrix/capability gates fail | Surprising — these four *were* run locally and were green. Suspect an environment difference, not the change. |

### Where `test_topology.py` is most likely to fail, and why

Its pure-Python half — the descriptor's fields and all 19 validation paths — *was* verified
locally against a stand-in for `ttnn`'s enums. What was **not** verified is everything that
touches real pybind objects:

- **`CoreRangeSet` equality.** The golden test asserts
  `recipes.ring_cores() == recipes.core_points(GOLDEN_RING_CORES)`. This relies on
  `CoreRangeSet.__eq__` existing and being order-sensitive. Order-sensitivity is well evidenced —
  it is the documented cause of the most expensive bug in the Wormhole corpus — but the operator
  itself was assumed.
- **`CoreRangeSet.subtract(...).num_cores()`** in the prefetch-mapping test. Used elsewhere in
  the repo, so lower risk.
- **`ttnn.CoreCoord` in a `MagicMock` return**, which `test_recipes.py` already does, so this
  should hold.

If `CoreRangeSet` equality turns out not to work, the fix is mechanical rather than a design
problem: compare `num_cores()` plus an explicit ordered walk, or compare `repr()`. **Do not
weaken the test to membership-only** — order is the load-bearing property, and a ring built from
a `set` produced bit-for-bit identical *wrong* output across runs, which is what made it look
like anything except an ordering bug for weeks.

### The `ruff format` caveat

`uv tool run ruff` resolves 0.16.7; the repo pins `>=0.11.0`. The two **formatters** disagree,
not just the `I001` lint — a single `ruff format` over `src/ tests/ qualification/` rewrote 16
files, 14 of them unrelated. Those were reverted, but it means the files this work touched are
internally consistent under 0.16.7 and may still be reformatted by the pinned version. That is
cosmetic. Accept the reformat as a separate commit; do not read it as a defect.

---

## Result

### 2026-09-15, `wh-glx6u-05`, ttnn 0.77.0, Python 3.10.21 — green, and it earned its place

**The hypothesis was wrong in the way that mattered.** It predicted the gates pass. On the first
run, `pytest -m host` did not even collect, and four tests failed once it did. Three were real
defects in the new code's tests, invisible until now for exactly the reason this experiment
exists: `tests/models/galaxy/` fails collection on `import ttnn`, so those assertions had never
executed.

Final state on `858bb9f`:

```
pytest_host        2641 passed, 87 skipped, 6898 deselected, 81 subtests passed, 1 failed  (77s)
taxonomy           exit 0
import_boundaries  exit 0
support_docs       exit 1   <-- 8 broken links, by decision; see below
hw_matrix          exit 0
bh_capabilities    exit 0
ruff check         All checks passed!          (ruff 0.11.0, CI scope)
ruff format        431 files already formatted (ruff 0.11.0, CI scope)
mypy               Success: no issues found in 6 source files  (mypy 1.15.0)
```

**The count is ~2641, not the predicted 2612–2613.** The prediction was made from a Mac venv
without `ttnn`; 2642 host tests collect here.

### What it found

**Three stale Galaxy assertions**, all fixed:

- `test_galaxy_mesh_gate_is_an_allowlist_keyed_on_available_geometry` asserted
  `supported_galaxy_architectures() == (WORMHOLE_B0,)`. Phase 3 registered `_resolve_blackhole`,
  so it returns both — which is the point of phase 3, and which `validate_galaxy_mesh`'s own
  docstring documents.
- `test_mesh_validation_requires_wormhole_galaxy` expected a *"Wormhole only"* rejection of a
  Blackhole mesh, for the same reason.
- `test_descriptor_validates_against_a_live_grid` expected `resolve_galaxy_chip_topology` to
  reject a device reporting 8 DRAM views. It cannot: `_resolve_wormhole` builds the descriptor
  *from* the reported count, so the guard compares a value with itself and is unreachable by that
  path.

Both fail-closed halves were kept rather than deleted, retargeted at `ttnn.device.Arch.QUASAR`,
which genuinely has no descriptor.

**One real lint error, which the wrong ruff had hidden.** Under the locked 0.11.0 the tree has
exactly one `I001`, in `tests/host/test_galaxy_arch_taxonomy.py` — a file this branch added; `main`
is clean. Under the unpinned 0.16.7 the same tree reports **12** errors, 11 of them on files
`main` also "fails", which made the whole gate look like a version artifact to wave through. The
predicted "expected and cosmetic" reformat is likewise not real: 0.11.0 reports 431 files already
formatted, and no reformat commit is needed.

**`test_topology.py`'s `CoreRangeSet` equality held.** The README's most-likely-failure prediction
did not fire: `CoreRangeSet.__eq__` exists, is order-sensitive, and the golden tables pass, as does
`subtract(...).num_cores()`. That risk is now closed by measurement.

### The one remaining failure, and why it stays

`validate_support_docs.py` reports 8 broken local links, and
`tests/host/test_support_documentation.py` wraps it, so `pytest -m host` is red on the same cause:

```
docs/bh_galaxy_experiments/README.md: broken local link '../blackhole_galaxy_port_plan.md'
docs/bh_galaxy_experiments/README.md: broken local link '../../../my-tt-dev-tools/exabox/README.md'
docs/bh_galaxy_experiments/README.md: broken local link '../bh_galaxy_deferred_work.md'
docs/bh_galaxy_handoff.md: broken local link 'blackhole_galaxy_port_plan.md'   (x2)
docs/bh_galaxy_port_log.md: broken local link 'blackhole_galaxy_port_plan.md'
docs/bh_galaxy_port_log.md: broken local link 'bh_galaxy_deferred_work.md'
docs/bh_galaxy_port_log.md: broken local link '../../my-tt-dev-tools/exabox/README.md'
```

This is a **deliberate docs-location decision, not a code defect**: `blackhole_galaxy_port_plan.md`
and `bh_galaxy_deferred_work.md` are personal development documents and now live in
`my-tt-dev-tools/agents-context/tttv2-2d-modules-bh/`, not in this repository. The remaining port
documents here — this vault, the handoff, the port log — are slated to follow them, leaving only
user- and contributor-facing docs in tree. Until that move happens the links dangle.

**Worth knowing regardless: those links were always broken in CI.** The validator resolves links
on disk, so `../../../my-tt-dev-tools/exabox/README.md` only ever resolved because both repos are
siblings on the development laptop. It can never resolve in a clean checkout, and it predates the
docs decision.
