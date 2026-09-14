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

*Not run.*
