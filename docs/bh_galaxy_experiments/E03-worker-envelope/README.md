# E03 — The prefetcher-free worker envelope

**Hardware: one 32-device Blackhole Galaxy. Three patches, one per arm.**
**Base SHA: `746bb77`** (`src/tt_transformers/models/galaxy/topology.py` only).

This is the largest genuinely unanswered geometry question in the port, and it is worth roughly
**22 worker cores** — about 22 % of the envelope.

---

## Why it exists

The reference port's Blackhole worker envelope is `cols 1..10 x rows 0..7`. Two of those bounds
exist for reasons that **do not apply to a prefetcher-free path**, and the reference has no
prefetcher-free Blackhole worker range to copy.

**Column 0.** The reference excludes it because that is where its prefetcher senders live.
Milestone 1 has no senders, so there is no such reason — `cols 0..10` is plausible, worth 10
cores on a `12 x 10` grid. That is an inference, not a reading.

**Rows 8–9.** The reference sets `sub_core_max_y = 7`, and its recorded rationale is *"to match
the 24-core ring geometry"* — 8 rows × 3 columns. With no ring there is no extent to match. Two
facts corroborate treating it as prefetcher baggage: it is written `7 if is_blackhole else 9`
rather than gated on `use_prefetcher`, and the reference's own Llama path does not apply it at
all. Worth 12 cores.

## The current position, and why it is split

`topology.py` ships **conservative on the column, optimistic on the rows**:

```python
BLACKHOLE_FIRST_WORKER_COLUMN = 1          # conservative: keep column 0 out
BLACKHOLE_SUB_DEVICE_MAX_Y: int | None = None   # optimistic: rows 0-9
```

That asymmetry is deliberate. The failures are not symmetrical:

- **Too many worker cores** puts tensors on a column reserved for a reason nobody wrote down, and
  every failure in that class is silent. So the column stays out until measured.
- **The row cap is different** — it has a *recorded* rationale that demonstrably does not apply
  without a ring. Inheriting it would be cargo-culting a constraint, not being careful.

## Hypotheses

| Arm | Patch | Envelope at `12 x 10` | Cores | Hypothesis |
| --- | --- | --- | --- | --- |
| **A** | *(none — as shipped)* | cols 1–10 × rows 0–9 | 100 | works |
| **B** | `arm-b-column-0.diff` | cols 0–10 × rows 0–9 | 110 | works, and is 10 cores better |
| **C** | `arm-c-row-cap-7.diff` | cols 1–10 × rows 0–7 | 80 | works; the reference's exact envelope |
| **D** | `arm-d-both.diff` | cols 0–10 × rows 0–7 | 88 | control for interaction |

## Run

```bash
# Arm A first, unpatched, as the control.
MESH_DEVICE=BHGLX pytest tests/models/galaxy/test_topology_bh_galaxy.py -sv

# Then each arm, one at a time, reverting between.
git apply docs/bh_galaxy_experiments/E03-worker-envelope/arm-b-column-0.diff
MESH_DEVICE=BHGLX pytest tests/models/galaxy/test_topology_bh_galaxy.py -sv
git checkout -- src/tt_transformers/models/galaxy/topology.py
```

**The probe alone is necessary but not sufficient.** It confirms the envelope resolves and
validates, which is a host-side property. What it cannot see is the empirical reason column 11
is excluded in the first place — *prefill warmup*. Once [E04](../E04-module-capability-gates/)
lands and the modules accept a Blackhole mesh, re-run the winning arm against a real prefill and
compare warmup time against arm A. **That is the measurement that actually decides arm B.**

## What each outcome means

**B resolves and prefill warmup is unchanged.** Adopt it: set
`BLACKHOLE_FIRST_WORKER_COLUMN = 0` and record the measurement next to the constant, replacing
the inference that is there now. Ten cores.

**B resolves but prefill warmup regresses.** *Keep arm A*, and this is the important outcome —
it means column 0 carries the same unwritten property as column 11 on this chassis, which
nothing in the reference would have told you. Record it next to the constant in those terms, so
nobody re-derives the inference and tries again.

**C is required (A or B fails where C passes).** The row cap is **not** prefetcher baggage, and
the reference's unconditional `7 if is_blackhole else 9` is right for a reason its own comment
misstates. That is worth knowing beyond this experiment: it means other constants the reference
justifies by ring geometry may also have a second, unrecorded reason, and the deferred prefetcher
work should re-examine them rather than trusting the stated rationale.

**D behaves differently from B and C individually.** The two bounds interact, which nothing
predicts. Do not adopt anything; record the observation and treat the envelope as needing real
characterization rather than a two-constant choice.

### Whatever the outcome

Both constants must stay **named, with the measurement written beside them**. The value of this
experiment is not the extra cores — it is replacing two inferences with two facts. A silently
adopted number is worth much less than a documented one, because the next person to look at a
prefetcher-free envelope will otherwise redo exactly this reasoning.

---

## Result

*Not run.*

Record here: per arm, whether the descriptor resolved, the worker core count, and — once E04
lands — prefill warmup against arm A.
