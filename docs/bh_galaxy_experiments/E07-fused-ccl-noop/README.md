# E07 — Do the standard collectives carry data where the fused five no-op?

**Hardware: one 32-device Blackhole Galaxy. Specification, not a patch.**
**Depends on [E04](../E04-module-capability-gates/).**

**The most dangerous item in the port plan**, because the failure mode moves no data and raises
nothing, so the collective *appears* to run.

---

## Why it exists

Blackhole Galaxy needs a 2D-torus fabric. A 2D fabric uses a different packet header. Every CCL
kernel written only for the 1D header is therefore unavailable — and **the two ways that
manifests are not the same**:

**Safe.** The fused `all_gather_minimal_matmul_async` *fails to compile* against the 2D-torus
`HybridMeshPacketHeader`. Loud, and nothing to do about it.

**Dangerous.** The five galaxy-specific fused CCLs — `fused_rms_minimal`,
`llama_rs_create_heads`, `all_gather_concat`, `llama_rs_matmul`, `llama_reduce_scatter` — use
1D-multicast writers that **no-op**. The 2D fabric does *not* reject what it cannot run.

**Do not generalize the compile error into an expectation of loud failure.** That inference is
the trap, and it is stated here because it is easy to make.

Milestone 1 mostly dodges this: the prefetcher-free path already uses the standard collectives —
the ones the reference falls back *to* — and `GalaxyCapabilities.has_fused_ccl` is `False` on the
Blackhole descriptor. This experiment confirms that dodge actually works rather than assuming it.

## Hypothesis

Every collective the Blackhole decode path issues **moves data**. Specifically: each is a
standard/stable op pinned to the worker sub-device, and each produces a result that differs from
its input in the way a real collective would.

## What to build

The test has to be designed around the failure mode, and the failure mode is silence. So the
usual shape — run the op, check PCC — is insufficient: a no-op writer leaves the output buffer
holding whatever was there, and if that happens to be a zeroed buffer or a plausible-looking
prior value, PCC against a reference may not be the first thing to fail.

Build it as a **positive-control** suite:

1. **Pre-fill each collective's output buffer with a recognisable poison pattern** before the
   call — not zeros, which are too easy to confuse with a legitimate result.
2. Run the collective.
3. Assert the output is **not** the poison pattern. That is the no-op detector, and it is
   independent of whether the numerics are right.
4. *Then* assert PCC against a CPU reference.

Step 3 is the whole point. Without it, a no-op that happens to leave believable bytes behind
reads as a numerics problem and gets debugged as one.

5. Enumerate which ops the path actually issues, and assert **none of the five** appears. That
   list is: `fused_rms_minimal`, `llama_rs_create_heads`, `all_gather_concat`, `llama_rs_matmul`,
   `llama_reduce_scatter`.

## Run

```bash
MESH_DEVICE=BHGLX pytest tests/models/galaxy/test_collectives_bh_galaxy.py -sv
```

## What each outcome means

**Every collective moves data, PCC holds.** The prefetcher-free path's dodge is real. Record the
op list — it is the evidence that `has_fused_ccl=False` is wired correctly, and the deferred
prefetcher work will need exactly that list when it builds the hybrid route.

**A collective leaves the poison pattern.** A 1D-multicast writer reached the 2D fabric. Find
which op and which call site. This is the single most valuable failure to catch here, because
every downstream module would otherwise produce plausible, wrong numbers.

**One of the five fused ops appears in the issued list.** The capability flag is not reaching
that call site. An E04 wiring defect, not a fabric one — and worth fixing as a capability check
rather than an architecture branch, because collapsing the two axes is what the deferred
prefetcher work explicitly cannot afford.

**It hangs.** Mismatched topology and fabric. `plans.py` says `Topology.Ring` everywhere, which
is right; on Blackhole the pairing is `Topology.Ring ↔ FABRIC_2D_TORUS_XY`, **not** the Wormhole
`Topology.Ring ↔ FABRIC_1D_RING`. Carrying the Wormhole rule here reproduces the whole class of
opaque hang that caused most of the observed hangs across the Wormhole corpus.

---

## Result

*Not run.*
