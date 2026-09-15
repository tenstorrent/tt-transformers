# Blackhole Galaxy — banked experiments

**Everything the deviceless work could not answer, prepared so that someone with a machine can
answer it without re-deriving anything.** Each folder holds one question, what it decides, the
exact command to run, and — where code has to change — a patch to apply.

Written because the port's phases 1–4 are complete and unrunnable: there is no Blackhole Galaxy
and no Wormhole Galaxy access from here. See
[bh_galaxy_port_log.md](../bh_galaxy_port_log.md) for what was built and why, and
[blackhole_galaxy_port_plan.md](../blackhole_galaxy_port_plan.md) for the plan these serve.

---

## How to use this

1. Read the folder's `README.md`. It states the hypothesis, the command, and — the part that
   matters — **what each outcome means for the code**. An experiment whose result you cannot act
   on is not worth a hardware window.
2. Apply `patch.diff` if the folder has one: `git apply docs/bh_galaxy_experiments/<E>/patch.diff`.
   Every patch records the SHA it was generated against. If it fails to apply, the file moved;
   re-read the target and redo the change by hand rather than forcing it.

   **`spec` in the table below means there is deliberately no patch.** For E05–E07 the shape of
   the change depends on what E04 reveals, and a plausible-looking patch for an unmeasured
   failure is worse than none — it invites someone to apply it and believe the result. Those
   folders specify the *test* precisely instead, which is the part that does not depend on the
   outcome.
3. Run the command. Capture the whole log, not the exit code.
4. **Write the result back into the folder's `README.md`** under `## Result`, and update the
   `Status` column below. A measured answer that stays in a terminal has to be measured again.

### Rules that apply to every device run here

These are not ceremony; each one is a failure the Wormhole project already paid for.

- **One pytest node id per process.** The ttnn program cache belongs to the mesh device and the
  process, so two models in one process share compiled programs and buffer addresses and hang at
  the second model's first decode. The weight cache is keyed on `MeshDevice.id()`, so a
  multi-test process silently regenerates every weight — measured at 4–5× slower and 899.3 GB
  written in one three-hour window.
- **Re-run an already-qualified numerical control after any reset.** A reset that reports success
  does not guarantee a healthy mesh: one episode chased for hours as a tracing defect was mesh
  state, giving 8 distinct answers out of 12 with logit magnitudes 1e14–1e17 against a healthy
  ~31 peak.
- **`ls /dev/tenstorrent | wc -l` is not a health check** — use `/sys/class/tenstorrent`. Reset
  with `tt-smi -glx_reset`, not `-r`; neither recovers a board off the PCIe bus.
- **Three fresh processes, byte-identical output** is the bar for any correctness claim, not
  three PCC passes. The fused-RMSNorm stats CB aliases whatever the allocator left on the first
  core of the norm shard grid and has produced PCC of 0.0977 / 0.1394 / 0.1701 / 0.9999 across
  processes **on an unchanged test**.
- **A wrong `HF_HOME` makes `hf_config_or_skip` skip every real-checkpoint test**, so the run
  looks green and measured nothing. `LLAMA_DIR` / `HF_MODEL` unset makes a Galaxy test open and
  close the mesh cleanly with zero correctness signal. Do not mistake clean teardown for a
  result.

### Where to run

The harness is in [`my-tt-dev-tools/exabox/`](../../../my-tt-dev-tools/exabox/README.md): Claude
on the laptop, jobs on the cluster, the login node as control plane only. Reserve **one**
`bh-glx-*` node — one node is one 32-chip Blackhole Galaxy; the `*_podN` partitions are 4-host
scale-out nobody here needs.

One note that is not yet in that README: the partition survey lists a **`cpu_only`** partition.
E00 needs Linux and `ttnn`, not silicon, so it is worth trying there before spending a Galaxy
window on it.

---

## The index

Tier 1 blocks or validates work already written. Tier 2 is phase-5 bring-up, banked so the first
window is spent measuring rather than authoring.

| # | Question | Decides | Hardware | Ships | Status |
| --- | --- | --- | --- | --- | --- |
| [E00](E00-host-gates/) | Do the real host gates pass? | whether phases 1–4 are sound at all | none — Linux + `ttnn` | commands | **not run** |
| [E01](E01-wh-byte-identical/) | Is Wormhole decode byte-identical after the phase-2 refactor? | whether the topology descriptor may keep the qualified path | **WH** Galaxy | procedure + script | **not run** |
| [E02](E02-bh-day0-probe/) | What does a Blackhole Galaxy actually report? | §8 Q1, Q3, Q4, Q5 — four open questions in one read-only run | BH Galaxy | in-tree test | **not run** |
| [E03](E03-worker-envelope/) | Does column 0 rejoin the workers, and does the row cap apply? | §8 Q2 — worth ~22 worker cores | BH Galaxy | **patch**, 3 arms | **not run** |
| [E04](E04-module-capability-gates/) | Do the 2D modules accept a Blackhole mesh once gated on capability? | phase 5's first task, and it unblocks every module suite | BH Galaxy | **patch** | **not run** |
| [E05](E05-multilayer-residual/) | Does the residual survive more than one layer? | B5 — invisible at one layer, wrong at eighty | BH Galaxy | spec | **not run** |
| [E06](E06-mlp-dram-interleaved/) | Does DRAM-interleaved MLP decode hold PCC? | B3 — the silent PCC≈0 channel-order failure | BH Galaxy | spec | **not run** |
| [E07](E07-fused-ccl-noop/) | Do the standard collectives carry data where the fused five no-op? | B4 — the most dangerous item in the plan | BH Galaxy | spec | **not run** |

### Suggested order

**E00 first, always** — it needs no silicon, and it is the only thing that can tell you the
deviceless work is sound before you spend a window on it. **E01's host half is the same shape**:
its `dump_resolved_geometry.py` diff needs `ttnn` but no device, and it is a stronger check on
phase 2 than the device run it precedes. Both belong on any Linux box before any allocation.

[bh_galaxy_handoff.md](../bh_galaxy_handoff.md) stages all of this against a real Wormhole
window, with times and a failure playbook.

Then, on the first Blackhole window: **E02**, which is read-only and answers four questions in
about as long as opening the mesh takes. Then **E03**, whose arms are one-line constant flips,
then **E04**, which unblocks everything after it.

**E01 is on a different machine and a different schedule.** It is the highest-value single
experiment here — it is the only thing that can confirm the phase-2 refactor did not move a
hardware-qualified path — but it needs Wormhole Galaxy, and the port can proceed without it at
the cost of carrying that risk.

E05–E07 are the bring-up sequence, cheapest first, and they follow the ordering the Wormhole
evidence established: norm before MLP before attention.

---

## What is deliberately *not* here

- **Anything the prefetcher needs.** Milestone 1 is prefetcher-free; the deferred work and its
  own open questions are in [bh_galaxy_deferred_work.md](../bh_galaxy_deferred_work.md).
- **Performance numbers.** No Blackhole performance figure exists anywhere in the reference, both
  Blackhole Galaxy CI SKUs upstream are `release_ready: false`, and the reference's prefetcher
  configuration forfeits every fused collective — so the Wormhole speedup does not carry by
  construction. The payoff is unmeasured, not merely unported. **Do not invent a number here.**
- **`concat-32` batched prefill.** Unproven on Wormhole, where it does not fit L1 at any
  supported length for either model.
