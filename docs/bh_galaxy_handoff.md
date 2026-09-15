# Blackhole Galaxy port — handoff

**Written for: whoever picks this up next with machine access** — an engineer or an agent session
with SSH to a Wormhole Galaxy now, and to a Blackhole Galaxy later. It assumes you have not read
the rest of `docs/`.

Branch: `tttv2-galaxy-2d-modules-port`. Six commits, **none pushed**, no PR open.

---

## 1. Where the work stands in one paragraph

Phases 1–4 of [blackhole_galaxy_port_plan.md](blackhole_galaxy_port_plan.md) — everything that
needs no hardware — are written and committed. The Galaxy geometry now lives in a validated
per-architecture descriptor, Blackhole resolves against it, and a disabled matrix node and
capability geometry are registered. **What has never run is anything that imports `ttnn`**, which
includes every Galaxy test suite. Four host gates and 307 host tests were run on a Mac in a
`ttnn`-free venv, differentially; that is real but partial coverage, and
[bh_galaxy_port_log.md](bh_galaxy_port_log.md) §0 and §6 say exactly what it does and does not
cover.

So the job now is to close that gap, in a specific order, and the Wormhole Galaxy window closes
most of it.

---

## 2. The Wormhole window: what it is actually for

You are reserving a Wormhole Galaxy. It gives you **two** things, and people forget the second:

1. **A 32-device Wormhole Galaxy mesh** — needed for exactly one experiment,
   [E01](bh_galaxy_experiments/E01-wh-byte-identical/).
2. **A Linux x86_64 host with `ttnn` installable** — which is all that
   [E00](bh_galaxy_experiments/E00-host-gates/) and the geometry diff need. No device is opened.

**Most of the value in this window is in (2), and it costs no device time.** Do those first, and
do not touch the device until they are green.

### Why E01 matters more than anything else on this machine

Phase 2 replaced every intra-chip core coordinate in `recipes.py` and `prefetch.py` with a
topology descriptor. That is a rewrite of the geometry source of a path that **was qualified on
silicon** — 56 of 57 cheap module ids green, ~50 minutes of device time. The port plan's own exit
criterion for that phase is a Wormhole run with bit-identical decode output, and it says the
criterion is not optional.

It has never run. Two host tests pin the *resolver*; nothing pins the *callers*. That is the gap.

---

## 3. The run plan, staged

Four stages. **Each one gates the next.** Times are wall-clock estimates with the reasoning
attached, so you can tell whether you are on track or something is wrong.

| Stage | What | Device? | Estimate |
| --- | --- | --- | --- |
| 0 | Environment | no | 15–25 min |
| 1 | [E00](bh_galaxy_experiments/E00-host-gates/) — the real host gates | no | 10–20 min |
| 2 | Host-side geometry diff | no | ~2 min |
| 3 | [E01](bh_galaxy_experiments/E01-wh-byte-identical/) — device confirmation | **yes** | 50–70 min |

**Reserve ~3 hours.** About one of those is device time. If the window is shorter, stages 0–2 run
on any Linux box with `ttnn` — including, per the exabox partition survey, the `cpu_only`
partition — so only stage 3 truly needs the Galaxy.

### One scheduling fact worth knowing up front

**No Galaxy module suite touches a checkpoint.** All eight `tests/modules/*/test_*_2d_wh_galaxy.py`
files and all three `tests/models/galaxy/test_*_wh_galaxy.py` files have zero references to
`hf_config_or_skip`, `LLAMA_DIR`, `HF_MODEL` or `from_pretrained` — verified by grep, not assumed.

So **there is no 138 GB / ~26 min weight staging on this path**, and the trap where a wrong
`HF_HOME` makes `hf_config_or_skip` skip every real-checkpoint test and the run looks green
having measured nothing **cannot fire here**. That trap is real for model-level suites; it is not
your problem in this window.

---

### Stage 0 — environment (15–25 min, no device)

```bash
git fetch && git checkout tttv2-galaxy-2d-modules-port
uv python install 3.12
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[test,examples]'
python -c "import ttnn; print(ttnn.__version__)"     # expect 0.77.0
```

The package is pure Python; its only heavy dependency is the `ttnn` wheel from PyPI. There is no
`tt-metal` build and no C++ toolchain. Most of the estimate is the wheel download.

**If `ttnn` will not install:** check the interpreter is cp310 or cp312 and the platform is
`manylinux_2_34_x86_64`. Those are the only wheels published. This is also why none of this ran
on a Mac.

---

### Stage 1 — E00, the real host gates (10–20 min, no device)

Commands are in [E00's README](bh_galaxy_experiments/E00-host-gates/). Expected:
**`pytest -m host` reports 2612–2613 passed**, and the six tool gates are clean.

**This is the stage that can invalidate phases 1–4, and it costs nothing. Do not skip it to save
time.**

**If it is red**, stop and do not spend device time. Two outcomes are worth pre-empting:

- **`tests/models/galaxy/test_topology.py` fails.** The most likely single failure, and E00's
  README explains precisely why and what to do. In short: its pure-Python half was verified, but
  its `CoreRangeSet` equality assertions were not, because that needs real pybind. If equality
  does not behave, the fix is mechanical — **but do not weaken the test to membership-only.**
  Order is the load-bearing property; a ring built from a `set` once produced bit-for-bit
  identical *wrong* output across runs.
- **`ruff format --check` rewrites files.** Expected and cosmetic. The repo pins `ruff>=0.11.0`
  and the unpinned 0.16.7 used during development formats differently. Land the reformat as its
  own commit; it is not a defect.

---

### Stage 2 — the host-side geometry diff (~2 min, no device)

**This is the strongest single check available, and it was added because it is cheaper and more
complete than the device run it precedes.**

Everything phase 2 could have broken is resolved on the host before any module hot path runs:
memory configs, program configs, core range sets, sub-device partitions. So resolve all of it on
both commits and diff the text.

```bash
D=docs/bh_galaxy_experiments/E01-wh-byte-identical/dump_resolved_geometry.py
git stash -u
git checkout 0a3e045 && python $D > /tmp/base.txt          # phase 1: last commit before the descriptor
git checkout tttv2-galaxy-2d-modules-port && python $D > /tmp/head.txt
git stash pop
diff -u /tmp/base.txt /tmp/head.txt && echo "IDENTICAL"
```

The script uses only API present on **both** commits — verified symbol by symbol and
signature by signature — so it runs unmodified from either checkout.

**Expected: `IDENTICAL`.**

**If the diff is non-empty:** you have found the phase-2 regression, on the host, in two minutes,
with the exact field named. That is the best possible outcome short of identity. Read the
differing line — it names the model, the mode and the field — and fix before going near the
device. First suspect is **iteration order** wherever a `CoreRangeSet` is rebuilt from the
descriptor's tuples.

**If a section prints `FAILED TO RESOLVE`:** the dump is deliberately section-guarded so the rest
still runs. A section that fails on *both* commits identically is not a regression — it means the
script's assumptions about that helper are wrong, and the diff for the other sections is still
valid. A section that fails on only one commit **is** the finding.

---

### Stage 3 — E01 on the device (50–70 min device time)

Only if stages 1 and 2 are green. Full procedure in
[E01's README](bh_galaxy_experiments/E01-wh-byte-identical/).

**Stage 2 changes what stage 3 has to prove.** If the host-resolved configs are byte-identical,
the device programs built from them are identical by construction, so running *both* commits on
device is no longer the primary evidence. **Run HEAD only**, and keep the base commit in reserve
for triage.

```bash
# The marker was renamed in phase 1: `-m galaxy` now selects NOTHING.
pytest tests/modules tests/models/galaxy --collect-only -q -m galaxy_wh | grep '::' > /tmp/ids.txt
wc -l /tmp/ids.txt        # expect ~57

# One pytest node id per process. This is not style: the ttnn program cache belongs
# to the mesh device and the process, and the weight cache is keyed on MeshDevice.id().
while read -r id; do
  MESH_DEVICE=TG timeout 900 pytest "$id" -sv 2>&1 | tee "/tmp/wh-$(echo "$id" | tr '/:[]' '____').log"
done < /tmp/ids.txt
```

**Expected: 56 of 57 green**, in roughly 50 minutes. Individual suites run 7–107 s; most of the
rest is mesh open and close per process.

**One known-red baseline, and it is not yours.** `attention_decode_with_active_prefetch` passes
setup, call, PCC and its own cleanup and **still exits non-zero**, because
`ttnn.close_mesh_device` hangs in the fixture afterwards: the test starts the DRAM producer, but
attention decode is precisely the module that must *not* consume the global CB, so nothing drains
the ring. That is a test-design defect recorded before this work. Treat it as red and move on.

**Then check stability on the two highest-signal suites** — three fresh processes each, output
compared, not just PCC-passing:

```bash
for i in 1 2 3; do
  MESH_DEVICE=TG pytest tests/modules/mlp/test_mlp_2d_wh_galaxy.py -sv 2>&1 | tee /tmp/mlp-$i.log
done
```

The bar is three fresh processes because the fused-RMSNorm stats CB binds to the first core of
the norm input shard grid and aliases whatever the allocator left there — it has produced PCC of
0.0977 / 0.1394 / 0.1701 / 0.9999 **on an unchanged test**. One run that passes proves less than
it looks like.

---

## 4. If stage 3 fails

Work down this list; it is ordered by how often each cause turns out to be the real one.

**A suite fails on HEAD.** Do not conclude the refactor broke it. **Re-run that one node id on
`0a3e045` first.** The pre-existing state was 56/57, so a red suite may predate this work
entirely. Only a suite that is green on base and red on HEAD is a regression — and if stage 2 was
`IDENTICAL`, that combination is surprising enough to be worth stopping and thinking about rather
than patching.

**PCC passes but you suspect the output moved.** This is the failure the byte-identical bar
exists for, because the ordering defect produces *stable* wrong output that PCC thresholds
accept. If stage 2 was identical this should be impossible; if you see it anyway, the dump script
is missing a code path, and finding which one is more valuable than the test result.

**The mesh hangs with no traceback.** Almost always mismatched topology and fabric, or a wrong
link count — the ring and line CCLs index ethernet channels by link. Debug with
**worker-sub-device-scoped** `synchronize()` fences, never a whole-device sync. Be aware that a
`TT_FATAL` out of `enqueue_mesh_workload` leaves the mesh **un-drainable**: SIGTERM cannot service
it, and SIGKILL plus a reset is the only recovery.

**You need a reset.** `tt-smi -glx_reset`, **not** `-r`. Neither recovers a board that has fallen
off the PCIe bus. Check health with `/sys/class/tenstorrent` — `ls /dev/tenstorrent | wc -l` is
not a health check.

**After any reset, re-run an already-qualified numerical control before trusting anything.** A
reset that reports success does not guarantee a healthy mesh: one episode chased for hours as a
tracing defect turned out to be mesh state, producing 8 distinct answers out of 12 with logit
magnitudes of 1e14–1e17 against a healthy peak of about 31.

**The window runs out.** Stages 0–2 are the ones that matter most and none of them needs the
device. If you got through stage 2 with `IDENTICAL`, record that in
[E01's README](bh_galaxy_experiments/E01-wh-byte-identical/) under `## Result` and the window was
a success — stage 3 is confirmation, not the primary evidence.

---

## 5. Recording results

**A measured answer that stays in a terminal has to be measured again.** For every experiment you
run:

1. Write the outcome into that folder's `README.md` under `## Result` — the readings verbatim,
   not a summary.
2. Update the `Status` column in
   [bh_galaxy_experiments/README.md](bh_galaxy_experiments/README.md).
3. If it changed a decision, add it to [bh_galaxy_port_log.md](bh_galaxy_port_log.md) with the
   `[measure]` tag replaced by what you measured.

**Negative results count.** Several experiments have outcomes that say "keep the current
behaviour and write down why" — those are worth as much as the green ones, because the
alternative is that the next person redoes the same inference and tries the same thing.

---

## 6. The experiments, and running the patched ones

Every experiment lives in [docs/bh_galaxy_experiments/](bh_galaxy_experiments/) with its own
README stating the hypothesis, the command, and what each outcome means. **Read the folder; this
section only covers the mechanics.**

| # | Hardware | Ships | Note |
| --- | --- | --- | --- |
| [E00](bh_galaxy_experiments/E00-host-gates/) | none | commands | stage 1 above |
| [E01](bh_galaxy_experiments/E01-wh-byte-identical/) | **WH Galaxy** | procedure + dump script | stages 2–3 above |
| [E02](bh_galaxy_experiments/E02-bh-day0-probe/) | BH Galaxy | in-tree test | run first on the first BH window |
| [E03](bh_galaxy_experiments/E03-worker-envelope/) | BH Galaxy | **3 patches** | one-line constant flips, 3 arms |
| [E04](bh_galaxy_experiments/E04-module-capability-gates/) | BH Galaxy | **1 patch** | unblocks E05–E07 |
| [E05](bh_galaxy_experiments/E05-multilayer-residual/) | BH Galaxy | spec | no patch, deliberately |
| [E06](bh_galaxy_experiments/E06-mlp-dram-interleaved/) | BH Galaxy | spec | no patch, deliberately |
| [E07](bh_galaxy_experiments/E07-fused-ccl-noop/) | BH Galaxy | spec | no patch, deliberately |

### Applying a patch

All patches are generated against **`746bb77`** and were verified to apply against it.

```bash
git apply --check docs/bh_galaxy_experiments/E04-module-capability-gates/patch.diff   # dry run first
git apply        docs/bh_galaxy_experiments/E04-module-capability-gates/patch.diff
# ... run the experiment ...
git checkout -- src/ tests/                                                            # revert between arms
```

**Always `--check` first.** If it fails to apply, a target file has moved: **re-read the target
and redo the change by hand.** Do not force it, and do not `git apply --3way` your way past a
conflict you have not read — every one of these patches touches a constant or a gate whose
surrounding comment is the justification, and a patch that lands in the wrong place silently
detaches the two.

**Revert between arms.** E03 has three arms that each modify the same two constants; applying two
of them produces arm D by accident.

### Two things not to do on the Wormhole machine

- **Do not run the Blackhole experiments there.** E02–E07 need `MESH_DEVICE=BHGLX` and a
  Blackhole mesh; on Wormhole they will fail at the architecture gate, which tells you nothing.
- **Do not run the expensive suites.** Executor, trace and model-level suites are tens of minutes
  each, there are ~65 of them, and the next code change invalidates them. Nothing in this window
  needs them.

---

## 7. After the Wormhole window

The two open risks, in priority order:

1. **The 2D module gates still accept Wormhole only**, so no module suite runs on Blackhole yet.
   This is phase 5's first task and it is specified with a working patch in
   [E04](bh_galaxy_experiments/E04-module-capability-gates/). **Its host half needs no Blackhole
   machine** — `pytest -m host tests/modules/mlp tests/modules/rmsnorm` after applying the patch
   is runnable on the Wormhole host, or any Linux box, and it is where a resolution error would
   surface. Worth doing in this window if stages 0–3 finish early.
2. **The Blackhole side has been measured zero times.** [E02](bh_galaxy_experiments/E02-bh-day0-probe/)
   answers four of the plan's five open hardware questions in one read-only run that allocates no
   tensor and runs no collective, so it cannot leave the mesh needing a reset. It is the right
   first thing on the first Blackhole window.

Nothing is pushed and no PR is open. [The plan's §7 phase 0 item 3](blackhole_galaxy_port_plan.md)
records that push/PR sequencing is still unsettled — one combined Wormhole+Blackhole PR was the
earlier leaning. That decision is still open and is not mine to make.
