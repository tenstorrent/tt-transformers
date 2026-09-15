# Blackhole Galaxy port — handoff

**Written for: whoever picks this up next with machine access** — an engineer or an agent session
with SSH to a Wormhole Galaxy now, and to a Blackhole Galaxy later. It assumes you have not read
the rest of `docs/`.

Branch: `tttv2-galaxy-2d-modules-port`. Six commits, **none pushed**, no PR open.

---

## 1. Where the work stands in one paragraph

> **Status: the Wormhole window described below was run on 2026-09-15 and all four stages are
> complete.** Phase 2's exit criterion is met — host geometry byte-identical, 56/57 on device,
> numerics stable across processes. The stage instructions are kept because several of them were
> *wrong* in ways that produce convincing wrong answers, and the corrections are inline; the
> readings are in each experiment folder under `## Result`. What remains is Blackhole, which has
> been measured zero times. Skip to §7.

Phases 1–4 of the port plan — everything that needs no hardware — are written and committed. The
Galaxy geometry now lives in a validated per-architecture descriptor, Blackhole resolves against
it, and a disabled matrix node and capability geometry are registered. **What had never run was
anything that imports `ttnn`**, which includes every Galaxy test suite. Four host gates and 307
host tests were run on a Mac in a `ttnn`-free venv, differentially; that was real but partial
coverage, and [bh_galaxy_port_log.md](bh_galaxy_port_log.md) §0 and §6 say exactly what it did and
did not cover.

That gap is now closed for Wormhole. It was worth closing: the real host gates found three stale
assertions in the new Galaxy tests and one real lint error, none of which the Mac could see.

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
uv pip install -e '.[test,examples,dev]'
python -c "import ttnn, importlib.metadata as m; print(m.version('ttnn'))"    # expect 0.77.0
```

The package is pure Python; its only heavy dependency is the `ttnn` wheel from PyPI. There is no
`tt-metal` build and no C++ toolchain. Most of the estimate is the wheel download.

**Two corrections from the first real run.** `ruff` and `mypy` live in the **`dev`** extra, so
without it stage 1's last three gates have no binary. And `ttnn` publishes no `__version__` —
`import ttnn; print(ttnn.__version__)` raises `AttributeError` on a perfectly good install, which
is an alarming way to start; ask `importlib.metadata` instead.

If a hash-locked environment is wanted rather than a resolved one, CI's is
`constraints/locks/host-py310.txt` plus `constraints/locks/build-dev-py310.txt`.

**If `ttnn` will not install:** check the interpreter is cp310 or cp312 and the platform is
`manylinux_2_34_x86_64`. Those are the only wheels published. This is also why none of this ran
on a Mac.

---

### Stage 1 — E00, the real host gates (10–20 min, no device)

Commands are in [E00's README](bh_galaxy_experiments/E00-host-gates/). Expected:
**`python -m pytest -q -m host` reports ~2641 passed**, and the seven tool gates are clean.

**This is the stage that can invalidate phases 1–4, and it costs nothing. Do not skip it to save
time.**

**It has now been run, and it was worth every minute of the estimate**: it found three stale
assertions in the new Galaxy tests and one real lint error, all fixed. The readings and the
diagnosis are in E00's README under `## Result`. Two of the three predictions below turned out
to be wrong in useful ways, so they are corrected rather than deleted:

- **`test_topology.py`'s `CoreRangeSet` equality held.** This was flagged as the most likely
  single failure, because the operator was assumed rather than exercised. It exists, it is
  order-sensitive, and the golden tables pass. Risk closed.
- **`test_topology.py` did fail, for a different reason** — two of its assertions encoded the
  phase-1 world in which Wormhole was the only architecture with a descriptor. So did one in
  `test_recipes.py`. These are the stale-test failures, not ordering failures.
- **`ruff format --check` does not rewrite anything.** The predicted cosmetic reformat was an
  artifact of running an unpinned `ruff` over `.`; with the locked **0.11.0** and CI's scope,
  431 files are already formatted. Do not land a reformat commit — and do pin the version, because
  0.16.7 reports 12 lint errors on this tree and the one real error hides among them.

---

### Stage 2 — the host-side geometry diff (~2 min, no device)

**This is the strongest single check available, and it was added because it is cheaper and more
complete than the device run it precedes.**

Everything phase 2 could have broken is resolved on the host before any module hot path runs:
memory configs, program configs, core range sets, sub-device partitions. So resolve all of it on
both commits and diff the text.

```bash
# Copy the script OUT of the worktree: it postdates the base commit, so checking
# that commit out deletes it. And strip UMD's stdout logging, whose timestamps
# differ on every run.
cp docs/bh_galaxy_experiments/E01-wh-byte-identical/dump_resolved_geometry.py /tmp/dump.py
LOG='^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+ \|'
git stash -u
git checkout 0a3e045 && python /tmp/dump.py | grep -Ev "$LOG" > /tmp/base.txt   # last commit before the descriptor
git checkout tttv2-galaxy-2d-modules-port && python /tmp/dump.py | grep -Ev "$LOG" > /tmp/head.txt
git stash pop
diff -u /tmp/base.txt /tmp/head.txt && echo "IDENTICAL"
```

The script uses only API present on **both** commits — verified symbol by symbol and
signature by signature — so it runs unmodified from either checkout.

**Expected: `IDENTICAL`. This has now been run, and it is.** 408 lines, 395 resolved geometry
fields, the same sha256 on both commits. Phase 2's exit criterion is met on the host; the readings
are in [E01's README](bh_galaxy_experiments/E01-wh-byte-identical/) under `## Result`.

**Both of the deviations in that snippet were found by getting them wrong, and each returns a
convincing wrong answer rather than an error.** Running the script from its in-tree path makes the
base dump fail with `No such file or directory`, which leaves an empty base file and a diff that
is the whole head dump — 359 lines that read as total regression. And the script **opens the
cluster** despite allocating nothing, so it logs ~17 timestamped lines to stdout: the first honest
comparison came back as 47 differing lines, every one of them a clock reading. Strip them and
it is byte-identical.

It also silently measured only part of what it claims to before being fixed:
`ttnn.SDPAProgramConfig` on 0.77.0 raises `TypeError` from both `__repr__` and `__str__`, which
voided the *mesh-derived helpers* and *resolved placements* sections — between them most of what
phase 2 could have moved. The section guard that kept the rest of the dump alive is what made the
loss easy to miss. **If a section prints `FAILED TO RESOLVE`, do not accept the diff for the
others without reading why.**

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

**Run on 2026-09-15: 56 of 57, in 50 minutes, mean 53 s.** Exactly the pre-refactor baseline, so
no regression. Readings per file are in E01's README.

**One known-red baseline, and it is not yours.** `attention_decode_with_active_prefetch` passes
setup, call, PCC and its own cleanup and **still exits non-zero**, because
`ttnn.close_mesh_device` hangs in the fixture afterwards: the test starts the DRAM producer, but
attention decode is precisely the module that must *not* consume the global CB, so nothing drains
the ring. That is a test-design defect recorded before this work. Treat it as red and move on.

**Budget 15 minutes for it, though.** It does not exit quickly: its log prints `PASSED` and then
the process hangs after `Clearing program cache on MeshDevice 0` until `timeout` kills it, so it
burns the full 900 s — 15 of the sweep's 50 minutes for a verdict you already have. Deselect the
id, or give it a short timeout. SIGTERM was enough to clear it: no lingering process, 32 devices
still present, the next suite green, **no reset needed**.

**Then check stability on the two highest-signal suites** — three fresh processes each, output
compared, not just PCC-passing:

**Run one node id, not the file**: the MLP file holds both models, and two models in one process
is the pattern that hangs at the second model's first decode.

```bash
MLP='tests/modules/mlp/test_mlp_2d_wh_galaxy.py::test_mlp_2d_wh_galaxy_decode_batch_32_repeat[wormhole_b0-device_params0-llama-8192x28672-mesh_device0]'
NORM='tests/modules/rmsnorm/test_rmsnorm_2d_wh_galaxy.py::test_rmsnorm_2d_wh_galaxy_final_norm_decode_batch_32_fused_residual_repeat[wormhole_b0-device_params0-llama-final-8192-mesh_device0]'
for i in 1 2 3; do
  for id in "$MLP" "$NORM"; do
    MESH_DEVICE=TG python -m pytest -p pcc_probe "$id" -sv 2>&1 | tee "/tmp/stab-$i.log"
  done
done
```

The bar is three fresh processes because the fused-RMSNorm stats CB binds to the first core of
the norm input shard grid and aliases whatever the allocator left there — it has produced PCC of
0.0977 / 0.1394 / 0.1701 / 0.9999 **on an unchanged test**. One run that passes proves less than
it looks like.

**"Output compared" needs one extra piece: these suites print no PCC on success.** They assert
internally, so three green runs are indistinguishable from three *identical* green runs, which is
the entire question. Wrap the comparison functions from a plugin **outside** the repository — so
the checkout stays a clean pull target — and load it with `-p`:

```python
# pcc_probe.py, on PYTHONPATH, outside the repo
import functools

def _wrap(fn, label):
    @functools.wraps(fn)
    def probe(*args, **kwargs):
        result = fn(*args, **kwargs)
        print(f"PCC_PROBE {label} -> {result!r}", flush=True)
        return result
    probe._pcc_probe = True
    return probe

def pytest_runtest_setup(item):
    for name in ("comp_pcc", "mlp_pcc"):           # rmsnorm imports one, mlp the other
        fn = getattr(item.module, name, None)
        if callable(fn) and not getattr(fn, "_pcc_probe", False):
            setattr(item.module, name, _wrap(fn, name))
    import tests.modules._mlp_2d_galaxy as shared  # assert_mlp_pcc calls this module-global
    if not getattr(shared.mlp_pcc, "_pcc_probe", False):
        shared.mlp_pcc = _wrap(shared.mlp_pcc, "mlp_pcc")
```

**Run on 2026-09-15: identical to 16 significant digits**, three processes, both suites — MLP
`0.9982189986169618`, fused-residual RMSNorm `0.9999860329520437` and `0.9999979576039204`.

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

The two open risks, in priority order. **Risk 1 from the earlier revision — phase 2 carrying an
unverified refactor of a qualified path — is closed**, so what follows is what is left:

1. **The 2D module gates still accept Wormhole only**, so no module suite runs on Blackhole yet.
   This is phase 5's first task and it is specified with a working patch in
   [E04](bh_galaxy_experiments/E04-module-capability-gates/). **Its host half needs no Blackhole
   machine** — `python -m pytest -m host tests/modules/mlp tests/modules/rmsnorm` after applying
   the patch is runnable on the Wormhole host, or any Linux box, and it is where a resolution
   error would surface. It was **not** done in the 2026-09-15 window: it is a production change on
   a new phase rather than verification of the old one, and it was left for whoever owns phase 5
   to decide.

   One thing measured in that window sharpens the task. The two gates have now **diverged**: the
   mesh contract `recipes.validate_galaxy_mesh` generalised in phase 3 and admits any architecture
   holding a topology descriptor — which is what three of E00's failures were about — while seven
   modules still compare `mesh_device.arch()` against `WORMHOLE_B0` directly. So a Blackhole mesh
   now passes the mesh gate and is then refused by every module. The seven, and what each needs:

   | Module | State |
   | --- | --- |
   | `mlp_2d`, `rmsnorm_2d` | E04's patch covers these two |
   | `prefetcher_2d` | **leave it.** Milestone 1 is prefetcher-free on Blackhole, so a Wormhole-only gate here is correct by design, not a gap |
   | `embedding_2d`, `lm_head_2d`, `rope_2d`, `sampling_2d` | same shape as E04's two, **not in the patch**, and needed before any model-level Blackhole run |
2. **The Blackhole side has been measured zero times.** [E02](bh_galaxy_experiments/E02-bh-day0-probe/)
   answers four of the plan's five open hardware questions in one read-only run that allocates no
   tensor and runs no collective, so it cannot leave the mesh needing a reset. It is the right
   first thing on the first Blackhole window.

Nothing is pushed and no PR is open. [The plan's §7 phase 0 item 3](blackhole_galaxy_port_plan.md)
records that push/PR sequencing is still unsettled — one combined Wormhole+Blackhole PR was the
earlier leaning. That decision is still open and is not mine to make.
