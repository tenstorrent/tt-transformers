# E01 — Is Wormhole decode byte-identical after the topology refactor?

**Hardware: one 32-device Wormhole Galaxy (`wh-glx6u-05`). No patch.**

**The highest-value single experiment here.** It is the only thing that can confirm phase 2 did
not move a hardware-qualified path, and it is on a different machine and a different schedule
from everything else in this vault.

---

## Why it exists

Phase 2 replaced every intra-chip core coordinate in `recipes.py` and `prefetch.py` with a
`GalaxyChipTopology` descriptor. That is the geometry source of a path that **was** qualified on
silicon — 56 of 57 cheap module ids green, roughly 50 minutes of device time.

The port plan's own exit criterion for that phase is *"one WH Galaxy run with **bit-identical**
decode output, not merely passing PCC"*, and it says the criterion is not optional.

**It has not been run, and there is no Wormhole Galaxy access either.** So phase 2 currently
rests on two things:

1. `test_topology.py`'s golden tables, which pin every core sequence element-wise and in order
   against the literals as they stood before the refactor;
2. `test_wormhole_link_counts_are_unchanged_by_the_clamp`, which pins the full ordered
   `num_links` tuple for both models.

Both are real, and both check the **resolver**. Neither checks the **callers**. That gap is what
this experiment closes.

## Do the host half first — it is cheaper and more complete

**Added after this folder was first written**, because it is a better measurement than the device
run and it needs no allocation.

Everything phase 2 could have broken is resolved **on the host, before any module hot path runs**:
memory configs, program configs, core range sets, sub-device partitions. So resolve all of it on
both commits and diff the text.

```bash
# Copy the script OUT of the worktree: it postdates the base commit, so
# `git checkout 0a3e045` deletes it. And strip UMD's stdout logging, whose
# timestamps differ on every run.
cp docs/bh_galaxy_experiments/E01-wh-byte-identical/dump_resolved_geometry.py /tmp/dump.py
LOG='^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+ \|'
git checkout 0a3e045 && python /tmp/dump.py | grep -Ev "$LOG" > /tmp/base.txt
git checkout tttv2-galaxy-2d-modules-port && python /tmp/dump.py | grep -Ev "$LOG" > /tmp/head.txt
diff -u /tmp/base.txt /tmp/head.txt && echo "IDENTICAL"
```

[`dump_resolved_geometry.py`](dump_resolved_geometry.py) uses only API present on **both**
commits — verified symbol by symbol and signature by signature — so it runs unmodified from
either checkout.

**Both caveats in that snippet were learned by getting them wrong**, and each one produces a
convincing wrong answer rather than an error:

- Running the script from its **in-tree path** on the base commit fails with
  `No such file or directory`, because the script postdates that commit. That leaves an empty base
  file, and the diff against it is the entire head dump — which reads as every field having moved.
- The script **does open the cluster**, contrary to what this folder said. It allocates nothing,
  enqueues nothing and opens no mesh, but resolution reaches bindings that initialise UMD: a run
  on `wh-glx6u-05` logged *"Opening user mode device driver"*, topology discovery, and all 32
  local chip ids, then closed them. So it needs a host **with devices** rather than any Linux box,
  and because UMD logs ~17 timestamped lines **to stdout**, a raw diff of two runs is never empty
  and blames whichever field the log landed beside.

**Expected: `IDENTICAL`.** A non-empty diff names the model, the mode and the field that moved —
which is the whole finding, obtained in two minutes instead of two hours.

**This changes what the device run has to prove.** If the host-resolved configs are identical,
the device programs built from them are identical by construction, so running *both* commits on
silicon is no longer the primary evidence. Run HEAD only and keep the base commit for triage.

## Hypothesis

Every Wormhole Galaxy module suite produces output byte-identical to the pre-refactor commit.

Not "passes PCC". Byte-identical. The distinction is the whole point: the
`CoreRangeSet`-ordering defect this refactor is most at risk of reintroducing produced
**bit-for-bit identical wrong output across runs** — stable, not noisy, which is exactly why PCC
thresholds did not catch it and why it took weeks to find.

## Run

Run the host diff above first. Then, on HEAD only:

```bash
# The marker was renamed in phase 1: `-m galaxy` now selects NOTHING.
pytest tests/modules tests/models/galaxy --collect-only -q -m galaxy_wh | grep '::' > /tmp/ids.txt

# One pytest node id per process. The ttnn program cache belongs to the mesh device
# and the process, and the weight cache is keyed on MeshDevice.id().
while read -r id; do
  MESH_DEVICE=TG timeout 900 pytest "$id" -sv 2>&1 | tee "/tmp/wh-$(echo "$id" | tr '/:[]' '____').log"
done < /tmp/ids.txt
```

**No checkpoint is needed.** All eight `tests/modules/*/test_*_2d_wh_galaxy.py` files and all
three `tests/models/galaxy/test_*_wh_galaxy.py` files have zero references to
`hf_config_or_skip`, `LLAMA_DIR`, `HF_MODEL` or `from_pretrained` — so there is no 138 GB weight
staging on this path, and the trap where a wrong `HF_HOME` silently skips every real-checkpoint
test cannot fire here.

**Keep the base commit for triage, not for a full second pass.** Re-run a *failing* node id on
`0a3e045` to establish whether it predates this work; the pre-existing state was 56/57.

### Two things that will otherwise waste the window

**`attention_decode_with_active_prefetch` is a known-red baseline.** It passes setup, call, PCC
and its own cleanup, and still exits non-zero because `ttnn.close_mesh_device` hangs in the
fixture afterwards: the test starts the DRAM producer, but attention decode is precisely the
module that must *not* consume the global CB, so nothing drains the ring. Treat it as red on
both sides of the comparison, not as a regression this refactor introduced.

**Three fresh processes, not one.** The fused-RMSNorm stats CB binds to the first core of the
norm input shard grid and aliases whatever the allocator left there, which has produced PCC of
0.0977 / 0.1394 / 0.1701 / 0.9999 across processes on an unchanged test. A single run that
matches proves less than you think.

## What each outcome means

**Byte-identical across the board.** Phase 2 is confirmed and the descriptor may keep the
qualified path. Record it in `docs/validation.md` and in the port log — this is the evidence the
whole refactor is missing, and it should be findable later.

**PCC passes but output differs.** *This is the dangerous outcome and the one to look hardest
at.* Something in the resolved geometry moved in a way the golden tables did not catch, which
means the tables are checking the wrong thing. First suspect: **iteration order** somewhere a
`CoreRangeSet` is rebuilt from the descriptor. `core_points` preserves the tuple order it is
given, so check that the *tuple* is what you expect, then that nothing downstream reconstructs
the set from an unordered collection.

**A suite fails outright.** Compare against the base SHA before concluding anything. The base run
is not ceremony — 56 of 57 was the state before this work, so a red suite may predate it.

**It cannot be run at all.** Then say so, plainly, wherever phase 2 is described. Carrying an
unverified refactor of a qualified path is a legitimate decision; carrying it *silently* is not.

---

## Result

### Host half: `IDENTICAL` — 2026-09-15, `wh-glx6u-05`

**Phase 2's exit criterion is met on the host.** Base `0a3e045` against head `2270e06`, ttnn
0.77.0, Python 3.10.21:

```
base commit: 0a3e04548e5d83e517f3ba9e8cd5abc4814ddd6c
head commit: 2270e0684a31c42f4917414156f444a63831ec15
base dump exit=0                     head dump exit=0
base FAILED TO RESOLVE: 0            head FAILED TO RESOLVE: 0
base clean lines: 408                head clean lines: 408
geometry field lines: 395
RESULT: IDENTICAL
5dfa0412cec0c5a470c138fbc34fc85435af6de51f2ae5094a4da6ac5cc30b98  base.clean.txt
5dfa0412cec0c5a470c138fbc34fc85435af6de51f2ae5094a4da6ac5cc30b98  head.clean.txt
```

Identical sha256 over 408 lines and **395 resolved geometry fields**: every core set, the
20-entry prefetch sender/receiver mapping, the vocabulary and reduction arithmetic, every
mesh-derived helper, both models' resolved decode and prefill placements, and both models'
resource plans including the ordered `num_links` tuples and every collective's topology and
persistent output specs.

So the descriptor introduced in phase 2 reproduces the pre-descriptor geometry exactly, for
Llama-3.3-70B and Qwen3-32B, in both modes. The `CoreRangeSet`-ordering defect this refactor was
most at risk of reintroducing is excluded by construction, because order is part of what the
compared text records.

**The raw diff was not empty, and the reason matters.** The first comparison reported 47 differing
lines. Every one was a timestamped UMD log line — `Opening user mode device driver`, topology
discovery, chip ids, cluster teardown — which the dump emits to stdout, so the "difference" was
the clock. Not one resolved field differed. Both artifacts and the filtered pair are kept at
`/proj_sw/user_dev/ctr-apbernal/remote/logs/tt_transformers/20260915T131707Z-e01-geometry-diff2/`.

**Three defects in this folder's own procedure were found and fixed getting to that reading**, all
recorded above: the in-tree script path that cannot work on the base commit, the stdout log noise,
and the claim that no device is opened. A fourth was in the script — `ttnn.SDPAProgramConfig` on
0.77.0 raises `TypeError` from both `__repr__` and `__str__`, which silently voided the
*mesh-derived helpers* and *resolved placements* sections, i.e. most of what phase 2 could have
moved. `render` now falls back to an ordered field walk, which is why the dump is 425 lines rather
than the 356 the first run produced.

### Device half: 56 / 57 — 2026-09-15, `wh-glx6u-05`, `MESH_DEVICE=TG`, HEAD only

**Exactly the pre-existing baseline. No regression.** `858bb9f`, one pytest node id per process,
`timeout 900` each:

```
start=13:24:34   end=14:14:35   green: 56 / 57   total 50 min, mean 53s

 7/ 7  tests/models/galaxy/test_column_user_selector_wh_galaxy.py
 5/ 5  tests/models/galaxy/test_partition_wh_galaxy.py
 3/ 3  tests/models/galaxy/test_step7_page_table_placement_wh_galaxy.py
 2/ 2  tests/modules/attention/test_attention_2d_wh_galaxy.py
 2/ 2  tests/modules/embedding/test_embedding_2d_wh_galaxy.py
 2/ 2  tests/modules/lm_head/test_lm_head_2d_wh_galaxy.py
 4/ 4  tests/modules/mlp/test_mlp_2d_wh_galaxy.py
 7/ 8  tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py
 8/ 8  tests/modules/rmsnorm/test_rmsnorm_2d_wh_galaxy.py
 2/ 2  tests/modules/rope/test_rope_2d_wh_galaxy.py
 5/ 5  tests/modules/sampling/test_sampling_2d_wh_galaxy.py
 9/ 9  tests/modules/sampling/test_sampling_2d_wh_galaxy_stochastic.py
```

The one red is the documented baseline,
`test_prefetcher_2d_wh_galaxy_attention_decode_with_active_prefetch`, and its log confirms the
diagnosis rather than assuming it: the test prints **`PASSED`**, then the process hangs
immediately after `Clearing program cache on MeshDevice 0` — in the teardown fixture, after the
assertion. Not a regression, and not this refactor's.

**One correction to the estimate: that hang consumes the *whole* 900 s timeout.** This folder
described it as passing "and still exits non-zero", which suggests a fast failure; it is exit
`124` after 15 minutes. So a full sweep is 50 min of which 15 is one hung teardown, and the other
56 suites take 35 min at a 53 s mean. `timeout` sent only SIGTERM and it was sufficient — no
lingering process, 32 devices still present, and the next suite (rmsnorm) passed immediately
afterwards, so **no reset was needed**. Next time, deselect that id or give it a short timeout;
its verdict is known before it hangs.

### Stability: three fresh processes, values identical to 16 digits

The bar is the output, not the verdict, so the PCC each suite computes was captured rather than
inferred from green — the suites assert internally and print nothing on success. An out-of-tree
pytest plugin (`pcc_probe.py`, loaded with `-p`, so the checkout stayed clean) wrapped `mlp_pcc`
and `comp_pcc`. Three fresh processes each, of the two highest-signal ids:

```
mlp     decode_batch_32        (llama-8192x28672)   runs 1,2,3: 0.9982189986169618  (x2 per run)
rmsnorm final_norm_decode_32   (llama-final-8192)   runs 1,2,3: 0.9999860329520437
                                                                0.9999979576039204  (x2 per run)
IDENTICAL across all three processes, both suites
```

**This is the measurement the fused-RMSNorm risk exists for.** That stats CB binds to the first
core of the norm input shard grid and aliases whatever the allocator left there, which has
produced 0.0977 / 0.1394 / 0.1701 / 0.9999 across processes on an unchanged test. Here the
fused-residual decode path reproduces the same two correlations bit-for-bit in three separate
processes. Note the run was **not** the whole-file command this folder suggested: that file holds
both models, and two models in one process is the exact pattern the vault's own rules forbid.

### What this means

Phase 2 is confirmed on both halves: byte-identical host-resolved geometry, and the device state
unchanged from the 56/57 that preceded the refactor, with stable numerics. **The topology
descriptor may keep the qualified path.** The Wormhole window's purpose is discharged.
