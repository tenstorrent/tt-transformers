# Validation

This page summarizes validation of the cleaned package candidate. Results
apply only to the named revision or artifact, not as a blanket support
promise.

## Hardware

- Tested code: `9134c399334240e3e4d35dfa93013c6c7293a3d1`
- Executed: 2026-09-04 20:58–22:39 UTC
- Matrix: 42/42 enabled nodes passed (43 declared; the TG node is disabled and
  was not executed — see "Galaxy 8x4 and 2D modules" below)
- Stages: modules 30/30, runtime 1/1, smoke 3/3, end-to-end 8/8
- Hosts: Wormhole 23/23, Blackhole 19/19
- Meshes: N150 9/9, N300 6/6, T3K 8/8, P150 8/8, P150x4 11/11, TG 0/1 disabled
- Functional, lifecycle, missing-acceptance, and reset outcomes: zero

Both physical hosts ran concurrently while every node remained serialized
within its host. Post-run inventory found eight healthy Wormhole devices and
four healthy Blackhole devices, with no pytest process left behind.

The eight `MESH_DEVICE=P150` records were logical 1x1 executions on a physical
P150_X4 quietbox with `TT_VISIBLE_DEVICES` unset. They are not standalone-P150
product evidence.

## Cleaned package candidate

The cleanup candidate represented by the Git revision containing this page
passed the host suite from an independently unpacked sdist on CPython 3.10 and
3.12 with 2,170 passes, 28 intentional skips, 6,791 deselections, 5 warnings,
and 81 passing subtests per interpreter. Its sdist rebuilt a wheel
byte-for-byte identical to the checkout build. Two checkout builds also
produced identical wheels and identical normalized sdists. The package was
validated against `ttnn==0.77.0`.

Release artifact hashes belong in the immutable release/CI attestation. They
cannot be embedded in the sdist without changing that sdist's own digest.

The pre-cleanup non-editable-wheel host baseline was 2,170 passes, 28
intentional skips, 6,791 deselections, 5 warnings, and 81 passing subtests per
interpreter.

## Current evidence

The canonical current index contains 42 unique same-SHA records and has
SHA-256:

```text
e7e5b44a0f559d7ea41b197ea172c1bd5908f7cd595165e1182c452617ecaa91
```

The deterministic 91-file evidence archive was independently extracted and
verified. Its SHA-256 is:

```text
f5109114fcca10272580fb27e086fb9f86625c9bdb369019a051148f6a586684
```

The archive remains outside Git and must be copied to the project's approved
immutable artifact store before release publication.

## Historical evidence

The complete pre-cleanup evidence and reports are preserved by Git tag
[`tttv2-migration-audit-54648bc`](https://github.com/tenstorrent/tt_transformers/tree/tttv2-migration-audit-54648bc/qualification/evidence)
at commit `54648bcb966d6750ed6259e7716b8ad2900d596a`.

A deterministic archive created from that tag contains 310 files and has
SHA-256:

```text
72d38f7146299ce84c0f46725f2b6d48467c20d0b6dba3820bf817b152598222
```

The local archive manifest verified every file before cleanup. Repository
history retains the original exact-SHA records, and the audit tag makes the
raw files retrievable; the active source tree keeps only this compact summary.
The deterministic archive must be copied to the project's approved immutable
artifact store before publication.

## Galaxy 8x4 and 2D modules: not qualified here

The Galaxy 8x4 mesh support and the 2D tensor-parallel modules were ported from
`tt-metal` branch `apbernal/tttv2_wh_glx_2d_modules_milestone_c` at commit
`0036376d342bbd9de50ca8ea284b578838c44b29`. `docs/provenance/README.md`
describes that event. **Every Galaxy claim behind this code was measured on
that branch, on a 32-board Galaxy mesh, and none of it is re-qualified here.**

`tests/hardware/hardware-matrix.json` now declares **one** TG node,
`wh-tg-rmsnorm-2d-qk-norm`, and it is checked in **`enabled: false`** with
`disabled_classification: different_hardware_deferred`. It is a proposal, not a
gate: nothing selects it, `run_hardware_matrix.py` refuses it by design, and the
42-node result above is unchanged by its presence. It exists so the shape of a
Galaxy node class — the `TG` mesh value, a 32-board machine entry, the pool, the
cache requirement, the SKU provenance claim — is reviewable as a diff rather than
negotiated in the abstract. `fixture_policy.get_logical_sku` already maps a
32-device mesh to `TG`, so only the matrix entry was missing.

**Enabling it is an infrastructure commitment the port cannot make on its own.**
It means claiming a machine pool and extending the serial-reservation policy from
an 8-device host to a 32-board chassis. That last point is the real question:
`serialization.scope` is `physical_host` with one process, written for 8-device
hosts, and a Galaxy may share a chassis or fabric in ways the lock file does not
model. Settle that before flipping the flag.

The other nineteen Galaxy suites have no node. **Every ported device suite
therefore still lands as a file that no gate in this repository runs.**

### Galaxy module suites, executed by hand — evidence, not qualification

Because no node selects them, the WH Galaxy module suites were run directly on a
32-board `(8, 4)` mesh (`wh-glx6u-05`), one pytest node id per process, on
2026-09-11. **This is not a matrix result and must not be cited as one**: it was
driven by a shell loop, not `run_hardware_matrix.py`, so it carries none of the
evidence fields, serial-reservation guarantees or acceptance classification that
a node result carries.

- **57 module node ids, 53 passed, 50 minutes of device time.**
- Green across both Llama and Qwen shapes: attention 2D, embedding 2D, lm_head
  2D, MLP 2D, RMSNorm 2D (final-norm and q/k-norm), rope 2D, sampling 2D (exact
  and stochastic), column user selector, worker partition, page-table placement.
- **All 4 failures are in `tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py`.**
  Three are traced to one root cause, recorded below. The fourth
  (`attention_decode_with_active_prefetch`) fails and then hangs in teardown, so it
  was killed at its timeout and **its traceback was never written** — it is
  unexplained rather than explained by the same cause.

**Known defect — the global circular buffer cannot be re-placed after a prefill.**
`Prefetcher2D._release_global_cb()` runs on every `activate("prefill")` when
`release_global_cb_on_prefill` is set, and frees the buffer's L1 — but it does not
invoke `on_global_cb_released`, which is wired only into `cleanup()`. On Galaxy
that callback is what clears the mesh program cache and forgets the placement
record. Without it the cached decode programs stay resident, and so do their
semaphores; a semaphore is a 32-byte L1 allocation and `FreeListOpt::allocate`
prefers the smallest fitting block, so those blocks are taken below the buffer's
original address. The next `activate("decode")` then either trips the restore
guard (measured: 12 stray 32-byte blocks at 880352, below the free top 1368992
the first creation recorded) or fails outright in the allocator (792 064 B of CB
needed per bank against 709 152 B free). The mechanism is described exactly, and
independently, in `release_galaxy_global_cb_placement`'s own docstring — only the
wiring to the per-prefill release is missing.

Deliberately not patched here. Clearing the cache on every prefill→decode
transition is correct by that docstring's argument but recompiles every program
on the serving path; forgetting only the placement record is cheap and unsafe,
since that record is what keeps the address stable for programs still cached.
Which cost to pay is the module owners' call. **The restore guard raising is the
system working — do not relax it.**

Two further limits are worth stating plainly:

- **A green host suite is not evidence that these suites work.** The ported
  device tests are `device`-marked, so `pytest -m host` never collects them for
  execution. Four real undefined-name defects survived a green host run in this
  port and were caught only by `ruff`. Until a Galaxy node exists, lint is the
  only gate that reads those files at all.
- **The runner-derived reference comparisons are retired, and no recording will
  be produced.** In `tt-metal` the executor suites compared against tensors
  recorded by `GalaxyDirectRunner`, which wrapped the same model object the
  executor drives — so the gate validated the orchestration layer against a
  second orchestration layer, not Galaxy numerics. The runner is deliberately
  gone and nothing in this package can produce those tensors, so the
  comparisons were removed rather than reimplemented: two gates per model were
  deleted, and two were edited to keep the coverage that stands on its own. The
  surviving executor gates assert contracts and self-consistency — paged-KV
  capacity resolution and transactional bind/unbind, cross-slot agreement,
  program identity, teardown — not absolute values. The Galaxy numeric path is
  still checked against HuggingFace by `test_model_wh_galaxy.py` and the 2D
  module suites, and against the in-repo `.refpt` token assets by
  `test_executor_teacher_forced_accuracy`; none of that chain came from the
  runner. The follow-up worth building is the differential gate the 1D side
  already has (`test_w6_active15_padded16_trace_correctness`): two oracles from
  the same production executor in one process, no file and no second
  implementation. It is blocked today by the L1 address clash on a second KV
  allocation cycle in one process.

The two new model packages, `llama33_70b_galaxy` and `qwen3_32b_galaxy`,
deliberately ship **no** `support.json`. The manifest count below is therefore
still twelve against fourteen model directories: declaring a candidate geometry
that nothing in the matrix can select would be a false claim.

For the record, and as a developer-host observation rather than qualification
evidence: on the porting host the branch runs the host suite at parity with the
baseline it started from — 2,609 passes against a baseline 2,106, with the same
single pre-existing failure in both. That failure,
`test_benchmark_helper_does_not_require_private_tt_metal_infra_in_generic_ci`,
asserts that private tt-metal infrastructure is absent, and this host has it.
That run used `PYTHONPATH` against the checkout, not an unpacked sdist, so it
is not comparable to the cleaned-package figures above.

## Known limits

- All twelve model manifests remain experimental.
- The 42-node matrix covers selected regression cells, not every declared
  model geometry or workload.
- Four separate Llama 3.3 performance diagnostics at earlier code SHA
  `2883a949860d749adc2ed1af5525b27a9a547505` passed TTFT but missed their
  throughput floors; they were not rerun at the tested code above.
- Hardware results must be rerun when executable code, selectors, topology,
  target policy, or relevant test fixtures change. The current evidence binds
  only to `9134c399334240e3e4d35dfa93013c6c7293a3d1`; later commits do not
  inherit exact-SHA qualification.
- By that rule, the 42-node result above does **not** carry to the Galaxy port.
  The tested code is an ancestor of it, but the port modifies twelve production
  files under `src/tt_transformers/llm_runtime/` and
  `src/tt_transformers/modules/` — the code the module and runtime stages
  exercise. The 42 nodes must be rerun before the matrix result is claimed for
  any revision containing that port, independently of the separate question of
  Galaxy coverage.
- The two Galaxy model packages have no `support.json`, so the twelve
  experimental manifests above describe fourteen model directories.

Four deferrals the port recorded rather than resolved, all of them decisions for
the module and repository owners:

- **`MLP2D` and `RMSNorm2D` accept only the Galaxy `(8, 4)` mesh**
  (`WH_GALAXY_MESH_SHAPE`), where the pre-port version accepted any 2D mesh and
  its test parametrized `(4, 8)` as well. Kept narrow deliberately: it fails
  closed, it matches the only hardware that exists, and no node in the matrix
  declares a 2D mesh at all, so nothing exercisable regressed. **The `(4, 8)`
  coverage is a known deferral, not a silent narrowing** — restore the generic
  path when a non-Galaxy 2D mesh appears. The pre-port test is recoverable at
  `git show 38cbf1c:tests/modules/mlp/test_mlp_2d.py`.
- **`llm_runtime/decode.py` selects `prepare_rot_idxs` by `isinstance` on the
  rope config type.** 1D and 2D rope carry different config shapes
  (`device` vs `mesh_device`) and each module owns its own helper, and they
  cannot be merged behind one method because
  `test_legacy_rope_adapters_are_not_public` deliberately forbids `get_rot_idxs`
  on `RotarySetup1D`. So the branch is a type dispatch, not a model-family
  branch. `functools.singledispatch` over the config types, letting each rope
  module register its own helper, is the shape this wants. The import it needs is
  permitted by the boundary policy, so this is a tidiness question, not a
  blocker.
- **`tests/support/fixture_policy.get_updated_device_params` was a stub in this
  package before the port**, returning its input unchanged, so
  `dispatch_core_axis` reached `ttnn.open_mesh_device` verbatim and every device
  test setting it died at fixture setup. This is not a Galaxy issue: the
  package's own `tests/modules/sampling/test_legacy_sampling.py` passes the same
  key, so no device suite had run here since the extraction. Restored, with the
  Blackhole ROW→COL guard, but the owners should know the gap existed.
- **A differential correctness gate for the Galaxy executor does not exist yet.**
  The 1D side has one — `test_w6_active15_padded16_trace_correctness`, which
  builds two oracles from the same production executor in one process and
  compares them through `logits_oracle.assert_rowwise_logits_parity`, with no
  file and no second implementation. That is the right shape for Galaxy and is
  blocked today by an L1 address clash on a second KV allocation cycle in one
  process.
