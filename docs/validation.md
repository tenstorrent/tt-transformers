# Validation

This page summarizes validation of the cleaned package candidate. Results
apply only to the named revision or artifact, not as a blanket support
promise.

## Hardware

- Tested code: `9134c399334240e3e4d35dfa93013c6c7293a3d1`
- Executed: 2026-09-04 20:58–22:39 UTC
- Matrix: 42/42 nodes passed
- Stages: modules 30/30, runtime 1/1, smoke 3/3, end-to-end 8/8
- Hosts: Wormhole 23/23, Blackhole 19/19
- Meshes: N150 9/9, N300 6/6, T3K 8/8, P150 8/8, P150x4 11/11
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

`tests/hardware/hardware-matrix.json` declares no Galaxy, TG or 32-chip node.
The mesh values it does declare — N150, N300, T3K, P150, P150x4 — are all 1xN,
so no node selects a 2D mesh at all and no gate in this repository executes any
ported device suite. The suites land as files. Adding a Galaxy node class is a
prerequisite to be agreed with the repository owners, not a detail: it needs
new `mesh_device` values, a machine pool, a cache requirement, and the serial
lock semantics extended to a 32-board mesh.

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
