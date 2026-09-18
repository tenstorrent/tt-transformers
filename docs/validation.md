# Validation

This page summarizes validation of the cleaned package candidate. Results
apply only to the named revision or artifact, not as a blanket support
promise.

## Demo and benchmark coverage

The short `examples/<model>/demo.py` exercises public loading, chat-template
tokenization, `.generate()`, decoding, and cleanup. The separate `benchmark.py`
retains accuracy/performance workloads; existing hardware wrappers named
`test_demo.py` now delegate to that benchmark module.

A passing text-generation demo or reference-accuracy gate does not imply that
a benchmark meets its latency or throughput targets. Demo and benchmark
results must identify the tested entry point and revision independently.

## Hardware

- Tested release: `tt-transformers==2.0.0.dev0`
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

## Galaxy 8x4 and 2D modules: not qualified here

The Galaxy 8x4 mesh support and the 2D tensor-parallel modules are **not
qualified by the record above.** No enabled node in
`tests/hardware/hardware-matrix.json` selects them.

The matrix declares one `TG` node and one `BHGLX` node, both checked in
`enabled: false` with `disabled_classification: different_hardware_deferred`.
They are proposals rather than gates: nothing selects them and
`run_hardware_matrix.py` refuses them by design, so the enabled-node result is
unchanged by their presence. They exist so that the shape of a Galaxy node
class — the mesh value, a 32-board machine entry, the pool, the cache
requirement, the SKU provenance claim — is reviewable as a diff.

Every Galaxy device suite therefore lands as a file that no gate in this
repository runs. A green host suite is not evidence that those suites work:
they are `device`-marked, so `pytest -m host` never collects them for
execution.

The two Galaxy model packages, `llama33_70b_galaxy` and `qwen3_32b_galaxy`,
deliberately ship **no** `support.json`. The manifest count below is therefore
twelve against fourteen model directories: declaring a candidate geometry that
nothing in the matrix can select would be a false claim.

## Known limits

- All twelve model manifests remain experimental.
- The recorded 42-node matrix covers selected regression cells, not every declared
  model geometry or workload.
- The active matrix now contains 51 enabled nodes, including the HF generation
  and Wormhole cached-reference gates described in [tests](../tests/README.md).
  Their results require new evidence; the historical 42-node result does not
  qualify these additional gates.
- Llama 3.3 performance is not characterised by this record. No throughput or
  time-to-first-token figure here qualifies that model.
- Hardware results must be rerun when executable code, selectors, topology,
  target policy, or relevant test fixtures change. The current evidence binds
  only to `tt-transformers==2.0.0.dev0` as released; later changes do not
  inherit its qualification.
- By that rule, the recorded matrix result does **not** carry to the Galaxy 2D
  port. That port modifies production files under
  `src/tt_transformers/llm_runtime/` and `src/tt_transformers/modules/` — the
  code the module and runtime stages exercise — so the matrix must be rerun
  before its result is claimed for any revision containing the port,
  independently of the separate question of Galaxy coverage.
- The two Galaxy model packages have no `support.json`, so the twelve
  experimental manifests above describe fourteen model directories.
- `MLP2D` and `RMSNorm2D` accept only the Galaxy `(8, 4)` mesh
  (`WH_GALAXY_MESH_SHAPE`). The transposed `(4, 8)` orientation is rejected by
  construction rather than tested: the narrow contract fails closed and matches
  the only hardware that exists, and no node in the matrix declares a 2D mesh at
  all. Restoring a generic 2D path is a deferral to revisit when a non-Galaxy 2D
  mesh appears, not a silent narrowing.
- No differential correctness gate exists for the Galaxy executor.
