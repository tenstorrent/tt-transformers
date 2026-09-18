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

## Known limits

- All twelve model manifests remain experimental.
- The recorded 42-node matrix covers selected regression cells, not every declared
  model geometry or workload.
- The active matrix now contains 51 nodes, including the HF generation and
  Wormhole cached-reference gates described in [tests](../tests/README.md).
  Their results require new evidence; the historical 42-node result does not
  qualify these additional gates.
- Llama 3.3 performance is not characterised by this record. No throughput or
  time-to-first-token figure here qualifies that model.
- Hardware results must be rerun when executable code, selectors, topology,
  target policy, or relevant test fixtures change. The current evidence binds
  only to `tt-transformers==2.0.0.dev0` as released; later changes do not
  inherit its qualification.
