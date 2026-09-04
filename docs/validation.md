# Validation

This page separates the most recent hardware-qualified code from validation
of the cleaned package candidate. Results apply only to the named revision or
artifact, not as a blanket support promise.

## Hardware

- Tested code: `73d414f8b826a7da982df8c8229d4ac41ed8ba33`
- Matrix: 42/42 nodes passed
- Stages: modules 30/30, runtime 1/1, smoke 3/3, end-to-end 8/8
- Hosts: Wormhole 23/23, Blackhole 19/19
- Meshes: N150 9/9, N300 6/6, T3K 8/8, P150 8/8, P150x4 11/11
- Functional, lifecycle, missing-acceptance, and reset outcomes: zero

The eight `MESH_DEVICE=P150` records were logical 1x1 executions on a physical
P150_X4 quietbox with `TT_VISIBLE_DEVICES` unset. They are not standalone-P150
product evidence.

## Cleaned package candidate

The cleanup candidate represented by the Git revision containing this page
passed the host suite from an independently unpacked sdist on CPython 3.10 and
3.12 with 2,169 passes, 28 intentional skips, 6,791 deselections, 5 warnings,
and 81 passing subtests per interpreter. Its sdist rebuilt a wheel
byte-for-byte identical to the checkout build. Two checkout builds also
produced identical wheels and identical normalized sdists. The package was
validated against `ttnn==0.77.0`.

Release artifact hashes belong in the immutable release/CI attestation. They
cannot be embedded in the sdist without changing that sdist's own digest.

The pre-cleanup non-editable-wheel host baseline was 2,170 passes, 28
intentional skips, 6,791 deselections, 5 warnings, and 81 passing subtests per
interpreter.

## Archived evidence

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

## Known limits

- All twelve model manifests remain experimental.
- The 42-node matrix covers selected regression cells, not every declared
  model geometry or workload.
- Four separate Llama 3.3 performance diagnostics at earlier code SHA
  `2883a949860d749adc2ed1af5525b27a9a547505` passed TTFT but missed their
  throughput floors; they were not rerun at the tested code above.
- Hardware results must be rerun when executable code, selectors, topology,
  target policy, or relevant test fixtures change.
