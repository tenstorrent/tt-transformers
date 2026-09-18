# Qwen3 32B on WH Galaxy with TTTv2

The 2D tensor-parallel reconstruction of `Qwen/Qwen3-32B` for one Wormhole Galaxy `(8, 4)` mesh —
32 boards, 8 rows × 4 columns. Checkpoint revision `9216db5781bf21249d130ec9da846c4624c16137`, the
same revision the non-Galaxy `examples/qwen3_32b` package pins.

The package is `tt_transformers.models.qwen3_32b_galaxy`; the shared Galaxy runtime it builds on is
`tt_transformers.models.galaxy`.

## There is no demo in this directory, and that is deliberate

Every other `examples/<model>/` directory carries a `demo.py` and a `support.json`. This one carries
neither, for two separate reasons.

**No `demo.py`.** The demo that existed upstream was a *direct* demo: it drove the model
graph through a `GalaxyDirectRunner`, before model-owned executors existed. That runner was condemned
by the 2026-09-02 operator decision and is not part of this package, and all four of the demo's cases
called into it. Porting the file would have produced a module that cannot import. The executor path
that supersedes it is exercised by `tests/models/qwen3_32b_galaxy/test_executor_wh_galaxy.py` and
`test_trace_wh_galaxy.py`.

**No `support.json`.** A support manifest declares a candidate hardware geometry and a software
tuple. This model's declared geometry is a 32-device Galaxy mesh, and no node in
`tests/hardware/hardware-matrix.json` describes one — there is no `mesh_device` value, no machine
entry, and no serial-lock semantics for a 32-board mesh. Writing a manifest now would declare a
support contract that nothing in this repository can select, let alone qualify. It is left absent
until a Galaxy node class is agreed with the repository owners.

## What is here

| Surface | Location |
| --- | --- |
| Model, weights, HF adaptor, executor | `src/tt_transformers/models/qwen3_32b_galaxy/` |
| Shared Galaxy runtime (CCL, resources, plans, recipes, prefetch) | `src/tt_transformers/models/galaxy/` |
| 2D modules the model is built from | `src/tt_transformers/modules/*/[a-z_]*_2d.py` |
| Host tests | `tests/models/qwen3_32b_galaxy/*_host.py` |
| Device tests (need a Galaxy mesh; no matrix node selects them yet) | `tests/models/qwen3_32b_galaxy/*_wh_galaxy.py` |

This is a 64-layer model against Llama's 80, which is what made the two a useful pair: several
findings only became legible because the same measurement was taken on both.

## Qualification status

**Nothing here is qualified under this repository's hardware policy.** The device claims behind this
model were measured on a WH Galaxy `(8, 4)` on the tt-metal branch it was ported from, under a serial
house rule that closely matches this repository's `serialization` policy — one process per physical
host, external reservation, operator-only reset. They have **not** been re-measured here, because the
hardware matrix does not yet describe the hardware.

Two further limits are worth stating plainly:

- The executor reference tensors that `test_executor_wh_galaxy.py` compares against are not in this
  repository. They were 1.37 GB of `.pt`, one file above GitHub's 100 MB hard limit, and the
  generator that wrote them was itself runner-based. Those tests skip until the references are
  regenerated against an executor path.
- Performance is not characterised. No throughput or latency figure for this model is qualified
  here, and the performance investigation was deliberately left behind rather than ported
  mid-flight.

## Running the host tests

```sh
pytest tests/models/qwen3_32b_galaxy -m host
```

The device suites are marked `device`, `wormhole` and `galaxy`, so `-m host` excludes them and
`pytest -m host` in CI never attempts to open a mesh.
