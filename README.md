# TT Transformers

`tt-transformers` is the standalone home for reusable TTNN transformer
building blocks, runtimes, and model implementations for Tenstorrent devices.

> [!IMPORTANT]
> The repository is undergoing its initial extraction from `tt-metal`. No model
> is qualified from this standalone package until its support manifest and
> qualification report say otherwise.

## Installation

The first compatibility target is the published `ttnn==0.77.0` wheel on its
supported CPython 3.10 and 3.12, x86-64 manylinux platform matrix.

```bash
python -m pip install .
```

Examples and tests have separate dependency groups:

```bash
python -m pip install '.[examples]'
python -m pip install '.[test]'
python -m pip install '.[qualification]'  # only for live vLLM readiness clients
```

The package does not install firmware, drivers, checkpoints, tokenizers, or
model caches. The human-readable experimental matrix is in [`SUPPORT.md`](SUPPORT.md),
with per-example detail in [`examples/README.md`](examples/README.md) and each
example's machine-readable `support.json`. Concrete/runnable remains distinct
from qualified throughout these documents.

## Source provenance

The initial extraction is pinned to `tt-metal` revision
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`. The machine-readable file map
lives under `qualification/provenance/`. The `models/common/modules/moe`
subsystem is deliberately excluded; other boundary changes are recorded rather
than silently omitted.

See `TTTV2_MIGRATION_PLAN.md` for the migration design and
`TTTV2_MIGRATION_WORK_LOG.md` for current execution evidence.
