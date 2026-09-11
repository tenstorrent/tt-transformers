# TT Transformers

`tt-transformers` provides reusable TTNN transformer modules, model-neutral
LLM runtime components, sampling utilities, and model implementations for
Tenstorrent hardware.

> [!IMPORTANT]
> The package is currently a developer preview. Every model remains
> `experimental`; consult the [support matrix](https://github.com/tenstorrent/tt_transformers/blob/main/SUPPORT.md)
> before relying on a model, topology, or workload.

## Requirements

- Linux on x86-64
- CPython 3.10 or 3.12
- `ttnn==0.77.0`
- a compatible Tenstorrent driver, firmware, and device configuration for
  hardware execution

The Python package does not install firmware, drivers, model checkpoints,
tokenizers, or model caches.

Each tt-transformers release uses a pinned, publicly available TTNN package.
See [release compatibility](src/tt_transformers/README.md#release-compatibility)
for the release policy and the separate custom-runtime development workflow.

## Install

Install the library from a checkout:

```bash
git clone https://github.com/tenstorrent/tt_transformers.git
cd tt_transformers
python -m pip install .
```

Repository examples are not installed by the wheel. To run them from a clone,
install their dependencies as well:

```bash
python -m pip install -e '.[examples]'
```

Host tests use a separate extra:

```bash
python -m pip install -e '.[test]'
python -m pytest -m host
```

For fully reproducible maintainer environments, use the hash-locked files in
[`constraints/locks`](https://github.com/tenstorrent/tt_transformers/tree/main/constraints/locks).

## Verify the installation

```bash
python -c 'import tt_transformers; print(tt_transformers.__version__)'
```

The top-level import is deliberately lightweight. Model families are imported
from `tt_transformers.models`, while reusable building blocks live under
`tt_transformers.modules`, `tt_transformers.llm_runtime`, and
`tt_transformers.sampling`.

## Learn the package

- [Software architecture](docs/architecture.md)
- [Package guide](src/tt_transformers/README.md)
- [Models and deprecation policy](src/tt_transformers/models/README.md)
- [Reusable modules](https://github.com/tenstorrent/tt_transformers/blob/main/src/tt_transformers/modules/README.md)
- [LLM runtime](https://github.com/tenstorrent/tt_transformers/blob/main/src/tt_transformers/llm_runtime/README.md)
- [Sampling](https://github.com/tenstorrent/tt_transformers/blob/main/src/tt_transformers/sampling/README.md)
- [Repository examples](https://github.com/tenstorrent/tt_transformers/blob/main/examples/README.md)
- [Validation summary](https://github.com/tenstorrent/tt_transformers/blob/main/docs/validation.md)

## Development

See [CONTRIBUTING.md](https://github.com/tenstorrent/tt_transformers/blob/main/CONTRIBUTING.md),
the [test guide](https://github.com/tenstorrent/tt_transformers/blob/main/tests/README.md),
and the [tool guide](https://github.com/tenstorrent/tt_transformers/blob/main/tools/README.md).

To develop against a selected tt-metal commit or edit TTNN alongside
tt_transformers, see the [TTNN development guide](docs/ttnn-development.md).

Report security issues through the process in
[SECURITY.md](https://github.com/tenstorrent/tt_transformers/blob/main/SECURITY.md).

## Provenance

The initial standalone source was extracted from `tt-metal` commit
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`. MoE was intentionally excluded.
The compact source map is retained under
[`docs/provenance`](https://github.com/tenstorrent/tt_transformers/tree/main/docs/provenance).
