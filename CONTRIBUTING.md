# Contributing

Bugs are reported via [GitHub Issues](https://github.com/tenstorrent/tt-transformers/issues).
Bug fixes and new functionality are submitted via GitHub Pull Requests, which
are reviewed on a weekly cadence. By participating in this project, you agree
to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Set up a development environment

Use CPython 3.10 or 3.12 on Linux. For a normal editable checkout:

```bash
python -m pip install -e '.[examples,test,dev]'
pre-commit install
```

Maintainers and CI should prefer the hash-locked environments documented in
`constraints/README.md`.

To develop against a tt-metal checkout or a TTNN wheel built from a particular
commit, follow [Custom TTNN development](docs/ttnn-development.md). It provides
isolated source-linked and fixed-wheel environments while preserving release pins.

## Repository boundaries

- Production code may import only `tt_transformers`, declared third-party
  dependencies, and the Python standard library.
- Production code must not import repository `tests`, `examples`, or
  `qualification` modules.
- Examples are repository-only and may use `examples.common` plus documented
  qualification helpers.
- Test-only utilities belong under `tests/support`.
- New model support must update its `support.json`, documentation, and tests
  without claiming unexecuted hardware coverage.

## Before requesting review

```bash
pre-commit run --all-files
python -m pytest -m host
python tools/check_import_boundaries.py
python qualification/tools/validate_support_docs.py
python qualification/tools/run_hardware_matrix.py --validate
python tools/validate_lockfiles.py
python -m build
python -m twine check --strict dist/*
```

Hardware tests must identify the exact Git SHA, machine and physical SKU,
logical mesh, environment, selector, exit status, metrics, and teardown state.
One TT process may run per physical host. A skipped hardware test is not
evidence of support.

## Pull requests

Keep commits focused and explain user-visible behavior, validation performed,
and any remaining limitation. Never commit model checkpoints, caches,
credentials, machine-local logs, or migration work logs.
