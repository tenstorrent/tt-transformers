# CI and release dependency locks

`constraints/locks/` contains hash-complete, wheel-only dependency resolutions
for CPython 3.10 and 3.12 on x86-64 Linux.

- `base-py*.txt`: installed package runtime dependencies
- `host-py*.txt`: base plus repository example and test dependencies
- `qualification-py*.txt`: repository readiness-client dependencies
- `build-dev-py*.txt`: build, lint, typing, and publication tools

The locks use PyPI as the primary index and the PyTorch CPU index as a
supplemental index. Base and host locks select `ttnn==0.77.0`,
`torch==2.11.0+cpu`, `loguru==0.6.0`, and `transformers==5.12.1`.

They are maintainer/CI environments, not constraints imposed on downstream
applications. Normal users install the package metadata declared in
`pyproject.toml`.

Validate lock syntax and group invariants with:

```bash
python tools/validate_lockfiles.py
```
