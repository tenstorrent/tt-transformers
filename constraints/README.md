# Qualified dependency constraints

`host-py310.txt` and `host-py312.txt` freeze the complete dependency sets that
passed the standalone base, examples, and test import gates plus the full host
suite. Use PyPI as the primary index and the PyTorch CPU index as supplemental.
The locks intentionally select `torch==2.11.0+cpu`.

The version-only files remain convenient resolver constraints. Hash-complete,
wheel-only locks live under `constraints/locks/`:

- `base-py310.txt` and `base-py312.txt` for package runtime dependencies;
- `host-py310.txt` and `host-py312.txt` for base + examples + test;
- `qualification-py310.txt` and `qualification-py312.txt` for OpenAI/requests;
- `build-dev-py310.txt` and `build-dev-py312.txt` for build requirements,
  declared developer tools, and Twine's Markdown renderer.

Those eight files embed `--require-hashes`, record one resolver-selected wheel
SHA-256 per exact dependency, and pass clean strict installs plus `pip check`.
Per-wheel URL/index provenance is in
`qualification/reports/dependency-lock-evidence.json`; validation is provided
by `qualification/tools/validate_dependency_locks.py`.

PyYAML is already an exact hashed dependency in both base locks through TTNN;
there is deliberately no duplicate YAML-only environment.
