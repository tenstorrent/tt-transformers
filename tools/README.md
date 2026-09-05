# Repository tools

These tools support package policy, reproducible builds, and maintainer
validation. They are not installed by the wheel.

| Tool | Purpose | Writes repository files? |
|---|---|---|
| `audit_package_artifacts.py` | Validate wheel/sdist contents, metadata, paths, and source identity | Only when an output directory is explicitly supplied |
| `build_dependency_locks.py` | Generate hash-complete dependency locks from resolver reports | Yes; lock output is explicit |
| `check_import_boundaries.py` | Reject legacy or reverse-layer production imports | No |
| `normalize_sdist.py` | Normalize an existing sdist for reproducible bytes | Yes; rewrites the named archive |
| `probe_installed_package.py` | Probe a non-editable installed wheel outside the checkout | No |
| `strict_install_dependency_locks.py` | Install locks in fresh environments and record results | Writes only the requested work/output paths |
| `validate_lockfiles.py` | Validate lock syntax, hashes, and cross-group invariants | No |

Run tools from the repository root. Build output belongs under ignored `dist/`
or a temporary directory. Never commit build directories, environments,
resolver caches, credentials, or machine-local output.

Hardware/readiness-specific utilities are documented in
`qualification/README.md`.
