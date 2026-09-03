# CI and dependency constraints

Status: all eight target-specific artifact-hash locks are validated; the
consolidated CI consumes host, base, and build/dev locks but remains unobserved,
while post-qualification static-quality cleanup remains a release gate.

## Constraints

The qualified base/examples/test environments are frozen separately because
Python 3.10 and 3.12 select different transitive wheels and versions:

| Target | File | Packages | SHA256 |
| --- | --- | ---: | --- |
| CPython 3.10 | `constraints/host-py310.txt` | 63 | `b18a42fa787a81df5cca8bc4a6a56953016f14d4a654cb2d56e079d120f1ec44` |
| CPython 3.12 | `constraints/host-py312.txt` | 60 | `f1aeb029b428de852ff3e10fef19f75e9c9b8adf76ea21fb2bc7c843ee7fe7b5` |

Both select `ttnn==0.77.0`, `torch==2.11.0+cpu`, and `loguru==0.6.0` and
exactly reproduce the refreshed dependency-report freezes after excluding the
environment's bootstrap pip/setuptools tools. Run:

```bash
python -B qualification/tools/validate_constraints.py
```

The version-only constraints are complemented by eight target locks under
`constraints/locks`: base=31/29, host=64/61, qualification=20/19, and
build/dev=44/40 packages on CPython 3.10/3.12. Each selected wheel has an exact
SHA-256 plus URL/index provenance; all eight locks pass clean
`--require-hashes` installation and `pip check`.
See `qualification/reports/dependency-locks.md` and validate with:

```bash
python -B qualification/tools/validate_dependency_locks.py
```

Both base locks already contain hashed PyYAML 6.0.3, so no YAML-only lock is
invented. The consolidated workflow consumes host/base locks and creates its
build/Twine environment from the matching build/dev lock. It does not run live
qualification extras, although their two locks are independently strict-
installed. The workflow has not yet been observed on GitHub-hosted runners.

## Consolidated host workflow

The former overlapping workflows are replaced by
`.github/workflows/host.yml`, with exact CPython 3.10.19/3.12.13 matrix legs on
Ubuntu 24.04. Each leg:

1. uses PyPI primary and the PyTorch CPU index supplemental;
2. installs the target hash lock with `--require-hashes` and checks dependency consistency;
3. validates dependency locks and the explicit non-green static-quality
   baseline alongside taxonomy, documentation, provenance, import/API,
   capability, hardware-matrix, TTNN-compatibility, and readiness policy;
4. builds and audits wheel/sdist archives, then runs strict Twine validation;
5. installs the wheel non-editably and runs the complete `pytest -m host`
   selector without `PYTHONPATH`;
6. creates a second base-only environment and probes the installed wheel under
   isolated Python with optional dependencies absent.

The device-policy job performs only static scheduling validation. It never
imports TTNN or runs device tests, and preserves the one-node/one-process
hardware boundary.

The final local non-editable-wheel host selector passes identically on both
supported interpreters: 2,170 passed, 28 skipped, 6,791 deselected, 5 warnings,
and 81 passing subtests, with zero failures or errors. This is local evidence;
it does not convert the still-unobserved GitHub workflow into a CI pass.

This workflow has not yet produced an observed GitHub-hosted result. Its
Twine/build environment is separately provisioned from the hash-complete
build/dev lock; a clean local Twine 7.0.0/readme-renderer 46.0 strict check
passes both final artifacts.

## Deferred static-quality gate

A known-failing required lint/type job was not checked in. The exact isolated
baseline records 2,158 Ruff findings (628 legacy CLI-comparable fixable; JSON
classifies 505 safe and 154 unsafe fix applications), 167 files needing Ruff
formatting with 204 already formatted, and 489 mypy errors across 99 package
files. TTNN provides no complete typing marker/stubs, and the bounded Black
check did not complete reliably.
Phase 9 begins only after hardware parity by plan; it must select one formatter,
qualify the declared tool versions, establish a reviewed baseline, and then
make Ruff/format/mypy regressions required without masking real defects.
