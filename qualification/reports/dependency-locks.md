# Hash-complete dependency locks

Status: **eight target locks pass clean strict installation and `pip check`**

This evidence covers base, host (base + examples + test), qualification
(OpenAI + requests), and isolated build/dev tooling on both supported
interpreters. It never includes `tt-transformers` itself.

## Locked targets

| Target | Packages | Lock SHA-256 | Resolver | Selected indexes |
|---|---:|---|---|---|
| CPython 3.10 base | 31 | `224ed4d1da7b50ec9c341489e64b54c462eda2e16dda8eac41f1c25de8f00f08` | pip 23.0.1 | 29 PyPI; 2 PyTorch CPU |
| CPython 3.10 host | 64 | `7808a399f77588d1ff48c9a3173860d2b97f4b9613e8927e9b261c94ce1d635e` | pip 23.0.1 | 62 PyPI; 2 PyTorch CPU |
| CPython 3.10 qualification | 20 | `ea7fbd3222bbdcbc1431f793b934dac03bc53bb5cfbe000bf7ae24c83d031c02` | pip 23.0.1 | 20 PyPI |
| CPython 3.10 build/dev | 44 | `0c4f459c6d1402fc2d3595c3bf7f512fb1ea9ec1c50c6d0f45e6ad7a96cfdda0` | pip 23.0.1 | 44 PyPI |
| CPython 3.12 base | 29 | `74654b929c05a69a980f95926d630bcbc5253198cc400d4494466a9fd83fa4a8` | pip 25.0.1 | 26 PyPI; 3 PyTorch CPU |
| CPython 3.12 host | 61 | `ce2e5593562bda1098ed050244a53b61e8542f7a3aae9ad8334a0d293eade3af` | pip 25.0.1 | 58 PyPI; 3 PyTorch CPU |
| CPython 3.12 qualification | 19 | `3c3514aabb88ef7bc5edaacee6c84c9ba99529c51261b22cb6d08ff50c0fca2b` | pip 25.0.1 | 19 PyPI |
| CPython 3.12 build/dev | 40 | `ed0deb26dcbdfe67f54b71481ccfb9c8de16fa6f15938dc563e42ee713bc4916` | pip 25.0.1 | 40 PyPI |

Lock files:

- `constraints/locks/base-py310.txt`
- `constraints/locks/host-py310.txt`
- `constraints/locks/qualification-py310.txt`
- `constraints/locks/build-dev-py310.txt`
- `constraints/locks/base-py312.txt`
- `constraints/locks/host-py312.txt`
- `constraints/locks/qualification-py312.txt`
- `constraints/locks/build-dev-py312.txt`

Each lock embeds `--require-hashes`, `--only-binary :all:`, PyPI as the primary
index, and the PyTorch CPU index as supplemental. Every dependency is an exact
name/version pin with exactly the SHA-256 of the wheel selected by that target's
resolver report. Editable, VCS, filesystem-path, source-archive, unhashed, and
`tt-transformers` requirements are rejected.

## URL and index provenance

`dependency-lock-evidence.json` records, for every selected wheel:

- canonical distribution name and exact version;
- decoded wheel filename;
- resolver-selected HTTPS URL;
- `pypi` or `pytorch_cpu` index provenance;
- archive SHA-256; and
- whether it was a directly requested requirement.

The supplemental index is not described as strictly lower-priority because pip
searches all configured indexes. The evidence instead records the actual chosen
archive. Torch always resolves to `2.11.0+cpu` from
`download-r2.pytorch.org/whl/cpu`: SHA-256 `f378df...8b75` for CPython 3.10 and
`f82e2a...c338` for CPython 3.12. MarkupSafe is also selected from the PyTorch
index on both targets; Python 3.12 selects Jinja2 there as well. Python 3.10's
pip rejects that index's Jinja2 metadata name casing and selects the matching
PyPI wheel, which is captured rather than normalized away.

Qualification locks directly request the declared `openai>=1,<3` and
`requests>=2.31` extras while constraining shared packages to the proven host
versions. Build/dev directly covers `setuptools>=80`, wheel, build, the declared
Black/Ruff/mypy tools at their measured baseline versions, and the proven
Twine 7.0.0 + readme-renderer 46.0 Markdown toolchain.

## YAML ownership

No separate YAML environment is invented. `ttnn==0.77.0` already brings
PyYAML into both base closures, and both base locks contain a target wheel for
`pyyaml==6.0.3` with its exact hash. The validator requires that fact and the
evidence policy records `yaml_provided_by_base_closure=true`.

## Strict clean-install evidence

The original four environments were created under
`/tmp/gwang/tt-transformers-lock-install-v1`; four additional auxiliary
environments were created under
`/tmp/gwang/tt-transformers-aux-lock-install-v1`. Each ran its target lock with
`python -m pip install --require-hashes -r <lock>`, followed by `pip check`, an
exact Torch distribution probe, a `tt_transformers` absence probe, and a full
installed freeze comparison against the lock. A shared pip download cache was
used only to avoid downloading the same large wheel repeatedly; no installed
environment was shared.

All eight results pass:

- install exit 0 with hash enforcement;
- `pip check`: `No broken requirements found.`;
- installed packages exactly equal the lock plus venv bootstrap tooling;
- Torch is exactly `2.11.0+cpu` in base/host and absent from isolated
  qualification/build-dev environments; and
- `tt_transformers` is absent.

## Reproduction

Resolve using the target interpreter, exact version constraint, PyPI primary,
PyTorch CPU supplemental, `--only-binary=:all:`, `--ignore-installed`, and pip's
`--report`. Then regenerate locks:

```bash
python -B qualification/tools/build_dependency_locks.py \
  --resolver-dir /path/to/eight-pip-reports \
  --write
```

Run strict clean installs and merge their evidence:

```bash
python qualification/tools/strict_install_dependency_locks.py \
  --work-root /new/empty/path \
  --output /tmp/strict-install-results.json
python -B qualification/tools/build_dependency_locks.py \
  --resolver-dir /path/to/eight-pip-reports \
  --validation-results /tmp/strict-install-results.json \
  --write
python -B qualification/tools/validate_dependency_locks.py
```

The schema is `qualification/schemas/dependency-lock-evidence.schema.json`.
This is dependency artifact evidence only; it makes no TTNN semantic, model,
checkpoint, firmware, driver, or hardware claim.
