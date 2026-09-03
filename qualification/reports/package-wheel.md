# Phase 7 full package and wheel audit

Status: final post-triage wheel and sdist are published under `dist/`; archive policy and isolated base-wheel imports pass on CPython 3.10 and 3.12.

Audited distribution: `tt-transformers==0.1.0.dev0`

Compatibility tuple:

- `ttnn==0.77.0`;
- `torch==2.11.0`;
- `loguru==0.6.0`;
- CPython 3.10.19 and 3.12.13.

No example, test, qualification, YAML, or development extra was requested or installed. No model/checkpoint/cache path was opened and no TT hardware, remote host, or reset command was used.

## Release verdict

The final pure-Python wheel contains the complete current `src/tt_transformers` tree and imports from site-packages without access to the repository. Both Python environments imported all reusable modules, all runtime modules, all three shared executors, all twelve model packages, and every concrete core `model.py` while `transformers`, `tqdm`, and `pytest` were absent and actively blocked.

Two packaging defects were found and fixed before the final build:

1. `LMHead1D` no longer exposed the pinned `_nearest_32` alias consumed by eleven concrete model modules. The exact `nearest_32 as _nearest_32` import from `tt_transformers.tensor_utils` was restored. A static cross-module symbol audit then found zero unresolved internal imports.
2. `tensor_utils.serialize_config(..., fmt="yaml")` has a guarded PyYAML import but PyYAML was not declared directly. A non-base `yaml = ["PyYAML>=6.0,<7"]` extra was added, and dependency-root checking was extended across every installed layer. The qualified base tuple is unchanged.

The final rebuild also contains the subsequent TTNN 0.77 host-triage fix:
program configs without `to_json` serialize their supported public data fields
instead of relying on the broken 0.77 `SDPAProgramConfig.__repr__` binding.

## Final artifacts

Published artifact directory: `dist/`

| Artifact | Size | Archive entries | SHA-256 |
|---|---:|---:|---|
| `tt_transformers-0.1.0.dev0-py3-none-any.whl` | 597,778 bytes | 138 files | `8c55fac0a764fb9ae4d6ca514062ef2cb6cfe3097306a2f877bcad41269c50c2` |
| `tt_transformers-0.1.0.dev0.tar.gz` | 498,499 bytes | 143 files / 171 total members | `f161e13dedc5ce076d9553b677f0a1a4785996f932316f2325de9217376da644` |

These payloads were rebuilt from code SHA
`2883a949860d749adc2ed1af5525b27a9a547505`. Final hardware candidate
`73d414f8b826a7da982df8c8229d4ac41ed8ba33` has the identical
`src/tt_transformers` tree and `pyproject.toml`, so the same audited bytes,
source digest, and artifact hashes bind to that candidate. The intervening
hardware-matrix, runner-test, evidence, and report changes do not alter the
package payload or metadata.

Machine-readable evidence:

- `qualification/reports/package-wheel-manifest.json`: every wheel file with path, size, and content SHA-256, plus metadata summary and artifact hash;
- `qualification/reports/package-artifact-hashes.json`: wheel/sdist filenames, sizes, entry counts, SHA-256 hashes, pyproject hash, and deterministic source-tree identity.

The 138 wheel entries are exactly 132 source-identical Python files and six `.dist-info` metadata/license files. The audit compares every wheel Python payload byte-for-byte with the final `src/tt_transformers` file and rejects missing, extra, or changed Python content.

Release builds set `SOURCE_DATE_EPOCH=1788213269`, the pinned source revision's
commit timestamp. Setuptools honors it for wheel ZIP members. Because its sdist
staging tree otherwise carries wall-clock mtimes, `tools/normalize_sdist.py`
also normalizes tar member order, ownership and timestamps plus the gzip header.
Two independent builds then produced byte-identical artifacts with the hashes
above. The normalizer rejects absolute, traversing, and backslash member paths.

Final source identity:

```text
algorithm: sha256(path + NUL + content + NUL, sorted by path)
python files: 132
sha256: 7b03baf498e2a2252759d89813fcb898dd88daf573fb46f0e77e5c2cc6abad97
pyproject sha256: cc6442ca297411f6faffd72e90bab4a05fdc8af5f558f2d5d423684bec63f98c
```

Before publication, `dist/` contained only the known 8,980-byte scaffold
wheel with SHA-256
`d7a07395c3230f3de4d03579a6d7702b032df960d938044a497f97334f49a224`.
Both size and hash were asserted immediately before replacement. That one file
was overwritten and the new sdist was added; no unrelated file was removed.

## Build

The local isolated build attempt could not populate its temporary environment because sandbox networking could not download its `wheel` build requirement. The repository environment already had `setuptools==80.0.0` and a functioning `bdist_wheel` command, so the successful build used the local PEP 517 backend without isolation:

```bash
cd /localdev/gwang/tt_transformers
package_out=$(mktemp -d /tmp/gwang/tttv2-package-build-final.XXXXXX)
python -m build --no-isolation --skip-dependency-check --outdir "$package_out"
```

Result:

```text
Successfully built tt_transformers-0.1.0.dev0.tar.gz and tt_transformers-0.1.0.dev0-py3-none-any.whl
```

The wheel declares `Root-Is-Purelib: true` and `Tag: py3-none-any`.

## Metadata checks

The archive audit and independent `pkginfo` readers validated both artifacts:

- Core Metadata 2.4;
- name `tt-transformers`;
- version `0.1.0.dev0`;
- `Requires-Python: <3.13,>=3.10`;
- license expression `Apache-2.0` plus LICENSE and NOTICE payloads;
- base requirements exactly `ttnn==0.77.0`, `torch==2.11.0`, and `loguru==0.6.0`;
- extras `examples`, `test`, `qualification`, `yaml`, and `dev` remain conditional;
- non-empty Markdown long description;
- wheel `RECORD` contains exactly every archive path, uses SHA-256, and has correct sizes/digests;
- wheel and sdist PKG-INFO/METADATA agree on the checked identity/dependency fields.

The image's pre-existing `twine==4.0.2` executable failed before reading any artifact:

```text
KeyError: 'license'
```

That traceback originated in Twine's own `importlib_metadata` adapter, not the
built distribution. The canonical upload check was then rerun in a clean
temporary environment with `twine==7.0.0` and `readme-renderer==46.0` including
Markdown support:

```text
Checking dist/tt_transformers-0.1.0.dev0-py3-none-any.whl: PASSED
Checking dist/tt_transformers-0.1.0.dev0.tar.gz: PASSED
```

The command was `python -m twine check --strict` over both final artifacts and
exited zero. Direct Core Metadata, `pkginfo`, WHEEL/RECORD, and archive policy
validation also pass.

## Archive policy

`tools/audit_package_artifacts.py` rejects either archive if it contains:

- any `tests`, `examples`, or `qualification` path component;
- model/checkpoint/reference tensor or cache extensions (`.bin`, `.pt`, `.pth`, `.refpt`, `.safetensors`, `.tensorbin`, `.npy`, `.npz`);
- bytecode or `__pycache__`;
- a top-level legacy `models` namespace;
- executable Python imports from `models.*`, `tests`, `examples`, or `pytest`;
- absolute/traversing/backslash archive member paths;
- embedded workspace paths such as `/localdev`, `/home/gwang`, or `/tmp/gwang`;
- duplicate members;
- malformed/mismatched metadata or `RECORD` entries;
- wheel Python content that differs from the final source tree.

Final result: zero violations in both archives.

Reproduce the check and manifests with:

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/audit_package_artifacts.py \
  dist/tt_transformers-0.1.0.dev0-py3-none-any.whl \
  dist/tt_transformers-0.1.0.dev0.tar.gz \
  --source-root src \
  --output-dir qualification/reports
```

## Isolated installations

Fresh environments:

| Environment | Interpreter | Package location |
|---|---|---|
| `/tmp/gwang/tttv2-wheel-audit.yxKXYp/py310` | CPython 3.10.19 | `.../lib/python3.10/site-packages/tt_transformers` |
| `/tmp/gwang/tttv2-wheel-audit.yxKXYp/py312` | CPython 3.12.13 | `.../lib/python3.12/site-packages/tt_transformers` |

The initial installs resolved the wheel's exact base dependencies. The final
`dist/` wheel was reinstalled non-editably with `--force-reinstall --no-deps`,
retaining the already verified exact tuple. Neither environment contains
distributions or import specs for `transformers`, `tqdm`, or `pytest`.

Every installed probe ran from `/tmp` with `python -I`. Assertions verified that `/localdev/gwang/tt_transformers` was absent from `sys.path` and that `tt_transformers.__file__` and distribution files resolve under that environment's site-packages.

## Import surface

`tools/probe_installed_package.py` imports 76 named surfaces in each environment:

- root cache, environment, device ownership, device/tensor/mesh helpers;
- all reusable attention, embedding, lazy buffer/weight, LM head, MLP, RMSNorm, RoPE, sampling, and CCL modules;
- every LLM runtime and prefill module;
- `models.executor`, `models.llama3_executor`, and `models.qwen2_executor`;
- all twelve concrete package initializers;
- all twelve concrete core model modules.

Model families:

```text
deepseek_r1_distill_qwen_14b
llama32_1b
llama32_3b
llama33_70b
llama3_8b
mistral_7b
phi4
qwen25_72b
qwen25_7b
qwen25_coder_32b
qwen2_7b
qwen3_32b
```

Both probes report:

- 132 installed Python files;
- `tt-transformers==0.1.0.dev0`;
- `ttnn==0.77.0`;
- `torch==2.11.0`;
- `loguru==0.6.0`;
- no optional module loaded;
- static installed-source boundary pass;
- cache/device preflight pass;
- cleanup ownership surface pass.
- TTNN 0.77 SDPA program-config serialization pass.

The package imports TTNN, which prints its normal configuration debug line. It does not open a device.

To keep the repository off `sys.path`, stage and run the probe from `/tmp`:

```bash
cp tools/probe_installed_package.py /tmp/gwang/tttv2-wheel-audit.yxKXYp/probe_installed_package.py
cd /tmp
/tmp/gwang/tttv2-wheel-audit.yxKXYp/py310/bin/python -I \
  /tmp/gwang/tttv2-wheel-audit.yxKXYp/probe_installed_package.py
/tmp/gwang/tttv2-wheel-audit.yxKXYp/py312/bin/python -I \
  /tmp/gwang/tttv2-wheel-audit.yxKXYp/probe_installed_package.py
```

## Static boundary, preflight, and cleanup ownership

The installed probe parses all 132 wheel Python files and rejects legacy/test/example imports, embedded workspace paths, and unscoped TTNN default-device access. Both versions pass.

Cache/device preflight verifies:

- CI does not imply offline mode;
- a token-like environment value is redacted;
- the installed TTNN version in the cache identity is `0.77.0`;
- the default-device fallback ledger has ten entries and no active owner remains.

Cleanup ownership verifies callable surfaces for:

- device object-graph/model/DP cleanup;
- best-effort tensor-tree deallocation and orphan retry;
- attached/raised cleanup failures;
- scoped/compatibility default-device ownership;
- `ModelExecutor.cleanup`.

No cleanup call is executed because no TT resource is created.

## Dependency integrity

Both environments pass:

```bash
python -m pip check
```

Result: `No broken requirements found.`

## Acceptance summary

| Gate | CPython 3.10 | CPython 3.12 |
|---|---|---|
| Non-editable wheel install with exact base tuple | pass | pass |
| Repository absent from `sys.path` | pass | pass |
| Optional packages absent and blocked | pass | pass |
| Root/reusable/runtime/shared-executor imports | pass | pass |
| Twelve model package/core imports | pass | pass |
| Installed static boundary | pass | pass |
| Cache/device preflight | pass | pass |
| Cleanup ownership surfaces | pass | pass |
| TTNN 0.77 program-config serialization | pass | pass |
| `pip check` | pass | pass |

Archive, metadata, manifest, and source-identity gates also pass. This is a host packaging verdict only; it does not claim model correctness or hardware support.
