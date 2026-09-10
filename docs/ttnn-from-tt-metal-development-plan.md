# Plan: develop tt_transformers against TTNN built from a tt-metal commit

Status: approved design. The initial implementation and supported interfaces are
documented in [Custom TTNN development](ttnn-development.md). Examples below are
the original proposed interfaces; use the development guide for actual commands.

Prepared: 2026-09-10.

## Recommendation

Support two explicit TTNN installation modes:

| Mode | TTNN installation | Main use |
| --- | --- | --- |
| Source-linked development | Upstream editable installation linked to a tt-metal checkout and its native build | Editing tt-metal and tt_transformers together |
| Fixed wheel | A verified normal wheel built from a recorded source snapshot | Reproducible CI and development that only changes tt_transformers |

Make **source-linked TTNN a first-class option in the initial implementation**. A normal wheel installed next to a source checkout does not track that checkout's edits. The source-linked mode must use upstream editable-install support and the corresponding native build outputs.

The developer workflow should select a tt-metal checkout/ref, create an isolated environment, perform the initial native build, install TTNN in the chosen mode, and run tt_transformers directly from its own checkout. Developers can then edit both repositories; native changes go through an incremental rebuild rather than rebuilding and reinstalling a wheel after every edit. Export a fixed wheel when a source snapshot needs to be shared or tested in CI.

Keep the released tt_transformers package's `ttnn==0.77.0` requirement and normal release locks intact. The initial environment would install the `ttnn` distribution, editably or from a wheel, but would run tt_transformers from source without installing its distribution. A metadata-consistent editable installation of **tt_transformers itself** can follow if developers require its package entry points or distribution metadata.

The first supported build/test combination should be Linux x86-64, Ubuntu 22.04, CPython 3.10, Release, and CIv2 P150b. Add CPython 3.12 as a separate wheel/build profile after the first path works. Selecting an arbitrary commit means attempting that exact commit and reporting incompatibilities; it does not promise that every historical or future commit supports the requested environment.

## What already exists

| Finding | Evidence | Implication |
| --- | --- | --- |
| TTNN is pinned in package metadata, base/host locks, and release validation tools. | [pyproject.toml](../pyproject.toml), [locks](../constraints/README.md), [lock validator](../tools/validate_lockfiles.py), [installed-package probe](../tools/probe_installed_package.py), [artifact audit](../tools/audit_package_artifacts.py) | Replacing only the wheel is insufficient for a normal package install when its version differs from 0.77.0. |
| tt-metal packages compiled libraries, runtime assets, headers, and kernels into a wheel. | [setup.py at the inspected tt-metal main SHA](https://github.com/tenstorrent/tt-metal/blob/8d3e05c85c9b7b4d10fbfe6420563465ac468768/setup.py) | Reuse upstream packaging; do not distribute a copied Python directory or a lone shared library. |
| tt-metal has a custom editable install, and its editable build hook skips native compilation. CMake installs the Python extension into the source tree. | [editable hook](https://github.com/tenstorrent/tt-metal/blob/8d3e05c85c9b7b4d10fbfe6420563465ac468768/setup.py#L171), [native extension installation](https://github.com/tenstorrent/tt-metal/blob/8d3e05c85c9b7b4d10fbfe6420563465ac468768/ttnn/CMakeLists.txt) | Reuse the editable path, but explicitly build and refresh native outputs; `pip install -e` alone is insufficient. |
| tt-metal can package an existing build with `TT_FROM_PRECOMPILED_DIR`, followed by `auditwheel repair`. | [build-artifact.yaml](https://github.com/tenstorrent/tt-metal/blob/8d3e05c85c9b7b4d10fbfe6420563465ac468768/.github/workflows/build-artifact.yaml#L828) | A matched-container developer wheel can avoid a second native compile. |
| A separate manylinux/cibuildwheel route exists. | [wheels.yaml](https://github.com/tenstorrent/tt-metal/blob/8d3e05c85c9b7b4d10fbfe6420563465ac468768/.github/workflows/wheels.yaml) | Broader wheel portability is a separate build profile. |
| The shared CIv2 P150b pool and Llama checkpoint access already work for this repository. | [Successful Llama run](https://github.com/tenstorrent/tt_transformers/actions/runs/34314639402/job/102348307358), [current workflow](../.github/workflows/llama31-8b-p150b-civ2.yml) | Reuse the hardware selector and LFC downloader for a candidate-runtime test. |
| Model cache identity contains the TTNN package version, but not its wheel hash or complete build identity. | [cache_environment.py](../src/tt_transformers/cache_environment.py) | Development runtimes need additional cache separation. |
| Source imports already tolerate absent tt-transformers distribution metadata. | [package initializer](../src/tt_transformers/__init__.py), [cache version fallback](../src/tt_transformers/cache_environment.py) | Source-checkout development is a practical first path; provenance must record the source revision separately. |

The inspected tt-metal checkout is `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`. Relevant files were also checked against the observed remote `main`, `8d3e05c85c9b7b4d10fbfe6420563465ac468768`. The remote packaging code includes a newer recursive kernel-header inclusion rule. This is another reason to use the selected commit's packaging code and test a wheel outside the source checkout.

The existing hardware result proves TTNN 0.77.0 plus Llama 3.1 8B on P150b. It does not validate a wheel from current tt-metal main.

## 1. Define the runtime selection contract

Introduce a versioned runtime manifest and a small developer tool. The names below are proposed interfaces, not existing commands.

Inputs:

- tt-metal repository, initially fixed to `tenstorrent/tt-metal`;
- remote ref, tag, or full SHA; default `main`;
- an explicitly selected local tt-metal checkout for source-linked development, including local edits;
- alternatively, an existing wheel plus its build manifest;
- installation mode: `source` or `wheel`;
- target Python version, Linux architecture, build profile, and build flags;
- local-container or CI build execution.

Resolve `main` once through the remote repository and record the full SHA. Every checkout, image selection, build, artifact lookup, and test in that operation must use that resolved SHA. Repeating an operation from the saved manifest must not resolve `main` again.

When creating a checkout, use an isolated location with recursive submodules and sufficient Git tags/history for upstream version calculation. When attaching to an existing checkout, preserve its branch and edits; do not reset, clean, or switch it. Builds may write their expected generated outputs into that explicitly selected checkout/build directory.

Local uncommitted changes are supported in source-linked mode from the start. Record the base SHA, tracked patch digest, relevant untracked build inputs, and submodule state. Distinguish the current source fingerprint from the fingerprint used for the last successful native build. Do not describe a dirty checkout as exactly equal to its base commit. Initial shared CI builds should use committed revisions; exporting uncommitted changes requires an explicit reproducible source snapshot rather than a bare SHA claim.

Identify artifacts with at least:

- source repository and full commit SHA;
- submodule revisions;
- Python ABI, CPU architecture, and wheel platform tag;
- build type, Tracy/distributed/LTO flags, and relevant compiler/toolchain versions;
- builder image digest, runtime image digest, and build-recipe revision;
- actual wheel filename, package version, SHA-256, and dependency metadata;
- for source mode, editable origin, checkout identity, patch/content fingerprints, native build directory, and native output fingerprints instead of a wheel hash;
- build run/artifact IDs when built in CI;
- SFPI/JIT toolchain requirements and known driver/firmware expectations.

For wheel mode, use the wheel hash plus build inputs as the runtime fingerprint. For source mode, use the build record, relevant source content, and native outputs. Keep a stable workspace identity for the editable environment and a separate per-build/per-run fingerprint, so every source edit does not require recreating the virtual environment. A Python version string alone is not a reliable build identifier: the inspected upstream version scheme derives development versions from tags and distance, and clean builds do not necessarily include a Git SHA.

Preserve the wheel's actual version. Do not relabel an unrelated main-branch build as `0.77.0` to satisfy the existing dependency pin.

## 2. Build TTNN and support incremental development

### Source-linked mode

Bind the environment to a particular tt-metal checkout, Python ABI, native build directory, and toolchain. Perform the upstream native build and install/update its source-tree Python binding outputs, then use the upstream editable installation in the same environment. Do not install both a fixed TTNN wheel and editable TTNN in one environment.

The inspected upstream editable hook does not compile native code. Its CMake install rules populate `ttnn/ttnn/_ttnn.so`, with native library search paths reaching the matching build tree. The helper must run the required build/install targets, not merely rebuild an arbitrary library in a separate directory and assume Python will find it.

| Developer change | Required action before testing |
| --- | --- |
| tt_transformers Python modules/models | Start a fresh test/example process using the checkout. |
| TTNN Python code | Start a fresh process through the editable installation; normally no wheel reinstall or native build. |
| Host C++, native operators, bindings, or native headers | Incrementally rebuild/install the affected native outputs, then restart Python or the notebook kernel. |
| Device kernels or JIT headers | Let the target revision's JIT dependency tracking recompile affected kernels; invalidate/isolate the relevant JIT cache when needed. Shared native headers may also require a native rebuild. |
| CMake/build options, dependency metadata, Python ABI, or source ref | Reconfigure/rebuild and refresh editable metadata/dependency resolution as required; use a separate compatible environment/build profile when necessary. |

Editable installation does not provide automatic reload of already imported modules or loaded native libraries. Metadata/dependency changes can require reinstalling the editable package. [Setuptools editable-install behavior and limits](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)

Provide a `rebuild` operation that runs the selected revision's incremental build and required binding-install targets. Preflight should detect or conservatively check native build freshness before launching tests, refuse stale compiled inputs, and never overwrite native libraries underneath a running Python process. Reuse the existing CMake/Ninja dependency graph rather than inventing an incomplete list of file extensions that need rebuilding.

With container-based development, mount both checkouts and the native build tree at stable paths and run the editable environment inside that container. Do not assume a container-built extension is compatible with a different host Python or system libraries.

### Fixed-wheel mode and initial build profile

Use the selected commit's upstream build and packaging recipe inside a pinned Ubuntu 22.04 build container. Compile TTNN and its required native/runtime dependencies, then package the same tree through `TT_FROM_PRECOMPILED_DIR` and repair the wheel.

Implementation must establish the supported build arguments from the selected revision; do not assume that arbitrary older commits accept current build flags. Start with a tested recipe for current main and report unsupported revisions clearly.

Build outputs should include the repaired wheel, manifest, build logs, and a small import/runtime-assets inspection report. Reuse a cached artifact only when the complete build key matches and its hash verifies.

The existing in-place CI recipe describes Ubuntu 22.04 / manylinux_2_35 compatibility. Treat this as a matched-environment artifact, not a promise that it will work on every Linux host. For developers needing broader native-host portability, add the upstream manylinux/cibuildwheel route as a separate profile.

### Runtime completeness

A successful `import ttnn` is necessary but insufficient. In source mode, verify that TTNN Python, its extension, native libraries, runtime assets, and JIT inputs all resolve to the selected checkout/build. In wheel mode, verify those assets are supplied by the installed artifact and its declared runtime environment; exercise JIT compilation after removing the source checkout from import/library search paths. Preserve the wheel-only check to catch packaging omissions hidden by editable installs.

The inspected `setup.py` has `BUNDLE_SFPI = False`. Therefore the developer runtime container/host must supply the matching JIT toolchain separately. The wheel also does not install the kernel driver or firmware. Record those requirements with the artifact; do not silently combine a current-main wheel with the old image used to validate 0.77.0.

### CI ownership and access

Prefer a thin workflow in tt_transformers that explicitly checks out `tenstorrent/tt-metal` at the resolved SHA and invokes its build recipe on a CPU runner. Keep local and CI execution backed by the same recipe.

Do not call tt-metal's reusable workflows unchanged and assume they will check out tt-metal. Their checkout steps omit `repository:`, their local actions/scripts depend on tt-metal's layout, and reusable workflows operate in the caller's context. [GitHub workflow context and runner rules](https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations)

The inspected `wheels.yaml` also fails internal builds without `GARAGE_S3_ACCESS_KEY` and `GARAGE_S3_SECRET_KEY`. These secrets are not currently exposed to tt_transformers through its repository/organization secret listings. A copied build recipe must either support a deliberate cold/local-cache build or receive the intended cache access through infra.

Before enabling the build job, establish:

- a CPU runner with adequate resources; `tt-ubuntu-2204-large-stable` is upstream's current selector, but this repository has not tested its access/capacity;
- availability of the selected builder/runtime images and toolchain downloads through CIv2's proxy;
- whether shared compilation caching is available, and the expected cold-build duration if it is not;
- artifact retention and download access for developers.

P150b access is already proven; access to a particular CPU pool and build-cache credentials is a separate question. No runner registration or new hardware fleet is proposed.

An upstream-hosted build service is a viable later alternative if infra prefers owning the build. It would need a clean commit-addressed artifact interface and explicitly authorized cross-repository dispatch/download credentials.

## 3. Create a development environment without a dependency conflict

### Why an extra or constraints file is insufficient

The stable distribution declares `ttnn==0.77.0`. A constraints file narrows dependency selection; it cannot replace an incompatible requirement declared by that distribution. An extra also adds requirements rather than subtracting the base pin. [pip constraints](https://pip.pypa.io/en/stable/user_guide/#constraints-files), [dependency declarations](https://packaging.python.org/en/latest/specifications/pyproject-toml/#dependencies-optional-dependencies)

Installing a candidate wheel and then running a normal `pip install -e .` can replace it or fail resolution. Using `--no-deps` alone avoids resolution but leaves inconsistent installed metadata when the candidate version does not satisfy the pin; `pip check` should then report the conflict. [pip install](https://pip.pypa.io/en/stable/cli/pip_install/), [pip check](https://pip.pypa.io/en/stable/cli/pip_check/)

### Recommended first version: tt_transformers from source, with either TTNN mode

The developer tool should:

1. Create an isolated virtual environment for the selected workspace/build profile or fixed runtime artifact.
2. Read tt_transformers' direct requirements and requested development/example/test extras.
3. Replace only the TTNN selection in this development input with the chosen editable checkout or wheel.
4. Resolve that TTNN source/artifact's declared dependencies together with the other project requirements.
5. Prefer the existing tested non-TTNN pins. If they conflict, stop with a useful dependency report; require an explicit, recorded profile change before upgrading other dependencies.
6. Generate a separate development lock with exact dependency artifact hashes. In wheel mode, include the selected TTNN wheel hash. In source mode, install the hashed dependency closure first and record the editable TTNN source/build identity separately; a mutable editable checkout cannot be represented as an immutable wheel hash.
7. Install TTNN in the selected mode after preparing its dependencies and native build. Do not install the `tt-transformers` distribution in this initial interface.
8. Launch commands with this checkout's `src` on the child process's import path, plus the repository root when examples/tests need it.

This provides immediate feedback from edits without changing published package metadata. `pip check` must pass for installed distributions; the developer tool must additionally validate the source project's effective requirements, because pip does not see an uninstalled source project.

The doctor/preflight command must verify:

- TTNN's distribution version, installation mode, source/artifact provenance, Python ABI, and import origin;
- in wheel mode, `ttnn.__file__` points to the selected environment's installed wheel;
- in source mode, the TTNN Python origin, loaded `_ttnn` extension, native libraries, and runtime assets match the selected checkout/build, with no stale native inputs;
- `tt_transformers.__file__` points to the requested source checkout;
- no previously installed tt-transformers distribution shadows the checkout or supplies misleading metadata;
- runtime/import/library paths select only the intended wheel or source build, with no stale or competing tt-metal installation;
- tt_transformers examples/tests resolve from its own checkout, since upstream TTNN's editable `.pth` also exposes tt-metal's repository root and tools;
- the required JIT toolchain is available;
- the effective dependency graph matches the development manifest.

The launcher should scope environment changes to its child process. It should not rewrite global shell configuration or automatically modify a developer's driver/firmware.

### Follow-up if tt_transformers itself needs an editable distribution

Provide an explicitly selected development metadata path that declares the actual candidate TTNN requirement while default builds emit the stable requirement. This would require a deliberate packaging design, including consistent metadata across wheel and editable build hooks.

Static project dependency entries cannot simply be overwritten by a backend. If dynamic metadata is chosen, declare it through the packaging specification and preserve reproducible baseline output. Do not hand-edit an installed wheel's METADATA. [Static/dynamic metadata rules](https://packaging.python.org/en/latest/specifications/pyproject-toml/#dynamic)

This concerns tt_transformers' metadata and is deferred from its initial source-checkout workflow; editable TTNN is part of the first implementation. Broadly relaxing the public TTNN requirement is also a separate compatibility-policy decision.

## 4. Isolate caches and record both repositories

The developer runtime identity should namespace:

- converted model-weight caches;
- TTNN/Metal JIT caches;
- environment and build artifacts;
- execution evidence.

Use the runtime fingerprint, not only the TTNN version. Two builds can share a version string while differing in source, build flags, or binary contents. Editable package metadata can remain unchanged while the working tree changes. Update the source/build fingerprint after relevant edits and successful rebuilds; refuse to treat stale binaries as a successful new build. Use dependency-aware cache invalidation where established, with conservative cache separation when correctness is uncertain.

Initially the developer launcher can choose namespaced `TT_TRANSFORMERS_CACHE`, `TT_CACHE_PATH`, and `TT_METAL_CACHE` roots, using controls supported by the selected TTNN revision. Check explicit user cache paths instead of silently allowing them to bypass the runtime namespace.

Record tt_transformers' commit and dirty-source status separately from its fallback source version. Record a patch/content digest when attributing results from modified source. A reused runtime wheel does not imply that a changed model implementation has already been validated.

Apply the same dirty-source attribution to tt-metal in editable mode. Before exporting a wheel, freeze the intended source/build state and verify that Python files, native libraries, headers, and runtime assets come from that same state. Simply packaging a tree with stale native outputs would produce a misleading artifact.

Switching back to the stable environment should require no source edits and should leave the candidate environment/cache available for debugging.

## 5. Reuse the CIv2 hardware path

Add an opt-in workflow interface with a tt-metal ref or a verified fixed-runtime manifest. Use source-linked mode for local iteration and a fixed wheel for the initial shared CI path. Committing tt-metal edits and building that exact revision creates the reproducible bridge between them. Start with manual dispatch rather than building tt-metal for every tt_transformers PR.

Proposed flow:

`resolve ref → CPU build/cache lookup → resolve/install development dependencies → host checks → CIv2 hardware checks → upload evidence`

Use:

- separate TTNN wheels for each Python ABI;
- the matched runtime image from the build manifest;
- `tt-ubuntu-2204-P150b-viommu-stable` for the first hardware profile;
- the proven device options, `--device /dev/tenstorrent -e TT_GH_CI_INFRA`;
- the existing pinned Llama checkpoint downloader and CIv2 LFC source;
- the source-checkout environment and the same doctor checks used locally.

The current CPU preparation step downloads the baseline TTNN wheel from the host lock. The candidate path must replace that step's TTNN input and hash rather than accidentally installing both runtimes.

Validate in increasing scope:

1. Installed-wheel import and provenance, with the tt-metal source checkout unavailable.
2. A small actual TTNN operation that opens/closes the P150b and exercises JIT compilation.
3. Relevant reusable-module/runtime tests.
4. The existing Llama 3.1 8B token-accuracy example.

Require non-skipped evidence with the two repository SHAs, wheel hash, dependency lock, physical hardware identity, driver/firmware/toolchain, command, metrics, and teardown result. Separate build/dependency failures from API incompatibilities and device/model failures.

Keep the stable release CI path independently testable. Do not silently substitute main when a requested SHA fails. Do not schedule concurrent TT processes on one physical host or add automatic resets. Existing branches/runs must not be deleted or cancelled as part of this work without the user's instruction.

## Proposed developer experience

These interfaces are illustrative and do not exist yet:

```bash
# Attach an existing tt-metal checkout for development across both repositories.
python tools/ttnn_dev.py attach --tt-metal-checkout /path/to/tt-metal \
  --python 3.10 --executor local-container

# After a native change, refresh the existing build and Python binding outputs.
python tools/ttnn_dev.py rebuild --runtime /path/to/source-runtime-manifest.json

# Run in a fresh process using editable TTNN and this tt_transformers checkout.
python tools/ttnn_dev.py run --runtime /path/to/source-runtime-manifest.json -- \
  python -m pytest -m host

# Resolve current remote main and build a reusable TTNN artifact.
python tools/ttnn_dev.py build --tt-metal-ref main --python 3.10 --executor ci

# Create an isolated developer environment from that exact artifact.
python tools/ttnn_dev.py env --runtime /path/to/runtime-manifest.json

# Show the exact wheel, source checkout, dependency set, and JIT toolchain.
python tools/ttnn_dev.py doctor --runtime /path/to/runtime-manifest.json

# Run edited modules/tests from this checkout using the installed TTNN wheel.
python tools/ttnn_dev.py run --runtime /path/to/runtime-manifest.json -- \
  python -m pytest -m host

# On an allocated hardware host/container, with checkpoint access configured:
python tools/ttnn_dev.py run --runtime /path/to/runtime-manifest.json -- \
  python -m examples.llama3_8b.demo --case token-accuracy --optimizations performance
```

A local-container build should produce the same manifest format. A second developer should be able to use a downloaded artifact without rebuilding TTNN. Local development commands do not themselves acquire a CI hardware runner.

## Implementation sequence

| Phase | Deliverable | Acceptance |
| --- | --- | --- |
| 1. Runtime contract | Source/wheel modes, ref resolution, source/build fingerprints and doctor design | A moving ref resolves once; mode, dirty state, ABI, and stale-build errors are explicit. |
| 2. Incremental native build | One matched Ubuntu 22.04 / CPython 3.10 recipe with the required binding-install targets | A native change is rebuilt and observed by a fresh Python process; stale outputs are rejected. |
| 3. Source development environment | Editable TTNN, tt_transformers source launcher, dependency lock, cache namespace | Python edits in either repo are visible in fresh runs; native edits are visible after rebuilding; dependency checks and stable release validation pass. |
| 4. Wheel export and hardware CI | Snapshot packaging and opt-in candidate runtime in CIv2 P150b | The fixed wheel works without its source checkout; a TTNN operation, module checks, and Llama token accuracy pass with complete provenance. |
| 5. Documentation and reuse | Local setup, CI build/download, IDE/test execution, switch-back instructions, troubleshooting | A second developer reproduces the environment from the manifest and wheel. |
| 6. Additional profiles | CPython 3.12, then manylinux portability or tt_transformers editable metadata if needed | Each profile has its own build identity and validation; no inferred cross-ABI support. |

Likely implementation touch points:

- new `tools/ttnn_dev.py` and a small runtime-manifest schema;
- a shared native build/rebuild recipe, editable TTNN setup, and a CPU-only wheel build workflow;
- development dependency/validation helpers alongside the existing lock tools;
- `.github/workflows/llama31-8b-p150b-civ2.yml` for explicit candidate selection;
- `CONTRIBUTING.md`, `constraints/README.md`, and developer setup documentation;
- cache/preflight code only if environment-level namespacing cannot enforce the required identity.

No global dependency relaxation, backend migration, automatic nightly tracking, private package index, or firmware rollout is required for the first implementation.

## Completion criteria

The work is complete when a developer can select current tt-metal main, a full SHA, or a local checkout; edit Python code in either repository; incrementally rebuild native tt-metal changes; and observe the changes from a fresh tt_transformers test/example process. A committed source snapshot must also produce an attributable fixed wheel and a reproducible CIv2 P150b model result. The same tt_transformers checkout must still support the normal release-pinned installation and checks.

The original implementation prerequisites were agreement on the two-mode interface and confirmation of the CPU build environment/cache arrangement. This design recommends editable TTNN for development spanning both repositories and fixed wheels for reproducible CI. Current usage is documented in the development guide; validation evidence is recorded in runtime manifests and CI runs rather than inferred from this planning document.
