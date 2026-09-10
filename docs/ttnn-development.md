# Develop with a custom TTNN runtime

`tools/ttnn_dev.py` runs tt_transformers source against editable TTNN or a
fixed wheel bundle. The released package and normal locks retain
`ttnn==0.77.0`. The development environment installs TTNN and the resolved
dependencies, then imports tt_transformers from its checkout without installing
its distribution.

## Prerequisites

Use Linux x86-64 and CPython 3.10 or 3.12. The initial shared CI profile uses
3.10. The launcher needs `packaging`, plus `tomli` on Python 3.10.

Native builds require the selected tt-metal revision's build dependencies.
The initial recipe uses CMake, Ninja, Clang 20, and the upstream default
x86-64-v3 CPU target. Use a prepared tt-metal development container or a
configured native host. The tool installs Python/build utilities only in its
own environments; it does not install system build dependencies or drivers.

```bash
python3 -m venv .artifacts/ttnn-launcher
.artifacts/ttnn-launcher/bin/python -m pip install packaging tomli
TTNN_DEV_PYTHON="$PWD/.artifacts/ttnn-launcher/bin/python"
```

## Edit both repositories

Prefer a dedicated tt-metal worktree with recursive submodules initialized and
release tags available. Upstream installs native bindings into its source tree,
so different Python ABIs or build profiles should use different worktrees.

From a prepared native/container shell:

```bash
"$TTNN_DEV_PYTHON" tools/ttnn_dev.py attach \
  --tt-metal-checkout /path/to/tt-metal \
  --output "$PWD/.artifacts/ttnn-source" --python 3.10 --jobs 16

"$TTNN_DEV_PYTHON" tools/ttnn_dev.py doctor \
  --runtime "$PWD/.artifacts/ttnn-source/runtime.json"

"$TTNN_DEV_PYTHON" tools/ttnn_dev.py run \
  --runtime "$PWD/.artifacts/ttnn-source/runtime.json" -- \
  python -m pytest -q -m host tests/host/test_package_surface.py
```

`attach` resolves dependencies, performs the native build/install, and uses
upstream's editable TTNN backend. It writes an environment, `runtime.json`,
`environment.json`, and build/dependency records under the selected output.

If the checkout already selects another `build` directory, the tool refuses to
replace it. Use another worktree or explicitly adopt its current directory with
`--build-dir /absolute/path/to/build`. It will be configured with the requested
options. The default profile is Release without Tracy or multihost/OpenMPI.
Use `--distributed` for multihost support.

After editing native tt-metal code:

```bash
"$TTNN_DEV_PYTHON" tools/ttnn_dev.py rebuild \
  --runtime "$PWD/.artifacts/ttnn-source/runtime.json"

"$TTNN_DEV_PYTHON" tools/ttnn_dev.py run \
  --runtime "$PWD/.artifacts/ttnn-source/runtime.json" -- \
  python -m pytest -q -m host tests/host/test_package_surface.py
```

Existing TTNN Python-file edits are visible in a fresh process without building
another wheel. Native changes require `rebuild`, which refreshes the Python
binding outputs as well as the native libraries. Restart notebook kernels after
native changes. Metadata/dependency changes are resolved during rebuild.

Preflight checks source state, compiler/CMake dependency records, native outputs,
dependency metadata, and import origins. Builds hold an exclusive checkout lock;
launched commands hold shared locks. This prevents the tool from replacing
libraries while its commands use them. Independently launched Python processes
must also be stopped before rebuilding.

The tool never resets or cleans a supplied checkout. Local edits are supported
and recorded. Failed setup/build outputs remain for debugging. Repeat the same
`attach` arguments to retry an incomplete workspace; a partially created
environment or export can require a new output directory.

## Docker or Podman

Add `--executor local-container --image REGISTRY/IMAGE@sha256:DIGEST` to
`attach` or `build`. The launcher preserves absolute mount paths and records
the image. Subsequent `rebuild`, `doctor`, `run`, `env`, and `export`
commands also accept `--executor local-container`.

Add `--device` to a container `run` to expose `/dev/tenstorrent`. Use the
same prepared image and mount paths throughout a source workspace. Running a
container-built extension with unrelated host Python/system libraries is not
supported. The upstream build script requires checkout/build paths without
whitespace or glob characters.

## Fixed wheels

```bash
"$TTNN_DEV_PYTHON" tools/ttnn_dev.py build --tt-metal-ref main \
  --output "$PWD/.artifacts/ttnn-main" --python 3.10 --jobs 16

"$TTNN_DEV_PYTHON" tools/ttnn_dev.py env \
  --runtime "$PWD/.artifacts/ttnn-main/bundle/runtime.json" --python 3.10

"$TTNN_DEV_PYTHON" tools/ttnn_dev.py run \
  --runtime "$PWD/.artifacts/ttnn-main/bundle/runtime.json" -- \
  python -m pytest -q -m host tests/host/test_package_surface.py
```

`main` is resolved once to an immutable remote SHA. Full SHAs and tags also
work. The build uses upstream's native recipe, packages the compiled tree with
`TT_FROM_PRECOMPILED_DIR`, and runs `auditwheel repair`. Verified outputs
are reused only when their build/dependency inputs match.

To export an attached runtime, commit tt-metal edits, rebuild, then run:

```bash
"$TTNN_DEV_PYTHON" tools/ttnn_dev.py export \
  --runtime "$PWD/.artifacts/ttnn-source/runtime.json" \
  --output "$PWD/.artifacts/ttnn-bundle"
```

Shared wheel export requires a clean committed tt-metal tree. Copy the entire
bundle to another developer or CI job: it contains the wheel, hashed dependency
wheels/lock, manifest, and matching SFPI archive. `env` verifies and installs
the bundle without package-index access, then creates a runtime-assets overlay
combining the installed wheel with its SFPI toolchain. It does not rewrite wheel
metadata or require the tt-metal checkout.

Ubuntu 22.04-built artifacts require a compatible Linux/glibc/Python ABI. Other
platforms need their own build profile. Use `env --output /path/to/environment`
for a different environment location, then pass
`--environment /path/to/environment/environment.json` to `doctor` and `run`.

## Dependency behavior

The tool replaces only the TTNN selection in the development dependency input.
It preserves the other direct project requirements and uses the corresponding
host lock as constraints. Incompatible candidate dependencies fail resolution
rather than silently updating the rest of the stack.

Do not run `pip install -e .` for tt_transformers inside this environment:
that requests the release TTNN pin. `doctor` checks installed distributions
with `python -I -m pip check`, excluding source egg-info and caller
`PYTHONPATH`, and separately validates the source project's requirements.
It rejects an installed tt-transformers distribution that could shadow the source.
It also compares installed dependency versions against the development lock, so
changing an otherwise compatible package with pip cannot silently alter the runtime.

For IDE code analysis, select the environment's `venv/bin/python` and add this
checkout's `src` directory to the IDE's Python analysis paths. Launch tests and
examples through `ttnn_dev.py run` to retain the runtime checks and cache setup.

Editable TTNN is recorded separately from hashed dependency wheels; a mutable
checkout cannot have an immutable wheel hash. Source and fixed-wheel installs
must not coexist in one environment.

## CI and hardware

The `TTNN development runtime` workflow builds on
`tt-ubuntu-2204-large-stable`, verifies a fresh wheel environment, and uploads
the `ttnn-runtime` artifact. It does not require tt-metal's Garage credentials.

After the workflow is on the default branch:

```bash
"$TTNN_DEV_PYTHON" tools/ttnn_dev.py build --executor ci --tt-metal-ref main \
  --output "$PWD/.artifacts/ttnn-ci" --run-hardware
```

This uses authenticated `gh`, waits for completion, and downloads the bundle.
The initial CI profile is Python 3.10, Release, single-host support, and the
`examples,test` extras. Use native/container execution for other profiles.
Before merge, maintainers can choose a registered workflow branch through
`--workflow-ref`.

Optional hardware validation runs on a separate CIv2 P150b VM, without a
tt-metal checkout. It executes a JIT tensor operation, an exact RMSNorm test
with skipped-result rejection, and the Llama 3.1 8B token-accuracy example.
The checkpoint comes from CIv2 LFC and is verified against its pinned revision.
Set `MESH_DEVICE` and configure checkpoint access for equivalent local hardware
commands. The launcher does not acquire, reset, or update hardware.

A buildable TTNN commit is not automatically compatible with every module/model.
Failures remain visible; the tool does not substitute another wheel or commit.
Runner capacity, network allowlisting, and image access still apply.

## Provenance and caches

The manifest schema is [runtime.schema.json](../tools/ttnn_dev_support/runtime.schema.json).
Source manifests identify the local checkout/build and recorded inputs/outputs.
Fixed bundles use relative paths and SHA-256 checksums. Actual TTNN versions are
preserved; Git SHA, local edits, build flags, compiler/image information, and
artifact hashes supply additional identity.

`run` records both repositories' state and the command exit status.
Converted-weight and JIT caches are namespaced through
`TT_TRANSFORMERS_CACHE`, `TT_CACHE_PATH`, and `TT_METAL_CACHE`.
Inherited runtime/library paths are replaced in the child environment.
Original checkpoint caches such as `HF_HOME` can remain shared.

Use your normal environment to return to the release runtime. No release pins
or locks are changed by the tool. Keep build/execution directories out of Git;
the tool does not automatically delete branches, cancel runs, publish packages,
or change driver/firmware installations.
