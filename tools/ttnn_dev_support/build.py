"""One upstream native build/install recipe for editable and fixed-wheel runtimes."""

from __future__ import annotations

import os
import re
import tarfile
import tempfile
from pathlib import Path

from .common import (
    REPOSITORY,
    DevError,
    artifact_entry,
    clean_environment,
    command,
    digest,
    file_hash,
    git,
    host_tools,
    interpreter_info,
    load_runtime,
    read_json,
    runtime_lock,
    runtime_python,
    save_runtime,
    source_state,
    tool_fingerprint,
    write_json,
)
from .environment import (
    create_venv,
    dependency_files,
    install_dependencies,
    prepare_dependencies,
    requirements,
    wheel_metadata,
)

RECIPE = "tt-metal-native-v1"


def checkout_lock(checkout: Path) -> Path:
    path = Path(git(checkout, "rev-parse", "--git-path", "ttnn-dev.lock"))
    return path if path.is_absolute() else checkout / path


def resolve_ref(ref: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,254}", ref):
        raise DevError("Use a Git branch, tag, or full commit SHA.")
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return ref.lower()
    matches = command(
        [
            "git",
            "ls-remote",
            REPOSITORY,
            ref,
            f"{ref}^{{}}",
            f"refs/heads/{ref}",
            f"refs/tags/{ref}",
            f"refs/tags/{ref}^{{}}",
        ],
        capture=True,
    )
    rows = dict(line.split("\t", 1)[::-1] for line in matches.splitlines())
    # Prefer a peeled annotated tag to its tag-object SHA, but reject ambiguous names.
    values = {sha for name, sha in rows.items() if name + "^{}" not in rows}
    if len(values) != 1:
        raise DevError(f"Ref {ref!r} is missing or ambiguous; use a full ref or SHA.")
    return values.pop()


def clone_commit(sha: str, checkout: Path) -> None:
    if checkout.exists():
        if git(checkout, "rev-parse", "HEAD") != sha or source_state(checkout)["dirty"]:
            raise DevError("Build checkout does not match the requested clean commit; choose another output directory.")
        return
    command(["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, checkout])
    command(["git", "-C", checkout, "fetch", "--tags", "origin", sha])
    command(["git", "-C", checkout, "checkout", "--detach", sha])
    command(
        [
            "git",
            "-C",
            checkout,
            "-c",
            "url.https://github.com/.insteadOf=git@github.com:",
            "submodule",
            "update",
            "--init",
            "--recursive",
        ]
    )
    if git(checkout, "rev-parse", "HEAD") != sha:
        raise DevError("Resolved commit changed during checkout.")


def native_outputs(checkout: Path, build_dir: Path) -> dict:
    extensions = list((checkout / "ttnn/ttnn").glob("_ttnn*.so"))
    libraries = [path for path in (build_dir / "lib").glob("*.so*") if path.is_file()]
    if not extensions or not libraries:
        raise DevError("Build did not install source Python bindings and native libraries.")
    configuration = [path for path in (build_dir / "CMakeCache.txt", build_dir / "build.ninja") if path.is_file()]
    return {str(path.resolve()): file_hash(path) for path in sorted(set(extensions + libraries + configuration))}


def native_inputs(checkout: Path, build_dir: Path) -> dict:
    """Read actual compiler dependencies and CMake's configured input graph."""
    paths = set()
    dependencies = command(["ninja", "-C", build_dir, "-t", "deps"], capture=True)
    for line in dependencies.splitlines():
        if line.startswith("    "):
            path = Path(line[4:])
            paths.add(path if path.is_absolute() else build_dir / path)
    reply_dir = build_dir / ".cmake/api/v1/reply"
    indexes = sorted(reply_dir.glob("index-*.json"))
    if not indexes:
        raise DevError("CMake file API did not describe the configured input graph.")
    reference = read_json(indexes[-1])["reply"]["cmakeFiles-v1"]["jsonFile"]
    for entry in read_json(reply_dir / reference)["inputs"]:
        path = Path(entry["path"])
        paths.add(path if path.is_absolute() else checkout / path)
    records = {}
    for path in sorted(paths):
        if path.is_file():
            stat = path.stat()
            records[str(path.resolve())] = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
                "sha256": file_hash(path),
            }
    if not records:
        raise DevError("Native build did not expose any compiler/CMake dependencies.")
    return records


def sfpi_info(checkout: Path) -> dict:
    root = checkout / "runtime/sfpi"
    compiler = root / "compiler/bin/riscv-tt-elf-g++"
    if not compiler.is_file():
        raise DevError("This recipe requires the selected source revision's local runtime/sfpi toolchain.")
    return {
        "compiler_sha256": file_hash(compiler),
        "version": command([compiler, "--version"], capture=True).splitlines()[0],
    }


def native_build(settings: dict, venv: Path) -> None:
    checkout = Path(settings["checkout"])
    build_dir = Path(settings["build_dir"])
    for path in (checkout, build_dir):
        if re.search(r"\s|[?*\[\]]", str(path)):
            raise DevError("The upstream build script requires checkout/build paths without spaces or glob characters.")
    link = checkout / "build"
    if link.exists() or link.is_symlink():
        if link.resolve() != build_dir.resolve():
            raise DevError(
                "The checkout already selects another build directory. Use a separate worktree or explicitly "
                "attach with --build-dir pointing to its current build directory."
            )
    env = clean_environment(venv)
    query = build_dir / ".cmake/api/v1/query/cmakeFiles-v1"
    query.parent.mkdir(parents=True, exist_ok=True)
    query.touch()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(settings["jobs"])
    env["CPM_SOURCE_CACHE"] = str(Path(settings["work"]) / "cpm-cache")
    # Compile only runtime/bindings; upstream's install target also refreshes _ttnn.so.
    argv = [
        "bash",
        "build_metal.sh",
        "--build-dir",
        build_dir,
        "--install-prefix",
        build_dir,
        "--build-type",
        settings["build_type"],
        "--disable-profiler",
    ]
    if not settings["distributed"]:
        argv.append("--without-distributed")
    before = source_state(checkout)
    command(argv, cwd=checkout, env=env)
    if source_state(checkout)["fingerprint"] != before["fingerprint"]:
        raise DevError("tt-metal source changed during compilation. Rebuild a consistent source state.")


def attach(
    checkout: Path,
    project: Path,
    work: Path,
    *,
    python: str,
    build_dir: Path | None,
    build_type: str,
    jobs: int,
    distributed: bool,
    extras: list[str],
    image: str | None,
) -> Path:
    checkout, project, work = checkout.resolve(), project.resolve(), work.resolve()
    if Path(git(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout:
        raise DevError("--tt-metal-checkout must be a repository root.")
    if not (checkout / "build_metal.sh").is_file() or not (checkout / "ttnn/ttnn").is_dir():
        raise DevError("Selected checkout is not a supported tt-metal tree.")
    info = interpreter_info(python)
    settings = {
        "checkout": str(checkout),
        "project": str(project),
        "work": str(work),
        "build_dir": str((build_dir or work / "native").resolve()),
        "build_type": build_type,
        "jobs": jobs,
        "distributed": distributed,
        "extras": extras,
        "python": info,
        "image": image,
        "recipe": RECIPE,
    }
    workspace = work / "workspace.json"
    if workspace.exists():
        previous = read_json(workspace)
        if any(
            previous[key] != settings[key]
            for key in ("checkout", "project", "build_dir", "build_type", "distributed", "extras")
        ):
            raise DevError("Workspace belongs to a different source/build profile. Choose another --output.")
        if previous["python"]["abi"] != info["abi"]:
            raise DevError("Workspace Python ABI changed. Choose a new environment/build profile.")
    elif work.exists() and any(work.iterdir()):
        raise DevError("Output directory is not an existing runtime workspace or an empty directory.")
    elif not work.exists():
        work.mkdir(parents=True)
    write_json(workspace, settings)
    return rebuild(work / "runtime.json")


def rebuild(manifest: Path) -> Path:
    work = manifest.parent.resolve()
    settings = read_json(work / "workspace.json")
    checkout = Path(settings["checkout"])
    project = Path(settings["project"])
    with runtime_lock(checkout_lock(checkout), exclusive=True):
        tools = host_tools()
        venv = work / "venv"
        if not venv.exists():
            create_venv(settings["python"]["executable"], venv)
        python = runtime_python(venv)
        metadata_identity = digest(
            {
                "pyproject": file_hash(checkout / "pyproject.toml"),
                "setup": file_hash(checkout / "setup.py"),
                "project": requirements(project, settings["extras"]),
                "baseline": file_hash(
                    project / f"constraints/locks/host-py{settings['python']['minor'].replace('.', '')}.txt"
                ),
                "sha": git(checkout, "rev-parse", "HEAD"),
            }
        )
        dependencies = work / "dependencies"
        prepared = work / "prepared.json"
        if not prepared.exists() or read_json(prepared).get("identity") != metadata_identity:
            if dependencies.exists():
                # Keep failed/previous attempts for debugging instead of deleting their output.
                number = 1
                while (work / f"dependencies.previous-{number}").exists():
                    number += 1
                dependencies.rename(work / f"dependencies.previous-{number}")
            selected = prepare_dependencies(
                python, dependencies, project, checkout, editable=True, extras=settings["extras"]
            )
            write_json(prepared, {"identity": metadata_identity, "selected": selected})
        selected = read_json(prepared)["selected"]
        install_dependencies(python, dependencies)
        native_build(settings, venv)
        command([python, "-m", "pip", "install", "--no-deps", "--editable", checkout], env=clean_environment(venv))
        version = command(
            [python, "-I", "-c", "from importlib.metadata import version; print(version('ttnn'))"],
            capture=True,
            env=clean_environment(venv),
        )
        # Editable SCM metadata can change when a checkout becomes dirty; native provenance is separate.
        state = source_state(checkout)
        data = {
            "mode": "source",
            "metal": state,
            "python": settings["python"],
            "recipe": {
                "name": RECIPE,
                "tool_fingerprint": tool_fingerprint(),
                "build_type": settings["build_type"],
                "distributed": settings["distributed"],
                "tracy": False,
                "lto": False,
                "image": settings["image"],
                "builder_image": os.environ.get("TTNN_DEV_IMAGE"),
                "tools": tools,
            },
            "ttnn": {"version": version},
            "toolchain": sfpi_info(checkout),
            "dependencies": {
                "directory": "dependencies",
                "baseline_sha256": file_hash(
                    project / f"constraints/locks/host-py{settings['python']['minor'].replace('.', '')}.txt"
                ),
                "extras": settings["extras"],
                "requirements": selected["requirements"],
            },
            "source": {
                "checkout": str(checkout),
                "build_dir": settings["build_dir"],
                "native_fingerprint": state["native_fingerprint"],
                "outputs": native_outputs(checkout, Path(settings["build_dir"])),
                "inputs": native_inputs(checkout, Path(settings["build_dir"])),
            },
            "files": dependency_files(work, dependencies),
        }
        save_runtime(manifest, data)
        write_json(
            work / "environment.json",
            {
                "schema_version": 1,
                "runtime": str(manifest),
                "venv": str(venv),
                "project": str(project),
                "runtime_root": str(checkout),
            },
        )
    return work / "environment.json"


def export_wheel(source_manifest: Path, output: Path) -> Path:
    source = load_runtime(source_manifest)
    if source["mode"] != "source":
        raise DevError("Wheel export requires a source runtime.")
    checkout = Path(source["source"]["checkout"])
    with runtime_lock(checkout_lock(checkout), exclusive=True):
        state = source_state(checkout)
        if state["dirty"]:
            raise DevError("Commit local tt-metal changes before exporting a shared wheel.")
        if state["native_fingerprint"] != source["source"]["native_fingerprint"]:
            raise DevError("Source changed after the last build; rebuild before export.")
        if native_outputs(checkout, Path(source["source"]["build_dir"])) != source["source"]["outputs"]:
            raise DevError("Native outputs changed after the build; rebuild before export.")
        if output.exists() and any(output.iterdir()):
            raise DevError("Choose an empty output directory for the fixed wheel bundle.")
        output.mkdir(parents=True, exist_ok=True)
        bootstrap = source_manifest.parent / "wheel-tools"
        if not bootstrap.exists():
            create_venv(source["python"]["executable"], bootstrap)
        command(
            [
                bootstrap / "bin/python",
                "-m",
                "pip",
                "install",
                "build==1.3.0",
                "auditwheel==6.6.0",
                "patchelf==0.17.2.4",
            ]
        )
        env = clean_environment(bootstrap)
        env["TT_FROM_PRECOMPILED_DIR"] = str(checkout)
        raw = Path(tempfile.mkdtemp(prefix="wheel-raw-", dir=source_manifest.parent))
        command([bootstrap / "bin/python", "-m", "build", "--wheel", "--outdir", raw], cwd=checkout, env=env)
        wheels = list(raw.glob("*.whl"))
        if len(wheels) != 1:
            raise DevError("Expected one raw wheel.")
        env["LD_LIBRARY_PATH"] = str(Path(source["source"]["build_dir"]) / "lib")
        command([bootstrap / "bin/auditwheel", "repair", "--wheel-dir", output, wheels[0]], env=env)
        repaired = list(output.glob("*.whl"))
        if len(repaired) != 1:
            raise DevError("Expected one repaired wheel.")
        wheel = repaired[0]
        metadata = wheel_metadata(wheel)
        dependencies = output / "dependencies"
        settings = read_json(source_manifest.parent / "workspace.json")
        selected = prepare_dependencies(
            runtime_python(source_manifest.parent / "venv"),
            dependencies,
            Path(settings["project"]),
            wheel,
            editable=False,
            extras=settings["extras"],
        )
        toolchain_archive = output / "sfpi.tar.gz"
        with tarfile.open(toolchain_archive, "w:gz", dereference=True) as tar:
            for entry in sorted((checkout / "runtime/sfpi").iterdir()):
                tar.add(entry, arcname=entry.name)
        if source_state(checkout)["fingerprint"] != state["fingerprint"]:
            raise DevError("Source changed during packaging; this output must not be used.")
        data = {
            "mode": "wheel",
            "metal": state,
            "python": source["python"],
            "recipe": source["recipe"],
            "ttnn": {**metadata, "wheel": wheel.name, "sha256": file_hash(wheel)},
            "toolchain": {**source["toolchain"], "archive": toolchain_archive.name},
            "dependencies": {
                "directory": "dependencies",
                "baseline_sha256": source["dependencies"].get("baseline_sha256"),
                "extras": settings["extras"],
                "requirements": selected["requirements"],
            },
            "files": [
                artifact_entry(output, wheel),
                artifact_entry(output, toolchain_archive),
                *dependency_files(output, dependencies),
            ],
        }
        manifest = output / "runtime.json"
        save_runtime(manifest, data)
        return manifest
