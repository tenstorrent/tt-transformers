"""Resolve candidate dependencies, create environments, and verify import origins."""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .common import (
    DevError,
    artifact_entry,
    clean_environment,
    command,
    digest,
    file_hash,
    interpreter_info,
    load_environment,
    load_runtime,
    read_json,
    runtime_python,
    source_state,
    toml,
    tree_hash,
    within,
    write_json,
)

INDEX = "https://pypi.org/simple"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
WHEEL_HOSTS = {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}
PROBE = Path(__file__).with_name("probe.py")


def requirements(project: Path, extras: list[str]) -> list[str]:
    metadata = toml(project / "pyproject.toml")["project"]
    values = list(metadata["dependencies"])
    for extra in extras:
        if extra not in metadata.get("optional-dependencies", {}):
            raise DevError(f"Unknown project extra: {extra}")
        values.extend(metadata["optional-dependencies"][extra])
    return sorted({value for value in values if canonicalize_name(Requirement(value).name) != "ttnn"})


def create_venv(python: str | Path, venv: Path) -> dict:
    if venv.exists():
        raise DevError(f"Refusing to reuse an unverified environment: {venv}")
    info = interpreter_info(python)
    command([info["executable"], "-m", "venv", venv], env=clean_environment())
    command([venv / "bin/python", "-m", "pip", "install", "pip==26.2.1"], env=clean_environment(venv))
    return info


def wheel_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise DevError("Expected exactly one wheel METADATA file.")
        metadata = BytesParser().parsebytes(archive.read(names[0]))
    if canonicalize_name(metadata["Name"]) != "ttnn":
        raise DevError(f"Expected a TTNN wheel, got {metadata['Name']}")
    return {"version": metadata["Version"], "requires": metadata.get_all("Requires-Dist", [])}


def prepare_dependencies(
    python: Path,
    root: Path,
    project: Path,
    candidate: Path,
    *,
    editable: bool,
    extras: list[str],
) -> dict:
    """Resolve with existing pins; lock downloaded bytes instead of assuming pip reports contain hashes."""
    info = interpreter_info(python)
    root.mkdir(parents=True, exist_ok=False)
    requested = requirements(project, extras)
    tag = info["minor"].replace(".", "")
    baseline = project / f"constraints/locks/host-py{tag}.txt"
    constraints = []
    for line in baseline.read_text().splitlines():
        if line and not line.startswith(("#", "--", " ")):
            requirement = line.rstrip(" \\")
            if canonicalize_name(Requirement(requirement).name) != "ttnn":
                constraints.append(requirement)
    (root / "constraints.txt").write_text("\n".join(constraints) + "\n")
    report_path = root / "resolution.json"
    candidate_args = ["--editable", str(candidate)] if editable else [str(candidate)]
    command(
        [
            python,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--report",
            report_path,
            "--index-url",
            INDEX,
            "--extra-index-url",
            CPU_INDEX,
            "--only-binary=:all:",
            "--constraint",
            root / "constraints.txt",
            *requested,
            *candidate_args,
        ],
        env=clean_environment(python.parent.parent),
    )
    report = read_json(report_path)
    wheelhouse = root / "wheels"
    wheelhouse.mkdir()
    installed = {}
    urls = []
    ttnn_version = None
    for item in report["install"]:
        name = canonicalize_name(item["metadata"]["name"])
        version = item["metadata"]["version"]
        if name == "ttnn":
            ttnn_version = version
            if editable:
                continue
        url = item["download_info"]["url"]
        parsed = urlparse(url)
        if name != "ttnn" and (
            parsed.scheme != "https" or parsed.hostname not in WHEEL_HOSTS or parsed.username or parsed.password
        ):
            raise DevError(f"Expected a public HTTPS dependency wheel for {name}")
        if not unquote(parsed.path).endswith(".whl"):
            raise DevError(f"Dependency must be a wheel: {name}")
        urls.append(url)
        installed[name] = version
    if ttnn_version is None:
        raise DevError("Resolver did not select TTNN.")
    command(
        [python, "-m", "pip", "download", "--no-deps", "--dest", wheelhouse, *urls],
        env=clean_environment(python.parent.parent),
    )
    rows = []
    for wheel in sorted(wheelhouse.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(name))
        package = canonicalize_name(metadata["Name"])
        if installed.pop(package, None) != metadata["Version"]:
            raise DevError(f"Downloaded wheel differs from resolution: {wheel.name}")
        rows.append(f"{package}=={metadata['Version']} --hash=sha256:{file_hash(wheel)}")
    if installed:
        raise DevError(f"Missing dependency wheels: {sorted(installed)}")
    (root / "requirements.txt").write_text("\n".join(sorted(rows)) + "\n")
    return {"version": ttnn_version, "requirements": requested, "extras": extras}


def install_dependencies(python: Path, dependencies: Path) -> None:
    command(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--require-hashes",
            "--find-links",
            dependencies / "wheels",
            "-r",
            dependencies / "requirements.txt",
        ],
        env=clean_environment(python.parent.parent),
    )


def dependency_files(root: Path, dependencies: Path) -> list[dict]:
    return [artifact_entry(root, path) for path in sorted(dependencies.rglob("*")) if path.is_file()]


def extract_toolchain(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            within(destination, member.name)
            if not (member.isfile() or member.isdir()):
                raise DevError(f"Unexpected toolchain archive member: {member.name}")
        tar.extractall(destination, filter="data")


def prepare_overlay(python: Path, state: Path, data: dict, bundle: Path) -> Path:
    """Supply the exact SFPI build without modifying the installed wheel's files."""
    package = Path(
        command(
            [
                python,
                "-I",
                "-c",
                "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('ttnn').origin).parent)",
            ],
            capture=True,
            env=clean_environment(python.parent.parent),
        )
    )
    sfpi = state / "sfpi"
    extract_toolchain(within(bundle, data["toolchain"]["archive"]), sfpi)
    overlay = state / "runtime-root"
    overlay.mkdir()
    for path in package.iterdir():
        if path.name not in {"runtime", "__pycache__"}:
            (overlay / path.name).symlink_to(path, target_is_directory=path.is_dir())
    (overlay / "runtime").mkdir()
    for path in (package / "runtime").iterdir():
        if path.name != "sfpi":
            (overlay / "runtime" / path.name).symlink_to(path, target_is_directory=path.is_dir())
    (overlay / "runtime/sfpi").symlink_to(sfpi, target_is_directory=True)
    return overlay


def create_environment(manifest: Path, project: Path, state: Path, python: str) -> Path:
    data = load_runtime(manifest)
    if data["mode"] != "wheel":
        raise DevError("Source runtimes already own their environment; use attach/rebuild.")
    requested = interpreter_info(python)
    if requested["abi"] != data["python"]["abi"]:
        raise DevError(f"Python ABI mismatch: {requested['abi']} versus {data['python']['abi']}")
    if state.exists() and any(state.iterdir()):
        raise DevError(f"Choose a new environment directory: {state}")
    state.mkdir(parents=True, exist_ok=True)
    venv = state / "venv"
    info = create_venv(python, venv)
    if info["abi"] != data["python"]["abi"]:
        raise DevError(f"Python ABI mismatch: {info['abi']} versus {data['python']['abi']}")
    target = runtime_python(venv)
    install_dependencies(target, within(manifest.parent, data["dependencies"]["directory"]))
    overlay = prepare_overlay(target, state, data, manifest.parent)
    config = state / "environment.json"
    write_json(
        config,
        {
            "schema_version": 1,
            "runtime": str(manifest.resolve()),
            "venv": str(venv.resolve()),
            "project": str(project.resolve()),
            "runtime_root": str(overlay.resolve()),
        },
    )
    return config


def run_environment(config: Path, data: dict, *, identity: str) -> dict:
    settings = load_environment(config)
    venv = Path(settings["venv"])
    project = Path(settings["project"])
    env = clean_environment(venv)
    env["PYTHONPATH"] = os.pathsep.join([str(project / "src"), str(project)])
    env["TT_METAL_RUNTIME_ROOT"] = settings["runtime_root"]
    if data["mode"] == "source":
        env["TT_METAL_HOME"] = data["source"]["checkout"]
    cache = config.parent / "cache" / identity
    env["TT_TRANSFORMERS_CACHE"] = str(cache / "weights")
    env["TT_CACHE_PATH"] = str(cache / "weights/model")
    env["TT_METAL_CACHE"] = str(cache / "jit")
    return env


def verify_installed_wheel(wheel: Path, package: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        record = next(name for name in archive.namelist() if name.endswith(".dist-info/RECORD"))
        for name, checksum, _ in csv.reader(io.StringIO(archive.read(record).decode())):
            if not checksum or name.split("/", 1)[0] not in {"ttnn", "ttnn.libs"}:
                continue
            path = within(package.parent, name)
            algorithm, value = checksum.split("=", 1)
            expected = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).hex()
            if algorithm != "sha256" or not path.is_file() or file_hash(path) != expected:
                raise DevError(f"Installed wheel file changed: {name}")


def doctor(config: Path) -> dict:
    settings = load_environment(config)
    manifest = Path(settings["runtime"])
    data = load_runtime(manifest)
    project = Path(settings["project"])
    venv = Path(settings["venv"])
    python = runtime_python(venv)
    info = interpreter_info(python)
    if info["abi"] != data["python"]["abi"]:
        raise DevError("Environment Python ABI differs from its runtime.")
    if requirements(project, data["dependencies"]["extras"]) != data["dependencies"]["requirements"]:
        raise DevError("Project requirements changed; recreate the runtime environment.")
    baseline = data["dependencies"].get("baseline_sha256")
    if baseline and file_hash(project / f"constraints/locks/host-py{info['minor'].replace('.', '')}.txt") != baseline:
        raise DevError("Project dependency baseline changed; rebuild or select a matching runtime bundle.")
    state = None
    if data["mode"] == "source":
        checkout = Path(data["source"]["checkout"])
        if Path(settings["runtime_root"]).resolve() != checkout.resolve():
            raise DevError("Source runtime assets point at another checkout.")
        state = source_state(checkout)
        if state["native_fingerprint"] != data["source"]["native_fingerprint"]:
            raise DevError("Native tt-metal inputs changed; run rebuild before testing.")
        for path, checksum in data["source"]["outputs"].items():
            target = Path(path)
            if not target.is_file() or file_hash(target) != checksum:
                raise DevError(f"Native output changed or missing; run rebuild: {path}")
        for path, recorded in data["source"].get("inputs", {}).items():
            target = Path(path)
            if not target.is_file():
                raise DevError(f"Native input is missing; run rebuild: {path}")
            stat = target.stat()
            if (stat.st_mtime_ns, stat.st_size) != (recorded["mtime_ns"], recorded["size"]):
                if file_hash(target) != recorded["sha256"]:
                    raise DevError(f"Native compiler/CMake dependency changed; run rebuild: {path}")
        build_dir = Path(data["source"]["build_dir"])
        glob_check = build_dir / "CMakeFiles/VerifyGlobs.cmake"
        glob_stamp = build_dir / "CMakeFiles/cmake.verify_globs"
        if glob_check.is_file():
            before = glob_stamp.stat().st_mtime_ns if glob_stamp.exists() else None
            command(["cmake", "-P", glob_check], cwd=build_dir, env=clean_environment(venv), capture=True)
            after = glob_stamp.stat().st_mtime_ns if glob_stamp.exists() else None
            if before != after:
                raise DevError("CMake globbed inputs changed; run rebuild before testing.")
    project_state = source_state(project)
    identity = digest(
        {
            "runtime": data["fingerprint"],
            "source": state and state["fingerprint"],
            "project": project_state["fingerprint"],
        }
    )
    env = run_environment(config, data, identity=identity)
    # Source egg-info under project/src is not an installed distribution.
    command([python, "-I", "-m", "pip", "check"], env=env)
    output = command([python, "-I", PROBE, str(project)], env=env, capture=True)
    line = next((line for line in output.splitlines() if line.startswith("TTNN_DEV_PROBE=")), None)
    if not line:
        raise DevError("Runtime import probe produced no identity report.")
    probe = json.loads(line.split("=", 1)[1])
    if probe["tt_transformers_distribution"] is not None:
        raise DevError("Remove the installed tt-transformers distribution; this environment runs its source checkout.")
    if probe["ttnn_version"] != data["ttnn"]["version"]:
        raise DevError("Installed TTNN version differs from the selected runtime.")
    if "requires" in data["ttnn"] and sorted(probe["ttnn_requires"]) != sorted(data["ttnn"]["requires"]):
        raise DevError("Installed TTNN dependency metadata differs from the selected wheel.")
    if not Path(probe["project_origin"]).resolve().is_relative_to((project / "src").resolve()):
        raise DevError("tt_transformers was imported from the wrong checkout.")
    origin = Path(probe["ttnn_origin"]).resolve()
    extension = Path(probe["extension_origin"]).resolve()
    if data["mode"] == "source":
        checkout = Path(data["source"]["checkout"]).resolve()
        if not origin.is_relative_to(checkout / "ttnn") or not extension.is_relative_to(checkout / "ttnn"):
            raise DevError("Editable TTNN imports do not match the selected checkout/bindings.")
        direct = probe["direct_url"]
        if not direct or not direct.get("dir_info", {}).get("editable"):
            raise DevError("TTNN is not installed editably.")
        if Path(unquote(urlparse(direct["url"]).path)).resolve() != checkout:
            raise DevError("Editable metadata points at another tt-metal checkout.")
        permitted = [checkout / "ttnn", Path(data["source"]["build_dir"]).resolve()]
    else:
        if not origin.is_relative_to(venv.resolve()) or not extension.is_relative_to(venv.resolve()):
            raise DevError("TTNN was not imported from the selected wheel environment.")
        verify_installed_wheel(within(manifest.parent, data["ttnn"]["wheel"]), origin.parent)
        overlay = config.parent / "runtime-root"
        if overlay.is_symlink() or Path(settings["runtime_root"]).resolve() != overlay.resolve():
            raise DevError("Wheel runtime assets point at another environment.")
        for child in origin.parent.iterdir():
            if child.name not in {"runtime", "__pycache__"} and (overlay / child.name).resolve() != child.resolve():
                raise DevError(f"Wheel runtime asset link was changed: {child.name}")
        for child in (origin.parent / "runtime").iterdir():
            if child.name != "sfpi" and (overlay / "runtime" / child.name).resolve() != child.resolve():
                raise DevError(f"Wheel runtime asset link was changed: runtime/{child.name}")
        if (overlay / "runtime/sfpi").resolve() != (config.parent / "sfpi").resolve():
            raise DevError("Wheel SFPI link points at another environment.")
        permitted = [venv.resolve()]
    for library in probe["native_libraries"]:
        if not any(Path(library).resolve().is_relative_to(root) for root in permitted):
            raise DevError(f"Native library came from another runtime: {library}")
    compiler = Path(settings["runtime_root"]) / "runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
    if (
        not compiler.is_file()
        or not os.access(compiler, os.X_OK)
        or file_hash(compiler) != data["toolchain"]["compiler_sha256"]
    ):
        raise DevError("SFPI compiler does not match the runtime build.")
    if data["toolchain"].get("tree_sha256"):
        if tree_hash(Path(settings["runtime_root"]) / "runtime/sfpi") != data["toolchain"]["tree_sha256"]:
            raise DevError("SFPI toolchain files changed after the runtime build.")
    return {
        "status": "pass",
        "mode": data["mode"],
        "fingerprint": identity,
        "tt_metal": state or data["metal"],
        "tt_transformers": project_state,
        "python": info,
        "imports": probe,
        "cache": env["TT_METAL_CACHE"],
    }
