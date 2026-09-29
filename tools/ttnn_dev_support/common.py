# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Runtime identity, source provenance, locking, and subprocess helpers."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "https://github.com/tenstorrent/tt-metal.git"
SCHEMA_VERSION = 1


class DevError(RuntimeError):
    """An actionable refusal before using an inconsistent development runtime."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def tool_fingerprint() -> str:
    files = [ROOT / "tools/ttnn_dev.py", *Path(__file__).parent.glob("*.py")]
    return digest({str(path.relative_to(ROOT)): file_hash(path) for path in sorted(files)})


def tree_hash(root: Path) -> str:
    records = {}

    def visit(directory: Path, ancestors: set[Path]) -> None:
        resolved = directory.resolve()
        if resolved in ancestors:
            raise DevError(f"Toolchain contains a symlink cycle: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                visit(path, ancestors | {resolved})
            elif path.is_file():
                records[str(path.relative_to(root))] = file_hash(path)
            else:
                raise DevError(f"Unsupported toolchain entry: {path}")

    visit(root, set())
    return digest(records)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        raise DevError(f"Cannot read {path}: {error}") from error


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_environment(path: Path) -> dict:
    data = read_json(path)
    if data.get("schema_version") != 1:
        raise DevError("Unsupported development environment schema.")
    for key in ("runtime", "venv", "project", "runtime_root"):
        if not isinstance(data.get(key), str) or not Path(data[key]).is_absolute():
            raise DevError(f"Environment {key} must be an absolute path.")
    return data


def command(argv, *, cwd=None, env=None, capture=False) -> str:
    argv = [str(part) for part in argv]
    if not capture:
        print("+ " + shlex.join(argv), flush=True)
    result = subprocess.run(argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE if capture else None)
    if result.returncode:
        raise DevError(f"Command failed ({result.returncode}): {shlex.join(argv)}")
    return result.stdout.strip() if capture else ""


def git(checkout: Path, *args: str) -> str:
    # Container mounts may retain a different host UID. Trust only the checkout
    # explicitly selected by this invocation, without editing global git config.
    return command(["git", "-c", f"safe.directory={checkout.resolve()}", "-C", checkout, *args], capture=True)


def repository_url(checkout: Path) -> str | None:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={checkout.resolve()}",
            "-C",
            str(checkout),
            "config",
            "--get",
            "remote.origin.url",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    url = result.stdout.strip()
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "ssh"} or not parsed.hostname:
        return None
    host = parsed.hostname + (f":{parsed.port}" if parsed.port else "")
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as error:
            raise DevError("Use Python 3.11+ for the launcher, or install tomli on Python 3.10.") from error
    return tomllib.loads(path.read_text())


def interpreter_info(python: str | Path) -> dict:
    requested = str(python)
    if re.fullmatch(r"3\.(10|12)(\.\d+)?", requested):
        found = shutil.which("python" + ".".join(requested.split(".")[:2]))
        if not found:
            raise DevError(f"Python {requested} is not available; supply the target interpreter path.")
        python = found
    source = (
        "import json,platform,sys,sysconfig; print(json.dumps(dict("
        "version=platform.python_version(),minor=f'{sys.version_info.major}.{sys.version_info.minor}',"
        "abi=sys.implementation.cache_tag,platform=sysconfig.get_platform(),"
        "machine=platform.machine(),executable=sys.executable)))"
    )
    info = json.loads(command([python, "-c", source], capture=True))
    if re.fullmatch(r"3\.(10|12)\.\d+", requested) and info["version"] != requested:
        raise DevError(f"Requested Python {requested}, found {info['version']}; supply the exact interpreter path.")
    if info["minor"] not in {"3.10", "3.12"} or info["machine"] != "x86_64":
        raise DevError("The initial runtime profiles require CPython 3.10/3.12 on Linux x86-64.")
    if not info["platform"].startswith("linux-"):
        raise DevError("TTNN development runtimes require Linux.")
    return info


def clean_environment(venv: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "TT_METAL_HOME",
        "TT_METAL_RUNTIME_ROOT",
        "TT_METAL_KERNEL_PATH",
        "TT_FROM_PRECOMPILED_DIR",
        "LD_LIBRARY_PATH",
        "PYTHON_ENV_DIR",
        "VIRTUAL_ENV",
        "CCACHE_REMOTE_STORAGE",
        "CCACHE_REMOTE_ONLY",
        "TT_METAL_CACHE",
        "TT_CACHE_PATH",
        "TT_TRANSFORMERS_CACHE",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
    ):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    if venv:
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = str(venv / "bin") + os.pathsep + env["PATH"]
    return env


def source_state(checkout: Path) -> dict:
    """Hash working-tree changes, including submodules, without changing the checkout."""
    sha = git(checkout, "rev-parse", "HEAD")
    changes = git(checkout, "diff", "--binary", "HEAD", "--")
    # Native inputs conservatively include everything except TTNN's Python modules.
    native = git(checkout, "diff", "--binary", "HEAD", "--", ".", ":(glob,exclude)ttnn/ttnn/**/*.py")
    untracked = {}
    native_untracked = {}
    for name in git(checkout, "ls-files", "--others", "--exclude-standard", "-z").split("\0"):
        if not name:
            continue
        path = checkout / name
        if not path.is_file() or path.suffix not in {
            ".py",
            ".cpp",
            ".cc",
            ".c",
            ".h",
            ".hpp",
            ".inl",
            ".cmake",
            ".sh",
            ".txt",
            ".toml",
            ".json",
            ".yaml",
            ".yml",
            ".S",
            ".ld",
            ".in",
        }:
            continue
        untracked[name] = file_hash(path)
        if not (name.startswith("ttnn/ttnn/") and name.endswith(".py")):
            native_untracked[name] = untracked[name]
    submodules = {}
    for entry in git(checkout, "ls-files", "--stage", "-z").split("\0"):
        if entry.startswith("160000 "):
            name = entry.split("\t", 1)[1]
            child = checkout / name
            if not (child / ".git").exists():
                raise DevError(f"Submodule {name} is missing; initialize submodules before building.")
            submodules[name] = source_state(child)
    identity = {
        "sha": sha,
        "patch": hashlib.sha256(changes.encode()).hexdigest(),
        "untracked": untracked,
        "submodules": {name: state["fingerprint"] for name, state in submodules.items()},
    }
    native_identity = {**identity, "patch": hashlib.sha256(native.encode()).hexdigest(), "untracked": native_untracked}
    return {
        **identity,
        "repository": repository_url(checkout),
        "dirty": bool(changes or untracked or any(state["dirty"] for state in submodules.values())),
        "fingerprint": digest(identity),
        "native_fingerprint": digest(native_identity),
        "submodule_revisions": {name: state["sha"] for name, state in submodules.items()},
    }


def within(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise DevError(f"Runtime artifact path escapes its directory: {relative}")
    return path


def save_runtime(path: Path, data: dict) -> None:
    data = {key: value for key, value in data.items() if key != "fingerprint"}
    data["schema_version"] = SCHEMA_VERSION
    data["fingerprint"] = digest(data)
    write_json(path, data)


def load_runtime(path: Path) -> dict:
    data = read_json(path)
    if data.get("schema_version") != SCHEMA_VERSION or data.get("mode") not in {"source", "wheel"}:
        raise DevError("Unsupported runtime manifest; expected schema_version=1 and mode=source/wheel.")
    if data.get("fingerprint") != digest({key: value for key, value in data.items() if key != "fingerprint"}):
        raise DevError("Runtime manifest fingerprint mismatch.")
    required = {"metal", "python", "recipe", "ttnn", "dependencies", "toolchain"}
    if any(not isinstance(data.get(key), dict) for key in required):
        raise DevError("Runtime manifest is missing its source, Python, recipe, dependency, or toolchain description.")
    fields = {
        "metal": {"sha"},
        "python": {"abi", "minor"},
        "ttnn": {"version"},
        "dependencies": {"directory", "extras", "requirements"},
        "toolchain": {"compiler_sha256"},
    }
    if any(not keys <= data[section].keys() for section, keys in fields.items()):
        raise DevError("Runtime manifest contains an incomplete description.")
    if not isinstance(data.get("files"), list) or not data["files"]:
        raise DevError("Runtime manifest must declare its dependency artifacts.")
    seen = set()
    for entry in data["files"]:
        if not isinstance(entry, dict) or not {"path", "sha256"} <= entry.keys():
            raise DevError("Malformed runtime artifact entry.")
        if not isinstance(entry["path"], str) or not isinstance(entry["sha256"], str):
            raise DevError("Runtime artifact paths/checksums must be strings.")
        if entry["path"] in seen or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise DevError("Duplicate artifact path or invalid checksum.")
        seen.add(entry["path"])
        artifact = within(path.parent, entry["path"])
        if not artifact.is_file() or file_hash(artifact) != entry["sha256"]:
            raise DevError(f"Runtime artifact missing or modified: {artifact}")
    if data["dependencies"]["directory"] + "/requirements.txt" not in seen:
        raise DevError("Dependency lock is not covered by the runtime fingerprint.")
    if data["mode"] == "wheel":
        if data["ttnn"].get("wheel") not in seen or data["toolchain"].get("archive") not in seen:
            raise DevError("Wheel/toolchain is not covered by the runtime fingerprint.")
    elif (
        not isinstance(data.get("source"), dict)
        or not {"checkout", "build_dir", "native_fingerprint", "outputs"} <= data["source"].keys()
        or not data["source"]["outputs"]
    ):
        raise DevError("Source runtime does not identify its native build outputs.")
    return data


def artifact_entry(root: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": file_hash(path)}


@contextlib.contextmanager
def runtime_lock(path: Path, *, exclusive: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DevError(
                "Runtime is in use. Finish its launched commands before rebuilding or changing it."
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def host_tools() -> dict:
    tools = {}
    for name in ("cmake", "ninja", "clang++-20"):
        try:
            tools[name] = command([name, "--version"], capture=True).splitlines()[0]
        except (OSError, DevError) as error:
            raise DevError(f"Missing build tool {name}; run in the documented tt-metal development image.") from error
    tools["os"] = platform.platform()
    return tools


def runtime_python(venv: Path) -> Path:
    python = venv / "bin/python"
    if not python.is_file():
        raise DevError(f"Environment does not exist: {venv}. Run 'env' or 'attach' first.")
    return python
