#!/usr/bin/env python3
"""Validate release archive policy and emit deterministic artifact manifests."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import re
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath

COMMON_FORBIDDEN_TREE_PARTS = {"__pycache__", "model_cache", ".cache", "hardware-results"}
WHEEL_FORBIDDEN_TREE_PARTS = {
    *COMMON_FORBIDDEN_TREE_PARTS,
    ".github",
    "constraints",
    "docs",
    "examples",
    "qualification",
    "tests",
    "tools",
}
WHEEL_FORBIDDEN_SUFFIXES = {
    ".bin",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".refpt",
    ".safetensors",
    ".tensorbin",
}
SDIST_FORBIDDEN_SUFFIXES = {".bin", ".npy", ".npz", ".pt", ".pth", ".pyc", ".pyo", ".safetensors", ".tensorbin"}
SDIST_REQUIRED_ROOT_FILES = {
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "PKG-INFO",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "pyproject.toml",
}
SDIST_REQUIRED_DIRECTORIES = {
    ".github",
    "constraints",
    "docs",
    "examples",
    "qualification",
    "src",
    "tests",
    "tools",
}
ABSOLUTE_TEXT_PATTERNS = (b"/localdev/", b"/home/gwang/", b"/tmp/gwang/", b"C:\\")
EXPECTED_BASE_REQUIRES = {"ttnn==0.77.0", "torch==2.11.0", "loguru==0.6.0", "transformers==5.12.1"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_tree_identity(source_root: Path) -> dict[str, object]:
    files = sorted((source_root / "tt_transformers").rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(source_root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "algorithm": "sha256(path + NUL + content + NUL, sorted by path)",
        "python_files": len(files),
        "sha256": digest.hexdigest(),
    }


def validate_member_path(
    name: str,
    *,
    forbidden_parts: set[str],
    forbidden_suffixes: set[str],
) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"unsafe or absolute archive member: {name}")
    if forbidden_parts & set(path.parts):
        raise ValueError(f"forbidden tree/cache member: {name}")
    if path.suffix.lower() in forbidden_suffixes:
        raise ValueError(f"forbidden weight/reference/cache/bytecode payload: {name}")
    if path.parts and path.parts[0] == "models":
        raise ValueError(f"legacy top-level models namespace in archive: {name}")


def validate_python(name: str, data: bytes, *, enforce_production_boundary: bool) -> None:
    if enforce_production_boundary and any(pattern in data for pattern in ABSOLUTE_TEXT_PATTERNS):
        raise ValueError(f"workspace absolute path embedded in {name}")
    tree = ast.parse(data.decode("utf-8"), filename=name)
    for node in ast.walk(tree):
        imported: list[str] = []
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported = [node.module]
        for module in imported:
            root = module.split(".", 1)[0]
            if enforce_production_boundary:
                if root == "models":
                    raise ValueError(f"legacy models import in {name}:{node.lineno}: {module}")
                if root in {"tests", "examples", "qualification", "pytest"}:
                    raise ValueError(f"production repository-only import in {name}:{node.lineno}: {module}")


def parse_metadata(data: bytes, *, source: str) -> dict[str, object]:
    metadata = BytesParser(policy=default).parsebytes(data)
    if metadata["Name"] != "tt-transformers":
        raise ValueError(f"unexpected Name in {source}: {metadata['Name']}")
    if metadata["Version"] != "2.0.0.dev0":
        raise ValueError(f"unexpected Version in {source}: {metadata['Version']}")
    if metadata["Requires-Python"] != "<3.13,>=3.10":
        raise ValueError(f"unexpected Requires-Python in {source}: {metadata['Requires-Python']}")
    requires = metadata.get_all("Requires-Dist", [])
    base = {requirement for requirement in requires if ";" not in requirement}
    if base != EXPECTED_BASE_REQUIRES:
        raise ValueError(f"unexpected base requirements in {source}: {sorted(base)}")
    if metadata["License-Expression"] != "Apache-2.0":
        raise ValueError(f"unexpected license expression in {source}: {metadata['License-Expression']}")
    payload = metadata.get_payload()
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError(f"missing long description in {source}")
    return {
        "metadata_version": metadata["Metadata-Version"],
        "name": metadata["Name"],
        "version": metadata["Version"],
        "requires_python": metadata["Requires-Python"],
        "requires_dist": requires,
        "provides_extra": metadata.get_all("Provides-Extra", []),
        "license_expression": metadata["License-Expression"],
        "long_description_bytes": len(payload.encode()),
    }


def validate_record(files: dict[str, bytes], record_name: str) -> None:
    rows = list(csv.reader(io.StringIO(files[record_name].decode("utf-8"))))
    record_paths = {row[0] for row in rows}
    if record_paths != set(files):
        raise ValueError("wheel RECORD paths do not match archive members")
    for path, digest, size in rows:
        data = files[path]
        if path == record_name:
            if digest or size:
                raise ValueError("RECORD must have empty self hash/size")
            continue
        algorithm, encoded = digest.split("=", 1)
        if algorithm != "sha256":
            raise ValueError(f"non-sha256 RECORD digest for {path}")
        expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if encoded != expected or int(size) != len(data):
            raise ValueError(f"RECORD hash/size mismatch for {path}")


def audit_wheel(path: Path, source_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel members")
        files = {name: archive.read(name) for name in names if not name.endswith("/")}
    for name, data in files.items():
        validate_member_path(
            name,
            forbidden_parts=WHEEL_FORBIDDEN_TREE_PARTS,
            forbidden_suffixes=WHEEL_FORBIDDEN_SUFFIXES,
        )
        if name.endswith(".py"):
            validate_python(name, data, enforce_production_boundary=True)
    metadata_names = [name for name in files if name.endswith(".dist-info/METADATA")]
    wheel_names = [name for name in files if name.endswith(".dist-info/WHEEL")]
    record_names = [name for name in files if name.endswith(".dist-info/RECORD")]
    if not (len(metadata_names) == len(wheel_names) == len(record_names) == 1):
        raise ValueError("wheel must contain exactly one METADATA, WHEEL, and RECORD")
    metadata = parse_metadata(files[metadata_names[0]], source=metadata_names[0])
    wheel_text = files[wheel_names[0]].decode("utf-8")
    if "Root-Is-Purelib: true" not in wheel_text or "Tag: py3-none-any" not in wheel_text:
        raise ValueError("wheel must be pure Python with py3-none-any tag")
    validate_record(files, record_names[0])
    wheel_python = {name: data for name, data in files.items() if name.endswith(".py")}
    source_python = {
        str(path.relative_to(source_root)).replace("\\", "/"): path.read_bytes()
        for path in (source_root / "tt_transformers").rglob("*.py")
    }
    if wheel_python != source_python:
        missing = sorted(set(source_python) - set(wheel_python))
        extra = sorted(set(wheel_python) - set(source_python))
        changed = sorted(
            name for name in set(wheel_python) & set(source_python) if wheel_python[name] != source_python[name]
        )
        raise ValueError(f"wheel/source Python mismatch: missing={missing}, extra={extra}, changed={changed}")
    entries = [{"path": name, "size": len(data), "sha256": sha256(data)} for name, data in sorted(files.items())]
    return {
        **metadata,
        "tag": "py3-none-any",
        "entry_count": len(files),
        "source_python_files": len(source_python),
    }, entries


def audit_sdist(path: Path) -> dict[str, object]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate sdist members")
        files = {member.name: archive.extractfile(member).read() for member in members if member.isfile()}
    for name, data in files.items():
        validate_member_path(
            name,
            forbidden_parts=COMMON_FORBIDDEN_TREE_PARTS,
            forbidden_suffixes=SDIST_FORBIDDEN_SUFFIXES,
        )
        if name.endswith(".py"):
            validate_python(name, data, enforce_production_boundary=False)
    pkg_info = [
        name for name in files if PurePosixPath(name).name == "PKG-INFO" and len(PurePosixPath(name).parts) == 2
    ]
    if len(pkg_info) != 1:
        raise ValueError("sdist must contain exactly one root PKG-INFO")
    root_name = PurePosixPath(pkg_info[0]).parts[0]
    root_files = {
        PurePosixPath(name).parts[1]
        for name in files
        if len(PurePosixPath(name).parts) == 2 and PurePosixPath(name).parts[0] == root_name
    }
    if not SDIST_REQUIRED_ROOT_FILES <= root_files:
        raise ValueError(f"sdist lacks required root files: {sorted(SDIST_REQUIRED_ROOT_FILES - root_files)}")
    directories = {
        PurePosixPath(name).parts[1]
        for name in files
        if len(PurePosixPath(name).parts) > 2 and PurePosixPath(name).parts[0] == root_name
    }
    if not SDIST_REQUIRED_DIRECTORIES <= directories:
        raise ValueError(
            f"sdist lacks required source/test directories: {sorted(SDIST_REQUIRED_DIRECTORIES - directories)}"
        )
    metadata = parse_metadata(files[pkg_info[0]], source=pkg_info[0])
    return {**metadata, "entry_count": len(files)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    args = parser.parse_args()
    wheel_metadata, wheel_entries = audit_wheel(args.wheel, args.source_root)
    sdist_metadata = audit_sdist(args.sdist)
    source_identity = source_tree_identity(args.source_root)
    wheel_manifest = {
        "schema_version": 1,
        "artifact": args.wheel.name,
        "artifact_sha256": sha256(args.wheel.read_bytes()),
        "artifact_size": args.wheel.stat().st_size,
        "metadata": wheel_metadata,
        "source_tree": source_identity,
        "entries": wheel_entries,
    }
    hashes = {
        "schema_version": 1,
        "source_tree": source_identity,
        "pyproject_sha256": sha256((args.source_root.parent / "pyproject.toml").read_bytes()),
        "artifacts": [
            {
                "filename": artifact.name,
                "sha256": sha256(artifact.read_bytes()),
                "size": artifact.stat().st_size,
                "entry_count": metadata["entry_count"],
            }
            for artifact, metadata in ((args.wheel, wheel_metadata), (args.sdist, sdist_metadata))
        ],
    }
    write_json(args.output_dir / "package-wheel-manifest.json", wheel_manifest)
    write_json(args.output_dir / "package-artifact-hashes.json", hashes)
    print(json.dumps({"wheel": wheel_metadata, "sdist": sdist_metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
