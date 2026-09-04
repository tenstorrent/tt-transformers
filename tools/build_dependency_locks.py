#!/usr/bin/env python3
"""Build deterministic hash locks and provenance evidence from pip reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / "constraints/locks"
TARGETS = {
    "base-py310": ("3.10.19", "base", "base-py310.txt"),
    "host-py310": ("3.10.19", "host", "host-py310.txt"),
    "qualification-py310": ("3.10.19", "qualification", "qualification-py310.txt"),
    "build-dev-py310": ("3.10.19", "build-dev", "build-dev-py310.txt"),
    "base-py312": ("3.12.13", "base", "base-py312.txt"),
    "host-py312": ("3.12.13", "host", "host-py312.txt"),
    "qualification-py312": ("3.12.13", "qualification", "qualification-py312.txt"),
    "build-dev-py312": ("3.12.13", "build-dev", "build-dev-py312.txt"),
}
ALLOWED_HOSTS = {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}
BASE_REQUESTS = ["ttnn==0.77.0", "torch==2.11.0+cpu", "loguru==0.6.0"]
HOST_REQUESTS = [
    *BASE_REQUESTS,
    "transformers==5.12.1",
    "tqdm==4.66.3",
    "pytest==9.0.3",
    "pytest-cov==7.0.0",
    "pytest-timeout==2.4.0",
    "huggingface-hub>=0.30",
    "jsonschema>=4.23,<5",
    "PyYAML>=6.0,<7",
    "pytz>=2024.1",
]
QUALIFICATION_REQUESTS = ["openai>=1,<3", "requests>=2.31"]
BUILD_DEV_REQUESTS = [
    "setuptools>=80.0",
    "wheel",
    "build>=1.2.2",
    "mypy==1.15.0",
    "pre-commit>=4.0.0",
    "ruff==0.11.0",
    "twine==7.0.0",
    "readme-renderer[md]==46.0",
]
REQUESTS_BY_GROUP = {
    "base": BASE_REQUESTS,
    "host": HOST_REQUESTS,
    "qualification": QUALIFICATION_REQUESTS,
    "build-dev": BUILD_DEV_REQUESTS,
}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def index_name(url: str) -> str:
    host = urlparse(url).hostname
    if host == "files.pythonhosted.org":
        return "pypi"
    if host in {"download.pytorch.org", "download-r2.pytorch.org"}:
        return "pytorch_cpu"
    raise ValueError(f"unapproved archive host: {host}")


def parse_report(path: Path, *, expected_python: str, require_torch: bool) -> tuple[dict, list[dict]]:
    raw = path.read_bytes()
    report = json.loads(raw)
    actual_python = report["environment"]["python_full_version"]
    if actual_python != expected_python:
        raise ValueError(f"{path}: Python {actual_python} != {expected_python}")
    packages = []
    names = set()
    for item in report["install"]:
        name = canonical(item["metadata"]["name"])
        version = item["metadata"]["version"]
        if name in names:
            raise ValueError(f"{path}: duplicate package {name}")
        names.add(name)
        if name == "tt-transformers":
            raise ValueError(f"{path}: dependency lock must not contain tt-transformers")
        info = item["download_info"]
        url = info["url"]
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"{path}: editable/VCS/path/unapproved requirement URL {url}")
        if any(key in info for key in ("vcs_info", "dir_info")):
            raise ValueError(f"{path}: VCS/path requirement for {name}")
        filename = Path(unquote(parsed.path)).name
        if not filename.endswith(".whl"):
            raise ValueError(f"{path}: non-wheel archive selected for {name}: {filename}")
        archive_info = info.get("archive_info", {})
        sha256 = archive_info.get("hashes", {}).get("sha256")
        if sha256 is None and isinstance(archive_info.get("hash"), str):
            algorithm, separator, value = archive_info["hash"].partition("=")
            sha256 = value if separator and algorithm == "sha256" else None
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"{path}: missing SHA256 for {name}")
        packages.append(
            {
                "name": name,
                "version": version,
                "filename": filename,
                "url": url,
                "index": index_name(url),
                "sha256": sha256,
                "requested": bool(item.get("requested", False)),
            }
        )
    packages.sort(key=lambda item: item["name"])
    torch = [item for item in packages if item["name"] == "torch"]
    if require_torch and (len(torch) != 1 or torch[0]["version"] != "2.11.0+cpu" or torch[0]["index"] != "pytorch_cpu"):
        raise ValueError(f"{path}: torch resolution is not the required 2.11.0+cpu CPU-index wheel: {torch}")
    return {"sha256": digest(raw), "pip_version": report["pip_version"], "environment": report["environment"]}, packages


def lock_text(target: str, python: str, group: str, packages: list[dict]) -> str:
    lines = [
        f"# Hash-complete {group} dependency lock for CPython {python} x86-64 Linux.",
        "# Generated from pip's target-interpreter resolver report; do not add tt-transformers.",
        "--index-url https://pypi.org/simple",
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "--only-binary :all:",
        "--require-hashes",
        "",
    ]
    for package in packages:
        lines.append(f"{package['name']}=={package['version']} \\")
        lines.append(f"    --hash=sha256:{package['sha256']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolver-dir", type=Path, required=True)
    parser.add_argument("--validation-results", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    validations = {}
    if args.validation_results:
        validations.update(
            {item["target"]: item for item in json.loads(args.validation_results.read_text())["validations"]}
        )
    locks = []
    mismatches = []
    for target, (python, group, filename) in TARGETS.items():
        report_path = args.resolver_dir / f"{target}.json"
        report, packages = parse_report(report_path, expected_python=python, require_torch=group in {"base", "host"})
        text = lock_text(target, python, group, packages)
        lock_path = LOCK_DIR / filename
        if args.write:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(text)
        if not lock_path.is_file() or lock_path.read_text() != text:
            mismatches.append(str(lock_path.relative_to(ROOT)))
        locks.append(
            {
                "target": target,
                "python": python,
                "group": group,
                "lock_path": str(lock_path.relative_to(ROOT)),
                "lock_sha256": digest(text.encode()),
                "package_count": len(packages),
                "requested_requirements": REQUESTS_BY_GROUP[group],
                "resolver_report": report,
                "packages": packages,
                "strict_install_validation": validations.get(target),
            }
        )
    for target in ("base-py310", "base-py312"):
        lock = next(item for item in locks if item["target"] == target)
        pyyaml = [package for package in lock["packages"] if package["name"] == "pyyaml"]
        if len(pyyaml) != 1 or pyyaml[0]["version"] != "6.0.3":
            raise ValueError(f"{target}: base closure does not prove PyYAML 6.0.3: {pyyaml}")
    evidence = {
        "schema_version": 1,
        "status": "strict_installs_pass" if set(validations) == set(TARGETS) else "resolver_complete_install_pending",
        "indexes": {
            "primary": "https://pypi.org/simple",
            "supplemental": "https://download.pytorch.org/whl/cpu",
        },
        "policy": {
            "hash_algorithm": "sha256",
            "only_binary": True,
            "require_hashes": True,
            "reject_editable_vcs_path": True,
            "include_tt_transformers": False,
            "required_torch": "2.11.0+cpu",
            "yaml_provided_by_base_closure": True,
        },
        "locks": locks,
    }
    encoded = json.dumps(evidence, indent=2) + "\n"
    if args.manifest:
        if args.write:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(encoded)
        if not args.manifest.is_file() or args.manifest.read_text() != encoded:
            mismatches.append(str(args.manifest))
    if mismatches:
        print("dependency lock generation drift: " + ", ".join(mismatches))
        return 1
    print(
        "verified dependency locks: "
        + ", ".join(f"{lock['target']}={lock['package_count']}" for lock in locks)
        + f"; status={evidence['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
