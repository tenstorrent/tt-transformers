#!/usr/bin/env python3
"""Validate target-specific host constraints against qualification evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def expected_freeze(python_tag: str) -> list[str]:
    report = json.loads((ROOT / f"qualification/reports/dependencies-py{python_tag}.json").read_text())
    return [
        requirement
        for requirement in report["freeze"]
        if not requirement.lower().startswith(("pip==", "setuptools=="))
    ]


def validate_one(python_tag: str) -> tuple[int, str]:
    path = ROOT / f"constraints/host-py{python_tag}.txt"
    text = path.read_text(encoding="utf-8")
    requirements = [line for line in text.splitlines() if line and not line.startswith("#")]
    expected = expected_freeze(python_tag)
    if requirements != expected:
        raise ValueError(f"{path.name} differs from dependency evidence")
    if len(requirements) != len(set(requirement.lower() for requirement in requirements)):
        raise ValueError(f"{path.name} contains duplicate normalized requirements")
    for direct in ("ttnn==0.77.0", "torch==2.11.0+cpu", "loguru==0.6.0"):
        if direct not in requirements:
            raise ValueError(f"{path.name} is missing {direct}")
    return len(requirements), hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    results = {tag: validate_one(tag) for tag in ("310", "312")}
    print(
        "validated host constraints: "
        + ", ".join(f"py{tag}={count} packages sha256={digest}" for tag, (count, digest) in results.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
