#!/usr/bin/env python3
"""Validate that static-quality debt is explicit and bound to this source tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "qualification/reports/static-quality-baseline.json"


def source_digest() -> tuple[int, str]:
    digest = hashlib.sha256()
    files = sorted((ROOT / "src/tt_transformers").rglob("*.py"))
    for path in files:
        relative = path.relative_to(ROOT / "src").as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return len(files), digest.hexdigest()


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    count, digest = source_digest()
    if (count, digest) != (baseline["source_python_files"], baseline["source_tree_sha256"]):
        raise ValueError("static-quality baseline does not identify the current source tree")
    if baseline["ruff_check"]["findings"] != sum(baseline["ruff_check"]["by_code"].values()):
        raise ValueError("Ruff finding total does not match by-code counts")
    if any(baseline[tool]["status"] == "pass" for tool in ("ruff_check", "ruff_format", "mypy", "black")):
        raise ValueError("failing/deferred quality gates must not be mislabeled as passing")
    print(
        f"validated static-quality debt for {count} files: "
        f"ruff={baseline['ruff_check']['findings']}, mypy={baseline['mypy']['errors']}, "
        f"format={baseline['ruff_format']['would_reformat']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
