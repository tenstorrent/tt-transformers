#!/usr/bin/env python3
"""Probe the installed TTNN wheel for every unstable API used by TTTv2."""

from __future__ import annotations

import csv
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
USES = ROOT / "qualification/analysis/ttnn_unstable_uses.csv"


def resolve(root: Any, dotted_path: str) -> tuple[Any | None, str | None]:
    value = root
    for component in dotted_path.split(".")[1:]:
        try:
            value = getattr(value, component)
        except (AttributeError, ImportError) as error:
            return None, f"{type(error).__name__}: {error}"
    return value, None


def signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def main() -> int:
    import ttnn

    with USES.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    apis = sorted({row["api"] for row in rows})
    occurrences = {api: sum(row["api"] == api for row in rows) for api in apis}

    results = []
    for api in apis:
        value, error = resolve(ttnn, api)
        results.append(
            {
                "api": api,
                "occurrences": occurrences[api],
                "available": error is None,
                "signature": signature(value) if value is not None else None,
                "object_type": type(value).__name__ if value is not None else None,
                "error": error,
            }
        )

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ttnn_distribution_version": importlib.metadata.version("ttnn"),
        "ttnn_module_file": getattr(ttnn, "__file__", None),
        "apis": results,
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return int(any(not result["available"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
