from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.host
def test_static_import_boundaries() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_import_boundaries.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
