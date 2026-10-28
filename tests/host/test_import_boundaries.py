# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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


@pytest.mark.host
@pytest.mark.parametrize(
    ("root", "allowed_layers"),
    [("vllm", {"models"}), ("vllm_tt_plugin", {"models"}), ("yaml", {"root"})],
)
def test_layer_scoped_dependencies(root: str, allowed_layers: set[str]) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_import_boundaries as checker
    finally:
        sys.path.pop(0)

    allowed = {
        layer for layer in checker.DEPENDENCY_POLICY_LAYERS if checker._dependency_reason(f"{root}.x", layer) is None
    }
    assert allowed == allowed_layers
