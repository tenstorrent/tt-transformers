# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contract: every example benchmark reads ``MESH_DEVICE`` case-insensitively."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.common.trace_region_sizes import resolve_trace_region_size

ROOT = Path(__file__).resolve().parents[2]
_RESOLVER = "ttnn_mesh_device_param_from_env"
_BENCHMARKS = sorted(
    path
    for path in (ROOT / "examples").glob("*/benchmark.py")
    if any(
        isinstance(node, ast.FunctionDef) and node.name == _RESOLVER
        for node in ast.parse(path.read_text(encoding="utf-8")).body
    )
)


class _Unsupported(Exception):
    pass


def _mesh_table(tree: ast.Module, resolver: ast.FunctionDef) -> dict:
    """The resolver's mesh-name table: module-level ``_MESH_DEVICE_TO_SHAPE`` or a ``shapes`` local."""
    for node in (*tree.body, *ast.walk(resolver)):
        target = getattr(node, "target", None) or next(iter(getattr(node, "targets", [])), None)
        if isinstance(target, ast.Name) and target.id in {"_MESH_DEVICE_TO_SHAPE", "shapes"}:
            if isinstance(node.value, ast.Dict):
                return ast.literal_eval(node.value)
    raise AssertionError("no mesh table found")


def _load_resolver(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    resolver = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == _RESOLVER)
    table = _mesh_table(tree, resolver)
    fabric = SimpleNamespace(FABRIC_1D="FABRIC_1D", FABRIC_1D_RING="FABRIC_1D_RING")
    namespace = {
        "os": os,
        "ttnn": SimpleNamespace(FabricConfig=fabric),
        "UnsupportedConfiguration": _Unsupported,
        "_MESH_DEVICE_TO_SHAPE": table,
        "resolve_trace_region_size": resolve_trace_region_size,
    }
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[_RESOLVER], table


@pytest.mark.host
@pytest.mark.parametrize("path", _BENCHMARKS, ids=[path.parent.name for path in _BENCHMARKS])
def test_mesh_device_spelling_does_not_change_the_resolved_mesh(monkeypatch, path):
    resolve, table = _load_resolver(path)
    assert table
    for name in table:
        monkeypatch.setenv("MESH_DEVICE", name)
        expected = resolve()
        for spelling in (name.upper(), name.lower(), f" {name.lower()} "):
            monkeypatch.setenv("MESH_DEVICE", spelling)
            assert resolve() == expected, f"MESH_DEVICE={spelling!r}"


@pytest.mark.host
@pytest.mark.parametrize("model", ["qwen3_32b", "llama33_70b", "llama3_8b"])
def test_p150x4_spellings_resolve_to_the_same_shape_and_trace_region(monkeypatch, model):
    resolve, _ = _load_resolver(ROOT / "examples" / model / "benchmark.py")
    monkeypatch.setenv("MESH_DEVICE", "P150x4")
    canonical = resolve()
    monkeypatch.setenv("MESH_DEVICE", "P150X4")
    assert resolve() == canonical
    assert canonical["mesh_shape"] == (1, 4)
    assert canonical["trace_region_size"] > 0


@pytest.mark.host
@pytest.mark.parametrize("path", _BENCHMARKS, ids=[path.parent.name for path in _BENCHMARKS])
def test_unknown_mesh_device_is_still_unsupported(monkeypatch, path):
    resolve, _ = _load_resolver(path)
    monkeypatch.setenv("MESH_DEVICE", "NOT_A_MESH")
    with pytest.raises(_Unsupported, match="NOT_A_MESH"):
        resolve()
