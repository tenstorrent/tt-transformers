#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""List the hardware-matrix nodes whose tests depend on a set of changed files.

The dependency graph is static: Python imports (including imports inside functions), string literals that name a
repository module (the lazy `_EXPORTS` tables, the vLLM registry, `python -m` launches), the parent packages of every
imported module, and the `conftest.py` files pytest loads for each test. A non-Python file under a package (kernels,
configs, assets) counts as a change to the nearest enclosing package. Files that affect every test (`pyproject.toml`,
dependency locks) select every node. The selection errs toward running more: it never opens a device and is not
hardware evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "tests/hardware/hardware-matrix.json"
# Source trees scanned for imports, and the import root each one is reached by.
SOURCE_ROOTS = {"src": None, "tests": "tests", "examples": "examples", "qualification": "qualification"}
MODULE_ROOTS = frozenset({"tt_transformers", "tests", "examples", "qualification"})
GLOBAL_FILES = frozenset({"pyproject.toml"})
GLOBAL_PREFIXES = ("constraints/",)
DOCUMENTATION_SUFFIXES = frozenset({".md", ".rst", ".txt"})


@dataclass(frozen=True)
class Selection:
    nodes: list[dict[str, Any]]
    uncovered_device_tests: list[str]
    unmapped_files: list[str]
    global_files: list[str]

    def as_json(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "uncovered_device_tests": self.uncovered_device_tests,
            "unmapped_files": self.unmapped_files,
            "global_files": self.global_files,
        }


def module_name(relative: Path) -> str | None:
    """Dotted import name of a repository Python file, or None outside the scanned trees."""
    if relative.suffix != ".py" or relative.parts[0] not in SOURCE_ROOTS:
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or None


def _prefixes(name: str) -> Iterable[str]:
    parts = name.split(".")
    for end in range(1, len(parts) + 1):
        yield ".".join(parts[:end])


def _resolve_relative(name: str, package: str) -> str:
    level = len(name) - len(name.lstrip("."))
    base = package.split(".")[: len(package.split(".")) - (level - 1)] if level > 1 else package.split(".")
    rest = name[level:]
    return ".".join(base + ([rest] if rest else []))


def _package_of(module: str, relative: Path) -> str:
    return module if relative.name == "__init__.py" else module.rpartition(".")[0]


def imported_names(tree: ast.AST, module: str, relative: Path) -> set[str]:
    """Every module name `tree` may load, with all parent packages; over-approximates `from a import b`."""
    package = _package_of(module, relative)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative("." * node.level + (node.module or ""), package) if node.level else node.module
            if base:
                names.add(base)
                names.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.partition(":")[0]
            if text.startswith(".") and text.strip(".").replace(".", "").replace("_", "").isalnum():
                names.add(_resolve_relative(text, package))
            elif text.split(".", 1)[0] in MODULE_ROOTS and all(p.isidentifier() for p in text.split(".")):
                names.add(text)
    return {prefix for name in names for prefix in _prefixes(name)}


def uses_device_marker(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "device"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        for node in ast.walk(tree)
    )


def _conftest_names(relative: Path) -> set[str]:
    if relative.parts[0] != "tests":
        return set()
    return {".".join((*relative.parts[:end], "conftest")) for end in range(1, len(relative.parts))}


class DependencyGraph:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.files: dict[str, Path] = {}
        self.importers: dict[str, set[str]] = {}
        self.device_tests: set[str] = set()
        for top in SOURCE_ROOTS:
            for path in sorted((root / top).rglob("*.py")):
                relative = path.relative_to(root)
                if "__pycache__" in relative.parts:
                    continue
                name = module_name(relative)
                if name is None:
                    continue
                self.files[name] = relative
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
                if relative.parts[0] == "tests" and relative.name.startswith("test_") and uses_device_marker(tree):
                    self.device_tests.add(name)
                for imported in imported_names(tree, name, relative) | _conftest_names(relative):
                    if imported != name:
                        self.importers.setdefault(imported, set()).add(name)

    def changed_module(self, relative: Path) -> str | None:
        """The module a changed (possibly deleted) file stands for; data files map to their nearest package."""
        name = module_name(relative)
        if name is not None:
            return name
        if relative.parts[0] not in SOURCE_ROOTS or relative.suffix in DOCUMENTATION_SUFFIXES:
            return None
        for parent in relative.parents:
            if parent == Path("."):
                return None
            if (self.root / parent / "__init__.py").is_file():
                return module_name(parent / "__init__.py")
        return None

    def dependents(self, changed: Iterable[str]) -> dict[str, list[str]]:
        """Every module reached from `changed` through importers, with the import chain back to a changed module."""
        chains: dict[str, list[str]] = {name: [name] for name in changed}
        queue = deque(chains)
        while queue:
            current = queue.popleft()
            for importer in sorted(self.importers.get(current, ())):
                if importer not in chains:
                    chains[importer] = [importer, *chains[current]]
                    queue.append(importer)
        return chains


def selector_file(node: dict[str, Any]) -> Path:
    return Path(node["selector"]["target"].split("::", 1)[0])


def is_global(path: str) -> bool:
    return path in GLOBAL_FILES or path.startswith(GLOBAL_PREFIXES)


def select(changed_files: Iterable[str], matrix: dict[str, Any], graph: DependencyGraph) -> Selection:
    changed = sorted(set(changed_files))
    global_files = [path for path in changed if is_global(path)]
    modules: dict[str, str] = {}
    unmapped: list[str] = []
    for path in changed:
        if path in global_files:
            continue
        name = graph.changed_module(Path(path))
        if name is None:
            unmapped.append(path)
        else:
            modules.setdefault(name, path)
    chains = graph.dependents(modules)

    def explain(module: str) -> list[str]:
        chain = chains[module]
        return [*chain[:-1], f"{chain[-1]} ({modules[chain[-1]]})"]

    nodes = []
    targeted = set()
    for node in matrix["nodes"]:
        target = module_name(selector_file(node))
        targeted.add(target)
        if not node.get("enabled", True):
            continue
        if global_files:
            reason = [f"global file: {', '.join(global_files)}"]
        elif target in chains:
            reason = explain(target)
        else:
            continue
        nodes.append({"id": node["id"], "mesh_device": node["mesh_device"], "stage": node["stage"], "via": reason})

    uncovered = sorted(str(graph.files[name]) for name in chains if name in graph.device_tests - targeted)
    return Selection(nodes=nodes, uncovered_device_tests=uncovered, unmapped_files=unmapped, global_files=global_files)


def git_changed_files(base: str, root: Path = ROOT) -> list[str]:
    def git(*args: str) -> list[str]:
        result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
        return [line for line in result.stdout.splitlines() if line]

    changed = git("diff", "--name-only", "--no-renames", "--merge-base", base)
    return changed + git("ls-files", "--others", "--exclude-standard")


def render_text(selection: Selection) -> str:
    lines = []
    if selection.global_files:
        lines.append(f"Global files changed, every node selected: {', '.join(selection.global_files)}")
    lines.append(f"{len(selection.nodes)} affected hardware-matrix node(s)")
    for node in selection.nodes:
        lines.append(f"  {node['id']}  [{node['mesh_device']}, {node['stage']}]")
        if not selection.global_files:
            lines.append(f"      via {' -> '.join(node['via'])}")
    if selection.uncovered_device_tests:
        lines.append("Affected device tests that no matrix node runs:")
        lines.extend(f"  {path}" for path in selection.uncovered_device_tests)
    if selection.unmapped_files:
        lines.append("Changed files outside the import graph (not used for selection):")
        lines.extend(f"  {path}" for path in selection.unmapped_files)
    if selection.nodes:
        lines.append("Preview a node with: python qualification/tools/run_hardware_matrix.py --dry-run --node <id> ...")
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="git ref to diff against (merge base), plus working-tree and untracked files")
    source.add_argument("--files", nargs="+", help="changed paths relative to the repository root")
    result.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    result.add_argument("--json", action="store_true", help="print the selection as JSON")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    changed = args.files if args.files else git_changed_files(args.base)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    selection = select(changed, matrix, DependencyGraph())
    print(json.dumps(selection.as_json(), indent=2) if args.json else render_text(selection))
    return 0


if __name__ == "__main__":
    sys.exit(main())
