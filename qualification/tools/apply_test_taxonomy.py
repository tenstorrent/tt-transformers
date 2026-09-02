#!/usr/bin/env python3
"""Apply explicit, behavior-neutral pytest taxonomy marks to extracted tests."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
DEVICE_ARGUMENTS = {
    "device",
    "mesh_device",
    "ttnn_mesh_device",
    "bh_1d_mesh_device",
    "bh_2d_mesh_device",
    "t3k_single_board_mesh_device",
    "pcie_mesh_device",
}
TOPOLOGY_MARKS = {"wormhole", "blackhole", "n150", "n300", "t3k", "p150", "p300", "p150x4"}
PROVEN_DEVICE_MARKS = {
    ("tests/modules/mlp/test_mlp_2d.py", "test_ttnn_linear_2d_mesh_topology_bug"): ("wormhole",),
}


def test_functions(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            yield node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                    yield child


def existing_marks(node) -> set[str]:
    result = set()
    for decorator in node.decorator_list:
        value = ast.unparse(decorator)
        if value.startswith("pytest.mark."):
            result.add(value.split(".", 2)[2].split("(", 1)[0])
    return result


def is_clear_host_path(relative: str) -> bool:
    return relative.startswith(
        (
            "tests/host/",
            "tests/llm_runtime/",
            "tests/qualification/",
            "tests/integration/",
        )
    ) or relative in {
        "tests/hardware/test_device_lock.py",
        "tests/hardware/test_mesh_fixture_policy.py",
    }


def marks_for(path: Path, node) -> list[str]:
    relative = path.relative_to(ROOT).as_posix()
    args = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
    device = relative.startswith("tests/hardware/models/") or relative.endswith(
        ("test_p150x4_smoke.py", "test_t3k_batched_prefill_correctness.py")
    )
    if not is_clear_host_path(relative):
        device |= bool(args & DEVICE_ARGUMENTS) or any("mesh_device" in name for name in args)
    device |= (relative, node.name) in PROVEN_DEVICE_MARKS
    marks = ["device" if device else "host"]
    marks.extend(PROVEN_DEVICE_MARKS.get((relative, node.name), ()))
    if relative.startswith(("tests/models/", "tests/hardware/models/")):
        marks.append("model")
    if relative.startswith("tests/hardware/models/") or "/profiling/" in relative:
        marks.append("slow")

    if device:
        evidence = f"{relative}::{node.name}".lower()
        if "p150x4" in evidence:
            marks.extend(("blackhole", "p150x4"))
        elif "p300" in evidence:
            marks.extend(("blackhole", "p300"))
        elif "p150" in evidence:
            marks.extend(("blackhole", "p150"))
        if "t3k" in evidence:
            marks.extend(("wormhole", "t3k"))
        elif "n300" in evidence:
            marks.extend(("wormhole", "n300"))
        elif "n150" in evidence:
            marks.extend(("wormhole", "n150"))
        if "blackhole" in evidence and "blackhole" not in marks:
            marks.append("blackhole")
        if "wormhole" in evidence and "wormhole" not in marks:
            marks.append("wormhole")
    return list(dict.fromkeys(marks))


def import_insertion(tree: ast.Module, text: str) -> int:
    lines = text.splitlines(keepends=True)
    offsets = []
    total = 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    body_index = 0
    end_line = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        end_line = tree.body[0].end_lineno
        body_index = 1
    while body_index < len(tree.body):
        node = tree.body[body_index]
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
            break
        end_line = node.end_lineno
        body_index += 1
    if end_line == 0:
        return 0
    return offsets[end_line - 1] + len(lines[end_line - 1])


def apply_file(path: Path) -> tuple[int, int]:
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    functions = list(test_functions(tree))
    if not functions:
        return 0, 0
    edits = []
    needs_pytest = False
    marked = 0
    for node in functions:
        current = existing_marks(node)
        desired = marks_for(path, node)
        unwanted = (current & (TOPOLOGY_MARKS | {"host", "device"})) - set(desired)
        for decorator in node.decorator_list:
            value = ast.unparse(decorator)
            if not value.startswith("pytest.mark."):
                continue
            mark = value.split(".", 2)[2].split("(", 1)[0]
            if mark in unwanted:
                decorator_start = sum(len(line) for line in text.splitlines(keepends=True)[: decorator.lineno - 1])
                decorator_end = sum(len(line) for line in text.splitlines(keepends=True)[: decorator.end_lineno])
                edits.append((decorator_start, decorator_end, ""))
                current.remove(mark)
        wanted = [mark for mark in desired if mark not in current]
        if not wanted:
            continue
        first_line = min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)
        line_start = sum(len(line) for line in text.splitlines(keepends=True)[: first_line - 1])
        indent = " " * node.col_offset
        decorators = "".join(f"{indent}@pytest.mark.{mark}\n" for mark in wanted)
        edits.append((line_start, line_start, decorators))
        needs_pytest = True
        marked += 1
    if needs_pytest and not any(
        (isinstance(node, ast.Import) and any(alias.name == "pytest" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "pytest")
        for node in tree.body
    ):
        position = import_insertion(tree, text)
        edits.append((position, position, "\nimport pytest\n"))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    ast.parse(text, filename=str(path))
    path.write_text(text)
    return len(functions), marked


def augment_hardware_wrappers() -> int:
    count = 0
    for path in sorted((TESTS / "hardware/models").glob("*/test_*.py")):
        text = path.read_text()
        if "selected_topology_marks" not in text:
            text = text.replace(
                "import pytest\n",
                "import pytest\n\nfrom tests.support.marker_policy import selected_topology_marks\n",
                1,
            )
            tree = ast.parse(text, filename=str(path))
            assignment = next(
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
            )
            lines = text.splitlines(keepends=True)
            end = sum(len(line) for line in lines[: assignment.end_lineno])
            if isinstance(assignment.value, ast.List):
                addition = "\npytestmark.extend(selected_topology_marks(pytest))\n"
            else:
                addition = "\npytestmark = [pytestmark, *selected_topology_marks(pytest)]\n"
            text = text[:end] + addition + text[end:]
            ast.parse(text, filename=str(path))
            path.write_text(text)
            count += 1
    return count


def main() -> None:
    total = marked = 0
    for path in sorted(TESTS.rglob("test_*.py")):
        functions, changed = apply_file(path)
        total += functions
        marked += changed
    wrappers = augment_hardware_wrappers()
    print(f"audited {total} source test functions; added taxonomy to {marked}; topology augmentation in {wrappers} wrappers")


if __name__ == "__main__":
    main()
