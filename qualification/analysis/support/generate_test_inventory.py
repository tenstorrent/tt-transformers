#!/usr/bin/env python3
"""Generate a static, pinned-revision census of the in-scope TTTv2 tests.

This intentionally does not import pytest or any tt-metal module and does not
collect or execute tests. Parameterized definitions therefore remain one row
per source-level test function/method.
"""

from __future__ import annotations

import ast
import configparser
import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


PINNED_SHA = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
SOURCE_ROOT = Path("/localdev/gwang/tt-metal")
OUTPUT_ROOT = Path(__file__).resolve().parent


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_ROOT), *args], text=True).strip()


def classify(path: Path) -> str:
    value = path.as_posix()
    if value.startswith("models/common/readiness_check/"):
        return "qualification"
    if value in {
        "models/common/tests/test_bh_required_capabilities.py",
        "models/common/tests/test_validation_tools.py",
        "models/test_tttv2_validate_vllm_matrix.py",
    }:
        return "qualification"
    if "/tests/demos/" in value:
        return "demo"
    if "/tests/llm_runtime/" in value:
        return "llm_runtime"
    if "/tests/models/" in value or value.endswith("/test_llama3_8b_hf_adaptor.py"):
        return "model"
    if "/tests/modules/" in value:
        return "module"
    if "/tests/host/" in value:
        return "host_support"
    return "shared_support"


def in_scope_files() -> list[Path]:
    candidates: list[Path] = []
    for base in (SOURCE_ROOT / "models/common/tests", SOURCE_ROOT / "models/common/readiness_check"):
        for path in base.rglob("*.py"):
            rel = path.relative_to(SOURCE_ROOT)
            if "/modules/moe/" in f"/{rel.as_posix()}":
                continue
            if path.name.startswith("test_") or path.name == "demo.py":
                candidates.append(rel)
    candidates.append(Path("models/test_tttv2_validate_vllm_matrix.py"))
    # A demo helper can itself contain collected tests even though it is not named
    # demo.py. Include any Python file in the demo tree that defines a test.
    for path in (SOURCE_ROOT / "models/common/tests/demos").rglob("*.py"):
        rel = path.relative_to(SOURCE_ROOT)
        if rel not in candidates:
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_") for node in ast.walk(tree)):
                candidates.append(rel)
    return sorted(set(candidates), key=lambda item: item.as_posix())


def support_asset_files() -> list[Path]:
    """Return example/qualification support files that are not all pytest files."""

    paths: set[Path] = set()
    for base in (
        SOURCE_ROOT / "models/common/tests/demos",
        SOURCE_ROOT / "models/common/readiness_check",
        SOURCE_ROOT / "models/docs/tttv2",
    ):
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                paths.add(path.relative_to(SOURCE_ROOT))
    for path in (SOURCE_ROOT / "models").glob("tttv2_*"):
        if path.is_file():
            paths.add(path.relative_to(SOURCE_ROOT))
    for package in (
        "deepseek_r1_distill_qwen_14b",
        "llama32_1b",
        "llama32_3b",
        "llama33_70b",
        "llama3_8b",
        "mistral_7b",
        "phi4",
        "qwen25_72b",
        "qwen25_7b",
        "qwen25_coder_32b",
        "qwen2_7b",
        "qwen3_32b",
    ):
        for name in ("README.md", "demo.py"):
            path = SOURCE_ROOT / "models/common/models" / package / name
            if path.is_file():
                paths.add(path.relative_to(SOURCE_ROOT))
    paths.add(Path("models/common/llm_runtime/tttv2_demo_to_vllm_integration_skill.md"))
    paths.add(Path("tests/tests_common/cache_entries_counter.py"))
    tracked = set(git("ls-tree", "-r", "--name-only", PINNED_SHA).splitlines())
    return sorted((path for path in paths if path.as_posix() in tracked), key=lambda item: item.as_posix())


def support_asset_role(path: Path) -> str:
    value = path.as_posix()
    if value.startswith("models/common/tests/demos/"):
        if path.suffix == ".refpt":
            return "demo_reference"
        if "sample_prompts" in path.parts:
            return "demo_prompt"
        if path.name.startswith("generate_"):
            return "reference_generator"
        if path.name == "demo.py":
            return "runnable_demo_test"
        return "demo_support"
    if value.startswith("models/common/readiness_check/"):
        return "readiness_test" if path.name.startswith("test_") else "readiness_tool"
    if value == "tests/tests_common/cache_entries_counter.py":
        return "fixture_support"
    if value.startswith("models/docs/tttv2/"):
        return "hardware_or_migration_documentation"
    if path.name.endswith("required_capabilities.json"):
        return "blackhole_capability_manifest"
    if path.name.endswith("required_capabilities.schema.json"):
        return "blackhole_capability_schema"
    if "validate" in path.name:
        return "qualification_validator"
    if path.name.endswith("hardware_gate_runner.sh"):
        return "qualification_runner"
    if value.endswith("tttv2_demo_to_vllm_integration_skill.md"):
        return "integration_contract"
    if path.name == "README.md":
        return "model_support_documentation"
    if path.name == "demo.py":
        return "package_local_diagnostic_demo"
    return "qualification_support"


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return dotted(node.func)
    return ""


def marker_names(node: ast.AST, aliases: dict[str, str]) -> set[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        name = dotted(child)
        if name.startswith("pytest.mark."):
            found.add(name.split(".", 2)[2])
        elif isinstance(child, ast.Name) and child.id in aliases:
            found.add(aliases[child.id])
    return found


def fixture_metadata(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, str, str, str]:
    for decorator in node.decorator_list:
        if dotted(decorator) not in {"pytest.fixture", "fixture"}:
            continue
        scope = "function"
        autouse = "false"
        params = ""
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == "scope":
                    scope = ast.unparse(keyword.value)
                elif keyword.arg == "autouse":
                    autouse = ast.unparse(keyword.value).lower()
                elif keyword.arg == "params":
                    params = ast.unparse(keyword.value)
        return True, scope.strip("'\""), autouse, params
    return False, "", "", ""


def module_data(rel: Path) -> dict:
    source = (SOURCE_ROOT / rel).read_text()
    tree = ast.parse(source, filename=rel.as_posix())
    aliases: dict[str, str] = {}
    module_markers: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None:
                direct = dotted(value)
                if direct.startswith("pytest.mark."):
                    marker = direct.split(".", 2)[2]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            aliases[target.id] = marker
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == "pytestmark":
                        module_markers.update(marker_names(value, aliases))

    tests: list[dict] = []
    fixtures: list[dict] = []

    def visit(body: list[ast.stmt], parents: list[str]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_fixture, scope, autouse, params = fixture_metadata(node)
                args = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
                if args and parents and args[0] in {"self", "cls"}:
                    args = args[1:]
                decorators = set(module_markers)
                for decorator in node.decorator_list:
                    decorators.update(marker_names(decorator, aliases))
                qualname = ".".join([*parents, node.name])
                if is_fixture:
                    fixtures.append(
                        {
                            "definition_file": rel.as_posix(),
                            "fixture": node.name,
                            "line": node.lineno,
                            "scope": scope,
                            "autouse": autouse,
                            "params_expression": params,
                            "layer": classify(rel),
                        }
                    )
                if node.name.startswith("test_"):
                    device_refs = sorted(
                        name
                        for name in args
                        if name == "device"
                        or name.endswith("device")
                        or "mesh_device" in name
                        or name in {"silicon_arch_name", "silicon_arch_blackhole", "silicon_arch_wormhole_b0"}
                    )
                    model = ""
                    parts = rel.parts
                    for anchor in ("models", "demos"):
                        if anchor in parts:
                            index = parts.index(anchor)
                            if index + 1 < len(parts):
                                model = parts[index + 1]
                                break
                    tests.append(
                        {
                            "category": classify(rel),
                            "file": rel.as_posix(),
                            "blob_sha": git("rev-parse", f"{PINNED_SHA}:{rel.as_posix()}"),
                            "test": qualname,
                            "line": node.lineno,
                            "model": model,
                            "argument_names": ";".join(args),
                            "markers": ";".join(sorted(decorators)),
                            "device_fixture_signals": ";".join(device_refs),
                            "static_hardware_signal": "fixture" if device_refs else "none",
                        }
                    )
            elif isinstance(node, ast.ClassDef):
                visit(node.body, [*parents, node.name])

    visit(tree.body, [])
    all_markers = set(module_markers)
    for node in ast.walk(tree):
        all_markers.update(marker_names(node, aliases))
    return {
        "tests": tests,
        "fixtures": fixtures,
        "markers": all_markers,
        "source": source,
    }


def registered_markers() -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(SOURCE_ROOT / "pytest.ini")
    result: dict[str, str] = {}
    for line in parser.get("pytest", "markers").splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, description = line.partition(":")
        result[name.split("(", 1)[0].strip()] = description.strip()
    return result


def write_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    with (OUTPUT_ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    actual = git("rev-parse", "HEAD")
    if actual != PINNED_SHA:
        raise SystemExit(f"Refusing to inventory {actual}; expected {PINNED_SHA}")
    if git("status", "--short", "--untracked-files=no"):
        raise SystemExit("Refusing to inventory a source checkout with tracked changes")

    paths = in_scope_files()
    tests: list[dict] = []
    fixtures: list[dict] = []
    marker_files: defaultdict[str, set[str]] = defaultdict(set)
    file_rows: list[dict] = []

    for rel in paths:
        data = module_data(rel)
        tests.extend(data["tests"])
        fixtures.extend(data["fixtures"])
        for marker in data["markers"]:
            marker_files[marker].add(rel.as_posix())
        file_rows.append(
            {
                "category": classify(rel),
                "file": rel.as_posix(),
                "blob_sha": git("rev-parse", f"{PINNED_SHA}:{rel.as_posix()}"),
                "source_test_definitions": len(data["tests"]),
                "local_fixture_definitions": len(data["fixtures"]),
                "markers": ";".join(sorted(data["markers"])),
                "has_device_fixture_signal": "true" if any(row["device_fixture_signals"] for row in data["tests"]) else "false",
            }
        )

    # Ancestor conftests affect all or a subset of the selected tree.
    for rel in (Path("conftest.py"), Path("models/conftest.py"), Path("models/common/tests/conftest.py")):
        data = module_data(rel)
        for fixture in data["fixtures"]:
            fixture["layer"] = "ancestor_conftest"
            fixtures.append(fixture)
        for marker in data["markers"]:
            marker_files[marker].add(rel.as_posix())

    registry = registered_markers()
    marker_rows = []
    test_marker_counts = Counter(marker for test in tests for marker in test["markers"].split(";") if marker)
    for marker in sorted(set(registry) | set(marker_files)):
        marker_rows.append(
            {
                "marker": marker,
                "registered_in_pytest_ini": "true" if marker in registry else "false",
                "description": registry.get(marker, ""),
                "source_test_definition_occurrences": test_marker_counts[marker],
                "files": ";".join(sorted(marker_files[marker])),
            }
        )

    write_csv(
        "test_inventory.csv",
        tests,
        [
            "category",
            "file",
            "blob_sha",
            "test",
            "line",
            "model",
            "argument_names",
            "markers",
            "device_fixture_signals",
            "static_hardware_signal",
        ],
    )
    write_csv(
        "test_file_inventory.csv",
        file_rows,
        [
            "category",
            "file",
            "blob_sha",
            "source_test_definitions",
            "local_fixture_definitions",
            "markers",
            "has_device_fixture_signal",
        ],
    )
    write_csv(
        "fixture_inventory.csv",
        sorted(fixtures, key=lambda row: (row["definition_file"], row["line"])),
        ["definition_file", "fixture", "line", "scope", "autouse", "params_expression", "layer"],
    )
    write_csv(
        "marker_inventory.csv",
        marker_rows,
        ["marker", "registered_in_pytest_ini", "description", "source_test_definition_occurrences", "files"],
    )
    asset_rows = [
        {
            "path": rel.as_posix(),
            "blob_sha": git("rev-parse", f"{PINNED_SHA}:{rel.as_posix()}"),
            "role": support_asset_role(rel),
        }
        for rel in support_asset_files()
    ]
    write_csv("support_asset_inventory.csv", asset_rows, ["path", "blob_sha", "role"])
    category_counts = Counter(row["category"] for row in tests)
    print(f"source revision: {actual}")
    print(f"test files: {len(file_rows)}")
    print(f"source test definitions: {len(tests)}")
    print(f"fixture definitions: {len(fixtures)}")
    print(f"markers: {len(marker_rows)}")
    print(f"support assets: {len(asset_rows)}")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
