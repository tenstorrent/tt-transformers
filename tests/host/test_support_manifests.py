"""Host checks for support manifests and the Phase 3 support boundary."""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
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
}


@pytest.mark.host
def test_support_schema_and_all_model_manifests_have_required_shape():
    schema = json.loads((ROOT / "qualification/schemas/support-manifest.schema.json").read_text())
    required = set(schema["required"])
    paths = sorted((ROOT / "examples").glob("*/support.json"))
    if {path.parent.name for path in paths} != MODELS:
        raise AssertionError("support manifests must cover exactly the twelve extracted models")
    for path in paths:
        manifest = json.loads(path.read_text())
        if not required <= set(manifest):
            raise AssertionError(f"{path}: missing {required - set(manifest)}")
        if manifest["model"]["package"] != path.parent.name:
            raise AssertionError(f"{path}: package/path mismatch")
        if manifest["status"] != "experimental":
            raise AssertionError(f"{path}: extraction may not claim qualification")
        if manifest["validation"] != {"date": None, "git_sha": None, "evidence": []}:
            raise AssertionError(f"{path}: unsupported validation claim")


@pytest.mark.host
def test_examples_are_pytest_free_and_tests_depend_on_examples_only_one_way():
    for path in sorted((ROOT / "examples").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                raise AssertionError(f"{path}:{node.lineno}: example contains assert")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(name == "pytest" or name.startswith("tests") for name in names):
                    raise AssertionError(f"{path}:{node.lineno}: forbidden example import {names}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    raise AssertionError(f"{path}:{node.lineno}: test function remains in example")
                if any("pytest" in ast.unparse(decorator) for decorator in node.decorator_list):
                    raise AssertionError(f"{path}:{node.lineno}: pytest decorator remains")


@pytest.mark.host
def test_hardware_wrappers_import_public_example_modules():
    wrappers = sorted((ROOT / "tests/hardware/models").glob("*/test_*.py"))
    if len(wrappers) != 15:
        raise AssertionError(f"expected 15 hardware wrappers, got {len(wrappers)}")
    for path in wrappers:
        source = path.read_text()
        if "from examples." not in source or "run_" not in source:
            raise AssertionError(f"{path}: not delegated to a public example helper")
        if path.name == "test_demo.py":
            tree = ast.parse(source, filename=str(path))
            assert any(
                isinstance(node, ast.ImportFrom)
                and node.module == f"examples.{path.parent.name}"
                and any(alias.name == "benchmark" and alias.asname == "example" for alias in node.names)
                for node in tree.body
            ), f"{path}: hardware benchmark gate must delegate to benchmark.py"
