#!/usr/bin/env python3
"""Apply deterministic cache, Transformers, and TTNN ownership policy to examples."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
MANIFEST = ROOT / "qualification/extraction/example_runtime_policy_manifest.json"
SMOKE_MODELS = ("qwen25_coder_32b", "qwen3_32b")


def _lines_and_offsets(text: str) -> tuple[list[str], list[int]]:
    lines = text.splitlines(keepends=True)
    offsets = []
    total = 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    return lines, offsets


def _whole_lines(node: ast.AST, lines: list[str], offsets: list[int]) -> tuple[int, int]:
    return offsets[node.lineno - 1], offsets[node.end_lineno - 1] + len(lines[node.end_lineno - 1])


def _apply_edits(text: str, edits: list[tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _remove_forced_remote_code(text: str) -> str:
    return re.sub(r",\s*trust_remote_code\s*=\s*True", "", text)


def _normalize_cache_helper(text: str, path: Path) -> str:
    tree = ast.parse(text, filename=str(path))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "lazy_weight_cache_dir_for_demo"),
        None,
    )
    if function is None:
        return text
    lines, offsets = _lines_and_offsets(text)
    header = lines[function.lineno - 1].rstrip("\n")
    replacement = f'''{header}
    """Resolve the demo cache through standalone policy while preserving TT_CACHE_PATH/<topology>."""
    topology = get_device_name(mesh_device)
    return resolve_model_cache_path(
        hf_model_id=hf_model_id,
        hf_revision=DEMO_HF_REVISION,
        topology=topology,
        append_topology_to_tt_cache_path=True,
        mesh_device=mesh_device,
    )
'''
    start, end = _whole_lines(function, lines, offsets)
    text = text[:start] + replacement + text[end:]
    model = path.parent.name
    if "from tt_transformers.cache_environment import resolve_model_cache_path" not in text:
        text = text.replace(
            "import ttnn\n",
            "import ttnn\nfrom tt_transformers.cache_environment import resolve_model_cache_path\n"
            f"from tt_transformers.models.{model}.hf_adaptor import DEFAULT_HF_REVISION as DEMO_HF_REVISION\n",
            1,
        )
    text = text.replace(
        "``model_cache/<HF_MODEL>/<device_name>`` under the current working directory.",
        "the versioned standalone cache policy under `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache.",
    )
    text = text.replace(
        "``model_cache/<HF_MODEL>/<device_name>`` under the current working directory",
        "the versioned standalone cache policy under `TT_TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, or the user cache",
    )
    return text


def _normalize_smoke_scope(text: str, path: Path) -> str:
    tree = ast.parse(text, filename=str(path))
    scoped_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("run_")
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "SetDefaultDevice"
            for child in ast.walk(node)
        )
    }
    lines, offsets = _lines_and_offsets(text)
    removals = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "SetDefaultDevice"
        ):
            removals.append((*_whole_lines(node, lines, offsets), ""))
    text = _apply_edits(text, removals)
    if "from functools import wraps" not in text:
        text = text.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nfrom functools import wraps\nimport inspect\n", 1)
    if "from tt_transformers.device_ownership import default_device_scope" not in text:
        text = text.replace(
            "import ttnn\n",
            "import ttnn\nfrom tt_transformers.device_ownership import default_device_scope\n",
            1,
        )
    if "def _scoped_default_device(" not in text:
        marker = "def run_"
        index = text.index(marker)
        helper = '''def _scoped_default_device(function):
    """Run one smoke entry point under exact-restore TTNN default-device ownership."""

    @wraps(function)
    def invoke(mesh_device, *args, **kwargs):
        owner = f"{__name__}.{function.__name__}"
        with default_device_scope(ttnn, mesh_device, owner=owner):
            return function(mesh_device, *args, **kwargs)

    return invoke


'''
        text = text[:index] + helper + text[index:]
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines(keepends=True)
    insertions = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in scoped_names:
            continue
        if any(isinstance(decorator, ast.Name) and decorator.id == "_scoped_default_device" for decorator in node.decorator_list):
            continue
        first_line = min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)
        insertions.append(first_line - 1)
    for index in sorted(insertions, reverse=True):
        lines.insert(index, "@_scoped_default_device\n")
    text = "".join(lines)
    text = text.replace(
        "names = runner.__code__.co_varnames[: runner.__code__.co_argcount]",
        "names = tuple(inspect.signature(runner).parameters)",
    )
    return text


def normalize(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = _remove_forced_remote_code(text)
    if path.name == "demo.py":
        text = _normalize_cache_helper(text, path)
    if path.name == "smoke.py" and path.parent.name in SMOKE_MODELS:
        text = _normalize_smoke_scope(text, path)
    ast.parse(text, filename=str(path))
    return text


def findings(path: Path, text: str) -> dict[str, object]:
    tree = ast.parse(text, filename=str(path))
    forced = []
    cwd_cache = []
    default_device = []
    scoped = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if any(keyword.arg == "trust_remote_code" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords):
                forced.append(node.lineno)
            if isinstance(node.func, ast.Name) and node.func.id == "Path" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "model_cache":
                cwd_cache.append(node.lineno)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"SetDefaultDevice", "GetDefaultDevice"}:
                default_device.append(node.lineno)
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(decorator, ast.Name) and decorator.id == "_scoped_default_device" for decorator in node.decorator_list
        ):
            scoped.append(node.name)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "forced_trust_remote_code_true_lines": forced,
        "cwd_model_cache_lines": cwd_cache,
        "unscoped_default_device_lines": default_device,
        "scoped_smoke_functions": sorted(scoped),
    }


def apply_policy(*, write: bool) -> int:
    mismatches = []
    records = []
    for path in sorted(EXAMPLES.rglob("*.py")):
        expected = normalize(path)
        if path.read_text(encoding="utf-8") != expected:
            mismatches.append(str(path.relative_to(ROOT)))
            if write:
                path.write_text(expected, encoding="utf-8")
        records.append(findings(path, expected))
    actual_policy = {
        "forced_trust_remote_code_true": sum(len(record["forced_trust_remote_code_true_lines"]) for record in records),
        "cwd_relative_model_cache": sum(len(record["cwd_model_cache_lines"]) for record in records),
        "unscoped_default_device_access": sum(len(record["unscoped_default_device_lines"]) for record in records),
        "scoped_smoke_functions": sum(len(record["scoped_smoke_functions"]) for record in records),
    }
    required_policy = {
        "forced_trust_remote_code_true": 0,
        "cwd_relative_model_cache": 0,
        "unscoped_default_device_access": 0,
        "scoped_smoke_functions": 12,
    }
    policy_error = actual_policy != required_policy
    if policy_error:
        mismatches.append(f"policy counts {actual_policy!r} != {required_policy!r}")
    payload = {
        "schema_version": 1,
        "policy": actual_policy,
        "files": records,
    }
    expected_manifest = json.dumps(payload, indent=2) + "\n"
    if write:
        MANIFEST.write_text(expected_manifest, encoding="utf-8")
    if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != expected_manifest:
        mismatches.append(str(MANIFEST.relative_to(ROOT)))
    if policy_error:
        print("example runtime policy violation: " + ", ".join(mismatches))
        return 1
    if not write and mismatches:
        print("example runtime policy drift: " + ", ".join(mismatches))
        return 1
    print(f"example runtime policy verified across {len(records)} Python files")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    return apply_policy(write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
