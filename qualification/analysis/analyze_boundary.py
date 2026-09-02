#!/usr/bin/env python3
"""Reproduce the pinned TTTv2 production-boundary inventories.

This script reads source exclusively through Git objects. It never imports the
source tree and never writes to the source repository.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_REPO = Path("/localdev/gwang/tt-metal")
DEFAULT_REVISION = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
MODEL_FAMILIES = {
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
SHARED_MODEL_FILES = {
    "models/common/models/executor.py",
    "models/common/models/llama3_executor.py",
    "models/common/models/qwen2_executor.py",
}
PUBLIC_DUNDERS = {
    "__call__",
    "__enter__",
    "__exit__",
    "__getitem__",
    "__iter__",
    "__len__",
    "__next__",
    "__setitem__",
}
BASE_DEPENDENCIES = {"loguru", "torch", "ttnn"}
OPTIONAL_DEPENDENCIES = {"tqdm", "transformers"}


def git(source_repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_repo), *args],
        text=True,
        stderr=subprocess.PIPE,
    )


def is_in_scope(path: str) -> bool:
    if not path.endswith(".py"):
        return False
    if path.startswith("models/common/modules/"):
        return not path.startswith("models/common/modules/moe/")
    if path.startswith("models/common/llm_runtime/"):
        return True
    if path in SHARED_MODEL_FILES:
        return True
    parts = path.split("/")
    return (
        len(parts) >= 5
        and parts[:3] == ["models", "common", "models"]
        and parts[3] in MODEL_FAMILIES
    )


def layer_for(path: str) -> str:
    if path.startswith("models/common/modules/"):
        return "modules"
    if path.startswith("models/common/llm_runtime/"):
        return "llm_runtime"
    return "models"


def module_for(path: str) -> str:
    module = path[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def destination_module_for(path: str) -> str:
    if path.startswith("models/common/modules/"):
        tail = path[len("models/common/modules/") : -3].replace("/", ".")
        return f"tt_transformers.modules.{tail}".removesuffix(".__init__")
    if path.startswith("models/common/llm_runtime/"):
        tail = path[len("models/common/llm_runtime/") : -3].replace("/", ".")
        return f"tt_transformers.llm_runtime.{tail}".removesuffix(".__init__")
    tail = path[len("models/common/models/") : -3].replace("/", ".")
    return f"tt_transformers.models.{tail}".removesuffix(".__init__")


def source_text(source_repo: Path, revision: str, path: str) -> str:
    return git(source_repo, "show", f"{revision}:{path}")


def safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def one_line(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def parents_for(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def enclosing_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names: list[str] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(current.name)
        current = parents.get(current)
    return ".".join(reversed(names)) or "<module>"


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    signature = f"({safe_unparse(node.args)})"
    if node.returns is not None:
        signature += f" -> {safe_unparse(node.returns)}"
    if isinstance(node, ast.AsyncFunctionDef):
        signature = "async " + signature
    return signature


def public_name(name: str) -> bool:
    return not name.startswith("_") or name in PUBLIC_DUNDERS or name == "__init__"


def explicit_all(tree: ast.Module) -> set[str]:
    result: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            result.update(
                item.value for item in value.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return result


def class_signature(node: ast.ClassDef) -> str:
    bases = [safe_unparse(base) for base in node.bases]
    bases.extend(f"{kw.arg}={safe_unparse(kw.value)}" for kw in node.keywords)
    return f"({', '.join(bases)})" if bases else "()"


def public_symbol_rows(path: str, layer: str, tree: ast.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exports = explicit_all(tree)

    def export_status(name: str) -> str:
        if exports:
            return "explicit" if name in exports else "not_in___all__"
        return "candidate_no___all__"

    def decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
        return "|".join(safe_unparse(item) for item in node.decorator_list)

    def visit_class(node: ast.ClassDef, prefix: str = "") -> None:
        qualname = f"{prefix}.{node.name}" if prefix else node.name
        if public_name(node.name):
            rows.append(
                {
                    "layer": layer,
                    "source_path": path,
                    "line": node.lineno,
                    "qualname": qualname,
                    "kind": "class",
                    "signature": class_signature(node),
                    "decorators": decorators(node),
                    "export_status": export_status(node.name) if not prefix else "class_member",
                }
            )
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and public_name(child.name):
                rows.append(
                    {
                        "layer": layer,
                        "source_path": path,
                        "line": child.lineno,
                        "qualname": f"{qualname}.{child.name}",
                        "kind": "method",
                        "signature": function_signature(child),
                        "decorators": decorators(child),
                        "export_status": "class_member",
                    }
                )
            elif isinstance(child, ast.ClassDef):
                visit_class(child, qualname)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and public_name(node.name):
            rows.append(
                {
                    "layer": layer,
                    "source_path": path,
                    "line": node.lineno,
                    "qualname": node.name,
                    "kind": "function",
                    "signature": function_signature(node),
                    "decorators": decorators(node),
                    "export_status": export_status(node.name),
                }
            )
        elif isinstance(node, ast.ClassDef):
            visit_class(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            annotation = safe_unparse(node.annotation) if isinstance(node, ast.AnnAssign) else ""
            for target in targets:
                names = [item.id for item in ast.walk(target) if isinstance(item, ast.Name)]
                for name in names:
                    if public_name(name) and name != "__all__":
                        signature = f": {annotation}" if annotation else ""
                        if value is not None:
                            signature += f" = {one_line(safe_unparse(value), 240)}"
                        rows.append(
                            {
                                "layer": layer,
                                "source_path": path,
                                "line": node.lineno,
                                "qualname": name,
                                "kind": "constant_or_value",
                                "signature": signature,
                                "decorators": "",
                                "export_status": export_status(name),
                            }
                        )
    represented = {
        row["qualname"]
        for row in rows
        if row["export_status"] != "class_member" and "." not in row["qualname"]
    }
    import_origins: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                import_origins[alias.asname or alias.name] = f"{module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_origins[alias.asname or alias.name.split(".", 1)[0]] = alias.name
    all_line = next(
        (
            node.lineno
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        ),
        0,
    )
    for name in sorted(exports - represented):
        origin = import_origins.get(name, "")
        rows.append(
            {
                "layer": layer,
                "source_path": path,
                "line": all_line,
                "qualname": name,
                "kind": "explicit_export",
                "signature": f"re-export of {origin}" if origin else "explicit __all__ export",
                "decorators": "",
                "export_status": "explicit",
            }
        )
    return rows


def internal_resolution(module: str, imported_name: str, internal_modules: set[str]) -> str:
    if module in internal_modules:
        return module
    candidate = f"{module}.{imported_name}" if imported_name and imported_name != "*" else ""
    if candidate in internal_modules:
        return candidate
    if any(item.startswith(module + ".") for item in internal_modules):
        return candidate if candidate and any(item == candidate or item.startswith(candidate + ".") for item in internal_modules) else module
    return ""


def import_classification(module: str, imported_name: str, internal_modules: set[str]) -> tuple[str, str]:
    root = module.split(".", 1)[0]
    resolved = internal_resolution(module, imported_name, internal_modules)
    if root == "models":
        if resolved:
            return "internal_models_namespace", resolved
        if ".tests." in f".{module}." or module.endswith(".tests"):
            return "external_models_test", ""
        if module.startswith("models.tt_transformers"):
            return "external_models_tttv1", ""
        return "external_models_tt_metal", ""
    if root == "pytest":
        return "third_party_test", ""
    if root in BASE_DEPENDENCIES:
        return "third_party_base", ""
    if root in OPTIONAL_DEPENDENCIES:
        return "third_party_optional", ""
    if root in sys.stdlib_module_names or root == "__future__":
        return "stdlib", ""
    return "third_party_other", ""


def import_rows(
    path: str,
    layer: str,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    internal_modules: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        scope = enclosing_scope(node, parents)
        timing = "module" if scope == "<module>" else "nested"
        if isinstance(node, ast.Import):
            for alias in node.names:
                classification, resolved = import_classification(alias.name, "", internal_modules)
                rows.append(
                    {
                        "layer": layer,
                        "source_path": path,
                        "line": node.lineno,
                        "scope": scope,
                        "timing": timing,
                        "kind": "import",
                        "module": alias.name,
                        "imported_symbol": "",
                        "alias": alias.asname or "",
                        "root": alias.name.split(".", 1)[0],
                        "classification": classification,
                        "internal_target": resolved,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                classification, resolved = import_classification(module, alias.name, internal_modules)
                rows.append(
                    {
                        "layer": layer,
                        "source_path": path,
                        "line": node.lineno,
                        "scope": scope,
                        "timing": timing,
                        "kind": "from",
                        "module": module,
                        "imported_symbol": alias.name,
                        "alias": alias.asname or "",
                        "root": module.lstrip(".").split(".", 1)[0],
                        "classification": classification,
                        "internal_target": resolved,
                    }
                )
    return rows


def external_owner_treatment(module: str, symbol: str) -> tuple[str, str, str]:
    if module == "models.common.device_utils":
        return "tt_transformers.device_utils", "Internalize the narrow get_device_name helper and its minimal architecture/SKU logic.", "P0"
    if module == "models.common.lightweightmodule":
        return "tt_transformers.modules package core", "Internalize LightweightModule with its current construction/call contract; do not retain the old namespace.", "P0"
    if module in {"models.common.sampling", "models.common.sampling.sampling_params"}:
        return "tt_transformers.sampling", "Choose one canonical SamplingParams value and mechanically redirect all callers while preserving fields/defaults.", "P0"
    if module == "models.common.sampling.vocab_padding":
        return "tt_transformers.sampling.vocab_padding", "Internalize only the three imported vocabulary-padding helpers.", "P0"
    if module == "models.common.tensor_utils":
        return "tt_transformers.tensor_utils", "Extract the imported tensor/shape helpers and their transitive minimum; split host-only helpers later.", "P0"
    if module.startswith("models.common.tests."):
        return "examples/<model> plus tests/qualification", "Remove the production-to-tests edge; keep orchestration helpers with examples and assertions/fixtures with tests.", "P0"
    if module == "models.common.utility_functions" and symbol == "is_blackhole":
        return "tt_transformers.device_utils", "Replace the broad utility import with a narrow architecture predicate.", "P0"
    if module == "models.common.utility_functions" and symbol == "comp_pcc":
        return "tests/qualification", "Move PCC comparison out of production/example package code and into qualification helpers.", "P0"
    if module == "models.common.utils" and symbol == "LogProbsCalculator":
        return "tt_transformers.sampling.logprobs", "Internalize the narrow log-probability helper required by device sampling.", "P0"
    if module == "models.tt_transformers.tt.common" and symbol == "Mode":
        return "tt_transformers.llm_runtime neutral mode contract", "Replace with a TTTv2-owned enum/Literal or the existing string-only prefill/decode contract.", "P0"
    if module == "models.tt_transformers.tt.common" and symbol == "rope_scaling_model_factory":
        return "TTTv1 bridge removal", "Remove RotarySetup1D.from_model_args; do not carry the TTTv1 rope-scaling factory.", "P0"
    if module == "models.tt_transformers.tt.generator" and symbol == "create_submeshes":
        return "tt_transformers.mesh_utils", "Internalize the narrow submesh-construction helper and characterize ownership/cleanup.", "P0"
    if module == "models.tt_transformers.tt.model_config":
        return "TTTv1 bridge removal", "Remove the from_model_args bridge that consumes OpGroup/TensorGroup; do not internalize legacy model config.", "P0"
    if module == "models.tt_transformers.tt.rope":
        return "TTTv1 bridge removal", "Remove RotarySetup1D.from_model_args and use TTTv2-owned RoPE table construction paths.", "P0"
    return "unassigned", "Review and assign before extraction.", "P0"


def external_dependency_rows(imports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in imports:
        if not row["classification"].startswith("external_models_"):
            continue
        owner, treatment, priority = external_owner_treatment(row["module"], row["imported_symbol"])
        rows.append({**row, "proposed_owner": owner, "proposed_treatment": treatment, "priority": priority})
    return rows


def dotted_chain(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def ttnn_aliases(tree: ast.Module) -> dict[str, str]:
    aliases = {"ttnn": "ttnn"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ttnn":
                    aliases[alias.asname or "ttnn"] = "ttnn"
    return aliases


def canonical_chain(node: ast.AST, aliases: dict[str, str]) -> str:
    chain = dotted_chain(node)
    if not chain:
        return ""
    root, separator, tail = chain.partition(".")
    if root in aliases:
        return aliases[root] + (separator + tail if separator else "")
    return chain


def unstable_ttnn_rows(
    path: str,
    layer: str,
    source: str,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aliases = ttnn_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue
        api = canonical_chain(node, aliases)
        if not (api.startswith("ttnn.experimental") or api.startswith("ttnn._ttnn")):
            continue
        rows.append(
            {
                "tier": "private" if api.startswith("ttnn._ttnn") else "experimental",
                "api": api,
                "layer": layer,
                "source_path": path,
                "line": node.lineno,
                "scope": enclosing_scope(node, parents),
                "expression": one_line(ast.get_source_segment(source, node) or safe_unparse(node)),
            }
        )
    return rows


def literal_or_expression(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return safe_unparse(node)


def resolved_dynamic_keys(node: ast.AST, key_expression: str, parents: dict[ast.AST, ast.AST]) -> str:
    if not key_expression or key_expression.startswith(("'", '"')):
        return key_expression.strip("'\"")
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            for generator in current.generators:
                if isinstance(generator.target, ast.Name) and generator.target.id == key_expression:
                    if isinstance(generator.iter, (ast.Tuple, ast.List, ast.Set)):
                        values = [
                            item.value
                            for item in generator.iter.elts
                            if isinstance(item, ast.Constant) and isinstance(item.value, str)
                        ]
                        if values:
                            return "|".join(values)
        current = parents.get(current)
    return ""


def environment_rows(
    path: str,
    layer: str,
    source: str,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        access = ""
        key_node: ast.AST | None = None
        default_node: ast.AST | None = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            function = dotted_chain(node.func)
            if function in {"os.getenv", "os.environ.get"} and node.args:
                access = function
                key_node = node.args[0]
                default_node = node.args[1] if len(node.args) > 1 else None
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        default_node = keyword.value
        elif isinstance(node, ast.Subscript) and dotted_chain(node.value) == "os.environ":
            access = "os.environ[]"
            key_node = node.slice
        if not access:
            continue
        key_expression = literal_or_expression(key_node)
        rows.append(
            {
                "layer": layer,
                "source_path": path,
                "line": node.lineno,
                "scope": enclosing_scope(node, parents),
                "access": access,
                "key_expression": key_expression,
                "resolved_keys": resolved_dynamic_keys(node, key_expression, parents),
                "default_expression": literal_or_expression(default_node),
                "expression": one_line(ast.get_source_segment(source, node) or safe_unparse(node)),
            }
        )
    return rows


def runtime_assumption_rows(
    path: str,
    layer: str,
    source: str,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aliases = ttnn_aliases(tree)

    def add(node: ast.AST, category: str, assumption: str, impact: str, treatment: str) -> None:
        rows.append(
            {
                "category": category,
                "assumption": assumption,
                "layer": layer,
                "source_path": path,
                "line": getattr(node, "lineno", 0),
                "scope": enclosing_scope(node, parents),
                "expression": one_line(ast.get_source_segment(source, node) or safe_unparse(node)),
                "impact": impact,
                "proposed_treatment": treatment,
            }
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = canonical_chain(node.func, aliases) if isinstance(node.func, ast.Attribute) else safe_unparse(node.func)
        attribute = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if function == "ttnn.SetDefaultDevice":
            add(
                node,
                "global_device",
                "mutates TTNN process-global default device",
                "Construction can affect unrelated models/tests and is not restored by most provider paths.",
                "Pass mesh_device explicitly; if compatibility requires a default, scope and restore it deterministically.",
            )
        elif function == "ttnn.GetDefaultDevice":
            add(
                node,
                "global_device",
                "falls back to TTNN process-global default device",
                "Module construction depends on external process state when config/device fields are absent.",
                "Require or propagate explicit mesh_device while preserving characterized compatibility entry points.",
            )
        if function in {"Path", "pathlib.Path"} and node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str) and value and not value.startswith("/"):
                add(
                    node,
                    "cwd_relative_path",
                    f"relative path literal {value!r} resolves against current working directory",
                    "Cache/output placement and permissions change with invocation directory.",
                    "Adopt an explicit cache/output root and report it in preflight diagnostics.",
                )
        if attribute in {"mkdir", "exists", "is_file", "glob"}:
            add(
                node,
                "filesystem",
                f"filesystem {attribute} operation",
                "Requires filesystem visibility and, for mkdir, write permission.",
                "Document path ownership, errors, and cleanup; use the explicit cache policy.",
            )
        elif function == "open":
            add(
                node,
                "filesystem",
                "opens a caller/environment-selected output path",
                "Writes benchmark/qualification output as a runtime side effect.",
                "Keep under examples/qualification and require an explicit output path.",
            )
        elif function == "ttnn.load_tensor":
            add(
                node,
                "cache_io",
                "loads a TTNN tensor cache file",
                "Correctness depends on cache compatibility and readable local storage.",
                "Version cache keys with package, TTNN, checkpoint revision, conversion schema, architecture, and topology.",
            )
        elif function == "ttnn._ttnn.tensor.dump_tensor_flatbuffer":
            add(
                node,
                "cache_io",
                "writes a cache through a private TTNN API",
                "Packaging depends on a private symbol and local write permission.",
                "Qualify a public 0.77 equivalent or report the missing TTNN capability; do not vendor TTNN internals.",
            )
        elif attribute == "from_pretrained" and function.split(".")[0] in {
            "AutoConfig",
            "AutoModelForCausalLM",
            "AutoTokenizer",
        }:
            add(
                node,
                "checkpoint_io",
                "Hugging Face from_pretrained may read cache or access network",
                "Model construction depends on checkpoint authorization, revision/cache availability, and online/offline policy.",
                "Pin revisions and make online/offline and cache roots explicit; keep Transformers optional and lazily imported.",
            )
        if any(keyword.arg == "cache_file_name" for keyword in node.keywords):
            add(
                node,
                "cache_io",
                "passes cache_file_name to a TTNN conversion/allocation API",
                "TTNN may perform implicit local cache reads/writes.",
                "Route through the versioned package cache policy and document cache invalidation inputs.",
            )
    return rows


def tttv1_bridge_rows(
    path: str,
    layer: str,
    source: str,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    imports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "from_model_args":
            rows.append(
                {
                    "kind": "from_model_args_factory",
                    "layer": layer,
                    "source_path": path,
                    "line": node.lineno,
                    "scope": enclosing_scope(node, parents),
                    "symbol": f"{enclosing_scope(node, parents)}.{node.name}".removeprefix("<module>."),
                    "signature_or_import": function_signature(node),
                    "proposed_treatment": "Remove after characterization; this is an explicit TTTv1 ModelArgs bridge per the migration plan.",
                }
            )
    manual_legacy_methods = {
        ("models/common/modules/rope/rope_1d.py", "RotarySetup1D.get_rot_idxs"),
        ("models/common/modules/rope/rope_1d.py", "RotarySetup1D.get_rot_mats"),
        ("models/common/models/llama3_8b/model.py", "Llama3Transformer1D.forward"),
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        qualname = f"{enclosing_scope(node, parents)}.{node.name}".removeprefix("<module>.")
        if (path, qualname) in manual_legacy_methods:
            rows.append(
                {
                    "kind": "legacy_dispatch_or_adapter",
                    "layer": layer,
                    "source_path": path,
                    "line": node.lineno,
                    "scope": enclosing_scope(node, parents),
                    "symbol": qualname,
                    "signature_or_import": function_signature(node),
                    "proposed_treatment": "Characterize direct callers, then remove the TTTv1-only adapter while retaining the TTTv2 core path.",
                }
            )
    for row in imports:
        if not row["module"].startswith("models.tt_transformers"):
            continue
        owner, treatment, _ = external_owner_treatment(row["module"], row["imported_symbol"])
        rows.append(
            {
                "kind": "tttv1_namespace_import",
                "layer": layer,
                "source_path": path,
                "line": row["line"],
                "scope": row["scope"],
                "symbol": row["imported_symbol"],
                "signature_or_import": f"from {row['module']} import {row['imported_symbol']}",
                "proposed_treatment": f"Owner: {owner}. {treatment}",
            }
        )
    return rows


def dependency_policy(root: str, classifications: set[str]) -> tuple[str, str]:
    if classifications == {"stdlib"}:
        return "base-allowed", "Python standard library; no declaration required."
    if root in BASE_DEPENDENCIES:
        detail = {
            "ttnn": "Declare exact qualified TTNN version; separately qualify all experimental/private sites.",
            "torch": "Declare a directly qualified base dependency bound; runtime use is hard, not incidental.",
            "loguru": "Declare a directly qualified base dependency bound.",
        }[root]
        return "base-dependency", detail
    if root in OPTIONAL_DEPENDENCIES:
        return "examples/model-provider extra", "Keep optional; lazy exports must allow base-package import without this dependency."
    if root == "pytest":
        return "test-only", "Remove from production demo modules; declare only in the test extra."
    if root == "models":
        return "boundary-closure", "Rewrite internal edges to tt_transformers and close every external models.* edge using external_models_dependencies.csv."
    return "review", "Review whether this dependency is declared or should be eliminated."


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_tree_entry(line: str) -> tuple[str, str]:
    metadata, path = line.split("\t", 1)
    _mode, _kind, blob = metadata.split()
    return path, blob


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    resolved_revision = git(args.source_repo, "rev-parse", "--verify", f"{args.revision}^{{commit}}").strip()
    if resolved_revision != args.revision:
        raise SystemExit(f"revision resolved to {resolved_revision}, expected exact {args.revision}")

    tree_entries = dict(
        parse_tree_entry(line)
        for line in git(args.source_repo, "ls-tree", "-r", args.revision).splitlines()
    )
    paths = sorted(path for path in tree_entries if is_in_scope(path))
    internal_modules = {module_for(path) for path in paths}
    expected_layers = {"modules": 16, "llm_runtime": 21, "models": 76}
    actual_layers = Counter(layer_for(path) for path in paths)
    if dict(actual_layers) != expected_layers:
        raise SystemExit(f"source-scope assertion failed: {dict(actual_layers)} != {expected_layers}")

    file_rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    unstable_rows: list[dict[str, Any]] = []
    env_rows: list[dict[str, Any]] = []
    assumption_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []

    for path in paths:
        layer = layer_for(path)
        source = source_text(args.source_repo, args.revision, path)
        tree = ast.parse(source, filename=path)
        parents = parents_for(tree)
        per_file_imports = import_rows(path, layer, tree, parents, internal_modules)
        file_rows.append(
            {
                "layer": layer,
                "source_path": path,
                "source_module": module_for(path),
                "destination_module_candidate": destination_module_for(path),
                "git_blob_sha": tree_entries[path],
                "lines": len(source.splitlines()),
                "nonblank_lines": sum(bool(line.strip()) for line in source.splitlines()),
            }
        )
        symbol_rows.extend(public_symbol_rows(path, layer, tree))
        imports.extend(per_file_imports)
        unstable_rows.extend(unstable_ttnn_rows(path, layer, source, tree, parents))
        env_rows.extend(environment_rows(path, layer, source, tree, parents))
        assumption_rows.extend(runtime_assumption_rows(path, layer, source, tree, parents))
        bridge_rows.extend(tttv1_bridge_rows(path, layer, source, tree, parents, per_file_imports))

    imports.sort(key=lambda row: (row["source_path"], row["line"], row["module"], row["imported_symbol"]))
    external_rows = external_dependency_rows(imports)
    test_import_rows = [
        row
        for row in imports
        if row["classification"] in {"external_models_test", "third_party_test"}
    ]

    edge_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in imports:
        source_module = module_for(row["source_path"])
        target = row["internal_target"] or row["module"]
        edge_groups[(source_module, target, row["classification"])].append(row)
    edge_rows = []
    for (source_module, target, classification), grouped in sorted(edge_groups.items()):
        edge_rows.append(
            {
                "source_module": source_module,
                "target_module": target,
                "classification": classification,
                "imported_symbol_count": len(grouped),
                "site_count": len({(row["source_path"], row["line"]) for row in grouped}),
                "symbols": "|".join(sorted({row["imported_symbol"] for row in grouped if row["imported_symbol"]})),
            }
        )

    dependency_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in imports:
        dependency_groups[row["root"]].append(row)
    dependency_rows = []
    for root, grouped in sorted(dependency_groups.items()):
        classifications = {row["classification"] for row in grouped}
        policy, treatment = dependency_policy(root, classifications)
        dependency_rows.append(
            {
                "root": root,
                "classifications": "|".join(sorted(classifications)),
                "imported_symbol_count": len(grouped),
                "statement_site_count": len({(row["source_path"], row["line"]) for row in grouped}),
                "file_count": len({row["source_path"] for row in grouped}),
                "layers": "|".join(sorted({row["layer"] for row in grouped})),
                "policy_bucket": policy,
                "proposed_treatment": treatment,
            }
        )

    output = args.output_dir
    write_csv(
        output / "production_files.csv",
        file_rows,
        ["layer", "source_path", "source_module", "destination_module_candidate", "git_blob_sha", "lines", "nonblank_lines"],
    )
    write_csv(
        output / "public_symbols.csv",
        sorted(symbol_rows, key=lambda row: (row["source_path"], row["line"], row["qualname"])),
        ["layer", "source_path", "line", "qualname", "kind", "signature", "decorators", "export_status"],
    )
    write_csv(
        output / "imports.csv",
        imports,
        ["layer", "source_path", "line", "scope", "timing", "kind", "module", "imported_symbol", "alias", "root", "classification", "internal_target"],
    )
    write_csv(
        output / "import_edges.csv",
        edge_rows,
        ["source_module", "target_module", "classification", "imported_symbol_count", "site_count", "symbols"],
    )
    write_csv(
        output / "dependency_roots.csv",
        dependency_rows,
        ["root", "classifications", "imported_symbol_count", "statement_site_count", "file_count", "layers", "policy_bucket", "proposed_treatment"],
    )
    write_csv(
        output / "external_models_dependencies.csv",
        external_rows,
        ["priority", "layer", "source_path", "line", "scope", "timing", "module", "imported_symbol", "classification", "proposed_owner", "proposed_treatment"],
    )
    write_csv(
        output / "ttnn_unstable_uses.csv",
        sorted(unstable_rows, key=lambda row: (row["source_path"], row["line"], row["api"])),
        ["tier", "api", "layer", "source_path", "line", "scope", "expression"],
    )
    write_csv(
        output / "environment_variables.csv",
        sorted(env_rows, key=lambda row: (row["source_path"], row["line"], row["access"])),
        ["layer", "source_path", "line", "scope", "access", "key_expression", "resolved_keys", "default_expression", "expression"],
    )
    write_csv(
        output / "runtime_assumptions.csv",
        sorted(assumption_rows, key=lambda row: (row["source_path"], row["line"], row["category"], row["assumption"])),
        ["category", "assumption", "layer", "source_path", "line", "scope", "expression", "impact", "proposed_treatment"],
    )
    write_csv(
        output / "tttv1_bridges.csv",
        sorted(bridge_rows, key=lambda row: (row["source_path"], row["line"], row["kind"], row["symbol"])),
        ["kind", "layer", "source_path", "line", "scope", "symbol", "signature_or_import", "proposed_treatment"],
    )
    write_csv(
        output / "production_test_imports.csv",
        test_import_rows,
        ["layer", "source_path", "line", "scope", "timing", "kind", "module", "imported_symbol", "alias", "classification"],
    )

    resolved_env_keys: set[str] = set()
    for row in env_rows:
        if row["resolved_keys"]:
            resolved_env_keys.update(row["resolved_keys"].split("|"))
        else:
            expression = row["key_expression"]
            if expression.startswith(("'", '"')):
                resolved_env_keys.add(expression.strip("'\""))
    unstable_counts = Counter(row["api"] for row in unstable_rows)
    summary = {
        "source_repo": str(args.source_repo),
        "revision": args.revision,
        "scope": {
            "python_files": len(paths),
            "layers": dict(sorted(actual_layers.items())),
            "total_lines": sum(row["lines"] for row in file_rows),
            "moe_excluded": True,
        },
        "public_symbols": {
            "rows": len(symbol_rows),
            "by_kind": dict(sorted(Counter(row["kind"] for row in symbol_rows).items())),
            "files_with_explicit___all__": len(
                {
                    row["source_path"]
                    for row in symbol_rows
                    if row["export_status"] in {"explicit", "not_in___all__"}
                }
            ),
        },
        "imports": {
            "rows": len(imports),
            "edge_rows": len(edge_rows),
            "classifications": dict(sorted(Counter(row["classification"] for row in imports).items())),
            "external_models_rows": len(external_rows),
            "external_models_modules": sorted({row["module"] for row in external_rows}),
            "production_test_import_rows": len(test_import_rows),
        },
        "ttnn_unstable": {
            "occurrences": len(unstable_rows),
            "unique_apis": len(unstable_counts),
            "by_api": dict(sorted(unstable_counts.items())),
        },
        "environment": {
            "access_occurrences": len(env_rows),
            "resolved_keys": sorted(resolved_env_keys),
        },
        "runtime_assumptions": {
            "rows": len(assumption_rows),
            "by_category": dict(sorted(Counter(row["category"] for row in assumption_rows).items())),
        },
        "tttv1_bridges": {
            "rows": len(bridge_rows),
            "by_kind": dict(sorted(Counter(row["kind"] for row in bridge_rows).items())),
        },
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
