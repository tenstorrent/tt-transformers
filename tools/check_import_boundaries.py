#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Enforce the standalone package's static import-layer policy."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Import roots corresponding to the production dependency declarations in
# pyproject.toml. Distribution/import spelling differs for no entry here.
DECLARED_BASE_THIRD_PARTY_ROOTS = frozenset({"loguru", "torch", "ttnn", "transformers"})
DECLARED_OPTIONAL_THIRD_PARTY_ROOTS = frozenset({"tqdm", "yaml"})
# Installation requirements do not relax the import-layer or lazy-import policy.
RESTRICTED_ROOT_LAYERS = {
    "tqdm": frozenset({"models"}),
    "transformers": frozenset({"models"}),
    "yaml": frozenset({"root"}),
}
DEPENDENCY_POLICY_LAYERS = frozenset({"root", "modules", "sampling", "llm_runtime", "models"})
STDLIB_ROOTS = frozenset(sys.stdlib_module_names) | {"__future__"}
CANONICAL_SAMPLING_PARAMS_MODULE = "tt_transformers.sampling.sampling_params"


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    imported: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}: {self.imported}"


def imported_modules(tree: ast.AST) -> Iterable[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            if module:
                yield node.lineno, module


def _dependency_reason(imported: str, layer: str) -> str | None:
    if imported.startswith("."):
        return None
    root = imported.split(".", 1)[0]
    if root in STDLIB_ROOTS or root == "tt_transformers":
        return None
    if layer not in DEPENDENCY_POLICY_LAYERS:
        # Phase 3 dependency enforcement currently covers the runtime/model
        # product boundary. Foundation optional-path policy is tracked by its
        # own extraction qualification.
        return None
    if root in RESTRICTED_ROOT_LAYERS and layer not in RESTRICTED_ROOT_LAYERS[root]:
        return f"dependency {root} is not allowed in the {layer} layer"
    if root in DECLARED_BASE_THIRD_PARTY_ROOTS or root in DECLARED_OPTIONAL_THIRD_PARTY_ROOTS:
        return None
    return "third-party import root is not declared for the runtime/model product"


def _initializer_laziness_reason(relative: Path, imported: str) -> str | None:
    concrete_model_initializer = (
        len(relative.parts) == 3 and relative.parts[0] == "models" and relative.parts[-1] == "__init__.py"
    )
    if relative not in {Path("__init__.py"), Path("models/__init__.py")} and not concrete_model_initializer:
        return None
    root = imported.lstrip(".").split(".", 1)[0]
    if root in RESTRICTED_ROOT_LAYERS:
        return "package initializer cannot eagerly import a layer-scoped dependency"
    if relative == Path("__init__.py") and (
        imported in {".models", "tt_transformers.models"}
        or imported.startswith((".models.", "tt_transformers.models."))
    ):
        return "package root cannot eagerly import model packages"
    if relative == Path("models/__init__.py") and (
        imported.startswith(".")
        or imported == "tt_transformers.models"
        or imported.startswith("tt_transformers.models.")
    ):
        return "models package root must keep concrete and HF-backed models lazy"
    if concrete_model_initializer and (
        imported.startswith(".")
        or imported == "tt_transformers.models"
        or imported.startswith("tt_transformers.models.")
    ):
        return "concrete model package exports must resolve lazily through __getattr__"
    return None


def _structural_violations(path: Path, relative: Path, tree: ast.AST) -> list[Violation]:
    violations: list[Violation] = []
    layer = relative.parts[0] if len(relative.parts) > 1 else "root"
    if layer == "models" and path.name == "hf_generator.py":
        policy_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "tt_transformers.cache_environment"
            for alias in node.names
        }
        required = {"offline_mode", "report_model_preflight", "resolve_model_cache"}
        if not required <= policy_imports:
            violations.append(
                Violation(
                    path=path,
                    line=1,
                    imported="tt_transformers.cache_environment",
                    reason="HF generators must use the shared cache/environment policy",
                )
            )
        preflight_calls: list[ast.Call] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "report_model_preflight":
                    preflight_calls.append(node)
                if node.func.id == "Path" and node.args and isinstance(node.args[0], ast.Constant):
                    if node.args[0].value == "model_cache":
                        violations.append(
                            Violation(
                                path=path,
                                line=node.lineno,
                                imported='Path("model_cache")',
                                reason="HF generator cache defaults cannot resolve against CWD",
                            )
                        )
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
                if node.attr in {"getenv", "environ"}:
                    violations.append(
                        Violation(
                            path=path,
                            line=node.lineno,
                            imported=f"os.{node.attr}",
                            reason="HF generator environment access must use the typed shared policy",
                        )
                    )
        if len(preflight_calls) != 1:
            violations.append(
                Violation(
                    path=path,
                    line=1,
                    imported="report_model_preflight",
                    reason=f"HF generator must emit exactly one preflight report, found {len(preflight_calls)}",
                )
            )
        elif not any(keyword.arg == "cache_resolution" for keyword in preflight_calls[0].keywords):
            violations.append(
                Violation(
                    path=path,
                    line=preflight_calls[0].lineno,
                    imported="report_model_preflight",
                    reason="HF generator preflight must report the exact CacheResolution",
                )
            )
        cache_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "resolve_model_cache"
        ]
        if len(cache_calls) != 1 or not {
            "hf_model_id",
            "hf_revision",
            "topology",
            "mesh_device",
            "dtype",
        } <= {keyword.arg for call in cache_calls for keyword in call.keywords}:
            violations.append(
                Violation(
                    path=path,
                    line=cache_calls[0].lineno if cache_calls else 1,
                    imported="resolve_model_cache",
                    reason="HF generator cache resolution must include the complete identity inputs",
                )
            )
    for node in ast.walk(tree):
        if layer == "models" and isinstance(node, ast.Call):
            if any(
                keyword.arg == "trust_remote_code"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            ):
                violations.append(
                    Violation(
                        path=path,
                        line=node.lineno,
                        imported="trust_remote_code=True",
                        reason="model loaders cannot force execution of repository-supplied code",
                    )
                )
        if layer == "models" and isinstance(node, ast.Dict):
            if any(
                isinstance(key, ast.Constant)
                and key.value == "trust_remote_code"
                and isinstance(value, ast.Constant)
                and value.value is True
                for key, value in zip(node.keys, node.values)
            ):
                violations.append(
                    Violation(
                        path=path,
                        line=node.lineno,
                        imported='{"trust_remote_code": True}',
                        reason="model loaders cannot forward a forced remote-code opt-in",
                    )
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"SetDefaultDevice", "GetDefaultDevice"}
            and relative != Path("device_ownership.py")
        ):
            violations.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    imported=node.func.attr,
                    reason="direct TTNN default-device access is forbidden outside the scoped owner",
                )
            )
        if isinstance(node, ast.ImportFrom) and node.module in {
            "tt_transformers.sampling",
            "tt_transformers.sampling.generator",
        }:
            if any(alias.name == "SamplingParams" for alias in node.names):
                violations.append(
                    Violation(
                        path=path,
                        line=node.lineno,
                        imported=f"{node.module}.SamplingParams",
                        reason=(f"SamplingParams must use the canonical owner {CANONICAL_SAMPLING_PARAMS_MODULE}"),
                    )
                )
        if layer not in DEPENDENCY_POLICY_LAYERS or not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name == "from_model_args":
                violations.append(
                    Violation(
                        path=path,
                        line=child.lineno,
                        imported=f"{node.name}.from_model_args",
                        reason="TTTv1 from_model_args adapters are forbidden in runtime/model production",
                    )
                )
            elif node.name == "Llama3Transformer1D" and child.name == "forward":
                violations.append(
                    Violation(
                        path=path,
                        line=child.lineno,
                        imported="Llama3Transformer1D.forward",
                        reason="characterized TTTv1 model dispatcher is forbidden",
                    )
                )
    return violations


def check_file(path: Path, package_root: Path) -> list[Violation]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(package_root)
    layer = relative.parts[0] if len(relative.parts) > 1 else "root"
    violations: list[Violation] = []

    for line, imported in imported_modules(tree):
        reason: str | None = None
        if imported == "models" or imported.startswith("models."):
            reason = "legacy tt-metal namespace is forbidden"
        elif imported == "tests" or imported.startswith("tests."):
            reason = "production code cannot import tests"
        elif imported == "examples" or imported.startswith("examples."):
            reason = "installed code cannot import top-level examples"
        elif imported == "pytest" or imported.startswith("pytest."):
            reason = "pytest is not a production dependency"
        elif layer == "modules" and (
            imported == "tt_transformers.models"
            or imported.startswith("tt_transformers.models.")
            or imported == "tt_transformers.llm_runtime"
            or imported.startswith("tt_transformers.llm_runtime.")
        ):
            reason = "reusable modules cannot depend on runtime or model layers"
        elif layer == "llm_runtime" and (
            imported == "tt_transformers.models" or imported.startswith("tt_transformers.models.")
        ):
            reason = "model-neutral runtime cannot import concrete models"
        if reason is None:
            reason = _initializer_laziness_reason(relative, imported)
        if reason is None:
            reason = _dependency_reason(imported, layer)

        if reason:
            violations.append(Violation(path=path, line=line, imported=imported, reason=reason))

    violations.extend(_structural_violations(path, relative, tree))
    return violations


def check_tree(package_root: Path) -> list[Violation]:
    return [violation for path in sorted(package_root.rglob("*.py")) for violation in check_file(path, package_root)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", nargs="?", type=Path, default=Path("src/tt_transformers"))
    args = parser.parse_args()
    violations = check_tree(args.package_root)
    for violation in violations:
        print(violation)
    return bool(violations)


if __name__ == "__main__":
    raise SystemExit(main())
