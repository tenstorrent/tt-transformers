#!/usr/bin/env python3
"""Inventory current TTTv2 TTNN API use and probe host-visible wheel symbols."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import platform
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT / "src"


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


def parents_for(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def enclosing_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(current.name)
        current = parents.get(current)
    return ".".join(reversed(names)) or "<module>"


def usage_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    parent = parents.get(node)
    if isinstance(parent, ast.Call) and parent.func is node:
        return "call"
    current: ast.AST | None = node
    while current is not None:
        parent = parents.get(current)
        if isinstance(parent, ast.arg) and parent.annotation is current:
            return "annotation"
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            parent.returns is current
            or any(argument.annotation is current for argument in parent.args.args + parent.args.kwonlyargs)
        ):
            return "annotation"
        if isinstance(parent, ast.AnnAssign) and parent.annotation is current:
            return "annotation"
        if isinstance(parent, ast.Call):
            break
        current = parent
    return "value"


def canonical_api(chain: str, aliases: dict[str, str]) -> str:
    root, separator, tail = chain.partition(".")
    if root not in aliases:
        return ""
    return aliases[root] + (separator + tail if separator else "")


def aliases_for(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ttnn":
                    aliases[alias.asname or alias.name] = "ttnn"
        elif isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "ttnn" or node.module.startswith("ttnn.")
        ):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def source_digest(paths: list[Path], source_root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def inventory(source_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    paths = sorted(source_root.rglob("*.py"))
    sites_by_api: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dynamic_sites: list[dict[str, Any]] = []

    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        parents = parents_for(tree)
        aliases = aliases_for(tree)
        relative = str(path.relative_to(ROOT))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ttnn":
                        sites_by_api["ttnn"].append(
                            {
                                "path": relative,
                                "line": node.lineno,
                                "scope": enclosing_scope(node, parents),
                                "context": "import",
                                "expression": ast.get_source_segment(source, node) or "import ttnn",
                            }
                        )
            elif isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "ttnn" or node.module.startswith("ttnn.")
            ):
                for alias in node.names:
                    api = f"{node.module}.{alias.name}"
                    sites_by_api[api].append(
                        {
                            "path": relative,
                            "line": node.lineno,
                            "scope": enclosing_scope(node, parents),
                            "context": "import",
                            "expression": ast.get_source_segment(source, node) or api,
                        }
                    )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Attribute) and parent.value is node:
                continue
            chain = dotted_chain(node)
            api = canonical_api(chain, aliases)
            if not api:
                continue
            sites_by_api[api].append(
                {
                    "path": relative,
                    "line": node.lineno,
                    "scope": enclosing_scope(node, parents),
                    "context": usage_context(node, parents),
                    "expression": ast.get_source_segment(source, node) or chain,
                }
            )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or node.id not in aliases:
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Attribute) and parent.value is node:
                continue
            if isinstance(parent, (ast.Import, ast.ImportFrom, ast.alias)):
                continue
            sites_by_api[aliases[node.id]].append(
                {
                    "path": relative,
                    "line": node.lineno,
                    "scope": enclosing_scope(node, parents),
                    "context": usage_context(node, parents),
                    "expression": ast.get_source_segment(source, node) or node.id,
                }
            )

        # Instance/dynamic return members are not module attributes. Record only
        # high-confidence receivers whose local annotation is explicitly TTNN.
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            typed_receivers = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in function.args.args + function.args.kwonlyargs
                if argument.annotation is not None and "ttnn." in ast.unparse(argument.annotation)
            }
            if not typed_receivers:
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Attribute):
                    continue
                parent = parents.get(node)
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    continue
                chain = dotted_chain(node)
                receiver, separator, member = chain.partition(".")
                if not separator or receiver not in typed_receivers:
                    continue
                dynamic_sites.append(
                    {
                        "receiver_type": typed_receivers[receiver],
                        "member": member,
                        "path": relative,
                        "line": node.lineno,
                        "scope": enclosing_scope(node, parents),
                        "context": usage_context(node, parents),
                        "resolution_status": "not_module_resolvable_without_instance",
                    }
                )

    api_rows = [
        {
            "api": api,
            "occurrences": len(sites),
            "contexts": dict(sorted(Counter(site["context"] for site in sites).items())),
            "sites": sites,
        }
        for api, sites in sorted(sites_by_api.items())
    ]
    dynamic_sites.sort(key=lambda row: (row["receiver_type"], row["member"], row["path"], row["line"]))
    return api_rows, dynamic_sites, source_digest(paths, source_root)


def resolve(root: Any, api: str) -> tuple[Any | None, str | None]:
    value = root
    for component in api.split(".")[1:]:
        try:
            value = getattr(value, component)
        except (AttributeError, ImportError) as error:
            return None, f"{type(error).__name__}: {error}"
    return value, None


def safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def probe(source_root: Path) -> dict[str, Any]:
    import ttnn

    api_rows, dynamic_sites, digest = inventory(source_root)
    results = []
    for row in api_rows:
        value, error = resolve(ttnn, row["api"])
        if row["api"] == "ttnn":
            value, error = ttnn, None
        if row["api"].startswith("ttnn._ttnn"):
            tier = "private"
        elif row["api"].startswith("ttnn.experimental"):
            tier = "experimental"
        else:
            tier = "public"
        depth = row["api"].count(".")
        results.append(
            {
                **row,
                "tier": tier,
                "probe_kind": (
                    "module_root"
                    if row["api"] == "ttnn"
                    else "nested_or_enum_attribute"
                    if depth > 1 and tier == "public"
                    else "module_attribute"
                ),
                "resolution_status": "module_symbol_available" if error is None else "module_symbol_missing",
                "available": error is None,
                "object_type": type(value).__name__ if value is not None else None,
                "signature": safe_signature(value) if value is not None else None,
                "error": error,
            }
        )

    for row in dynamic_sites:
        receiver_type = row["receiver_type"]
        if not re.fullmatch(r"ttnn(?:\.[A-Za-z_][A-Za-z0-9_]*)+", receiver_type):
            row.update(
                {
                    "resolution_status": "complex_annotation_not_class_resolvable",
                    "descriptor_available": None,
                    "descriptor_type": None,
                    "signature": None,
                    "error": None,
                }
            )
            continue
        receiver, error = resolve(ttnn, receiver_type)
        if error is not None:
            row.update(
                {
                    "resolution_status": "receiver_type_missing",
                    "descriptor_available": False,
                    "descriptor_type": None,
                    "signature": None,
                    "error": error,
                }
            )
            continue
        try:
            descriptor = inspect.getattr_static(receiver, row["member"])
        except AttributeError as error:
            row.update(
                {
                    "resolution_status": "class_descriptor_missing_instance_or_dynamic_member",
                    "descriptor_available": False,
                    "descriptor_type": None,
                    "signature": None,
                    "error": f"AttributeError: {error}",
                }
            )
        else:
            row.update(
                {
                    "resolution_status": "class_descriptor_available",
                    "descriptor_available": True,
                    "descriptor_type": type(descriptor).__name__,
                    "signature": safe_signature(descriptor),
                    "error": None,
                }
            )

    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ttnn_distribution_version": importlib.metadata.version("ttnn"),
        "ttnn_module_file": getattr(ttnn, "__file__", None),
        "sys_path": sys.path,
        "source_checkout_on_sys_path": any(
            "/localdev/gwang/tt_transformers/src" in value
            or "/localdev/gwang/tt-metal" in value
            for value in sys.path
        ),
        "source_root": str(source_root),
        "source_sha256": digest,
        "unique_api_count": len(results),
        "occurrence_count": sum(row["occurrences"] for row in results),
        "available_count": sum(row["available"] for row in results),
        "missing_count": sum(not row["available"] for row in results),
        "apis": results,
        "dynamic_instance_member_count": len(dynamic_sites),
        "dynamic_instance_members": dynamic_sites,
        "dynamic_member_note": (
            "Instance and dynamic-return members are recorded separately and are not "
            "resolved by getattr on the ttnn module; no object/device construction is attempted."
        ),
    }


def emit_csv(report: dict[str, Any]) -> str:
    fields = [
        "python",
        "kind",
        "identity",
        "tier",
        "occurrences",
        "contexts",
        "resolution_status",
        "available",
        "object_type",
        "signature",
        "signature_quality",
        "plan_taxonomy",
        "official_source_evidence",
    ]

    def signature_quality(signature: str | None) -> str:
        if signature is None:
            return "not_inspectable"
        if "**" in signature:
            return "generic_wrapper"
        return "informative"

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in report["apis"]:
        quality = signature_quality(row["signature"])
        writer.writerow(
            {
                "python": report["python"],
                "kind": "module_api",
                "identity": row["api"],
                "tier": row["tier"],
                "occurrences": row["occurrences"],
                "contexts": json.dumps(
                    row["contexts"], sort_keys=True, separators=(",", ":")
                ),
                "resolution_status": row["resolution_status"],
                "available": str(row["available"]).lower(),
                "object_type": row["object_type"],
                "signature": row["signature"],
                "signature_quality": quality,
                "plan_taxonomy": (
                    "none: symbol and informative signature present"
                    if quality == "informative"
                    else "semantic or signature difference from source environment: signature evidence incomplete"
                ),
                "official_source_evidence": "",
            }
        )
    for row in report["dynamic_instance_members"]:
        missing = row["resolution_status"] == "class_descriptor_missing_instance_or_dynamic_member"
        writer.writerow(
            {
                "python": report["python"],
                "kind": "typed_instance_member",
                "identity": f"{row['receiver_type']}.{row['member']}",
                "tier": "instance",
                "occurrences": row.get("occurrences", 1),
                "contexts": "",
                "resolution_status": row["resolution_status"],
                "available": (
                    ""
                    if row.get("descriptor_available") is None
                    else str(row["descriptor_available"]).lower()
                ),
                "object_type": row.get("descriptor_type"),
                "signature": row.get("signature"),
                "signature_quality": signature_quality(row.get("signature")),
                "plan_taxonomy": (
                    "semantic or signature difference from source environment: "
                    "abstract annotation versus concrete runtime binding"
                    if missing
                    else "not a module symbol: dynamic/union instance evidence only"
                    if row["resolution_status"] == "complex_annotation_not_class_resolvable"
                    else "none: class descriptor present; instance semantics unqualified"
                ),
                "official_source_evidence": (
                    "v0.77.0 core.cpp lines 47-117 and compute_kernel_config.hpp lines 14-34"
                    if missing
                    else ""
                ),
            }
        )
    return output.getvalue()


def emit_legacy_csv(report: dict[str, Any]) -> str:
    """Deprecated detailed-site CSV kept only for downstream script compatibility."""
    fields = [
        "python",
        "api",
        "tier",
        "probe_kind",
        "occurrences",
        "contexts",
        "resolution_status",
        "available",
        "object_type",
        "signature",
        "error",
        "sites",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in report["apis"]:
        writer.writerow(
            {
                "python": report["python"],
                "api": row["api"],
                "tier": row["tier"],
                "probe_kind": row["probe_kind"],
                "occurrences": row["occurrences"],
                "contexts": json.dumps(row["contexts"], sort_keys=True),
                "resolution_status": row["resolution_status"],
                "available": str(row["available"]).lower(),
                "object_type": row["object_type"],
                "signature": row["signature"],
                "error": row["error"],
                "sites": json.dumps(row.get("sites", []), sort_keys=True),
            }
        )
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="omit per-site detail and aggregate dynamic instance members",
    )
    args = parser.parse_args()
    report = probe(args.source_root.resolve())
    if args.compact:
        for row in report["apis"]:
            row.pop("sites", None)
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in report["dynamic_instance_members"]:
            key = (row["receiver_type"], row["member"])
            item = grouped.setdefault(
                key,
                {
                    "receiver_type": row["receiver_type"],
                    "member": row["member"],
                    "occurrences": 0,
                    "resolution_status": row["resolution_status"],
                    "descriptor_available": row.get("descriptor_available"),
                    "descriptor_type": row.get("descriptor_type"),
                    "signature": row.get("signature"),
                    "error": row.get("error"),
                },
            )
            item["occurrences"] += 1
        report["dynamic_instance_members"] = [
            grouped[key] for key in sorted(grouped)
        ]
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        sys.stdout.write(emit_csv(report))
    return int(report["missing_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
