#!/usr/bin/env python3
"""Build and verify the source-to-extracted public API policy inventory."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "qualification" / "analysis"
SOURCE_SYMBOLS = ANALYSIS / "public_symbols.csv"
IMPORTS = ANALYSIS / "imports.csv"
PROVENANCE = ROOT / "qualification" / "provenance" / "source_inventory.csv"
POLICY = ANALYSIS / "public_api_policy.csv"
REMOVALS = ANALYSIS / "public_api_removals.csv"

sys.path.insert(0, str(ANALYSIS))
from analyze_boundary import module_for, public_symbol_rows  # noqa: E402

STATUSES = {
    "supported_public",
    "model_local_public",
    "compatibility_only",
    "private_by_policy",
}

COMPATIBILITY_REMOVED = {
    "Attention1D.from_model_args",
    "Embedding1D.from_model_args",
    "LMHead1D.from_model_args",
    "MLP1D.from_model_args",
    "MLP2D.from_model_args",
    "RMSNorm1D.from_model_args",
    "RMSNorm2D.from_model_args",
    "RotarySetup1D.from_model_args",
    "Penalties1D.from_model_args",
    "Sampling1D.from_model_args",
    "RotarySetup1D.get_rot_idxs",
    "RotarySetup1D.get_rot_mats",
    "Llama3Transformer1D.forward",
}

RELOCATED_NONPACKAGE = {
    "models/common/models/qwen25_coder_32b/demo.py",
    "models/common/models/qwen3_32b/demo.py",
}

# Phase 7 pins the default HF checkpoints while preserving every prior call.
# These signatures add an optional keyword, or change an optional revision
# default from a model-specific constant to ``None`` so a custom model ID does
# not silently inherit the default model's immutable revision.
OPTIONAL_HF_REVISION_EXTENSIONS = {
    ("models/common/models/llama32_1b/hf_adaptor.py", "load_tokenizer"),
    ("models/common/models/llama32_1b/hf_adaptor.py", "from_pretrained"),
    ("models/common/models/llama32_3b/hf_adaptor.py", "load_tokenizer"),
    ("models/common/models/llama32_3b/hf_adaptor.py", "from_pretrained"),
    ("models/common/models/llama33_70b/hf_adaptor.py", "load_tokenizer"),
    ("models/common/models/llama33_70b/hf_adaptor.py", "from_pretrained"),
    ("models/common/models/llama3_8b/hf_adaptor.py", "load_tokenizer"),
    ("models/common/models/llama3_8b/hf_adaptor.py", "load_converted_state_dict"),
    ("models/common/models/llama3_8b/hf_adaptor.py", "from_pretrained"),
}
HF_REVISION_DEFAULT_NONE = {
    ("models/common/models/mistral_7b/hf_adaptor.py", "load_tokenizer"),
    ("models/common/models/mistral_7b/hf_adaptor.py", "from_pretrained"),
}
IMMUTABLE_HF_REVISION_VALUES = {
    ("models/common/models/mistral_7b/hf_adaptor.py", "DEFAULT_HF_REVISION"): (
        " = 'c170c708c41dac9275d15a8fff4eca08d52bab71'"
    ),
}

POLICY_FIELDS = [
    "policy_id",
    "layer",
    "source_path",
    "source_line",
    "source_qualname",
    "source_kind",
    "source_signature",
    "source_export_status",
    "destination_path",
    "destination_qualname",
    "destination_signature",
    "policy_status",
    "evidence",
    "rationale",
    "human_review",
    "review_reason",
    "signature_check",
]

REMOVAL_FIELDS = [
    "policy_id",
    "source_path",
    "source_line",
    "source_qualname",
    "source_signature",
    "destination_path",
    "removal_status",
    "replacement_identity",
    "approval_evidence",
    "rationale",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def policy_id(row: dict[str, str]) -> str:
    material = "|".join(
        (
            row["layer"],
            row["source_path"],
            row["line"],
            row["qualname"],
            row["kind"],
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()[:20]


def canonical_signature(signature: str) -> str:
    return (
        signature.replace("models.common.models.", "tt_transformers.models.")
        .replace("models.common.llm_runtime.", "tt_transformers.llm_runtime.")
        .replace("models.common.modules.", "tt_transformers.modules.")
    )


def imported_evidence(
    source_rows: list[dict[str, str]],
) -> tuple[dict[str, set[str]], Counter[tuple[str, str]]]:
    source_paths = {row["source_path"] for row in source_rows}
    module_to_path = {module_for(path): path for path in source_paths}
    imported: dict[str, set[str]] = defaultdict(set)
    sites: Counter[tuple[str, str]] = Counter()
    for row in read_csv(IMPORTS):
        if row["kind"] != "from" or row["imported_symbol"] in {"", "*"}:
            continue
        target_path = module_to_path.get(row["module"])
        if target_path is None:
            continue
        symbol = row["imported_symbol"]
        imported[target_path].add(symbol)
        sites[(target_path, symbol)] += 1
    return imported, sites


def is_review_ambiguity(row: dict[str, str], status: str) -> bool:
    if (
        status != "private_by_policy"
        or row["layer"] != "models"
        or row["export_status"] != "candidate_no___all__"
        or row["kind"] not in {"class", "function", "constant_or_value"}
    ):
        return False
    name = row["qualname"]
    return (
        name.endswith(("Generator", "GeneratorConfig", "GenerationConfig"))
        or (name.startswith("build_") and name.endswith("_generator"))
        or name in {"DEFAULT_HF_MODEL", "DEFAULT_HF_REVISION"}
    )


def extracted_rows(
    source_rows: list[dict[str, str]],
    provenance: dict[str, dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for source_path in sorted({row["source_path"] for row in source_rows}):
        destination = provenance[source_path]["destination_path"].split(";", 1)[0]
        path = ROOT / destination
        if not path.exists():
            raise AssertionError(f"missing extracted destination: {destination}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        layer = next(row["layer"] for row in source_rows if row["source_path"] == source_path)
        symbols = public_symbol_rows(destination, layer, tree)
        lazy_origins: dict[str, str] = {}
        for node in tree.body:
            if not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "_EXPORTS"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Dict)
            ):
                continue
            for key, value in zip(node.value.keys, node.value.values):
                if not (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and isinstance(value, ast.Tuple)
                    and len(value.elts) == 2
                    and all(
                        isinstance(item, ast.Constant) and isinstance(item.value, str)
                        for item in value.elts
                    )
                ):
                    continue
                lazy_origins[key.value] = f"{value.elts[0].value}.{value.elts[1].value}"
        for symbol in symbols:
            origin = lazy_origins.get(symbol["qualname"])
            if symbol["kind"] == "explicit_export" and origin:
                symbol["signature"] = f"lazy re-export of {origin}"
        result[destination] = symbols
    return result


def match_destination(
    source: dict[str, str],
    candidates: list[dict[str, str]],
) -> tuple[dict[str, str] | None, str]:
    identity = [
        row
        for row in candidates
        if row["qualname"] == source["qualname"]
        and row["kind"] == source["kind"]
        and row["decorators"] == source["decorators"]
    ]
    for row in identity:
        if row["signature"] == source["signature"]:
            return row, "exact"
    expected = canonical_signature(source["signature"])
    for row in identity:
        if row["signature"] == expected:
            return row, "namespace_rewrite"
    lazy_expected = expected.replace("re-export of ", "lazy re-export of ", 1)
    for row in identity:
        if row["signature"] == lazy_expected:
            return row, "lazy_export_rewrite"
    key = (source["source_path"], source["qualname"])
    if key in OPTIONAL_HF_REVISION_EXTENSIONS:
        for row in identity:
            without_revision = row["signature"].replace(", hf_revision: str | None=None", "", 1)
            if without_revision == expected:
                return row, "compatible_optional_hf_revision"
    if key in HF_REVISION_DEFAULT_NONE:
        for row in identity:
            restored_default = row["signature"].replace(
                "hf_revision: str | None=None",
                "hf_revision: str | None=DEFAULT_HF_REVISION",
                1,
            )
            if restored_default == expected:
                return row, "compatible_hf_revision_default"
    if key in IMMUTABLE_HF_REVISION_VALUES:
        for row in identity:
            if row["signature"] == IMMUTABLE_HF_REVISION_VALUES[key]:
                return row, "immutable_default_hf_revision"
    return None, ""


def removal_replacement(row: dict[str, str], destination: str) -> str:
    qualname = row["qualname"]
    if row["source_path"] in RELOCATED_NONPACKAGE:
        if qualname == "device_params":
            replacement = "resolve_device_params"
        elif qualname == "hf_model_id":
            replacement = "default_hf_model_id"
        elif qualname.startswith("test_"):
            replacement = "run_" + qualname[len("test_") :]
        else:
            replacement = "<module policy moved to CLI main/open_mesh_device>"
        return f"{destination.split(';', 1)[0]}::{replacement}"
    if qualname == "RotarySetup1D.get_rot_idxs":
        return (
            "src/tt_transformers/modules/rope/rope_1d.py::prepare_rot_idxs"
        )
    if qualname == "RotarySetup1D.get_rot_mats":
        return (
            "src/tt_transformers/modules/rope/rope_1d.py::prepare_rot_idxs + "
            "src/tt_transformers/modules/rope/rope_1d.py::RotarySetup1D.decode_forward"
        )
    if qualname == "Llama3Transformer1D.forward":
        return (
            f"{destination}::Llama3Transformer1D.prefill_forward + "
            f"{destination}::Llama3Transformer1D.decode_forward"
        )
    owner = qualname.rsplit(".", 1)[0]
    return f"{destination}::{owner}.__init__ + {destination}::{owner}.from_config"


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_rows = read_csv(SOURCE_SYMBOLS)
    provenance = {row["source_path"]: row for row in read_csv(PROVENANCE)}
    imported, import_sites = imported_evidence(source_rows)
    extracted = extracted_rows(source_rows, provenance)

    top_level_status: dict[tuple[str, str], str] = {}
    top_level_evidence: dict[tuple[str, str], str] = {}
    for row in source_rows:
        if row["export_status"] == "class_member":
            continue
        key = (row["source_path"], row["qualname"])
        if row["qualname"] in COMPATIBILITY_REMOVED:
            status = "compatibility_only"
            evidence = "approved compatibility-removal ledger"
        elif row["source_path"] in RELOCATED_NONPACKAGE:
            status = "private_by_policy"
            evidence = "hybrid pytest/demo production surface relocated outside installed package"
        elif row["export_status"] == "explicit" or row["kind"] == "explicit_export":
            status = "model_local_public" if row["layer"] == "models" else "supported_public"
            evidence = "explicit source __all__ export"
        elif row["qualname"] in imported[row["source_path"]]:
            status = "model_local_public" if row["layer"] == "models" else "supported_public"
            count = import_sites[(row["source_path"], row["qualname"])]
            evidence = f"imported by {count} in-scope production site(s)"
        else:
            status = "private_by_policy"
            evidence = "no explicit __all__ or cross-module production-import evidence"
        top_level_status[key] = status
        top_level_evidence[key] = evidence

    policy_rows: list[dict[str, str]] = []
    removal_rows: list[dict[str, str]] = []
    for source in source_rows:
        identifier = policy_id(source)
        destination = provenance[source["source_path"]]["destination_path"]
        approved_removal = (
            source["qualname"] in COMPATIBILITY_REMOVED
            or source["source_path"] in RELOCATED_NONPACKAGE
        )

        if source["qualname"] in COMPATIBILITY_REMOVED:
            status = "compatibility_only"
            evidence = "approved compatibility-removal ledger"
        elif source["source_path"] in RELOCATED_NONPACKAGE:
            status = "private_by_policy"
            evidence = "hybrid pytest/demo production surface relocated outside installed package"
        elif source["export_status"] == "class_member":
            owner = source["qualname"].split(".", 1)[0]
            status = top_level_status.get(
                (source["source_path"], owner), "private_by_policy"
            )
            evidence = (
                f"protocol member inherits {status} policy from "
                f"{source['source_path']}::{owner}"
            )
        else:
            status = top_level_status[(source["source_path"], source["qualname"])]
            evidence = top_level_evidence[(source["source_path"], source["qualname"])]

        if status == "supported_public":
            rationale = (
                "Reusable module/runtime construction or protocol surface; preserve "
                "its module-qualified signature without adding a root eager export."
            )
        elif status == "model_local_public":
            rationale = (
                "Concrete-model package surface; preserve its model-local identity "
                "and signature without adding a root eager export."
            )
        elif status == "compatibility_only":
            rationale = (
                "Legacy compatibility surface approved for removal; the replacement "
                "identity is recorded in public_api_removals.csv."
            )
        elif source["source_path"] in RELOCATED_NONPACKAGE:
            rationale = (
                "Pytest/demo orchestration is not installed package API; behavior "
                "moved to explicit example and hardware-test owners."
            )
        else:
            rationale = (
                "Public-looking source name lacks export/import evidence and is "
                "private by policy; access is incidental and carries no compatibility promise."
            )

        if approved_removal:
            destination_qualname = ""
            destination_signature = ""
            signature_check = "approved_removal"
            removal_status = (
                "compatibility_removed"
                if source["qualname"] in COMPATIBILITY_REMOVED
                else "relocated_nonpackage"
            )
            removal_rows.append(
                {
                    "policy_id": identifier,
                    "source_path": source["source_path"],
                    "source_line": source["line"],
                    "source_qualname": source["qualname"],
                    "source_signature": source["signature"],
                    "destination_path": destination,
                    "removal_status": removal_status,
                    "replacement_identity": removal_replacement(source, destination),
                    "approval_evidence": (
                        "qualification/extraction/foundation_boundary.md"
                        if removal_status == "compatibility_removed"
                        else "qualification/provenance/source_inventory.csv split disposition"
                    ),
                    "rationale": rationale,
                }
            )
        else:
            primary_destination = destination.split(";", 1)[0]
            matched, signature_check = match_destination(
                source, extracted[primary_destination]
            )
            if matched is None:
                raise AssertionError(
                    f"unapproved missing/signature-changed symbol: "
                    f"{source['source_path']}::{source['qualname']}"
                )
            destination_qualname = matched["qualname"]
            destination_signature = matched["signature"]

        review = is_review_ambiguity(source, status)
        policy_rows.append(
            {
                "policy_id": identifier,
                "layer": source["layer"],
                "source_path": source["source_path"],
                "source_line": source["line"],
                "source_qualname": source["qualname"],
                "source_kind": source["kind"],
                "source_signature": source["signature"],
                "source_export_status": source["export_status"],
                "destination_path": destination,
                "destination_qualname": destination_qualname,
                "destination_signature": destination_signature,
                "policy_status": status,
                "evidence": evidence,
                "rationale": rationale,
                "human_review": "yes" if review else "no",
                "review_reason": (
                    "Analogous generator/HF symbol is explicit or imported in some "
                    "model families but only an unexported candidate in this family."
                    if review
                    else ""
                ),
                "signature_check": signature_check,
            }
        )

    return policy_rows, removal_rows


def emit(rows: list[dict[str, str]], fields: list[str], start: int, end: int | None) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
    if start == 0:
        writer.writeheader()
    writer.writerows(rows[start:end])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def check() -> None:
    expected_policy, expected_removals = build_rows()
    actual_policy = read_csv(POLICY)
    actual_removals = read_csv(REMOVALS)

    assert len(expected_policy) == len(actual_policy) == 1760
    assert len({row["policy_id"] for row in actual_policy}) == 1760
    assert {row["policy_status"] for row in actual_policy} <= STATUSES
    assert actual_policy == expected_policy
    assert actual_removals == expected_removals

    source = read_csv(SOURCE_SYMBOLS)
    assert len(source) == len(actual_policy)
    assert all(row["policy_status"] for row in actual_policy)
    assert all(
        row["policy_status"] in {"model_local_public", "supported_public"}
        for row in actual_policy
        if row["source_export_status"] == "explicit"
    )

    root_tree = ast.parse(
        (ROOT / "src" / "tt_transformers" / "__init__.py").read_text(encoding="utf-8")
    )
    root_exports = {
        value.value
        for node in root_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
        and target.id == "__all__"
        for value in (
            node.value.elts
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set))
            else []
        )
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    assert root_exports == {"__version__"}

    print(f"verified {len(actual_policy)} unique policy rows")
    print("status", dict(sorted(Counter(row["policy_status"] for row in actual_policy).items())))
    print("signature", dict(sorted(Counter(row["signature_check"] for row in actual_policy).items())))
    print("removals", dict(sorted(Counter(row["removal_status"] for row in actual_removals).items())))
    print("human_review", sum(row["human_review"] == "yes" for row in actual_policy))
    print("root_exports", sorted(root_exports))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit", choices=("policy", "removals"), help="emit deterministic CSV"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--write", action="store_true", help="regenerate the checked policy CSV")
    args = parser.parse_args()
    policy_rows, removal_rows = build_rows()
    if args.write:
        if args.emit:
            parser.error("--write cannot be combined with --emit")
        write_csv(POLICY, policy_rows, POLICY_FIELDS)
        check()
    elif args.emit == "policy":
        emit(policy_rows, POLICY_FIELDS, args.start, args.end)
    elif args.emit == "removals":
        emit(removal_rows, REMOVAL_FIELDS, args.start, args.end)
    else:
        check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
