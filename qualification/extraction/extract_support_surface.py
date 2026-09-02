#!/usr/bin/env python3
"""Mechanically extract the non-production TTTv2 surface from a pinned commit.

The provenance CSV is authoritative.  This script writes only tests/, examples/,
and qualification/ destinations.  Hybrid/split sources are copied whole in
Phase 2; semantic separation and filtering remain explicitly deferred.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = Path("/localdev/gwang/tt-metal")
PINNED_SHA = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
INVENTORY = ROOT / "qualification/provenance/source_inventory.csv"
MANIFEST = ROOT / "qualification/extraction/support_copy_manifest.csv"
SUPPORT_ASSETS = ROOT / "qualification/analysis/support/support_asset_inventory.csv"
UNASSIGNED = ROOT / "qualification/extraction/support_unassigned.csv"
TARGET_ROOTS = {"tests", "examples", "qualification"}
BINARY_SUFFIXES = {".bz2", ".jpeg", ".jpg", ".png", ".pptx", ".refpt", ".webp"}


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), *args], text=True).strip()


def source_blob(blob_sha: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), "cat-file", "blob", blob_sha])


def source_git_mode(source_path: str) -> str:
    record = git_text("ls-tree", PINNED_SHA, source_path)
    mode, _kind, _blob_and_path = record.split(maxsplit=2)
    return mode


def module_name(path: str, *, strip_src: bool = False) -> str | None:
    if not path.endswith(".py"):
        return None
    value = path[:-3]
    if strip_src and value.startswith("src/"):
        value = value[4:]
    if value.endswith("/__init__"):
        value = value[: -len("/__init__")]
    return value.replace("/", ".")


def target_destinations(row: dict[str, str]) -> list[str]:
    return [
        destination
        for destination in row["destination_path"].split(";")
        if destination and destination.split("/", 1)[0] in TARGET_ROOTS
    ]


def load_rows() -> list[dict[str, str]]:
    with INVENTORY.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 366
    selected = [row for row in rows if row["disposition"] != "excluded" and target_destinations(row)]
    assert len(selected) == 206
    assert sum(len(target_destinations(row)) for row in selected) == 223
    return rows


def build_import_map(rows: list[dict[str, str]]) -> dict[str, str]:
    mapping = {
        "models.common.models": "tt_transformers.models",
        "models.common.llm_runtime": "tt_transformers.llm_runtime",
        "models.common.modules": "tt_transformers.modules",
        "models.common.sampling": "tt_transformers.sampling",
        "models.common.tests.demos.cleanup_utils": "tests.integration.cleanup_utils",
        "models.common.tests.demos.run_helpers": "tests.integration.run_helpers",
        "models.common.tests.demos.llama3_8b.demo_utils": "examples.llama3_8b.demo_utils",
        "models.common.tests.demos.llama3_8b": "examples.llama3_8b",
        "models.common.tests.demos": "tests.integration",
        "models.common.tests.conftest": "tests.conftest",
        "models.common.tests.utils": "tests.support.helpers",
        "models.common.utility_functions": "tests.support.comparison",
        "models.tt_transformers.tt.generator": "tt_transformers.mesh_utils",
        "tests.tests_common.cache_entries_counter": "tests.support.cache_entries_counter",
    }
    for row in rows:
        # These sources are explicitly split by symbol/ownership.  Only the
        # narrow mappings above are safe before Phase 3 performs that split.
        if row["category"] in {"boundary_source", "boundary_support"}:
            continue
        old = module_name(row["source_path"])
        if old is None:
            continue
        destinations = row["destination_path"].split(";")
        preferred = next((value for value in destinations if value.startswith("src/tt_transformers/")), None)
        if preferred is None:
            preferred = next((value for value in destinations if value.startswith("qualification/tools/")), None)
        if preferred is None:
            preferred = next(
                (
                    value
                    for value in destinations
                    if value.startswith("tests/") and Path(value).name != "conftest.py"
                ),
                None,
            )
        if preferred is None:
            preferred = next((value for value in destinations if value.startswith("examples/")), None)
        new = module_name(preferred, strip_src=True) if preferred else None
        if new:
            mapping.setdefault(old, new)
    return mapping


def remap_module(name: str, mapping: dict[str, str], destination: str) -> str:
    contextual = {
        "models.common.tests.demos.llama3_8b.demo_utils": (
            "tests.models.llama3_8b.demo_utils"
            if destination.startswith("tests/")
            else "examples.llama3_8b.demo_utils"
        )
    }
    if name in contextual:
        return contextual[name]
    for old in sorted(mapping, key=len, reverse=True):
        if name == old or name.startswith(old + "."):
            return mapping[old] + name[len(old) :]
    return name


def rewrite_python_imports(data: bytes, destination: str, mapping: dict[str, str]) -> tuple[bytes, int]:
    text = data.decode("utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return data, 0
    lines = text.splitlines(keepends=True)
    offsets = []
    total = 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "models.tt_transformers.tt.common" and {alias.name for alias in node.names} == {
                "Mode"
            }:
                replacement = "tt_transformers.modules.mode"
            else:
                replacement = remap_module(node.module, mapping, destination)
            if replacement == node.module:
                continue
            line = lines[node.lineno - 1]
            match = re.search(r"\bfrom\s+" + re.escape(node.module) + r"\s+import\b", line[node.col_offset :])
            if match is None:
                continue
            start = offsets[node.lineno - 1] + node.col_offset + match.start() + match.group(0).index(node.module)
            edits.append((start, start + len(node.module), replacement))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                replacement = remap_module(alias.name, mapping, destination)
                if replacement == alias.name or not hasattr(alias, "lineno"):
                    continue
                start = offsets[alias.lineno - 1] + alias.col_offset
                edits.append((start, start + len(alias.name), replacement))
    for start, end, replacement in sorted(set(edits), reverse=True):
        text = text[:start] + replacement + text[end:]
    return text.encode("utf-8"), len(set(edits))


def existing_owned_paths() -> set[str]:
    if not MANIFEST.exists():
        return set()
    with MANIFEST.open(newline="") as stream:
        return {row["destination_path"] for row in csv.DictReader(stream)}


def write_file(destination: str, data: bytes, allowed_existing: set[str], *, executable: bool) -> None:
    path = ROOT / destination
    if path.exists() and destination not in allowed_existing:
        raise RuntimeError(f"refusing to overwrite pre-existing destination: {destination}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o755 if executable else 0o644)


def main() -> None:
    boundary_manifest = ROOT / "qualification/extraction/support_boundary_manifest.json"
    if boundary_manifest.exists() and os.getenv("TT_TRANSFORMERS_REEXTRACT_PHASE2") != "1":
        raise SystemExit(
            "Phase 3 support boundary is already closed; refusing to overwrite it. "
            "Remove the boundary manifest or set TT_TRANSFORMERS_REEXTRACT_PHASE2=1 for an intentional rebuild."
        )
    assert git_text("rev-parse", PINNED_SHA) == PINNED_SHA
    rows = load_rows()
    import_map = build_import_map(rows)
    selected = [row for row in rows if row["disposition"] != "excluded" and target_destinations(row)]
    allowed_existing = existing_owned_paths()
    by_destination: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        actual = git_text("rev-parse", f"{PINNED_SHA}:{row['source_path']}")
        if actual != row["git_blob_sha"]:
            raise RuntimeError(f"blob mismatch for {row['source_path']}: {actual}")
        for destination in target_destinations(row):
            by_destination[destination].append(row)

    # Existing scaffold content is intentionally outside the extraction target.
    assert "examples/README.md" not in by_destination
    assert "qualification/schemas/support-manifest.schema.json" not in by_destination
    assert len(by_destination) == 220

    manifest_rows: list[dict[str, str]] = []
    for destination, source_rows in sorted(by_destination.items()):
        if destination == "tests/conftest.py":
            order = {"conftest.py": 0, "models/conftest.py": 1, "models/common/tests/conftest.py": 2}
            chunks = []
            import_edits = 0
            for row in sorted(source_rows, key=lambda item: order[item["source_path"]]):
                raw = source_blob(row["git_blob_sha"])
                rewritten, count = rewrite_python_imports(raw, destination, import_map)
                import_edits += count
                chunks.append(
                    f"\n# --- Pinned source segment: {row['source_path']} @ {row['git_blob_sha']} ---\n".encode()
                    + rewritten.rstrip()
                    + b"\n"
                )
            output = b"".join(chunks).lstrip(b"\n")
            destination_executable = False
            transformation = f"merged_fixture_snapshots;mechanical_import_rewrites={import_edits};phase3_fixture_reduction_deferred"
        else:
            blobs = {row["git_blob_sha"] for row in source_rows}
            if len(blobs) != 1:
                raise RuntimeError(f"destination collision with different blobs: {destination}")
            raw = source_blob(next(iter(blobs)))
            import_edits = 0
            if Path(destination).suffix == ".py":
                output, import_edits = rewrite_python_imports(raw, destination, import_map)
            else:
                output = raw
            split = any(row["disposition"] == "split" for row in source_rows)
            if split:
                transformation = "whole_pinned_snapshot;phase3_semantic_split_or_filter_deferred"
            else:
                transformation = "content_preserving_path_move"
            if import_edits:
                transformation += f";mechanical_import_rewrites={import_edits}"
            modes = {source_git_mode(row["source_path"]) for row in source_rows}
            if len(modes) != 1:
                raise RuntimeError(f"destination collision with different source modes: {destination}")
            destination_executable = modes == {"100755"}
        write_file(destination, output, allowed_existing, executable=destination_executable)
        output_sha256 = hashlib.sha256(output).hexdigest()
        for row in source_rows:
            raw = source_blob(row["git_blob_sha"])
            manifest_rows.append(
                {
                    "source_path": row["source_path"],
                    "source_blob_sha": row["git_blob_sha"],
                    "source_git_mode": source_git_mode(row["source_path"]),
                    "category": row["category"],
                    "disposition": row["disposition"],
                    "destination_path": destination,
                    "destination_sha256": output_sha256,
                    "destination_mode": "0755" if destination_executable else "0644",
                    "raw_blob_size": str(len(raw)),
                    "destination_size": str(len(output)),
                    "exact_raw_copy": str(output == raw).lower(),
                    "transformation": transformation,
                    "boundary_cleanup": row["boundary_cleanup"],
                }
            )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_path",
        "source_blob_sha",
        "source_git_mode",
        "category",
        "disposition",
        "destination_path",
        "destination_sha256",
        "destination_mode",
        "raw_blob_size",
        "destination_size",
        "exact_raw_copy",
        "transformation",
        "boundary_cleanup",
    ]
    with MANIFEST.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    inventory_paths = {row["source_path"] for row in rows}
    with SUPPORT_ASSETS.open(newline="") as stream:
        support_assets = list(csv.DictReader(stream))
    unassigned_rows = []
    for asset in support_assets:
        if asset["path"] in inventory_paths:
            continue
        actual = git_text("rev-parse", f"{PINNED_SHA}:{asset['path']}")
        if actual != asset["blob_sha"]:
            raise RuntimeError(f"support-inventory blob mismatch for {asset['path']}")
        unassigned_rows.append(
            {
                "source_path": asset["path"],
                "source_blob_sha": asset["blob_sha"],
                "role": asset["role"],
                "reason": "support inventory has no provenance row or assigned destination",
            }
        )
    assert len(unassigned_rows) == 0
    with UNASSIGNED.open("w", newline="") as stream:
        fields = ["source_path", "source_blob_sha", "role", "reason"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(unassigned_rows)
    print(f"extracted {len(selected)} source rows into {len(by_destination)} destinations")
    print(f"wrote {len(manifest_rows)} row/destination verification records")
    print(f"recorded {len(unassigned_rows)} support-inventory assets without assigned destinations")


if __name__ == "__main__":
    main()
