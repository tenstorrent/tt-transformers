#!/usr/bin/env python3
"""Verify pinned foundation extraction and exact standalone adaptations."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path("/localdev/gwang/tt-metal")
SOURCE_REVISION = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
INVENTORY = ROOT / "qualification/provenance/source_inventory.csv"
CANONICAL_SAMPLING_SOURCES = {
    "models/common/sampling/__init__.py",
    "models/common/sampling/generator.py",
    "models/common/sampling/sampling_params.py",
}
ADAPTED_SAMPLING_SOURCES = {
    "models/common/modules/sampling/params.py": (
        "16dc0f49c838c881e435a62ec63bd45b7b09f8bd3485e79e75ea5eeb7592a593",
        (
            'if name == "seed":\n                return normalized',
            "return [normalized for _ in selected]",
        ),
        (),
    ),
    "models/common/modules/sampling/sampling_state_1d.py": (
        "5d46e8e4712ed37d8725f4bef83460a3efcde6aa065e63c83d48107efaf5f7ed",
        (
            "prefill sampling rows must be a request-ordered active prefix",
            "refresh_prefill_request_ordered(",
            "state.penalty_history_valid = not prepared.penalties_enabled and not survivor_slots",
        ),
        ("native device-sampled prefill currently requires exactly one active request",),
    ),
    "models/common/modules/sampling/seed_manager_1d.py": (
        "ff09347e2c58c8e77ad43e0ff24e25c49f5cfdbedfd09510ef9f74a77900128e",
        (
            "def refresh_prefill_request_ordered(",
            "slot_values = self._advance_seed_values",
            "request_values[row] = slot_values[destination]",
        ),
        (),
    ),
}
SOURCES = CANONICAL_SAMPLING_SOURCES | set(ADAPTED_SAMPLING_SOURCES)
PHASE4_TENSOR_UTILS_SOURCE = "models/common/tensor_utils.py"
PHASE4_TENSOR_UTILS_BLOB = "382e92e35d38fb80d60b80e2cd9206538b8fe417"
PHASE4_TENSOR_UTILS_SHA256 = "b74028873bd694498b274010f2003d532fda9de8a271e4a9633c53b72dc7e85c"
PHASE4_TENSOR_UTILS_DESTINATION = ROOT / "src/tt_transformers/tensor_utils.py"


@dataclass(frozen=True)
class InventoryRow:
    source_path: str
    destination_path: Path
    blob_sha: str


def git_output(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-C", str(SOURCE_REPO), *args],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def selected_rows() -> list[InventoryRow]:
    with INVENTORY.open(newline="", encoding="utf-8") as stream:
        rows = {row["source_path"]: row for row in csv.DictReader(stream) if row["source_path"] in SOURCES}
    if set(rows) != SOURCES:
        raise ValueError(f"missing foundation inventory rows: {SOURCES - set(rows)}")
    result = []
    for source_path in sorted(SOURCES):
        destination = rows[source_path]["destination_path"].split(";", 1)[0]
        result.append(
            InventoryRow(
                source_path=source_path,
                destination_path=ROOT / destination,
                blob_sha=rows[source_path]["git_blob_sha"],
            )
        )
    return result


def canonicalize_generator(text: str) -> str:
    tree = ast.parse(text)
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SamplingParams"]
    if len(matches) != 1:
        raise ValueError(f"expected one duplicate SamplingParams class, found {len(matches)}")
    node = matches[0]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list]) - 1
    end = node.end_lineno
    lines = text.splitlines(keepends=True)
    while end < len(lines) and not lines[end].strip():
        end += 1
    del lines[start:end]
    rewritten = "".join(lines)
    anchor = "from ._utils import clamp, is_default_value, split_list\n"
    if rewritten.count(anchor) != 1:
        raise ValueError("sampling generator import anchor drift")
    rewritten = rewritten.replace(
        anchor,
        anchor + "from .sampling_params import SamplingParams\n",
    )
    ast.parse(rewritten)
    return rewritten


def canonical_sampling_initializer(text: str) -> str:
    tree = ast.parse(text)
    export_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_EXPORTS" for target in node.targets)
    )
    if not isinstance(export_node.value, ast.Dict):
        raise ValueError("sampling _EXPORTS must be a literal dict")
    exports = []
    for key, value in zip(export_node.value.keys, export_node.value.values):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(value, ast.Tuple)
            and len(value.elts) == 2
            and all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts)
        ):
            raise ValueError("unsupported sampling export entry")
        module_name = value.elts[0].value
        attribute = value.elts[1].value
        if key.value == "SamplingParams":
            module_name = ".sampling_params"
        exports.append((key.value, module_name, attribute))

    lines = [
        "# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.",
        "# SPDX-License-Identifier: Apache-2.0",
        "",
        '"""Model-neutral sampling values and helpers.',
        "",
        "The lazy exports preserve the pinned helper surface while keeping one canonical",
        "SamplingParams identity in sampling_params.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from importlib import import_module",
        "",
        "_EXPORTS = {",
        *(f'    "{name}": ("{module}", "{attribute}"),' for name, module, attribute in exports),
        "}",
        "",
        "__all__ = list(_EXPORTS)",
        "",
        "",
        "def __getattr__(name: str):",
        "    try:",
        "        module_name, attribute = _EXPORTS[name]",
        "    except KeyError as error:",
        '        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error',
        "    value = getattr(import_module(module_name, __name__), attribute)",
        "    globals()[name] = value",
        "    return value",
        "",
        "",
        "def __dir__():",
        "    return sorted(set(globals()) | set(__all__))",
        "",
    ]
    rewritten = "\n".join(lines)
    ast.parse(rewritten)
    return rewritten


def expected_content(row: InventoryRow) -> bytes:
    if row.source_path not in CANONICAL_SAMPLING_SOURCES:
        raise ValueError(f"{row.source_path} is exact-hash verified, not mechanically rewritten")
    blob = str(git_output("rev-parse", f"{SOURCE_REVISION}:{row.source_path}")).strip()
    if blob != row.blob_sha:
        raise ValueError(f"blob mismatch for {row.source_path}: {blob} != {row.blob_sha}")
    text = bytes(git_output("cat-file", "blob", f"{SOURCE_REVISION}:{row.source_path}", binary=True)).decode("utf-8")
    if row.source_path.endswith("/generator.py"):
        text = canonicalize_generator(text)
    elif row.source_path.endswith("/__init__.py"):
        text = canonical_sampling_initializer(text)
    return text.encode("utf-8")


def verify_adapted_sampling(row: InventoryRow) -> list[str]:
    """Verify one intentional post-extraction sampling adaptation fail-closed."""

    failures = []
    blob = str(git_output("rev-parse", f"{SOURCE_REVISION}:{row.source_path}")).strip()
    if blob != row.blob_sha:
        failures.append(f"blob mismatch for {row.source_path}: {blob} != {row.blob_sha}")
        return failures
    if not row.destination_path.exists():
        failures.append(f"missing {row.destination_path.relative_to(ROOT)}")
        return failures

    expected_hash, required, forbidden = ADAPTED_SAMPLING_SOURCES[row.source_path]
    data = row.destination_path.read_bytes()
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != expected_hash:
        failures.append(
            f"intentional sampling adaptation drift {row.destination_path.relative_to(ROOT)} "
            f"(expected {expected_hash}, got {actual_hash})"
        )
        return failures
    text = data.decode("utf-8")
    for fragment in required:
        if fragment not in text:
            failures.append(f"missing sampling adaptation in {row.destination_path.relative_to(ROOT)}: {fragment!r}")
    for fragment in forbidden:
        if fragment in text:
            failures.append(f"retired sampling behavior in {row.destination_path.relative_to(ROOT)}: {fragment!r}")
    ast.parse(text, filename=str(row.destination_path))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite canonical sampling destinations; intentional module adaptations remain exact-hash verified",
    )
    args = parser.parse_args()
    failures = []
    for row in selected_rows():
        if row.source_path in ADAPTED_SAMPLING_SOURCES:
            failures.extend(verify_adapted_sampling(row))
            continue
        expected = expected_content(row)
        if args.write:
            row.destination_path.parent.mkdir(parents=True, exist_ok=True)
            row.destination_path.write_bytes(expected)
        if not row.destination_path.exists():
            failures.append(f"missing {row.destination_path.relative_to(ROOT)}")
        elif row.destination_path.read_bytes() != expected:
            failures.append(
                f"content mismatch {row.destination_path.relative_to(ROOT)} "
                f"(expected {hashlib.sha256(expected).hexdigest()}, "
                f"got {hashlib.sha256(row.destination_path.read_bytes()).hexdigest()})"
            )
    tensor_blob = str(git_output("rev-parse", f"{SOURCE_REVISION}:{PHASE4_TENSOR_UTILS_SOURCE}")).strip()
    if tensor_blob != PHASE4_TENSOR_UTILS_BLOB:
        failures.append(f"tensor_utils source blob mismatch: expected {PHASE4_TENSOR_UTILS_BLOB}, got {tensor_blob}")
    if not PHASE4_TENSOR_UTILS_DESTINATION.exists():
        failures.append(f"missing {PHASE4_TENSOR_UTILS_DESTINATION.relative_to(ROOT)}")
    else:
        tensor_hash = hashlib.sha256(PHASE4_TENSOR_UTILS_DESTINATION.read_bytes()).hexdigest()
        if tensor_hash != PHASE4_TENSOR_UTILS_SHA256:
            failures.append(f"Phase 4 tensor_utils drift: expected {PHASE4_TENSOR_UTILS_SHA256}, got {tensor_hash}")
    if failures:
        print("\n".join(failures))
        return 1
    print(
        "verified 3 pinned canonical-sampling files, 3 exact-hash standalone sampling adaptations, "
        "and Phase 4 tensor_utils normalization"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
