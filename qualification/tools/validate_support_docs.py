#!/usr/bin/env python3
"""Validate public support manifests and repository documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
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
)
REQUIRED_HEADINGS = (
    "### Purpose",
    "### Status and checkpoint",
    "### Candidate software tuple",
    "### Declared hardware geometry",
    "### Proven limits and features",
    "### Checkpoint, cache, and offline requirements",
    "### Install, run, and collect",
    "### Correctness criterion",
    "### Validation and evidence",
    "### Known and unsupported gaps",
)
REMOVED_PATH_FRAGMENTS = (
    "qualification/analysis/",
    "qualification/extraction/",
    "qualification/reports/",
    "docs/provenance/",
    "TTTV2_MIGRATION_PLAN.md",
    "TTTV2_MIGRATION_WORK_LOG.md",
    "SESSION_WORK_LOG.md",
)


def _local_markdown_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    links = []
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = target.strip("<>").split("#", 1)[0]
        if target and not target.startswith(("http://", "https://", "mailto:")):
            links.append(target)
    return links


def validate() -> list[str]:
    errors: list[str] = []
    schema_path = ROOT / "qualification/schemas/support-manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    manifest_paths = sorted((ROOT / "examples").glob("*/support.json"))
    if [path.parent.name for path in manifest_paths] != list(MODELS):
        errors.append("support manifests do not cover the exact model set")

    for path in manifest_paths:
        for filename in ("demo.py", "benchmark.py"):
            if not path.with_name(filename).is_file():
                errors.append(f"{path}: {filename} is missing")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for error in validator.iter_errors(manifest):
            location = "/".join(map(str, error.absolute_path)) or "<root>"
            errors.append(f"{path}:{location}: {error.message}")
        if manifest.get("status") != "experimental":
            errors.append(f"{path}: model must remain experimental")
        software = manifest.get("software", {})
        if software.get("tt_transformers") != "2.0.0.dev0":
            errors.append(f"{path}: wrong tt-transformers version")
        if software.get("ttnn") != "0.77.0":
            errors.append(f"{path}: wrong TTNN target")
        revision = manifest.get("model", {}).get("hf_revision", "")
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            errors.append(f"{path}: Hugging Face revision is not immutable")
        validation = manifest.get("validation", {})
        if validation.get("date") is not None or validation.get("git_sha") is not None:
            errors.append(f"{path}: experimental manifest claims a validation identity")
        if validation.get("evidence"):
            errors.append(f"{path}: experimental manifest claims promoted evidence")
        reference_root = manifest.get("assets", {}).get("reference_root")
        if not isinstance(reference_root, str) or not (ROOT / reference_root).is_dir():
            errors.append(f"{path}: missing reference_root {reference_root!r}")

        readme = path.with_name("README.md")
        if not readme.is_file():
            errors.append(f"{path}: README.md is missing")
            continue
        text = readme.read_text(encoding="utf-8")
        for heading in REQUIRED_HEADINGS:
            if heading not in text:
                errors.append(f"{readme}: missing heading {heading!r}")
        if "tt-transformers==2.0.0.dev0" not in text:
            errors.append(f"{readme}: package version is stale")
        if "models.common" in text or "models/common" in text:
            errors.append(f"{readme}: legacy namespace/path remains")

    required_docs = (
        ROOT / "README.md",
        ROOT / "SUPPORT.md",
        ROOT / "examples/README.md",
        ROOT / "docs/architecture.md",
        ROOT / "docs/validation.md",
        ROOT / "src/tt_transformers/README.md",
        ROOT / "tests/README.md",
        ROOT / "tools/README.md",
        ROOT / "qualification/README.md",
    )
    for path in required_docs:
        if not path.is_file():
            errors.append(f"required documentation is missing: {path}")

    for path in sorted(ROOT.rglob("*.md")):
        if any(
            part
            in {
                ".git",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
                ".venv",
                "build",
                "dist",
            }
            for part in path.parts
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(fragment in text for fragment in REMOVED_PATH_FRAGMENTS):
            errors.append(f"{path}: references a removed migration path")
        for target in _local_markdown_links(path):
            if not (path.parent / target).resolve().exists():
                errors.append(f"{path}: broken local link {target!r}")

    matrix_path = ROOT / "tests/hardware/hardware-matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    nodes = matrix.get("nodes", [])
    if len(nodes) != 51 or len({node.get("id") for node in nodes}) != 51:
        errors.append("hardware matrix must contain 51 unique nodes")
    if sum(node.get("mesh_device") == "P150" for node in nodes) != 12:
        errors.append("hardware matrix must retain twelve logical P150 nodes")

    for path in (
        ROOT / "TTTV2_MIGRATION_WORK_LOG.md",
        ROOT / "hardware-results",
        ROOT / "qualification/analysis",
        ROOT / "qualification/evidence",
        ROOT / "qualification/extraction",
        ROOT / "qualification/reports",
    ):
        if path.exists():
            errors.append(f"migration-only path remains: {path}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Validated {len(MODELS)} experimental support manifests and public documentation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
