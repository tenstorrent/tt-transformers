#!/usr/bin/env python3
"""Validate experimental example manifests and generated support documentation."""

from __future__ import annotations

import argparse
import json
import re
import sys
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
PINNED_SHA = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
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


def validate() -> list[str]:
    errors: list[str] = []
    schema_path = ROOT / "qualification/schemas/support-manifest.schema.json"
    schema = json.loads(schema_path.read_text())
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        errors.append(f"support schema is invalid: {error}")
        return errors
    validator = Draft202012Validator(schema)

    manifests = {}
    paths = sorted((ROOT / "examples").glob("*/support.json"))
    found = {path.parent.name for path in paths}
    if found != set(MODELS):
        errors.append(f"support manifest packages differ: expected={sorted(MODELS)} found={sorted(found)}")

    for path in paths:
        model = path.parent.name
        manifest = json.loads(path.read_text())
        manifests[model] = manifest
        for error in validator.iter_errors(manifest):
            errors.append(f"{path}:{'/'.join(map(str, error.absolute_path))}: {error.message}")
        if manifest.get("status") != "experimental":
            errors.append(f"{path}: active extraction status must remain experimental")
        validation = manifest.get("validation")
        if validation != {"date": None, "git_sha": None, "evidence": []}:
            errors.append(f"{path}: unsupported validation/evidence claim: {validation}")
        revision = manifest.get("model", {}).get("hf_revision")
        if revision == "UNPINNED":
            if "HF revision is not pinned" not in manifest.get("known_gaps", []):
                errors.append(f"{path}: UNPINNED revision lacks explicit known gap")
        elif not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            errors.append(f"{path}: revision must be UNPINNED or a 40-character lowercase SHA")
        if manifest.get("software") != {
            "tt_transformers": "0.1.0.dev0",
            "ttnn": "0.77.0",
            "python": ["3.10", "3.12"],
            "torch": "2.11.0",
            "transformers": "5.12.1",
        }:
            errors.append(f"{path}: candidate software tuple drifted")
        if not manifest.get("hardware"):
            errors.append(f"{path}: no declared candidate hardware rows")
        assets = manifest.get("assets", {})
        if assets.get("remote_code_default") is not False:
            errors.append(f"{path}: remote-code default must be false")
        if assets.get("cwd_relative_cache_default") is not False:
            errors.append(f"{path}: cwd-relative cache default must be false")
        if assets.get("cache_resolver") != "tt_transformers.cache_environment.resolve_model_cache_path":
            errors.append(f"{path}: standalone cache resolver is missing")
        if assets.get("tt_cache_path_semantics") != "append topology exactly once":
            errors.append(f"{path}: TT_CACHE_PATH topology semantics drifted")

        readme_path = path.with_name("README.md")
        readme = readme_path.read_text()
        if readme.count("<!-- BEGIN GENERATED SUPPORT -->") != 1 or readme.count(
            "<!-- END GENERATED SUPPORT -->"
        ) != 1:
            errors.append(f"{readme_path}: generated support markers must occur exactly once")
        for heading in REQUIRED_HEADINGS:
            if heading not in readme:
                errors.append(f"{readme_path}: missing {heading}")
        required_text = (
            manifest["model"]["hf_id"],
            "experimental",
            "concrete and runnable",
            "ttnn==0.77.0",
            "torch==2.11.0",
            "transformers==5.12.1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "TT_TRANSFORMERS_CACHE",
            "trust_remote_code` is not enabled by default",
            f"python -m examples.{model}.demo",
            f"tests/hardware/models/{model}/test_demo.py",
            "Correctness criterion",
            f"pinned source revision `{PINNED_SHA}`",
            "Last standalone hardware validation date: **none**",
        )
        for token in required_text:
            if token not in readme:
                errors.append(f"{readme_path}: missing required token {token!r}")
        if revision == "UNPINNED" and "UNPINNED — release blocker" not in readme:
            errors.append(f"{readme_path}: unpinned blocker is not prominent")
        if revision != "UNPINNED" and revision not in readme:
            errors.append(f"{readme_path}: pinned revision missing")
        for hardware in manifest.get("hardware", []):
            row_tokens = (
                hardware["architecture"],
                hardware["sku"],
                hardware["mesh"],
                str(hardware["tp"]),
                str(hardware["dp"]),
            )
            if not all(token in readme for token in row_tokens):
                errors.append(f"{readme_path}: hardware row is not rendered: {hardware}")
        for limit in manifest.get("limits", {}).get("source_proven", []):
            if limit not in readme:
                errors.append(f"{readme_path}: source-proven limit missing: {limit}")
        for target in (
            path,
            ROOT / "tests/hardware/models" / model / "test_demo.py",
            ROOT / "qualification/analysis/support/support_baseline.md",
            ROOT / "qualification/analysis/support/hardware_evidence.csv",
        ):
            if not target.exists():
                errors.append(f"{readme_path}: linked target missing: {target}")

    root_matrix = (ROOT / "SUPPORT.md").read_text()
    example_matrix = (ROOT / "examples/README.md").read_text()
    for model in MODELS:
        manifest = manifests.get(model)
        if not manifest:
            continue
        for matrix_name, matrix in (("SUPPORT.md", root_matrix), ("examples/README.md", example_matrix)):
            if matrix.count(f"[{model}]") != 1:
                errors.append(f"{matrix_name}: {model} row must occur exactly once")
            if manifest["model"]["hf_id"] not in matrix:
                errors.append(f"{matrix_name}: missing HF ID for {model}")
        if "none at pinned SHA" not in root_matrix or "none at pinned SHA" not in example_matrix:
            errors.append("support matrices must state evidence absence explicitly")
    if "| qualified |" in root_matrix.lower() or "| qualified |" in example_matrix.lower():
        errors.append("support matrix contains an unsupported qualified row")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    errors = validate()
    if errors:
        print("Support documentation validation FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Validated 12 experimental support manifests, READMEs, and both support matrices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
