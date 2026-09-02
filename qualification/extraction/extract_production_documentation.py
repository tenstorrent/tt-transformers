#!/usr/bin/env python3
"""Extract and verify the five pinned production documentation destinations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPOSITORY = Path("/localdev/gwang/tt-metal")
SOURCE_REVISION = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
MANIFEST = ROOT / "qualification/extraction/production_documentation_manifest.json"
FILES = {
    "models/common/llm_runtime/prefill/README.md": (
        "176e4e5a4f517aa3760bea30c85333d03f812e8e",
        "src/tt_transformers/llm_runtime/prefill/README.md",
    ),
    "models/common/llm_runtime/README.md": (
        "976f63eb3605f48745378343fa93ceeaeec0b966",
        "src/tt_transformers/llm_runtime/README.md",
    ),
    "models/common/llm_runtime/tttv2_demo_to_vllm_integration_skill.md": (
        "b643ace0dd9fec9b2911cdb1b775d7d5e0512ec1",
        "src/tt_transformers/llm_runtime/tttv2_demo_to_vllm_integration_skill.md",
    ),
    "models/common/modules/README.md": (
        "117f5daf847f465091a1acc993c6f93fbfb404bd",
        "src/tt_transformers/modules/README.md",
    ),
    "models/common/sampling/README.md": (
        "2f4fbc45a16324817ac8e8bd5a7213bb17c9cb46",
        "src/tt_transformers/sampling/README.md",
    ),
}


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(SOURCE_REPOSITORY), *args],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def rewrite(text: str) -> tuple[str, int]:
    replacements = (
        (
            "/localdev/gwang/vllm_duo/tt-metal-too/models/common/tests/llm_runtime/test_executor_integration.py",
            "tests/llm_runtime/test_executor_integration.py",
        ),
        ("/localdev/gwang/vllm_duo/tt-metal-too/python_env", "<tt-metal-checkout>/python_env"),
        ("/localdev/gwang/vllm_duo/tt-metal-too", "<tt-metal-checkout>"),
        ("/localdev/gwang/vllm_duo/vllm", "<vllm-checkout>"),
        ("models/common/models/llama3_8b/README.md", "examples/llama3_8b/README.md"),
        ("models/common/tests/llm_runtime", "tests/llm_runtime"),
        ("models/common/tests/modules", "tests/modules"),
        ("models/common/tests/demos", "tests/hardware/models"),
        ("models/common/llm_runtime", "src/tt_transformers/llm_runtime"),
        ("models/common/modules", "src/tt_transformers/modules"),
        ("models/common/models", "src/tt_transformers/models"),
        ("models/common/sampling", "src/tt_transformers/sampling"),
        ("models.common.llm_runtime", "tt_transformers.llm_runtime"),
        ("models.common.modules", "tt_transformers.modules"),
        ("models.common.models", "tt_transformers.models"),
        ("models.common.sampling", "tt_transformers.sampling"),
        ("models.tt_transformers.tt.common", "tt_transformers.modules.mode"),
    )
    count = 0
    for old, new in replacements:
        matches = text.count(old)
        text = text.replace(old, new)
        count += matches
    return text, count


def expected() -> tuple[dict[str, str], list[dict[str, object]]]:
    outputs = {}
    records = []
    for source_path, (blob_sha, destination_path) in FILES.items():
        actual_blob = git("rev-parse", f"{SOURCE_REVISION}:{source_path}").decode().strip()
        if actual_blob != blob_sha:
            raise RuntimeError(f"{source_path}: expected blob {blob_sha}, got {actual_blob}")
        raw = git("show", f"{SOURCE_REVISION}:{source_path}")
        output, rewrite_count = rewrite(raw.decode("utf-8"))
        outputs[destination_path] = output
        records.append(
            {
                "source_path": source_path,
                "source_blob_sha": blob_sha,
                "destination_path": destination_path,
                "source_size": len(raw),
                "destination_size": len(output.encode("utf-8")),
                "destination_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "path_rewrite_count": rewrite_count,
            }
        )
    return outputs, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    outputs, records = expected()
    payload = {
        "schema_version": 1,
        "source_revision": SOURCE_REVISION,
        "files": records,
    }
    mismatches = []
    for destination_path, output in outputs.items():
        path = ROOT / destination_path
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output, encoding="utf-8")
        if not path.is_file() or path.read_text(encoding="utf-8") != output:
            mismatches.append(destination_path)
    expected_manifest = json.dumps(payload, indent=2) + "\n"
    if args.write:
        MANIFEST.write_text(expected_manifest, encoding="utf-8")
    if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != expected_manifest:
        mismatches.append(str(MANIFEST.relative_to(ROOT)))
    if mismatches:
        print("documentation extraction mismatch: " + ", ".join(mismatches))
        return 1
    print(f"verified {len(records)} pinned documentation destinations and manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
