from __future__ import annotations

import pytest

import json
from pathlib import Path

import tt_transformers


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.host
def test_base_package_import_has_a_version() -> None:
    assert tt_transformers.__version__


@pytest.mark.host
def test_support_manifest_schema_is_valid_json() -> None:
    schema_path = ROOT / "qualification/schemas/support-manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["status"]["enum"] == [
        "qualified",
        "experimental",
        "deprecated",
        "legacy",
    ]
