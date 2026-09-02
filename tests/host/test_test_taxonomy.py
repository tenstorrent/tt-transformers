"""Host gate for the explicit pytest marker taxonomy."""

import pytest

from qualification.tools.audit_test_taxonomy import audit


@pytest.mark.host
def test_explicit_test_taxonomy_is_complete_and_consistent():
    result = audit()
    assert not result["errors"], "\n".join(result["errors"])
    assert result["counts"]["host"] + result["counts"]["device"] == result["functions"]
    assert result["counts"]["device"] > 0
    assert result["counts"]["model"] > 0
