# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Deterministic host gate for example support documentation."""

import pytest

from qualification.tools.validate_support_docs import validate


@pytest.mark.host
def test_support_documentation_and_manifests_are_consistent():
    errors = validate()
    assert not errors, "\n".join(errors)
