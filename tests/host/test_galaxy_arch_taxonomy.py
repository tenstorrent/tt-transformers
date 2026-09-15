# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host gates for the per-architecture Galaxy taxonomy.

Galaxy is a *topology*, and both architectures ship one. The taxonomy treated
`galaxy` as an arch-implying SKU pinned to Wormhole, which made a Blackhole
Galaxy test unexpressible: `audit_test_taxonomy` enforces SKU => arch, so
`galaxy` + `blackhole` failed the gate outright. These tests pin the split that
fixes it, and the three plumbing sites that have to move with it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from tests.support import fixture_policy
from tests.support.marker_policy import selected_topology_marks

from qualification.tools.audit_test_taxonomy import ROOT, SKU_ARCH, TAXONOMY, configured_markers, marks

TESTS = ROOT / "tests"


@pytest.mark.host
def test_galaxy_marker_is_split_per_architecture_and_registered():
    """Both Galaxy markers exist, carry their own arch, and are declared."""

    assert SKU_ARCH["galaxy_wh"] == "wormhole"
    assert SKU_ARCH["galaxy_bh"] == "blackhole"
    # The collapsed marker must be gone, not merely joined: leaving it in place
    # would keep `galaxy` implying Wormhole for anything that had not re-marked.
    assert "galaxy" not in SKU_ARCH
    assert "galaxy" not in TAXONOMY
    assert {"galaxy_wh", "galaxy_bh"} <= TAXONOMY
    assert {"galaxy_wh", "galaxy_bh"} <= configured_markers()


@pytest.mark.host
def test_no_test_still_carries_the_collapsed_galaxy_marker():
    """No suite in the tree uses the pre-split marker.

    A leftover `@pytest.mark.galaxy` is not a stale name: `galaxy` is no longer
    in the marker registry, so it would be an unregistered mark that selects
    nothing, and `-m galaxy_wh` would silently skip that suite.
    """

    offenders: list[str] = []
    for path in sorted(TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if "galaxy" in marks(node):
                offenders.append(f"{Path(path).relative_to(ROOT).as_posix()}:{node.lineno}:{node.name}")
    assert not offenders, "re-mark these as galaxy_wh or galaxy_bh:\n" + "\n".join(offenders)


@pytest.mark.host
def test_marker_policy_maps_both_galaxy_mesh_names(monkeypatch):
    """`MESH_DEVICE` names the topology, so it must name the architecture too."""

    def names(mesh_device: str) -> list[str]:
        monkeypatch.setenv("MESH_DEVICE", mesh_device)
        return [mark.name for mark in selected_topology_marks(pytest)]

    assert names("TG") == ["wormhole", "galaxy_wh"]
    # `BHGLX` is the name `device_utils.get_device_name` already returns for a
    # 32-device Blackhole mesh; this extends that convention rather than adding one.
    assert names("BHGLX") == ["blackhole", "galaxy_bh"]
    assert names("bhglx") == ["blackhole", "galaxy_bh"]
    assert names("") == []


@pytest.mark.host
def test_logical_sku_distinguishes_the_two_galaxies(monkeypatch):
    """A 32-device mesh is not necessarily `TG`.

    Keying on device count alone collapsed Blackhole Galaxy onto the Wormhole
    entry, which is the same failure the reference port hit and fixed the same
    way. It also reported a Blackhole SKU (`P150x4`) for a four-device Wormhole
    mesh, which predates Galaxy entirely.
    """

    monkeypatch.setattr(fixture_policy, "_is_blackhole_cluster", lambda: False)
    assert fixture_policy.get_logical_sku(None, 32) == "TG"
    assert fixture_policy.get_logical_sku(None, 8) == "T3K"
    assert fixture_policy.get_logical_sku(None, 4) == "N150x4"

    monkeypatch.setattr(fixture_policy, "_is_blackhole_cluster", lambda: True)
    assert fixture_policy.get_logical_sku(None, 32) == "BHGLX"
    assert fixture_policy.get_logical_sku(None, 4) == "P150x4"


@pytest.mark.host
def test_logical_sku_accepts_what_its_callers_actually_pass(monkeypatch):
    """The argument is a parametrization, not an open `MeshDevice`.

    `tests/conftest.py` resolves `trace_region_size` -- a *device-open*
    parameter -- from `request.param`, which is an `int` or a `(rows, columns)`
    tuple. The previous body called `.get_num_devices()` on it, which raises
    `AttributeError` for both. Nothing caught it because both call paths are
    currently unreachable: no suite sets `TRACE_MODEL_KEY_PARAM` or
    `trace_model_key`. Registering a Blackhole Galaxy trace node makes them
    reachable, so pin the contract now.
    """

    monkeypatch.setattr(fixture_policy, "_is_blackhole_cluster", lambda: False)
    assert fixture_policy.get_logical_sku(None, (8, 4)) == "TG"
    assert fixture_policy.get_logical_sku(None, 32) == "TG"

    class _Mesh:
        def get_num_devices(self):
            return 32

    assert fixture_policy.get_logical_sku(None, _Mesh()) == "TG"
    # An unrecognised parametrization yields no SKU rather than a wrong one; the
    # fixture logs and leaves `trace_region_size` unset.
    assert fixture_policy.get_logical_sku(None, None) is None


@pytest.mark.host
def test_blackhole_galaxy_sku_token_resolves_to_its_own_targets_entry():
    """`BHGLX` must not normalize onto the Wormhole Galaxy SKU.

    The SKU token returned above is fed to `resolve_trace_region_size`, which
    normalizes through `_SKU_ALIASES`. If `BHGLX` had no entry it would fall
    through as an unknown token and silently resolve no trace region at all.
    """

    from examples.common.model_targets import normalize_sku

    assert normalize_sku("BHGLX") == "bh_galaxy_perf"
    assert normalize_sku("TG") == "wh_galaxy_perf"
