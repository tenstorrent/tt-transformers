# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A model that needs one fabric refuses a mesh opened with another."""

from types import SimpleNamespace

import pytest
import ttnn

from tt_transformers import device_utils
from tt_transformers.device_utils import require_fabric_config


def _mesh(num_devices: int) -> SimpleNamespace:
    return SimpleNamespace(get_num_devices=lambda: num_devices)


@pytest.mark.host
def test_ring_model_accepts_a_mesh_opened_with_a_ring(monkeypatch):
    monkeypatch.setattr(device_utils.ttnn, "get_fabric_config", lambda: ttnn.FabricConfig.FABRIC_1D_RING)

    require_fabric_config(_mesh(8), ttnn.FabricConfig.FABRIC_1D_RING, model="org/ring-model")


@pytest.mark.host
@pytest.mark.parametrize("active", [ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.DISABLED])
def test_ring_model_rejects_a_mesh_opened_with_another_fabric(monkeypatch, active):
    monkeypatch.setattr(device_utils.ttnn, "get_fabric_config", lambda: active)

    with pytest.raises(RuntimeError, match=rf"org/ring-model needs fabric FABRIC_1D_RING .* opened with {active.name}"):
        require_fabric_config(_mesh(8), ttnn.FabricConfig.FABRIC_1D_RING, model="org/ring-model")


@pytest.mark.host
def test_single_device_mesh_needs_no_fabric(monkeypatch):
    def unexpected():
        raise AssertionError("a single-device mesh has no fabric to check")

    monkeypatch.setattr(device_utils.ttnn, "get_fabric_config", unexpected)

    require_fabric_config(_mesh(1), ttnn.FabricConfig.FABRIC_1D_RING, model="org/ring-model")
