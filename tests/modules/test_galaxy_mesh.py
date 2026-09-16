# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The module-layer Galaxy mesh gate, which every 2D module now shares."""

from unittest.mock import MagicMock

import pytest
import ttnn

from tt_transformers.modules.galaxy_mesh import (
    GALAXY_ARCHITECTURES,
    GALAXY_DEVICE_COUNT,
    GALAXY_MESH_SHAPE,
    is_galaxy_architecture_name,
    require_galaxy_mesh,
)


def _mesh(shape=GALAXY_MESH_SHAPE, devices=GALAXY_DEVICE_COUNT, arch=ttnn.device.Arch.WORMHOLE_B0):
    mesh = MagicMock()
    mesh.shape = shape
    mesh.get_num_devices.return_value = devices
    mesh.arch.return_value = arch
    return mesh


@pytest.mark.host
@pytest.mark.parametrize("arch", sorted(GALAXY_ARCHITECTURES, key=str), ids=str)
def test_both_galaxy_architectures_pass_the_gate(arch):
    """Wormhole and Blackhole Galaxy are the same mesh, and the gate says so.

    Blackhole Galaxy declares the same `device_topology { dims: [8, 4] }`, so a
    gate that rejected it by name rejected a mesh it could describe.
    """

    assert require_galaxy_mesh("Module2D", _mesh(arch=arch)) == GALAXY_MESH_SHAPE


@pytest.mark.host
def test_gate_refuses_an_architecture_with_no_galaxy(expect_error):
    """The allowlist is still an allowlist.

    `QUASAR` is the negative control throughout this branch: it is a real
    member of the pinned `ttnn`'s `Arch` enum -- naming one that does not exist
    would raise `AttributeError` and silently disable the check -- and it has no
    Galaxy.
    """

    with expect_error(ValueError, "Galaxy architecture"):
        require_galaxy_mesh("Module2D", _mesh(arch=ttnn.device.Arch.QUASAR))


@pytest.mark.host
@pytest.mark.parametrize(
    "shape,devices,error",
    [
        ((4, 8), GALAXY_DEVICE_COUNT, r"Galaxy mesh \(8, 4\)"),
        (GALAXY_MESH_SHAPE, 31, "exactly 32 devices"),
    ],
)
def test_gate_refuses_a_mesh_that_is_not_a_galaxy(shape, devices, error, expect_error):
    with expect_error(ValueError, error):
        require_galaxy_mesh("Module2D", _mesh(shape=shape, devices=devices))


@pytest.mark.host
def test_gate_raises_rather_than_asserts_on_a_missing_mesh(expect_error):
    """`python -O` strips `assert`; this gate must survive it.

    `TypeError` rather than `ValueError` because the argument is of the wrong
    type, not out of range -- and either way not `AssertionError`, which is the
    proof that the gate is not compiled away.
    """

    with expect_error(TypeError, "requires a mesh_device"):
        require_galaxy_mesh("Module2D", None)


@pytest.mark.host
@pytest.mark.parametrize(
    "architecture",
    [
        ttnn.device.Arch.WORMHOLE_B0,
        ttnn.device.Arch.BLACKHOLE,
        "wormhole",
        "wormhole_b0",
        "blackhole",
        "BLACKHOLE",
    ],
)
def test_architecture_names_accept_every_spelling_callers_use(architecture):
    """`Sampling2DConfig.architecture` holds any of these, depending on caller.

    One site passes `mesh_device.arch()`, another the bare string `"wormhole"`,
    and the field's own default is `"wormhole_b0"`. All three have to pass the
    same gate, which is why it is a substring test and not an equality one.
    """

    assert is_galaxy_architecture_name(architecture)


@pytest.mark.host
@pytest.mark.parametrize("architecture", [ttnn.device.Arch.QUASAR, "quasar", "grayskull", None])
def test_architecture_names_refuse_everything_else(architecture):
    assert not is_galaxy_architecture_name(architecture)
