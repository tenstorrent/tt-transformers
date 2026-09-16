# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The Galaxy mesh facts the 2D modules need, without depending on a model package.

`models/galaxy/topology.py` owns the real per-architecture geometry, but
`modules/` never imports `models/` -- the dependency runs the other way, at four
sites and never back. What is here is what a module's own mesh gate needs, and
it can live at this layer because none of it is per-architecture geometry:

* the mesh **shape** is identical on both Galaxy architectures, because Blackhole
  Galaxy declares the same `device_topology { dims: [8, 4] }`;
* the **architecture set** is an allowlist of Galaxy architectures the modules
  will run on. That is a weaker claim than "this architecture has a qualified
  intra-chip geometry", which is `recipes.validate_galaxy_mesh`'s job and is
  keyed on a topology descriptor existing. The model layer runs that one.

`modules/tt_ccl.py` already knows `BHGLX` for the same reason: a module-layer
fact about Galaxy hardware is not a model-layer concern.
"""

from __future__ import annotations

from typing import Any

import ttnn

#: The Galaxy logical mesh, identical on both architectures.
GALAXY_MESH_SHAPE = (8, 4)
GALAXY_DEVICE_COUNT = 32

#: Architectures whose Galaxy meshes the 2D modules accept.
GALAXY_ARCHITECTURES = frozenset({ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE})

#: The same allowlist for the config fields that carry an architecture *name*
#: rather than a `ttnn.device.Arch`. `Sampling2DConfig.architecture` is one: its
#: callers pass `mesh_device.arch()` at one site and the literal `"wormhole"` at
#: another, so the field is validated by substring and has to stay that way.
#: `WORMHOLE_B0` stringifies with the suffix, which is why the token is the
#: prefix and not the whole name.
GALAXY_ARCHITECTURE_NAMES = ("wormhole", "blackhole")


def is_galaxy_architecture_name(architecture: Any) -> bool:
    """Whether an architecture *name* is one the 2D modules accept.

    Deliberately not an equality test: the field this serves holds an
    `Arch` enum, a bare `"wormhole"`, or a full `"wormhole_b0"` depending on
    the caller, and all three have to pass the same gate.
    """

    text = str(architecture).lower()
    return any(name in text for name in GALAXY_ARCHITECTURE_NAMES)


def require_galaxy_mesh(name: str, mesh_device: Any) -> tuple[int, int]:
    """Return the cluster shape, or raise if this is not a Galaxy mesh we accept.

    Raised rather than asserted: `python -O` strips `assert`, and this gate is
    the only thing between a wrong mesh and tensors placed on cores the loaded
    sub-device manager does not own -- which hangs the host with no traceback
    rather than reporting anything.
    """

    if mesh_device is None:
        raise TypeError(f"{name} requires a mesh_device; none was configured and no default is available")
    shape = tuple(mesh_device.shape)
    if shape != GALAXY_MESH_SHAPE:
        raise ValueError(f"{name} requires Galaxy mesh {GALAXY_MESH_SHAPE}, got {shape}")
    if mesh_device.get_num_devices() != GALAXY_DEVICE_COUNT:
        raise ValueError(f"{name} requires exactly {GALAXY_DEVICE_COUNT} devices, got {mesh_device.get_num_devices()}")
    architecture = mesh_device.arch()
    if architecture not in GALAXY_ARCHITECTURES:
        raise ValueError(f"{name} requires a Galaxy architecture, got {architecture}")
    return shape
