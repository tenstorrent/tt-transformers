# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The three model-layer Galaxy architecture gates, and what they still refuse.

`tests/modules/test_galaxy_mesh.py` covers the *module*-layer gate, which asks
the weaker question: "is this a Galaxy mesh of an architecture the 2D modules
run on". These three are the model layer, and they ask the stronger one: "does
this architecture have intra-chip geometry written down". Both halves are
tested, because the inventory that listed only the module half was wrong by
three gates and the omission was found on silicon rather than here.

Each gate was widened against the error it actually produced on a Blackhole
Galaxy on 2026-09-16, not ahead of it:

    models/galaxy/ccl.py       ValueError: Galaxy CCL requires Wormhole B0, got Arch.BLACKHOLE
    models/galaxy/resources.py ValueError: Galaxy resources require Wormhole B0

`recipes.GALAXY_ARCHITECTURE` is the third. It was not a refusal but a constant
asserting that Galaxy *is* Wormhole, so widening it means removing the claim;
the test below pins that it is gone rather than quietly re-resolving to
Wormhole under a name that reads architecture-neutral.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import ttnn

from tt_transformers.models.galaxy import recipes
from tt_transformers.models.galaxy.ccl import _validate_galaxy
from tt_transformers.models.galaxy.resources import GalaxyResourcesConfig
from tt_transformers.models.galaxy.topology import supported_galaxy_architectures

GALAXY_ARCHITECTURES = tuple(sorted(supported_galaxy_architectures(), key=str))

#: The negative control throughout this branch. A real member of the pinned
#: `ttnn`'s `Arch` enum -- naming one that does not exist raises
#: `AttributeError` inside the gate and silently disables the check -- and one
#: that genuinely has no Galaxy.
NO_GALAXY_ARCH = ttnn.device.Arch.QUASAR


def _mesh(arch=ttnn.device.Arch.WORMHOLE_B0, shape=(8, 4), devices=32):
    return SimpleNamespace(
        shape=shape,
        get_num_devices=lambda: devices,
        arch=lambda: arch,
    )


def _mode_plan(mode):
    """The smallest plan `GalaxyResourcesConfig` will accept.

    Deliberately not a real envelope: this file tests the architecture gate,
    and a plan carrying Wormhole core ranges would make a Blackhole case pass
    for the wrong reason.
    """

    return SimpleNamespace(mode=mode)


# --------------------------------------------------------------------------- #
# Gate 1 -- models/galaxy/ccl.py
# --------------------------------------------------------------------------- #


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("arch", GALAXY_ARCHITECTURES, ids=str)
def test_ccl_gate_accepts_every_architecture_with_a_topology(arch):
    """Both Galaxy architectures reach the CCL layer.

    This is the gate that blocked every Blackhole device suite: every 2D module
    takes a `tt_ccl`, so its refusal stopped the run before a module resolved
    anything.
    """

    assert _validate_galaxy(_mesh(arch=arch), (8, 4), arch) is None


@pytest.mark.host
@pytest.mark.model
def test_ccl_gate_still_refuses_an_architecture_with_no_topology(expect_error):
    """The contract stays an allowlist keyed on a descriptor existing.

    Widening to `supported_galaxy_architectures()` rather than to a wider set
    of names is what keeps this fail-closed: an architecture passes only once
    someone has written its intra-chip geometry down.
    """

    with expect_error(ValueError, "no topology for"):
        _validate_galaxy(_mesh(arch=NO_GALAXY_ARCH), (8, 4), NO_GALAXY_ARCH)


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("arch", GALAXY_ARCHITECTURES, ids=str)
def test_ccl_gate_keeps_the_mesh_shape_a_hard_equality(arch, expect_error):
    """`(8, 4)` is architecture-invariant on Galaxy, so only the arch line moved.

    Blackhole Galaxy declares the same `device_topology { dims: [8, 4] }`
    (`single_bh_galaxy_mesh_graph_descriptor.textproto`), so everything derived
    from the shape carries -- the weight splits, the mesh mappers, the
    row/column sharding, `n_kv_heads == 8` per mesh row. Widening the shape
    check as well would have given up that invariant for nothing.
    """

    with expect_error(ValueError, r"logical mesh shape \(8, 4\)"):
        _validate_galaxy(_mesh(arch=arch, shape=(4, 8)), (4, 8), arch)


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("arch", GALAXY_ARCHITECTURES, ids=str)
def test_ccl_gate_keeps_the_device_count(arch, expect_error):
    with expect_error(ValueError, "exactly 32 devices"):
        _validate_galaxy(_mesh(arch=arch, devices=31), (8, 4), arch)


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("arch", GALAXY_ARCHITECTURES, ids=str)
def test_ccl_gate_refuses_a_mesh_whose_architecture_disagrees(arch, expect_error):
    """The resolved architecture and the device's must be the same one.

    With two supported architectures this check does real work for the first
    time: before, any mismatch was also a Wormhole refusal, so it could not be
    reached independently.
    """

    other = next(candidate for candidate in GALAXY_ARCHITECTURES if candidate != arch)
    with expect_error(ValueError, "does not match the resolved architecture"):
        _validate_galaxy(_mesh(arch=other), (8, 4), arch)


# --------------------------------------------------------------------------- #
# Gate 2 -- models/galaxy/resources.py
# --------------------------------------------------------------------------- #


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("arch", GALAXY_ARCHITECTURES, ids=str)
def test_resources_config_accepts_every_architecture_with_a_topology(arch):
    """The refusal that sat immediately behind the CCL one."""

    config = GalaxyResourcesConfig(
        architecture=arch,
        prefill=_mode_plan("prefill"),
        decode=_mode_plan("decode"),
    )
    assert config.architecture is arch


@pytest.mark.host
@pytest.mark.model
def test_resources_config_still_refuses_an_architecture_with_no_topology(expect_error):
    with expect_error(ValueError, "no topology for"):
        GalaxyResourcesConfig(
            architecture=NO_GALAXY_ARCH,
            prefill=_mode_plan("prefill"),
            decode=_mode_plan("decode"),
        )


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("arch", GALAXY_ARCHITECTURES, ids=str)
def test_resources_config_keeps_the_mesh_shape_a_hard_equality(arch, expect_error):
    with expect_error(ValueError, r"logical mesh shape \(8, 4\)"):
        GalaxyResourcesConfig(
            architecture=arch,
            prefill=_mode_plan("prefill"),
            decode=_mode_plan("decode"),
            mesh_shape=(4, 8),
        )


# --------------------------------------------------------------------------- #
# Gate 3 -- recipes.GALAXY_ARCHITECTURE
# --------------------------------------------------------------------------- #


@pytest.mark.host
@pytest.mark.model
def test_the_constant_asserting_galaxy_is_wormhole_is_gone():
    """`GALAXY_ARCHITECTURE` is not renamed-and-kept; the claim is retracted.

    A module asking "what architecture is Galaxy" can only be asking about a
    mesh once there are two, so the answer has to come from one --
    `mesh_device.arch()`, or `galaxy_topology(mesh_device)`. Keeping a
    no-argument constant under an architecture-neutral name would have made
    every remaining caller read as correct while resolving Wormhole.
    """

    assert not hasattr(recipes, "GALAXY_ARCHITECTURE")

    import tt_transformers.models.galaxy as galaxy_package

    assert "GALAXY_ARCHITECTURE" not in galaxy_package.__all__
    with pytest.raises(AttributeError):
        galaxy_package.GALAXY_ARCHITECTURE


@pytest.mark.host
@pytest.mark.model
def test_the_wormhole_default_is_named_as_a_default_and_reads_through():
    """The replacement names its architecture and cannot drift from it.

    `WORMHOLE_GALAXY_ARCHITECTURE` exists because the no-argument helpers in
    `recipes` still describe Wormhole. It is defined as
    `WORMHOLE_GALAXY_TOPOLOGY.architecture` rather than as a second literal,
    so the constant and the descriptor it documents cannot disagree.
    """

    assert recipes.WORMHOLE_GALAXY_ARCHITECTURE is ttnn.device.Arch.WORMHOLE_B0
    assert recipes.WORMHOLE_GALAXY_ARCHITECTURE is recipes._DEFAULT_TOPOLOGY.architecture


@pytest.mark.host
@pytest.mark.model
def test_the_reserved_worker_core_count_is_the_wormhole_one_by_name():
    """`GALAXY_CCL_RESERVED_WORKER_CORES` is 4 because Wormhole has 4 links.

    Pinned because getting this wrong does not raise: starving
    `all_reduce_async` of worker cores warns and then segmentation-faults. The
    architecture-aware route is `galaxy_ccl_reserved_worker_cores(mesh_device)`,
    which returns 2 on Blackhole.
    """

    assert recipes.GALAXY_CCL_RESERVED_WORKER_CORES == 4
    assert recipes.GALAXY_CCL_RESERVED_WORKER_CORES == recipes.GALAXY_FABRIC_LINKS[ttnn.device.Arch.WORMHOLE_B0]
    assert recipes.GALAXY_FABRIC_LINKS[ttnn.device.Arch.BLACKHOLE] == 2
