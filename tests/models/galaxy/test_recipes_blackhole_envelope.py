# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The matmul-rectangle helpers on both Galaxy architectures, on the host.

Three helpers in `recipes.py` searched the worker envelope but read it from the
Wormhole default, so on a Blackhole mesh they would have returned Wormhole
columns -- silently, and inside a program config, which is the shape of failure
this port keeps paying for. They now take a topology.

This file is the cheap check that would catch the reference's own `x=6`
shard-grid failure before hardware: **every resolved placement must lie inside
the worker envelope it will run on.** It runs with no device, because the
descriptor can be built for a named compute grid.

The Wormhole half is a regression guard, not new coverage: these helpers feed a
hardware-qualified path, so their Wormhole answers must not move. The numbers
are written as literals on purpose -- a test that recomputed them from the same
descriptor would pass whatever the descriptor said.
"""

from __future__ import annotations

import pytest
import ttnn

from tt_transformers.models.galaxy import recipes
from tt_transformers.models.galaxy.topology import (
    BLACKHOLE_GALAXY_COMPUTE_GRID,
    galaxy_chip_topology,
)

#: The deployed Blackhole Galaxy grid, measured on silicon: 12 x 10 tensix,
#: 8 DRAM views, dispatch on column 11.
BLACKHOLE_GRID = BLACKHOLE_GALAXY_COMPUTE_GRID

#: Wormhole's answers as they stood before the helpers took a topology.
#:
#: Three columns, because `worker_cores()` is `x=1..3` and `x=5..6` split by the
#: `x=4` prefetch sender column, so the largest rectangle anchored at `(1, 0)`
#: stops at `x=3`.
WORMHOLE_RECTANGLE_WIDTH = 3
WORMHOLE_RECTANGLE_HEIGHT = 10


def _blackhole(grid=BLACKHOLE_GRID):
    return galaxy_chip_topology(ttnn.device.Arch.BLACKHOLE, compute_grid=grid)


def _wormhole():
    return recipes._DEFAULT_TOPOLOGY


# --------------------------------------------------------------------------- #
# Wormhole must not move
# --------------------------------------------------------------------------- #


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("height", [1, 2, 4, 10])
def test_wormhole_rectangle_is_unchanged_by_the_new_parameter(height):
    """Passing the Wormhole descriptor explicitly equals passing nothing.

    The parameter defaults to `None` -> Wormhole, so the qualified path keeps
    its exact answers. Asserting equality between the two call forms is what
    makes the refactor provably non-regressive rather than merely plausible.
    """

    implicit = recipes.dense_matmul_worker_rectangle(height)
    explicit = recipes.dense_matmul_worker_rectangle(height, _wormhole())
    assert implicit == explicit
    box = implicit.bounding_box()
    assert (box.start.x, box.start.y) == (1, 0)
    assert box.end.x - box.start.x + 1 == WORMHOLE_RECTANGLE_WIDTH
    assert box.end.y - box.start.y + 1 == height


@pytest.mark.host
@pytest.mark.model
def test_wormhole_full_height_rectangle_is_unchanged():
    implicit = recipes.worker_matmul_rectangle()
    assert implicit == recipes.worker_matmul_rectangle(_wormhole())
    assert implicit.num_cores() == WORMHOLE_RECTANGLE_WIDTH * WORMHOLE_RECTANGLE_HEIGHT


@pytest.mark.host
@pytest.mark.model
def test_wormhole_dense_matmul_program_config_is_unchanged():
    """The program config is what actually reaches the device, so pin it too.

    Llama's decode QKV geometry: 32 rows, `local_dim` 1024, `local_qkv_size`
    1280 per device on an `(8, 4)` mesh.
    """

    implicit = recipes.dense_matmul_program_config(32, 1024, 1280)
    explicit = recipes.dense_matmul_program_config(32, 1024, 1280, _wormhole())
    for field in ("in0_block_w", "per_core_M", "per_core_N", "out_block_h", "out_block_w"):
        assert getattr(implicit, field) == getattr(explicit, field), field
    assert implicit.allowed_worker_cores == explicit.allowed_worker_cores
    assert implicit.allowed_worker_cores.bounding_box().end.x == WORMHOLE_RECTANGLE_WIDTH


# --------------------------------------------------------------------------- #
# Blackhole resolves, and stays inside its own envelope
# --------------------------------------------------------------------------- #


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("height", [1, 2, 4, 10])
def test_blackhole_rectangle_is_contained_in_the_blackhole_envelope(height):
    """The containment check, which is the point of the file.

    A rectangle that left the envelope would put kernels either on the dispatch
    column -- *"Kernels cannot be placed on dispatch cores!"* -- or on cores
    belonging to no sub-device, which is the `TT_FATAL ... Kernel group cores do
    not match sub device cores` class. Both are loud on the device and free to
    rule out here.
    """

    topology = _blackhole()
    workers = recipes.worker_cores(topology)
    rectangle = recipes.dense_matmul_worker_rectangle(height, topology)

    assert rectangle.subtract(workers).num_cores() == 0, (
        f"the height-{height} Blackhole matmul rectangle leaves the worker envelope"
    )
    box = rectangle.bounding_box()
    assert box.end.y - box.start.y + 1 == height


@pytest.mark.host
@pytest.mark.model
def test_blackhole_rectangle_is_wider_than_wormholes_and_says_why():
    """Ten columns, not three, and the difference is the absent prefetcher.

    Wormhole's rectangle stops at three columns because the `x=4` prefetch
    sender column splits its envelope. Blackhole milestone 1 has no senders, so
    `worker_cores()` is the single rectangle `x=1..10` and the search finds all
    of it. This is the one place the prefetcher-free path is *cheaper* than
    Wormhole rather than more expensive, and it is worth pinning so a later
    change that reintroduces a split column shows up here.
    """

    topology = _blackhole()
    box = recipes.dense_matmul_worker_rectangle(4, topology).bounding_box()
    assert (box.start.x, box.start.y) == (1, 0)
    assert box.end.x == BLACKHOLE_GRID[0] - 2, "the rectangle should stop one short of the dispatch column"
    assert box.end.x - box.start.x + 1 == 10
    assert box.end.x < BLACKHOLE_GRID[0] - 1


@pytest.mark.host
@pytest.mark.model
def test_blackhole_rectangle_never_touches_the_dispatch_column():
    """Stated separately from containment, because it is the empirical one.

    Every other exclusion here follows from the grid. This one does not: column
    11 is inside `compute_with_storage_grid_size()` and the reference excludes
    it because folding it into the worker sub-device *"regresses prefill
    warmup"*. The descriptor carries it as `reserved_columns`, and this asserts
    the matmul search honours it.
    """

    topology = _blackhole()
    dispatch_column = BLACKHOLE_GRID[0] - 1
    assert topology.reserved_columns == (dispatch_column,)
    for height in (1, 4, 10):
        rectangle = recipes.dense_matmul_worker_rectangle(height, topology)
        assert all(
            core.x != dispatch_column
            for core_range in rectangle.ranges()
            for core in (core_range.start, core_range.end)
        )


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize(
    "rows,local_k,local_n,case",
    [
        (32, 1024, 1280, "llama-decode-qkv"),
        (32, 1024, 1024, "llama-decode-wo"),
        (32, 640, 2048, "qwen-decode-qkv"),
        (2048, 1024, 1280, "llama-prefill-2048-qkv"),
    ],
)
def test_blackhole_dense_matmul_program_config_resolves_and_is_contained(rows, local_k, local_n, case):
    """A program config is the artefact that reaches the device, so check that.

    Containment of `allowed_worker_cores` is the check; the grid dimensions are
    asserted to agree with it because
    `MatmulMultiCoreReuseMultiCastProgramConfig` confines the op only when the
    set it is given *is* the rectangle its `compute_with_storage_grid_size`
    describes. A config whose two halves disagree silently uses the bounding
    box.
    """

    topology = _blackhole()
    workers = recipes.worker_cores(topology)
    config = recipes.dense_matmul_program_config(rows, local_k, local_n, topology)

    allowed = config.allowed_worker_cores
    assert allowed.subtract(workers).num_cores() == 0, f"{case}: allowed_worker_cores leaves the envelope"
    box = allowed.bounding_box()
    # Reads back as a `CoreCoord`, not the tuple it was constructed from.
    grid = config.compute_with_storage_grid_size
    assert (grid.x, grid.y) == (box.end.x - box.start.x + 1, box.end.y - box.start.y + 1), (
        f"{case}: the program config's grid and its allowed cores disagree"
    )
    assert config.per_core_N * grid.x * recipes.TILE >= local_n, f"{case}: per_core_N does not cover N"


@pytest.mark.host
@pytest.mark.model
def test_blackhole_full_height_rectangle_is_contained():
    topology = _blackhole()
    rectangle = recipes.worker_matmul_rectangle(topology)
    assert rectangle.subtract(recipes.worker_cores(topology)).num_cores() == 0
    assert rectangle.num_cores() == 10 * 10


# --------------------------------------------------------------------------- #
# Harvesting robustness
# --------------------------------------------------------------------------- #


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("grid", [(13, 10), (12, 10), (11, 10)], ids=["unharvested", "deployed", "2x-harvested"])
def test_the_rectangle_follows_the_grid_rather_than_naming_a_width(grid):
    """12 x 10 is the deployed shape, not a guarantee: harvesting is per-part.

    A mesh whose chassis harvests differently moves the dispatch column with
    the grid width, so the envelope and every rectangle inside it move too.
    Nothing here names 10 or 11.
    """

    topology = _blackhole(grid)
    workers = recipes.worker_cores(topology)
    rectangle = recipes.dense_matmul_worker_rectangle(4, topology)

    assert rectangle.subtract(workers).num_cores() == 0
    box = rectangle.bounding_box()
    assert box.end.x == grid[0] - 2
    assert topology.reserved_columns == (grid[0] - 1,)


@pytest.mark.host
@pytest.mark.model
def test_a_height_taller_than_the_envelope_is_refused_not_clamped():
    """Fail closed. A clamped height would silently change the matmul's M split.

    The envelope is 10 rows tall, so 11 has no rectangle. `recipes` raises
    rather than returning the tallest available, because a program config built
    for a height it did not get resolves `per_core_M` against the wrong grid_y
    and is wrong rather than slow.
    """

    topology = _blackhole()
    with pytest.raises(ValueError, match="no worker rectangle"):
        recipes.dense_matmul_worker_rectangle(11, topology)
