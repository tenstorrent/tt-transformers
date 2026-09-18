# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for RotarySetup2D on a Blackhole Galaxy.

The Blackhole counterpart of `test_rope_2d_wh_galaxy.py`. `RotarySetup2D`
issues no collective and runs no matmul, so there is no `num_links`, no program
config and no compute kernel config to re-derive here; the only thing it places
on cores is the decode cos/sin pair and the decode transformation matrix, and
**that placement is the whole risk**.

## The trap, and why it is live on Blackhole and not on Wormhole

`recipes.rope_core_grids(mesh_device, use_qk_fused=...)` returns
`(core_grid, batch_grid)`. `batch_grid` carries the decode cos/sin shards, so it
has to lie inside the worker sub-device like every other decode placement.
Taking the first `rows` cores of the *whole* compute grid instead -- which is
what the Wormhole suite this file is modelled on does, and what it can afford to
do because it loads no sub-device manager -- puts shards on cores no sub-device
owns, and `ttnn.embedding` then builds the decode tables on cores the loaded
manager does not own and aborts with

    TT_FATAL ... Kernel group cores do not match sub device cores
                 for programmable core type TENSIX

`rope_core_grids` resolves the anchor from the mesh rather than defaulting to
Wormhole precisely because the anchor is per-architecture. On Blackhole the
compute grid is 12 x 10 with **dispatch on column 11, inside it**, so a grid
taken from the compute grid rather than from the worker envelope is wrong here
in a way it is not on Wormhole. The containment assertion is cheap, it is
host-side, and it is the single most useful thing in this file.

`core_grid` -- the first element of that pair -- is the full compute grid, and it
*does* include the dispatch column on Blackhole. That is harmless only because
`RotarySetup2D` stores it and never resolves a placement from it; every real
placement comes from `batch_grid`. Asserted below rather than assumed, because
if the module ever derives a grid from it, this file should be the thing that
notices.

## Three node ids, three independent failure domains

`has_fused_qk_rotary` is `False` on Blackhole, so the **non-fused rotary pair is
the Blackhole route** and the non-fused decode geometry is the primary claim.
The static contract trace was against the non-fused op, so it is
directly reusable.

* `..._decode_on_worker_subdevice` loads the descriptor's worker envelope as the
  only sub-device and runs decode inside it. This is the production decode shape
  and the only test here that can hit the `TT_FATAL` above.
* `..._prefill_full_grid` loads **no** sub-device manager, because
  `prefill_forward` calls `ttnn.clone`, which compiles a program over the full
  compute grid and would abort under a narrow sub-device -- the same `TT_FATAL`,
  for the same reason, which is why `_materialize_table_copy` writes its tilized
  prefill copy from the host instead of cloning on device.
* `..._decode_fused_qk_pair` covers `use_qk_fused=True` on one geometry, and
  also loads no sub-device manager, so that a failure of the sub-device route
  cannot take out both decode claims at once.

Splitting them is not tidiness: one pytest node id per process is already
required here (the ttnn program cache belongs to the mesh device and the
process), so the split costs mesh opens and buys independent diagnosis of three
different things.

No checkpoint is read: the cos/sin tables are built from `torch` on the host, so
there is no path along which this skips instead of running.
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from examples.common.auto_compose import to_torch_auto_compose
from loguru import logger
from tests.models.galaxy.galaxy_hardware import BHGLX_DEVICE_PARAMS
from tests.modules._bh_galaxy_hardware import (
    bh_galaxy_topology,
    deallocate_tensor,
    record_module_output,
)
from tests.support.comparison import comp_pcc

from tt_transformers.models.galaxy.recipes import rope_core_grids, worker_cores
from tt_transformers.modules.lazy_weight import LazyWeight
from tt_transformers.modules.rope.rope_2d import RotarySetup2D, RotarySetup2DConfig

#: The Wormhole suite's constants, unchanged: both Galaxy models use
#: `head_dim = 128`, and both are qualified over a 2048-entry table.
_HEAD_DIM = 128
_MAX_SEQ_LEN = 2048

#: Galaxy physical batch, and the users one mesh column holds. Both are
#: architecture-invariant -- `(8, 4)` is the same mesh on both Galaxy
#: architectures -- and `RotarySetup2D` rejects anything else.
_BATCH = 32
_USERS_PER_COLUMN = 8

_PCC = 0.99


def _rope_tables(max_seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the cos/sin tables on the host. Identical to the Wormhole suite."""

    positions = torch.arange(max_seq_len, dtype=torch.float32)
    frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().bfloat16()[None, None], angles.sin().bfloat16()[None, None]


def _module(mesh_device, cos, sin, theta, core_grid, batch_grid, *, use_qk_fused: bool) -> RotarySetup2D:
    return RotarySetup2D.from_config(
        RotarySetup2DConfig(
            LazyWeight(source=cos, device=mesh_device),
            LazyWeight(source=sin, device=mesh_device),
            max_batch_size=_BATCH,
            rope_theta=theta,
            use_qk_fused=use_qk_fused,
            core_grid=core_grid,
            batch_grid=batch_grid,
        )
    )


def _release(module: RotarySetup2D, *tensors) -> None:
    """Free device tensors, then the module's own weights, and never raise.

    Teardown runs after a failure as often as after a pass, and a teardown that
    raises replaces the real failure with its own.
    """

    targets = list(tensors) + [
        getattr(module, name, None)
        for name in (
            "cos_matrix",
            "sin_matrix",
            "cos_matrix_prefill",
            "sin_matrix_prefill",
            "transformation_mat",
            "transformation_mat_prefill",
        )
    ]
    for target in targets:
        try:
            deallocate_tensor(target)
        except BaseException as error:  # noqa: BLE001 - teardown must not mask the real failure
            logger.warning("rope_2d BH Galaxy teardown could not free a tensor: {}", error)
    for weight in (
        module.config.cos_matrix,
        module.config.sin_matrix,
        module.config._decode_trans_mat,
        module.config._prefill_trans_mat,
    ):
        weight._value = None


def _ordered_cores(grid: ttnn.CoreRangeSet) -> list[tuple[int, int]]:
    """Flatten a `CoreRangeSet` to `[(x, y)]` in constructor sequence.

    Order is kept rather than collapsed into a set on purpose: `CoreRangeSet`
    preserves the sequence it was built from, the same point-cores built from a
    `list` and from a `set` are unequal objects with different iteration order,
    and kernels that walk a grid depend on that order.
    """

    return [
        (x, y)
        for core_range in grid.ranges()
        for y in range(core_range.start.y, core_range.end.y + 1)
        for x in range(core_range.start.x, core_range.end.x + 1)
    ]


def _assert_batch_grid_inside_worker_envelope(topology, batch_grid: ttnn.CoreRangeSet, *, rows: int) -> None:
    """The load-bearing host-side check: `batch_grid` never leaves the envelope.

    Cheap, deterministic, and it fails before any device work, so it is the
    first thing each test does. Everything it compares against comes from the
    descriptor -- `worker_cores` is the single canonical converter from the
    descriptor's rectangles to a `CoreRangeSet` -- so no core coordinate, grid
    width or column index is named here.
    """

    cores = _ordered_cores(batch_grid)
    envelope = _ordered_cores(worker_cores(topology))
    dispatch_column = topology.compute_grid[0] - 1

    assert batch_grid.num_cores() == rows, f"batch_grid holds {batch_grid.num_cores()} cores, expected {rows}"
    assert len(cores) == rows and len(set(cores)) == rows, f"batch_grid enumerates {cores}, expected {rows} distinct"
    escaped = [core for core in cores if core not in set(envelope)]
    assert not escaped, (
        f"batch_grid leaves the worker envelope at {escaped}; the decode cos/sin shards would sit on cores "
        "the loaded sub-device manager does not own and `ttnn.embedding` would abort with "
        "'Kernel group cores do not match sub device cores'"
    )
    assert all(x != dispatch_column for x, _ in cores), (
        f"batch_grid touches the dispatch column {dispatch_column}, which is inside the "
        f"{topology.compute_grid} compute grid on Blackhole"
    )
    logger.info("BH Galaxy rope batch_grid ({} cores): {}", rows, cores)


def _assert_batch_grid_order(topology, batch_grid: ttnn.CoreRangeSet, *, rows: int) -> None:
    """Assert the placement's *order*, not only its membership.

    Deliberately called after the numeric claim has been made and recorded: the
    expectation is that a row-wise subgrid anchored at the descriptor's first
    worker core enumerates the envelope's first `rows` cores row-wise, and if
    Blackhole's `num_cores_to_corerangeset_in_subcoregrids` decomposes that set
    differently, the honest reading is an ordering convention to write down
    rather than a placement defect. Failing here still leaves the recorded
    output and the PCC values from this process on disk.
    """

    envelope = _ordered_cores(worker_cores(topology))
    expected = sorted(envelope, key=lambda core: (core[1], core[0]))[:rows]
    assert _ordered_cores(batch_grid) == expected, (
        f"batch_grid enumerates {_ordered_cores(batch_grid)}, expected the envelope's first {rows} cores "
        f"row-wise from the descriptor anchor {topology.sampling_start_core}: {expected}"
    )


def _load_worker_sub_device(mesh_device, topology) -> tuple[object, ttnn.CoreRangeSet]:
    """Load the descriptor's worker envelope as the one and only sub-device.

    This is the prefetcher-free shape `bh_galaxy_mode_plan` describes -- one
    sub-device over the envelope, `SubDeviceId(0)` as the worker -- but it is
    built here rather than through that helper because `GalaxyModePlan` rejects
    a plan with no collectives, and `RotarySetup2D` issues none. Driving the
    helper would mean inventing a collective plan and letting
    `create_galaxy_resources` allocate persistent buffers and semaphores for a
    collective this module never calls, which is more device state and more
    risk than the thing being tested.

    The *geometry* still comes from `recipes.worker_cores`, so there is no
    second construction path for the envelope; only the sub-device manager's
    create/load/unload is local, and the harness exposes that only inside
    `_CCLOnlySubdeviceOwner`, which needs the same collective-bearing config.
    """

    cores = worker_cores(topology)
    worker_id = ttnn.SubDeviceId(0)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
    try:
        mesh_device.load_sub_device_manager(manager)
        mesh_device.set_sub_device_stall_group([worker_id])
    except BaseException:
        # A manager that was created and never loaded still has to be removed,
        # or the mesh closes with it outstanding.
        mesh_device.remove_sub_device_manager(manager)
        raise
    logger.info("loaded the BH Galaxy worker sub-device: {} cores", cores.num_cores())
    return manager, cores


def _unload_worker_sub_device(mesh_device, manager) -> None:
    """Unload and remove the manager, best effort.

    Each step is independent so that a failure in one does not leave the others
    undone, and none of them may replace the test's own failure with its own:
    this runs in a `finally`, after the interesting exception.
    """

    for step, action in (
        ("reset the stall group", lambda: mesh_device.reset_sub_device_stall_group()),
        ("clear the loaded manager", lambda: mesh_device.clear_loaded_sub_device_manager()),
        ("remove the manager", lambda: mesh_device.remove_sub_device_manager(manager)),
    ):
        try:
            action()
        except BaseException as error:  # noqa: BLE001 - teardown must not mask the real failure
            logger.warning("rope_2d BH Galaxy teardown could not {}: {}", step, error)


def _compose_decode(outputs, mesh_device, *, rows_per_column: int) -> list[torch.Tensor]:
    """Compose the decode cos/sin pair to `[1, 1, rows, head_dim]`.

    The composer is the Wormhole suite's: mesh rows carry replicas (so the
    first is taken) and mesh columns carry the position groups, in column
    order.
    """

    mesh_rows, mesh_columns = (int(value) for value in tuple(mesh_device.shape))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=(mesh_rows, mesh_columns))
    composed = []
    for output in outputs:
        value = ttnn.to_torch(output, mesh_composer=composer)
        assert value.shape[1] == rows_per_column * mesh_columns, (value.shape, rows_per_column, mesh_columns)
        assert value.shape[2] == mesh_rows, (value.shape, mesh_rows)
        composed.append(value[:, :, :1].permute(0, 2, 1, 3))
    return composed


def _assert_no_column_permutation(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    rows_per_column: int,
    case: str,
) -> None:
    """Correlate per mesh column: a shard-order bug is a permutation.

    The decode indices are replicated over mesh rows and sharded on axis 0 over
    the 4 mesh columns, so column `c` holds users `c * rows_per_column ...`. A
    column-order mismatch leaves every column well correlated with *some*
    reference column and none with its own, which aggregate PCC reports only as
    "wrong" -- indistinguishable from bad numerics.

    **The diagonal-best test is separable on these tables, measured rather than
    assumed.** RoPE tables are smooth in position, so neighbouring position
    groups could in principle correlate as well as a group does with itself, and
    then this check would fail on a *correct* result. Computed on the host for
    both thetas: the diagonal is 1.0 and the largest off-diagonal is **0.673**
    (cos, columns 2 vs 3; sin stays under 0.39). So the margin is ~0.33, and a
    permuted column would also fail the aggregate `_PCC` bar outright rather
    than slipping under it.
    """

    columns = actual.shape[-2] // rows_per_column
    for column in range(columns):
        start = column * rows_per_column
        actual_slice = actual[:, :, start : start + rows_per_column]
        scores = [
            comp_pcc(
                reference[:, :, other * rows_per_column : (other + 1) * rows_per_column],
                actual_slice,
                _PCC,
            )[1]
            for other in range(columns)
        ]
        logger.info("{}: mesh column {} correlates {}", case, column, scores)
        best = max(range(columns), key=lambda other: scores[other])
        assert best == column, (
            f"{case}: mesh column {column} correlates best with reference column {best} (scores {scores}); "
            "that is a shard-order permutation, not a numeric error"
        )


def _assert_pair(references, actuals, *, case: str) -> None:
    for name, reference, actual in zip(("cos", "sin"), references, actuals):
        assert actual.shape == reference.shape, (name, actual.shape, reference.shape)
        passing, pcc = comp_pcc(reference, actual, _PCC)
        # `comp_pcc` logs nothing on success, so print every stage's value.
        logger.info("rope_2d BH Galaxy {} {}: PCC {}", name, case, pcc)
        assert passing, f"{case} {name}: PCC {pcc} below {_PCC}"


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "theta",
    [pytest.param(500000.0, id="llama-theta500k"), pytest.param(1000000.0, id="qwen-theta1m")],
)
@torch.no_grad()
def test_rope_2d_bh_galaxy_decode_on_worker_subdevice(mesh_device, theta):
    """Qualify the non-fused decode pair inside the Blackhole worker sub-device.

    The primary Blackhole claim. `has_fused_qk_rotary` is `False` here, so the
    non-fused pair is the route the models take, and this runs it in the shape
    production runs it: the descriptor's worker envelope loaded as the only
    sub-device, with the decode cos/sin height-sharded onto a `batch_grid`
    inside it.

    The reference is the host cos/sin table indexed by the decode positions --
    never a second TT path. Positions are `arange(32)`, as on Wormhole, which
    makes both the across-column order (column `c` holds users `8c..8c+7`) and
    the within-column order observable in the composed result.
    """

    topology = bh_galaxy_topology(mesh_device)
    core_grid, batch_grid = rope_core_grids(mesh_device, use_qk_fused=False)
    _assert_batch_grid_inside_worker_envelope(topology, batch_grid, rows=_USERS_PER_COLUMN)
    # The pair's `core_grid` is the raw compute grid and includes the dispatch
    # column on Blackhole. It is inert -- `RotarySetup2D` stores it and resolves
    # no placement from it -- and this is where that stops being an assumption.
    assert (int(core_grid.x), int(core_grid.y)) == topology.compute_grid

    cos, sin = _rope_tables(_MAX_SEQ_LEN, _HEAD_DIM, theta)
    positions = torch.arange(_BATCH, dtype=torch.int32)
    references = (cos[:, :, positions.long(), :], sin[:, :, positions.long(), :])

    manager = None
    module = None
    rot_idxs = None
    try:
        manager, _ = _load_worker_sub_device(mesh_device, topology)
        module = _module(mesh_device, cos, sin, theta, core_grid, batch_grid, use_qk_fused=False)
        rot_idxs = module.get_rot_idxs(positions)
        actuals: list[torch.Tensor] = []
        for invocation in range(2):
            case = f"decode invocation {invocation} (theta {theta})"
            outputs = module.decode_forward(rot_idxs)
            try:
                actuals = _compose_decode(outputs, mesh_device, rows_per_column=_USERS_PER_COLUMN)
            finally:
                for output in outputs:
                    deallocate_tensor(output)
            _assert_pair(references, actuals, case=case)
            for name, reference, actual in zip(("cos", "sin"), references, actuals):
                _assert_no_column_permutation(
                    reference, actual, rows_per_column=_USERS_PER_COLUMN, case=f"{case} {name}"
                )
            # The tables are resident weights, not per-call allocations; if a
            # forward freed one, the second invocation would read freed L1.
            assert module.cos_matrix.is_allocated()
            assert module.sin_matrix.is_allocated()

        record_module_output(f"rope_2d_bh_galaxy_decode_theta{int(theta)}", *actuals)
        _assert_batch_grid_order(topology, batch_grid, rows=_USERS_PER_COLUMN)
    finally:
        if module is not None:
            _release(module, rot_idxs)
        if manager is not None:
            _unload_worker_sub_device(mesh_device, manager)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "theta",
    [pytest.param(500000.0, id="llama-theta500k"), pytest.param(1000000.0, id="qwen-theta1m")],
)
@torch.no_grad()
def test_rope_2d_bh_galaxy_prefill_full_grid(mesh_device, theta):
    """Qualify the prefill tables at 128 and 2048, on the default manager.

    **No sub-device manager is loaded, and that is a requirement rather than a
    simplification.** `prefill_forward` slices the tilized host-written table
    copy and hands the slice to `ttnn.clone`, which compiles a program over the
    full compute grid; under a narrow worker sub-device that aborts with
    `Kernel group cores do not match sub device cores`, which is exactly why
    `_materialize_table_copy` writes the prefill copy from the host rather than
    cloning it on device. Production runs Blackhole prefill on the full compute
    grid for the same reason.

    So this node id also carries the cheapest confirmation that a
    full-compute-grid program is legal on a Blackhole Galaxy whose dispatch
    column sits inside that grid.
    """

    topology = bh_galaxy_topology(mesh_device)
    core_grid, batch_grid = rope_core_grids(mesh_device, use_qk_fused=False)
    _assert_batch_grid_inside_worker_envelope(topology, batch_grid, rows=_USERS_PER_COLUMN)

    cos, sin = _rope_tables(_MAX_SEQ_LEN, _HEAD_DIM, theta)
    module = None
    recorded: list[torch.Tensor] = []
    try:
        module = _module(mesh_device, cos, sin, theta, core_grid, batch_grid, use_qk_fused=False)
        for seq_len in (128, 2048):
            references = (cos[:, :, :seq_len], sin[:, :, :seq_len])
            actuals: list[torch.Tensor] = []
            for invocation in range(2):
                outputs = module.prefill_forward(start_pos=0, seq_len=seq_len)
                try:
                    actuals = [to_torch_auto_compose(output)[:, :, :seq_len, :_HEAD_DIM] for output in outputs]
                finally:
                    for output in outputs:
                        deallocate_tensor(output)
                _assert_pair(references, actuals, case=f"prefill {seq_len} invocation {invocation} (theta {theta})")
            recorded.extend(actuals)

        record_module_output(f"rope_2d_bh_galaxy_prefill_theta{int(theta)}", *recorded)
    finally:
        if module is not None:
            _release(module)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("theta", [pytest.param(500000.0, id="llama-theta500k")])
@torch.no_grad()
def test_rope_2d_bh_galaxy_decode_fused_qk_pair(mesh_device, theta):
    """Cover the fused-QK decode geometry, which is **not** the Blackhole route.

    `has_fused_qk_rotary` is `False` on the Blackhole descriptor, so the
    condition that selects fused QK on a prefetcher mesh may simply never fire
    here. This node id exists because the geometry is expressible and cheap to
    check, not because the port depends on it: **a failure here is
    expected-and-recorded, not a defect**, and it should be read as evidence
    about the fused path's availability rather than as a regression in
    `RotarySetup2D`. Whatever it does, it must not be taken as a reason to turn
    `has_fused_qk_rotary` on.

    It runs on the default sub-device manager rather than the worker envelope,
    so that a failure of the sub-device route cannot take out both decode node
    ids at once. The fused geometry doubles `batch_grid` to 16 cores and repeats
    each column's 8 positions, which the host reference reproduces exactly.
    """

    topology = bh_galaxy_topology(mesh_device)
    assert not topology.capabilities.has_fused_qk_rotary, (
        "the Blackhole descriptor now reports has_fused_qk_rotary=True; this node id was written as the "
        "non-primary fused geometry and its expectations need revisiting before it is read as a gate"
    )
    rows = _USERS_PER_COLUMN * 2
    core_grid, batch_grid = rope_core_grids(mesh_device, use_qk_fused=True)
    _assert_batch_grid_inside_worker_envelope(topology, batch_grid, rows=rows)

    cos, sin = _rope_tables(_MAX_SEQ_LEN, _HEAD_DIM, theta)
    positions = torch.arange(_BATCH, dtype=torch.int32)
    # `prepare_rot_idxs` lays the batch out as `[columns, users]`, repeats each
    # column's users once for Q and once for K, and shards axis 0 over the mesh
    # columns -- so the composed order is `8c..8c+7` twice, per column.
    fused_positions = positions.reshape(-1, _USERS_PER_COLUMN).repeat(1, 2).reshape(-1)
    references = (cos[:, :, fused_positions.long(), :], sin[:, :, fused_positions.long(), :])

    module = None
    rot_idxs = None
    try:
        module = _module(mesh_device, cos, sin, theta, core_grid, batch_grid, use_qk_fused=True)
        rot_idxs = module.get_rot_idxs(positions)
        actuals: list[torch.Tensor] = []
        for invocation in range(2):
            case = f"fused-QK decode invocation {invocation} (theta {theta})"
            outputs = module.decode_forward(rot_idxs)
            try:
                actuals = _compose_decode(outputs, mesh_device, rows_per_column=rows)
            finally:
                for output in outputs:
                    deallocate_tensor(output)
            _assert_pair(references, actuals, case=case)
            for name, reference, actual in zip(("cos", "sin"), references, actuals):
                _assert_no_column_permutation(reference, actual, rows_per_column=rows, case=f"{case} {name}")

        record_module_output(f"rope_2d_bh_galaxy_decode_fused_theta{int(theta)}", *actuals)
        _assert_batch_grid_order(topology, batch_grid, rows=rows)
    finally:
        if module is not None:
            _release(module, rot_idxs)
