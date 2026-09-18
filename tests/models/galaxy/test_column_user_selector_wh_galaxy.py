# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware qualification for the Galaxy column user selector.

Decode logits leave ``LMHead2D`` with all 32 users present on every column,
while ``Sampling2D`` consumes one column's eight users. ``GalaxyColumnUserSelector``
bridges the two with a one-hot matmul whose selector rows differ per column.
That composition is the only unqualified step in the device sampling path, so it
is worth qualifying on its own — a failure here is a placement problem, whereas
the same failure inside a 70B demo is a needle in a haystack.

Two tests, deliberately ordered:

1. the selector alone, on a tensor whose values name their user; and
2. the selector feeding ``Sampling2D``, which is exactly what
   ``<model>.sample_decode`` does.

Run::

    pytest tests/models/galaxy/test_column_user_selector_wh_galaxy.py -v
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import ttnn
from examples.common.auto_compose import to_torch_auto_compose
from tests.models.galaxy.galaxy_hardware import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_USERS_PER_COLUMN,
    deallocate,
)

from tt_transformers.models.galaxy.collectives import GalaxyColumnUserSelector, compose_galaxy_sampled_tokens
from tt_transformers.models.galaxy.recipes import (
    prefetch_sender_cores,
    ring_cores,
    sampling_core_grids,
    width_sharded_memory_config,
    worker_cores,
)
from tt_transformers.modules.sampling.sampling_2d import Sampling2D


def _stage_column_replicated(source: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Shard the width over mesh rows and replicate the users over columns.

    This is the placement ``LMHead2D`` decode output has after its column
    all-reduce: the vocabulary is row-sharded, the physical batch is everywhere.
    """

    return ttnn.from_torch(
        source,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=GALAXY_MESH_SHAPE),
    )


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_column_user_selector_gives_each_column_its_own_users(mesh_device: ttnn.MeshDevice):
    """Column ``c`` must receive exactly users ``8c .. 8c + 7``, in order."""

    width = 256
    source = torch.arange(GALAXY_PHYSICAL_BATCH, dtype=torch.bfloat16).reshape(1, 1, -1, 1).repeat(1, 1, 1, width)
    selector = GalaxyColumnUserSelector(mesh_device)
    staged = selected = None
    try:
        staged = _stage_column_replicated(source, mesh_device)
        for _ in range(2):  # repeat invocation must reuse the cached selector
            selected = selector(staged)
            try:
                composed = to_torch_auto_compose(selected).float()
                message = f"expected the four column slices to compose back to 32 users, got {tuple(composed.shape)}"
                assert tuple(composed.shape[-2:]) == (GALAXY_PHYSICAL_BATCH, width), message
                users = composed.reshape(-1, width)[:GALAXY_PHYSICAL_BATCH, 0]
                message = f"column user order is wrong: {users.tolist()}"
                assert torch.equal(users, torch.arange(GALAXY_PHYSICAL_BATCH, dtype=users.dtype)), message
            finally:
                deallocate(selected)
                selected = None
    finally:
        deallocate(staged)
        selector.release()


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_column_user_selector_feeds_sampling_2d(mesh_device: ttnn.MeshDevice):
    """Selector plus ``Sampling2D`` reproduces a per-user argmax.

    The vocabulary is Llama-3.3-70B's, whose padded width equals its logical
    width, so the assertion is only about user placement.
    """

    vocab_size = padded_vocab_size = 128256
    logits = torch.full((1, 1, GALAXY_PHYSICAL_BATCH, padded_vocab_size), -20.0, dtype=torch.bfloat16)
    expected = torch.arange(GALAXY_PHYSICAL_BATCH, dtype=torch.int64) * 1013
    logits[0, 0, torch.arange(GALAXY_PHYSICAL_BATCH), expected] = 10.0

    sub_core_grids, topk_grid, start_core = sampling_core_grids()
    sampler = Sampling2D(
        vocab_size,
        padded_vocab_size,
        mesh_device,
        sub_core_grids=sub_core_grids,
        sub_core_grid_topk=topk_grid,
        start_core=start_core,
    )
    selector = GalaxyColumnUserSelector(mesh_device)
    staged = selected = output = None
    try:
        staged = _stage_column_replicated(logits, mesh_device)
        selected = selector(staged)
        assert tuple(selected.shape)[-2] == GALAXY_USERS_PER_COLUMN
        output = sampler.decode_forward(selected, top_k=32, top_p=1.0, temperature=0.0, forced_argmax=True)
        actual = to_torch_auto_compose(output).reshape(-1)[:GALAXY_PHYSICAL_BATCH].to(torch.int64)
        assert torch.equal(actual, expected), f"sampled {actual.tolist()}, expected {expected.tolist()}"
    finally:
        deallocate(output)
        deallocate(selected)
        deallocate(staged)
        selector.release()
        sampler.release()


# ---------------------------------------------------------------------------
# The composition gap the two cases above could not see.
#
# Both of them build their input with `memory_config=ttnn.DRAM_MEMORY_CONFIG`
# and with **no sub-device manager loaded** - the one layout a default
# `ttnn.matmul` accepts and the one layout the real model never produces. They
# passed 3/3 on silicon while the production sampling path was failing twice
# over on the same line:
#
#   * `LMHead2D.decode_forward` returns its output under the shared recipe's
#     `lm_head_output_memcfg`, which is `width_sharded_memory_config(
#     padded_local_vocab, ring)` - WIDTH_SHARDED for both models -
#     and the matmul requires `in1` INTERLEAVED;
#   * decode runs under a loaded sub-device manager, and a matmul that
#     resolves its own grid takes the full compute grid, which reaches the
#     `x=0` and `x=4` prefetch sender columns:
#         TT_FATAL @ program.cpp:2205: num_intersections == num_cores
#         Kernel group cores do not match sub device cores
#
# So the cases below stage the input in the LM head's **own** placement, at both
# models' resolved widths, and run the selector under the **decode** sub-device
# partition (`galaxy_decode_mode_plan`'s senders + workers). They are the
# regression tests for both defects: without the fix in
# `GalaxyColumnUserSelector.__call__` the first aborts on the WIDTH_SHARDED
# `in1` and, once that is satisfied, on the sub-device core mismatch.
# ---------------------------------------------------------------------------

#: `(id, vocab_size, padded_local_vocab)` for the two models. The widths are
#: `galaxy_padded_vocab_size(vocab) // 8`, i.e. what a host probe measured on
#: both models: Llama 24 cores of (32, 672), Qwen 24 cores of (32, 800).
_MODEL_VOCABULARIES = [
    pytest.param(128256, 16128, id="llama-3.3-70b"),
    pytest.param(151936, 19200, id="qwen3-32b"),
]


@contextlib.contextmanager
def _loaded_decode_partition(mesh_device: ttnn.MeshDevice):
    """Load the canonical Galaxy decode sub-device manager for the block.

    Senders and workers, `SubDeviceId(1)` as the worker id, stalling on the
    workers - byte for byte what `galaxy_decode_mode_plan` builds and what
    `Prefetcher2D._configure_mode` loads before a decode step. Nothing here
    needs the prefetcher itself: the defect is about which cores a *program* may use,
    and that is decided by the loaded manager alone.
    """

    senders = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in prefetch_sender_cores()])
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([senders]), ttnn.SubDevice([worker_cores()])], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])
    try:
        yield
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)


def _stage_lm_head_decode_output(
    source: torch.Tensor, mesh_device: ttnn.MeshDevice, padded_local_vocab: int
) -> ttnn.Tensor:
    """Stage a tensor exactly as `LMHead2D.decode_forward` leaves its output.

    Vocabulary width-sharded over the eight mesh rows and over the 24 ring cores
    within each device, the physical batch replicated over the four columns.
    """

    return ttnn.from_torch(
        source,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=width_sharded_memory_config(padded_local_vocab, ring_cores()),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=GALAXY_MESH_SHAPE),
    )


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@pytest.mark.parametrize("vocab_size,padded_local_vocab", _MODEL_VOCABULARIES)
@torch.no_grad()
def test_column_user_selector_accepts_the_lm_head_decode_placement(
    mesh_device: ttnn.MeshDevice, vocab_size: int, padded_local_vocab: int
):
    """Both defects at once: the real placement, under the real decode partition."""

    padded_vocab_size = padded_local_vocab * GALAXY_MESH_SHAPE[0]
    source = (
        torch.arange(GALAXY_PHYSICAL_BATCH, dtype=torch.bfloat16)
        .reshape(1, 1, -1, 1)
        .repeat(1, 1, 1, padded_vocab_size)
    )
    selector = GalaxyColumnUserSelector(mesh_device)
    staged = selected = None
    try:
        staged = _stage_lm_head_decode_output(source, mesh_device, padded_local_vocab)
        assert staged.is_sharded(), "the staged input must reproduce the LM head's WIDTH_SHARDED output"
        print(f"[selector] staged {tuple(staged.shape)} {staged.memory_config().memory_layout}", flush=True)
        program_config = selector.resolved_program_config(padded_local_vocab)
        print(
            f"[selector] program config: allowed_worker_cores={program_config.allowed_worker_cores} "
            f"per_core_N={program_config.per_core_N} out_subblock_w={program_config.out_subblock_w}",
            flush=True,
        )
        with _loaded_decode_partition(mesh_device):
            for _ in range(2):  # the cached selector and the cached program config on re-entry
                selected = selector(staged)
                try:
                    assert tuple(selected.shape)[-2] == GALAXY_USERS_PER_COLUMN
                    composed = ttnn.to_torch(
                        selected,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 2), mesh_shape=GALAXY_MESH_SHAPE),
                    ).float()
                    message = f"composed to {tuple(composed.shape)}, expected 32 users of {padded_vocab_size}"
                    assert tuple(composed.shape[-2:]) == (GALAXY_PHYSICAL_BATCH, padded_vocab_size), message
                    users = composed.reshape(-1, padded_vocab_size)[:GALAXY_PHYSICAL_BATCH, 0]
                    message = f"column user order is wrong: {users.tolist()}"
                    assert torch.equal(users, torch.arange(GALAXY_PHYSICAL_BATCH, dtype=users.dtype)), message
                finally:
                    deallocate(selected)
                    selected = None
    finally:
        deallocate(staged)
        selector.release()


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_column_user_selection_is_bit_exact(mesh_device: ttnn.MeshDevice):
    """A one-hot matmul has to be a *copy*, and by default it is not.

    Every case above uses values a `bfloat16` mantissa holds exactly - small
    integers, or a single peak against a flat floor - so none of them can see
    what this does: with `compute_kernel_config` unset, `ttnn.matmul` takes its
    default math fidelity and truncates its inputs' mantissas. Measured over a
    32 x 153600 tensor of decode-logit magnitudes, the "exact row gather"
    changed **4 300 324 of 4 915 200 values**, by up to 0.875.

    A `bfloat16` ulp at magnitude 15 is 0.125, so that is several ulps, and it
    flips an argmax: Qwen's device greedy sampling disagreed with the host
    argmax in 4 of 32 slots by gaps of 0.125 to 0.5, none of them a tie.
    Without `exact_gather_compute_kernel_config` as the selector's default this
    fails on the first assertion.
    """

    padded_local_vocab = 19200
    padded_vocab_size = padded_local_vocab * GALAXY_MESH_SHAPE[0]
    torch.manual_seed(20260829)
    source = (torch.rand(1, 1, GALAXY_PHYSICAL_BATCH, padded_vocab_size) * 40.0 - 20.0).to(torch.bfloat16)
    selector = GalaxyColumnUserSelector(mesh_device)
    staged = selected = None
    try:
        staged = _stage_lm_head_decode_output(source, mesh_device, padded_local_vocab)
        with _loaded_decode_partition(mesh_device):
            selected = selector(staged)
            composed = ttnn.to_torch(
                selected,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 2), mesh_shape=GALAXY_MESH_SHAPE),
            )
        got = composed.reshape(-1, padded_vocab_size)[:GALAXY_PHYSICAL_BATCH].to(torch.float32)
        want = source.reshape(-1, padded_vocab_size).to(torch.float32)
        delta = (got - want).abs()
        changed = int((delta > 0).sum())
        print(
            f"[selector] gather changed {changed}/{want.numel()} values, max |delta| {float(delta.max())}", flush=True
        )
        assert changed == 0, (
            f"the one-hot selector changed {changed} of {want.numel()} values, by up to "
            f"{float(delta.max())}; a gather that is not a copy flips an argmax"
        )
    finally:
        deallocate(selected)
        deallocate(staged)
        selector.release()


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.model
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@pytest.mark.parametrize("vocab_size,padded_local_vocab", _MODEL_VOCABULARIES)
@torch.no_grad()
def test_column_user_selector_feeds_sampling_2d_under_the_decode_partition(
    mesh_device: ttnn.MeshDevice, vocab_size: int, padded_local_vocab: int
):
    """Selector plus `Sampling2D`, in the placement and partition decode uses.

    This is `<model>.sample_decode` with the model taken out: the same input
    layout, the same loaded sub-device manager, the same sampler grids. The
    sampled tokens are composed by their **distribution** and not by their
    labels, for the reason `compose_galaxy_sampled_tokens` documents.
    """

    padded_vocab_size = padded_local_vocab * GALAXY_MESH_SHAPE[0]
    logits = torch.full((1, 1, GALAXY_PHYSICAL_BATCH, padded_vocab_size), -20.0, dtype=torch.bfloat16)
    expected = torch.arange(GALAXY_PHYSICAL_BATCH, dtype=torch.int64) * (vocab_size // GALAXY_PHYSICAL_BATCH)
    logits[0, 0, torch.arange(GALAXY_PHYSICAL_BATCH), expected] = 10.0

    sub_core_grids, topk_grid, start_core = sampling_core_grids()
    sampler = Sampling2D(
        vocab_size,
        padded_vocab_size,
        mesh_device,
        sub_core_grids=sub_core_grids,
        sub_core_grid_topk=topk_grid,
        start_core=start_core,
    )
    selector = GalaxyColumnUserSelector(mesh_device)
    staged = selected = output = None
    try:
        staged = _stage_lm_head_decode_output(logits, mesh_device, padded_local_vocab)
        with _loaded_decode_partition(mesh_device):
            selected = selector(staged)
            assert tuple(selected.shape)[-2] == GALAXY_USERS_PER_COLUMN
            output = sampler.decode_forward(selected, top_k=32, top_p=1.0, temperature=0.0, forced_argmax=True)
            actual = compose_galaxy_sampled_tokens(output, mesh_device=mesh_device, users=GALAXY_PHYSICAL_BATCH).to(
                torch.int64
            )
        print(f"[selector] sampled {actual.tolist()}", flush=True)
        assert torch.equal(actual, expected), f"sampled {actual.tolist()}, expected {expected.tolist()}"
    finally:
        deallocate(output)
        deallocate(selected)
        deallocate(staged)
        selector.release()
        sampler.release()
