# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for Sampling2D on a Wormhole Galaxy."""

import contextlib

import pytest
import torch
import ttnn
from examples.common.auto_compose import to_torch_auto_compose

from tt_transformers.device_utils import GALAXY_L1_SMALL_SIZE
from tt_transformers.models.galaxy.recipes import (
    prefetch_sender_cores,
    ring_cores,
    sampling_core_grids,
    width_sharded_memory_config,
    worker_cores,
)
from tt_transformers.modules.sampling.sampling_2d import Sampling2D


def _deallocate(tensor):
    if tensor is not None:
        tensor.deallocate(True)


@contextlib.contextmanager
def _decode_partition(mesh_device):
    """Load the production decode sender/worker partition and stall on the worker."""

    senders = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in prefetch_sender_cores()])
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([senders]), ttnn.SubDevice([worker_cores()])], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])
    try:
        yield ttnn.SubDeviceId(1)
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "l1_small_size": GALAXY_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)
def test_sampling_2d_wh_galaxy_exact_padded_vocab_exclusion(mesh_device):
    vocab_size = 151936
    padded_vocab_size = 152064
    batch = 32
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    sub_core_grid_topk = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9))])
    sampler = Sampling2D(
        vocab_size,
        padded_vocab_size,
        mesh_device,
        sub_core_grids=sub_core_grids,
        sub_core_grid_topk=sub_core_grid_topk,
        start_core=ttnn.CoreCoord(1, 0),
    )
    logits = torch.full((1, 1, batch, padded_vocab_size), -20.0, dtype=torch.bfloat16)
    expected = torch.arange(batch, dtype=torch.int64) * 97
    logits[0, 0, torch.arange(batch), expected] = 10.0
    logits[..., vocab_size:] = 1000.0
    tt_logits = ttnn.from_torch(
        logits,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=(8, 4)),
    )

    try:
        for _ in range(2):
            output = sampler.decode_forward(
                tt_logits,
                top_k=32,
                top_p=1.0,
                temperature=0.0,
                seed=7,
                forced_argmax=True,
            )
            try:
                actual = to_torch_auto_compose(output).reshape(-1)[:batch].to(torch.int64)
                assert torch.equal(actual, expected)
                assert torch.all(actual < vocab_size)
            finally:
                _deallocate(output)
    finally:
        _deallocate(tt_logits)
        sampler.release()


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "l1_small_size": GALAXY_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)
def test_sampling_2d_wh_galaxy_subdevice_ordering_on_program_cache_hit(mesh_device):
    """Repeated decode sampling stays correct under the loaded decode partition.

    The chain crosses no sub-device boundary only because `ccl_sub_device_id` and
    `sub_core_grids` say so. Given neither, `ttnn.all_gather` and the int32 `ttnn.add`
    resolve their placement to `get_sub_device_ids()` position 0 - the *prefetch sender*
    partition - and fast dispatch, which orders programs per sub-device and nothing
    wider, never orders them against the `topk` that feeds them. The accidental
    serializers that hide this (kernel-binary load stalling on the stall group, the
    gather's semaphore `Synchronize`, the host JIT gap) all exist only while the program
    cache is cold, so call 0 is right and calls 1..3 return an earlier call's tokens.

    Four calls with distinct peaks is therefore the whole test: one to warm the cache,
    three that were wrong before the placements were named. Every assertion is on the
    sampled tokens alone - no barrier, no intermediate readback - because a host
    round-trip mid-chain drains the device and masks exactly what this covers.
    """

    vocab_size = 151936
    padded_vocab_size = 152064
    batch = 32
    calls = 4
    sub_core_grids, sub_core_grid_topk, start_core = sampling_core_grids()

    with _decode_partition(mesh_device) as worker_sub_device_id:
        sampler = Sampling2D(
            vocab_size,
            padded_vocab_size,
            mesh_device,
            sub_core_grids=sub_core_grids,
            sub_core_grid_topk=sub_core_grid_topk,
            start_core=start_core,
            ccl_sub_device_id=lambda: worker_sub_device_id,
        )
        try:
            for call in range(calls):
                # Peaks move every call, so a stale answer is a *wrong* answer rather
                # than an accidentally equal one.
                expected = (torch.arange(batch, dtype=torch.int64) * 97 + call) % vocab_size
                logits = torch.full((1, 1, batch, padded_vocab_size), -20.0, dtype=torch.bfloat16)
                logits[0, 0, torch.arange(batch), expected] = 10.0
                logits[..., vocab_size:] = 1000.0
                tt_logits = ttnn.from_torch(
                    logits,
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=(8, 4)),
                )
                output = None
                try:
                    output = sampler.decode_forward(
                        tt_logits,
                        top_k=32,
                        top_p=1.0,
                        temperature=0.0,
                        seed=7,
                        forced_argmax=True,
                    )
                    actual = to_torch_auto_compose(output).reshape(-1)[:batch].to(torch.int64)
                    assert torch.equal(actual, expected), f"call {call} sampled an earlier call's tokens"
                finally:
                    _deallocate(output)
                    _deallocate(tt_logits)
        finally:
            sampler.release()


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "l1_small_size": GALAXY_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)
def test_sampling_2d_wh_galaxy_bfloat8_b_logits_under_the_decode_partition(mesh_device):
    """Sampled decode on the LM head's own dtype, inside the loaded decode partition.

    Both tests above build their logits as `bfloat16`, so
    `Sampling2D.decode_forward`'s first branch — the typecast it takes when the
    logits are *not* already `bfloat16` — has never run on this mesh. The Galaxy
    LM head hands the executor **BFLOAT8_B** decode logits, so serving takes that
    branch on every sampled step, and it aborted the moment anything got far
    enough to reach it:

        ttnn.typecast(<[1,1,32,16128] BFLOAT8_B TILE, interleaved>,
                      dtype=BFLOAT16, sub_core_grids={[1-0 - 3-9], [5-0 - 6-9]})
        TT_FATAL @ program.cpp:2205: num_intersections == num_cores
        Kernel group cores do not match sub device cores for TENSIX

    Two conditions have to hold together, which is why neither existing test
    catches it: the logits must need the typecast, **and** a decode sub-device
    manager must be loaded. `..._exact_padded_vocab_exclusion` has neither and
    `..._subdevice_ordering_on_program_cache_hit` has only the second.
    """

    vocab_size = 151936
    padded_vocab_size = 152064
    batch = 32
    sub_core_grids, sub_core_grid_topk, start_core = sampling_core_grids()

    with _decode_partition(mesh_device) as worker_sub_device_id:
        sampler = Sampling2D(
            vocab_size,
            padded_vocab_size,
            mesh_device,
            sub_core_grids=sub_core_grids,
            sub_core_grid_topk=sub_core_grid_topk,
            start_core=start_core,
            ccl_sub_device_id=lambda: worker_sub_device_id,
        )
        # A peak of 10.0 in a block of -20.0 survives bfloat8_b's shared
        # exponent, so the argmax is exact even though the values are not.
        expected = torch.arange(batch, dtype=torch.int64) * 97
        logits = torch.full((1, 1, batch, padded_vocab_size), -20.0, dtype=torch.bfloat16)
        logits[0, 0, torch.arange(batch), expected] = 10.0
        logits[..., vocab_size:] = 1000.0
        tt_logits = ttnn.from_torch(
            logits,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=(8, 4)),
        )
        try:
            assert tt_logits.dtype == ttnn.bfloat8_b, "this test is only a test while the input needs a typecast"
            output = sampler.decode_forward(
                tt_logits,
                top_k=32,
                top_p=1.0,
                temperature=0.0,
                seed=7,
                forced_argmax=True,
            )
            try:
                actual = to_torch_auto_compose(output).reshape(-1)[:batch].to(torch.int64)
                assert torch.equal(actual, expected)
                assert torch.all(actual < vocab_size)
            finally:
                _deallocate(output)
        finally:
            _deallocate(tt_logits)
            sampler.release()


#: The Galaxy decode LM head's own output placement: `[32, 16128]` of padded
#: local vocabulary, width-sharded in L1 over the 24-core `gather_in0` ring
#: (`recipes.ring_cores`, `lm_head_2d.py:330`). Those two numbers are Llama's
#: `129024 // 8` and the ring's `16128 // 24 = 672` columns per core, i.e. 21
#: whole tiles, which is why this test uses Llama's vocabulary and the two above
#: use Qwen's: `152064 // 8 // 24` is 792 columns, and a width-sharded L1 shard
#: must be a whole number of tiles wide.
_LLAMA_VOCAB_SIZE = 128256
_LLAMA_PADDED_VOCAB_SIZE = 129024


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "l1_small_size": GALAXY_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)
def test_sampling_2d_wh_galaxy_ring_sharded_logits_under_the_decode_partition(mesh_device):
    """Sampled decode on the LM head's own *placement*, inside the decode partition.

    This test and the one above it differ in exactly one thing: this one's
    logits are **width-sharded in L1 over the ring**, which is what the Galaxy
    decode LM head actually hands the sampler, and the one above's are
    interleaved in DRAM. That single difference is the whole of the defect this
    covers.

    `ttnn.typecast` takes its `sub_core_grids` into account only for an
    *interleaved* input. Given a sharded one it resolves the program's cores
    from the shard spec, and for a `CoreRangeSet` of 24 scattered single cores
    that is the **bounding box** `(1,0)-(6,9)`. Under the loaded decode
    partition - senders `x in {0, 4}` at eight of ten rows, workers
    `x in {1..3, 5..6}` at all ten - exactly two cores of that box, `(4,3)` and
    `(4,8)`, belong to no sub-device, so 58 of 60 cores intersect and the
    program aborts before it runs:

        TT_FATAL @ program.cpp:2205: num_intersections == num_cores
        Kernel group cores do not match sub device cores for TENSIX

    Measured three ways rather than argued: the executor aborts at
    `sampling_2d.py:445`; the identical call on an *interleaved* tensor of the
    same shape and dtype under the identical partition **places**; and printing
    the real tensor's memory config at the call site and then trying seven
    placements of it in situ leaves only the two that go through an interleaved
    tensor first surviving.
    """

    batch = 32
    sub_core_grids, sub_core_grid_topk, start_core = sampling_core_grids()
    ring = ring_cores()
    local_vocab = _LLAMA_PADDED_VOCAB_SIZE // 8

    with _decode_partition(mesh_device) as worker_sub_device_id:
        sampler = Sampling2D(
            _LLAMA_VOCAB_SIZE,
            _LLAMA_PADDED_VOCAB_SIZE,
            mesh_device,
            sub_core_grids=sub_core_grids,
            sub_core_grid_topk=sub_core_grid_topk,
            start_core=start_core,
            ccl_sub_device_id=lambda: worker_sub_device_id,
        )
        # A peak of 10.0 in a block of -20.0 survives bfloat8_b's shared
        # exponent, so the argmax is exact even though the values are not.
        expected = torch.arange(batch, dtype=torch.int64) * 97
        logits = torch.full((1, 1, batch, _LLAMA_PADDED_VOCAB_SIZE), -20.0, dtype=torch.bfloat16)
        logits[0, 0, torch.arange(batch), expected] = 10.0
        logits[..., _LLAMA_VOCAB_SIZE:] = 1000.0
        interleaved = ttnn.from_torch(
            logits,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=(8, 4)),
        )
        tt_logits = None
        try:
            # The LM head produces this placement on device; building it the
            # same way keeps the test's input a device tensor of the LM head's
            # own memory config rather than a host round trip's.
            tt_logits = ttnn.interleaved_to_sharded(interleaved, width_sharded_memory_config(local_vocab, ring))
            assert tt_logits.memory_config().is_sharded(), "this test is only a test while the input is sharded"
            assert tt_logits.dtype == ttnn.bfloat8_b, "this test is only a test while the input needs a typecast"
            output = sampler.decode_forward(
                tt_logits,
                top_k=32,
                top_p=1.0,
                temperature=0.0,
                seed=7,
                forced_argmax=True,
            )
            try:
                actual = to_torch_auto_compose(output).reshape(-1)[:batch].to(torch.int64)
                assert torch.equal(actual, expected)
                assert torch.all(actual < _LLAMA_VOCAB_SIZE)
            finally:
                _deallocate(output)
        finally:
            _deallocate(tt_logits)
            _deallocate(interleaved)
            sampler.release()


@pytest.mark.device
@pytest.mark.wormhole
@pytest.mark.galaxy_wh
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "l1_small_size": GALAXY_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)
def test_sampling_2d_wh_galaxy_replicated_users_on_the_lm_head_placement(mesh_device):
    """The placement the Galaxy decode graph actually produces.

    Every other test in this file shards the physical batch over the four mesh
    columns, so each device holds `32 // 4 == 8` users. **The decode graph does
    not do that.** It replicates the batch across the columns and shards the
    *vocabulary* over the eight rows, so `lm_head_2d` hands the sampler all 32
    users and one eighth of the vocabulary on every device - `model.py` says so
    at the site where it builds the sampler's config, and the common runtime
    requires the same of the output, reshaping the sampled tokens to the shape
    of a `ReplicateTensorToMesh` token tensor (`decode.py:693`), which is 32
    wide per device.

    With the sharded default that mismatch reaches the vocabulary mask add and
    cannot be broadcast:

        ttnn.add([1,1,32,16128] bf16, [1,1,8,16128] bf16)
        TT_THROW @ binary_ng_device_operation.cpp:224 Invalid subtile broadcast type

    So this asserts **exact argmax correctness for all 32 users, on every one of
    the 32 devices**, not merely that nothing aborts: a placement that merely
    stops aborting would still sample the wrong rows, and a token comparison
    between two execution paths could not tell the difference.
    """

    batch = 32
    sub_core_grids, sub_core_grid_topk, start_core = sampling_core_grids()
    ring = ring_cores()
    local_vocab = _LLAMA_PADDED_VOCAB_SIZE // 8

    with _decode_partition(mesh_device) as worker_sub_device_id:
        sampler = Sampling2D(
            _LLAMA_VOCAB_SIZE,
            _LLAMA_PADDED_VOCAB_SIZE,
            mesh_device,
            sub_core_grids=sub_core_grids,
            sub_core_grid_topk=sub_core_grid_topk,
            start_core=start_core,
            replicate_users=True,
            ccl_sub_device_id=lambda: worker_sub_device_id,
        )
        expected = torch.arange(batch, dtype=torch.int64) * 97
        logits = torch.full((1, 1, batch, _LLAMA_PADDED_VOCAB_SIZE), -20.0, dtype=torch.bfloat16)
        logits[0, 0, torch.arange(batch), expected] = 10.0
        logits[..., _LLAMA_VOCAB_SIZE:] = 1000.0
        # Vocabulary over the eight rows, batch replicated across the four
        # columns: the decode graph's own placement. Every winning token here is
        # below `local_vocab`, so only mesh row 0 holds a peak and the answer
        # exists on the other seven rows only after the all-gather.
        interleaved = ttnn.from_torch(
            logits,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=(8, 4)),
        )
        tt_logits = None
        try:
            tt_logits = ttnn.interleaved_to_sharded(interleaved, width_sharded_memory_config(local_vocab, ring))
            assert tuple(tt_logits.shape)[-2:] == (batch, local_vocab), "the batch is not replicated per device"
            output = sampler.decode_forward(
                tt_logits,
                top_k=32,
                top_p=1.0,
                temperature=0.0,
                seed=7,
                forced_argmax=True,
            )
            try:
                shards = ttnn.get_device_tensors(output.cpu())
                assert len(shards) == 32, f"expected one shard per device, got {len(shards)}"
                for index, shard in enumerate(shards):
                    actual = ttnn.to_torch(shard).reshape(-1)[:batch].to(torch.int64)
                    assert torch.equal(actual, expected), f"device {index} sampled {actual.tolist()}"
                    assert torch.all(actual < _LLAMA_VOCAB_SIZE)
            finally:
                _deallocate(output)
        finally:
            _deallocate(tt_logits)
            _deallocate(interleaved)
            sampler.release()
