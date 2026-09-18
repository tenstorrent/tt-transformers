# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for greedy Galaxy sampling on Blackhole.

Modelled on `test_sampling_2d_wh_galaxy.py`, and deliberately *not* a port of
it, because the pipeline those four tests exercise does not exist here.

`GalaxyCapabilities.has_distributed_sampling` is `False` on the Blackhole
descriptor: `ttnn.sampling`'s pipeline is unavailable, and greedy decoding
routes through **all-gather plus `ttnn.argmax`** instead - which also changes
the shape callers see, from the sampler's `[1, 1, users]` per device to
`ttnn.argmax`'s `[1, 1, 32]` with `keepdim=False`. That is a caller-visible
change, so it is asserted rather than inferred.

**`Sampling2D` cannot express that route.** `decode_forward` has exactly one
terminal op, `ttnn.sampling`, on every path that reaches a device: `forced_argmax`
is not a route but a *value*, normalized in `_update_call_buffers` into
`k_values[slot] = 1` and fed to the same `ttnn.sampling` call, and
`ttnn.manual_seed` is issued unconditionally just above it. There is no branch
that gathers the full logits and takes an argmax, and adding one is a change
under `src/` that this suite does not make. So the suite is written at the level
that *is* testable on hardware today - the all-gather of the logits plus
`ttnn.argmax` over the gathered result, against a CPU argmax - which settles
whether the Blackhole greedy route works at all. What `Sampling2D` would need is
in the report.

The last test in the file is the complementary probe: it calls
`Sampling2D.decode_forward` for real, so that `has_distributed_sampling=False`
is a *measurement* on this chassis rather than a descriptor claim inherited from
the reference port. It is last because it is the only test here expected to
abort, and because it is the only one whose failure mode is not already covered
by the test above it.

Every op in the route below is pinned: `sub_core_grids` on the eltwise,
untilize and argmax programs, `subdevice_id` on the gather. An interleaved
eltwise resolves its cores from `device->compute_with_storage_grid_size()`, and
under the loaded worker sub-device manager that is
`TT_FATAL @ program.cpp:2205 Kernel group cores do not match sub device cores`,
so an unpinned op in this chain is not a style question.
"""

import math

import pytest
import torch
import ttnn
from loguru import logger
from tests.models.galaxy.galaxy_hardware import BHGLX_DEVICE_PARAMS
from tests.modules._bh_galaxy_hardware import (
    bh_galaxy_mode_plan,
    bh_galaxy_resources_config,
    bh_galaxy_topology,
    record_module_output,
    require_bh_galaxy_ccl_resources,
)

from tt_transformers.models.galaxy.recipes import (
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_ROWS,
    TILE,
    galaxy_fabric_links,
    galaxy_padded_vocab_size,
    sampling_core_grids,
    worker_cores,
)
from tt_transformers.models.galaxy.resources import (
    GalaxyCollectivePlan,
    GalaxyResourceKey,
    GalaxyTensorSpec,
)
from tt_transformers.modules.sampling.sampling_2d import Sampling2D
from tt_transformers.sampling.vocab_padding import build_invalid_vocab_mask

#: The two Galaxy geometries. The padded vocabulary is a literal for the same
#: reason the LM-head suite writes it as one: recomputing the expectation from
#: the function under test would pass whatever that function said. `dim` is
#: carried because it names the model, not because sampling reads it.
_GEOMETRIES = [
    pytest.param(8192, 128256, 129024, id="llama-8192"),
    pytest.param(5120, 151936, 153600, id="qwen-5120"),
]


def _deallocate(tensor):
    if tensor is not None:
        tensor.deallocate(True)


def _logits_all_gather_plan(batch, local_vocab, num_links):
    """Return the axis-0 logits gather plan the mode plans are built around.

    `GalaxyModePlan` requires at least one collective and
    `GalaxyCollectivePlan` at least one persistent output, so this exists to
    make the plan legal - and it is the honest shape of the collective this
    suite runs, not a placeholder. Nothing here is consumed: the route uses the
    stable `ttnn.all_gather`, which allocates its own semaphores and its own
    output. What the suite needs from the plan is the **worker envelope and the
    loaded sub-device manager** that `bh_galaxy_mode_plan` builds over it.
    """

    local = (1, 1, batch, local_vocab)
    gathered = (1, 1, batch, local_vocab * GALAXY_ROWS)
    return GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_gather", 0, local, batch),
        topology=ttnn.Topology.Linear,
        num_links=num_links,
        persistent_output_specs=(GalaxyTensorSpec(gathered, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),),
    )


def _worker_resources(mesh_device, topology, collectives):
    """Create CCL-only resources over the prefetcher-free Blackhole envelope.

    `require_bh_galaxy_ccl_resources` starts no prefetch producer, so nothing
    has to be drained - unlike Wormhole, where a suite that starts the producer
    without a consumer passes all its own assertions and then hangs in
    `ttnn.close_mesh_device`.
    """

    prefill = bh_galaxy_mode_plan("prefill", collectives, mesh_device, topology=topology)
    decode = bh_galaxy_mode_plan("decode", collectives, mesh_device, topology=topology)
    config = bh_galaxy_resources_config(mesh_device, prefill=prefill, decode=decode)
    return require_bh_galaxy_ccl_resources(mesh_device, config=config)


def _greedy_logits(vocab_size, padded_vocab_size, call):
    """Return `(logits, expected)` whose winners are spread over all eight rows.

    Two properties are deliberate.

    **Every mesh row owns some user's winning token.** The vocabulary is sharded
    over the eight rows, so a gather that moved no data would leave 28 of the 32
    devices holding no peak for most users and the argmax would be wrong almost
    everywhere. A test whose winners all sat in row 0's shard would pass on a
    no-op gather for every device in row 0.

    **The padding is poisoned with `+1000`, not zeros.** Zeros are too easy to
    confuse with a legitimate result; `+1000` beats every valid logit, so an
    unapplied invalid-vocabulary mask sampled a padded token id rather than
    merely a different one. `call` moves every peak, so a program-cache hit that
    returned an earlier call's tokens is a *wrong* answer rather than an
    accidentally equal one (the lesson of the repeated-decode case).
    """

    batch = GALAXY_PHYSICAL_BATCH
    local_vocab = padded_vocab_size // GALAXY_ROWS
    # The last row's shard runs past `vocab_size`, so the offset inside a shard
    # is bounded by what is still a valid token id there, minus a tile.
    span = vocab_size - (GALAXY_ROWS - 1) * local_vocab - TILE
    assert 0 < span <= local_vocab
    expected = torch.tensor(
        [(user % GALAXY_ROWS) * local_vocab + (user * 619 + call * 97) % span for user in range(batch)],
        dtype=torch.int64,
    )
    assert int(expected.max()) < vocab_size
    logits = torch.full((1, 1, batch, padded_vocab_size), -20.0, dtype=torch.bfloat16)
    logits[0, 0, torch.arange(batch), expected] = 10.0
    logits[..., vocab_size:] = 1000.0
    # A fixture bug would otherwise read as a device defect.
    assert torch.equal(logits[0, 0, :, :vocab_size].float().argmax(dim=-1), expected)
    return logits, expected


def _shard_over_rows(logits, mesh_device, dtype):
    """Place a `[1, 1, batch, padded_vocab]` tensor the way the LM head hands it over.

    Vocabulary over the eight mesh rows, physical batch **replicated** across
    the four columns - `dims=(3, None)`. That is the placement the decode graph
    produces, not one of two equivalent choices: `model.py` builds the sampler
    for it, and the common runtime reshapes the sampled tokens to a
    `ReplicateTensorToMesh` token tensor, 32 wide per device. The
    invalid-vocabulary mask is placed the same way, because it is added to the
    logits shard for shard.
    """

    return ttnn.from_torch(
        logits,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=GALAXY_MESH_SHAPE),
    )


def _device_tokens(tensor, batch):
    """Return one `[devices, batch]` int64 tensor of every device's tokens."""

    shards = ttnn.get_device_tensors(tensor.cpu())
    assert len(shards) == 32, f"expected one shard per device, got {len(shards)}"
    return torch.stack([ttnn.to_torch(shard).reshape(-1)[:batch].to(torch.int64) for shard in shards])


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@torch.no_grad()
def test_sampling_2d_bh_galaxy_greedy_pipeline_and_grids(mesh_device):
    """The capability, the grids, and that both geometries resolve. Cheapest first.

    No device program runs here, and that is the point: every placement
    `Sampling2D` and the argmax route would use is resolved against the
    descriptor `bh_galaxy_topology` has already validated *against this device*,
    so a containment error fails in a second rather than inside a kernel. This
    is the check that would have caught the reference port's own `x=6`
    shard-grid failure - a grid named independently of the partition that has to
    contain it - before it reached silicon.

    Both geometries are constructed in one process on purpose. `Sampling2D`'s
    per-call state is `LazyBuffer`, so nothing is materialized on device and
    there are no compiled programs or buffer addresses to share; the one-node-id
    -per-process rule exists for tests that dispatch, which this one does not.
    """

    topology = bh_galaxy_topology(mesh_device)
    assert topology.capabilities.has_distributed_sampling is False, (
        "the Blackhole descriptor now claims distributed sampling. This suite's whole shape - "
        "all-gather plus ttnn.argmax rather than ttnn.sampling - follows from that flag being "
        "False, and the [1, 1, 32] return shape it asserts is argmax's, not the sampler's"
    )
    assert topology.capabilities.has_prefetcher is False

    workers = worker_cores(topology)
    sub_core_grids, sub_core_grid_topk, start_core = sampling_core_grids(topology)
    assert sub_core_grids == workers
    assert sub_core_grid_topk.subtract(workers).num_cores() == 0, (
        "the top-k grid leaves the worker envelope; on device that is 'Kernel group cores do not "
        "match sub device cores' rather than a wrong answer"
    )
    assert sub_core_grids.subtract(workers).num_cores() == 0
    for name, grid in (("sampling", sub_core_grids), ("top-k", sub_core_grid_topk)):
        for core_range in grid.ranges():
            for core in (core_range.start, core_range.end):
                assert core.x not in topology.reserved_columns, (
                    f"the {name} grid reaches reserved column {core.x}; that column carries dispatch "
                    "and is excluded by measurement, not by derivation"
                )
    box = workers.bounding_box()
    assert (start_core.x, start_core.y) == (box.start.x, box.start.y), (
        "Sampling2D anchors its sub-grids at the first worker core; an anchor outside the envelope "
        "places shards on cores belonging to no sub-device"
    )

    # Two links, not four. Over-requesting deadlocks with no traceback.
    num_links = galaxy_fabric_links(mesh_device)
    assert num_links == 2
    assert topology.ccl_reserved_worker_cores == num_links

    resolved = []
    samplers = []
    try:
        for _dim, vocab_size, padded_vocab_size in (geometry.values for geometry in _GEOMETRIES):
            assert galaxy_padded_vocab_size(vocab_size) == padded_vocab_size, (
                "Blackhole keeps the Wormhole padded vocabulary; if this moved, the ring size moved "
                "and every constraint stated in terms of ring_size has to be re-walked"
            )
            sampler = Sampling2D(
                vocab_size,
                padded_vocab_size,
                mesh_device,
                sub_core_grids=sub_core_grids,
                sub_core_grid_topk=sub_core_grid_topk,
                start_core=start_core,
                replicate_users=True,
                num_gather_links=num_links,
            )
            samplers.append(sampler)
            config = sampler.config
            assert config.padded_vocab_size == padded_vocab_size
            assert config.users_per_shard == GALAXY_PHYSICAL_BATCH, (
                "with the batch replicated across the four columns every device holds all 32 users"
            )
            assert config.invalid_vocab_mask is not None, (
                "the ring-exact padding is wider than the vocabulary on both geometries, so the "
                "invalid-logits mask is load-bearing - for Llama-3.3-70B for the first time"
            )
            resolved.append(padded_vocab_size // GALAXY_ROWS)
    finally:
        for sampler in samplers:
            sampler.release()

    logger.info(
        f"blackhole sampling grids: workers={workers.num_cores()} topk={sub_core_grid_topk.num_cores()} "
        f"start=({start_core.x}, {start_core.y}) links={num_links} local_vocab={resolved}"
    )
    record_module_output(
        "sampling_2d_bh_galaxy_greedy_pipeline_and_grids",
        torch.tensor(
            [
                workers.num_cores(),
                sub_core_grid_topk.num_cores(),
                start_core.x,
                start_core.y,
                num_links,
                *resolved,
            ],
            dtype=torch.int64,
        ),
    )


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("dim,vocab_size,padded_vocab_size", _GEOMETRIES)
@torch.no_grad()
def test_sampling_2d_bh_galaxy_all_gather_argmax_greedy(mesh_device, dim, vocab_size, padded_vocab_size):
    """The Blackhole greedy route, end to end, against a CPU argmax.

    This is the route `has_distributed_sampling=False` implies, run at the level
    that is testable today: typecast the `bfloat8_b` logits the LM head
    produces, add the invalid-vocabulary mask, all-gather the vocabulary axis,
    untilize, argmax. Every op is pinned to the worker sub-device.

    Three calls, with the peaks moved between them. The first warms the program
    cache; the two after it are the ones that were wrong on Wormhole when an op
    in the chain resolved its placement to a *default* - fast dispatch orders
    programs per sub-device and nothing wider, so the accidental serializers
    that hide such a defect (the kernel-binary load, the gather's semaphore
    `Synchronize`, the host JIT gap) all exist only while the cache is cold.

    The reference is `torch.argmax` on host. Never another TT path: a comparison
    between two device paths would agree with itself while both were wrong.
    """

    batch = GALAXY_PHYSICAL_BATCH
    topology = bh_galaxy_topology(mesh_device)
    assert galaxy_padded_vocab_size(vocab_size) == padded_vocab_size
    local_vocab = padded_vocab_size // GALAXY_ROWS
    workers = worker_cores(topology)
    num_links = galaxy_fabric_links(mesh_device)
    assert num_links == 2, f"Blackhole Galaxy has two fabric links per direction, not {num_links}"

    mask = build_invalid_vocab_mask(vocab_size, padded_vocab_size, batch)
    assert mask is not None, "the ring-exact padding always leaves an invalid tail to mask"

    resources = _worker_resources(mesh_device, topology, (_logits_all_gather_plan(batch, local_vocab, num_links),))
    tt_mask = None
    recorded = None
    try:
        resources.activate("decode")
        worker_sub_device_id = resources.context("decode").worker_sub_device_id
        tt_mask = _shard_over_rows(mask, mesh_device, ttnn.bfloat16)

        for call in range(3):
            logits, expected = _greedy_logits(vocab_size, padded_vocab_size, call)
            tt_logits = _shard_over_rows(logits, mesh_device, ttnn.bfloat8_b)
            typecast = masked = gathered = untilized = tokens = None
            try:
                assert tt_logits.dtype == ttnn.bfloat8_b, (
                    "the Galaxy LM head hands the sampler bfloat8_b logits, and untilize cannot take "
                    "a block-float tensor, so the typecast below is part of the route rather than a "
                    "convenience"
                )
                typecast = ttnn.typecast(tt_logits, dtype=ttnn.bfloat16, sub_core_grids=workers)
                masked = ttnn.add(
                    typecast,
                    tt_mask,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    sub_core_grids=workers,
                )
                gathered = ttnn.all_gather(
                    masked,
                    dim=3,
                    num_links=num_links,
                    cluster_axis=0,
                    topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    subdevice_id=worker_sub_device_id,
                )
                assert tuple(gathered.shape) == (1, 1, batch, padded_vocab_size), (
                    f"the axis-0 gather produced {tuple(gathered.shape)}; every device must hold the "
                    f"whole padded vocabulary before the argmax"
                )
                # A single untilize of the full row, deliberately. The chunked
                # path exists for unpinned Wormhole grids; on a Blackhole worker
                # sub-device the DRAM-interleaved `ttnn.split` it needs does not
                # honour `sub_core_grids` and aborts, while a single untilize of
                # a wider row than either geometry's is measured to compile there
                # (`tt_sampling.py`, the 155648-wide Qwen Galaxy case).
                untilized = ttnn.untilize(gathered, use_multicore=True, sub_core_grids=workers)
                tokens = ttnn.argmax(untilized, dim=-1, keepdim=False, sub_core_grids=workers)
                # The return shape, which is the caller-visible difference from
                # `ttnn.sampling` and the reason the capability flag is not
                # cosmetic. Asserted as "32 tokens, batch-major" rather than
                # literally `(1, 1, 32)` so that a leading unit dimension ttnn
                # may or may not keep does not fail a run that computed the
                # right answer; the exact shape is logged for the record.
                shape = tuple(tokens.shape)
                assert shape[-1] == batch and math.prod(shape) == batch, (
                    f"the Blackhole greedy route returns argmax's shape; expected {batch} tokens in a "
                    f"batch-major shape like (1, 1, {batch}) and got {shape}"
                )
                logger.info(f"call {call}: greedy route returned shape {shape} dtype {tokens.dtype}")

                if call == 0:
                    # Localize before trusting the tokens: the gather's own
                    # output, read back per device. Aggregate agreement would
                    # not distinguish a row whose shard never arrived from one
                    # whose values are merely poor, and the winners are spread
                    # over all eight rows precisely so that it can. The three
                    # indices are positions in `get_device_tensors` order, not a
                    # claim about which (row, column) each one is - every device
                    # must hold the whole gathered vocabulary, so the check does
                    # not need the mapping.
                    shards = ttnn.get_device_tensors(gathered.cpu())
                    assert len(shards) == 32
                    for index in (0, 1, 31):
                        shard = ttnn.to_torch(shards[index]).float().reshape(batch, padded_vocab_size)
                        assert torch.all(shard[:, vocab_size:] < -1e30), (
                            f"device {index}: the invalid-vocabulary mask did not survive the gather; "
                            f"max value in the padded tail is {float(shard[:, vocab_size:].max())}"
                        )
                        valid = shard[:, :vocab_size]
                        assert torch.equal(valid.argmax(dim=-1), expected), (
                            f"device {index} gathered logits whose per-user maxima are "
                            f"{valid.argmax(dim=-1).tolist()}, expected {expected.tolist()}; a wrong "
                            "row block means the gather's per-device channel order is wrong, not that "
                            "its numerics are"
                        )
                        assert int((valid == 10.0).sum()) == batch, (
                            f"device {index} holds {int((valid == 10.0).sum())} peaks, expected one "
                            "per user; a duplicated row block would show up here"
                        )
                        logger.info(f"device {index} gathered logits: {tuple(shard.shape)}, peaks located")

                actual = _device_tokens(tokens, batch)
                for index in range(actual.shape[0]):
                    assert torch.equal(actual[index], expected), (
                        f"call {call}, device {index} sampled {actual[index].tolist()}, expected {expected.tolist()}"
                    )
                assert torch.all(actual < vocab_size)
                logger.info(f"call {call}: all 32 devices agree with the CPU argmax")
                recorded = actual
            finally:
                for tensor in (tokens, untilized, gathered, masked, typecast, tt_logits):
                    _deallocate(tensor)
        resources.synchronize("decode")
    finally:
        _deallocate(tt_mask)
        resources.cleanup()

    record_module_output(
        f"sampling_2d_bh_galaxy_all_gather_argmax_{'llama8192' if dim == 8192 else 'qwen5120'}",
        recorded,
    )


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("dim,vocab_size,padded_vocab_size", _GEOMETRIES[:1])
@torch.no_grad()
def test_sampling_2d_bh_galaxy_distributed_sampling_probe(mesh_device, dim, vocab_size, padded_vocab_size):
    """Call `Sampling2D.decode_forward` for real, to measure the capability flag.

    **Schedule this last.** `has_distributed_sampling=False` is currently a
    descriptor claim carried from the reference port, and the whole shape of
    this file follows from it, so it is worth one node id to turn it into a
    reading on this chassis. Two outcomes are useful: if the chain aborts, the
    flag is confirmed and `Sampling2D` needs the source change named in the
    report; if it passes, Blackhole needs no new sampling route at all and this
    file's other test becomes a redundant belt.

    It is the only test here expected to abort, which is why it is last and why
    it is one geometry rather than two. The fabric risk is bounded: the only
    collectives in the chain are the two axis-0 `ttnn.all_gather` calls that
    `..._all_gather_argmax_greedy` above already exercises, and `ttnn.sampling`
    and `ttnn.manual_seed` are single-device programs that cannot hang on the
    fabric - they abort or they answer.

    `forced_argmax=True` is passed, and is expected *not* to be sufficient on
    its own: `_update_call_buffers` turns it into `k_values[slot] = 1` inside
    the same `ttnn.sampling` call rather than routing around it.
    """

    batch = GALAXY_PHYSICAL_BATCH
    topology = bh_galaxy_topology(mesh_device)
    assert topology.capabilities.has_distributed_sampling is False
    local_vocab = padded_vocab_size // GALAXY_ROWS
    num_links = galaxy_fabric_links(mesh_device)
    sub_core_grids, sub_core_grid_topk, start_core = sampling_core_grids(topology)

    resources = _worker_resources(mesh_device, topology, (_logits_all_gather_plan(batch, local_vocab, num_links),))
    sampler = None
    tt_logits = None
    output = None
    try:
        resources.activate("decode")
        worker_sub_device_id = resources.context("decode").worker_sub_device_id
        sampler = Sampling2D(
            vocab_size,
            padded_vocab_size,
            mesh_device,
            sub_core_grids=sub_core_grids,
            sub_core_grid_topk=sub_core_grid_topk,
            start_core=start_core,
            replicate_users=True,
            num_gather_links=num_links,
            ccl_sub_device_id=lambda: worker_sub_device_id,
        )
        logits, expected = _greedy_logits(vocab_size, padded_vocab_size, 0)
        tt_logits = _shard_over_rows(logits, mesh_device, ttnn.bfloat8_b)
        output = sampler.decode_forward(
            tt_logits,
            top_k=32,
            top_p=1.0,
            temperature=0.0,
            seed=7,
            forced_argmax=True,
        )
        actual = _device_tokens(output, batch)
        for index in range(actual.shape[0]):
            assert torch.equal(actual[index], expected), (
                f"device {index} sampled {actual[index].tolist()}, expected {expected.tolist()}"
            )
        assert torch.all(actual < vocab_size)
        logger.info("ttnn.sampling ran on Blackhole Galaxy and agreed with the CPU argmax on all 32 devices")
        record_module_output(
            f"sampling_2d_bh_galaxy_distributed_sampling_probe_{'llama8192' if dim == 8192 else 'qwen5120'}",
            actual,
        )
    finally:
        _deallocate(output)
        _deallocate(tt_logits)
        if sampler is not None:
            sampler.release()
        resources.cleanup()
