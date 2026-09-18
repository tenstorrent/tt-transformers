# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for LMHead2D on a Blackhole Galaxy.

Modelled on `test_lm_head_2d_wh_galaxy.py`, and the differences are all recipe,
not style.

**The vocabulary padding is expected to be unchanged, and that expectation is
the test.** `galaxy_padded_vocab_size` pads to `GALAXY_ROWS * RING_ALIGNMENT`,
and `RING_ALIGNMENT` is `TILE * RING_CORE_COUNT` - a function of the ring size.
The Galaxy reference holds `RING_SIZE = 24` on Blackhole *deliberately*, keeping
all the weight-sharding math by widening receivers-per-reader from 2 to 3
(`8 * 3 = 24 = 12 * 2`), so Llama-3.3-70B still pads 128256 -> 129024 and
Qwen3-32B still pads 151936 -> 153600. Beware the `24 -> 16` figure that
circulates for Blackhole: that is TTTv1's **1D** LM head, a different path with
different weight sharding, and carrying it into Galaxy vocabulary arithmetic
breaks the invariant `test_vocabulary_padding_is_identical_on_both_architectures`
asserts on the host and this file asserts on the device.

**The ring *size* survives; the ring *path* does not.** `has_ring_matmul` is
`False` here, `ring_core_coords` is `None`, and the 24-core `gather_in0` matmul
is therefore unavailable. That is exactly what the prefetcher-free LM head is:
`LMHead2D`'s own defaults - `decode_program_configs` resolved to `(None,)`,
`decode_input_memcfg`/`decode_output_memcfg` interleaved L1, prefill interleaved
DRAM, weights DRAM-interleaved - which is the `program_config=None`
DRAM-interleaved matmul the milestone-1 recipe calls for. No source change is
needed to express it; the three-argument constructor *is* it.

**No sub-device manager is loaded, and that is forced rather than chosen.**
`LMHead2D._project` finishes with `ttnn.add(logits, staged_mask, ...)`, which
takes no `sub_core_grids`. On Wormhole that never mattered: the Galaxy decode
output placement is width-sharded over the ring, so `decode_stage_mask` is true
and the add runs on the logits' own cores, inside the partition. Prefetcher-free
Blackhole has no ring to shard onto, the output is interleaved, and an
interleaved eltwise resolves its cores from
`device->compute_with_storage_grid_size()` - which under a loaded worker
sub-device manager is `TT_FATAL @ program.cpp:2205 Kernel group cores do not
match sub device cores`. So the whole suite runs with no manager loaded, which
in turn rules out `all_reduce_async` for the column reduction: its senders would
be chosen from the default sub-device while the semaphores sit on the worker
envelope, and a sender on an uncovered core **hangs**.
The reduction therefore goes through the same plain `ttnn.all_reduce` the
Wormhole suite qualified. What is missing from `src/` to run this module inside
the worker partition is recorded rather than worked around here.
"""

import gc

import pytest
import torch
import ttnn
from loguru import logger
from tests.models.galaxy.galaxy_hardware import BHGLX_DEVICE_PARAMS
from tests.modules._bh_galaxy_hardware import bh_galaxy_topology, record_module_output
from tests.support.comparison import comp_pcc

from tt_transformers.models.galaxy.recipes import (
    GALAXY_COLUMNS,
    GALAXY_MESH_SHAPE,
    GALAXY_ROWS,
    RING_ALIGNMENT,
    RING_CORE_COUNT,
    TILE,
    galaxy_fabric_links,
    galaxy_padded_vocab_size,
    lm_head_reduce_core_count,
    pad_ring_width,
    ring_cores,
    worker_cores,
)
from tt_transformers.modules.lazy_weight import LazyWeight
from tt_transformers.modules.lm_head.lm_head_2d import LMHead2D

#: The two Galaxy geometries, with the padded vocabulary written as a literal.
#:
#: A literal, and not `galaxy_padded_vocab_size(vocab)`, because a test that
#: recomputed the expectation from the function under test would pass whatever
#: the function said. These are the numbers the Wormhole suites and the host
#: gates carry, and the claim of this file is that Blackhole does not move them.
_GEOMETRIES = [
    pytest.param(8192, 128256, 129024, id="llama-8192"),
    pytest.param(5120, 151936, 153600, id="qwen-5120"),
]

#: Checkpoint-vocabulary padding, which is **not** the ring-exact padding and is
#: pinned here so the two can never be conflated. Qwen3-32B's checkpoint pads
#: 151936 -> 152064 for tile/device alignment; the ring-exact width is 153600.
#: Llama-3.3-70B's 128256 is already a multiple of `GALAXY_ROWS * TILE`, so its
#: checkpoint padding is a no-op and only the ring-exact rule pads it at all -
#: which is why the invalid-logits mask is load-bearing for Llama for the first
#: time on this path.
_CHECKPOINT_PADDED_VOCAB = {128256: 128256, 151936: 152064}


class _ColumnAllReduce:
    """Synchronous adapter satisfying LMHead2D's borrowed-input/owned-output contract.

    Identical to the Wormhole suite's adapter except for `num_links`, which is
    **read off the architecture rather than named**: `galaxy_fabric_links`
    returns 4 on Wormhole and 2 on Blackhole, because the Blackhole mesh graph
    descriptor declares `channels { count: 2 }` and a live mesh reported
    `get_num_links` of 2 on both cluster axes. The ring and line CCLs index
    ethernet channels by link, so over-requesting does not degrade - it
    **deadlocks the host with no traceback**.

    `cluster_axis=1` is a column reduction, which on Blackhole runs on device
    and needs `FABRIC_2D_TORUS_XY`; `BHGLX_DEVICE_PARAMS` opens the mesh with it.
    The Wormhole `FABRIC_1D_RING` throws `IndexError: map::at` on the first
    cross-column route.
    """

    cluster_axis = 1
    consumes_input = False
    returns_owned_output = True

    def __init__(self, mesh_device, *, num_links):
        self.mesh_device = mesh_device
        self.num_links = num_links

    def __call__(self, tensor):
        return ttnn.all_reduce(
            tensor,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=ttnn.Topology.Linear,
            memory_config=tensor.memory_config(),
        )


def _deallocate(tensor):
    if tensor is not None:
        tensor.deallocate(True)


def _compose_columns(tensor, mesh_device, padded_vocab_size):
    """Return `[1, GALAXY_COLUMNS, 32, padded_vocab]`, one column replica each.

    The LM head shards the vocabulary over the eight mesh rows and replicates
    the reduced logits across the four mesh columns, which is the mirror image
    of `compose_2d_sharded_tensor`: mesh axis 0 concatenates into tensor dim 3
    and mesh axis 1 into tensor dim 1. Keeping the replicas rather than dropping
    them is deliberate - they are the only evidence that the column reduction
    produced the *same* answer everywhere, and a per-column check is what
    distinguishes a channel-order permutation from bad numerics.
    """

    composed = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=GALAXY_MESH_SHAPE),
    )
    assert tuple(composed.shape) == (1, GALAXY_COLUMNS, 32, padded_vocab_size), (
        f"composed logits have shape {tuple(composed.shape)}; expected one column replica of "
        f"[32, {padded_vocab_size}] per mesh column"
    )
    return composed.float()


def _row_correlations(reference, actual, vocab_size, local_vocab):
    """Return the own-row PCC of every mesh row's vocabulary shard.

    Aggregate PCC is close to useless for a column-local sharding bug: those
    land in 0 - 0.01 and say nothing about where. Correlating each mesh row's
    own slice localizes it, and the off-diagonal report below turns the *shape*
    of the failure into a name.
    """

    values = []
    for row in range(GALAXY_ROWS):
        start = row * local_vocab
        stop = min(start + local_vocab, vocab_size)
        if start >= stop:
            continue
        _, pcc = comp_pcc(reference[..., start:stop], actual[..., start:stop], 0.99)
        values.append((row, start, stop, float(pcc)))
    return values


def _permutation_report(reference, actual, vocab_size, local_vocab):
    """Describe, for each mesh row, which reference row its logits match best.

    A channel-order mismatch shows up as a permutation - every row well
    correlated with *some* reference row, just not its own - and that signature
    is immediately distinguishable from bad numerics, where the correlation is
    poor everywhere. Built only when a diagonal entry has already failed, so
    the 64 correlations cost nothing on the passing path.
    """

    lines = []
    for row in range(GALAXY_ROWS):
        start = row * local_vocab
        stop = min(start + local_vocab, vocab_size)
        if start >= stop:
            continue
        width = stop - start
        best_row, best_pcc = None, -2.0
        for candidate in range(GALAXY_ROWS):
            other = candidate * local_vocab
            if other + width > vocab_size:
                continue
            _, pcc = comp_pcc(reference[..., other : other + width], actual[..., start:stop], 0.99)
            if float(pcc) > best_pcc:
                best_row, best_pcc = candidate, float(pcc)
        lines.append(f"row {row} correlates best with reference row {best_row} at pcc {best_pcc:.4f}")
    return "; ".join(lines)


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("dim,vocab_size,padded_vocab_size", _GEOMETRIES)
@torch.no_grad()
def test_lm_head_2d_bh_galaxy_vocabulary_padding_is_ring_exact(mesh_device, dim, vocab_size, padded_vocab_size):
    """The padded vocabulary and its divisor-exactness, on the live Blackhole mesh.

    Cheapest test in the file and the one that protects the other: it runs
    against the descriptor `bh_galaxy_topology` has already validated *against
    this device* - compute grid, DRAM views, every named core group inside the
    grid, the worker envelope clear of the dispatch column - so a chassis that
    is not the one these numbers were measured on fails here rather than in a
    matmul.

    Divisor-exactness is asserted explicitly because the failure it prevents has
    no traceback. `all_reduce_async`'s reduction compute kernel opens with

        cb_in.wait_front(num_blocks * block_num_tiles);   // ring_size * shard

    on **every** output core, so a width with no divisor in the chosen core
    count leaves the last core waiting for tiles the fabric will never send. The
    program never signals completion, the host blocks in
    `FDMeshCommandQueue::wait_for_outstanding_reads`, and the mesh has to be
    reset.
    """

    topology = bh_galaxy_topology(mesh_device)

    assert RING_CORE_COUNT == 24 and RING_ALIGNMENT == TILE * RING_CORE_COUNT, (
        f"the Galaxy ring size is {RING_CORE_COUNT}, not 24. Every constraint stated in terms of "
        "`ring_size` has to be re-walked before this suite means anything: `RING_ALIGNMENT`, "
        "`pad_ring_width`, `galaxy_padded_vocab_size`, and `lm_head_reduce_core_count`'s divisor "
        "search, which the padding is what makes exact by construction"
    )
    assert galaxy_padded_vocab_size(vocab_size) == padded_vocab_size, (
        f"Blackhole pads vocab {vocab_size} to {galaxy_padded_vocab_size(vocab_size)}, not the "
        f"{padded_vocab_size} both architectures are expected to share. **The ring size moved**, and "
        "rule 8's whole dependency chain needs re-walking: the reference holds RING_SIZE = 24 on "
        "Blackhole deliberately, by widening receivers-per-reader from 2 to 3, and the `24 -> 16` "
        "figure that circulates for Blackhole belongs to the 1D LM head, a different path with "
        "different weight sharding"
    )
    assert padded_vocab_size % (GALAXY_ROWS * RING_ALIGNMENT) == 0

    local_vocab = padded_vocab_size // GALAXY_ROWS
    assert local_vocab % TILE == 0
    assert pad_ring_width(local_vocab) == local_vocab, (
        f"per-device vocabulary {local_vocab} is not a whole number of 24-core ring rows, so "
        "`lm_head_reduce_core_count`'s divisor search is no longer exact by construction"
    )

    # The two paddings are different things and must not be conflated: the
    # checkpoint's tile/device alignment is not the ring-exact width.
    checkpoint_padded = _CHECKPOINT_PADDED_VOCAB[vocab_size]
    minimum_padded = -(-vocab_size // (GALAXY_ROWS * TILE)) * (GALAXY_ROWS * TILE)
    assert checkpoint_padded == minimum_padded
    assert padded_vocab_size > checkpoint_padded, (
        "the ring-exact padding must be strictly wider than the checkpoint-vocabulary padding; "
        f"got {padded_vocab_size} against {checkpoint_padded}"
    )
    assert padded_vocab_size - vocab_size < GALAXY_ROWS * TILE * GALAXY_ROWS

    # The staging the decode all-reduce would use, resolved on this mesh's own
    # envelope and per-link reserve. Blackhole has 100 worker cores and reserves
    # 2 (one per fabric link) against Wormhole's 50 and 4, so the count differs
    # from the qualified Wormhole one and is derived rather than carried.
    workers = worker_cores(topology)
    assert workers.num_cores() > 0
    reserved = topology.ccl_reserved_worker_cores
    assert reserved == galaxy_fabric_links(mesh_device) == 2, (
        f"Blackhole Galaxy reserves {reserved} worker cores for one fabric link each; the link "
        "budget is 2, and an over-requested num_links deadlocks with no traceback"
    )
    reduce_cores = lm_head_reduce_core_count(local_vocab, workers.num_cores(), reserved_worker_cores=reserved)
    assert (local_vocab // TILE) % reduce_cores == 0, (
        f"{reduce_cores} cores do not divide {local_vocab // TILE} tiles evenly; the reduction's "
        "last core would wait for tiles the fabric never sends and the host would block in "
        "wait_for_outstanding_reads with no traceback"
    )
    assert reduce_cores <= workers.num_cores() - reserved

    # Containment: every LM-head placement must lie inside the resolved worker
    # sub-device. This is the class of failure that caught the reference's `x=6`
    # ring shard grid - a grid named independently of the partition that has to
    # contain it - and it is free to rule out here.
    placement = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(*topology.sampling_start_core), reduce_cores, workers, row_wise=True
    )
    assert placement.num_cores() == reduce_cores
    assert placement.subtract(workers).num_cores() == 0, (
        "the LM-head reduction placement leaves the worker envelope; on device that is either "
        "'Kernels cannot be placed on dispatch cores!' or 'Kernel group cores do not match sub "
        "device cores'"
    )
    for core_range in placement.ranges():
        for core in (core_range.start, core_range.end):
            assert core.x not in topology.reserved_columns, (
                f"the LM-head reduction placement reaches reserved column {core.x}; that column is "
                "excluded by measurement, not by derivation"
            )

    # The ring *path* is absent even though the ring *size* arithmetic above is
    # preserved, and asking for one must fail loudly rather than hand back
    # Wormhole's coordinates.
    assert topology.capabilities.has_ring_matmul is False
    assert topology.ring_core_coords is None
    with pytest.raises(ValueError, match="has no ring matmul path"):
        ring_cores(topology)

    logger.info(
        f"blackhole lm head vocabulary: padded={padded_vocab_size} local={local_vocab} "
        f"tiles={local_vocab // TILE} workers={workers.num_cores()} reduce_cores={reduce_cores}"
    )
    record_module_output(
        f"lm_head_2d_bh_galaxy_ring_exact_padding_{'llama8192' if dim == 8192 else 'qwen5120'}",
        torch.tensor(
            [padded_vocab_size, local_vocab, local_vocab // TILE, workers.num_cores(), reduce_cores, reserved],
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
def test_lm_head_2d_bh_galaxy_decode_reference(mesh_device, dim, vocab_size, padded_vocab_size):
    """Run each full-size geometry against a CPU matmul and reuse its allocations.

    The reference is `torch.matmul`, never another TT path: the whole point is
    that a wrong per-device channel order yields PCC ~ 0 *silently*, and two
    device paths would agree with each other while both were wrong.

    Three claims beyond the aggregate PCC, each aimed at a silent failure:

    * **the padded tail is unusably negative**, which for Llama-3.3-70B is new
      on this path. Under the old minimal padding `padded_vocab_size ==
      vocab_size` and `LMHead2D`'s invalid-logits mask was identically zero; the
      ring-exact width gives Llama 768 padded columns, so the mask is
      load-bearing for the first time and a dropped mask would let a padded
      column win an argmax. Asserted as `< -1e30` rather than `isneginf`,
      because the logits are `bfloat8_b` and a block-float shared exponent is
      not required to round-trip an infinity;
    * **the four column replicas agree with each other to 0.9999**, which is the
      only evidence that the axis-1 reduction produced the same answer on every
      column device. Tight rather than bitwise: a cross-device sum is
      arrival-order sensitive at the last bit, while a column that missed the
      reduction entirely lands near 0.5;
    * **each mesh row's vocabulary shard correlates with its own reference
      slice**. A permutation - every row matching *some* other row - is the
      channel-order signature, and the report distinguishes it from numerics.
    """

    torch.manual_seed(2)
    assert galaxy_padded_vocab_size(vocab_size) == padded_vocab_size
    topology = bh_galaxy_topology(mesh_device)
    assert topology.capabilities.has_prefetcher is False
    num_links = galaxy_fabric_links(mesh_device)
    assert num_links == 2, f"Blackhole Galaxy has two fabric links per direction, not {num_links}"

    local_vocab = padded_vocab_size // GALAXY_ROWS
    weight = torch.randn((dim, padded_vocab_size), dtype=torch.bfloat16)
    weight[:, vocab_size:] = 0
    hidden = torch.randn((1, 1, 32, dim), dtype=torch.bfloat16)
    # The full padded weight, then slice the *result*. `weight[:, vocab_size:]`
    # is zeroed, so the extra columns contribute nothing, and multiplying the
    # contiguous tensor avoids the temporary copy a non-contiguous column slice
    # of a 2 GB bfloat16 weight would make on host.
    reference = torch.matmul(hidden, weight)[..., :vocab_size].float()
    lazy_input = LazyWeight(source=hidden, device=mesh_device, dtype=ttnn.bfloat16)
    module = LMHead2D(
        [LazyWeight(source=weight, device=mesh_device, dtype=ttnn.bfloat8_b)],
        vocab_size,
        _ColumnAllReduce(mesh_device, num_links=num_links),
    )
    # `LMHead2D`'s defaults *are* the prefetcher-free recipe, so assert them
    # rather than assume them: no program config, interleaved placements, and a
    # mask that is added in place rather than staged into a shard grid.
    assert module.config.decode_program_configs == (None,)
    assert module.config.prefill_program_configs == (None,)
    assert not module.config.decode_output_memcfg.is_sharded()
    assert not module.config.prefill_output_memcfg.is_sharded()
    assert module.config.decode_stage_mask is False
    assert module.config.prefill_stage_mask is False
    assert module.config.padded_vocab_size == padded_vocab_size

    recorded = {}
    try:
        # PrefillRuntime extracts the final token rows before LM-head projection,
        # so both modes consume the same physical 32-row output batch here.
        for mode, mode_forward in (("decode", module.decode_forward), ("prefill", module.prefill_forward)):
            for iteration in range(2):
                output = mode_forward(lazy_input)
                try:
                    composed = _compose_columns(output, mesh_device, padded_vocab_size)
                finally:
                    _deallocate(output)

                # The mask, on every column replica. The tail is tile-aligned on
                # both geometries (128256 and 151936 are both multiples of 32),
                # so it is whole masked tiles rather than a mixed block.
                tail = composed[:, :, :, vocab_size:]
                assert torch.all(tail < -1e30), (
                    f"{mode} iteration {iteration}: the invalid-logits mask did not reach the padded "
                    f"tail [{vocab_size}:{padded_vocab_size}]; max value there is {float(tail.max())}"
                )

                valid = composed[0, :, :, :vocab_size]
                column_pccs = [
                    float(comp_pcc(reference, valid[column : column + 1], 0.99)[1]) for column in range(GALAXY_COLUMNS)
                ]
                logger.info(
                    f"{mode} iteration {iteration} per-mesh-column pcc: "
                    + ", ".join(f"column {column}={pcc:.5f}" for column, pcc in enumerate(column_pccs))
                )
                bad_columns = [column for column, pcc in enumerate(column_pccs) if pcc < 0.99]
                assert not bad_columns, (
                    f"{mode} iteration {iteration}: mesh column(s) {bad_columns} disagree with the CPU "
                    f"reference ({column_pccs}); a column near 0.5 never received the axis-1 reduction"
                )
                for column in range(1, GALAXY_COLUMNS):
                    _, replica_pcc = comp_pcc(valid[0:1], valid[column : column + 1], 0.9999)
                    assert float(replica_pcc) >= 0.9999, (
                        f"{mode} iteration {iteration}: mesh column {column} differs from column 0 at "
                        f"pcc {float(replica_pcc):.6f}, so the axis-1 reduction did not replicate"
                    )

                actual = valid[:1]
                rows = _row_correlations(reference, actual, vocab_size, local_vocab)
                # comp_pcc logs nothing on success, so a passing stage would
                # otherwise tell us nothing.
                logger.info(
                    f"{mode} iteration {iteration} per-mesh-row pcc: "
                    + ", ".join(f"row {row}[{start}:{stop}]={pcc:.5f}" for row, start, stop, pcc in rows)
                )
                bad = [(row, pcc) for row, _, _, pcc in rows if pcc < 0.99]
                assert not bad, (
                    f"{mode} iteration {iteration}: mesh row(s) {[row for row, _ in bad]} correlate "
                    f"poorly with their own vocabulary slice ({bad}). "
                    + _permutation_report(reference, actual, vocab_size, local_vocab)
                    + ". Every row matching some *other* row is a per-device channel-order "
                    "permutation; poor correlation everywhere is numerics"
                )

                # The aggregate number, which is `column_pccs[0]`, is logged last
                # so a passing run still prints the headline PCC.
                logger.info(f"{mode} iteration {iteration} aggregate pcc={column_pccs[0]:.5f}")
                recorded[mode] = actual.contiguous()
    finally:
        _deallocate(lazy_input._value)
        lazy_input._value = None
        module.release()

    record_module_output(
        f"lm_head_2d_bh_galaxy_decode_{'llama8192' if dim == 8192 else 'qwen5120'}",
        recorded["decode"],
        recorded["prefill"],
    )
    del hidden, lazy_input, module, reference, weight, recorded
    gc.collect()
