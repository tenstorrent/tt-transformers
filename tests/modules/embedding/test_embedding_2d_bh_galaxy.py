# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for Embedding2D on a Blackhole Galaxy.

The Blackhole counterpart of `test_embedding_2d_wh_galaxy.py`, and deliberately
the same test: same stages, same geometries, same constants, same composer. The
`(8, 4)` mesh shape is architecture-invariant, so the row-replicated /
column-sharded vocabulary arithmetic carries over unchanged and a divergence
between the two architectures' numbers is then attributable to the architecture
rather than to the test.

**Why this is the cheapest thing in the port, and why it should run first.**
`Embedding2D` names no core coordinate, no program config, no compute kernel
config and issues no collective, so there is nothing here to re-derive for
Blackhole. What it does confirm, in one node id and a few seconds of device
time, is that a 32-device Blackhole Galaxy mesh opens with
`BHGLX_DEVICE_PARAMS`, that the topology descriptor resolves against the grid
the silicon reports, and that a module produces correct numbers on it. Every
other suite in the batch depends on all three.

**The one Blackhole-specific thing it measures is not in the module.** Both of
`Embedding2D`'s resolved output placements are *interleaved* -- L1 for decode,
DRAM for prefill -- and the production Galaxy Llama config records what that
means for this op:

    `ttnn.embedding` takes its program grid from a *sharded* output's shard
    grid, and only from there: with an interleaved output - L1 or DRAM - it
    spreads over the whole compute grid

(`models/llama33_70b_galaxy/model.py`, `embedding_config`). So the placement
here is a grid chosen independently of any partition -- the same defect class as
the RoPE `batch_grid` -- and on Blackhole the whole compute grid is 12 x 10 with
the dispatch column *inside* it at column 11, which the descriptor carries as
`reserved_columns` because the reference measured a prefill-warmup regression
when it was folded into the worker envelope.

That makes this suite the batch's answer to a question every other suite
depends on: **does a full-compute-grid program run on a Blackhole Galaxy?** It
is asked here rather than anywhere else because it is asked cheapest here, and
because nothing else in this file can fail for an unrelated reason. Two
consequences, both deliberate:

* **No sub-device manager is loaded.** Under a loaded worker sub-device an
  interleaved-output embedding would compile onto cores the manager does not
  own and abort with `Kernel group cores do not match sub device cores`; that is
  a property of the placement, not of Blackhole, and it is not what this suite
  is for. Production avoids it by naming the residual placement for decode and
  by running prefill under a full-grid sub-device. Neither is available to a
  module test that owns no residual stream, so this runs the module's own
  defaults on the default manager, exactly as the Wormhole suite does.
* **The interleaved placements are asserted**, so that if the module's defaults
  ever become sharded, this file's premise is re-examined rather than silently
  invalidated.

No checkpoint is read: `torch.randn` weights and synthetic token ids, so there
is no `HF_HOME` path along which this can skip instead of running.
"""

from __future__ import annotations

import gc

import pytest
import torch
import ttnn
from loguru import logger
from tests.models.galaxy.galaxy_hardware import BHGLX_DEVICE_PARAMS
from tests.modules._bh_galaxy_hardware import (
    bh_galaxy_topology,
    compose_2d_sharded_tensor,
    deallocate_tensor,
    record_module_output,
)
from tests.support.comparison import comp_pcc

from tt_transformers.modules.embedding.embedding_2d import Embedding2D
from tt_transformers.modules.lazy_weight import LazyWeight

#: The stages the Wormhole suite qualified, in its order: decode at the Galaxy
#: physical batch, then the two sequential prefill lengths, each run twice so a
#: second invocation against a warm program cache is covered.
_STAGES = (("decode", 32), ("prefill", 128), ("prefill", 2048))

#: The correctness bar. Same value as the Wormhole suite so the two are
#: comparable; the byte-identity across processes is the stronger claim and is
#: made by the job script from `record_module_output`.
_PCC = 0.99


def _require_interleaved_placements(module: Embedding2D) -> None:
    """Assert the module resolved the interleaved placements this file assumes.

    Not a tautology: it is the premise of the whole file. An interleaved output
    makes `ttnn.embedding` spread over the entire compute grid, which is the
    Blackhole question this suite exists to answer, and it is also why no
    sub-device manager is loaded here. If a future default is sharded instead,
    both of those facts change and this fails rather than quietly measuring
    something else.
    """

    for name in ("weights_memcfg", "decode_output_memcfg", "prefill_output_memcfg"):
        memory_config = getattr(module.config, name)
        assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, (
            f"Embedding2D resolved a sharded {name} ({memory_config}); this suite assumes the interleaved "
            "defaults, which is what makes `ttnn.embedding` take the whole compute grid and what makes it "
            "safe to run with no sub-device manager loaded"
        )


def _record_descriptor_facts(topology) -> None:
    """Log and check the descriptor facts the run's interpretation rests on.

    The dispatch column is the only one of these that is empirical rather than
    derived, and it is the one that matters here: it lies *inside*
    `compute_with_storage_grid_size()`, so the grid this module's interleaved
    output spreads over contains it.
    """

    dispatch_column = topology.compute_grid[0] - 1
    assert topology.reserved_columns == (dispatch_column,), (
        f"expected the dispatch column {dispatch_column} to be the reserved column, got {topology.reserved_columns}"
    )
    assert not topology.capabilities.has_prefetcher
    assert all(x != dispatch_column for x, _ in topology.worker_coords)
    logger.info(
        "BH Galaxy descriptor: compute grid {}, {} DRAM views, {} worker cores, dispatch column {} "
        "(inside the compute grid, and inside the grid an interleaved `ttnn.embedding` spreads over)",
        topology.compute_grid,
        topology.dram_views,
        len(topology.worker_coords),
        dispatch_column,
    )


def _assert_column_shards_are_not_permuted(
    reference: torch.Tensor,
    actual: torch.Tensor,
    mesh_columns: int,
    *,
    case: str,
) -> None:
    """Correlate per mesh column, because aggregate PCC hides a permutation.

    The vocabulary is sharded over the mesh's 4 columns on the hidden
    dimension, so a column-order mismatch in the mapper or the composer is a
    *permutation* of hidden slices: every column correlates well with some
    reference column, just not with its own. Aggregate PCC on a mis-ordered
    composition lands in 0-0.01 and says only "wrong"; the diagonal-best test
    below distinguishes a permutation from bad numerics, where correlation is
    poor everywhere.
    """

    local = actual.shape[-1] // mesh_columns
    for column in range(mesh_columns):
        actual_slice = actual[..., column * local : (column + 1) * local]
        scores = [
            comp_pcc(reference[..., other * local : (other + 1) * local], actual_slice, _PCC)[1]
            for other in range(mesh_columns)
        ]
        logger.info("{}: mesh column {} correlates {} against reference columns", case, column, scores)
        best = max(range(mesh_columns), key=lambda other: scores[other])
        assert best == column, (
            f"{case}: mesh column {column} correlates best with reference column {best} "
            f"(scores {scores}); that is a column-order permutation, not a numeric error"
        )


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.galaxy_bh
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [BHGLX_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "vocab_size,dim,embed_scale",
    [
        pytest.param(128256, 8192, 1.0, id="llama-8192"),
        pytest.param(151936, 5120, 5120**0.5, id="qwen-5120"),
    ],
)
@torch.no_grad()
def test_embedding_2d_bh_galaxy_reference(mesh_device, vocab_size, dim, embed_scale):
    """Qualify decode batch 32 and sequential prefill lengths against torch.

    The constants are the Wormhole suite's, unchanged: Llama-3.3-70B is
    `vocab 128256, dim 8192, scale 1.0`; Qwen3-32B is `vocab 151936, dim 5120`
    with the `sqrt(dim)` scale that exercises the multiply path. Qwen's
    checkpoint-padded vocabulary (151936 -> 152064) is deliberately *not* used:
    the padding belongs to the checkpoint loader, and changing the constant
    would make these numbers incomparable to the qualified Wormhole ones for no
    coverage the sharding math does not already have.

    The reference is `torch.nn.functional.embedding` on the CPU -- never a
    second TT path -- and each stage is compared aggregate *and* per mesh
    column.
    """

    topology = bh_galaxy_topology(mesh_device)
    _record_descriptor_facts(topology)
    mesh_columns = int(tuple(mesh_device.shape)[1])
    assert dim % mesh_columns == 0

    torch.manual_seed(0)
    weight = torch.randn((vocab_size, dim), dtype=torch.bfloat16)
    module = Embedding2D(LazyWeight(source=weight, device=mesh_device), embed_scale=embed_scale)
    _require_interleaved_placements(module)

    recorded: list[torch.Tensor] = []
    try:
        for mode, token_count in _STAGES:
            token_ids = torch.randint(0, vocab_size, (1, token_count), dtype=torch.int32)
            reference = torch.nn.functional.embedding(token_ids.long(), weight).float() * embed_scale
            lazy_ids = LazyWeight(source=token_ids.reshape(1, 1, 1, token_count), device=mesh_device)
            actual = None

            for invocation in range(2):
                case = f"{mode} {token_count} tokens, invocation {invocation}"
                output = module.forward(lazy_ids, mode=mode)
                try:
                    actual = compose_2d_sharded_tensor(output, mesh_device).reshape(1, token_count, dim).float()
                finally:
                    deallocate_tensor(output)
                passing, pcc = comp_pcc(reference, actual, _PCC)
                # `comp_pcc` logs nothing on success, so a passing stage would
                # otherwise tell the run's reader nothing at all.
                logger.info("embedding_2d BH Galaxy dim {} {}: PCC {}", dim, case, pcc)
                assert passing, f"{case}: PCC {pcc} below {_PCC}"
                _assert_column_shards_are_not_permuted(reference, actual, mesh_columns, case=case)

            recorded.append(actual)
            deallocate_tensor(lazy_ids._value)
            lazy_ids._value = None
            del lazy_ids, reference, token_ids
            gc.collect()

        # One file per node id per process; the job script compares the three
        # processes' files with `torch.equal`. Three PCC passes are not a
        # result, a byte-identical triple is.
        record_module_output(f"embedding_2d_bh_galaxy_dim{dim}", *recorded)
    finally:
        deallocate_tensor(getattr(module, "weights", None))
        module.config.weights._value = None
        del module, weight, recorded
        gc.collect()
