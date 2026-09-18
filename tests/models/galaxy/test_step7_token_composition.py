# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host qualification of the sampled-token composition.

Measured twice and byte-identically on an `(8, 4)` Galaxy:
`GalaxyDirectRunner.decode_sampled` returns **one mesh column's eight users
repeated four times** instead of 32 distinct users. The focused selector
qualification (`test_column_user_selector_wh_galaxy.py`) passes 3/3 on the same
hardware, so the arithmetic is right and the **readback** is wrong.

This file pins the mechanism down **on host, with no mesh**, which is the only
reason it can exist: the Galaxy was unrecoverable when this was found, and a
diagnosis that can only be checked on hardware that is down is a diagnosis
nobody can review.

Three claims, in order:

1. `auto_compose._compose_nd_sharded` maps a `PlacementShard` to a concatenation
   on its `dim` and a `PlacementReplicate` to `shape_override=1`, i.e. skipped.
   So `to_torch_auto_compose` is **correct for correctly labelled tensors** —
   the defect is not a bug in the auto-composer.
2. Given topology labels that name the replicated mesh axis as sharded and the
   sharded one as replicated — which is what an op inheriting its *activation's*
   labels produces, per `collectives.compose_galaxy_logits` — the resulting
   composition of a 32-device `(8, 4)` token tensor is exactly
   `users[0:8]` repeated four times in its leading 32 values. That is the
   measured signature, reproduced arithmetically.
3. The composition that follows the distribution instead —
   `ConcatMesh2dToTensor(dims=(0, user axis))` then mesh row 0 — recovers all 32
   distinct users.

Claims 2 and 3 are host **models** of the two compositions over an `(8, 4)` mesh,
not device measurements; they are exact about the index arithmetic, which is
where the defect lives, and say nothing about what silicon does.
`test_qwen_device_sampling_claims_with_an_explicit_token_composition` is the
device case that closes the loop.

Run::

    pytest tests/models/galaxy/test_step7_token_composition.py -v
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from examples.common.auto_compose import _compose_nd_sharded
from tests.models.galaxy.galaxy_hardware import (
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_USERS_PER_COLUMN,
)

#: `(8, 4)`: eight devices per mesh column carrying the vocabulary shards, four
#: mesh columns carrying `GALAXY_USERS_PER_COLUMN` users each.
_MESH_ROWS, _MESH_COLUMNS = GALAXY_MESH_SHAPE


def _captured_composer_config(monkeypatch, placements, dist_shape):
    """Return the `MeshComposerConfig` `_compose_nd_sharded` would build."""

    captured = {}

    def fake_create_mesh_composer(device, config):
        captured["device"] = device
        captured["config"] = config
        return "composer-sentinel"

    monkeypatch.setattr(ttnn, "create_mesh_composer", fake_create_mesh_composer)
    result = _compose_nd_sharded(None, placements, dist_shape)
    assert result == "composer-sentinel", "the composer the module returns is the one it created"
    return captured["config"]


@pytest.mark.host
@pytest.mark.model
def test_a_replicated_mesh_axis_is_skipped_and_a_sharded_one_is_concatenated(monkeypatch):
    """Claim 1: the auto-composer is correct for correctly labelled tensors.

    A tensor whose eight-device axis really is replicated and whose four-device
    axis really is sharded on the user dimension composes to 32 users, because
    `shape_override=1` on the replicated axis skips its concatenation.
    """

    config = _captured_composer_config(
        monkeypatch,
        [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)],
        [_MESH_ROWS, _MESH_COLUMNS],
    )
    text = str(config)
    assert "dims: [0, 3]" in text, f"expected the replicated axis to fall back to dim 0, got {text}"
    assert "MeshShape([1, 4])" in text, f"expected the replicated axis to be skipped with override 1, got {text}"


@pytest.mark.host
@pytest.mark.model
def test_labels_that_name_the_wrong_axis_concatenate_the_replicas(monkeypatch):
    """Claim 1, the other half: the same code with the labels swapped.

    This is the label set an op inheriting its *activation's* topology produces:
    `Shard(3)` on the axis that in fact holds identical copies, `Replicate` on the
    axis that in fact holds the distinct users.
    """

    config = _captured_composer_config(
        monkeypatch,
        [ttnn.PlacementShard(3), ttnn.PlacementReplicate()],
        [_MESH_ROWS, _MESH_COLUMNS],
    )
    text = str(config)
    assert "dims: [3, 0]" in text, f"expected a concatenation on dim 3 for the mislabelled axis, got {text}"
    assert "MeshShape([8, 1])" in text, (
        f"expected the eight identical copies to be concatenated and the four distinct columns skipped, got {text}"
    )


def _per_device_tokens() -> list[list[torch.Tensor]]:
    """The token tensor as 32 devices hold it after `Sampling2D.decode_forward`.

    Mesh column `c` sampled users `8c .. 8c + 7`; the eight devices of that column
    hold identical copies, because they all-gathered the whole vocabulary between
    them before sampling.
    """

    users = torch.arange(GALAXY_PHYSICAL_BATCH, dtype=torch.int64)
    return [
        [
            users[column * GALAXY_USERS_PER_COLUMN : (column + 1) * GALAXY_USERS_PER_COLUMN].clone()
            for _ in range(_MESH_ROWS)
        ]
        for column in range(_MESH_COLUMNS)
    ]


def _compose_as_the_labels_say(devices: list[list[torch.Tensor]]) -> torch.Tensor:
    """`dims=[3, 0]`, `override=[8, 1]`: concatenate the eight copies, skip the columns."""

    return torch.cat(devices[0], dim=-1)


def _compose_as_the_distribution_says(devices: list[list[torch.Tensor]]) -> torch.Tensor:
    """`dims=(0, user axis)` then mesh row 0: concatenate the four columns."""

    stacked = torch.stack([torch.cat([devices[c][r] for c in range(_MESH_COLUMNS)], dim=-1) for r in range(_MESH_ROWS)])
    return stacked[0]


@pytest.mark.host
@pytest.mark.model
def test_the_mislabelled_composition_reproduces_the_measured_signature():
    """Claim 2: eight users repeated four times, in the leading 32 values."""

    devices = _per_device_tokens()
    composed = _compose_as_the_labels_say(devices)
    assert composed.numel() == _MESH_ROWS * GALAXY_USERS_PER_COLUMN == 64, (
        f"the mislabelled composition should yield {_MESH_ROWS} copies of "
        f"{GALAXY_USERS_PER_COLUMN} users, got {composed.numel()}"
    )
    leading = composed[:GALAXY_PHYSICAL_BATCH]
    expected = torch.arange(GALAXY_USERS_PER_COLUMN, dtype=torch.int64).repeat(
        GALAXY_PHYSICAL_BATCH // GALAXY_USERS_PER_COLUMN
    )
    assert torch.equal(leading, expected), (
        f"the measured signature is column 0's eight users repeated four times; this composition gave "
        f"{leading.tolist()}"
    )
    # And the harm: 25 of 32 slots carry another user's token, which is exactly
    # the shape of the 7/32 agreement that was measured (slot 4 aside, whose
    # host argmax differs for its own reason).
    wrong = [slot for slot in range(GALAXY_PHYSICAL_BATCH) if int(leading[slot]) != slot]
    assert len(wrong) == GALAXY_PHYSICAL_BATCH - GALAXY_USERS_PER_COLUMN == 24, (
        f"expected the first column's 8 slots to be right and the other 24 wrong, got {len(wrong)} wrong"
    )


@pytest.mark.host
@pytest.mark.model
def test_the_composition_that_follows_the_distribution_recovers_all_32_users():
    """Claim 3: the fix pattern gives 32 distinct users, in order."""

    devices = _per_device_tokens()
    composed = _compose_as_the_distribution_says(devices)
    assert composed.numel() == GALAXY_PHYSICAL_BATCH, f"expected 32 users, got {composed.numel()}"
    assert torch.equal(composed, torch.arange(GALAXY_PHYSICAL_BATCH, dtype=torch.int64)), (
        f"the distribution-following composition must return user i in slot i, got {composed.tolist()}"
    )


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("column", range(_MESH_COLUMNS))
def test_every_mesh_column_holds_its_own_users_and_nothing_else(column: int):
    """The premise the two compositions differ about, stated on its own."""

    devices = _per_device_tokens()
    for row in range(_MESH_ROWS):
        held = devices[column][row]
        first = column * GALAXY_USERS_PER_COLUMN
        assert torch.equal(held, torch.arange(first, first + GALAXY_USERS_PER_COLUMN, dtype=torch.int64))
        assert torch.equal(held, devices[column][0]), "the eight devices of a mesh column hold identical tokens"
