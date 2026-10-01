# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only contract tests for Qwen GDN decode-slot remapping."""

from types import MethodType

import pytest
import torch

from tt_transformers.models.qwen38.gdn import tp as gdn_tp
from tt_transformers.models.qwen38.gdn.tp import TPGatedDeltaNet


def _fake_gdn(*, packed_valid: bool, batch_size: int = 8):
    """Build the smallest state owner that exercises remap_slots without a device."""
    gdn = object.__new__(TPGatedDeltaNet)
    gdn.B = batch_size
    gdn.K = 4
    gdn.rec_state = {"rows": list(range(batch_size))}
    gdn.conv_states = [{"rows": [(tap, slot) for slot in range(batch_size)]} for tap in range(gdn.K)]
    gdn.conv_hist_packed = {
        "address": object(),
        "rows": [(slot, slot & 1) for slot in range(batch_size)],
    }
    gdn._hist_packed_valid = packed_valid
    gdn._conv_win_stale = False
    events = []

    def sync_conv_taps(self):
        events.append("sync_taps")

    def gather_indices(self, buf, idx, dim):
        events.append(("gather", dim, id(buf)))
        old = list(buf["rows"])
        buf["rows"] = [old[src] for src in idx]

    def sync_conv_hist_packed(self, slot=None):
        assert slot is None
        events.append("sync_hist")
        # Model the production rebuild: the logical history follows the remapped taps,
        # while its physical row selector is encoded for the destination slot parity.
        self.conv_hist_packed["rows"] = [
            (self.conv_states[0]["rows"][dst][1], dst & 1) for dst in range(self.B)
        ]
        self._hist_packed_valid = True

    def remap_conv_hist_packed(self, idx):
        events.append("remap_packed")
        old = list(self.conv_hist_packed["rows"])
        self.conv_hist_packed["rows"] = [(old[src][0], dst & 1) for dst, src in enumerate(idx)]
        self._hist_packed_valid = True

    gdn.sync_conv_taps = MethodType(sync_conv_taps, gdn)
    gdn._gather_indices = MethodType(gather_indices, gdn)
    gdn._sync_conv_hist_packed = MethodType(sync_conv_hist_packed, gdn)
    gdn._remap_conv_hist_packed = MethodType(remap_conv_hist_packed, gdn)
    return gdn, events


@pytest.mark.parametrize("packed_valid", [True, False], ids=["packed-valid", "packed-invalid"])
def test_gdn_slot_remap_identity_is_a_true_noop(packed_valid):
    gdn, events = _fake_gdn(packed_valid=packed_valid)
    packed_address = gdn.conv_hist_packed["address"]
    packed_rows = list(gdn.conv_hist_packed["rows"])

    gdn.remap_slots(list(range(gdn.B)))

    assert events == []
    assert gdn.rec_state["rows"] == list(range(gdn.B))
    assert gdn.conv_hist_packed["address"] is packed_address
    assert gdn.conv_hist_packed["rows"] == packed_rows
    assert gdn._hist_packed_valid is packed_valid
    assert gdn._conv_win_stale is False


@pytest.mark.parametrize("packed_valid", [True, False], ids=["packed-valid", "packed-invalid"])
@pytest.mark.parametrize(
    "remap",
    [
        pytest.param([1, 2, 3, 4, 5, 6, 7, 0], id="parity-changing-rotation"),
        pytest.param([2, 3, 4, 5, 6, 7, 0, 1], id="parity-preserving-rotation"),
        pytest.param([1, 2, 0, 3, 4, 5, 6, 7], id="mixed-parity-with-hold-rows"),
    ],
)
def test_gdn_slot_remap_rebuilds_packed_history_for_destination_parity(remap, packed_valid):
    gdn, events = _fake_gdn(packed_valid=packed_valid)
    packed_address = gdn.conv_hist_packed["address"]

    gdn.remap_slots(remap)

    assert gdn.rec_state["rows"] == remap
    for tap in range(gdn.K):
        assert gdn.conv_states[tap]["rows"] == [(tap, src) for src in remap]
    assert gdn.conv_hist_packed["rows"] == [(src, dst & 1) for dst, src in enumerate(remap)]
    assert gdn.conv_hist_packed["address"] is packed_address
    assert gdn._hist_packed_valid is True
    assert gdn._conv_win_stale is True

    assert events[0] == "sync_taps"
    assert events[1][0:2] == ("gather", 0)
    if packed_valid:
        # The packed view may be newer than the unpacked taps after fused plain decode, so it is
        # translated directly before the taps are independently gathered.
        assert events[2] == "remap_packed"
        assert [event[1] for event in events[3:]] == [1] * gdn.K
    else:
        # The canonical taps move first; only then may the invalid packed mirror be rebuilt.
        assert [event[1] for event in events[2:-1]] == [1] * gdn.K
        assert events[-1] == "sync_hist"


def test_valid_packed_remap_updates_persistent_buffer_without_device_allocation(monkeypatch):
    """The host-assisted parity translation must not allocate a request-time device tensor."""
    batch_size = 4
    old = torch.zeros(batch_size, 1, 1, 8, 2, dtype=torch.bfloat16)
    for src in range(batch_size):
        old[src, ..., src & 1 :: 2, :] = src + 1

    persistent = object()
    device_shard = object()
    mesh = object()
    mapper = object()
    captured = {}

    monkeypatch.setattr(gdn_tp.ttnn, "get_device_tensors", lambda tensor: [device_shard])
    monkeypatch.setattr(gdn_tp.ttnn, "to_torch", lambda tensor: old.clone())
    monkeypatch.setattr(gdn_tp.ttnn, "ShardTensorToMesh", lambda mesh_arg, dim: mapper)

    def from_torch(tensor, **kwargs):
        captured["host"] = tensor.clone()
        captured["from_torch"] = kwargs
        return object()

    def copy_host_to_device_tensor(host, destination):
        captured["destination"] = destination

    monkeypatch.setattr(gdn_tp.ttnn, "from_torch", from_torch)
    monkeypatch.setattr(gdn_tp.ttnn, "copy_host_to_device_tensor", copy_host_to_device_tensor)

    gdn = object.__new__(TPGatedDeltaNet)
    gdn.conv_hist_packed = persistent
    gdn.mesh = mesh
    gdn._hist_packed_valid = False
    remap = [1, 2, 3, 0]

    gdn._remap_conv_hist_packed(remap)

    expected = torch.zeros_like(old)
    for dst, src in enumerate(remap):
        expected[dst, ..., dst & 1 :: 2, :] = old[src, ..., src & 1 :: 2, :]
    assert torch.equal(captured["host"], expected)
    assert captured["from_torch"]["device"] is None
    assert captured["from_torch"]["mesh_mapper"] is mapper
    assert captured["destination"] is persistent
    assert gdn._hist_packed_valid is True
