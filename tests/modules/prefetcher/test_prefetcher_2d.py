# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ttnn

from tt_transformers.modules.prefetcher.prefetcher_2d import (
    GlobalCBPlacement,
    Prefetcher2D,
    Prefetcher2DConfig,
    Prefetcher2DModeConfig,
)


class FakeMesh:
    def __init__(self, shape=(8, 4), count=32, arch=ttnn.device.Arch.WORMHOLE_B0):
        self.shape = shape
        self._count = count
        self._arch = arch
        self.events = []
        self.fail_next_stall = False
        self.fail_remove = None

    def get_num_devices(self):
        return self._count

    def arch(self):
        return self._arch

    def create_sub_device_manager(self, subdevices, local_l1_size):
        manager = f"manager-{len([event for event in self.events if event[0] == 'create'])}"
        self.events.append(("create", tuple(subdevices), local_l1_size, manager))
        return manager

    def load_sub_device_manager(self, manager):
        self.events.append(("load", manager))

    def set_sub_device_stall_group(self, stall_group):
        self.events.append(("stall", tuple(stall_group)))
        if self.fail_next_stall:
            self.fail_next_stall = False
            raise RuntimeError("stall failure")

    def reset_sub_device_stall_group(self):
        self.events.append(("reset_stall",))

    def clear_loaded_sub_device_manager(self):
        self.events.append(("clear_manager",))

    def remove_sub_device_manager(self, manager):
        self.events.append(("remove", manager))
        if manager == self.fail_remove:
            raise RuntimeError(f"remove failure: {manager}")


class FakeTensor:
    def __init__(self, mesh, address, size=128):
        self._mesh = mesh
        self._address = address
        self._size = size

    def device(self):
        return self._mesh

    def buffer_address(self):
        return self._address

    def buffer_size(self):
        return self._size


class ResourceHarness:
    def __init__(self):
        self.created_cbs = []
        self.metadata = []
        self.deallocated = []
        self.prefetch_events = []
        self.fail_metadata_once = False
        self.fail_start_number = None
        self.fail_stop_once = False
        self.start_count = 0

    def create_global_cb(self, mesh, mapping, size):
        value = ("global-cb", len(self.created_cbs), tuple(mapping), size)
        self.created_cbs.append(value)
        return value

    def create_metadata(self, addresses, **kwargs):
        if self.fail_metadata_once:
            self.fail_metadata_once = False
            raise RuntimeError("metadata failure")
        value = {"addresses": addresses.clone(), **kwargs}
        self.metadata.append(value)
        return value

    def deallocate(self, resource):
        self.deallocated.append(resource)

    def start(self, context):
        self.start_count += 1
        self.prefetch_events.append(("start", context.mode, context.weights, context.global_cb))
        if self.start_count == self.fail_start_number:
            raise RuntimeError("prefetch start failure")
        return ("prefetch-result", self.start_count)

    def stop(self, mesh, result):
        self.prefetch_events.append(("stop", mesh, result))
        if self.fail_stop_once:
            self.fail_stop_once = False
            raise RuntimeError("prefetch stop failure")
        self.deallocate(result)
        return ("sync-result", result[-1])

    def kwargs(self):
        return {
            "create_global_cb": self.create_global_cb,
            "create_address_metadata": self.create_metadata,
            "deallocate": self.deallocate,
            "dram_prefetch_start": self.start,
            "dram_prefetch_stop": self.stop,
        }


@pytest.fixture
def resources():
    return ResourceHarness()


def make_config(
    mesh=None,
    expected_weight_count=2,
    global_cb_size=4096,
    defer_global_cb=False,
    release_global_cb_on_prefill=False,
    global_cb_headroom=0,
    on_global_cb_released=None,
):
    mesh = FakeMesh() if mesh is None else mesh
    return Prefetcher2DConfig(
        mesh_device=mesh,
        architecture=ttnn.device.Arch.WORMHOLE_B0,
        prefill=Prefetcher2DModeConfig(
            mode="prefill",
            sub_devices=("all-workers",),
            worker_sub_device_id="prefill-worker",
            stall_group=("prefill-worker",),
        ),
        decode=Prefetcher2DModeConfig(
            mode="decode",
            sub_devices=("senders", "workers"),
            worker_sub_device_id="decode-worker",
            stall_group=("prefetch-worker", "decode-worker"),
        ),
        sender_receiver_mapping=(("sender-0", "receiver-0"), ("sender-1", "receiver-1")),
        global_cb_size=global_cb_size,
        expected_weight_count=expected_weight_count,
        address_repeat_count=2,
        address_memory_config="address-memcfg",
        address_mesh_mapper="address-mapper",
        defer_global_cb=defer_global_cb,
        on_global_cb_released=on_global_cb_released,
        release_global_cb_on_prefill=release_global_cb_on_prefill,
        global_cb_headroom=global_cb_headroom,
    )


def initialized_owner(resources, expected_weight_count=2, **kwargs):
    owner = Prefetcher2D(make_config(expected_weight_count=expected_weight_count, **kwargs), **resources.kwargs())
    owner.initialize()
    return owner


def seal_one(owner, *, size=128):
    weight = FakeTensor(owner.config.mesh_device, 101, size)
    owner.register_weight("weight", weight)
    owner.seal()
    return weight


@pytest.mark.host
def test_config_is_frozen_and_fails_closed_for_wh_8x4():
    cfg = make_config()
    with pytest.raises(FrozenInstanceError):
        cfg.mesh_shape = (4, 8)

    with pytest.raises(ValueError, match="mesh device shape"):
        make_config(FakeMesh(shape=(4, 8)))
    with pytest.raises(ValueError, match="exactly 32 devices"):
        make_config(FakeMesh(count=16))
    with pytest.raises(ValueError, match="architecture does not match"):
        make_config(FakeMesh(arch=ttnn.device.Arch.BLACKHOLE))
    assert isinstance(cfg.prefill.sub_devices, tuple)
    assert isinstance(cfg.decode.stall_group, tuple)
    assert isinstance(cfg.sender_receiver_mapping, tuple)


@pytest.mark.host
def test_address_repeat_count_tracks_active_readers_not_dummy_global_cb_mappings():
    cfg = replace(
        make_config(),
        sender_receiver_mapping=(
            ("active-0", "receiver-0"),
            ("active-1", "receiver-1"),
            ("dummy", "remaining-workers"),
        ),
    )
    assert cfg.address_repeat_count == 2

    with pytest.raises(ValueError, match="cannot exceed"):
        replace(cfg, address_repeat_count=4)


@pytest.mark.host
def test_initialize_creates_only_both_managers_and_is_idempotent(resources):
    owner = initialized_owner(resources)
    owner.initialize()

    creates = [event for event in owner.config.mesh_device.events if event[0] == "create"]
    assert len(creates) == 2
    assert creates[0][1] == ("all-workers",)
    assert creates[1][1] == ("senders", "workers")
    assert resources.created_cbs == []


@pytest.mark.host
def test_registration_is_ordered_borrowed_and_compatibility_validated(resources):
    calls = []

    def validate(name, tensor, existing):
        calls.append((name, tensor, existing))
        if name == "bad":
            raise ValueError("incompatible prefetch weight")

    owner = Prefetcher2D(make_config(), validate_weight_compatibility=validate, **resources.kwargs())
    owner.initialize()
    mesh = owner.config.mesh_device
    first = FakeTensor(mesh, 101)
    second = FakeTensor(mesh, 202)
    owner.register_weight("layer.0.w1", first)
    with pytest.raises(ValueError, match="incompatible"):
        owner.register_weight("bad", second)
    owner.register_weight("layer.0.w2", second)
    prefill, decode = owner.seal()

    assert calls[-1] == ("layer.0.w2", second, (first,))
    assert owner.borrowed_weights == (first, second)
    assert prefill.weights == ()
    assert decode.weights == (first, second)
    assert dict(decode.weight_addresses) == {"layer.0.w1": 101, "layer.0.w2": 202}
    assert resources.metadata[0]["addresses"].tolist() == [[101, 202], [101, 202]]
    with pytest.raises(RuntimeError, match="sealed"):
        owner.register_weight("layer.1.w1", FakeTensor(mesh, 303))
    assert owner.seal() == (prefill, decode)


@pytest.mark.host
def test_seal_loads_decode_manager_before_global_cb_allocation(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101))

    owner.seal()

    assert owner.config.mesh_device.events[-2:] == [
        ("load", "manager-1"),
        ("stall", (ttnn.SubDeviceId(0), ttnn.SubDeviceId(1))),
    ]


@pytest.mark.host
def test_default_registration_rejects_duplicate_buffer_and_other_mesh(resources):
    owner = initialized_owner(resources)
    owner.register_weight("first", FakeTensor(owner.config.mesh_device, 101))
    with pytest.raises(ValueError, match="distinct device buffers"):
        owner.register_weight("alias", FakeTensor(owner.config.mesh_device, 101))
    with pytest.raises(TypeError, match="materialized"):
        owner.register_weight("lazy", object())
    with pytest.raises(ValueError, match="different mesh"):
        owner.register_weight("other", FakeTensor(FakeMesh(), 202))


@pytest.mark.host
def test_deferred_global_cb_is_allocated_on_first_decode_activation(resources):
    """`defer_global_cb` moves the allocation from `seal()` to `activate("decode")`.

    The global CB is ~774 kB of L1 per sender/receiver core and nothing frees it,
    so a prefill program needing static circular buffers on those cores cannot be
    placed while it is resident. Prefill never reads it, so it must not exist yet.

    Three properties matter, and all three are checked here:

    1. sealing allocates nothing, and prefill still gets `global_cb=None`;
    2. the first `activate("decode")` allocates exactly one buffer, *before* the
       prefetch program is started - the prefetch program is what reads it;
    3. the decode context object that modules captured at build time sees the
       buffer, not the `None` it was sealed with. Module configs hold the context
       *object* (`MLP2DConfig.decode_prefetch_context`), so binding a replacement
       context would silently leave every built module with no global CB.
    """

    owner = initialized_owner(resources, expected_weight_count=1, defer_global_cb=True)
    weight = FakeTensor(owner.config.mesh_device, 101, 128)
    owner.register_weight("weight", weight)
    prefill, decode = owner.seal()

    assert resources.created_cbs == []
    assert prefill.global_cb is None
    assert decode.global_cb is None

    # Prefill must be activatable with no buffer in existence at all.
    owner.activate("prefill")
    assert resources.created_cbs == []

    owner.activate("decode")
    assert len(resources.created_cbs) == 1
    assert decode.global_cb == resources.created_cbs[0]
    # The prefetch program was handed the buffer, not None.
    starts = [event for event in resources.prefetch_events if event[0] == "start"]
    assert starts[-1][3] == resources.created_cbs[0]

    # Re-activating decode must not allocate a second buffer.
    owner.activate("decode")
    assert len(resources.created_cbs) == 1
    owner.cleanup()


@pytest.mark.host
def test_seal_derives_cb_size_and_rejects_undersized_configuration(resources):
    derived = Prefetcher2D(make_config(expected_weight_count=2, global_cb_size=None), **resources.kwargs())
    derived.initialize()
    derived.register_weight("small", FakeTensor(derived.config.mesh_device, 101, size=128))
    derived.register_weight("large", FakeTensor(derived.config.mesh_device, 202, size=300))
    _, decode = derived.seal()
    assert derived.resolved_global_cb_size == 600
    assert decode.global_cb[-1] == 600

    configured = initialized_owner(resources, expected_weight_count=1, global_cb_size=128)
    configured.register_weight("large", FakeTensor(configured.config.mesh_device, 303, size=100))
    _, decode = configured.seal()
    assert configured.resolved_global_cb_size == 128
    assert decode.global_cb[-1] == 128


@pytest.mark.host
def test_seal_is_transactional_and_retryable_after_metadata_failure(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101))
    resources.fail_metadata_once = True

    with pytest.raises(RuntimeError, match="metadata failure"):
        owner.seal()
    assert not owner.sealed
    assert resources.deallocated == []

    _, decode = owner.seal()
    assert owner.sealed
    assert decode.global_cb == resources.created_cbs[1]


@pytest.mark.host
def test_activation_starts_stops_and_releases_repeat_results(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    weight = seal_one(owner)

    decode = owner.activate("decode")
    first_result = owner.prefetch_result
    owner.activate("decode")
    second_result = owner.prefetch_result
    owner.activate("prefill")

    assert decode.worker_sub_device_id == "decode-worker"
    assert first_result != second_result
    assert owner.prefetch_result is None
    assert first_result in resources.deallocated
    assert second_result in resources.deallocated
    assert first_result not in owner.owned_resources
    assert second_result not in owner.owned_resources
    assert weight not in owner.owned_resources
    assert [event[0] for event in resources.prefetch_events] == ["start", "stop", "start", "stop"]
    stall_events = [event[1] for event in owner.config.mesh_device.events if event[0] == "stall"]
    assert stall_events[0] == (ttnn.SubDeviceId(0), ttnn.SubDeviceId(1))
    assert stall_events[1] == (ttnn.SubDeviceId(0), ttnn.SubDeviceId(1))
    assert stall_events[2] == owner.config.decode.stall_group


@pytest.mark.host
def test_default_ttnn_session_includes_addresses_and_deallocates_stop_sentinel(monkeypatch):
    start = MagicMock(return_value="sentinel")
    deallocate = MagicMock()
    monkeypatch.setattr(ttnn, "dram_prefetcher", start)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    owner = object.__new__(Prefetcher2D)
    owner.config = SimpleNamespace(prefetch_num_layers=1)
    context = SimpleNamespace(weights=("w1", "w2"), weight_address_metadata="addresses", global_cb="cb")

    assert owner._default_dram_prefetch_start(context) == "sentinel"
    start.assert_called_once_with(["w1", "w2", "addresses"], num_layers=1, global_cb="cb")
    assert Prefetcher2D._default_dram_prefetch_stop("mesh", "sentinel") is None
    deallocate.assert_called_once_with("sentinel")


@pytest.mark.host
def test_borrow_context_requires_exact_sealed_subdevice_policy(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)

    context = owner.borrow_context(
        "decode",
        sub_devices=("senders", "workers"),
        worker_sub_device_id="decode-worker",
        stall_group=("prefetch-worker", "decode-worker"),
        local_l1_size=0,
    )

    assert owner.mesh_device is owner.config.mesh_device
    assert context is owner.context("decode")
    with pytest.raises(ValueError, match="do not match"):
        owner.borrow_context(
            "decode",
            sub_devices=("workers",),
            worker_sub_device_id="decode-worker",
            stall_group=("prefetch-worker", "decode-worker"),
            local_l1_size=0,
        )


@pytest.mark.host
def test_failed_activation_rolls_back_mode_and_running_prefetch(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.activate("decode")
    previous_result = owner.prefetch_result
    resources.fail_start_number = 2

    with pytest.raises(RuntimeError, match="prefetch start failure"):
        owner.activate("decode")

    assert owner.active_mode == "decode"
    assert owner.prefetch_result is not None
    assert owner.prefetch_result != previous_result
    assert resources.start_count == 3


@pytest.mark.host
def test_failed_stop_preserves_active_session_ownership(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.activate("decode")
    active_result = owner.prefetch_result
    resources.fail_stop_once = True

    with pytest.raises(RuntimeError, match="prefetch stop failure"):
        owner.activate("prefill")

    assert owner.active_mode == "decode"
    assert owner.prefetch_result == active_result
    owner.cleanup()
    assert active_result in resources.deallocated


@pytest.mark.host
def test_failed_stall_transition_restores_previous_mode_without_publishing_target(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.activate("prefill")
    owner.config.mesh_device.fail_next_stall = True

    with pytest.raises(RuntimeError, match="stall failure"):
        owner.activate("decode")

    assert owner.active_mode == "prefill"
    assert owner.prefetch_result is None
    assert owner.config.mesh_device.events[-2:] == [
        ("load", "manager-0"),
        ("stall", ("prefill-worker",)),
    ]


@pytest.mark.host
def test_cleanup_is_idempotent_releases_owned_results_and_never_weights(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    weight = seal_one(owner)
    _, decode = owner.seal()
    owner.activate("decode")
    prefetch_result = owner.prefetch_result
    metadata = decode.weight_address_metadata

    owner.cleanup()
    owner.cleanup()

    events = owner.config.mesh_device.events
    assert events.count(("reset_stall",)) == 1
    assert events.count(("clear_manager",)) == 1
    assert events.count(("remove", "manager-1")) == 1
    assert events.count(("remove", "manager-0")) == 1
    assert prefetch_result in resources.deallocated
    assert metadata in resources.deallocated
    assert weight not in resources.deallocated
    with pytest.raises(RuntimeError, match="cleaned up"):
        owner.activate("decode")


@pytest.mark.host
def test_cleanup_continues_after_failure_and_remains_idempotent(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.config.mesh_device.fail_remove = "manager-1"

    with pytest.raises(RuntimeError, match="remove failure"):
        owner.cleanup()
    owner.cleanup()
    assert ("remove", "manager-0") in owner.config.mesh_device.events


@pytest.mark.host
def test_context_manager_cleans_up_on_failure(resources):
    owner = Prefetcher2D(make_config(expected_weight_count=1), **resources.kwargs())
    with pytest.raises(RuntimeError, match="body failure"):
        with owner:
            raise RuntimeError("body failure")
    assert ("remove", "manager-1") in owner.config.mesh_device.events
    assert ("remove", "manager-0") in owner.config.mesh_device.events


@pytest.mark.host
def test_releasing_the_global_cb_on_prefill_recreates_it_on_the_next_decode(resources):
    """`release_global_cb_on_prefill` gives the buffer a per-mode lifetime.

    `defer_global_cb` only helps the *first* prefill; once decode has run, the
    buffer is resident again and a prefill after a decode is unplaceable, which is
    what a second runner in one process does. This releases and recreates it.

    The properties that matter: the release drops the sealed decode context's
    reference too (nothing else can free the L1), and the next decode activation
    makes a *new* buffer and rebinds it, so the prefetch program is never handed
    None.
    """

    owner = initialized_owner(
        resources, expected_weight_count=1, defer_global_cb=True, release_global_cb_on_prefill=True
    )
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101, 128))
    prefill, decode = owner.seal()

    owner.activate("decode")
    assert len(resources.created_cbs) == 1
    first = decode.global_cb
    assert first == resources.created_cbs[0]

    owner.activate("prefill")
    # Both references gone, so the C++ destructor can free the L1.
    assert decode.global_cb is None
    assert owner._global_cb is None
    assert prefill.global_cb is None

    owner.activate("decode")
    assert len(resources.created_cbs) == 2
    assert decode.global_cb == resources.created_cbs[1]
    starts = [event for event in resources.prefetch_events if event[0] == "start"]
    assert starts[-1][3] == resources.created_cbs[1]
    owner.cleanup()


@pytest.mark.host
def test_cleanup_takes_the_global_cb_back_out_of_every_context_it_handed_out(resources):
    """Cleanup must break the references it gave away, not just its own.

    A `Prefetcher2DContext` is captured by *value* at module construction -
    `MLP2DConfig.decode_prefetch_context` holds the context object and reads
    `getattr(context, "global_cb", None)` at call time - so dropping
    `self._contexts` leaves every already-built module holding a context whose
    `global_cb` is still the live buffer. There is no `deallocate` on a
    `global_circular_buffer`; its L1 is freed by the C++ destructor, so one
    surviving Python reference keeps ~774 kB per sender/receiver core allocated
    for the life of the process.

    Measured: after a model was closed, deleted and
    `gc.collect()`-ed, 923 776 of every 1 393 472 B L1 bank was still allocated
    and the second model in the process could not create its own buffer:

        TT_FATAL @ bank_manager.cpp:462 Out of Memory: Not enough space to
        allocate 55444480 B L1 buffer across 70 banks, where each bank needs to
        store 792064 B ... (allocated: 923776 B, free: 469696 B)

    Without the loop in `cleanup()` this test fails on the `is None` assertions
    with the buffer still bound to both contexts.
    """

    owner = initialized_owner(resources, expected_weight_count=1, defer_global_cb=True)
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101, 128))
    prefill, decode = owner.seal()
    owner.activate("decode")
    assert decode.global_cb == resources.created_cbs[0]

    owner.cleanup()

    assert decode.global_cb is None, "the sealed decode context still holds the global circular buffer"
    assert prefill.global_cb is None
    assert owner._global_cb is None


@pytest.mark.host
def test_release_on_prefill_leaves_no_context_holding_the_global_cb(resources):
    """No context may hold the buffer after a release. Pins an invariant; does
    **not** discriminate a fix.

    Written to test a hypothesis that turned out to be wrong, and kept because the
    invariant is worth pinning and because the refutation is worth recording. The
    hypothesis was that `_release_global_cb` had the same reference-retention
    defect `cleanup()` had - it cleared the owner's reference and the *decode*
    context's, and a
    `Prefetcher2DContext` is captured by value at module construction, so perhaps
    module-held references survived and that was why the release "ran and did not
    help" on hardware.

    It is not that. The decode context is the only context the buffer is ever
    assigned to (`_ensure_global_cb`), the owner's `_contexts` map is *not*
    cleared by the release path the way `cleanup()` cleared it, and the sealed
    contexts modules receive are the same objects that map holds. So the two
    references the old code cleared were the two that exist, and this test passes
    against the old code and the new one alike - which is exactly how it was
    established, in two minutes on the host, that this was not the explanation.

    `_release_global_cb` now loops over every context anyway, so it cannot drift
    away from `cleanup()`, but that is symmetry rather than a fix, and it is not
    evidence for anything on device.

    What remains true and measured: the buffer's L1 *is* returned when the last
    reference goes - 792 256 B per bank in a two-pool run. Why the release does
    not move the clash address is still open.
    """

    owner = initialized_owner(
        resources, expected_weight_count=1, defer_global_cb=True, release_global_cb_on_prefill=True
    )
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101, 128))
    prefill, decode = owner.seal()

    owner.activate("decode")
    assert len(resources.created_cbs) == 1
    first = decode.global_cb
    assert first == resources.created_cbs[0]

    owner.activate("prefill")
    # Both references gone, so the C++ destructor can free the L1.
    assert decode.global_cb is None
    assert owner._global_cb is None
    assert prefill.global_cb is None

    owner.activate("decode")
    assert len(resources.created_cbs) == 2
    assert decode.global_cb == resources.created_cbs[1]
    starts = [event for event in resources.prefetch_events if event[0] == "start"]
    assert starts[-1][3] == resources.created_cbs[1]
    owner.cleanup()


@pytest.mark.host
def test_release_without_defer_is_rejected(expect_error):
    with expect_error(ValueError, "requires defer_global_cb"):
        make_config(defer_global_cb=False, release_global_cb_on_prefill=True)


class ScriptedL1:
    """A small faithful model of the L1 free list, with one governing rule.

    `FreeListOpt::allocate` takes the **smallest free block that fits**, and L1 is
    allocated top-down inside the block it picks. Everything the global-CB
    placement fix depends on follows from that one rule, so the model derives its
    free blocks from its allocated ones rather than tracking them, and cannot
    drift out of agreement with itself:

    * a buffer created while something large is resident lands below it, and an
      allocated address never moves;
    * a request that fits a **gap** above the low region lands in that gap and
      does not lower the free top - which is why one reservation was not enough
      on silicon: the buffer came back 32 736 B high on both models;
    * a global circular buffer is **two** allocations, a `size`-byte data buffer
      and a 192-byte config page, and by the same rule a small gap anywhere in L1
      captures the config page - which on silicon put it 850 kB from its own data
      buffer.

    Tests build a hole by seeding `blocks` with a block that leaves one.
    """

    CONFIG_PAGE = 192
    TOP = 1_000_032

    def __init__(self, *, blocks=None, creation_extra=0):
        #: allocated blocks, address -> size. One resident block at the very top.
        self.blocks = dict(blocks) if blocks else {1_000_000: 32}
        #: bytes the creation allocates *beyond* the data buffer and its config
        #: page. On silicon this is not constant, which is why a fixed headroom
        #: cannot pin the buffer.
        self.creation_extra = creation_extra
        self.reads = 0
        self.reserved = []
        self.released = []

    def free_blocks(self):
        """Derive the free blocks: the complement of the allocated ones."""

        free, cursor = [], 0
        for address in sorted(self.blocks):
            if address > cursor:
                free.append((cursor, address - cursor))
            cursor = address + self.blocks[address]
        if cursor < self.TOP:
            free.append((cursor, self.TOP - cursor))
        return free

    def l1_block_table(self, mesh):
        del mesh
        self.reads += 1
        table = [(address, size, True) for address, size in self.blocks.items()]
        table.extend((address, size, False) for address, size in self.free_blocks())
        return tuple(table)

    def _allocate(self, size):
        candidates = [block for block in self.free_blocks() if block[1] >= size]
        if not candidates:
            raise AssertionError(f"the scripted L1 cannot fit {size} bytes")
        address, extent = min(candidates, key=lambda block: (block[1], block[0]))
        placed = address + extent - size  # top-down inside the chosen block
        self.blocks[placed] = size
        return placed

    def reserve_l1(self, mesh, size):
        del mesh
        self.reserved.append(size)
        return ("reservation", self._allocate(size), size)

    def release(self, resource):
        self.released.append(resource)
        if isinstance(resource, tuple) and resource[0] == "reservation":
            self.blocks.pop(resource[1], None)

    def create_global_cb(self, mesh, mapping, size):
        del mesh, mapping
        if self.creation_extra:
            self._allocate(self.creation_extra)
        data = self._allocate(size)
        self._allocate(self.CONFIG_PAGE)
        return ("global-cb", data, size)

    def release_global_cb(self, cb):
        """Free the blocks the creation made, as dropping the last handle does."""

        data = cb[1]
        for address in (data, data - self.CONFIG_PAGE):
            self.blocks.pop(address, None)


class NonLoweringL1(ScriptedL1):
    """An L1 whose reservations never take anything: the loops must be bounded."""

    def reserve_l1(self, mesh, size):
        del mesh
        self.reserved.append(size)
        return ("reservation", None, size)


def _headroom_owner(resources, l1, *, headroom, create_global_cb=None):
    owner = Prefetcher2D(
        make_config(
            expected_weight_count=1,
            defer_global_cb=True,
            release_global_cb_on_prefill=True,
            global_cb_headroom=headroom,
        ),
        create_global_cb=create_global_cb or resources.create_global_cb,
        create_address_metadata=resources.create_metadata,
        deallocate=l1.release,
        dram_prefetch_start=resources.start,
        dram_prefetch_stop=resources.stop,
        l1_block_table=l1.l1_block_table,
        reserve_l1=l1.reserve_l1,
    )
    owner.initialize()
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101, 128))
    owner.seal()
    return owner


@pytest.mark.host
def test_the_global_cb_is_created_under_reserved_headroom():
    """The headroom is taken before the buffer and given back after it.

    L1 is allocated top-down and a buffer's address never moves, so anything
    long-lived allocated while the ~774 kB global circular buffer is resident is
    stranded *below* it for the life of the process - and the Galaxy prefill
    embedding's static circular buffers reach 630080, so one stranded byte down
    there makes every later prefill unplaceable. Measured on `(8, 4)`: a 32-byte
    block at 545760 is attributed to the first decode, and it is gone once
    64 kiB of headroom is reserved first.

    Without the reservation this fails on `reserved == [4096]`.
    """

    resources = ResourceHarness()
    l1 = ScriptedL1()
    owner = _headroom_owner(resources, l1, headroom=4096, create_global_cb=l1.create_global_cb)

    owner.activate("decode")

    assert l1.reserved == [4096], "the headroom was not reserved before the global CB"
    assert l1.released, "the headroom was never released"


@pytest.mark.host
def test_a_recreated_global_cb_returns_to_its_first_l1_blocks_over_a_leftover_gap():
    """The buffer comes back to the same blocks even when a gap would swallow the reservation.

    `release_global_cb_on_prefill` frees the buffer on the way into prefill and
    `_ensure_global_cb` recreates it on the way back into decode - and decode
    programs already in the ttnn program cache hold its address. Measured with a
    *fixed* headroom on `(8, 4)`:
    the buffer landed at 447296 and then at 414880, 32 416 B apart, and the decode
    after the recreation hung with no output for four minutes.

    This models the free list the production path presented on the second
    creation: a decode block inside the old headroom, leaving a gap of **exactly**
    the bytes the free top has to come down by. A single reservation of that size
    lands in the gap and lowers nothing, which is why the buffer came back 32 736 B
    high on both models.
    """

    resources = ResourceHarness()
    l1 = ScriptedL1()
    owner = _headroom_owner(resources, l1, headroom=4096, create_global_cb=l1.create_global_cb)

    owner.activate("decode")
    first = dict(l1.blocks)
    cb = owner._global_cb
    owner.activate("prefill")
    l1.release_global_cb(cb)
    # The first decode's own 1024 B, placed so that the leftover gap above it is
    # exactly the distance the free top now has to travel: 997440 - 995904 = 1536,
    # and 1000000 - 997440 - 1024 = 1536.
    l1.blocks[997_440] = 1024

    owner.activate("decode")

    assert dict(l1.blocks) == {**first, 997_440: 1024}, "the recreated buffer moved"


@pytest.mark.host
def test_a_free_gap_cannot_capture_the_global_cb_config_page():
    """A 192-byte hole anywhere in L1 would take the config page; it is held first.

    A global circular buffer is two allocations and a cached program holds the
    address of **both** - `CircularBufferImpl::set_global_circular_buffer`
    captures ``buffer_address()`` and ``config_address()`` once, and
    `dispatch.cpp` re-sends the captured pair on every launch. Since
    `FreeListOpt::allocate` takes the smallest free block that fits, a small hole
    in the resident region captures the config page and moves it away from its own
    data buffer. Measured on `(8, 4)`, two fresh processes: the data
    buffer came back at 510816 both times while the config page moved from 510624
    to **1367872**, and the chunked-prefill claim failed on it.

    Without the gap-filling pass the config page lands in the hole and the block
    set no longer matches.
    """

    resources = ResourceHarness()
    l1 = ScriptedL1()
    owner = _headroom_owner(resources, l1, headroom=4096, create_global_cb=l1.create_global_cb)

    owner.activate("decode")
    first = dict(l1.blocks)
    cb = owner._global_cb
    owner.activate("prefill")
    l1.release_global_cb(cb)
    # A 32-byte block at 999776 leaves a 192-byte hole at 999808, exactly the
    # config page's size, high above where the buffer goes.
    l1.blocks[999_776] = 32

    owner.activate("decode")

    assert dict(l1.blocks) == {**first, 999_776: 32}, "the config page did not come back where it was"
    assert 999_808 not in l1.blocks, "the hole was reserved but never given back"


@pytest.mark.host
def test_a_recreated_global_cb_that_moves_is_reported_rather_than_used(expect_error):
    """A buffer that cannot be put back where it was is an error, not a silent read.

    Decode programs in the ttnn program cache hold the old address; continuing
    with a buffer somewhere else reads the wrong L1 and produces no error at all -
    on silicon it hangs. So the owner checks and raises.
    """

    resources = ResourceHarness()
    l1 = ScriptedL1()
    owner = _headroom_owner(resources, l1, headroom=4096, create_global_cb=l1.create_global_cb)
    owner.activate("decode")
    cb = owner._global_cb
    owner.activate("prefill")
    l1.release_global_cb(cb)
    # The next creation costs more than the first did, so the buffer lands lower.
    l1.creation_extra = 512
    with expect_error(RuntimeError, "did not land on its original L1 blocks"):
        owner.activate("decode")


@pytest.mark.host
def test_a_free_top_that_never_comes_down_is_reported(expect_error):
    """A reservation loop that cannot reach its target is bounded, not spun on."""

    resources = ResourceHarness()
    l1 = NonLoweringL1()
    owner = _headroom_owner(resources, l1, headroom=4096, create_global_cb=l1.create_global_cb)
    with expect_error(RuntimeError, "could not lower the L1 free top"):
        owner.activate("decode")


@pytest.mark.host
def test_a_gap_that_never_fills_is_reported(expect_error):
    """The gap-filling pass is bounded too, and says which pass gave up."""

    resources = ResourceHarness()
    l1 = NonLoweringL1(blocks={1_000_000: 32, 999_776: 32})
    owner = _headroom_owner(resources, l1, headroom=4096, create_global_cb=l1.create_global_cb)
    with expect_error(RuntimeError, "could not fill the free L1 gaps"):
        owner.activate("decode")


@pytest.mark.host
def test_zero_headroom_creates_the_global_cb_exactly_as_before():
    """The default is byte-for-byte the previous behaviour: no reservation, no query."""

    resources = ResourceHarness()
    l1 = ScriptedL1()
    owner = _headroom_owner(resources, l1, headroom=0)

    owner.activate("decode")

    assert l1.reserved == []
    assert l1.reads == 0, "zero headroom must not even query the allocator"
    assert len(resources.created_cbs) == 1


@pytest.mark.host
def test_negative_headroom_is_rejected(expect_error):
    with expect_error(ValueError, "global_cb_headroom cannot be negative"):
        make_config(global_cb_headroom=-1)


@pytest.mark.host
def test_a_shared_placement_record_pins_a_second_owner_to_the_first_ones_blocks():
    """Two owners on one mesh put the buffer in the same L1 blocks, or it is reported.

    The ttnn program cache belongs to the **mesh device** and outlives any one
    model, and two structurally identical models hash to the same program-cache
    keys - so the second model's decode reuses programs compiled for the first,
    and those programs carry the first model's global circular buffer addresses.
    Measured on `(8, 4)`: with the record held per owner, the second Qwen model's
    first decode hung in `FDMeshCommandQueue::wait_for_outstanding_reads` with
    both models' weights already resident.

    Without the shared record the second owner records a fresh free top and the
    buffer lands wherever the second model's own resident L1 puts it.
    """

    resources = ResourceHarness()
    l1 = ScriptedL1()
    placement = GlobalCBPlacement()

    first = Prefetcher2D(
        make_config(
            expected_weight_count=1,
            defer_global_cb=True,
            release_global_cb_on_prefill=True,
            global_cb_headroom=4096,
        ),
        create_global_cb=l1.create_global_cb,
        create_address_metadata=resources.create_metadata,
        deallocate=l1.release,
        dram_prefetch_start=resources.start,
        dram_prefetch_stop=resources.stop,
        l1_block_table=l1.l1_block_table,
        reserve_l1=l1.reserve_l1,
        global_cb_placement=placement,
    )
    first.initialize()
    first.register_weight("weight", FakeTensor(first.config.mesh_device, 101, 128))
    first.seal()
    first.activate("decode")
    cb = first._global_cb
    blocks = dict(l1.blocks)
    first.cleanup()
    l1.release_global_cb(cb)

    assert placement.blocks is not None, "the first owner did not record its placement"

    second = Prefetcher2D(
        make_config(
            expected_weight_count=1,
            defer_global_cb=True,
            release_global_cb_on_prefill=True,
            global_cb_headroom=4096,
        ),
        create_global_cb=l1.create_global_cb,
        create_address_metadata=resources.create_metadata,
        deallocate=l1.release,
        dram_prefetch_start=resources.start,
        dram_prefetch_stop=resources.stop,
        l1_block_table=l1.l1_block_table,
        reserve_l1=l1.reserve_l1,
        global_cb_placement=placement,
    )
    second.initialize()
    second.register_weight("weight", FakeTensor(second.config.mesh_device, 101, 128))
    second.seal()
    # The second model's resident L1 differs by one block, which is enough to move
    # the buffer unless the recorded free top is reproduced.
    l1.blocks[999_936] = 64
    second.activate("decode")

    assert dict(l1.blocks) == {**blocks, 999_936: 64}, "the second owner's buffer did not land on the first's blocks"


@pytest.mark.host
def test_an_owner_with_no_shared_record_keeps_its_own():
    """The default is the previous per-owner behaviour, unchanged."""

    resources = ResourceHarness()
    l1 = ScriptedL1()
    owner = _headroom_owner(resources, l1, headroom=4096, create_global_cb=l1.create_global_cb)

    owner.activate("decode")

    assert owner._global_cb_placement.blocks is not None
    assert (
        owner._global_cb_placement is not _headroom_owner(resources, ScriptedL1(), headroom=4096)._global_cb_placement
    )


@pytest.mark.host
def test_a_gap_that_is_not_a_multiple_of_32_is_taken_as_far_as_it_can_be():
    """An unaligned gap leaves a sliver, and a sliver cannot hold the config page.

    An L1 reservation is a whole number of 32-byte units, so a 936-byte hole can
    only be taken 928 bytes deep. That is safe rather than approximate: the 8-byte
    leftover is smaller than anything the creation allocates, the smallest of which
    is the 192-byte config page.
    """

    resources = ResourceHarness()
    l1 = ScriptedL1()
    owner = _headroom_owner(resources, l1, headroom=4096, create_global_cb=l1.create_global_cb)

    owner.activate("decode")
    first = dict(l1.blocks)
    cb = owner._global_cb
    owner.activate("prefill")
    l1.release_global_cb(cb)
    # A 64-byte block at 999000 leaves a 936-byte hole at 999064, which is 8 bytes
    # more than a multiple of 32.
    l1.blocks[999_000] = 64

    owner.activate("decode")

    assert dict(l1.blocks) == {**first, 999_000: 64}, "the unaligned gap captured part of the creation"
    assert 928 in l1.reserved, "the gap was not taken as far as 32-byte units allow"


@pytest.mark.host
def test_cleanup_announces_the_global_cb_release_once_the_buffer_is_gone():
    """`on_global_cb_released` fires from `cleanup`, and only after the release.

    A program in the ttnn program cache captures the global circular buffer's
    ``buffer_address()`` and ``config_address()`` once and re-sends the pair on
    every launch, and it holds its semaphores - 32 bytes of L1 each - for as long
    as it is cached. So when the buffer goes away, something has to be able to
    retire the programs that still refer to it. The module does not know how many
    models share the mesh, so it announces the release and the caller decides.

    Measured consequence of having no such announcement, on `(8, 4)`: a
    second model in one process found 77 extra 32-byte L1 blocks resident, which
    fragmented the free list enough to displace a 64 kB allocation 109 376 B
    downward and made the buffer unplaceable at its recorded address.
    """

    resources = ResourceHarness()
    l1 = ScriptedL1()
    announced: list[str] = []

    owner = Prefetcher2D(
        make_config(
            expected_weight_count=1,
            defer_global_cb=True,
            release_global_cb_on_prefill=True,
            global_cb_headroom=4096,
            on_global_cb_released=lambda: announced.append("released"),
        ),
        create_global_cb=l1.create_global_cb,
        create_address_metadata=resources.create_metadata,
        deallocate=l1.release,
        dram_prefetch_start=resources.start,
        dram_prefetch_stop=resources.stop,
        l1_block_table=l1.l1_block_table,
        reserve_l1=l1.reserve_l1,
    )
    owner.initialize()
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101, 128))
    owner.seal()
    owner.activate("decode")

    assert announced == [], "the release was announced while the buffer was still live"
    owner.cleanup()
    assert announced == ["released"], "cleanup did not announce the global CB release"

    # Terminal and idempotent, exactly as `cleanup` itself is: a second cleanup
    # must not retire the mesh's programs a second time.
    owner.cleanup()
    assert announced == ["released"], "cleanup announced the release more than once"


@pytest.mark.host
def test_an_owner_with_no_release_listener_behaves_exactly_as_before():
    """The hook defaults to `None`, and that path is byte-for-byte the old one."""

    resources = ResourceHarness()
    l1 = ScriptedL1()
    owner = Prefetcher2D(
        make_config(expected_weight_count=1, defer_global_cb=True),
        create_global_cb=l1.create_global_cb,
        create_address_metadata=resources.create_metadata,
        deallocate=l1.release,
        dram_prefetch_start=resources.start,
        dram_prefetch_stop=resources.stop,
        l1_block_table=l1.l1_block_table,
        reserve_l1=l1.reserve_l1,
    )
    assert owner.config.on_global_cb_released is None
    owner.initialize()
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101, 128))
    owner.seal()
    owner.activate("decode")
    owner.cleanup()


@pytest.mark.host
def test_no_persistent_sender_is_dispatched_while_a_trace_capture_is_in_progress(resources):
    """A capture region records the consumer; the sender must not be launched.

    `ttnn.dram_prefetcher` is a persistent program that blocks on global-CB
    credits from the decode graph. Inside a capture region that graph is recorded
    rather than run, so a launched sender strands itself on the prefetch sender
    cores — and the prefill subdevice covers those cores, which is what hung
    `ttnn.end_trace_capture` when it was measured. Everything else about the
    activation is unchanged, the stall group included.
    """

    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    capturing = {"value": False}
    owner.set_capture_probe(lambda: capturing["value"])

    # Eager: the sender is launched exactly as it always was.
    owner.activate("decode")
    assert [event[0] for event in resources.prefetch_events] == ["start"]
    assert owner.prefetch_result is not None

    # Capturing: the running sender is stopped and no new one is launched.
    capturing["value"] = True
    owner.activate("decode")
    assert [event[0] for event in resources.prefetch_events] == ["start", "stop"]
    assert owner.prefetch_result is None

    # The stall group is still where `_start_prefetch` would have left it, so a
    # blocking finish inside the capture region waits on the worker subdevice
    # only, never on the prefetch senders.
    stall_events = [event[1] for event in owner.config.mesh_device.events if event[0] == "stall"]
    assert stall_events[-1] == owner.config.decode.stall_group

    # And the capture is over: the next activation launches the sender again,
    # immediately before the replay that consumes it.
    capturing["value"] = False
    owner.activate("decode")
    assert [event[0] for event in resources.prefetch_events] == ["start", "stop", "start"]
    assert owner.prefetch_result is not None


@pytest.mark.host
def test_capture_probe_is_rejected_unless_callable_or_none(resources, expect_error):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)

    with expect_error(TypeError, "capture probe"):
        owner.set_capture_probe("yes")
    owner.set_capture_probe(None)
    owner.activate("decode")
    assert owner.prefetch_result is not None


@pytest.mark.host
def test_the_graph_launch_site_moves_the_sender_out_of_the_operation_boundary(resources):
    """`activate` stops launching; the graph body launches instead.

    A boundary launch is dispatched *for real* while the graph that consumes its
    global-CB credits is only being recorded, so capture and replay see different
    sender state. TTTv1 launches `ttnn.dram_prefetcher` from inside `forward`
    (`models/demos/llama3_70b_galaxy/tt/llama_model.py:790-808`) for exactly that
    reason, and this is the seam that lets this owner do the same.
    """

    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    assert owner.sender_launch_site == "boundary"

    owner.set_sender_launch_site("graph")
    owner.activate("decode")
    assert [event[0] for event in resources.prefetch_events] == []
    assert owner.prefetch_result is None
    # The stall group is still exactly where a boundary launch would leave it.
    stall_events = [event[1] for event in owner.config.mesh_device.events if event[0] == "stall"]
    assert stall_events[-1] == owner.config.decode.stall_group

    owner.launch_sender()
    assert [event[0] for event in resources.prefetch_events] == ["start"]
    assert owner.prefetch_result is not None

    # A second body launch replaces the sender rather than stranding it.
    owner.launch_sender()
    assert [event[0] for event in resources.prefetch_events] == ["start", "stop", "start"]


@pytest.mark.host
def test_the_boundary_site_refuses_a_graph_launch(resources, expect_error):
    """Exactly one site launches, so the unselected one must refuse."""

    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.activate("decode")
    with expect_error(RuntimeError, "requires set_sender_launch_site"):
        owner.launch_sender()
    assert [event[0] for event in resources.prefetch_events] == ["start"]


@pytest.mark.host
def test_a_graph_launch_outside_decode_is_refused(resources, expect_error):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.set_sender_launch_site("graph")
    owner.activate("prefill")
    with expect_error(RuntimeError, "decode-only"):
        owner.launch_sender()


@pytest.mark.host
def test_an_unknown_sender_launch_site_is_refused(resources, expect_error):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    with expect_error(ValueError, "boundary"):
        owner.set_sender_launch_site("wherever")
    assert owner.sender_launch_site == "boundary"
