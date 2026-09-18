# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host cover for the capture-phase free deferral.

The defect these tests pin: a DRAM buffer freed inside a trace capture body
returns its address to the allocator while the graph being recorded still writes
into it, so a later allocation in the same body lands on a Galaxy collective's
fabric-write destination. Measured on silicon as 18 of 40 traced prefill replays
matching eager at 80 Llama layers, against 40 of 40 with the frees held.

Everything here is host-only: `CaptureFreeBindings` names the two free entry
points so the mechanism can be driven against fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests.models.galaxy.test_resources import (
    FakeMesh,
    FakePrefetcherWithCaptureProbe,
    FakeTTNN,
    resource_config,
)

from tt_transformers.models.galaxy.capture_frees import CaptureFreeBindings, CaptureFreeDeferral

DRAM = "dram"
L1 = "l1"


class FakeTensor:
    """A tensor whose `deallocate` is a method on a swappable class."""

    def __init__(self, address: int, *, buffer_type: str = DRAM, size: int = 4):
        self.address = address
        self.buffer_type = buffer_type
        self.freed = 0
        self._size = size

    # The seam under test: `tensor.deallocate(True)`.
    def deallocate(self, force: bool = False) -> None:
        self.freed += 1

    def is_allocated(self) -> bool:
        return self.freed == 0

    def memory_config(self):
        return SimpleNamespace(buffer_type=self.buffer_type)

    def buffer_address(self) -> int:
        return self.address

    def volume(self) -> int:
        return self._size

    def element_size(self) -> int:
        return 1


class FakeModule:
    """Stands in for `ttnn`, whose `deallocate` is resolved at call time."""

    def __init__(self):
        self.freed: list[FakeTensor] = []

    def deallocate(self, tensor, *args, **kwargs):
        self.freed.append(tensor)
        FakeTensor.deallocate(tensor, True)


def bindings(module: FakeModule) -> CaptureFreeBindings:
    return CaptureFreeBindings(module=module, tensor_type=FakeTensor, dram_buffer_type=DRAM)


def deferral(module: FakeModule, capturing, **kwargs) -> CaptureFreeDeferral:
    return CaptureFreeDeferral(capturing=capturing, bindings=bindings(module), **kwargs)


@pytest.mark.host
@pytest.mark.model
def test_a_dram_free_inside_a_capture_body_is_held_until_the_phase_ends():
    module = FakeModule()
    capturing = True
    tensor = FakeTensor(0x1000)

    with deferral(module, lambda: capturing) as hold:
        tensor.deallocate(True)
        # This is the whole defect: at this instant the unmodified tree has
        # returned the address to the allocator, and the capture body is still
        # allocating.
        assert tensor.freed == 0
        assert hold.deferred_count == 1
        assert hold.deferred_bytes == 4

    assert tensor.freed == 1
    assert hold.flushed_count == 1


@pytest.mark.host
@pytest.mark.model
def test_the_module_level_entry_point_is_held_too():
    module = FakeModule()
    tensor = FakeTensor(0x2000)

    with deferral(module, lambda: True) as hold:
        module.deallocate(tensor)
        assert tensor.freed == 0
        assert hold.deferred_count == 1

    assert tensor.freed == 1


@pytest.mark.host
@pytest.mark.model
def test_an_l1_free_is_never_held():
    """`b1_L20_capture_hold_ablation`: a program's static CBs need that L1 back.

    It aborted with `Statically allocated circular buffers in program 880 clash
    with L1 buffers on core range [1-0 - 3-9]` and never reached a replay.
    """

    module = FakeModule()
    tensor = FakeTensor(0x3000, buffer_type=L1)

    with deferral(module, lambda: True) as hold:
        tensor.deallocate(True)
        assert tensor.freed == 1
        assert hold.deferred_count == 0


@pytest.mark.host
@pytest.mark.model
def test_nothing_is_held_outside_a_capture_region():
    module = FakeModule()
    tensor = FakeTensor(0x4000)

    with deferral(module, lambda: False) as hold:
        tensor.deallocate(True)
        module.deallocate(FakeTensor(0x4100))
        assert tensor.freed == 1
        assert hold.deferred_count == 0


@pytest.mark.host
@pytest.mark.model
def test_a_borrowed_plan_buffer_is_dropped_from_the_flush_rather_than_freed():
    """`b5_L80_release_after_capture`, which aborted for want of this guard.

    It force-freed three buffers the Galaxy resource owner still lends to every
    collective, and the next op died with
    `TT_FATAL @ device_operation.hpp:456 input_tensor.is_allocated()`.
    """

    module = FakeModule()
    borrowed = FakeTensor(0x5000)
    transient = FakeTensor(0x5100)

    with deferral(module, lambda: True, protected_addresses=lambda: frozenset({0x5000})) as hold:
        borrowed.deallocate(True)
        transient.deallocate(True)

    assert borrowed.freed == 0
    assert transient.freed == 1
    assert hold.protected_count == 1
    assert hold.flushed_count == 1


@pytest.mark.host
@pytest.mark.model
def test_the_hold_spans_every_capture_region_in_one_phase():
    """One phase captures decode and then prefill; the flush waits for both.

    `mesh_device.cpp:1337`: a second capture re-enables "safe" allocation while
    the first trace is live, so releasing decode's addresses before prefill's
    capture body allocates is the same defect one operation later.
    """

    module = FakeModule()
    first = FakeTensor(0x6000)
    second = FakeTensor(0x6100)
    hold = deferral(module, lambda: True)

    with hold:
        first.deallocate(True)
        with hold:
            second.deallocate(True)
        # The inner region ended and nothing was released.
        assert first.freed == 0
        assert second.freed == 0

    assert first.freed == 1
    assert second.freed == 1


@pytest.mark.host
@pytest.mark.model
def test_the_entry_points_are_restored_even_when_the_body_raises(expect_error):
    module = FakeModule()
    original_module_free = module.deallocate
    original_tensor_free = FakeTensor.deallocate
    tensor = FakeTensor(0x7000)

    with expect_error(RuntimeError, "capture failed"):
        with deferral(module, lambda: True):
            tensor.deallocate(True)
            raise RuntimeError("capture failed")

    assert module.deallocate == original_module_free
    assert FakeTensor.deallocate is original_tensor_free
    # A failed capture still owes the allocator its addresses back.
    assert tensor.freed == 1


@pytest.mark.host
@pytest.mark.model
def test_a_tensor_freed_twice_inside_the_body_is_freed_once_on_flush():
    module = FakeModule()
    tensor = FakeTensor(0x8000)

    with deferral(module, lambda: True) as hold:
        tensor.deallocate(True)
        module.deallocate(tensor)
        assert hold.deferred_count == 2

    assert tensor.freed == 1


@pytest.mark.host
@pytest.mark.model
def test_the_resource_owner_binds_its_own_capture_probe_and_its_own_lent_buffers():
    """The owner is the only thing that knows both halves of the question."""

    mesh = FakeMesh()
    fake_ttnn = FakeTTNN()
    config = resource_config()
    prefetcher = FakePrefetcherWithCaptureProbe(mesh, config)

    addresses = iter(range(0x10000, 0x20000, 0x100))

    def allocate_tensor(mesh_device, spec):
        return FakeTensor(next(addresses))

    from tt_transformers.models.galaxy import GalaxyResourceBindings, create_galaxy_resources

    bound = GalaxyResourceBindings(
        create_semaphore=fake_ttnn.create_semaphore,
        reset_semaphore=fake_ttnn.reset_semaphore,
        allocate_tensor=allocate_tensor,
        deallocate_tensor=fake_ttnn.deallocate_tensor,
        synchronize=fake_ttnn.synchronize,
    )
    owner = create_galaxy_resources(mesh, config=config, prefetcher=prefetcher, bindings=bound)
    capturing = {"value": False}
    owner.set_capture_probe(lambda: capturing["value"])
    owner.activate("decode")

    hold = owner.defer_capture_frees()
    assert hold._capturing() is False
    capturing["value"] = True
    assert hold._capturing() is True

    lent = hold._protected_addresses()
    assert lent, "the decode plan's persistent buffers must be reported as borrowed"
    assert all(isinstance(address, int) for address in lent)
