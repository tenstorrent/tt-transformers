# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hold device DRAM deallocations for as long as a trace capture is open.

Why this exists, in one paragraph. tt-metal's allocator says of a live trace
(`tt_metal/impl/allocator/allocator.cpp:113-126`) that buffers allocated while
one exists "are meant to ... have a lifetime that ends before the trace is
executed", and there is no enforcement behind that comment. The unwritten half
of the same contract is what this module implements: a buffer *freed* inside a
capture body returns its address to the allocator **while that address is still
recorded in the trace being captured**, so a later allocation in the same body
can be placed on top of a buffer the captured graph still writes into. Locally
that is harmless - this device's program order is what the recycling assumed.
Across the mesh it is not: a Galaxy collective's destination is written by *peer*
devices over the fabric, the host barrier that orders those writes against a free
cannot be recorded into a trace (`GalaxyResources._synchronize`), and on replay
there is no barrier at all. The result is a fabric write landing in whatever
tensor now owns the address, on a random layer of a random replay.

Measured on silicon at 80 Llama layers with 40 traced prefill replays per arm:

| arm | replays matching eager | bit-identical to eager |
| --- | --- | --- |
| unmodified tree | 18 of 40 | 10 of 40 |
| every capture-body DRAM free suppressed, held through the replays | 40 of 40 | 37 of 40 |
| the same, but the whole hold **released before the replays** | **40 of 40** | **40 of 40** |

`b6` is what this module is: the hold has to last exactly as long as the capture
phase, and not one moment longer. Nothing needs to stay resident - `b6` releases
1110 MB of it, runs an eager prefill that allocates freely, and then replays
forty times bit-identically. The price is a transient DRAM peak while capturing,
which `b6` shows fits on a WH Galaxy at 80 layers.

L1 is deliberately **not** deferred. `b1_L20_capture_hold_ablation` held L1 too
and never reached a replay: a prefill program's statically allocated circular
buffers need the L1 the previous program frees, and it aborted with
`Statically allocated circular buffers in program 880 clash with L1 buffers`.

**This is not the whole fix, and it does not pretend to be.** Upstream's traced
Galaxy model (`models/demos/llama3_70b_galaxy/tt/llama_ccl.py`) needs none of
this, because no fabric-written buffer is ever freed inside its graph body: every
CCL destination and intermediate is pre-allocated at process lifetime and
double-buffered. Reaching that discipline is a change to every 2D module's
ownership contract. This module buys the same invariant - no address recycled
inside a capture body - at the capture boundary instead, in one place, with no
change to any module's behaviour outside a capture region.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import ttnn
from loguru import logger


@dataclass(frozen=True)
class CaptureFreeBindings:
    """The two entry points every device free in this stack passes through.

    `src/tt_transformers/modules/**/*_2d.py` frees a tensor either as
    `ttnn.deallocate(tensor)` - resolved on the module object at call time - or
    as `tensor.deallocate(True)`, which is `ttnn.Tensor.deallocate`. Both are
    named here rather than reached for directly so that the whole mechanism is
    host-testable against fakes; production takes the defaults.
    """

    module: Any = ttnn
    tensor_type: Any = ttnn.Tensor
    dram_buffer_type: Any = ttnn.BufferType.DRAM


def _buffer_address(tensor: Any) -> int | None:
    getter = getattr(tensor, "buffer_address", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except BaseException:
        return None


class CaptureFreeDeferral:
    """Defer DRAM frees issued while `capturing()` is true; flush on exit.

    Re-entrant: an inner `with` block is a no-op, so the outermost region owns
    the flush. That matters because one capture phase captures *every* operation
    - `TraceCompiler.capture_all` records decode and then prefill - and freeing
    decode's addresses before prefill's capture body allocates is the same defect
    one operation later (`mesh_device.cpp:1337`: a second capture re-enables
    "safe" allocation while the first trace is live).
    """

    def __init__(
        self,
        *,
        capturing: Callable[[], bool],
        protected_addresses: Callable[[], frozenset[int]] | None = None,
        bindings: CaptureFreeBindings | None = None,
    ) -> None:
        self._capturing = capturing
        self._protected_addresses = protected_addresses
        self._bindings = bindings or CaptureFreeBindings()
        self._deferred: list[Any] = []
        self._deferred_ids: set[int] = set()
        self._installed = False
        self._depth = 0
        self._originals: dict[str, Any] = {}
        #: Diagnostics, read by the device tests and worth logging: how many
        #: frees were held, and how many bytes of DRAM that peak cost.
        self.deferred_count = 0
        self.deferred_bytes = 0
        self.flushed_count = 0
        self.protected_count = 0

    # Public surface

    def __enter__(self) -> CaptureFreeDeferral:
        self._depth += 1
        if self._depth == 1:
            self._install()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._depth -= 1
        if self._depth > 0:
            return False
        try:
            self._remove()
        finally:
            self.flush()
        return False

    def flush(self) -> int:
        """Free everything held, except a borrowed plan-owned buffer.

        The exception is load-bearing and was measured: `b5` released its whole
        hold including three buffers the Galaxy resource owner still hands to
        every collective, and the next op died with
        `TT_FATAL @ device_operation.hpp:456 input_tensor.is_allocated()`.
        `l1_L80_borrowed_buffer_free_watch` then established that the unmodified
        capture body frees exactly one such buffer and that it is in L1, so this
        guard should never fire on today's tree - it is here so that the day a
        module does free a plan buffer, the consequence is a leak and a count,
        not a corrupted persistent buffer.
        """

        protected = self._protected_addresses() if self._protected_addresses is not None else frozenset()
        held = self._deferred
        self._deferred = []
        self._deferred_ids = set()
        freed = 0
        for tensor in held:
            if protected and _buffer_address(tensor) in protected:
                self.protected_count += 1
                continue
            try:
                is_allocated = getattr(tensor, "is_allocated", None)
                if callable(is_allocated) and not is_allocated():
                    continue
                self._originals["tensor_free"](tensor, True)
                freed += 1
            except BaseException:
                # A flush is cleanup. One tensor that cannot be freed must not
                # strand the rest of a 1.1 GB hold.
                continue
        self.flushed_count += freed
        if held:
            # Logged because it is the only place a device log records that the
            # hold engaged at all, and how much it cost. Every arm of the
            # investigation had to be read for exactly these two numbers.
            logger.info(
                f"Released the trace capture hold: {freed} DRAM tensors freed, "
                f"{self.protected_count} left with their borrowing collective, "
                f"{self.deferred_count} frees held at a peak of {self.deferred_bytes / 1e6:.1f} MB per device"
            )
        return freed

    # Private implementation

    def _install(self) -> None:
        if self._installed:
            return
        module = self._bindings.module
        tensor_type = self._bindings.tensor_type
        original_module_free = module.deallocate
        original_tensor_free = tensor_type.deallocate
        deferral = self

        def module_free(tensor, *args, **kwargs):
            if deferral._hold(tensor):
                return None
            return original_module_free(tensor, *args, **kwargs)

        def tensor_free(tensor, *args, **kwargs):
            if deferral._hold(tensor):
                return None
            return original_tensor_free(tensor, *args, **kwargs)

        self._originals = {
            "module_free": original_module_free,
            "tensor_free": original_tensor_free,
        }
        module.deallocate = module_free
        tensor_type.deallocate = tensor_free
        self._installed = True

    def _remove(self) -> None:
        if not self._installed:
            return
        self._bindings.module.deallocate = self._originals["module_free"]
        self._bindings.tensor_type.deallocate = self._originals["tensor_free"]
        self._installed = False

    def _hold(self, tensor: Any) -> bool:
        if tensor is None:
            return False
        if not self._capturing():
            return False
        if not self._is_dram(tensor):
            return False
        if id(tensor) not in self._deferred_ids:
            self._deferred.append(tensor)
            self._deferred_ids.add(id(tensor))
            try:
                self.deferred_bytes += int(tensor.volume()) * int(tensor.element_size())
            except BaseException:
                pass
        self.deferred_count += 1
        return True

    def _is_dram(self, tensor: Any) -> bool:
        memory_config = getattr(tensor, "memory_config", None)
        if not callable(memory_config):
            return False
        try:
            return memory_config().buffer_type == self._bindings.dram_buffer_type
        except BaseException:
            return False
