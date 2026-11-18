# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blocking and asynchronous device-to-host output read lifecycle."""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
import ttnn


@dataclass(frozen=True)
class PendingRead:
    """One retained asynchronous host destination and its completion events."""

    value: Any
    events: tuple[Any, ...]
    sequence: int
    _owner: Any = field(repr=False, compare=False)
    _completed: bool = field(default=False, repr=False, compare=False)


class OutputReader:
    """Own pending output-read destinations/events until completion or drain.

    `DecodeRuntime.consume` uses `read` for blocking output. vLLM's
    asynchronous path uses `submit`, returns the retained host value and
    events to its scheduler, and later calls `complete`. Executor cleanup
    calls `drain` to retire anything the scheduler did not complete.
    """

    def __init__(self, mesh_device: Any):
        self.mesh_device = mesh_device
        self._sequences = itertools.count()
        self._pending: dict[int, PendingRead] = {}
        self._pending_by_value_id: dict[int, int] = {}
        self._owner = object()
        self._lock = threading.Lock()

    # Public API

    def read(self, value: Any, *, blocking: bool = True) -> Any | PendingRead:
        """Read a nested output payload to host.

        Blocking calls return the completed host payload directly. Async calls
        return a ``PendingRead`` whose ``value`` and ``events`` can be adapted to
        the existing vLLM tuple contract while this reader retains ownership.
        """

        return self._read(value, blocking=blocking)

    def submit(self, value: Any) -> PendingRead:
        """Submit an asynchronous read and retain its resources."""

        result = self._read(value, blocking=False)
        assert isinstance(result, PendingRead)
        return result

    def read_synchronized(self, value: Any) -> Any:
        """Submit nested host copies and synchronize the device once.

        This is the canonical readback path and it encodes a correctness
        invariant, not just a performance choice:

            ``ttnn.execute_trace(..., blocking=True)`` fences COMPUTE, but it does
            NOT fence the device->host readback DMA that ``Tensor.cpu`` /
            ``ttnn.to_torch`` start. Consuming the host buffer, or letting the
            next ``execute_trace`` overwrite the source tensor, before that DMA
            drains corrupts the host heap. On multi-host runs the failure is
            rank-asymmetric and invisible to CPU memory checkers (it is a device
            DMA, not a CPU instruction).

        The discipline here is what makes it safe: submit every host copy
        non-blocking, retain each destination, then issue a single
        ``ttnn.synchronize_device`` that drains all of them before returning. A
        caller must not read the returned buffers before this method returns.

        Per-step host round-trips should go through this method (or an equivalent
        drain) rather than a bare non-blocking ``cpu()``. Where a model allows
        it, prefer keeping state on-device and gathering once over reading back
        every step. For a cross-host step that additionally needs an MPI barrier,
        see :func:`tt_transformers.distributed.cross_host_rendezvous`.
        """

        retained_destinations: list[Any] = []
        try:
            host_value, submitted = _read_to_host(
                value,
                blocking=False,
                retained_destinations=retained_destinations,
            )
            if submitted:
                ttnn.synchronize_device(self.mesh_device)
        except BaseException:
            _synchronize_after_failed_submission(self.mesh_device)
            raise
        return host_value

    def complete(self, pending_or_value: PendingRead | Any) -> Any:
        """Synchronize and retire a pending read, returning its host payload.

        Passing the exact unwrapped ``PendingRead.value`` supports compatibility
        facades that return ``(host_value, events)`` to external schedulers.
        Completion is idempotent.
        """

        return self._complete_pending(pending_or_value)

    def drain(self) -> None:
        """Complete every pending read; repeated drains are no-ops."""

        with self._lock:
            pending_reads = tuple(self._pending.values())
        failures = []
        for pending in pending_reads:
            try:
                self._complete_pending(pending)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError(f"Failed to drain {len(failures)} pending output read(s)") from failures[0]

    # Private implementation

    def _read(self, value: Any, *, blocking: bool) -> Any | PendingRead:
        retained_destinations: list[Any] = []
        try:
            host_value, submitted = _read_to_host(
                value,
                blocking=blocking,
                retained_destinations=retained_destinations,
            )
        except BaseException:
            if not blocking:
                _synchronize_after_failed_submission(self.mesh_device)
            raise
        if blocking:
            return host_value

        sequence = next(self._sequences)
        if not submitted:
            return PendingRead(value=host_value, events=(), sequence=sequence, _owner=self._owner, _completed=True)

        record_event = getattr(ttnn, "record_event", None)
        event_synchronize = getattr(ttnn, "event_synchronize", None)
        if not callable(record_event) or not callable(event_synchronize):
            _synchronize_after_failed_submission(self.mesh_device)
            raise RuntimeError("Asynchronous output reads require ttnn.record_event and ttnn.event_synchronize")
        try:
            event = record_event(self.mesh_device, 0)
        except BaseException:
            _synchronize_after_failed_submission(self.mesh_device)
            raise

        pending = PendingRead(value=host_value, events=(event,), sequence=sequence, _owner=self._owner)
        with self._lock:
            self._pending[sequence] = pending
            self._pending_by_value_id[id(host_value)] = sequence
        return pending

    def _complete_pending(self, pending_or_value: PendingRead | Any) -> Any:
        if isinstance(pending_or_value, PendingRead):
            candidate = pending_or_value
            if candidate._owner is not self._owner:
                raise ValueError("PendingRead is not owned by this OutputReader")
            sequence = candidate.sequence
        else:
            with self._lock:
                sequence = self._pending_by_value_id.get(id(pending_or_value))
            if sequence is None:
                return pending_or_value
            candidate = None

        with self._lock:
            pending = self._pending.get(sequence)
        if pending is None:
            if candidate is not None:
                if candidate._completed:
                    return candidate.value
                raise ValueError("PendingRead is not active in this OutputReader")
            return pending_or_value
        if candidate is not None and pending is not candidate:
            raise ValueError("PendingRead is not owned by this OutputReader")

        synchronize = getattr(ttnn, "event_synchronize", None)
        if pending.events and not callable(synchronize):
            raise RuntimeError("Pending output reads require ttnn.event_synchronize")
        for event in pending.events:
            synchronize(event)

        with self._lock:
            current = self._pending.pop(sequence, None)
            if current is not None:
                self._pending_by_value_id.pop(id(current.value), None)
                object.__setattr__(current, "_completed", True)
        return pending.value


def _read_to_host(
    value: Any,
    *,
    blocking: bool,
    retained_destinations: list[Any],
) -> tuple[Any, bool]:
    if value is None:
        return None, False
    if isinstance(value, tuple):
        converted = [
            _read_to_host(item, blocking=blocking, retained_destinations=retained_destinations) for item in value
        ]
        return tuple(item for item, _ in converted), any(submitted for _, submitted in converted)
    if isinstance(value, list):
        converted = [
            _read_to_host(item, blocking=blocking, retained_destinations=retained_destinations) for item in value
        ]
        return [item for item, _ in converted], any(submitted for _, submitted in converted)
    if isinstance(value, dict):
        converted = {
            key: _read_to_host(item, blocking=blocking, retained_destinations=retained_destinations)
            for key, item in value.items()
        }
        return (
            {key: item for key, (item, _) in converted.items()},
            any(submitted for _, submitted in converted.values()),
        )
    if isinstance(value, torch.Tensor):
        return value.cpu(), False
    if isinstance(value, ttnn.Tensor):
        if value.storage_type() == ttnn.StorageType.HOST:
            return value, False
        host_value = value.cpu(blocking=blocking)
        if not blocking:
            retained_destinations.append(host_value)
        return host_value, not blocking

    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        try:
            host_value = cpu(blocking=blocking)
        except TypeError:
            return cpu(), False
        if not blocking:
            retained_destinations.append(host_value)
        return host_value, not blocking
    return value, False


def _synchronize_after_failed_submission(mesh_device: Any) -> None:
    synchronize = getattr(ttnn, "synchronize_device", None)
    if callable(synchronize):
        synchronize(mesh_device)
