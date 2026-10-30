# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Rank-aware helpers for single- vs multi-host (cross-box) mesh execution.

These primitives are opt-in and self-contained. Nothing in the existing
single-host code paths calls them yet (call-site adoption is a follow-up), so
importing this module and leaving it unused keeps single-host behaviour
byte-for-byte unchanged.

The central rule this module encodes:

    Single-host vs. multi-host must branch on the host RANK COUNT, never on the
    mesh shape.

A ``1x4`` mesh may be one host owning four devices, or two hosts owning two
devices each; the shape alone cannot tell them apart. The authoritative source
is the distributed-context / MPI world size. :func:`num_host_ranks` reads that
world size (never the mesh geometry); :func:`is_multi_rank` is the path
selector other code should branch on.

The cross-host correctness rule this module encodes:

    ``ttnn.execute_trace(..., blocking=True)`` fences COMPUTE, but not the
    device->host readback DMA started by ``ttnn.to_torch`` / ``Tensor.cpu``.

A rank that consumes the host buffer, or lets the next ``execute_trace``
overwrite the source tensor, before that readback DMA drains corrupts its host
heap. The corruption is rank-asymmetric (the send-first rank trips; recv-first
is incidentally safe) and invisible to CPU memory checkers because it is a
device DMA, not a CPU instruction. The fix is to drain with
``ttnn.synchronize_device(mesh)`` immediately after the per-step readback (or to
keep state on-device and gather once). :func:`cross_host_rendezvous` performs
that drain and, only when multi-rank, the cross-host MPI barrier.
"""

from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

import ttnn
from loguru import logger

T = TypeVar("T")

# OpenMPI (the launcher `tt-run` uses) publishes the world size and this rank's
# index through these environment variables. They are the fallback when a ttnn
# distributed context has not been initialized in this process.
_MPI_WORLD_SIZE_ENV = "OMPI_COMM_WORLD_SIZE"
_MPI_WORLD_RANK_ENV = "OMPI_COMM_WORLD_RANK"

# Substrings seen in the transient failure where the cross-host fabric link does
# not finish training at mesh-open and the router sync times out. Matching is
# case-insensitive. A genuine, non-transient open failure will not match these
# and is re-raised immediately.
_FABRIC_OPEN_RETRY_SUBSTRINGS = (
    "fabric router",
    "router sync",
    "fabric sync",
    "sync timeout",
    "routing table",
    "fabric init",
)


def _ttnn_context_world_size() -> int | None:
    """Return the distributed-context world size, or ``None`` when unavailable.

    Uses ttnn's distributed context only when it is both present (a multihost
    build) and initialized. Single-host ttnn wheels do not expose these symbols,
    so the lookups are defensive ``getattr`` probes rather than hard references.
    """

    is_initialized = getattr(ttnn, "distributed_context_is_initialized", None)
    world_size = getattr(ttnn, "distributed_context_world_size", None) or getattr(
        ttnn, "distributed_context_get_size", None
    )
    if not callable(is_initialized) or not callable(world_size):
        return None
    try:
        if not is_initialized():
            return None
        return int(world_size())
    except Exception:  # pragma: no cover - defensive against backend variance
        return None


def _ttnn_context_world_rank() -> int | None:
    """Return this process's distributed-context rank, or ``None`` when unavailable."""

    is_initialized = getattr(ttnn, "distributed_context_is_initialized", None)
    world_rank = getattr(ttnn, "distributed_context_world_rank", None) or getattr(
        ttnn, "distributed_context_get_rank", None
    )
    if not callable(is_initialized) or not callable(world_rank):
        return None
    try:
        if not is_initialized():
            return None
        return int(world_rank())
    except Exception:  # pragma: no cover - defensive against backend variance
        return None


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def num_host_ranks() -> int:
    """Return the number of host ranks (processes) cooperating on this run.

    This is the authoritative single-host vs. multi-host signal. It is derived
    from the distributed-context / MPI world size and MUST NOT be inferred from
    the mesh shape: a ``1x4`` mesh can be one host with four devices or two
    hosts with two devices each, and only the world size distinguishes them.

    Resolution order:

    1. The ttnn distributed context world size, when a context is initialized.
    2. The ``OMPI_COMM_WORLD_SIZE`` environment variable set by the launcher.
    3. ``1`` (a plain, single-process run).

    A value of ``1`` means single-host; anything greater means cross-host.
    """

    context_size = _ttnn_context_world_size()
    if context_size is not None and context_size >= 1:
        return context_size
    env_size = _env_int(_MPI_WORLD_SIZE_ENV)
    if env_size is not None and env_size >= 1:
        return env_size
    return 1


def host_rank() -> int:
    """Return this process's host-rank index in ``[0, num_host_ranks())``.

    Resolved from the ttnn distributed context, then ``OMPI_COMM_WORLD_RANK``,
    then ``0`` for a plain single-process run.
    """

    context_rank = _ttnn_context_world_rank()
    if context_rank is not None and context_rank >= 0:
        return context_rank
    env_rank = _env_int(_MPI_WORLD_RANK_ENV)
    if env_rank is not None and env_rank >= 0:
        return env_rank
    return 0


def is_multi_rank() -> bool:
    """Return whether more than one host rank is cooperating on this run.

    This is the correct predicate for selecting cross-host code paths. Do not
    branch on ``mesh_device.shape``; see :func:`num_host_ranks`.
    """

    return num_host_ranks() > 1


def cross_host_rendezvous(mesh_device) -> None:
    """Drain the per-step readback DMA, then barrier across hosts when multi-rank.

    Call this once per traced step, immediately after reading this step's
    output(s) back to host. It enforces the readback-DMA-drain invariant that
    ``ttnn.execute_trace`` does not provide on its own:

    * ``ttnn.synchronize_device(mesh_device)`` blocks until the device->host
      readback DMA has fully drained, so neither this rank's consumption of the
      host buffer nor the next ``execute_trace`` can race the in-flight copy.
    * When (and only when) the run spans multiple hosts, it then issues
      ``ttnn.distributed_context_barrier()`` so no rank races ahead into the
      next step before every peer has drained its own readback.

    In a single-host run this is exactly one ``synchronize_device`` and no
    barrier, matching the discipline already used by
    :meth:`tt_transformers.llm_runtime.output_reader.OutputReader.read_synchronized`.
    Prefer keeping state on-device and gathering once over a per-step host
    round-trip where the model allows it; this helper is for the cases that must
    read back every step.
    """

    ttnn.synchronize_device(mesh_device)
    if not is_multi_rank():
        return
    barrier = getattr(ttnn, "distributed_context_barrier", None)
    if not callable(barrier):
        raise RuntimeError(
            "Cross-host rendezvous requires ttnn.distributed_context_barrier, which this "
            "ttnn build does not provide. Multi-host runs need a distributed (multihost) "
            "ttnn build; see docs/ttnn-development.md."
        )
    barrier()


def _is_transient_fabric_open_error(error: BaseException) -> bool:
    """Return whether ``error`` looks like a retriable fabric-router-sync timeout."""

    message = str(error).lower()
    return any(substring in message for substring in _FABRIC_OPEN_RETRY_SUBSTRINGS)


def open_mesh_device_with_retry(
    open_callable: Callable[[], T],
    *,
    attempts: int = 5,
    backoff_seconds: float = 2.0,
) -> T:
    """Call ``open_callable`` to open a mesh, retrying transient fabric sync timeouts.

    The cross-host fabric link intermittently fails to finish training when the
    mesh is first opened, surfacing as a fabric-router-sync timeout. Those
    failures clear on a retry. This wrapper retries only errors that match that
    signature (see ``_FABRIC_OPEN_RETRY_SUBSTRINGS``); any other error is
    re-raised immediately.

    This is a no-op on the common path: a successful first open returns right
    away and never sleeps, so single-host behaviour is unchanged. ``open_callable``
    should fully open (and, on failure, fully tear down) the mesh so that each
    attempt starts from a clean state.
    """

    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return open_callable()
        except Exception as error:
            if attempt >= attempts or not _is_transient_fabric_open_error(error):
                raise
            last_error = error
            logger.warning(
                "Mesh open failed with a transient fabric-router-sync timeout "
                f"(attempt {attempt}/{attempts}); retrying in {backoff_seconds}s: {error}"
            )
            if backoff_seconds > 0:
                time.sleep(backoff_seconds)

    # Unreachable: the loop either returns or re-raises on the final attempt.
    assert last_error is not None
    raise last_error
