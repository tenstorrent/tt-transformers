"""Host-side ownership helpers for runnable examples."""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import tempfile
from pathlib import Path

import ttnn


class UnsupportedConfiguration(RuntimeError):
    """A requested example geometry or asset is deliberately unsupported."""


class TemporaryPathFactory:
    """Small runtime counterpart of pytest's tmp_path_factory."""

    def __init__(self):
        self._root = Path(tempfile.mkdtemp(prefix="tt_transformers_example_"))

    def mktemp(self, name: str) -> Path:
        path = self._root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)


@contextlib.contextmanager
def _device_lock():
    lock_path = Path(os.environ.get("TT_DEVICE_LOCK_PATH", "/tmp/tt_device.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _parent_shape(system_shape: tuple[int, int], requested: tuple[int, int]) -> tuple[int, int]:
    if requested == (1, 1):
        return requested
    if requested[0] * requested[1] == system_shape[0] * system_shape[1]:
        return requested
    if requested[0] <= system_shape[0] and requested[1] <= system_shape[1]:
        return system_shape
    rotated = (system_shape[1], system_shape[0])
    if requested[0] <= rotated[0] and requested[1] <= rotated[1]:
        return rotated
    raise UnsupportedConfiguration(f"requested mesh {requested} does not fit system mesh {system_shape}")


@contextlib.contextmanager
def open_mesh_device(parameters: dict):
    """Open one example mesh with the same full-parent/submesh ownership rule as tests."""

    params = dict(parameters)
    requested = tuple(params.pop("mesh_shape"))
    fabric = params.pop("fabric_config", None)
    parent = None
    submesh = None
    with _device_lock():
        try:
            system = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
            parent_shape = _parent_shape(system, requested)
            if fabric is not None:
                ttnn.set_fabric_config(
                    fabric,
                    ttnn.FabricReliabilityMode.STRICT_INIT,
                    None,
                    ttnn.FabricTensixConfig.DISABLED,
                )
            parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(parent_shape), **params)
            if requested != parent_shape:
                submesh = parent.create_submesh(ttnn.MeshShape(requested))
                yield submesh
            else:
                yield parent
        finally:
            if submesh is not None:
                ttnn.close_mesh_device(submesh)
            if parent is not None:
                ttnn.close_mesh_device(parent)
            if fabric is not None:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
