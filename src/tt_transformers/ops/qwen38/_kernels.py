# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resolve package-owned device kernel paths without external source trees."""

from pathlib import Path

KERNEL_ROOT = Path(__file__).resolve().parent / "kernels"


def kernel_path(operation: str, role: str, name: str) -> str:
    path = KERNEL_ROOT / operation / role / name
    if not path.is_file():
        raise FileNotFoundError(f"missing bundled {operation} kernel: {path}")
    return str(path)


def include_path(operation: str, role: str) -> str:
    path = KERNEL_ROOT / operation / role
    if not path.is_dir():
        raise FileNotFoundError(f"missing bundled {operation} include directory: {path}")
    return str(path)
