# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone TTNN transformer building blocks, runtimes, and models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tt-transformers")
except PackageNotFoundError:
    __version__ = "2.0.0.dev0"

__all__ = ["__version__"]
