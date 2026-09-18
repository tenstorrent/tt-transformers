# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-request sampling values for the Galaxy `(8, 4)` 2D tensor models.

This dataclass was defined inside `GalaxyDirectRunner`'s module in tt-metal.
That runner was condemned by the 2026-09-02 operator decision and is not part of
this package, but the policy type it happened to host has no coupling to it: both
models' demos and several device suites pass one of these around. It is lifted
here verbatim so those call sites survive the runner's absence.

`GalaxyDirectGeneration`, the runner's other public dataclass, was deliberately
*not* lifted — nothing outside the runner and its own tests used it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GalaxySamplingPolicy"]


@dataclass(frozen=True)
class GalaxySamplingPolicy:
    """Per-request sampling values. ``temperature == 0`` means greedy."""

    top_k: int = 1
    top_p: float = 1.0
    temperature: float = 0.0
    seed: int | None = None
    on_device: bool = False

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0 or self.top_k == 1
