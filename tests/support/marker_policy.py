"""Collection-safe topology marks derived only from explicit MESH_DEVICE."""

from __future__ import annotations

import os


def selected_topology_marks(pytest_module):
    """Return architecture/SKU marks only when the requested geometry proves them."""

    selected = os.environ.get("MESH_DEVICE", "").strip().upper()
    mapping = {
        "N150": ("wormhole", "n150"),
        "N300": ("wormhole", "n300"),
        "T3K": ("wormhole", "t3k"),
        "P150": ("blackhole", "p150"),
        "P300": ("blackhole", "p300"),
        "P150X4": ("blackhole", "p150x4"),
        "TG": ("wormhole", "galaxy"),
    }
    return [getattr(pytest_module.mark, name) for name in mapping.get(selected, ())]
