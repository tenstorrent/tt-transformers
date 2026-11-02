# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RowsTopK (ops/sampling/rows_topk.py): its kernels ship with the package, and on device it returns the exact
top-32 values and indices of the live rows, sorted by value then lower index, rows >= B copies of row 0; widths above
65536 return UINT32 indices."""

import re
from pathlib import Path

import pytest
import torch
import ttnn

from tt_transformers.ops.sampling import rows_topk
from tt_transformers.ops.sampling.rows_topk import RowsTopK

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.host
def test_named_kernels_ship_and_are_covered_by_package_data():
    names = set(re.findall(r'"([A-Za-z0-9_]+\.cpp)"', Path(rows_topk.__file__).read_text()))
    assert names == {"rows_topk_scan.cpp", "rows_topk_merge.cpp"}
    globs = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"][
        "tt_transformers.ops.sampling"
    ]
    package_root = Path(rows_topk.__file__).parent
    packaged = {path for glob in globs for path in package_root.glob(glob)}
    for name in names:
        kernel = rows_topk.KERNEL_DIR / name
        assert kernel.is_file(), name
        assert kernel in packaged, name
        for include in re.findall(r'#include "([^"]+)"', kernel.read_text()):
            assert include.startswith("api/"), f"{name}: {include}"


K = 32


def _reference_order(row, k):
    return sorted(range(row.numel()), key=lambda c: (-row[c].item(), c))[:k]


@pytest.mark.device
@pytest.mark.blackhole
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [pytest.param({"mesh_shape": (1, 1), "trace_region_size": 32 << 20}, id="p150-1x1")],
    indirect=True,
)
@pytest.mark.parametrize("width", [62080, 124160], ids=["w62080", "w124160_uint32"])
@pytest.mark.parametrize("rows", [1, 2, 8])
def test_rows_topk_is_exact(ttnn_mesh_device, width, rows):
    torch.manual_seed(0)
    op = RowsTopK(ttnn_mesh_device, width, K)
    lg = (torch.randn(1, 1, 32, width) * 3).bfloat16()
    lg[0, 0, 0, 1000] = lg[0, 0, 0, 2000] = 50.0  # an exact tie at the top: lower index first
    lg[0, 0, 1, 3] = lg[0, 0, 1, width - 7] = 45.0  # a tie across the first and last scan cores
    x = ttnn.from_torch(
        lg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=ttnn_mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v, i = op(x, rows)
    assert i.dtype == (ttnn.uint32 if width > 65536 else ttnn.uint16)
    gv = ttnn.to_torch(v).float().reshape(32, K)
    gi = ttnn.to_torch(i).long().reshape(32, K) & 0xFFFFFFFF
    for r in range(32):
        src = 0 if r >= rows else r
        row = lg[0, 0, src].float()
        if r < 2:  # the rows with planted ties: exact order
            order = _reference_order(row, K)
            assert torch.equal(gi[r], torch.tensor(order)), (r, gi[r][:8].tolist(), order[:8])
            assert torch.equal(gv[r], row[order]), r
        else:
            assert torch.equal(gv[r], torch.sort(row, descending=True).values[:K]), r
