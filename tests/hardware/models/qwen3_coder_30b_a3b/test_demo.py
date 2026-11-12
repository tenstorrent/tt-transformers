# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hardware gates delegated to the pytest-free public example module."""

import pytest
from examples.qwen3_coder_30b_a3b import benchmark as example
from tests.support.marker_policy import selected_topology_marks

try:
    _MESH_PARAM = example.mesh_parameters()
except example.UnsupportedConfiguration as error:
    pytest.skip(str(error), allow_module_level=True)

pytestmark = [
    pytest.mark.parametrize("ttnn_mesh_device", [_MESH_PARAM], indirect=True, ids=["P150x4"]),
    *selected_topology_marks(pytest),
]


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.blackhole
@pytest.mark.p150x4
@pytest.mark.parametrize("case", example.CASES)
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
def test_qwen3_coder_30b_a3b(case, ttnn_mesh_device, require_blackhole_mesh_device, optimizations):
    example.run_case(case, ttnn_mesh_device, optimizations)
