"""Hardware gates delegated to the pytest-free public example module."""

import os

import pytest
from examples.deepseek_r1_distill_qwen_14b import benchmark as example
from tests.support.marker_policy import selected_topology_marks

try:
    _MESH_PARAM = example.ttnn_mesh_device_param_from_env()
except example.UnsupportedConfiguration as error:
    pytest.skip(str(error), allow_module_level=True)

pytestmark = pytest.mark.parametrize(
    "ttnn_mesh_device", [_MESH_PARAM], indirect=True, ids=[os.environ.get("MESH_DEVICE", "mesh").strip() or "mesh"]
)

pytestmark = [pytestmark, *selected_topology_marks(pytest)]


@pytest.fixture(scope="module")
def mesh_device(ttnn_mesh_device):
    return ttnn_mesh_device


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param("token-accuracy", id="token-accuracy"),
        pytest.param("batch-1", id="batch-1"),
        pytest.param("batch-32", id="batch-32"),
        pytest.param("batch-32-ci", id="batch-32-ci"),
        pytest.param("eval-32", id="eval-32"),
        pytest.param("ci-b1-DP-2", id="ci-b1-DP-2"),
        pytest.param("ci-b1-DP-4", id="ci-b1-DP-4"),
        pytest.param("ci-b1-DP-8", id="ci-b1-DP-8"),
        pytest.param("ci-b1-DP-16", id="ci-b1-DP-16"),
        pytest.param("ci-b1-DP-32", id="ci-b1-DP-32"),
    ],
)
@pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
def test_deepseek_r1_qwen_14b(test_config, mesh_device, optimizations):
    try:
        example.run_deepseek_r1_qwen_14b(test_config=test_config, mesh_device=mesh_device, optimizations=optimizations)
    except example.UnsupportedConfiguration as error:
        pytest.skip(str(error))
