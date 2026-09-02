"""Hardware gates delegated to the pytest-free public example module."""

import os

import pytest

from tests.support.marker_policy import selected_topology_marks

from examples.llama3_8b import demo as example

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
@pytest.mark.parametrize('test_config', [pytest.param('token-accuracy', id='token-accuracy-repeat_batch-1-prefetcher-off'), 'batch-1', pytest.param('batch-32', id='batch-32-repeat_batch-1-prefetcher-off'), 'batch-32-ci', pytest.param('eval-32-repeat-3', id='eval-32-repeat_batch-3-prefetcher-off-perf-report-off'), pytest.param('eval-32-repeat-1', id='eval-32-repeat_batch-1-prefetcher-off-perf-report-on'), 'ci-b1-DP-2', 'ci-b1-DP-4', 'ci-b1-DP-8', 'ci-b1-DP-16', 'ci-b1-DP-32'])
@pytest.mark.parametrize('optimizations', ['performance', 'accuracy'])
@pytest.mark.usefixtures('silicon_arch_name')
def test_llama3_8b(test_config, ttnn_mesh_device, optimizations):
    try:
        example.run_llama3_8b(test_config=test_config, ttnn_mesh_device=ttnn_mesh_device, optimizations=optimizations)
    except example.UnsupportedConfiguration as error:
        pytest.skip(str(error))

@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize('optimizations', ['performance', 'accuracy'])
@pytest.mark.usefixtures('silicon_arch_name')
def test_llama3_8b_bh_seeded_cross_cardinality(ttnn_mesh_device, optimizations):
    try:
        example.run_llama3_8b_bh_seeded_cross_cardinality(ttnn_mesh_device=ttnn_mesh_device, optimizations=optimizations)
    except example.UnsupportedConfiguration as error:
        pytest.skip(str(error))
