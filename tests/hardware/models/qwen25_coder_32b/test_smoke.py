# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware smoke gates delegated to the pytest-free public example module."""

import os

import pytest
from examples.qwen25_coder_32b import smoke as example
from tests.support.marker_policy import selected_topology_marks


@pytest.fixture
def device_params(request, galaxy_type):
    return example.resolve_device_params(request, galaxy_type)


@pytest.fixture(scope="module")
def hf_model_id():
    return example.default_hf_model_id()


pytestmark = [
    pytest.mark.parametrize("mesh_device", [{"T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 8))], indirect=True),
    pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True),
]

pytestmark.extend(selected_topology_marks(pytest))

_slow = pytest.mark.slow


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@_slow
@pytest.mark.parametrize("seq_len", [128])
def test_qwen25_coder_32b_prefill_smoke(mesh_device, hf_model_id, seq_len, tmp_path_factory):
    try:
        example.run_qwen25_coder_32b_prefill_smoke(
            mesh_device=mesh_device, hf_model_id=hf_model_id, seq_len=seq_len, tmp_path_factory=tmp_path_factory
        )
    except example.UnsupportedConfiguration as error:
        pytest.skip(str(error))


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@_slow
def test_qwen25_coder_32b_decode_one_step(mesh_device, hf_model_id, tmp_path_factory):
    try:
        example.run_qwen25_coder_32b_decode_one_step(
            mesh_device=mesh_device, hf_model_id=hf_model_id, tmp_path_factory=tmp_path_factory
        )
    except example.UnsupportedConfiguration as error:
        pytest.skip(str(error))


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@_slow
@pytest.mark.parametrize("seq_len", [128])
def test_qwen25_coder_32b_executor_prefill_smoke(mesh_device, hf_model_id, seq_len, tmp_path_factory):
    try:
        example.run_qwen25_coder_32b_executor_prefill_smoke(
            mesh_device=mesh_device, hf_model_id=hf_model_id, seq_len=seq_len, tmp_path_factory=tmp_path_factory
        )
    except example.UnsupportedConfiguration as error:
        pytest.skip(str(error))


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@_slow
def test_qwen25_coder_32b_teacher_forcing_prefill_vs_hf(mesh_device, hf_model_id, tmp_path_factory):
    try:
        example.run_qwen25_coder_32b_teacher_forcing_prefill_vs_hf(
            mesh_device=mesh_device, hf_model_id=hf_model_id, tmp_path_factory=tmp_path_factory
        )
    except example.UnsupportedConfiguration as error:
        pytest.skip(str(error))


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@_slow
def test_qwen25_coder_32b_eager_traced_prefill_logits_match(mesh_device, hf_model_id, tmp_path_factory):
    try:
        example.run_qwen25_coder_32b_eager_traced_prefill_logits_match(
            mesh_device=mesh_device, hf_model_id=hf_model_id, tmp_path_factory=tmp_path_factory
        )
    except example.UnsupportedConfiguration as error:
        pytest.skip(str(error))


@pytest.mark.device
@pytest.mark.model
@pytest.mark.slow
@_slow
def test_qwen25_coder_32b_numerical_divergence_vs_hf(mesh_device, hf_model_id, tmp_path_factory):
    try:
        example.run_qwen25_coder_32b_numerical_divergence_vs_hf(
            mesh_device=mesh_device, hf_model_id=hf_model_id, tmp_path_factory=tmp_path_factory
        )
    except example.UnsupportedConfiguration as error:
        pytest.skip(str(error))
