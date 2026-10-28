# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tt_transformers.models.qwen38 import model_config
from tt_transformers.models.qwen38.gdn.tp import _resolve_kda_tile_input


@pytest.mark.host
@pytest.mark.model
@pytest.mark.skipif(
    not hasattr(ttnn, "QkvCausalConv1dSiluProgramConfig"),
    reason="requires ttnn.QkvCausalConv1dSiluProgramConfig",
)
def test_native_kda_program_config_uses_the_public_ttnn_surface():
    config = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256)
    assert config.channel_chunk_size == 256


@pytest.mark.host
@pytest.mark.model
def test_defaults_kda_prefill_to_wheel_owned_tile_program():
    assert model_config.Qwen38ServingConfig().environment()["QWEN36_KDA_TILE_IN"] == "1"


@pytest.mark.host
@pytest.mark.model
def test_serving_defaults_leave_process_wide_tt_flags_alone():
    assert not any(name.startswith("TT_") for name in model_config.Qwen38ServingConfig().environment())


@pytest.mark.host
@pytest.mark.model
def test_import_does_not_change_the_environment():
    import importlib
    import os

    before = dict(os.environ)
    importlib.reload(model_config)
    assert dict(os.environ) == before


@pytest.mark.host
@pytest.mark.model
def test_serving_defaults_apply_without_overriding(monkeypatch):
    for name in model_config.Qwen38ServingConfig().environment():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QWEN_SDPA_BF8", "0")
    model_config.Qwen38ServingConfig().apply_environment_defaults()
    import os

    assert os.environ["QWEN_SDPA_BF8"] == "0"
    assert os.environ["QWEN36_GDN_CONV"] == "kda"
    assert os.environ["DFLASH_WEIGHTS"] == "incoai/Qwen3.8-27B-DFlash2"


@pytest.mark.host
@pytest.mark.model
def test_accepts_wheel_owned_tile_input(monkeypatch):
    monkeypatch.setenv("QWEN36_KDA_TILE_IN", "1")
    assert _resolve_kda_tile_input() is True


@pytest.mark.host
@pytest.mark.model
def test_stock_ttnn_accepts_row_major_input(monkeypatch):
    monkeypatch.setenv("QWEN36_KDA_TILE_IN", "0")
    assert _resolve_kda_tile_input() is False


@pytest.mark.host
@pytest.mark.model
def test_stock_ttnn_rejects_unknown_layout_selector(monkeypatch):
    import pytest

    monkeypatch.setenv("QWEN36_KDA_TILE_IN", "maybe")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _resolve_kda_tile_input()


@pytest.mark.host
@pytest.mark.model
def test_default_decode_uses_the_composite_path_without_build_levels(monkeypatch):
    monkeypatch.delattr(ttnn.KernelDescriptor, "BuildOptLevel", raising=False)
    assert model_config.Qwen38ServingConfig().environment()["QWEN36_GDN_DECODE_FUSED"] == "0"


@pytest.mark.host
@pytest.mark.model
def test_default_decode_uses_the_fused_op_with_build_levels(monkeypatch):
    monkeypatch.setattr(ttnn.KernelDescriptor, "BuildOptLevel", object(), raising=False)
    assert model_config.Qwen38ServingConfig().environment()["QWEN36_GDN_DECODE_FUSED"] == "2"
