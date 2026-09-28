# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from tt_transformers.models.qwen38 import model_config
from tt_transformers.models.qwen38.gdn.tp import _resolve_v6_kda_tile_input


def test_native_kda_program_config_uses_the_public_ttnn_surface():
    import ttnn

    config = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256)
    assert config.channel_chunk_size == 256


def test_v6_defaults_kda_prefill_to_wheel_owned_tile_program():
    assert model_config._QWEN36_SERVING_OPT_DEFAULTS["QWEN36_KDA_TILE_IN"] == "1"


def test_v6_accepts_wheel_owned_tile_input(monkeypatch):
    monkeypatch.setenv("QWEN36_KDA_TILE_IN", "1")
    assert _resolve_v6_kda_tile_input() is True


def test_stock_ttnn_accepts_row_major_input(monkeypatch):
    monkeypatch.setenv("QWEN36_KDA_TILE_IN", "0")
    assert _resolve_v6_kda_tile_input() is False


def test_stock_ttnn_rejects_unknown_layout_selector(monkeypatch):
    import pytest

    monkeypatch.setenv("QWEN36_KDA_TILE_IN", "maybe")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _resolve_v6_kda_tile_input()
