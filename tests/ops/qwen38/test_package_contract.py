# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import ttnn

from tt_transformers.ops.qwen38 import _legacy
from tt_transformers.ops.qwen38._kernels import KERNEL_ROOT, kernel_path
from tt_transformers.ops.qwen38.gdn_decode_step import _FUSED, _PLAIN
from tt_transformers.ops.qwen38.gdn_spec_step import _SPEC
from tt_transformers.ops.qwen38.qkv_causal_conv1d_silu_tile import _DFBS


def test_all_required_kernel_sources_are_wheel_owned():
    expected = {
        "gdn_decode_step/dataflow/reader_gdn_decode_step.cpp",
        "gdn_decode_step/dataflow/writer_gdn_decode_step.cpp",
        "gdn_decode_step/dataflow/reader_gdn_decode_step_conv.cpp",
        "gdn_decode_step/dataflow/writer_gdn_decode_step_conv.cpp",
        "gdn_decode_step/compute/gdn_decode_step.cpp",
        "gdn_decode_step/compute/gdn_decode_step_conv.cpp",
        "gdn_spec_step/dataflow/reader_gdn_spec_step.cpp",
        "gdn_spec_step/dataflow/writer_gdn_spec_step.cpp",
        "gdn_spec_step/compute/gdn_spec_step.cpp",
        "qkv_causal_conv1d_silu/dataflow/reader_qkv_causal_conv1d_silu_tile.cpp",
        "qkv_causal_conv1d_silu/dataflow/writer_qkv_causal_conv1d_silu.cpp",
        "qkv_causal_conv1d_silu/compute/qkv_causal_conv1d_silu_tile.cpp",
    }
    present = {str(path.relative_to(KERNEL_ROOT)) for path in KERNEL_ROOT.rglob("*.cpp")}
    assert present == expected
    for relative in expected:
        operation, role, name = relative.split("/", 2)
        assert Path(kernel_path(operation, role, name)).is_file()


@pytest.mark.parametrize(
    ("table", "expected_count"),
    [(_PLAIN, 29), (_FUSED, 42), (_SPEC, 52)],
)
def test_dfb_slots_are_dense_and_stable(table, expected_count):
    slots = sorted(slot for slot, _, _ in table.values())
    assert slots == list(range(expected_count))


def test_tile_kda_dfb_slots_match_the_v1_generated_program():
    assert _DFBS == {
        "act_tile": 0,
        "shift": 1,
        "shifted": 2,
        "weights": 3,
        "partial": 4,
        "output": 5,
    }


@pytest.mark.parametrize(
    ("operation", "role", "filename", "template_count", "entry", "runtime_arity", "pragma"),
    [
        ("gdn_decode_step", "dataflow", "reader_gdn_decode_step.cpp", 8, "reader", 2, None),
        ("gdn_decode_step", "compute", "gdn_decode_step_conv.cpp", 4, "compute", 1, "O2"),
        ("gdn_spec_step", "dataflow", "reader_gdn_spec_step.cpp", 18, "reader", 2, "Os"),
        ("gdn_spec_step", "dataflow", "writer_gdn_spec_step.cpp", 11, "writer", 2, "Os"),
        ("gdn_spec_step", "compute", "gdn_spec_step.cpp", 6, "compute", 1, None),
        (
            "qkv_causal_conv1d_silu",
            "dataflow",
            "reader_qkv_causal_conv1d_silu_tile.cpp",
            2,
            "reader",
            2,
            None,
        ),
        (
            "qkv_causal_conv1d_silu",
            "compute",
            "qkv_causal_conv1d_silu_tile.cpp",
            2,
            "compute",
            2,
            None,
        ),
    ],
)
def test_legacy_source_has_one_entry_and_explicit_bindings(
    operation, role, filename, template_count, entry, runtime_arity, pragma
):
    source = _legacy.legacy_source(
        operation=operation,
        role=role,
        filename=filename,
        dfb_ids={"first": 0, "last": 51},
        tensor_tokens=[("input", template_count, 0), ("output", template_count + 4, 16)],
        template_count=template_count,
        entry_name=entry,
        runtime_arity=runtime_arity,
        opt_pragma=pragma,
    )
    assert "TT_KERNEL" not in source
    assert source.count("kernel_main") == 1
    assert "DFBBindingToken first{0u}" in source
    assert "DFBBindingToken last{51u}" in source
    assert f"TensorBindingToken<{template_count}u, 0u>" in source
    assert f"TensorBindingToken<{template_count + 4}u, 16u>" in source
    assert source.count("get_compile_time_arg_val(") == template_count
    assert source.count("get_arg_val<uint32_t>(") == runtime_arity
    assert "#pragma GCC optimize" not in source


def test_required_build_levels_are_exposed_by_ttnn():
    levels = ttnn.KernelDescriptor.BuildOptLevel
    assert levels.O2.name == "O2"
    assert levels.Os.name == "Os"


def test_legacy_source_caches_only_normalized_source_shape(monkeypatch):
    _legacy._legacy_source_cached.cache_clear()
    reads = 0
    original_read_text = Path.read_text

    def counted_read_text(path, *args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read_text)
    kwargs = dict(
        operation="qkv_causal_conv1d_silu",
        role="compute",
        filename="qkv_causal_conv1d_silu_tile.cpp",
        dfb_ids={"first": 0, "last": 5},
        tensor_tokens=[],
        template_count=2,
        entry_name="compute",
        runtime_arity=2,
    )
    first = _legacy.legacy_source(**kwargs)
    second = _legacy.legacy_source(**{**kwargs, "dfb_ids": dict(kwargs["dfb_ids"])})
    changed = _legacy.legacy_source(**{**kwargs, "runtime_arity": 1})

    assert first is second
    assert first != changed
    assert reads == 2
    assert _legacy._legacy_source_cached.cache_info().hits == 1


def test_stock_api_cannot_express_required_kernel_build_levels():
    # This is the release-significant stock-0.79 limitation.  Removing this
    # assertion requires replacing the source-pragmas with the real enum and
    # re-running the native parity/performance matrix.
    import ttnn

    assert not hasattr(ttnn, "KernelBuildOptLevel")
    assert not hasattr(ttnn._ttnn, "KernelBuildOptLevel")


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_scalar_validation_rejects_invalid_epsilon(bad):
    with pytest.raises(ValueError):
        _legacy.validate_scalars(None, 128, bad, 1e-6)
