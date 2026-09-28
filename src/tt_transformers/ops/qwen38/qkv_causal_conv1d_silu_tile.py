# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TILE-input QKV causal convolution for the stock-TTNN v6 runtime.

Stock TTNN 0.79 ships the row-major KDA prefill operation, while Unified V1's
fork also accepts TILE input/history.  This descriptor owns only that missing
TILE program.  The three device kernels are copied byte-for-byte from the V1
oracle and launched through ``ttnn.generic_op``.
"""

from __future__ import annotations

from collections.abc import Sequence

import ttnn

from ._legacy import allocate_output, cb_descriptor, compute_descriptor, core_geometry, make_kernel, per_core_runtime

_DFBS = {
    "act_tile": 0,
    "shift": 1,
    "shifted": 2,
    "weights": 3,
    "partial": 4,
    "output": 5,
}


def _work_distribution(device, total: int):
    """Match the native KDA ``distribute_prep`` contiguous work split."""

    grid = device.compute_with_storage_grid_size()
    count = min(int(total), int(grid.x) * int(grid.y))
    core_ranges, coords = core_geometry(device, count)
    base, remainder = divmod(int(total), count)
    rows = []
    offset = 0
    for index in range(count):
        item_count = base + int(index < remainder)
        rows.append((offset, item_count))
        offset += item_count
    return core_ranges, coords, rows


def _require_tensor(tensor, name: str, *, layout, dtype, device=None) -> None:
    try:
        tensor.buffer_address()
    except Exception as error:
        raise ValueError(f"{name} must be an allocated device tensor") from error
    if tensor.layout != layout:
        raise ValueError(f"{name} must use {layout}, got {tensor.layout}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {tensor.dtype}")
    if device is not None and tensor.device() != device:
        raise ValueError(f"{name} must be on the same device as input")
    is_sharded = getattr(tensor, "is_sharded", None)
    if callable(is_sharded) and is_sharded():
        raise ValueError(f"{name} must use interleaved memory")


def qkv_causal_conv1d_silu_tile(
    input_tensor,
    history,
    taps: Sequence,
    q_width: int,
    k_width: int,
    v_width: int,
    *,
    channel_chunk_size: int,
    memory_config=None,
    compute_kernel_config=None,
):
    """Run the V1 TILE-input four-tap KDA prefill kernel on stock TTNN.

    Inputs are interleaved BF16 TILE tensors ``[1,T,Q+K+V]`` and
    ``[1,3,Q+K+V]``.  Four TILE BF16 tap tensors each contain one scalar per
    channel.  Outputs are three interleaved BF16 TILE tensors.
    """

    if len(taps) != 4:
        raise ValueError(f"exactly four convolution taps are required, got {len(taps)}")
    widths = (int(q_width), int(k_width), int(v_width))
    if any(width <= 0 or width % 32 for width in widths):
        raise ValueError(f"Q/K/V widths must be positive and tile aligned, got {widths}")
    channels = sum(widths)
    channel_chunk_size = int(channel_chunk_size)
    if channel_chunk_size <= 0 or channel_chunk_size % 32 or channels % channel_chunk_size:
        raise ValueError(
            "channel_chunk_size must be positive, tile aligned, and divide Q+K+V; "
            f"got chunk={channel_chunk_size}, channels={channels}"
        )

    device = input_tensor.device()
    _require_tensor(input_tensor, "input", layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    _require_tensor(history, "history", layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    for index, tap in enumerate(taps):
        _require_tensor(tap, f"tap{index}", layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    input_shape = tuple(int(value) for value in input_tensor.shape)
    history_shape = tuple(int(value) for value in history.shape)
    if len(input_shape) != 3 or input_shape[0] != 1 or input_shape[2] != channels:
        raise ValueError(f"input must be [1,T,{channels}], got {input_shape}")
    sequence = input_shape[1]
    if sequence <= 0 or sequence % 32:
        raise ValueError(f"input sequence must be positive and tile aligned, got {sequence}")
    if history_shape != (1, 3, channels):
        raise ValueError(f"history must be [1,3,{channels}], got {history_shape}")
    for index, tap in enumerate(taps):
        tap_shape = tuple(int(value) for value in tap.shape)
        if tap_shape[-1] != channels:
            raise ValueError(f"tap{index} last dimension must be {channels}, got {tap_shape}")

    qt, kt, vt = (width // 32 for width in widths)
    block_ct = channel_chunk_size // 32
    num_blocks = channels // channel_chunk_size
    mt = sequence // 32
    core_ranges, coords, work = _work_distribution(device, mt * num_blocks)
    runtime = per_core_runtime(coords, work)
    out_mc = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
    outputs = [allocate_output((1, sequence, width), input_tensor, ttnn.bfloat16, out_mc) for width in widths]
    compute_cfg = compute_descriptor(device, compute_kernel_config)

    reader = make_kernel(
        operation="qkv_causal_conv1d_silu",
        role="dataflow",
        filename="reader_qkv_causal_conv1d_silu_tile.cpp",
        dfb_ids=_DFBS,
        algorithm_compile_args=[block_ct, num_blocks],
        bindings=[
            ("input", input_tensor),
            ("history", history),
            ("tap0", taps[0]),
            ("tap1", taps[1]),
            ("tap2", taps[2]),
            ("tap3", taps[3]),
        ],
        template_count=2,
        entry_name="reader",
        runtime_args=runtime,
        runtime_arity=2,
        core_ranges=core_ranges,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = make_kernel(
        operation="qkv_causal_conv1d_silu",
        role="dataflow",
        filename="writer_qkv_causal_conv1d_silu.cpp",
        dfb_ids=_DFBS,
        algorithm_compile_args=[qt, kt, vt, block_ct, num_blocks],
        bindings=[("q", outputs[0]), ("k", outputs[1]), ("v", outputs[2])],
        template_count=5,
        entry_name="writer",
        runtime_args=runtime,
        runtime_arity=2,
        core_ranges=core_ranges,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = make_kernel(
        operation="qkv_causal_conv1d_silu",
        role="compute",
        filename="qkv_causal_conv1d_silu_tile.cpp",
        dfb_ids=_DFBS,
        algorithm_compile_args=[block_ct, num_blocks],
        bindings=[],
        template_count=2,
        entry_name="compute",
        runtime_args=runtime,
        runtime_arity=2,
        core_ranges=core_ranges,
        config=compute_cfg,
    )
    cbs = [
        cb_descriptor(_DFBS["act_tile"], 4 * block_ct, ttnn.bfloat16, core_ranges),
        cb_descriptor(_DFBS["shift"], 9, ttnn.bfloat16, core_ranges),
        cb_descriptor(_DFBS["shifted"], 2 * block_ct, ttnn.bfloat16, core_ranges),
        cb_descriptor(_DFBS["weights"], 4 * block_ct, ttnn.bfloat16, core_ranges),
        cb_descriptor(_DFBS["partial"], 2 * block_ct, ttnn.bfloat16, core_ranges),
        cb_descriptor(_DFBS["output"], 2 * block_ct, ttnn.bfloat16, core_ranges),
    ]
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([input_tensor, history, *taps, *outputs], program)
    return tuple(outputs)
