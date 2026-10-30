# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stock-TTNN generic-op launcher for fused/plain GDN decode."""

from __future__ import annotations

import math

import ttnn

from ._legacy import (
    allocate_output,
    cb_descriptor,
    compute_descriptor,
    core_geometry,
    float_bits,
    make_kernel,
    per_core_runtime,
    validate_dimensions,
    validate_scalars,
)

_PLAIN = {
    "q_in": (0, "kt", "bf16"),
    "k_in": (1, "kt", "bf16"),
    "v_in": (2, "vt", "bf16"),
    "beta_s": (3, 1, "fp32"),
    "g_s": (4, 1, "fp32"),
    "state_in": (5, "kv", "fp32"),
    "w_in": (6, "vt", "bf16"),
    "scaler": (7, 1, "fp32"),
    "eps_l2": (8, 1, "bf16"),
    "eps_norm": (9, 1, "bf16"),
    "mask": (10, 1, "bf16"),
    "tmp": (11, "max", "fp32"),
    "stats": (12, 1, "fp32"),
    "scratch": (13, 1, "fp32"),
    "inv": (14, 1, "fp32"),
    "qn": (15, "kt", "fp32"),
    "kn": (16, "kt", "fp32"),
    "vm": (17, "vt", "fp32"),
    "dec": (18, 1, "fp32"),
    "hd": (19, "kv", "fp32"),
    "vread": (20, "vt", "fp32"),
    "delta": (21, "vt", "fp32"),
    "kt": (22, "kt", "fp32"),
    "outer": (23, "kv", "fp32"),
    "hn": (24, "kv", "fp32"),
    "o": (25, "vt", "fp32"),
    "on": (26, "vt", "fp32"),
    "hnew": (27, "kv", "fp32"),
    "out": (28, "vt", "out"),
}

_FUSED = {
    "hist": (0, 3, "bf16"),
    "taps": (1, 4, "bf16"),
    "cur": (2, 1, "bf16"),
    "sel": (3, "ct", "bf16"),
    "z_in": (4, "vt", "bf16"),
    "a_s": (5, 1, "fp32"),
    "b_s": (6, 1, "fp32"),
    "dtb_s": (7, 1, "fp32"),
    "nea_s": (8, 1, "fp32"),
    "state_in": (9, "kv", "fp32"),
    "w_in": (10, "vt", "bf16"),
    "scaler": (11, 1, "fp32"),
    "eps_l2": (12, 1, "bf16"),
    "eps_norm": (13, 1, "bf16"),
    "mask": (14, 1, "bf16"),
    "tmp": (15, "max", "fp32"),
    "stats": (16, 1, "fp32"),
    "scratch": (17, 1, "fp32"),
    "inv": (18, 1, "fp32"),
    "qn": (19, "kt", "fp32"),
    "kn": (20, "kt", "fp32"),
    "vm": (21, "vt", "fp32"),
    "dec": (22, 1, "fp32"),
    "hd": (23, "kv", "fp32"),
    "vread": (24, "vt", "fp32"),
    "delta": (25, "vt", "fp32"),
    "kt": (26, "kt", "fp32"),
    "outer": (27, "kv", "fp32"),
    "hn": (28, "kv", "fp32"),
    "o": (29, "vt", "fp32"),
    "on": (30, "vt", "fp32"),
    "conv_p": (31, 1, "fp32"),
    "out_acc": (32, "vt", "fp32"),
    "acc2": (33, "vt", "fp32"),
    "qc": (34, "kt", "fp32"),
    "kc": (35, "kt", "fp32"),
    "vc": (36, "vt", "fp32"),
    "beta_t": (37, 1, "fp32"),
    "zs": (38, "vt", "fp32"),
    "hnew": (39, "kv", "fp32"),
    "out": (40, "vt", "out"),
    "wshift": (41, 4, "bf16"),
}


def _cbs(table, *, kt, vt, output_dtype, core_ranges):
    counts = {"kt": kt, "vt": vt, "kv": kt * vt, "ct": 2 * kt + vt, "max": max(kt, vt)}
    dtypes = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "out": output_dtype}
    return [
        cb_descriptor(slot, counts.get(entries, entries), dtypes[kind], core_ranges)
        for _, (slot, entries, kind) in sorted(table.items(), key=lambda item: item[1][0])
    ]


def gdn_decode_step(
    qkv,
    beta,
    g,
    state,
    weight,
    num_value_heads,
    num_key_heads,
    key_dim,
    value_dim,
    *,
    scale=None,
    l2_epsilon=1e-6,
    norm_epsilon=1e-6,
    memory_config=None,
    compute_kernel_config=None,
    output_dtype=ttnn.bfloat16,
    conv_hist=None,
    conv_taps=None,
    qkvz_dim=0,
):
    """Run one GDN decode step using a wheel-owned ``ttnn.generic_op`` program."""

    nv, nk, dk, dv = num_value_heads, num_key_heads, key_dim, value_dim
    validate_dimensions(nv, nk, dk, dv)
    scale = validate_scalars(scale, dk, l2_epsilon, norm_epsilon)
    if output_dtype not in (ttnn.bfloat16, ttnn.float32):
        raise ValueError("output_dtype must be bfloat16 or float32")
    if (conv_hist is None) != (conv_taps is None):
        raise ValueError("conv_hist and conv_taps must be supplied together")
    fused = conv_hist is not None
    if nv > 32 or (fused and 2 * nv > 32):
        raise ValueError("value-head geometry does not fit one tile row")

    kt, vt = dk // 32, dv // 32
    device = qkv.device()
    B = int(qkv.shape[-2]) if fused else 1
    if fused:
        expected_qkvz = 2 * nk * dk + 2 * nv * dv
        if int(qkvz_dim) != expected_qkvz or qkvz_dim % 32:
            raise ValueError(f"qkvz_dim must equal {expected_qkvz} and be tile aligned")
        if B < 1 or B > 32:
            raise ValueError("fused decode batch must be in [1, 32]")
        group = 1 if B == 1 else 2
        capacity = int(device.compute_with_storage_grid_size().x) * int(device.compute_with_storage_grid_size().y)
        while nv * math.ceil(B / group) > capacity:
            group += 2
        groups = math.ceil(B / group)
        work = [
            (head, group_index * group, min(group, B - group_index * group))
            for head in range(nv)
            for group_index in range(groups)
        ]
    else:
        work = [(head, 0, 1) for head in range(nv)]

    core_ranges, coords = core_geometry(device, len(work))
    reader_rt = per_core_runtime(coords, work if fused else ((head, 1) for head, _, _ in work))
    writer_rt = per_core_runtime(coords, work if fused else ((head, 1) for head, _, _ in work))
    compute_rt = per_core_runtime(coords, ((nu,) for _, _, nu in work) if fused else ((1,) for _ in work))
    table = _FUSED if fused else _PLAIN
    dfbs = {name: slot for name, (slot, _, _) in table.items()}
    out_mc = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
    output = allocate_output((1, B, nv * dv), qkv, output_dtype, out_mc)
    compute_cfg = compute_descriptor(device, compute_kernel_config)

    if fused:
        z_tile0 = 2 * nk * kt + nv * vt
        reader_alg = [
            kt,
            vt,
            nk,
            nv,
            z_tile0,
            qkvz_dim // 32,
            int(beta.dtype == ttnn.float32),
            int(g.dtype == ttnn.float32),
            float_bits(l2_epsilon),
            float_bits(norm_epsilon),
        ]
        writer_alg = [kt, vt, nk, nv]
        reader_bindings = [
            ("qkv", qkv),
            ("dtb", beta),
            ("nea", g),
            ("state", state),
            ("weight", weight),
            ("hist", conv_hist),
            ("taps", conv_taps),
        ]
        writer_bindings = [("state_out", state), ("out", output), ("qkv_w", qkv), ("hist_w", conv_hist)]
        reader_name, writer_name, compute_name = (
            "reader_gdn_decode_step_conv.cpp",
            "writer_gdn_decode_step_conv.cpp",
            "gdn_decode_step_conv.cpp",
        )
        reader_template, writer_template = 10, 4
        reader_arity = writer_arity = 3
        compute_pragma = "O2"
    else:
        reader_alg = [
            kt,
            vt,
            nk,
            nv,
            int(beta.dtype == ttnn.float32),
            int(g.dtype == ttnn.float32),
            float_bits(l2_epsilon),
            float_bits(norm_epsilon),
        ]
        writer_alg = [kt, vt]
        reader_bindings = [("qkv", qkv), ("beta", beta), ("g", g), ("state", state), ("weight", weight)]
        writer_bindings = [("state_out", state), ("out", output)]
        reader_name, writer_name, compute_name = (
            "reader_gdn_decode_step.cpp",
            "writer_gdn_decode_step.cpp",
            "gdn_decode_step.cpp",
        )
        reader_template, writer_template = 8, 2
        reader_arity = writer_arity = 2
        compute_pragma = None

    compute_alg = [kt, vt, float_bits(scale), float_bits(1.0 / dv)]
    reader = make_kernel(
        operation="gdn_decode_step",
        role="dataflow",
        filename=reader_name,
        dfb_ids=dfbs,
        algorithm_compile_args=reader_alg,
        bindings=reader_bindings,
        template_count=reader_template,
        entry_name="reader",
        runtime_args=reader_rt,
        runtime_arity=reader_arity,
        core_ranges=core_ranges,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = make_kernel(
        operation="gdn_decode_step",
        role="dataflow",
        filename=writer_name,
        dfb_ids=dfbs,
        algorithm_compile_args=writer_alg,
        bindings=writer_bindings,
        template_count=writer_template,
        entry_name="writer",
        runtime_args=writer_rt,
        runtime_arity=writer_arity,
        core_ranges=core_ranges,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = make_kernel(
        operation="gdn_decode_step",
        role="compute",
        filename=compute_name,
        dfb_ids=dfbs,
        algorithm_compile_args=compute_alg,
        bindings=[],
        template_count=4,
        entry_name="compute",
        runtime_args=compute_rt,
        runtime_arity=1,
        core_ranges=core_ranges,
        config=compute_cfg,
        opt_pragma=compute_pragma,
    )
    program = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        semaphores=[],
        cbs=_cbs(table, kt=kt, vt=vt, output_dtype=output_dtype, core_ranges=core_ranges),
    )
    io = [qkv, beta, g, state, weight]
    if fused:
        io.extend([conv_hist, conv_taps])
    io.append(output)
    return ttnn.generic_op(io, program)
