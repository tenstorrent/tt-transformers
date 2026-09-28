# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stock-TTNN generic-op launcher for speculative GDN recurrence."""

from __future__ import annotations

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

_SPEC = {
    "src_in": (0, "2ch", "bf16"),
    "taps": (1, "ch", "bf16"),
    "z_in": (2, "vt", "bf16"),
    "ab_in": (3, 1, "bf16"),
    "sel": (4, "3k", "bf16"),
    "mask_T": (5, 1, "bf16"),
    "e_t": (6, "t", "bf16"),
    "rsel": (7, "t", "bf16"),
    "csel": (8, 2, "bf16"),
    "w_in": (9, "vt", "bf16"),
    "scaler": (10, 1, "fp32"),
    "eps_l2": (11, 1, "bf16"),
    "eps_norm": (12, 1, "bf16"),
    "dtb_s": (13, 1, "fp32"),
    "nea_s": (14, 1, "fp32"),
    "state_in": (15, "kv", "fp32"),
    "ctrl_r": (16, 1, "ctrl"),
    "wc": (17, "ch", "bf16"),
    "shift": (18, "chk", "bf16"),
    "cv": (19, "ch", "fp32"),
    "tmp": (20, "max", "fp32"),
    "stats": (21, 1, "fp32"),
    "scratch": (22, 1, "fp32"),
    "inv": (23, 1, "fp32"),
    "qc": (24, "kt", "fp32"),
    "kc": (25, "kt", "fp32"),
    "vc": (26, "vt", "fp32"),
    "qn": (27, "kt", "fp32"),
    "kn": (28, "kt", "fp32"),
    "vm": (29, "vt", "fp32"),
    "kt": (30, "kt", "fp32"),
    "g1": (31, 1, "fp32"),
    "a_s": (32, 1, "fp32"),
    "b_s": (33, 1, "fp32"),
    "beta_t": (34, "t", "fp32"),
    "dec": (35, "t", "fp32"),
    "eb": (36, 1, "fp32"),
    "hd": (37, "kv", "fp32"),
    "vread": (38, "vt", "fp32"),
    "delta": (39, "vt", "fp32"),
    "outer": (40, "kv", "fp32"),
    "hn": (41, "kv", "fp32"),
    "gq": (42, "vt", "fp32"),
    "acc_a": (43, "vt", "fp32"),
    "acc_b": (44, "vt", "fp32"),
    "on": (45, "vt", "fp32"),
    "zs": (46, "vt", "fp32"),
    "hnew": (47, "hnew", "fp32"),
    "out": (48, "vt", "out"),
    "wout": (49, "ch", "bf16"),
    "ctrl_w": (50, 1, "ctrl"),
    "bounce": (51, "ch", "bf16"),
}


def _cbs(*, kt, vt, T, K, hnew_depth, output_dtype, ctrl_bytes, core_ranges):
    ch = 2 * kt + vt
    counts = {
        "kt": kt,
        "vt": vt,
        "kv": kt * vt,
        "ch": ch,
        "2ch": 2 * ch,
        "3k": 3 + K,
        "t": T,
        "chk": ch * K,
        "max": max(kt, vt),
        "hnew": hnew_depth * kt * vt,
    }
    dtypes = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "out": output_dtype, "ctrl": ttnn.uint32}
    result = []
    for _, (slot, entries, kind) in sorted(_SPEC.items(), key=lambda item: item[1][0]):
        result.append(
            cb_descriptor(
                slot,
                counts.get(entries, entries),
                dtypes[kind],
                core_ranges,
                raw_page_bytes=ctrl_bytes if kind == "ctrl" else None,
            )
        )
    return result


def gdn_spec_step(
    qkvzab,
    win_a,
    win_b,
    ring,
    ctrl,
    taps,
    dt_bias,
    neg_exp_A,
    weight,
    num_value_heads,
    num_key_heads,
    key_dim,
    value_dim,
    T,
    B,
    qkvz_dim,
    *,
    conv_kernel=4,
    scale=None,
    l2_epsilon=1e-6,
    norm_epsilon=1e-6,
    hnew_depth=2,
    memory_config=None,
    compute_kernel_config=None,
    output_dtype=ttnn.bfloat16,
):
    """Run the fused candidate recurrence through ``ttnn.generic_op``."""

    nv, nk, dk, dv = num_value_heads, num_key_heads, key_dim, value_dim
    validate_dimensions(nv, nk, dk, dv)
    scale = validate_scalars(scale, dk, l2_epsilon, norm_epsilon)
    T, B, K = int(T), int(B), int(conv_kernel)
    if T < 1 or B < 1:
        raise ValueError("T and B must be positive")
    if K < 2 or K > 4:
        raise ValueError("conv_kernel must be in [2, 4]")
    Lw = K - 1 + T
    if Lw > 32:
        raise ValueError("convolution window must fit one tile")
    if B * T > 32 and 32 % T:
        raise ValueError("one user's candidate rows may not cross a tile row")
    if 2 * nv > 32:
        raise ValueError("a|b heads must fit one tile row")
    if hnew_depth not in (2, 4):
        raise ValueError("hnew_depth must be 2 or 4")
    if output_dtype not in (ttnn.bfloat16, ttnn.float32):
        raise ValueError("output_dtype must be bfloat16 or float32")
    expected_qkvz = 2 * nk * dk + 2 * nv * dv
    if int(qkvz_dim) != expected_qkvz or qkvz_dim % 32:
        raise ValueError(f"qkvz_dim must equal {expected_qkvz} and be tile aligned")
    if win_a.buffer_address() == win_b.buffer_address():
        raise ValueError("win_a and win_b must be distinct buffers")

    kt, vt = dk // 32, dv // 32
    C = 2 * nk * dk + nv * dv
    Ct = C // 32
    R = int(qkvzab.shape[-2])
    if not (B * T <= R <= ((B * T + 31) // 32) * 32):
        raise ValueError("qkvzab row count is outside the candidate tile row")
    device = qkvzab.device()
    core_ranges, coords = core_geometry(device, B * nv)
    work = [(u, h) for u in range(B) for h in range(nv)]
    reader_rt = per_core_runtime(coords, work)
    writer_rt = per_core_runtime(coords, work)
    compute_rt = per_core_runtime(coords, ((u,) for u, _ in work))
    dfbs = {name: slot for name, (slot, _, _) in _SPEC.items()}
    ctrl_bytes = int(ctrl.buffer_aligned_page_size())
    out_mc = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
    output = allocate_output((1, R, nv * dv), qkvzab, output_dtype, out_mc)
    compute_cfg = compute_descriptor(device, compute_kernel_config)
    z_tile0 = C // 32
    w_tiles = int(qkvzab.padded_shape[-1]) // 32

    reader_alg = [
        kt,
        vt,
        nk,
        nv,
        T,
        B,
        K,
        Lw,
        Ct,
        z_tile0,
        qkvz_dim // 32,
        w_tiles,
        int(dt_bias.dtype == ttnn.float32),
        int(neg_exp_A.dtype == ttnn.float32),
        float_bits(l2_epsilon),
        float_bits(norm_epsilon),
        ctrl_bytes,
        0xFFFFFFFF,
    ]
    writer_alg = [kt, vt, nk, nv, T, B, Lw, Ct, nv * vt, ctrl_bytes, 0xFFFFFFFF]
    compute_alg = [kt, vt, T, K, float_bits(scale), float_bits(1.0 / dv)]
    reader_bindings = [
        ("qkv", qkvzab),
        ("win_a", win_a),
        ("win_b", win_b),
        ("ring", ring),
        ("ctrl", ctrl),
        ("taps", taps),
        ("dtb", dt_bias),
        ("nea", neg_exp_A),
        ("weight", weight),
    ]
    writer_bindings = [
        ("ring_out", ring),
        ("out", output),
        ("win_a_w", win_a),
        ("win_b_w", win_b),
        ("ctrl_w", ctrl),
    ]
    reader = make_kernel(
        operation="gdn_spec_step",
        role="dataflow",
        filename="reader_gdn_spec_step.cpp",
        dfb_ids=dfbs,
        algorithm_compile_args=reader_alg,
        bindings=reader_bindings,
        template_count=18,
        entry_name="reader",
        runtime_args=reader_rt,
        runtime_arity=2,
        core_ranges=core_ranges,
        config=ttnn.ReaderConfigDescriptor(),
        opt_pragma="Os",
    )
    writer = make_kernel(
        operation="gdn_spec_step",
        role="dataflow",
        filename="writer_gdn_spec_step.cpp",
        dfb_ids=dfbs,
        algorithm_compile_args=writer_alg,
        bindings=writer_bindings,
        template_count=11,
        entry_name="writer",
        runtime_args=writer_rt,
        runtime_arity=2,
        core_ranges=core_ranges,
        config=ttnn.WriterConfigDescriptor(),
        opt_pragma="Os",
    )
    compute = make_kernel(
        operation="gdn_spec_step",
        role="compute",
        filename="gdn_spec_step.cpp",
        dfb_ids=dfbs,
        algorithm_compile_args=compute_alg,
        bindings=[],
        template_count=6,
        entry_name="compute",
        runtime_args=compute_rt,
        runtime_arity=1,
        core_ranges=core_ranges,
        config=compute_cfg,
    )
    program = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        semaphores=[],
        cbs=_cbs(
            kt=kt,
            vt=vt,
            T=T,
            K=K,
            hnew_depth=hnew_depth,
            output_dtype=output_dtype,
            ctrl_bytes=ctrl_bytes,
            core_ranges=core_ranges,
        ),
    )
    return ttnn.generic_op(
        [qkvzab, win_a, win_b, ring, ctrl, taps, dt_bias, neg_exp_A, weight, output],
        program,
    )
