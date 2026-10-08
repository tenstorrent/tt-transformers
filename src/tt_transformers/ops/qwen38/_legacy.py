# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Legacy ``ProgramDescriptor`` lowering for the Qwen GDN kernels.

TTNN 0.79 exposes ``generic_op`` but not Metal 2.0 ``ProgramSpec``.  The
shipped kernels are retained byte-for-byte as provenance inputs and are
lowered to inline legacy sources at descriptor construction time.  Only the
generated binding/entry shim changes; the algorithmic bodies are untouched.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path

import ttnn

from ._kernels import KERNEL_ROOT, kernel_path

_TILE = ttnn.Tile((32, 32))
_TILE_DESCRIPTOR = ttnn.TileDescriptor(_TILE)
_GDN_INCLUDE_PREFIX = "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/"


def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _replace_private_includes(body: str) -> str:
    for role in ("dataflow", "compute"):
        directory = KERNEL_ROOT / "gdn_decode_step" / role
        for helper in directory.glob("*.hpp"):
            body = body.replace(
                f'#include "{_GDN_INCLUDE_PREFIX}{role}/{helper.name}"',
                f'#include "{helper}"',
            )
    return body


def legacy_source(
    *,
    operation: str,
    role: str,
    filename: str,
    dfb_ids: Mapping[str, int],
    tensor_tokens: Sequence[tuple[str, int, int]],
    template_count: int,
    entry_name: str,
    runtime_arity: int,
    opt_pragma: str | None = None,
) -> str:
    """Generate the legacy entry/binding shim around one unchanged source."""

    # The TILE KDA prefill path is intentionally eager so it can hand five
    # intermediate activations to the drafter after every prompt chunk.  That
    # means this helper is reached once per reader/writer/compute kernel for
    # every GDN layer, not just while the program cache is warming.  Keep the
    # public API permissive (callers naturally have dict/list inputs), but
    # canonicalize its immutable source-shape inputs before entering the
    # cache.  Tensor addresses and runtime arguments are deliberately *not*
    # cached here; make_kernel still rebuilds those for every invocation.
    return _legacy_source_cached(
        operation,
        role,
        filename,
        tuple(dfb_ids.items()),
        tuple(tuple(token) for token in tensor_tokens),
        int(template_count),
        entry_name,
        int(runtime_arity),
        opt_pragma,
    )


@cache
def _legacy_source_cached(
    operation: str,
    role: str,
    filename: str,
    dfb_ids: tuple[tuple[str, int], ...],
    tensor_tokens: tuple[tuple[str, int, int], ...],
    template_count: int,
    entry_name: str,
    runtime_arity: int,
    opt_pragma: str | None,
) -> str:
    """Cached implementation for :func:`legacy_source`.

    The generated source depends only on the normalized arguments above.
    Device/tensor state is carried by ``KernelDescriptor`` compile/common/
    runtime arguments, which remain freshly constructed by ``make_kernel``.
    """

    body = Path(kernel_path(operation, role, filename)).read_text()
    body = _replace_private_includes(body)
    if body.count("TT_KERNEL void") != 1:
        raise ValueError(f"{filename}: expected exactly one Metal 2.0 entry")
    body = body.replace("TT_KERNEL void", "static FORCE_INLINE void", 1)

    prefix = [
        "// qwen38-v6 legacy-wrapper ABI version 1",
        '#include "api/dataflow/dataflow_buffer.h"',
        '#include "api/tensor/tensor_binding_token.h"',
    ]
    prefix.append("namespace dfb {")
    prefix.extend(f"constexpr DFBBindingToken {name}{{{slot}u}};" for name, slot in dfb_ids)
    prefix.append("}  // namespace dfb")
    prefix.append("namespace tensor {")
    for name, cta_offset, crta_offset in tensor_tokens:
        prefix.append(f"using {name}_t = ::tensor_accessor::TensorBindingToken<{cta_offset}u, {crta_offset}u>;")
        prefix.append(f"constexpr {name}_t {name}{{}};")
    prefix.append("}  // namespace tensor")

    template_args = ", ".join(f"get_compile_time_arg_val({index})" for index in range(template_count))
    runtime_args = ", ".join(f"get_arg_val<uint32_t>({index})" for index in range(runtime_arity))
    suffix = f"void kernel_main() {{ {entry_name}<{template_args}>({runtime_args}); }}"
    source = "\n".join(prefix) + "\n" + body + "\n" + suffix + "\n"
    if "TT_KERNEL" in source:
        raise ValueError(f"{filename}: generated legacy source still contains TT_KERNEL")
    if source.count("kernel_main") != 1:
        raise ValueError(f"{filename}: generated legacy source must contain one kernel_main")
    return source


def _tensor_binding_layout(
    algorithm_compile_args: Sequence[int], bindings: Sequence[tuple[str, object]]
) -> tuple[list[int], list[int], list[tuple[str, int, int]]]:
    compile_args = list(algorithm_compile_args)
    common_runtime_args: list[int] = []
    tokens = []
    for name, tensor in bindings:
        accessor = ttnn.TensorAccessorArgs(tensor)
        tokens.append((name, len(compile_args), 4 * len(common_runtime_args)))
        compile_args.extend(accessor.get_compile_time_args())
        common_runtime_args.append(int(tensor.buffer_address()))
        common_runtime_args.extend(accessor.get_common_runtime_args())
    return compile_args, common_runtime_args, tokens


def make_kernel(
    *,
    operation: str,
    role: str,
    filename: str,
    dfb_ids: Mapping[str, int],
    algorithm_compile_args: Sequence[int],
    bindings: Sequence[tuple[str, object]],
    template_count: int,
    entry_name: str,
    runtime_args,
    runtime_arity: int,
    core_ranges,
    config,
    opt_pragma: str | None = None,
):
    compile_args, common_runtime_args, tokens = _tensor_binding_layout(algorithm_compile_args, bindings)
    source = legacy_source(
        operation=operation,
        role=role,
        filename=filename,
        dfb_ids=dfb_ids,
        tensor_tokens=tokens,
        template_count=template_count,
        entry_name=entry_name,
        runtime_arity=runtime_arity,
        opt_pragma=opt_pragma,
    )
    build_opt_level = None
    if opt_pragma is not None:
        build_levels = getattr(ttnn.KernelDescriptor, "BuildOptLevel", None)
        if build_levels is None:
            raise RuntimeError(
                "qwen38-v6-ops requires a TTNN build that exposes "
                "KernelDescriptor.BuildOptLevel; the compiler pragma fallback "
                "cannot reproduce the native LTO link optimization level"
            )
        try:
            build_opt_level = getattr(build_levels, opt_pragma)
        except AttributeError as exc:
            raise ValueError(f"unsupported kernel build optimization level {opt_pragma!r}") from exc
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=compile_args,
        runtime_args=runtime_args,
        common_runtime_args=common_runtime_args,
        config=config,
        opt_level=build_opt_level,
        compiler_include_paths=[
            str(KERNEL_ROOT / "gdn_decode_step" / "dataflow"),
            str(KERNEL_ROOT / "gdn_decode_step" / "compute"),
        ],
    )


def cb_descriptor(slot: int, entries: int, dtype, core_ranges, *, raw_page_bytes: int | None = None):
    if entries <= 0:
        raise ValueError(f"CB {slot} entries must be positive")
    page_size = int(raw_page_bytes or _TILE.get_tile_size(dtype))
    kwargs = {
        "buffer_index": slot,
        "data_format": dtype,
        "page_size": page_size,
    }
    if raw_page_bytes is None:
        kwargs["tile"] = _TILE_DESCRIPTOR
    fmt = ttnn.CBFormatDescriptor(**kwargs)
    return ttnn.CBDescriptor(
        total_size=entries * page_size,
        core_ranges=core_ranges,
        format_descriptors=[fmt],
    )


def core_geometry(device, count: int):
    grid = device.compute_with_storage_grid_size()
    capacity = int(grid.x) * int(grid.y)
    if count <= 0 or count > capacity:
        raise ValueError(f"operation requires {count} cores but grid capacity is {capacity}")
    cores = ttnn.num_cores_to_corerangeset(count, grid, row_wise=True)
    coords = [ttnn.CoreCoord(index % int(grid.x), index // int(grid.x)) for index in range(count)]
    return cores, coords


def per_core_runtime(coords: Iterable, values: Iterable[Sequence[int]]):
    args = ttnn.RuntimeArgs()
    for core, row in zip(coords, values, strict=True):
        args[int(core.x)][int(core.y)] = [int(value) for value in row]
    return args


def compute_descriptor(device, compute_kernel_config=None):
    resolved = ttnn.init_device_compute_kernel_config(
        device.arch(),
        compute_kernel_config,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    if getattr(resolved, "packer_l1_acc", False):
        raise ValueError("GDN operations do not support packer_l1_acc")
    desc = ttnn.ComputeConfigDescriptor()
    for name in (
        "math_fidelity",
        "math_approx_mode",
        "fp32_dest_acc_en",
        "dst_full_sync_en",
        "bfp8_pack_precise",
        "enable_trisc2_rvv",
    ):
        if hasattr(resolved, name) and hasattr(desc, name):
            setattr(desc, name, getattr(resolved, name))
    if not desc.fp32_dest_acc_en:
        raise ValueError("GDN operations require fp32 destination accumulation")
    return desc


def allocate_output(shape, source, dtype, memory_config):
    return ttnn.empty(
        list(shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=source.device(),
        memory_config=memory_config,
    )


def validate_dimensions(nv: int, nk: int, dk: int, dv: int):
    for name, value in (("num_value_heads", nv), ("num_key_heads", nk), ("key_dim", dk), ("value_dim", dv)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if nv % nk:
        raise ValueError("num_value_heads must be a multiple of num_key_heads")
    if dk % 32 or dv % 32:
        raise ValueError("key_dim and value_dim must be tile aligned")


def validate_scalars(scale: float | None, dk: int, l2_epsilon: float, norm_epsilon: float):
    scale = 1.0 / math.sqrt(dk) if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if not math.isfinite(l2_epsilon) or not math.isfinite(norm_epsilon):
        raise ValueError("epsilon values must be finite")
    if l2_epsilon <= 0 or norm_epsilon <= 0:
        raise ValueError("epsilon values must be positive")
    return scale
