# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn
from loguru import logger


### Math operations ###
def _nearest_32(x):
    return math.ceil(x / 32) * 32


def nearest_32(
    x,
):  # needs refctoring; to match alias called in some scripts (e.g. test_padding_test in unit tests)
    return _nearest_32(x)


### Profiling ###


### Turn flags on/off ###


### Tensor conversion ###


### Tilizing / Untilizing ###


### Measuring accuracy and other metrics ###


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    atol_delta = torch.max(torch.abs(golden - calculated)).item()
    rtol_delta = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()
    return (
        torch.allclose(golden, calculated, rtol, atol, True),
        f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}",
    )


def comp_pcc(golden, calculated, pcc=0.99, rtol=1e-05, atol=1e-04):
    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, 0.0

    # Test if either is completely zero — but a zero tensor is also a constant tensor,
    # so fall back to allclose instead of a hard 0.0: zero-vs-small-constant may be
    # within the caller's tolerances.
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.warning("One tensor is all zero. PCC undefined; falling back to allclose.")
        result = torch.allclose(golden, calculated, rtol=rtol, atol=atol)
        return result, float(result)

    golden = torch.squeeze(golden).flatten()
    calculated = torch.squeeze(calculated).flatten()

    # For now, mask all infs and nans (to zero) so that we check the rest... TODO
    # Skip this for integer types which don't have NaN/Inf values.
    if golden.dtype.is_floating_point:
        # FP8 doesn't support isfinite/nan_to_num and bfloat16 products lose precision,
        # so correlate these in float32.
        if golden.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16):
            golden = golden.to(torch.float32)
            calculated = calculated.to(torch.float32)

        # Zero out NaN/Inf, preserving the historical PCC values. nan_to_num allocates a
        # full-size copy of each tensor, so only do it when invalid values are actually
        # present; on the common all-finite path the tensors stay as views and no copy is
        # made (this short-circuit is what keeps peak memory near 1x of one input).
        if not bool((torch.isfinite(golden) & torch.isfinite(calculated)).all()):
            golden = torch.nan_to_num(golden, nan=0.0, posinf=0.0, neginf=0.0)
            calculated = torch.nan_to_num(calculated, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.equal(golden, calculated):
        return True, 1.0

    # Integer tensors must be correlated in floating point (centering/products would
    # otherwise truncate/overflow). float32 keeps the working set small.
    if not golden.dtype.is_floating_point:
        golden = golden.to(torch.float32)
        calculated = calculated.to(torch.float32)

    # Pearson r with float64 *accumulation* (dtype= on the reductions) over the float32
    # data: no float64 copy of either tensor is materialized, so peak memory stays near
    # 1x of one input on large tensors while matching a full-float64 correlation to
    # |Δ|<1e-9 across the high-PCC (>=0.999) range.
    n = golden.numel()
    g_centered = golden - (golden.sum(dtype=torch.float64) / n).to(golden.dtype)
    c_centered = calculated - (calculated.sum(dtype=torch.float64) / n).to(calculated.dtype)
    cov = (g_centered * c_centered).sum(dtype=torch.float64)
    g_sq_sum = g_centered.pow(2).sum(dtype=torch.float64)
    c_sq_sum = c_centered.pow(2).sum(dtype=torch.float64)
    denom = torch.sqrt(g_sq_sum * c_sq_sum)
    # pow/sum stay in float32 before the reduction; large-magnitude tensors (e.g. ldexp)
    # can overflow to inf here even though float64 accumulation would be finite.
    if not math.isfinite(denom.item()) or not math.isfinite(cov.item()):
        g_centered64 = g_centered.to(torch.float64)
        c_centered64 = c_centered.to(torch.float64)
        cov = (g_centered64 * c_centered64).sum()
        denom = torch.sqrt(g_centered64.pow(2).sum() * c_centered64.pow(2).sum())
    cal_pcc = (cov / denom).item()

    # Zero variance -> denom == 0 -> cal_pcc is nan: PCC is undefined for constant tensors.
    # Fall back to allclose rather than returning a misleading 1.0.
    if math.isnan(cal_pcc):
        logger.warning("PCC is NaN (zero variance / constant tensor). Falling back to allclose check.")
        result = torch.allclose(golden, calculated, rtol=rtol, atol=atol)
        return result, float(result)

    return cal_pcc >= pcc, cal_pcc


### Conv related helpers ###


def is_blackhole():
    ARCH_NAME = ttnn.get_arch_name()
    return "blackhole" in ARCH_NAME


def is_wormhole_b0():
    ARCH_NAME = ttnn.get_arch_name()
    return "wormhole_b0" in ARCH_NAME


# produces a tensor where each element in a page is the page number
# this tensor is easy to debug and visualize


# ── transformers 5.x Cache API compatibility ────────────────────────────────
# transformers 5.x removed the legacy Cache API: DynamicCache no longer exposes
# from_legacy_cache / to_legacy_cache / key_cache / value_cache (per-layer KV now
# lives at cache.layers[i].keys/.values). These helpers work on both 4.x and 5.x.


def hf_cache_to_legacy(cache):
    """Export a transformers Cache to the legacy tuple-of-(key, value) format."""
    if hasattr(cache, "to_legacy_cache"):  # transformers < 5.x
        return cache.to_legacy_cache()
    return tuple((layer.keys, layer.values) for layer in cache.layers)  # transformers >= 5.x


def copy_to_buffer(src: "ttnn.Tensor", dst: "ttnn.Tensor", target_dtype) -> None:
    """Convert ``src`` to ``dst``'s layout/dtype/shape/memcfg and write it into
    ``dst``. ``dst``'s device buffer is preserved (no reallocation) so any
    captured trace and the DRAM prefetcher's recorded buffer addresses remain
    valid. The final ``ttnn.to_memory_config`` with ``output_tensor=dst`` both
    reshards to ``dst``'s memory config and copies into ``dst``'s buffer.
    """
    converted = src

    if converted.layout != dst.layout:
        converted = ttnn.to_layout(converted, layout=dst.layout)

    if converted.dtype != target_dtype:
        converted = ttnn.typecast(converted, dtype=target_dtype)

    if tuple(converted.shape) != tuple(dst.shape):
        converted = ttnn.reshape(converted, list(dst.shape))

    ttnn.to_memory_config(converted, dst.memory_config(), output_tensor=dst)
