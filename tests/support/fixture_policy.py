# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Narrow fixture policy helpers internalized from tt-metal tests."""

from examples.common.trace_region_sizes import resolve_trace_region_size


def get_updated_device_params(params):
    return dict(params or {})


def get_logical_sku(request, mesh_device):
    del request
    count = mesh_device.get_num_devices()
    return {1: "N150", 2: "N300", 4: "P150x4", 8: "T3K", 32: "TG"}.get(count, f"{count}dev")


def get_supported_trace_region_size(request, mesh_device):
    param = getattr(request, "param", {})
    if isinstance(param, dict) and "trace_region_size" in param:
        return param["trace_region_size"]
    model_key = param.get("trace_model_key") if isinstance(param, dict) else None
    return resolve_trace_region_size(model_key, get_logical_sku(request, mesh_device)) if model_key else None
