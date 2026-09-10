"""Narrow fixture policy helpers internalized from tt-metal tests."""

from examples.common.trace_region_sizes import resolve_trace_region_size


def get_updated_device_params(params):
    """Translate dispatch-core parameters into the `open_mesh_device` keyword.

    `ttnn.open_mesh_device` takes `dispatch_core_config`; test parametrizations
    express the intent as `dispatch_core_axis` / `dispatch_core_type`, which is
    what tt-metal's `get_updated_device_params` folded together. The narrowed
    helper here returned its parameters unchanged, so those two keys reached
    `open_mesh_device` verbatim and every device test setting one failed at
    fixture setup with `unexpected keyword argument 'dispatch_core_axis'`.
    Nothing caught it because no device suite runs in this repository's CI.

    `fabric_tensix_config` is read but left in place: the `mesh_device` fixture
    pops it separately for `set_fabric`.
    """

    import ttnn

    updated = dict(params or {})
    if "dispatch_core_config" in updated:
        return updated

    axis = updated.pop("dispatch_core_axis", None)
    core_type = updated.pop("dispatch_core_type", None)
    if axis is None and core_type is None:
        return updated

    fabric_tensix_config = updated.get("fabric_tensix_config")
    if axis == ttnn.DispatchCoreAxis.ROW and ttnn.device.is_blackhole():
        # ROW dispatch on Blackhole needs both fabric configs; without them the only
        # working choice is COL, which is the substitution tt-metal forced here too.
        if not (updated.get("fabric_config") and fabric_tensix_config):
            axis = ttnn.DispatchCoreAxis.COL

    updated["dispatch_core_config"] = ttnn.DispatchCoreConfig(core_type, axis, fabric_tensix_config)
    return updated


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
