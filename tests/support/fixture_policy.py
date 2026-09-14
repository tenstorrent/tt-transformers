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


def _requested_device_count(requested) -> int | None:
    """Return the device count a `mesh_device` parametrization asks for.

    Callers pass `request.param` -- an `int`, or a `(rows, columns)` tuple --
    because this runs *before* `open_mesh_device`, to choose a device-open
    parameter (`trace_region_size`). The name of the second argument used to say
    `mesh_device`, and the body called `.get_num_devices()` on it, which raises
    `AttributeError` on both shapes a caller actually passes. That never fired
    because both call paths are currently unreachable: nothing sets
    `TRACE_MODEL_KEY_PARAM` or `trace_model_key` in a `device_params` dict.
    """

    if isinstance(requested, tuple) and len(requested) == 2:
        return requested[0] * requested[1]
    if isinstance(requested, int):
        return requested
    get_num_devices = getattr(requested, "get_num_devices", None)
    return get_num_devices() if callable(get_num_devices) else None


def _is_blackhole_cluster() -> bool:
    """Report the architecture without an open mesh device.

    For the case that matters here -- a 32-device mesh, where count alone cannot
    tell Blackhole Galaxy from Wormhole Galaxy -- cluster type answers exactly,
    and it answers at this point in fixture setup, where no `MeshDevice` exists
    yet. Only the three Galaxy members are named, because they are the ones this
    repository already depends on (`tests/conftest.py`'s `is_galaxy`); naming an
    unverified member would raise `AttributeError` inside the `try` and silently
    disable the whole check. Everything else defers to the arch query.
    """

    import ttnn

    try:
        cluster_type = ttnn.cluster.get_cluster_type()
        if cluster_type == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
            return True
        if cluster_type in (ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG):
            return False
    except Exception:  # noqa: BLE001 - control plane cannot name the cluster; ask the arch instead
        pass
    try:
        return bool(ttnn.device.is_blackhole())
    except Exception:  # noqa: BLE001 - no device open and no cluster: do not guess Blackhole
        return False


def get_logical_sku(request, mesh_device):
    """Return the SKU token for the submesh a test asked for.

    Keyed on device count *and* architecture. Count alone reported `TG` for a
    32-device Blackhole mesh, collapsing Blackhole Galaxy onto the Wormhole
    entry, and `P150x4` for a four-device Wormhole one. `device_utils.
    get_device_name` already branches on arch, but it needs an open
    `MeshDevice`, which does not exist yet at this call site.
    """

    del request
    count = _requested_device_count(mesh_device)
    if count is None:
        return None
    if _is_blackhole_cluster():
        names = {1: "P150", 2: "P300", 4: "P150x4", 8: "P150x8", 32: "BHGLX"}
    else:
        names = {1: "N150", 2: "N300", 4: "N150x4", 8: "T3K", 32: "TG"}
    return names.get(count, f"{count}dev")


def get_supported_trace_region_size(request, mesh_device):
    param = getattr(request, "param", {})
    if isinstance(param, dict) and "trace_region_size" in param:
        return param["trace_region_size"]
    model_key = param.get("trace_model_key") if isinstance(param, dict) else None
    return resolve_trace_region_size(model_key, get_logical_sku(request, mesh_device)) if model_key else None
