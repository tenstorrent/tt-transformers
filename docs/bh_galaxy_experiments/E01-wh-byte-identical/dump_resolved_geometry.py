#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dump every host-resolved Galaxy placement, so two commits can be diffed.

**This needs `ttnn` importable, but opens no device.** It runs in seconds on any
Linux box with the package installed, including the Wormhole Galaxy host before
its allocation is spent on anything.

Why it exists: phase 2 replaced the intra-chip core coordinates in `recipes.py`
and `prefetch.py` with a `GalaxyChipTopology` descriptor. Everything that could
have gone wrong in that change is resolved **on the host, before a module hot
path runs** -- memory configs, program configs, core range sets, sub-device
partitions. So the cheapest complete check is not a device run at all: resolve
the lot on both commits and diff the text.

`test_topology.py`'s golden tables already pin the descriptor's own tuples. This
pins what the *callers* build out of them, which is the half those tables cannot
reach.

Usage. **Copy this script out of the worktree first.** It was added after the
base commit, so `git checkout 0a3e045` deletes it and the base dump cannot run
from its in-tree path -- the mistake produces an empty base file and a "diff"
that is just the whole head dump, which looks like a total regression:

    cp docs/bh_galaxy_experiments/E01-wh-byte-identical/dump_resolved_geometry.py /tmp/dump.py
    git checkout 0a3e045    # phase 1: the last commit before the descriptor
    python /tmp/dump.py > /tmp/base.txt
    git checkout <head>
    python /tmp/dump.py > /tmp/head.txt
    diff -u /tmp/base.txt /tmp/head.txt && echo "IDENTICAL"

Deliberately uses only the API that exists on **both** commits, so it can be run
from the newer checkout against the older one without editing. Every section is
independently guarded: a section that cannot resolve prints its exception and the
rest still dumps, because a partial diff is worth more than an aborted one.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock

import ttnn

# Geometry for both validated Galaxy models, matching the host suites.
LLAMA = dict(dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=128256)
QWEN = dict(dim=5120, hidden_dim=25600, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=151936)
PREFILL_LENGTHS = (128, 2048)


def wormhole_galaxy_mesh():
    """A mocked Wormhole Galaxy, identical to the one the host suites drive.

    Mesh mappers need a live device, and placement identity does not depend on
    them, so they are replaced with sentinels exactly as `test_plans.py` does.
    `compute_with_storage_grid_size` returns a real `CoreCoord` because
    `ttnn.num_cores_to_corerangeset` is a pybind binding and rejects a
    duck-typed stand-in.
    """

    ttnn.ShardTensor2dMesh = lambda *args, **kwargs: "shard-2d-mapper"
    ttnn.ReplicateTensorToMesh = lambda *args, **kwargs: "replicate-mapper"

    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = (8, 4)
    mesh.get_num_devices.return_value = 32
    mesh.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    mesh.dram_grid_size.return_value = SimpleNamespace(x=12, y=1)
    mesh.compute_with_storage_grid_size.return_value = ttnn.CoreCoord(7, 10)
    return mesh


def render(value: object) -> str:
    """Render one value deterministically, tolerating a broken pybind `__repr__`.

    `ttnn.SDPAProgramConfig` on 0.77.0 raises `TypeError` out of **both**
    `__repr__` and `__str__` -- the binding cannot convert its own
    `std::string` return value -- which aborted two entire sections of this dump
    the first time it ran against real `ttnn`. Every one of its fields reads back
    fine, so fall back to an ordered field walk. That is not a workaround with a
    cost: a field-wise line is a *finer* diff than the repr string it replaces,
    because it names the field that moved.
    """

    try:
        return repr(value)
    except TypeError:
        pass

    fields = []
    for name in sorted(attribute for attribute in dir(value) if not attribute.startswith("_")):
        try:
            attribute = getattr(value, name)
        except Exception:  # noqa: BLE001 - an unreadable attribute must not abort the dump
            fields.append(f"{name}=<unreadable>")
            continue
        if callable(attribute):
            continue
        fields.append(f"{name}={render(attribute)}")
    return f"<{type(value).__qualname__} {' '.join(fields)}>"


def emit(label: str, value: object) -> None:
    print(f"{label} = {render(value)}")


def emit_dataclass(prefix: str, instance: object) -> None:
    """Print one line per field, sorted by name, so the diff is line-oriented."""

    for field in sorted(dataclasses.fields(instance), key=lambda f: f.name):
        emit(f"{prefix}.{field.name}", getattr(instance, field.name))


def section(title: str):
    def decorator(function):
        def wrapper(*args, **kwargs):
            print(f"\n##### {title}")
            try:
                function(*args, **kwargs)
            except Exception:  # noqa: BLE001 - a failed section must not abort the dump
                print(f"!!!!! {title} FAILED TO RESOLVE")
                traceback.print_exc(file=sys.stdout)

        return wrapper

    return decorator


@section("core sets")
def dump_core_sets() -> None:
    from tt_transformers.models.galaxy import recipes

    for name in (
        "ring_cores",
        "ring_receiver_cores",
        "ring_hop_cores",
        "worker_cores",
        "topk_cores",
        "prefetch_sender_cores",
    ):
        emit(name, getattr(recipes, name)())

    emit("RING_CORE_COUNT", recipes.RING_CORE_COUNT)
    emit("RING_ALIGNMENT", recipes.RING_ALIGNMENT)
    emit("GALAXY_CCL_RESERVED_WORKER_CORES", recipes.GALAXY_CCL_RESERVED_WORKER_CORES)
    emit("GALAXY_MATMUL_CB_BUDGET", recipes.GALAXY_MATMUL_CB_BUDGET)
    emit("worker_matmul_rectangle", recipes.worker_matmul_rectangle())
    for height in (1, 2, 4, 8, 10):
        emit(f"dense_matmul_worker_rectangle({height})", recipes.dense_matmul_worker_rectangle(height))


@section("prefetch mapping")
def dump_prefetch_mapping() -> None:
    from tt_transformers.models.galaxy import prefetch

    mapping = prefetch.galaxy_sender_receiver_mapping()
    emit("sender_receiver_mapping.length", len(mapping))
    for index, (sender, receivers) in enumerate(mapping):
        emit(f"sender_receiver_mapping[{index}].sender", sender)
        emit(f"sender_receiver_mapping[{index}].receivers", receivers)
    for count in (5, 15, 45):
        emit(f"galaxy_address_memory_config({count})", prefetch.galaxy_address_memory_config(count))


@section("vocabulary and reduction arithmetic")
def dump_vocabulary() -> None:
    from tt_transformers.models.galaxy import recipes

    for model_name, model in (("llama", LLAMA), ("qwen", QWEN)):
        padded = recipes.galaxy_padded_vocab_size(model["vocab_size"])
        emit(f"{model_name}.galaxy_padded_vocab_size", padded)
        local = padded // recipes.GALAXY_ROWS
        emit(f"{model_name}.local_padded_vocab", local)
        emit(f"{model_name}.pad_ring_width(local)", recipes.pad_ring_width(local))
        # Positional, because the reserve became a keyword-only parameter at head
        # and passing it would not resolve on the base commit.
        emit(f"{model_name}.lm_head_reduce_core_count", recipes.lm_head_reduce_core_count(local, 50))


@section("mesh-derived helpers")
def dump_mesh_helpers(mesh) -> None:
    from tt_transformers.models.galaxy import recipes

    emit("galaxy_prefill_mode_plan_cores", recipes.galaxy_prefill_mode_plan_cores(mesh))
    emit("sampling_core_grids", recipes.sampling_core_grids())
    for fused in (False, True):
        emit(f"rope_core_grids(use_qk_fused={fused})", recipes.rope_core_grids(mesh, use_qk_fused=fused))
    for decode in (True, False):
        for length in (128, 2048, 8192):
            emit(
                f"sdpa_program_config(length={length}, decode={decode})",
                recipes.sdpa_program_config(length, decode=decode, sub_core_grids=recipes.worker_cores()),
            )
    emit(
        "chunked_sdpa_program_config",
        recipes.chunked_sdpa_program_config(sub_core_grids=recipes.worker_cores()),
    )
    for local_k, local_n in ((1024, 3584), (2048, 2048)):
        emit(
            f"dram_sharded_weight_memory_config({local_k}, {local_n})",
            recipes.dram_sharded_weight_memory_config(mesh, local_k, local_n),
        )
        emit(
            f"ring_matmul_program_config({local_k}, {recipes.pad_ring_width(local_n)})",
            recipes.ring_matmul_program_config(local_k, recipes.pad_ring_width(local_n)),
        )


@section("resolved placements")
def dump_placements(mesh) -> None:
    from tt_transformers.models.galaxy import recipes

    for model_name, model in (("llama", LLAMA), ("qwen", QWEN)):
        geometry = recipes.GalaxyDenseGeometry(**model, max_seq_len=2048, prefill_sequence_lengths=PREFILL_LENGTHS)
        emit_dataclass(f"{model_name}.geometry", geometry)
        emit_dataclass(f"{model_name}.decode", recipes.resolve_galaxy_decode_placements(geometry, mesh))
        emit_dataclass(f"{model_name}.prefill", recipes.resolve_galaxy_prefill_placements(geometry, mesh))


@section("resolved resource plans")
def dump_resource_plans(mesh) -> None:
    from tt_transformers.models.galaxy import recipes
    from tt_transformers.models.galaxy.plans import build_galaxy_resources_config

    for model_name, model in (("llama", LLAMA), ("qwen", QWEN)):
        geometry = recipes.GalaxyDenseGeometry(**model, max_seq_len=2048, prefill_sequence_lengths=PREFILL_LENGTHS)
        decode = recipes.resolve_galaxy_decode_placements(geometry, mesh)
        config = build_galaxy_resources_config(mesh, geometry, decode)

        for mode in ("prefill", "decode"):
            plan = getattr(config, mode)
            emit(f"{model_name}.{mode}.worker_sub_device_id", plan.worker_sub_device_id)
            emit(f"{model_name}.{mode}.stall_group", plan.stall_group)
            emit(f"{model_name}.{mode}.sub_devices.count", len(plan.sub_devices))
            emit(f"{model_name}.{mode}.semaphore_cores", plan.semaphore_cores)
            emit(f"{model_name}.{mode}.worker_cores", plan.worker_cores)
            # num_links is the value phase 1's clamp touches; the ordered tuple is
            # what `test_wormhole_link_counts_are_unchanged_by_the_clamp` pins.
            emit(
                f"{model_name}.{mode}.num_links",
                tuple(collective.num_links for collective in plan.collectives),
            )
            for index, collective in enumerate(plan.collectives):
                emit(f"{model_name}.{mode}.collectives[{index}].key", collective.key)
                emit(f"{model_name}.{mode}.collectives[{index}].topology", collective.topology)
                emit(f"{model_name}.{mode}.collectives[{index}].persistent", collective.persistent_output_specs)
                emit(f"{model_name}.{mode}.collectives[{index}].intermediate", collective.intermediate_output_specs)


def ttnn_version() -> str:
    """Return the installed `ttnn` version. It publishes no `__version__`."""

    try:
        return importlib.metadata.version("ttnn")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    print(f"# ttnn {ttnn_version()}")
    mesh = wormhole_galaxy_mesh()
    dump_core_sets()
    dump_prefetch_mapping()
    dump_vocabulary()
    dump_mesh_helpers(mesh)
    dump_placements(mesh)
    dump_resource_plans(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
