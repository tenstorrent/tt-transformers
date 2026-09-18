# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lazy model-local exports; optional HF dependencies load on access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "GALAXY_COLUMNS": ("tt_transformers.models.galaxy.recipes", "GALAXY_COLUMNS"),
    "GALAXY_DEVICE_COUNT": ("tt_transformers.models.galaxy.recipes", "GALAXY_DEVICE_COUNT"),
    "GALAXY_GLOBAL_CB_SIZE": ("tt_transformers.models.galaxy.prefetch", "GALAXY_GLOBAL_CB_SIZE"),
    "GALAXY_MESH_SHAPE": ("tt_transformers.models.galaxy.recipes", "GALAXY_MESH_SHAPE"),
    "GALAXY_PHYSICAL_BATCH": ("tt_transformers.models.galaxy.recipes", "GALAXY_PHYSICAL_BATCH"),
    "GALAXY_ROWS": ("tt_transformers.models.galaxy.recipes", "GALAXY_ROWS"),
    "GALAXY_USERS_PER_COLUMN": ("tt_transformers.models.galaxy.recipes", "GALAXY_USERS_PER_COLUMN"),
    "GalaxyAttentionCollectives": ("tt_transformers.models.galaxy.collectives", "GalaxyAttentionCollectives"),
    "GalaxyAttentionKVSpec": ("tt_transformers.models.galaxy.kv_contract", "GalaxyAttentionKVSpec"),
    "GalaxyCCL": ("tt_transformers.models.galaxy.ccl", "GalaxyCCL"),
    "GalaxyCCLCollaborator": ("tt_transformers.models.galaxy.ccl", "GalaxyCCLCollaborator"),
    "GalaxyCCLConfig": ("tt_transformers.models.galaxy.ccl", "GalaxyCCLConfig"),
    "GalaxyCCLContext": ("tt_transformers.models.galaxy.ccl", "GalaxyCCLContext"),
    "GalaxyCollectivePlan": ("tt_transformers.models.galaxy.resources", "GalaxyCollectivePlan"),
    "GalaxyCollectiveResources": ("tt_transformers.models.galaxy.ccl", "GalaxyCollectiveResources"),
    "GalaxyColumnAllReduce": ("tt_transformers.models.galaxy.collectives", "GalaxyColumnAllReduce"),
    "GalaxyColumnUserSelector": ("tt_transformers.models.galaxy.collectives", "GalaxyColumnUserSelector"),
    "GalaxyDecodePlacements": ("tt_transformers.models.galaxy.recipes", "GalaxyDecodePlacements"),
    "GalaxyDenseGeometry": ("tt_transformers.models.galaxy.recipes", "GalaxyDenseGeometry"),
    "GalaxyModePlan": ("tt_transformers.models.galaxy.resources", "GalaxyModePlan"),
    "GalaxyModeResources": ("tt_transformers.models.galaxy.ccl", "GalaxyModeResources"),
    "GalaxyPagedAttentionConfig": ("tt_transformers.models.galaxy.kv_contract", "GalaxyPagedAttentionConfig"),
    "GalaxyPagedKVContract": ("tt_transformers.models.galaxy.kv_contract", "GalaxyPagedKVContract"),
    "GalaxyPrefillPlacements": ("tt_transformers.models.galaxy.recipes", "GalaxyPrefillPlacements"),
    "GalaxyResourceBindings": ("tt_transformers.models.galaxy.resources", "GalaxyResourceBindings"),
    "GalaxyResourceKey": ("tt_transformers.models.galaxy.ccl", "GalaxyResourceKey"),
    "GalaxyResources": ("tt_transformers.models.galaxy.resources", "GalaxyResources"),
    "GalaxyResourcesConfig": ("tt_transformers.models.galaxy.resources", "GalaxyResourcesConfig"),
    "GalaxySamplingPolicy": ("tt_transformers.models.galaxy.sampling_policy", "GalaxySamplingPolicy"),
    "GalaxyTensorSpec": ("tt_transformers.models.galaxy.resources", "GalaxyTensorSpec"),
    "TTNNGalaxyCCLResourceFactory": ("tt_transformers.models.galaxy.resources", "TTNNGalaxyCCLResourceFactory"),
    "WORMHOLE_GALAXY_ARCHITECTURE": (
        "tt_transformers.models.galaxy.recipes",
        "WORMHOLE_GALAXY_ARCHITECTURE",
    ),
    "build_galaxy_decode_collectives": ("tt_transformers.models.galaxy.plans", "build_galaxy_decode_collectives"),
    "build_galaxy_prefetcher": ("tt_transformers.models.galaxy.prefetch", "build_galaxy_prefetcher"),
    "build_galaxy_prefetcher_config": ("tt_transformers.models.galaxy.prefetch", "build_galaxy_prefetcher_config"),
    "build_galaxy_prefill_collectives": ("tt_transformers.models.galaxy.plans", "build_galaxy_prefill_collectives"),
    "build_galaxy_resources_config": ("tt_transformers.models.galaxy.plans", "build_galaxy_resources_config"),
    "chunked_sdpa_program_config": ("tt_transformers.models.galaxy.recipes", "chunked_sdpa_program_config"),
    "compute_kernel_config": ("tt_transformers.models.galaxy.recipes", "compute_kernel_config"),
    "create_galaxy_resources": ("tt_transformers.models.galaxy.resources", "create_galaxy_resources"),
    "deallocate_if_allocated": ("tt_transformers.models.galaxy.collectives", "deallocate_if_allocated"),
    "dram_sharded_weight_memory_config": ("tt_transformers.models.galaxy.recipes", "dram_sharded_weight_memory_config"),
    "galaxy_address_memory_config": ("tt_transformers.models.galaxy.prefetch", "galaxy_address_memory_config"),
    "galaxy_decode_mode_plan": ("tt_transformers.models.galaxy.plans", "galaxy_decode_mode_plan"),
    "galaxy_dram_prefetch_start": ("tt_transformers.models.galaxy.prefetch", "galaxy_dram_prefetch_start"),
    "galaxy_padded_vocab_size": ("tt_transformers.models.galaxy.recipes", "galaxy_padded_vocab_size"),
    "galaxy_prefill_mode_plan": ("tt_transformers.models.galaxy.plans", "galaxy_prefill_mode_plan"),
    "galaxy_runtime_tensor_factory": ("tt_transformers.models.galaxy.collectives", "galaxy_runtime_tensor_factory"),
    "galaxy_sender_receiver_mapping": ("tt_transformers.models.galaxy.prefetch", "galaxy_sender_receiver_mapping"),
    "resolve_galaxy_decode_placements": ("tt_transformers.models.galaxy.recipes", "resolve_galaxy_decode_placements"),
    "resolve_galaxy_prefill_placements": ("tt_transformers.models.galaxy.recipes", "resolve_galaxy_prefill_placements"),
    "rope_core_grids": ("tt_transformers.models.galaxy.recipes", "rope_core_grids"),
    "sampling_core_grids": ("tt_transformers.models.galaxy.recipes", "sampling_core_grids"),
    "select_galaxy_resource": ("tt_transformers.models.galaxy.plans", "select_galaxy_resource"),
    "validate_galaxy_mesh": ("tt_transformers.models.galaxy.recipes", "validate_galaxy_mesh"),
    "worker_cores": ("tt_transformers.models.galaxy.recipes", "worker_cores"),
}

__all__ = [
    "GALAXY_COLUMNS",
    "GALAXY_DEVICE_COUNT",
    "GALAXY_GLOBAL_CB_SIZE",
    "GALAXY_MESH_SHAPE",
    "GALAXY_PHYSICAL_BATCH",
    "GALAXY_ROWS",
    "GALAXY_USERS_PER_COLUMN",
    "GalaxyAttentionCollectives",
    "GalaxyAttentionKVSpec",
    "GalaxyCCL",
    "GalaxyCCLCollaborator",
    "GalaxyCCLConfig",
    "GalaxyCCLContext",
    "GalaxyCollectivePlan",
    "GalaxyCollectiveResources",
    "GalaxyColumnAllReduce",
    "GalaxyColumnUserSelector",
    "GalaxyDecodePlacements",
    "GalaxyDenseGeometry",
    "GalaxyModePlan",
    "GalaxyModeResources",
    "GalaxyPagedAttentionConfig",
    "GalaxyPagedKVContract",
    "GalaxyPrefillPlacements",
    "GalaxyResourceBindings",
    "GalaxyResourceKey",
    "GalaxyResources",
    "GalaxyResourcesConfig",
    "GalaxySamplingPolicy",
    "GalaxyTensorSpec",
    "TTNNGalaxyCCLResourceFactory",
    "WORMHOLE_GALAXY_ARCHITECTURE",
    "build_galaxy_decode_collectives",
    "build_galaxy_prefetcher",
    "build_galaxy_prefetcher_config",
    "build_galaxy_prefill_collectives",
    "build_galaxy_resources_config",
    "chunked_sdpa_program_config",
    "compute_kernel_config",
    "create_galaxy_resources",
    "deallocate_if_allocated",
    "dram_sharded_weight_memory_config",
    "galaxy_address_memory_config",
    "galaxy_decode_mode_plan",
    "galaxy_dram_prefetch_start",
    "galaxy_padded_vocab_size",
    "galaxy_prefill_mode_plan",
    "galaxy_runtime_tensor_factory",
    "galaxy_sender_receiver_mapping",
    "resolve_galaxy_decode_placements",
    "resolve_galaxy_prefill_placements",
    "rope_core_grids",
    "sampling_core_grids",
    "select_galaxy_resource",
    "validate_galaxy_mesh",
    "worker_cores",
]


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
