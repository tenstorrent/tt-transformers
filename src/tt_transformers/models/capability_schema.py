# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for the vLLM ``model_capabilities`` contract.

Every serving model declares a class-level ``model_capabilities`` dict on its
``*Generator`` (see ``models/<m>/vllm_generator.py``). The plugin snapshots that
class attribute before any instance exists, so it must stay a plain class-level
dict — never a property or an instance field.

The contract is deliberately terse: a subclass dict *replaces* the inherited one
rather than merging, so an **absent key means "not supported"** and no reader
assumes otherwise — except the keys whose ``reader_default`` is ``True``, where an
absent key means supported. Because absence is meaningful,
a key that upstream starts requiring after the split is silently "declined" here
until someone notices. This schema exists so that "someone" is a test: it names
every capability key this repo owns, records its type and its reader-side default,
and lets a host test assert that no generator declares a key the schema does not
know and that every required key is present.

This module imports no ``ttnn`` and evaluates nothing on device: it is pure data
so it can be read from the host suite on runners that do not have ``ttnn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class CapabilityKey:
    """One declared capability key: its type, reader-side default, and meaning.

    ``reader_default`` is the value a plugin reader assumes when the key is
    **absent** from a generator's ``model_capabilities`` dict. For most keys that
    means "not supported" (``False`` for the boolean flags, ``None`` for the
    advertised limits); the keys whose ``reader_default`` is ``True`` are the
    exceptions. ``required`` marks the keys every serving generator in this repo
    declares today; a model that dropped one would be changing the contract, and
    the schema test is meant to catch exactly that.
    """

    name: str
    value_type: str
    reader_default: Any
    required: bool
    doc: str


_CAPABILITY_KEYS: tuple[CapabilityKey, ...] = (
    CapabilityKey(
        name="supports_prefix_caching",
        value_type="bool",
        reader_default=False,
        required=True,
        doc="Generator can serve requests that reuse a cached prompt prefix.",
    ),
    CapabilityKey(
        name="supports_async_decode",
        value_type="bool",
        reader_default=False,
        required=True,
        doc="Generator supports the plugin's asynchronous decode output path.",
    ),
    CapabilityKey(
        name="supports_sample_on_device",
        value_type="bool",
        reader_default=False,
        required=True,
        doc="Generator can sample tokens on the device rather than on the host.",
    ),
    CapabilityKey(
        name="accepts_trace_mode",
        value_type="bool",
        reader_default=False,
        required=True,
        doc="Generator accepts a caller-chosen trace mode (none / decode_only / all).",
    ),
    CapabilityKey(
        name="supports_chunked_prefill",
        value_type="bool",
        reader_default=False,
        required=True,
        doc=(
            "Generator can prefill a prompt over several steps, each starting at a "
            "block-aligned offset into the paged KV cache. Absent, the plugin turns "
            "chunked prefill off and raises max_num_batched_tokens to max_model_len."
        ),
    ),
    CapabilityKey(
        name="max_device_top_k",
        value_type="int",
        reader_default=None,
        required=True,
        doc=(
            "Largest top-k the device sampler supports. A TTTv2-only key: it is not "
            "part of the tt-metal capability contract, so the schema describes what "
            "this repo owns. Only meaningful alongside supports_sample_on_device. "
            "Absent, the plugin sends an unbounded or larger top-k to the device, "
            "which rejects it."
        ),
    ),
    CapabilityKey(
        name="fabric_config",
        value_type="dict (kwargs: {'config': ttnn.FabricConfig})",
        reader_default=None,
        required=False,
        doc=(
            "Fabric the plugin must configure for this model's supported geometries. "
            "The value is a kwargs dict whose member is a ttnn.FabricConfig enum, so "
            "it is not a Python literal and cannot be evaluated on a host without "
            "ttnn. Present on the generators that pin an explicit fabric, which is "
            "FABRIC_1D, the plugin's default for their meshes; absent means the "
            "plugin's own per-mesh default applies (vllm-tt-plugin #116)."
        ),
    ),
    CapabilityKey(
        name="supports_device_penalties",
        value_type="bool",
        reader_default=True,
        required=False,
        doc=(
            "Whether the device applies repetition/presence penalties. An exception "
            "to 'absent means not supported': the plugin reads this key with a default "
            "of True, so an absent key means supported. Generators whose executor owns no "
            "device penalty state (SamplingState1D) declare False."
        ),
    ),
    CapabilityKey(
        name="supports_stochastic_device_prefill",
        value_type="bool",
        reader_default=True,
        required=False,
        doc=(
            "Whether a nonzero-temperature prefill row may be sampled on device. "
            "Generators whose executor owns no SamplingState1D declare False: their "
            "prefill sampler seeds from a fixed per-slot buffer, so the first token "
            "would depend on the slot rather than the request and would ignore its "
            "seed. The plugin does not currently read this key, so the declaration "
            "records the limit without changing where such a prefill is sampled."
        ),
    ),
)


CAPABILITY_SCHEMA: MappingProxyType[str, CapabilityKey] = MappingProxyType({key.name: key for key in _CAPABILITY_KEYS})

KNOWN_CAPABILITY_KEYS: frozenset[str] = frozenset(CAPABILITY_SCHEMA)

REQUIRED_CAPABILITY_KEYS: frozenset[str] = frozenset(name for name, key in CAPABILITY_SCHEMA.items() if key.required)
