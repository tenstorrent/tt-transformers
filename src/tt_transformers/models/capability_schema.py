# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for the vLLM ``model_capabilities`` contract.

Every serving model declares a class-level ``model_capabilities`` dict on its
``*Generator`` (see ``models/<m>/vllm_generator.py``). The plugin snapshots that
class attribute before any instance exists, so it must stay a plain class-level
dict — never a property or an instance field.

The contract is deliberately terse: a subclass dict *replaces* the inherited one
rather than merging, so an **absent key means "not supported"** and no reader
assumes otherwise — with one documented exception, ``supports_device_penalties``,
which the plugin reads with a default of ``True``. Because absence is meaningful,
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
    **absent** from a generator's ``model_capabilities`` dict. For every key that
    means "not supported" (``False`` for the boolean flags, ``None`` for the
    advertised limits) except ``supports_device_penalties``, whose reader default
    is ``True``. ``required`` marks the keys every serving generator in this repo
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
        name="max_device_top_k",
        value_type="int",
        reader_default=None,
        required=False,
        doc=(
            "Largest top-k the device sampler supports. A TTTv2-only key: it is not "
            "part of the tt-metal capability contract, so the schema describes what "
            "this repo owns. Only meaningful alongside supports_sample_on_device."
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
            "ttnn. Present on the SKUs whose geometries need a non-default (ring) "
            "fabric or that pin an explicit one; absent means the plugin's own "
            "per-mesh default applies (vllm-tt-plugin #116)."
        ),
    ),
    CapabilityKey(
        name="supports_device_penalties",
        value_type="bool",
        reader_default=True,
        required=False,
        doc=(
            "Whether the device applies repetition/presence penalties. THE exception "
            "to 'absent means not supported': the plugin reads this key with a default "
            "of True, so an absent key means supported. No generator declares it today."
        ),
    ),
)


CAPABILITY_SCHEMA: MappingProxyType[str, CapabilityKey] = MappingProxyType({key.name: key for key in _CAPABILITY_KEYS})

KNOWN_CAPABILITY_KEYS: frozenset[str] = frozenset(CAPABILITY_SCHEMA)

REQUIRED_CAPABILITY_KEYS: frozenset[str] = frozenset(name for name, key in CAPABILITY_SCHEMA.items() if key.required)
