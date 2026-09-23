# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral sampling values and helpers.

One canonical SamplingParams identity lives in sampling_params. The on-device
sampling implementation is ``tt_transformers.modules.sampling``; this package
carries only the shared value types and the log-probability helper.

The TTTv1-era ``SamplingGenerator`` / ``TTSampling`` / ``TTPenalties`` surface
was removed. Nothing in this package consumed it, and tt-metal keeps and uses
its own copy under ``models/common/sampling``. Their sampling-parameter helpers
duplicated ``tt_transformers.modules.sampling.params``, which is the one to
import.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "LogProbsCalculator": (".tt_log_probs", "LogProbsCalculator"),
    "LogProbsResult": (".tt_log_probs", "LogProbsResult"),
    "SamplingParams": (".sampling_params", "SamplingParams"),
    "split_list": ("._utils", "split_list"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
