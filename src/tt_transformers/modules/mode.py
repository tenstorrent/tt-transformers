# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral execution mode values for reusable transformer modules."""

from enum import Enum


class Mode(Enum):
    """Execution paths supported by reusable prefill/decode modules."""

    DECODE = "decode"
    PREFILL = "prefill"


def normalize_mode(mode: str | Mode) -> str:
    """Return the string value used by reusable prefill/decode dispatch."""

    return mode.value if isinstance(mode, Mode) else mode
