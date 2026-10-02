# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


def filter_none(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


def split_list(lst, n):
    """Split list into n equal parts."""
    chunk_size = len(lst) // n
    return [list(lst[i * chunk_size : (i + 1) * chunk_size]) for i in range(n)]
