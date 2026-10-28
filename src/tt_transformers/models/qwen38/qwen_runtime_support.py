# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Small host-only compatibility types used by the Qwen vLLM runtime.

Keeping these simple value types here avoids depending on the removed
``models.common`` source tree.
"""

from dataclasses import dataclass

import torch


@dataclass
class TokenResult:
    token: int
    text: str
    logprobs: list[float] | None = None


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[probs_sum - probs_sort > p] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.gather(probs_idx, -1, torch.multinomial(probs_sort, num_samples=1))
