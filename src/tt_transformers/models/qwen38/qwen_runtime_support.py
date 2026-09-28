# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Small host-only compatibility types used by the Qwen vLLM runtime.

The v6 serving path does not call the legacy demo/chat helpers, but the
specialized runtime retains those public convenience methods.  Keeping their
simple value types here avoids depending on the removed ``models.common``
source tree.
"""

from dataclasses import dataclass
from enum import Enum

import torch
from PIL import Image


class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    ipython = "ipython"


class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


@dataclass
class TokenResult:
    token: int
    text: str
    logprobs: list[float] | None = None


@dataclass
class CompletionMessage:
    content: str
    role: str = Role.assistant.value


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[probs_sum - probs_sort > p] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.gather(probs_idx, -1, torch.multinomial(probs_sort, num_samples=1))


def extract_images_from_messages(messages):
    images = []
    for message in messages:
        for content in message.get("content", []):
            if content.get("type") == "image" and "image" in content:
                images.append(content["image"])
    return images


def create_vision_mask(tokens, vision_token):
    locations = [i for i, token in enumerate(tokens) if token == vision_token]
    if not locations:
        return []
    if len(locations) == 1:
        return [[locations[0], -1]]
    masks = [[left, right] for left, right in zip(locations[:-1], locations[1:])]
    masks.append([locations[-1], len(tokens)])
    last_end = masks[-1][1]
    for mask in reversed(masks):
        if mask[0] == mask[1] - 1:
            mask[1] = last_end
        last_end = mask[1]
    return masks


def encode_content(content, images, image_token):
    if isinstance(content, Image.Image):
        images.append(content)
        if image_token is None:
            raise ValueError("image content requires an image token")
        return image_token
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(encode_content(item, images, image_token) for item in content)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content["text"]
        if content.get("type") == "image":
            images.append(content["image"])
            if image_token is None:
                raise ValueError("image content requires an image token")
            return image_token
    raise ValueError(f"Unknown content format: {content}")
