# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral prompt helpers retained from the legacy demo boundary."""


def _chat_template_ids(encoded):
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return list(encoded)


def encode_prompt_hf(tokenizer, prompt_text, system_prompt_text=None):
    chat = []
    if isinstance(prompt_text, str):
        if system_prompt_text:
            chat.append({"role": "system", "content": system_prompt_text})
        if prompt_text:
            chat.append({"role": "user", "content": prompt_text})
        encoded = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
    else:
        encoded = tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)
    return _chat_template_ids(encoded)


def get_padded_prefill_len(seq_len: int) -> int:
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    return 2 ** (seq_len - 1).bit_length()
