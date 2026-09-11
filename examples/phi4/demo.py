# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate text with the public phi4 model API."""

from __future__ import annotations

import argparse
import os

from examples.common.runtime import mesh_device_parameters_from_env, open_mesh_device
from tt_transformers.models.phi4.hf_generator import DEFAULT_HF_MODEL, from_pretrained


def run(mesh_device, messages, *, hf_model=None, max_new_tokens=40, max_seq_len=2048):
    hf_model = hf_model or os.environ.get("HF_MODEL") or DEFAULT_HF_MODEL
    model = from_pretrained(
        mesh_device,
        hf_model=hf_model,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )
    try:
        inputs = model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        continuation = outputs[:, inputs["input_ids"].shape[1] :]
        return model.tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]
    finally:
        model.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Explain paged attention briefly.")
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args(argv)
    messages = [{"role": "user", "content": args.prompt}]
    with open_mesh_device(mesh_device_parameters_from_env()) as mesh_device:
        text = run(
            mesh_device,
            messages,
            hf_model=args.hf_model,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
        )
    print(text)


if __name__ == "__main__":
    main()
