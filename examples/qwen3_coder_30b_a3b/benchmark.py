# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Coder-30B-A3B accuracy and performance cases on a four-device Blackhole mesh.

token-accuracy  teacher-forced top-1 / top-5 against the Hugging Face reference (eager)
perf-batch1     TTFT and decode throughput for one user (traced, on-device top-k sampling)
perf-batch32    the same for 32 users
perf-prefill    TTFT for one user at 128, 1024 and 4096 prompt tokens (prefill traced at 128 and 1024)
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
from pathlib import Path

import torch
import ttnn
from loguru import logger

from examples.common.prompting import encode_prompt_hf
from examples.common.run_helpers import run_perf_benchmark, run_teacher_forcing, warmup_demo_executor
from examples.common.runtime import UnsupportedConfiguration, open_mesh_device
from tt_transformers.models.qwen3_coder_30b_a3b.hf_generator import (
    DEFAULT_HF_MODEL,
    EagerQwen3Coder30BA3BExecutor,
    TracedQwen3Coder30BA3BExecutor,
    _load_model,
    load_tokenizer,
)
from tt_transformers.sampling.sampling_params import SamplingParams

REFERENCE = Path("tests/assets/reference_outputs/qwen3_coder_30b_a3b/Qwen3-Coder-30B-A3B-Instruct.refpt")
PROMPTS = Path("examples/assets/sample_prompts/input_data_questions_prefill_128.json")
BOOK = Path("tests/assets/reference_inputs/tale-of-two-cities.txt.bz2")
PREFILL_LENGTHS = (128, 1024, 4096)
BLOCK_SIZE = 32
MAX_SEQ_LEN = 2048
DECODE_TOKENS = 128
# measured on Blackhole P300x2 (four p300c dies), ttnn 0.79.0, "performance" precision
MIN_ACCURACY = {"top1": 97.0, "top5": 99.0}  # measured 98.2 / 99.8
CASES = ("token-accuracy", "perf-batch1", "perf-batch32", "perf-prefill")


def mesh_parameters() -> dict:
    """The 1x4 Blackhole ring the model runs on, with room for the decode and prefill traces."""
    if os.environ.get("MESH_DEVICE", "").strip().upper() != "P150X4":
        raise UnsupportedConfiguration("Qwen3-Coder-30B-A3B runs on MESH_DEVICE=P150x4 (P150_X4 or P300_X2)")
    return {
        "mesh_shape": (1, 4),
        "trace_region_size": 512 << 20,
        "num_command_queues": 1,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    }


def _paged_kv(executor, model, mesh_device):
    args = model.model_args
    blocks_per_user = args.max_seq_len // BLOCK_SIZE
    shape = (
        blocks_per_user * args.max_batch_size,
        args.n_kv_heads // mesh_device.get_num_devices(),
        BLOCK_SIZE,
        args.head_dim,
    )
    kv_cache = executor.allocate_kv_cache(shape, torch.bfloat16, args.n_layers)
    page_table = torch.arange(blocks_per_user * args.max_batch_size, dtype=torch.int32)
    return kv_cache, page_table.reshape(args.max_batch_size, blocks_per_user)


def token_accuracy(model, mesh_device) -> dict:
    if not REFERENCE.is_file():
        raise UnsupportedConfiguration(f"reference file not found: {REFERENCE}")
    reference = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    tokens, prompt_len = reference["reference_tokens"].reshape(-1), int(reference["prompt_len"])
    executor = EagerQwen3Coder30BA3BExecutor(model, mesh_device)
    kv_cache, page_table = _paged_kv(executor, model, mesh_device)
    result = run_teacher_forcing(
        executor,
        prompt_tokens=tokens[:prompt_len].unsqueeze(0),
        reference_tokens=tokens,
        top5_tokens=reference["top5_tokens"],
        kv_cache=kv_cache,
        page_table=page_table,
    )
    top1, top5 = result.top1_accuracy() * 100, result.top5_accuracy() * 100
    logger.info(f"token accuracy: top1 {top1:.1f}%, top5 {top5:.1f}%")
    if top1 < MIN_ACCURACY["top1"] or top5 < MIN_ACCURACY["top5"]:
        raise AssertionError(f"accuracy top1 {top1:.1f}% / top5 {top5:.1f}% below {MIN_ACCURACY}")
    return {"top1": top1, "top5": top5}


def _traced(model, mesh_device, tokens, lens, sampling):
    """A traced executor, its paged KV cache and page table, warmed up on this prompt shape."""
    executor = TracedQwen3Coder30BA3BExecutor(model, mesh_device, ondevice_decode_loop=True)
    kv_cache, page_table = _paged_kv(executor, model, mesh_device)
    warmup_demo_executor(
        executor,
        kv_cache=kv_cache,
        page_table=page_table,
        prefill_compile_case=(tokens, lens),
        prefill_sampling_params=sampling,
        prefill_compile_execution=executor.traced_prefill_execution,
    )
    return executor, kv_cache, page_table


def _timed(executor, kv_cache, page_table, tokens, lens, sampling, decode_tokens):
    return run_perf_benchmark(
        executor,
        tokens=tokens,
        kv_cache=kv_cache,
        page_table=page_table,
        num_decode_tokens=decode_tokens,
        max_batch_size=tokens.shape[0],
        prompt_lens=lens,
        sampling_params=sampling,
    )


def performance(model, mesh_device, users: int) -> dict:
    tokenizer = load_tokenizer(os.environ.get("HF_MODEL", DEFAULT_HF_MODEL))
    sampling = SamplingParams(temperature=0.0, top_k=32, top_p=0.08)
    prompts = [entry["prompt"] for entry in json.loads(PROMPTS.read_text())]
    encoded = [list(encode_prompt_hf(tokenizer, prompt)) for prompt in (prompts * users)[:users]]
    lens = torch.tensor([len(ids) for ids in encoded])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokens = torch.tensor([ids + [pad] * (int(lens.max()) - len(ids)) for ids in encoded])
    executor, kv_cache, page_table = _traced(model, mesh_device, tokens, lens, sampling)
    result = _timed(executor, kv_cache, page_table, tokens, lens, sampling, DECODE_TOKENS)
    metrics = {"ttft_ms": result.ttft_ms, "decode_tok_s_u": result.tok_s_u, "decode_tok_s": result.tok_s}
    logger.info(f"performance, {users} users: {metrics}")
    return metrics


def prefill(model, mesh_device) -> dict:
    """TTFT at each prompt length, from the reference book; each length runs once untimed first."""
    tokenizer = load_tokenizer(os.environ.get("HF_MODEL", DEFAULT_HF_MODEL))
    sampling = SamplingParams(temperature=0.0, top_k=32, top_p=0.08)
    with bz2.open(BOOK, "rt", encoding="utf-8") as book:
        text = tokenizer.encode(book.read(), add_special_tokens=False)
    cases = [(torch.tensor([text[:length]]), torch.tensor([length])) for length in PREFILL_LENGTHS]
    executor, kv_cache, page_table = _traced(model, mesh_device, *cases[0], sampling)
    metrics = {}
    for (tokens, lens), length in zip(cases, PREFILL_LENGTHS):
        _timed(executor, kv_cache, page_table, tokens, lens, sampling, 4)
        metrics[f"ttft_ms_{length}"] = _timed(executor, kv_cache, page_table, tokens, lens, sampling, 4).ttft_ms
    logger.info(f"prefill: {metrics}")
    return metrics


def run_case(case: str, mesh_device, optimizations: str = "performance") -> dict:
    users = 32 if case == "perf-batch32" else 1
    llm = _load_model(
        mesh_device,
        hf_model=os.environ.get("HF_MODEL", DEFAULT_HF_MODEL),
        instruct=False,
        max_batch_size=users,
        max_seq_len=2 * PREFILL_LENGTHS[-1] if case == "perf-prefill" else MAX_SEQ_LEN,
        optimizations=optimizations,
    )
    try:
        if case == "token-accuracy":
            return token_accuracy(llm.model, mesh_device)
        if case == "perf-prefill":
            return prefill(llm.model, mesh_device)
        return performance(llm.model, mesh_device, users)
    finally:
        llm.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=CASES, default=CASES[0])
    parser.add_argument("--optimizations", choices=("performance", "accuracy"), default="performance")
    args = parser.parse_args(argv)
    with open_mesh_device(mesh_parameters()) as mesh_device:
        print(run_case(args.case, mesh_device, args.optimizations))


if __name__ == "__main__":
    main()
