# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Plain generation helpers shared by the concrete text-model wrappers.

Inputs and returned sequences are CPU PyTorch tensors. TT staging, compilation,
traces, and KV storage remain owned by the composed executor. Greedy selection
can use its device sampler; stochastic selection uses host filtering to preserve
HF temperature/top-k/top-p semantics without the device sampler's candidate cap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import wraps
from numbers import Integral, Real
from threading import Lock

import torch

_UNSET = object()
# Transformers 5.12 stores unspecified GenerationConfig fields as None and
# applies these defaults at generation time. Keep the supported subset here;
# legacy wrapper defaults are used only without checkpoint generation metadata.
_HF_SAMPLING_DEFAULTS = {"do_sample": False, "temperature": 1.0, "top_k": 50, "top_p": 1.0}


def _exclusive_operation(function):
    @wraps(function)
    def locked(llm, *args, **kwargs):
        lock = vars(llm).setdefault("_hf_operation_lock", Lock())
        if not lock.acquire(blocking=False):
            raise RuntimeError(f"cannot {function.__name__}: another operation on this model is active")
        try:
            return function(llm, *args, **kwargs)
        finally:
            lock.release()

    return locked


def _validate_checkpoint_options(llm):
    # Checkpoint defaults must not silently request behavior outside this API.
    neutral = {
        "num_beams": (1,),
        "num_beam_groups": (1,),
        "num_return_sequences": (1,),
        "repetition_penalty": (1.0,),
        "encoder_repetition_penalty": (1.0,),
        "no_repeat_ngram_size": (0,),
        "encoder_no_repeat_ngram_size": (0,),
        "min_length": (0,),
        "min_new_tokens": (None, 0),
        "max_time": (None,),
        "bad_words_ids": (None,),
        "force_words_ids": (None,),
        "forced_bos_token_id": (None,),
        "forced_eos_token_id": (None,),
        "suppress_tokens": (None,),
        "begin_suppress_tokens": (None,),
        "stop_strings": (None,),
        "sequence_bias": (None,),
        "watermarking_config": (None,),
        "exponential_decay_length_penalty": (None,),
        "constraints": (None,),
        "return_dict_in_generate": (False,),
        "output_scores": (False,),
        "output_logits": (False,),
        "output_attentions": (False,),
        "output_hidden_states": (False,),
        "typical_p": (1.0,),
        "epsilon_cutoff": (0.0,),
        "eta_cutoff": (0.0,),
        "min_p": (None,),
        "top_h": (None,),
        "penalty_alpha": (None,),
    }
    config = getattr(llm, "_hf_generation_config", None)
    if config is None:
        return
    unsupported = []
    for name, allowed in neutral.items():
        value = config.get(name, allowed[0]) if isinstance(config, Mapping) else getattr(config, name, allowed[0])
        if value is not None and value not in allowed:
            unsupported.append(name)
    if unsupported:
        raise ValueError(f"Unsupported checkpoint generation option(s): {', '.join(sorted(unsupported))}")


def _default(llm, name, fallback=None, *, legacy=None):
    for index, config in enumerate(
        (getattr(llm, "_hf_generation_config", None), getattr(llm, "generation_config", None))
    ):
        if config is None:
            continue
        for key in (name, legacy) if legacy else (name,):
            value = config.get(key) if isinstance(config, Mapping) else getattr(config, key, None)
            if value is not None:
                return value
        if index == 0 and name in _HF_SAMPLING_DEFAULTS:
            return _HF_SAMPLING_DEFAULTS[name]
    return fallback


def _integer(name, value, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _token_ids(value, name, vocab_size):
    if value is None:
        return ()
    if isinstance(value, Integral):
        value = (value,)
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a token ID, a list of token IDs, or None")
    result = tuple(dict.fromkeys(_integer(name, token) for token in value))
    if any(token >= vocab_size for token in result):
        raise ValueError(f"{name} must contain IDs below vocabulary size {vocab_size}")
    return result


def _select(logits, *, batch_size, vocab_size, do_sample, temperature, top_k, top_p, active):
    if not isinstance(logits, torch.Tensor) or logits.ndim not in (2, 3):
        raise TypeError("executor must return host logits with shape [batch, 1, vocab] or [batch, vocab]")
    if logits.shape[0] < batch_size or logits.shape[-1] < vocab_size:
        raise ValueError("executor logits do not cover the requested batch and vocabulary")
    scores = (logits[:, -1, :] if logits.ndim == 3 else logits)[:batch_size, :vocab_size].float().cpu()
    selected = torch.zeros(batch_size, dtype=torch.long)
    scores = scores[active]
    if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any(dim=-1).all():
        raise ValueError("executor returned invalid logits for an active sequence")
    if not do_sample:
        selected[active] = scores.argmax(dim=-1)
        return selected
    scores = scores / temperature
    if top_k:
        cutoff = scores.topk(min(top_k, vocab_size), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    if top_p < 1.0:
        sorted_scores, indices = scores.sort(dim=-1)
        remove = sorted_scores.softmax(dim=-1).cumsum(dim=-1) <= (1.0 - top_p)
        remove[:, -1] = False
        scores = scores.masked_fill(torch.zeros_like(remove).scatter(1, indices, remove), -torch.inf)
    selected[active] = torch.multinomial(scores.softmax(dim=-1), num_samples=1).squeeze(1)
    return selected


def _prepare_execution(llm, tokens, lengths, page_table, *, device_greedy, decode):
    executor = llm.executor
    if not hasattr(llm, "_hf_kv_cache"):
        # Existing callers may already have allocated through this same executor.
        if executor.kv_cache_manager.bound_context is None:
            llm._hf_kv_cache = executor.allocate_kv_cache()
        else:
            llm._hf_kv_cache = None  # The executor can use its bound cache directly.
    cache = llm._hf_kv_cache
    sampling = None
    if device_greedy:
        from tt_transformers.sampling.sampling_params import SamplingParams

        sampling = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    prefill = dict(
        tokens=tokens,
        prompt_lens=lengths,
        page_table=page_table[: len(lengths)],
        empty_slots=list(range(len(lengths))),
        kv_cache=cache,
        sampling_params=sampling,
    )
    positions = torch.full((page_table.shape[0],), -1, dtype=torch.long)
    positions[: len(lengths)] = lengths
    decode_args = dict(
        tokens=torch.zeros(page_table.shape[0], dtype=torch.long),
        start_pos=positions,
        page_table=page_table,
        kv_cache=cache,
        sampling_params=sampling,
        reset_batch=True,
    )
    trace = executor.config.trace
    needs_warmup = trace.mode != "none" and not getattr(llm, "_hf_warmed_up", False)
    compile_target = {"execution": executor.eager_execution} if needs_warmup else {}
    # Sampling buffers and decode programs must exist before sampled prefill.
    # First compile eagerly so configured trace warmup can register the preferred
    # top-k prefill workspace before any argmax alias sharing its hidden trace.
    if device_greedy:
        executor.compile_decode(**decode_args, **compile_target)
    executor.compile_prefill(**prefill, **compile_target)
    if not device_greedy and decode:
        executor.compile_decode(**decode_args, **compile_target)
    if needs_warmup:
        # The warmup ledger compiles all configured programs before trace capture.
        # It always includes logits, allowing host sampling on a sampled executor.
        can_sample = executor.config.device_sampling_enabled
        prefill_args = dict(kv_cache=cache, can_sample_on_device=can_sample)
        warmup_decode = dict(**prefill_args, max_batch_size=page_table.shape[0], num_blocks=page_table.shape[1])
        executor.warmup_model_prefill(enable_trace=False, **prefill_args)
        executor.warmup_model_decode(enable_trace=False, **warmup_decode)
        if trace.prefill_enabled:
            executor.warmup_model_prefill(enable_trace=True, **prefill_args)
        executor.compile_prefill(**prefill)
        if decode or device_greedy:
            executor.compile_decode(**decode_args)
        if trace.decode_enabled:
            executor.warmup_model_decode(enable_trace=True, **warmup_decode)
        llm._hf_warmed_up = True
    return prefill, sampling


def _reset_request(llm):
    executor = llm.executor
    controller = getattr(executor, "sampling_state_controller", None)
    if controller is not None:
        controller.reset(executor.sampling_state)
    # Native Sampling1D without a state controller resets admission at the first
    # decode via reset_batch=True. Host requests never retain token histories.


@_exclusive_operation
@torch.inference_mode()
def generate(
    llm,
    input_ids,
    attention_mask=None,
    *,
    max_new_tokens=_UNSET,
    do_sample=_UNSET,
    temperature=_UNSET,
    top_k=_UNSET,
    top_p=_UNSET,
    eos_token_id=_UNSET,
    pad_token_id=_UNSET,
    **unsupported_kwargs,
):
    """Return prompts followed by at most ``max_new_tokens`` generated tokens.

    Accept rank-two CPU integer tensors and contiguous left/right padding masks.
    Without a mask, a configured pad ID distinct from EOS identifies padding.
    Explicit ``eos_token_id=None`` disables EOS stopping. Per-call arguments do
    not mutate checkpoint defaults. Completed rows receive the pad ID while
    other rows continue. A zero token budget returns a copy of the input.

    Supported options are the parameters above; beam search, streaming, custom
    processors, cache continuation, and output objects are not implemented.
    One request may run at a time per loaded model. Call ``cleanup()`` when done.
    """
    if unsupported_kwargs:
        raise TypeError(f"Unsupported generate option(s): {', '.join(sorted(unsupported_kwargs))}")
    _validate_checkpoint_options(llm)
    if getattr(llm, "_hf_terminal", False):
        raise RuntimeError("model is terminal after cleanup; load a new model")
    executor = getattr(llm, "executor", None)
    if executor is None:
        raise RuntimeError("generate requires the executor attached by hf_generator.from_pretrained()")
    if executor.terminal:
        raise RuntimeError("executor is terminal; load a new model")
    if getattr(llm, "_hf_generating", False):
        raise RuntimeError("concurrent generate calls on one model are not supported")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a rank-two CPU integer tensor")
    if input_ids.device.type != "cpu" or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be a CPU int32 or int64 tensor; the executor handles TT transfer")
    batch_size, width = input_ids.shape
    capacity = int(llm.model.config.max_batch_size)
    vocab_size = int(llm.model.config.vocab_size)
    context = min(int(llm.model.config.max_seq_len), int(llm.runtime_config.max_context_len))
    if batch_size < 1 or width < 1:
        raise ValueError("input_ids must contain at least one nonempty prompt")
    if batch_size > capacity:
        raise ValueError(f"batch size {batch_size} exceeds model capacity {capacity}")
    if torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size):
        raise ValueError(f"input_ids must be within vocabulary size {vocab_size}")
    if max_new_tokens is _UNSET:
        max_new_tokens = _default(llm, "max_new_tokens", 128, legacy="max_decode_tokens")
    max_new_tokens = _integer("max_new_tokens", max_new_tokens)
    default_temperature = _default(llm, "temperature", 1.0)
    if do_sample is _UNSET:
        do_sample = _default(llm, "do_sample", default_temperature > 0)
    if not isinstance(do_sample, bool):
        raise ValueError("do_sample must be bool")
    if temperature is _UNSET:
        temperature = default_temperature if default_temperature > 0 else 1.0
    if isinstance(temperature, bool) or not isinstance(temperature, Real) or not math.isfinite(temperature):
        raise ValueError("temperature must be a finite number")
    if do_sample and temperature <= 0:
        raise ValueError("temperature must be positive when do_sample=True")
    if top_k is _UNSET:
        top_k = _default(llm, "top_k", 50)
    top_k = _integer("top_k", top_k)
    if top_p is _UNSET:
        top_p = _default(llm, "top_p", 1.0)
    if isinstance(top_p, bool) or not isinstance(top_p, Real) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if eos_token_id is _UNSET:
        eos_token_id = _default(
            llm, "eos_token_id", getattr(llm.tokenizer, "eos_token_id", None), legacy="stop_token_ids"
        )
        if eos_token_id == ():
            eos_token_id = getattr(llm.tokenizer, "eos_token_id", None)
    eos_ids = _token_ids(eos_token_id, "eos_token_id", vocab_size)
    if pad_token_id is _UNSET:
        pad_token_id = _default(llm, "pad_token_id", getattr(llm.tokenizer, "pad_token_id", None))
    input_pad_token_id = pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_ids[0] if eos_ids else 0
    pad_token_id = _integer("pad_token_id", pad_token_id)
    if pad_token_id >= vocab_size:
        raise ValueError(f"pad_token_id must be below vocabulary size {vocab_size}")
    if attention_mask is None:
        mask = (
            input_ids != input_pad_token_id
            if input_pad_token_id is not None and input_pad_token_id not in eos_ids
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
    else:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must be a CPU tensor with the same shape as input_ids")
        if attention_mask.device.type != "cpu" or not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only zero and one on CPU")
        mask = attention_mask.bool()
    lengths = mask.sum(dim=-1)
    if torch.any(lengths == 0):
        raise ValueError("every input row must contain at least one unmasked prompt token")
    if torch.any(lengths + max_new_tokens > context):
        raise ValueError(f"prompt length plus max_new_tokens exceeds context capacity {context}")
    tokens = torch.full((batch_size, int(lengths.max())), pad_token_id, dtype=torch.long)
    for row in range(batch_size):
        indices = mask[row].nonzero().flatten()
        if int(indices[-1] - indices[0] + 1) != int(lengths[row]):
            raise ValueError("attention_mask supports contiguous prompts with left or right padding only")
        tokens[row, : lengths[row]] = input_ids[row, mask[row]]
    if max_new_tokens == 0:
        return input_ids.to(torch.long).clone()
    block_size = executor.paged_kv_cache_config.block_size
    blocks = (lengths + max_new_tokens + block_size - 1) // block_size
    available = executor.paged_kv_cache_config.num_blocks
    if available is None or int(blocks.sum()) > available:
        raise ValueError(
            f"request requires {int(blocks.sum())} KV blocks, but executor has {available} resolved blocks"
        )
    page_table = torch.zeros((capacity, int(blocks.max())), dtype=torch.int32)
    offset = 0
    for row, count in enumerate(blocks.tolist()):
        page_table[row, :count] = torch.arange(offset, offset + count, dtype=torch.int32)
        offset += count
    device_greedy = executor.config.device_sampling_enabled and not do_sample
    llm._hf_generating = True
    try:
        _reset_request(llm)
        prefill, sampling = _prepare_execution(
            llm, tokens, lengths, page_table, device_greedy=device_greedy, decode=max_new_tokens > 1
        )
        output = executor.prefill_forward(**prefill)
        finished = torch.zeros(batch_size, dtype=torch.bool)
        continuation = []
        for step in range(max_new_tokens):
            if sampling is not None:
                if not isinstance(output, tuple):
                    raise TypeError("device sampling must return (token_ids, log_probs)")
                next_tokens = output[0].reshape(-1)[:batch_size].to(device="cpu", dtype=torch.long)
                if (
                    next_tokens.numel() != batch_size
                    or torch.any(next_tokens[~finished] < 0)
                    or torch.any(next_tokens[~finished] >= vocab_size)
                ):
                    raise ValueError("device sampler returned invalid token IDs")
            else:
                logits = output[0] if isinstance(output, tuple) else output
                next_tokens = _select(
                    logits,
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    active=~finished,
                )
            next_tokens = next_tokens.masked_fill(finished, pad_token_id)
            continuation.append(next_tokens)
            for eos in eos_ids:
                finished |= next_tokens == eos
            if finished.all() or step + 1 == max_new_tokens:
                break
            decode_tokens = torch.full((capacity,), pad_token_id, dtype=torch.long)
            decode_tokens[:batch_size] = next_tokens
            positions = torch.full((capacity,), -1, dtype=torch.long)
            positions[:batch_size] = torch.where(finished, -1, lengths + step)
            output = executor.decode_forward(
                decode_tokens,
                positions,
                page_table=page_table,
                kv_cache=llm._hf_kv_cache,
                sampling_params=sampling,
                reset_batch=True,
                read_from_device=True,
            )
        result = torch.cat((input_ids.to(torch.long), torch.stack(continuation, dim=1)), dim=1)
        _reset_request(llm)
        return result
    except BaseException as primary:
        try:
            _reset_request(llm)
        except BaseException as failure:
            primary.cleanup_failures = (*getattr(primary, "cleanup_failures", ()), failure)
        raise
    finally:
        llm._hf_generating = False
    # Request histories are reset before and after each admission. No
    # public cache escapes; overwritten prompt pages hide all previous KV data.


@_exclusive_operation
def cleanup(llm):
    """Release the executor before model tensors without closing its borrowed mesh."""
    if getattr(llm, "_hf_generating", False):
        raise RuntimeError("cannot clean up a model during generate()")
    llm._hf_terminal = True
    if getattr(llm, "_hf_cleaned_up", False):
        return
    executor = getattr(llm, "executor", None)
    if executor is not None:
        # Failed executor release can retain KV tensors still referenced by the
        # model. Only that owner may retry their release; traversing the model
        # graph now would free borrowed tensors behind the manager's back.
        executor.cleanup()
    model = getattr(llm, "model", None)
    if model is not None:
        from tt_transformers.device_utils import cleanup_object_graph

        cleanup_object_graph(model)
    llm._hf_cleaned_up = True
    llm._hf_kv_cache = None
