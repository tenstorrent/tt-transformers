# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device grammar sampling: host checks of the packed mask and Sampling1D's grammar members.

The masks here are synthetic: hand-built allowed-token sets, with no grammar engine.
Device checks of the same members are in test_sampling_grammar_device.py.
"""

from types import SimpleNamespace

import pytest
import torch

from tt_transformers.modules.sampling import sampling_1d
from tt_transformers.modules.sampling.sampling_1d import Sampling1D
from tt_transformers.sampling.grammar_bitmask import (
    all_allowed_grammar_bitmask,
    format_grammar_bitmask,
    grammar_layout_error,
    packed_grammar_width,
    unpack_grammar_bitmask,
)


def packed_mask(rows: int, vocab_size: int, allowed: dict[int, list[int]]) -> torch.Tensor:
    """A vLLM-style mask: rows in ``allowed`` allow only those tokens, the rest allow all."""
    width = packed_grammar_width(vocab_size)
    mask = torch.full((rows, width), -1, dtype=torch.int32)
    for row, token_ids in allowed.items():
        words = torch.zeros(width, dtype=torch.int64)
        for token_id in token_ids:
            words[token_id // 32] |= 1 << (token_id % 32)
        mask[row] = words.to(torch.int32)
    return mask


# ---------------------------------------------------------------------------
# format_grammar_bitmask
# ---------------------------------------------------------------------------


@pytest.mark.host
def test_padding_blocks_ids_past_the_vocab_and_frees_unused_rows():
    grammar = torch.tensor([[-1, (1 << 17) | (1 << 18)]], dtype=torch.int32)

    formatted = format_grammar_bitmask(grammar, vocab_size=50, padded_vocab_size=96, max_batch_size=4)

    assert formatted.dtype == torch.int32
    assert formatted.shape == (4, 3)
    # Token 50 (bit 18 of word 1) is past the vocab; word 2 is padding.
    assert formatted[0].tolist() == [-1, 1 << 17, 0]
    assert torch.all(formatted[1:] == -1)


@pytest.mark.host
def test_formatting_does_not_modify_the_callers_mask():
    grammar = torch.tensor([[-1, -1]], dtype=torch.int32)

    format_grammar_bitmask(grammar, vocab_size=50, padded_vocab_size=64, max_batch_size=2)

    assert grammar.tolist() == [[-1, -1]]


@pytest.mark.host
def test_sign_bit_word_round_trips():
    """Token 32w + 31 sets the int32 sign bit; it must stay allowed."""
    vocab_size = 64
    grammar = packed_mask(1, vocab_size, {0: [31, 63]})

    formatted = format_grammar_bitmask(grammar, vocab_size=vocab_size, padded_vocab_size=64, max_batch_size=1)

    allowed = unpack_grammar_bitmask(formatted, vocab_size)
    assert torch.nonzero(allowed[0]).flatten().tolist() == [31, 63]


@pytest.mark.host
@pytest.mark.parametrize(
    ("grammar", "error"),
    [
        pytest.param(torch.zeros((1, 2), dtype=torch.int32), "no allowed token", id="empty-row"),
        pytest.param(torch.tensor([[0, 1 << 18]], dtype=torch.int32), "no allowed token", id="only-past-vocab"),
        pytest.param(torch.ones((1, 1), dtype=torch.int32), "packed width", id="wrong-width"),
        pytest.param(torch.ones((5, 2), dtype=torch.int32), "exceeds sampler batch", id="too-many-rows"),
        pytest.param(torch.ones((2,), dtype=torch.int32), "must have shape", id="one-dimensional"),
    ],
)
def test_invalid_masks_are_rejected(grammar, error):
    with pytest.raises(ValueError, match=error):
        format_grammar_bitmask(grammar, vocab_size=50, padded_vocab_size=96, max_batch_size=4)


@pytest.mark.host
def test_mask_dtype_is_checked():
    with pytest.raises(TypeError, match="torch.int32"):
        format_grammar_bitmask(
            torch.ones((1, 2), dtype=torch.int64), vocab_size=50, padded_vocab_size=96, max_batch_size=4
        )


@pytest.mark.host
def test_unaligned_padded_vocab_is_rejected():
    with pytest.raises(ValueError, match="multiple of 32"):
        format_grammar_bitmask(
            torch.ones((1, 2), dtype=torch.int32), vocab_size=50, padded_vocab_size=60, max_batch_size=4
        )


@pytest.mark.host
def test_unpack_matches_the_allowed_sets():
    vocab_size = 100
    allowed = {0: [0, 33, 99], 2: [64]}
    grammar = packed_mask(3, vocab_size, allowed)

    table = unpack_grammar_bitmask(grammar, vocab_size)

    assert torch.nonzero(table[0]).flatten().tolist() == [0, 33, 99]
    assert bool(table[1].all())
    assert torch.nonzero(table[2]).flatten().tolist() == [64]


@pytest.mark.host
def test_all_allowed_mask_shape():
    mask = all_allowed_grammar_bitmask(32, 128256)

    assert mask.shape == (32, 4008)
    assert torch.all(mask == -1)


# ---------------------------------------------------------------------------
# grammar_layout_error
# ---------------------------------------------------------------------------


@pytest.mark.host
@pytest.mark.parametrize(
    ("padded_vocab_size", "num_vocab_shards"),
    [
        pytest.param(128256, 1, id="llama-1x1"),
        pytest.param(128256, 2, id="llama-1x2"),
        pytest.param(152064, 8, id="qwen25-1x8"),
    ],
)
def test_word_aligned_layouts_are_accepted(padded_vocab_size, num_vocab_shards):
    assert grammar_layout_error(padded_vocab_size, num_vocab_shards) is None


@pytest.mark.host
@pytest.mark.parametrize(
    ("padded_vocab_size", "num_vocab_shards"),
    [
        # 3216 tokens per device is 100.5 mask words.
        pytest.param(25728, 8, id="half-word-shard"),
        pytest.param(32016, 1, id="unaligned-single-device"),
    ],
)
def test_word_straddling_layouts_are_rejected(padded_vocab_size, num_vocab_shards):
    assert "whole 32-token mask words" in grammar_layout_error(padded_vocab_size, num_vocab_shards)


# ---------------------------------------------------------------------------
# Sampling1D grammar members, with fake device tensors
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, name, shape=(), dtype="bf16"):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"FakeTensor({self.name})"


class FakeMesh:
    def __init__(self, shape):
        self.shape = shape

    def get_num_devices(self):
        return self.shape[0] * self.shape[1]


def _sampler(*, mesh_shape=(1, 2), vocab_size=128256, valid_vocab_size=None, max_batch_size=32, argmax=True):
    sampler = object.__new__(Sampling1D)
    sampler.config = SimpleNamespace(
        mesh_device=FakeMesh(mesh_shape),
        vocab_size=vocab_size,
        valid_vocab_size=vocab_size if valid_vocab_size is None else valid_vocab_size,
        max_batch_size=max_batch_size,
        allow_force_argmax=argmax,
        sub_core_grids=None,
    )
    sampler.load_device_buffers = lambda: None
    sampler._log_probs_calculator = SimpleNamespace(set_log_probs_mode=lambda enabled: None)
    return sampler


@pytest.fixture
def fake_ttnn(monkeypatch):
    """Record device calls instead of running them."""
    calls = []
    ttnn = sampling_1d.ttnn

    def record(name, result=None):
        def op(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result(*args, **kwargs) if callable(result) else result

        return op

    monkeypatch.setattr(ttnn, "bfloat16", "bf16", raising=False)
    monkeypatch.setattr(
        ttnn,
        "from_torch",
        record("from_torch", lambda tensor, **kw: FakeTensor("from_torch", tuple(tensor.shape))),
    )
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", record("copy"))
    monkeypatch.setattr(ttnn, "deallocate", record("deallocate"))
    monkeypatch.setattr(ttnn, "ShardTensor2dMesh", record("shard_mapper", lambda *a, **kw: ("shard", kw["dims"])))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", record("replicate_mapper", "replicate"))
    monkeypatch.setattr(ttnn, "reshape", record("reshape", lambda t, shape, **_: FakeTensor("reshape", shape)))
    for name in ("bitwise_right_shift", "bitwise_and", "to_layout", "typecast", "subtract", "multiply", "add"):
        monkeypatch.setattr(ttnn, name, record(name, lambda *a, _n=name, **kw: FakeTensor(_n, a[0].shape)))
    return calls


def _names(calls):
    return [name for name, _args, _kwargs in calls]


@pytest.mark.host
def test_enable_allocates_an_all_allowed_mask_sharded_like_the_logits(fake_ttnn):
    sampler = _sampler(mesh_shape=(1, 2), vocab_size=128256)

    sampler.enable_device_grammar()
    sampler.enable_device_grammar()  # idempotent

    assert sampler.device_grammar_enabled
    from_torch = [(args, kwargs) for name, args, kwargs in fake_ttnn if name == "from_torch"]
    assert len(from_torch) == 2
    (mask,), mask_kwargs = from_torch[0]
    assert mask.shape == (32, 4008) and torch.all(mask == -1)
    assert mask_kwargs["mesh_mapper"] == ("shard", (None, 1))
    (shift,), shift_kwargs = from_torch[1]
    assert shift.tolist() == list(range(32))
    assert shift_kwargs["mesh_mapper"] == "replicate"


@pytest.mark.host
def test_enable_rejects_a_word_straddling_layout_before_allocating(fake_ttnn):
    sampler = _sampler(mesh_shape=(1, 8), vocab_size=25728)

    with pytest.raises(ValueError, match="whole 32-token mask words"):
        sampler.enable_device_grammar()

    assert not sampler.device_grammar_enabled
    assert fake_ttnn == []


@pytest.mark.host
def test_update_validates_before_copying(fake_ttnn):
    sampler = _sampler(vocab_size=128, valid_vocab_size=100, max_batch_size=32)
    sampler.enable_device_grammar()
    fake_ttnn.clear()

    with pytest.raises(ValueError, match="no allowed token"):
        sampler.update_grammar_bitmask(torch.zeros((2, 4), dtype=torch.int32))
    assert fake_ttnn == []

    sampler.update_grammar_bitmask(packed_mask(2, 100, {1: [99]}))
    assert _names(fake_ttnn) == ["shard_mapper", "from_torch", "copy"]
    (formatted,), _ = [(a, k) for n, a, k in fake_ttnn if n == "from_torch"][0]
    assert formatted.shape == (32, 4)
    assert formatted[0].tolist()[:3] == [-1, -1, -1]
    assert torch.nonzero(unpack_grammar_bitmask(formatted, 128)[1]).flatten().tolist() == [99]
    assert torch.all(formatted[2:] == -1)


@pytest.mark.host
def test_update_with_none_stages_an_all_allowed_mask(fake_ttnn):
    sampler = _sampler()
    sampler.enable_device_grammar()
    fake_ttnn.clear()

    sampler.update_grammar_bitmask(None)

    (formatted,), _ = [(a, k) for n, a, k in fake_ttnn if n == "from_torch"][0]
    assert torch.all(formatted == -1)
    assert "copy" in _names(fake_ttnn)


@pytest.mark.host
def test_update_requires_enable():
    with pytest.raises(RuntimeError, match="enable_device_grammar"):
        _sampler().update_grammar_bitmask(None)


@pytest.mark.host
def test_mask_unpacks_to_the_local_vocab_shard_out_of_place(fake_ttnn):
    sampler = _sampler(mesh_shape=(1, 8), vocab_size=32768)
    sampler._grammar_bitmask = FakeTensor("mask", (32, 128))
    sampler._grammar_shift = FakeTensor("shift", (32,))
    logits = FakeTensor("logits", (1, 1, 32, 4096))

    masked = sampler._apply_grammar_bitmask(logits)

    reshapes = [args[1] for name, args, _ in fake_ttnn if name == "reshape"]
    assert reshapes == [(32, 128, 1), (1, 1, 32, 4096)]
    assert masked.name == "add"
    add_args = [args for name, args, _ in fake_ttnn if name == "add"][0]
    assert add_args[0] is logits
    freed = [args[0] for name, args, _ in fake_ttnn if name == "deallocate"]
    assert logits not in freed and sampler._grammar_bitmask not in freed and sampler._grammar_shift not in freed


@pytest.mark.host
def test_mask_must_cover_the_logits(fake_ttnn):
    sampler = _sampler(mesh_shape=(1, 8), vocab_size=32768)
    sampler._grammar_bitmask = FakeTensor("mask", (32, 128))
    sampler._grammar_shift = FakeTensor("shift", (32,))

    with pytest.raises(ValueError, match="does not cover the logits"):
        sampler._apply_grammar_bitmask(FakeTensor("logits", (1, 1, 32, 8192)))
    assert fake_ttnn == []


@pytest.mark.host
def test_grammar_requires_enable_before_any_device_work():
    sampler = _sampler()

    with pytest.raises(RuntimeError, match="enable_device_grammar"):
        sampler.decode_forward(FakeTensor("logits"), apply_grammar=True)


@pytest.mark.host
@pytest.mark.parametrize("enable_log_probs", [True, [False, True]])
def test_grammar_rejects_logprobs(enable_log_probs):
    sampler = _sampler(argmax=False)
    sampler._grammar_bitmask = FakeTensor("mask", (32, 4008))

    with pytest.raises(ValueError, match="logprobs"):
        sampler.decode_forward(
            FakeTensor("logits"),
            k=object(),
            p=object(),
            temp=object(),
            enable_log_probs=enable_log_probs,
            apply_grammar=True,
        )


@pytest.mark.host
@pytest.mark.parametrize("path", ["argmax", "topk"])
def test_decode_samples_the_masked_copy_and_frees_it(fake_ttnn, path):
    sampler = _sampler(argmax=True)
    sampler._grammar_bitmask = FakeTensor("mask", (32, 4008))
    sampler._grammar_shift = FakeTensor("shift", (32,))
    masked = FakeTensor("masked")
    sampler._apply_grammar_bitmask = lambda logits: masked
    seen = []
    sampler._sample_argmax = lambda logits, out: seen.append(logits) or ("tokens", None)
    sampler._sample_topk = lambda logits, *rest: seen.append(logits) or ("tokens", None)
    params = {} if path == "argmax" else {"k": object(), "p": object(), "temp": object()}

    assert sampler.decode_forward(FakeTensor("logits"), apply_grammar=True, **params) == ("tokens", None)

    assert seen == [masked]
    assert [args[0] for name, args, _ in fake_ttnn if name == "deallocate"] == [masked]


@pytest.mark.host
def test_decode_without_grammar_is_unchanged(fake_ttnn):
    sampler = _sampler()
    logits = FakeTensor("logits")
    seen = []
    sampler._sample_argmax = lambda value, out: seen.append(value) or ("tokens", None)

    sampler.decode_forward(logits)

    assert seen == [logits]
    assert fake_ttnn == []


@pytest.mark.host
def test_release_frees_the_grammar_buffers(fake_ttnn):
    sampler = _sampler()
    mask, shift = FakeTensor("mask"), FakeTensor("shift")
    sampler._grammar_bitmask, sampler._grammar_shift = mask, shift
    sampler._log_probs_calculator = None
    sampler.config.index_offsets = None
    for name in ("invalid_vocab_mask", "invalid_vocab_tail_mask", "seeds", "user_ids"):
        setattr(sampler.config, name, None)

    sampler.release()

    assert [args[0] for name, args, _ in fake_ttnn if name == "deallocate"] == [mask, shift]
    assert not sampler.device_grammar_enabled
