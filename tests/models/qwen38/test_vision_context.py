# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tt_transformers.models.qwen38.vision.context import VisionContext


def _rope_setup():
    return SimpleNamespace(
        spatial_merge_size=2,
        inv_freq=torch.tensor([1.0, 0.1, 0.01]),
        mrope_section=[1, 1, 1],
        attention_scaling=1.0,
    )


def test_multi_image_context_maps_each_item_and_prepares_mrope():
    image_id = 99
    # Grids produce 4 and 2 packed tokens after 2x2 spatial merge.
    grids = torch.tensor([[1, 4, 4], [1, 2, 4]])
    tokens = torch.tensor([[10, image_id, image_id, image_id, image_id, 11, image_id, image_id, 12]])
    context = VisionContext.for_images(torch.zeros(6, 8), grids, image_token_id=image_id)

    context.prepare(tokens, _rope_setup())

    assert len(context.placeholder_mapping) == 2
    assert context.placeholder_mapping[0].embedding_start == 0
    assert context.placeholder_mapping[0].embedding_end == 4
    assert context.placeholder_mapping[1].embedding_start == 4
    assert context.placeholder_mapping[1].embedding_end == 6
    assert context.placeholder_positions("image").tolist() == [1, 2, 3, 4, 6, 7]
    assert context.mrope_cos.shape == (tokens.shape[1], 6)
    assert context.mrope_sin.shape == context.mrope_cos.shape
    assert isinstance(context.rope_delta, int)


def test_context_rejects_placeholder_or_embedding_count_mismatches():
    setup = _rope_setup()
    grid = torch.tensor([[1, 4, 4]])  # four merged tokens
    with pytest.raises(ValueError, match="placeholder count 3"):
        VisionContext.for_images(torch.zeros(4, 8), grid, image_token_id=99).prepare(
            torch.tensor([[99, 99, 99]]), setup
        )
    with pytest.raises(ValueError, match="embeddings have 3 rows"):
        VisionContext.for_images(torch.zeros(3, 8), grid, image_token_id=99).prepare(
            torch.tensor([[99, 99, 99, 99]]), setup
        )


def test_preparing_another_request_cannot_mutate_the_first_context():
    setup = _rope_setup()
    first = VisionContext.for_images(torch.zeros(1, 8), [[1, 2, 2]], image_token_id=99)
    second = VisionContext.for_images(torch.zeros(2, 8), [[1, 2, 4]], image_token_id=99)

    first.prepare(torch.tensor([[1, 99, 2]]), setup)
    first_cos = first.mrope_cos.clone()
    first_positions = first.placeholder_positions().clone()
    second.prepare(torch.tensor([[3, 99, 99, 4]]), setup)

    assert torch.equal(first.mrope_cos, first_cos)
    assert torch.equal(first.placeholder_positions(), first_positions)
    assert first.placeholder_mapping is not second.placeholder_mapping


def test_decode_rope_accepts_one_explicit_delta_per_spec_candidate_row():
    from tt_transformers.models.qwen38 import model as model_module

    fake = SimpleNamespace(
        args=SimpleNamespace(rope_head_dim=6, rope_theta=10000.0),
        _rope_delta_for=lambda: 999,
    )
    positions = torch.tensor([10, 11, 20, 21], dtype=torch.int32)
    deltas = torch.tensor([5, 5, -3, -3], dtype=torch.int32)

    cos, sin = model_module.Qwen36Model._rope_tp_cos_sin_decode_torch(fake, positions, rope_deltas=deltas)

    inv_freq = 1.0 / (
        fake.args.rope_theta ** (torch.arange(0, fake.args.rope_head_dim, 2).float() / fake.args.rope_head_dim)
    )
    freqs = torch.outer((positions + deltas).float(), inv_freq)
    expected = torch.cat([freqs, freqs], dim=-1)
    assert torch.equal(cos, expected.cos().reshape(1, 4, 1, 6).to(torch.bfloat16))
    assert torch.equal(sin, expected.sin().reshape(1, 4, 1, 6).to(torch.bfloat16))


def test_vllm_gather_packs_four_images_in_declared_order():
    from tt_transformers.models.qwen38.qwen36_vllm import Qwen36ForCausalLM

    pixels = [torch.full((i + 1, 3), i, dtype=torch.float32) for i in range(4)]
    grids = [torch.tensor([1, 2, 2 * (i + 1)]) for i in range(4)]
    packed_pixels, packed_grids = Qwen36ForCausalLM._gather_user_visual(
        {"pixel_values": [pixels], "image_grid_thw": [grids]},
        "pixel_values",
        "image_grid_thw",
    )

    assert packed_pixels.shape == (10, 3)
    assert [int(packed_pixels[sum(range(i + 1)), 0]) for i in range(4)] == [0, 1, 2, 3]
    assert torch.equal(packed_grids, torch.stack(grids).to(torch.int32))


def test_vllm_gather_selects_visual_row_from_mixed_batch():
    from tt_transformers.models.qwen38.qwen36_vllm import Qwen36ForCausalLM

    pixels = [torch.ones((2, 3)), torch.full((1, 3), 2.0)]
    grids = [torch.tensor([1, 2, 4]), torch.tensor([1, 2, 2])]
    kwargs = {
        "pixel_values": [None, pixels, None],
        "image_grid_thw": [None, grids, None],
    }

    assert Qwen36ForCausalLM._gather_user_visual(kwargs, "pixel_values", "image_grid_thw", user=0) is None
    packed_pixels, packed_grids = Qwen36ForCausalLM._gather_user_visual(
        kwargs, "pixel_values", "image_grid_thw", user=1
    )
    assert packed_pixels.shape == (3, 3)
    assert torch.equal(packed_grids, torch.stack(grids).to(torch.int32))
    assert Qwen36ForCausalLM._has_visual(kwargs, "pixel_values")


def test_qwen_vision_warmup_uses_packed_patch_contract_once(monkeypatch):
    from tt_transformers.models.qwen38 import qwen36_vllm as vllm_module

    calls = []
    mesh = SimpleNamespace(num_program_cache_entries=lambda: 17)
    vision_cfg = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=16,
    )

    def get_image_features(pixels, grid):
        calls.append((pixels.clone(), grid.clone()))
        return SimpleNamespace(embeddings=torch.zeros(1, 8, dtype=torch.bfloat16))

    model = SimpleNamespace(
        vision_model=SimpleNamespace(dtype=torch.bfloat16),
        args=SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_cfg)),
        mesh_device=mesh,
        get_image_features=get_image_features,
    )
    wrapper = SimpleNamespace(model=[model])
    synced = []
    monkeypatch.setenv("QWEN36_ENABLE_VISION", "1")
    monkeypatch.setattr(vllm_module.ttnn, "synchronize_device", lambda device: synced.append(device))

    vllm_module.Qwen36ForCausalLM._warmup_qwen_vision(wrapper)
    vllm_module.Qwen36ForCausalLM._warmup_qwen_vision(wrapper)

    assert len(calls) == 2
    assert [pixels.shape for pixels, _ in calls] == [(1024, 1536), (3596, 1536)]
    assert all(pixels.dtype == torch.bfloat16 for pixels, _ in calls)
    assert torch.equal(calls[0][1], torch.tensor([[1, 32, 32]], dtype=torch.int32))
    assert torch.equal(calls[1][1], torch.tensor([[1, 58, 62]], dtype=torch.int32))
    assert synced == [mesh, mesh]
    assert wrapper._qwen_vision_warmed is True


@pytest.mark.parametrize(
    "pixels,grid,error",
    [
        (torch.zeros(4, 8), torch.tensor([1, 2]), "shape \\[N, 3\\]"),
        (torch.zeros(4, 8), torch.tensor([[1, 0, 4]]), "strictly positive"),
        (torch.zeros(6, 8), torch.tensor([[1, 3, 2]]), "divisible"),
        (torch.zeros(3, 8), torch.tensor([[1, 2, 2]]), "requires 4"),
    ],
)
def test_vision_wrapper_rejects_malformed_packed_inputs_before_device(pixels, grid, error):
    from tt_transformers.models.qwen38.vision.model import DropInVisionTransformer

    fake = SimpleNamespace(spatial_merge_size=2)
    with pytest.raises(ValueError, match=error):
        DropInVisionTransformer.forward(fake, pixels, grid)


def test_vision_wrapper_rejects_unwarmed_patch_bucket_before_device(monkeypatch):
    from tt_transformers.models.qwen38.vision.model import DropInVisionTransformer

    fake = SimpleNamespace(spatial_merge_size=2)
    monkeypatch.setenv("QWEN36_VISION_MAX_PATCHES", "15")
    with pytest.raises(ValueError, match="pre-warmed vision limit"):
        DropInVisionTransformer.forward(fake, torch.zeros(16, 8), torch.tensor([[1, 4, 4]]))


def test_spec_short_prefill_threads_request_context_and_physical_slot(monkeypatch):
    from tt_transformers.models.qwen38 import model as model_module

    calls = []
    context = object()
    fake = SimpleNamespace(_chunked_chunk_size=2048)
    fake._build_request_rope = lambda tokens, vision, slot=0: calls.append(("rope", vision, slot, tokens.shape[1]))

    def masked(*args, **kwargs):
        calls.append(("masked", kwargs))
        return "logits", "hidden"

    fake.prefill_masked_bucket = masked
    monkeypatch.setattr(model_module.ttnn, "deallocate", lambda tensor: None)
    seen = []

    result = model_module.Qwen36Model._prefill_for_spec_b1(
        fake,
        torch.arange(17, dtype=torch.int32).reshape(1, -1),
        torch.zeros(1, 4, dtype=torch.int32),
        17,
        lambda hidden, start, valid: seen.append((hidden, start, valid)),
        vision_context=context,
        slot=3,
    )

    assert result == "logits"
    assert calls[0] == ("rope", context, 3, 17)
    assert calls[1][1]["vision_tokens"] is context
    assert calls[1][1]["vision_slot"] == 3
    assert seen == [("hidden", 0, 17)]


def test_spec_long_prefill_splices_vision_in_full_chunk_and_tail(monkeypatch):
    from tt_transformers.models.qwen38 import model as model_module

    context = object()
    calls = []
    fake = SimpleNamespace(_chunked_chunk_size=2048, device=object())
    fake._build_request_rope = lambda tokens, vision, slot=0: calls.append(("rope", vision, slot, tokens.shape[1]))
    fake._reset_gdn_state_for_new_sequence = lambda: calls.append(("reset",))
    fake._vis_row_offset_for = lambda tokens, start, vision: 11 if start else 0

    def stage(tokens, vision, offset):
        calls.append(("stage", tokens.shape[1], vision, offset))
        return [object()]

    fake._set_vision_merge = stage

    def full(*args, **kwargs):
        calls.append(("full", kwargs))
        return "full-hidden"

    fake._forward_prefill_chunk_masked_tp = full

    def masked(*args, **kwargs):
        calls.append(("tail", kwargs))
        return "logits", "tail-hidden"

    fake.prefill_masked_bucket = masked
    monkeypatch.setattr(model_module.ttnn, "deallocate", lambda tensor: None)
    monkeypatch.setattr(model_module.ttnn, "synchronize_device", lambda device: None)
    seen = []

    result = model_module.Qwen36Model._prefill_for_spec_b1(
        fake,
        torch.arange(2050, dtype=torch.int32).reshape(1, -1),
        torch.zeros(1, 40, dtype=torch.int32),
        2050,
        lambda hidden, start, valid: seen.append((hidden, start, valid)),
        vision_context=context,
        slot=5,
    )

    assert result == "logits"
    assert ("stage", 2048, context, 0) in calls
    full_call = next(call for call in calls if call[0] == "full")
    assert full_call[1]["vision_tokens"] is context
    tail_call = next(call for call in calls if call[0] == "tail")
    assert tail_call[1]["vision_tokens"] is context
    assert tail_call[1]["vis_row_offset"] == 11
    assert tail_call[1]["vision_slot"] == 5
    assert seen == [("full-hidden", 0, 2048), ("tail-hidden", 2048, 2)]
