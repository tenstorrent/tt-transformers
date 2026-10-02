import pytest
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError
from transformers import AutoConfig

from tests.support.helpers import hf_config_or_skip, stable_model_seed

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from tt_transformers.tensor_utils import align_shape_to_tile


@pytest.mark.host
def test_stable_model_seed_deterministic() -> None:
    assert stable_model_seed("llama-3") == stable_model_seed("llama-3")


@pytest.mark.host
def test_stable_model_seed_distinct() -> None:
    assert stable_model_seed("llama-3") != stable_model_seed("mistral-7b")


@pytest.mark.host
def test_stable_model_seed_uint32_range() -> None:
    seed = stable_model_seed("llama-3")
    assert 0 <= seed < 2**32


@pytest.mark.host
def test_align_shape_to_tile():
    """Test align_shape_to_tile (replacement for deprecated ttnn.pad_to_tile_shape)."""

    assert align_shape_to_tile([1, 2, 3, 4]) == [1, 2, 32, 32]
    assert align_shape_to_tile([1, 1, 32, 32]) == [1, 1, 32, 32]
    assert align_shape_to_tile([1, 1, 33, 65]) == [1, 1, 64, 96]
    assert align_shape_to_tile([1, 1, 1, 1]) == [1, 1, 32, 32]
    assert align_shape_to_tile([1, 384, 49, 96]) == [1, 384, 64, 96]
    assert align_shape_to_tile([1, 9, 49, 768]) == [1, 9, 64, 768]
    assert align_shape_to_tile([2, 4, 64, 128]) == [2, 4, 64, 128]
    assert align_shape_to_tile((1, 1, 10, 10)) == [1, 1, 32, 32]
    assert align_shape_to_tile([7]) == [32]
    assert align_shape_to_tile([7, 50]) == [32, 64]
    assert align_shape_to_tile([2, 3, 4, 5, 6, 7]) == [2, 3, 4, 5, 32, 32]
    assert align_shape_to_tile([1, 1, 10, 10], tile_size=16) == [1, 1, 16, 16]
    assert align_shape_to_tile([1, 1, 17, 33], tile_size=16) == [1, 1, 32, 48]

    original = [1, 1, 10, 10]
    align_shape_to_tile(original)
    # ensure input is not mutated
    assert original == [1, 1, 10, 10]


def _hub_error(error_type: type[Exception]) -> Exception:
    import httpx

    if error_type is LocalEntryNotFoundError:
        return error_type("cannot reach the hub")
    response = httpx.Response(
        403 if error_type is GatedRepoError else 404, request=httpx.Request("GET", "https://hf.co")
    )
    return error_type("hub error", response=response)


def _raising_from_pretrained(cause: Exception):
    def from_pretrained(*_args, **_kwargs):
        raise OSError("config unavailable\nsecond line") from cause

    return from_pretrained


@pytest.mark.host
@pytest.mark.parametrize("error_type", [LocalEntryNotFoundError, GatedRepoError], ids=["offline", "gated"])
def test_hf_config_or_skip_skips_when_config_is_unavailable(monkeypatch, error_type) -> None:
    monkeypatch.setattr(AutoConfig, "from_pretrained", _raising_from_pretrained(_hub_error(error_type)))
    with pytest.raises(pytest.skip.Exception, match="org/model is not available: config unavailable$"):
        hf_config_or_skip("org/model")


@pytest.mark.host
def test_hf_config_or_skip_fails_on_an_unknown_model_id(monkeypatch) -> None:
    monkeypatch.setattr(AutoConfig, "from_pretrained", _raising_from_pretrained(_hub_error(RepositoryNotFoundError)))
    with pytest.raises(OSError, match="config unavailable"):
        hf_config_or_skip("org/misspelled")


@pytest.mark.host
def test_hf_config_or_skip_passes_arguments_through(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: calls.append((args, kwargs)) or "cfg")
    assert hf_config_or_skip("org/model", trust_remote_code=True) == "cfg"
    assert calls == [(("org/model",), {"trust_remote_code": True})]
