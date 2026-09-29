# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tt_transformers.models.qwen38.weights import resolve_hf_weights


@pytest.mark.host
@pytest.mark.model
def test_resolve_hf_weights_threads_explicit_revision(monkeypatch):
    calls = []
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("CI", raising=False)

    def fake_snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        return "/cache/snapshots/deadbeef"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    assert resolve_hf_weights("org/model", "deadbeef") == "/cache/snapshots/deadbeef"
    assert calls == [("org/model", {"revision": "deadbeef", "local_files_only": False})]


@pytest.mark.host
@pytest.mark.model
@pytest.mark.parametrize("offline_variable", ["HF_HUB_OFFLINE", "CI"])
def test_resolve_hf_weights_respects_offline_policy(monkeypatch, offline_variable):
    calls = []
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv(offline_variable, "1" if offline_variable == "HF_HUB_OFFLINE" else "true")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, **kwargs: calls.append((repo, kwargs)) or "/cache/snapshots/deadbeef",
    )

    assert resolve_hf_weights("org/model", "deadbeef") == "/cache/snapshots/deadbeef"
    assert calls == [("org/model", {"revision": "deadbeef", "local_files_only": True})]


@pytest.mark.host
@pytest.mark.model
def test_resolve_hf_weights_accepts_embedded_revision(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, **kwargs: calls.append((repo, kwargs)) or "/cache/snapshots/cafe",
    )
    assert resolve_hf_weights("org/model@cafe") == "/cache/snapshots/cafe"
    assert calls[0][0] == "org/model"
    assert calls[0][1]["revision"] == "cafe"


@pytest.mark.host
@pytest.mark.model
def test_resolve_hf_weights_preserves_local_directory(tmp_path):
    assert resolve_hf_weights(str(tmp_path), "ignored") == str(Path(tmp_path).resolve())
