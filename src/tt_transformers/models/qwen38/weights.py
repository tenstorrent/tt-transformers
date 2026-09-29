# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pinned Hugging Face weight resolution for Qwen."""

import os
from pathlib import Path


def resolve_hf_weights(spec: str, revision: str | None = None) -> str:
    local = Path(spec).expanduser()
    if local.is_dir():
        return str(local.resolve())
    if not revision and "@" in spec:
        spec, revision = spec.rsplit("@", 1)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        spec,
        revision=revision or None,
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("CI") == "true",
    )
