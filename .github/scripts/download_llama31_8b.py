"""Populate the pinned Llama checkpoint from CIv2's Large File Cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

LFC_BASE = (
    "http://large-file-cache.large-file-cache.svc.cluster.local/"
    "/mldata/model_checkpoints/pytorch/huggingface"
)
MANIFEST = Path(__file__).with_name("llama31-8b-checkpoint.json")


def verify_file(path: Path, entry: dict) -> str:
    size = path.stat().st_size
    if size != entry["size"]:
        raise ValueError(f"{path.name}: expected {entry['size']} bytes, received {size}")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            if not entry["lfs"]:
                git_blob.update(chunk)
    if entry["lfs"]:
        # HF exposes the Git blob ID of each LFS pointer even for gated files.
        # Reconstruct that pointer from the cached bytes to verify the revision.
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{sha256.hexdigest()}\nsize {size}\n"
        ).encode()
        git_blob = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer)
    if git_blob.hexdigest() != entry["git_blob_sha1"]:
        raise ValueError(f"{path.name}: LFC content does not match the pinned HF revision")
    return sha256.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    cache = args.hf_home / "hub" / ("models--" + manifest["model"].replace("/", "--"))
    snapshot = cache / "snapshots" / manifest["revision"]
    snapshot.mkdir(parents=True, exist_ok=True)
    report = {
        "model": manifest["model"],
        "revision": manifest["revision"],
        "source": LFC_BASE,
        "verified_files": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        name = entry["name"]
        url = f"{LFC_BASE}/{manifest['model']}/{name}"
        destination = snapshot / name
        print(f"Downloading {name} from CIv2 LFC", flush=True)
        subprocess.run(
            [
                "wget", "--no-proxy", "--tries=5", "--retry-connrefused",
                "--waitretry=10", "--timeout=30", "--progress=dot:giga",
                "--continue", "--output-document", str(destination), url,
            ],
            check=True,
            timeout=1200,
        )
        checksum = verify_file(destination, entry)
        report["verified_files"].append({"name": name, "size": entry["size"], "sha256": checksum})
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Verified {name}: sha256:{checksum}", flush=True)
    (cache / "refs").mkdir(exist_ok=True)
    (cache / "refs" / "main").write_text(manifest["revision"])
    print(f"Pinned checkpoint ready: {manifest['model']}@{manifest['revision']}", flush=True)


if __name__ == "__main__":
    main()
