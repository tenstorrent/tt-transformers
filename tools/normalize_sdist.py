#!/usr/bin/env python3
"""Normalize a generated source distribution for byte-reproducible release builds."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe sdist member: {name!r}")


def normalized_member(member: tarfile.TarInfo, epoch: int) -> tarfile.TarInfo:
    safe_name(member.name)
    result = tarfile.TarInfo(member.name)
    result.mode = member.mode
    result.uid = 0
    result.gid = 0
    result.uname = ""
    result.gname = ""
    result.mtime = epoch
    result.type = member.type
    result.linkname = member.linkname
    result.size = member.size if member.isfile() else 0
    result.devmajor = member.devmajor
    result.devminor = member.devminor
    return result


def normalize(path: Path, epoch: int) -> None:
    with tarfile.open(path, "r:gz") as source:
        members = []
        for member in source.getmembers():
            payload = source.extractfile(member).read() if member.isfile() else None
            members.append((normalized_member(member, epoch), payload))

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as destination:
                    for member, payload in sorted(members, key=lambda item: item[0].name):
                        destination.addfile(member, io.BytesIO(payload) if payload is not None else None)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--epoch", type=int, default=None)
    args = parser.parse_args()
    epoch = args.epoch
    if epoch is None:
        raw_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if not raw_epoch:
            raise ValueError("--epoch or SOURCE_DATE_EPOCH is required")
        epoch = int(raw_epoch)
    if epoch < 0:
        raise ValueError("normalization epoch must be non-negative")
    normalize(args.sdist, epoch)
    print(f"normalized {args.sdist} at epoch {epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
