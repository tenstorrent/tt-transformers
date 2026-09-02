# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from tests.support.cache_entries_counter import CacheEntriesCounter


pytestmark = pytest.mark.host

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = "tests/tests_common/cache_entries_counter.py"
DESTINATION_PATH = "tests/support/cache_entries_counter.py"
SOURCE_BLOB = "028364b7e8c6fe1a5d8a5d60f804443b69030b67"


class FakeDevice:
    def __init__(self, entries: list[int]):
        self.entries = iter(entries)
        self.reads = 0

    def num_program_cache_entries(self) -> int:
        self.reads += 1
        return next(self.entries)


@pytest.mark.host
def test_measure_accumulates_program_cache_entry_deltas_and_reset_clears_total():
    counter = CacheEntriesCounter(FakeDevice([3, 8, 8, 10]))

    with counter.measure():
        pass
    with counter.measure():
        pass

    assert counter.total == 7
    counter.reset()
    assert counter.total == 0


@pytest.mark.host
def test_decorator_preserves_metadata_arguments_and_return_value():
    counter = CacheEntriesCounter(FakeDevice([10, 13]))

    @counter.decorator
    def operation(value: int, *, scale: int = 1) -> int:
        """Example decorated operation."""
        return value * scale

    assert operation(4, scale=3) == 12
    assert operation.__name__ == "operation"
    assert operation.__doc__ == "Example decorated operation."
    assert counter.total == 3


@pytest.mark.host
def test_measure_preserves_source_exception_behavior_without_a_post_read():
    device = FakeDevice([4, 99])
    counter = CacheEntriesCounter(device)

    with pytest.raises(RuntimeError, match="intentional"):
        with counter.measure():
            raise RuntimeError("intentional")

    assert counter.total == 0
    assert device.reads == 1


@pytest.mark.host
def test_pinned_helper_has_complete_standalone_provenance():
    with (ROOT / "qualification/provenance/source_inventory.csv").open(newline="", encoding="utf-8") as stream:
        inventory = {row["source_path"]: row for row in csv.DictReader(stream)}
    row = inventory[SOURCE_PATH]
    assert row == {
        "source_path": SOURCE_PATH,
        "destination_path": DESTINATION_PATH,
        "git_blob_sha": SOURCE_BLOB,
        "disposition": "renamed",
        "reason": (
            "Move the cache-entry accounting helper required by standalone device fixtures "
            "into repository-owned test support."
        ),
        "category": "test_support",
        "boundary_cleanup": "true",
    }

    with (ROOT / "qualification/extraction/support_copy_manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = {
            (entry["source_path"], entry["destination_path"]): entry
            for entry in csv.DictReader(stream)
        }
    copied = manifest[(SOURCE_PATH, DESTINATION_PATH)]
    destination = ROOT / DESTINATION_PATH
    assert copied["source_blob_sha"] == SOURCE_BLOB
    assert copied["destination_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert copied["destination_size"] == str(destination.stat().st_size)
    assert copied["exact_raw_copy"] == "true"
    assert copied["transformation"] == "content_preserving_path_move"


@pytest.mark.host
def test_device_fixtures_use_the_standalone_helper_owner():
    source = (ROOT / "tests/conftest.py").read_text(encoding="utf-8")

    assert source.count("from tests.support.cache_entries_counter import CacheEntriesCounter") == 2
    assert "tests.tests_common.cache_entries_counter" not in source
