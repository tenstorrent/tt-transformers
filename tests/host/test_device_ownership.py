from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from tt_transformers.device_ownership import (
    DEFAULT_DEVICE_FALLBACK_LEDGER,
    active_default_device_owners,
    compatibility_default_device,
    default_device_scope,
)


ROOT = Path(__file__).resolve().parents[2]


class FakeTTNN:
    def __init__(self, default=None):
        self.default = default
        self.set_calls = []
        self.fail_next_set = False

    def GetDefaultDevice(self):
        return self.default

    def SetDefaultDevice(self, device):
        self.set_calls.append(device)
        self.default = device
        if self.fail_next_set:
            self.fail_next_set = False
            raise RuntimeError("set failed after mutation")


@pytest.mark.host
def test_scope_restores_exact_previous_device():
    original = object()
    owned = object()
    ttnn = FakeTTNN(original)
    with default_device_scope(ttnn, owned, owner="builder"):
        assert ttnn.default is owned
        assert active_default_device_owners() == ("builder",)
    assert ttnn.default is original
    assert ttnn.set_calls == [owned, original]
    assert active_default_device_owners() == ()


@pytest.mark.host
def test_scope_restores_after_body_failure():
    original = object()
    ttnn = FakeTTNN(original)
    with pytest.raises(LookupError, match="body failed"):
        with default_device_scope(ttnn, object(), owner="failing-builder"):
            raise LookupError("body failed")
    assert ttnn.default is original
    assert active_default_device_owners() == ()


@pytest.mark.host
def test_nested_scopes_restore_in_lifo_order():
    original = object()
    outer = object()
    inner = object()
    ttnn = FakeTTNN(original)
    with default_device_scope(ttnn, outer, owner="outer"):
        assert active_default_device_owners() == ("outer",)
        with default_device_scope(ttnn, inner, owner="inner"):
            assert ttnn.default is inner
            assert active_default_device_owners() == ("outer", "inner")
        assert ttnn.default is outer
    assert ttnn.default is original
    assert ttnn.set_calls == [outer, inner, outer, original]


@pytest.mark.host
def test_repeated_scopes_do_not_retain_previous_owner():
    original = object()
    ttnn = FakeTTNN(original)
    for index in range(3):
        owned = object()
        with default_device_scope(ttnn, owned, owner=f"builder-{index}"):
            assert ttnn.default is owned
        assert ttnn.default is original
        assert active_default_device_owners() == ()


@pytest.mark.host
def test_failed_set_is_best_effort_restored():
    original = object()
    ttnn = FakeTTNN(original)
    ttnn.fail_next_set = True
    with pytest.raises(RuntimeError, match="set failed"):
        with default_device_scope(ttnn, object(), owner="builder"):
            raise AssertionError("body must not run")
    assert ttnn.default is original
    assert active_default_device_owners() == ()


@pytest.mark.host
def test_different_threads_cannot_observe_cross_owner_default():
    original = object()
    first_device = object()
    second_device = object()
    ttnn = FakeTTNN(original)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    observations = []

    def first_owner():
        with default_device_scope(ttnn, first_device, owner="first"):
            observations.append(("first", ttnn.default))
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second_owner():
        assert first_entered.wait(timeout=2)
        with default_device_scope(ttnn, second_device, owner="second"):
            observations.append(("second", ttnn.default))
            second_entered.set()

    first = threading.Thread(target=first_owner)
    second = threading.Thread(target=second_owner)
    first.start()
    second.start()
    assert first_entered.wait(timeout=2)
    assert not second_entered.wait(timeout=0.05)
    assert ttnn.default is first_device
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive() and not second.is_alive()
    assert observations == [("first", first_device), ("second", second_device)]
    assert ttnn.default is original


@pytest.mark.host
def test_compatibility_fallback_requires_registered_owner_and_active_default():
    owner = DEFAULT_DEVICE_FALLBACK_LEDGER[0].owner
    device = object()
    assert compatibility_default_device(FakeTTNN(device), owner=owner) is device
    with pytest.raises(ValueError, match="unregistered"):
        compatibility_default_device(FakeTTNN(device), owner="unknown")
    with pytest.raises(ValueError, match="explicit mesh_device"):
        compatibility_default_device(FakeTTNN(), owner=owner)


@pytest.mark.host
def test_fallback_ledger_is_exact_and_unique():
    owners = [record.owner for record in DEFAULT_DEVICE_FALLBACK_LEDGER]
    assert len(owners) == 10
    assert len(set(owners)) == 10
    assert all(record.rationale for record in DEFAULT_DEVICE_FALLBACK_LEDGER)


@pytest.mark.host
def test_production_default_device_access_matches_ledger_exactly():
    fallback_owners = []
    direct_accesses = []
    for path in (ROOT / "src/tt_transformers").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "compatibility_default_device":
                owner = next(
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "owner"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                )
                fallback_owners.append(owner)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"SetDefaultDevice", "GetDefaultDevice"}:
                if path.name != "device_ownership.py":
                    direct_accesses.append((path, node.lineno, node.func.attr))
    assert not direct_accesses
    assert sorted(fallback_owners) == sorted(record.owner for record in DEFAULT_DEVICE_FALLBACK_LEDGER)
