"""Tests for persisted trust-on-first-use SSH host verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.trainer_target.ssh_remote import SshHostTrustStore
from services.trainer_target.trainer_target import TrainerTargetError


def _verify(
    store: SshHostTrustStore,
    *,
    identity: str = "runpod:pod-1",
    key: str = "host-key-a",
    host: str = "203.0.113.10",
) -> None:
    store.verify_or_record(
        identity=identity,
        hostname=host,
        port=22022,
        key_type="ssh-ed25519",
        key_base64=key,
    )


def test_first_seen_key_is_persisted_and_reconnect_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "trusted_hosts.json"
    store = SshHostTrustStore(path)

    _verify(store)
    _verify(store, host="203.0.113.11")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == 1
    assert payload["hosts"]["runpod:pod-1"]["keyBase64"] == "host-key-a"
    assert payload["hosts"]["runpod:pod-1"]["endpoint"] == "203.0.113.10:22022"


def test_same_pod_rejects_changed_host_key(tmp_path: Path) -> None:
    store = SshHostTrustStore(tmp_path / "trusted_hosts.json")
    _verify(store)

    with pytest.raises(TrainerTargetError) as exc:
        _verify(store, key="host-key-b")

    assert exc.value.retryable is False
    assert "identity changed" in exc.value.detail


def test_new_pod_identity_accepts_its_own_key(tmp_path: Path) -> None:
    store = SshHostTrustStore(tmp_path / "trusted_hosts.json")

    _verify(store, identity="runpod:pod-1", key="host-key-a")
    _verify(store, identity="runpod:pod-2", key="host-key-b")

    payload = json.loads((tmp_path / "trusted_hosts.json").read_text(encoding="utf-8"))
    assert set(payload["hosts"]) == {"runpod:pod-1", "runpod:pod-2"}


def test_corrupt_trust_store_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "trusted_hosts.json"
    path.write_text("{not-json", encoding="utf-8")
    store = SshHostTrustStore(path)

    with pytest.raises(TrainerTargetError) as exc:
        _verify(store)

    assert exc.value.retryable is False
    assert "refusing" in exc.value.detail
