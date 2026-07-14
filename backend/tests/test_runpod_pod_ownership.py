"""Ownership scoping for the RunPod API-level pod termination.

`release_workspace` terminates a pod by id via the RunPod API. To keep that
from ever killing a same-account pod LTX Desktop didn't spawn (a corrupted
handle or a stray connect-UI request), it must first fetch the pod and confirm
its name is the app's pod name. These tests inject a fake `runpod` module via
pytest's `monkeypatch` (a real module object — no mocking libraries) so the
target's `import runpod` resolves to it.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from services.trainer_target.runpod_trainer_target import RunPodTrainerTarget
from services.trainer_target.trainer_target import (
    GpuOffer,
    NetworkVolume,
    TrainerCredentials,
    TrainerTargetError,
)
from state.lora_training_state import TargetHandle


class _FakeRunPod:
    """Minimal stand-in for the `runpod` SDK surface used by release_workspace."""

    api_key: str = ""

    def __init__(self) -> None:
        self.pods: dict[str, dict[str, Any]] = {}
        self.get_pod_calls: list[str] = []
        self.create_pod_calls: list[dict[str, Any]] = []
        self.stop_pod_calls: list[str] = []
        self.resume_pod_calls: list[str] = []
        self.terminate_pod_calls: list[str] = []

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        self.get_pod_calls.append(pod_id)
        return self.pods.get(pod_id)

    def get_pods(self) -> list[dict[str, Any]]:
        return list(self.pods.values())

    def terminate_pod(self, pod_id: str) -> None:
        self.terminate_pod_calls.append(pod_id)
        self.pods.pop(pod_id, None)

    def create_pod(self, **kwargs: Any) -> dict[str, str]:
        self.create_pod_calls.append(kwargs)
        self.pods["new-pod"] = {"id": "new-pod", "name": "ltx-desktop-lora"}
        return {"id": "new-pod"}

    def stop_pod(self, pod_id: str) -> None:
        self.stop_pod_calls.append(pod_id)

    def resume_pod(self, pod_id: str) -> None:
        self.resume_pod_calls.append(pod_id)
        if pod_id in self.pods:
            self.pods[pod_id]["desiredStatus"] = "RUNNING"


def _install_fake_runpod(monkeypatch: pytest.MonkeyPatch) -> _FakeRunPod:
    fake = _FakeRunPod()
    module = ModuleType("runpod")
    # `import runpod` binds the module object; attribute access (api_key,
    # get_pod, terminate_pod) must resolve to our fake's methods.
    module.api_key = ""  # type: ignore[attr-defined]
    module.get_pod = fake.get_pod  # type: ignore[attr-defined]
    module.get_pods = fake.get_pods  # type: ignore[attr-defined]
    module.create_pod = fake.create_pod  # type: ignore[attr-defined]
    module.stop_pod = fake.stop_pod  # type: ignore[attr-defined]
    module.resume_pod = fake.resume_pod  # type: ignore[attr-defined]
    module.terminate_pod = fake.terminate_pod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod", module)
    return fake


def _creds() -> TrainerCredentials:
    return TrainerCredentials(
        provider="runpod",
        workspace_dir="/ws",
        model_path="/m",
        text_encoder_path="/te",
        runpod_api_key="test-key",
    )


def test_list_pods_parses_provider_uptime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_runpod(monkeypatch)
    fake.pods["pod-1"] = {
        "id": "pod-1",
        "name": "ltx-desktop-lora",
        "desiredStatus": "RUNNING",
        "costPerHr": 1.5,
        "runtime": {
            "uptimeInSeconds": 3_725,
            "lastStartedAt": "2026-07-12T07:00:00Z",
        },
    }

    pod = RunPodTrainerTarget().list_pods(credentials=_creds())[0]
    assert pod.uptime_seconds == 3_725
    assert pod.last_started_at == "2026-07-12T07:00:00Z"
    assert pod.cost_per_hr == 1.5


def test_release_workspace_terminates_app_owned_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_runpod(monkeypatch)
    fake.pods["pod-1"] = {"id": "pod-1", "name": "ltx-desktop-lora"}

    RunPodTrainerTarget().release_workspace(
        credentials=_creds(),
        handle=TargetHandle(provider="runpod", pod_id="pod-1"),
    )

    assert fake.get_pod_calls == ["pod-1"]
    assert fake.terminate_pod_calls == ["pod-1"]


def test_release_workspace_refuses_pod_not_created_by_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_runpod(monkeypatch)
    # Same account, but a workload LTX Desktop didn't spawn.
    fake.pods["pod-2"] = {"id": "pod-2", "name": "my-other-workload"}

    with pytest.raises(TrainerTargetError) as exc:
        RunPodTrainerTarget().release_workspace(
            credentials=_creds(),
            handle=TargetHandle(provider="runpod", pod_id="pod-2"),
        )

    assert exc.value.retryable is False
    assert fake.terminate_pod_calls == []


def test_release_workspace_already_gone_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_runpod(monkeypatch)
    # get_pod returns None — pod was already terminated.

    RunPodTrainerTarget().release_workspace(
        credentials=_creds(),
        handle=TargetHandle(provider="runpod", pod_id="gone"),
    )

    assert fake.get_pod_calls == ["gone"]
    assert fake.terminate_pod_calls == []


def test_new_pod_is_terminated_when_readiness_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_runpod(monkeypatch)
    target = RunPodTrainerTarget()

    def fail_readiness(_runpod: Any, _pod_id: str) -> None:
        raise TrainerTargetError("SSH never became ready", retryable=True)

    monkeypatch.setattr(target, "_wait_until_ready", fail_readiness)

    with pytest.raises(TrainerTargetError, match="SSH never became ready"):
        target._create_pod(  # noqa: SLF001 - lifecycle invariant under test
            fake,
            _creds(),
            {"gpu_type_id": "NVIDIA A100 80GB PCIe"},
        )

    assert fake.terminate_pod_calls == ["new-pod"]
    assert "new-pod" not in fake.pods


def test_unavailable_gpu_directs_user_back_to_per_run_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_runpod(monkeypatch)
    target = RunPodTrainerTarget()

    def unavailable_create(**_kwargs: Any) -> dict[str, str]:
        raise RuntimeError("no instances available")

    monkeypatch.setattr(fake, "create_pod", unavailable_create)
    monkeypatch.setattr(target, "_available_training_gpus", lambda *_args, **_kwargs: [])

    with pytest.raises(TrainerTargetError) as exc:
        target._create_pod(  # noqa: SLF001 - capacity UX contract under test
            fake,
            _creds(),
            {"gpu_type_id": "NVIDIA A100 80GB PCIe"},
        )

    assert exc.value.code == "capacity_unavailable"
    assert "Return to GPU selection" in exc.value.detail
    assert "dataset and progress are preserved" in exc.value.detail
    assert "Settings" not in exc.value.detail


@pytest.mark.parametrize("action", ["stop", "resume"])
def test_stop_and_resume_refuse_foreign_pods(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    fake = _install_fake_runpod(monkeypatch)
    fake.pods["foreign"] = {"id": "foreign", "name": "other-workload"}
    target = RunPodTrainerTarget()

    with pytest.raises(TrainerTargetError, match="not created by LTX Desktop"):
        if action == "stop":
            target.stop_pod(credentials=_creds(), pod_id="foreign")
        else:
            target.resume_pod(credentials=_creds(), pod_id="foreign")

    assert fake.stop_pod_calls == []
    assert fake.resume_pod_calls == []


def test_ensure_workspace_resumes_app_owned_stopped_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_runpod(monkeypatch)
    fake.pods["stopped"] = {
        "id": "stopped",
        "name": "ltx-desktop-lora",
        "desiredStatus": "STOPPED",
    }
    target = RunPodTrainerTarget()
    monkeypatch.setattr(target, "_wait_until_ready", lambda _r, _p: None)

    handle = target.ensure_workspace(
        credentials=_creds(),
        handle=TargetHandle(provider="runpod", pod_id="stopped"),
    )

    assert handle.pod_id == "stopped"
    assert fake.resume_pod_calls == ["stopped"]


def test_volume_region_selection_refuses_when_no_qualifying_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = RunPodTrainerTarget()
    monkeypatch.setattr(
        target, "_storage_datacenters", lambda _credentials: ["EU-RO-1", "US-TX-1"]
    )
    unavailable = [
        GpuOffer(
            id="NVIDIA A100 80GB PCIe",
            label="A100",
            memory_gb=80,
            available=False,
        )
    ]
    monkeypatch.setattr(
        target,
        "_discover_gpus_graphql",
        lambda _credentials, _datacenter: unavailable,
    )

    with pytest.raises(TrainerTargetError, match="No storage-capable") as exc:
        target._pick_volume_datacenter(_creds())  # noqa: SLF001

    assert exc.value.retryable is False


def test_delete_network_volume_refuses_foreign_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = RunPodTrainerTarget()
    monkeypatch.setattr(
        target,
        "_list_volumes",
        lambda _credentials: [
            NetworkVolume(
                id="foreign",
                name="user-data",
                size_gb=500,
                datacenter_id="US-TX-1",
                created_by_app=False,
            )
        ],
    )

    with pytest.raises(TrainerTargetError, match="not created by LTX Desktop"):
        target.delete_network_volume(credentials=_creds(), volume_id="foreign")
