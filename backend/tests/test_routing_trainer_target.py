"""Tests for `RoutingTrainerTarget` provider dispatch.

`RoutingTrainerTarget` holds one concrete target per provider and routes each
call to the right one by `credentials.provider`. These tests use a small hand
-written fake target (no mocks) and assert that calls land on the backend the
credentials name, and that an unknown provider raises.
"""

from __future__ import annotations

import pytest

from services.trainer_target.routing_trainer_target import RoutingTrainerTarget
from services.trainer_target.trainer_target import (
    RemoteCommandStatus,
    TrainerCredentials,
    TrainerTargetError,
)
from state.lora_training_state import TargetHandle, TrainerProvider


class RecordingTarget:
    """Minimal fake target that records which methods it received.

    Only the handful of methods the tests exercise are implemented; that's
    enough to verify routing without standing up a real backend.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.test_connection_calls: list[TrainerCredentials] = []
        self.started_commands: list[str] = []

    def test_connection(self, *, credentials: TrainerCredentials) -> None:
        self.test_connection_calls.append(credentials)

    def start_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        command: str,
        workdir: str,
    ) -> str:
        del credentials, handle, workdir
        self.started_commands.append(command)
        return f"{self.label}-job"

    def poll_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> RemoteCommandStatus:
        del credentials, handle, remote_job_id
        return RemoteCommandStatus(state="succeeded", exit_code=0)


def _credentials(provider: TrainerProvider) -> TrainerCredentials:
    return TrainerCredentials(
        provider=provider,
        workspace_dir="/workspace",
        model_path="/workspace/models/m.safetensors",
        text_encoder_path="/workspace/models/enc",
    )


def test_dispatches_to_backend_named_by_provider() -> None:
    runpod = RecordingTarget("runpod")
    local = RecordingTarget("local")
    router = RoutingTrainerTarget({"runpod": runpod, "local": local})

    router.test_connection(credentials=_credentials("runpod"))
    router.test_connection(credentials=_credentials("local"))

    assert len(runpod.test_connection_calls) == 1
    assert runpod.test_connection_calls[0].provider == "runpod"
    assert len(local.test_connection_calls) == 1
    assert local.test_connection_calls[0].provider == "local"


def test_routes_start_command_and_returns_backends_job_id() -> None:
    runpod = RecordingTarget("runpod")
    local = RecordingTarget("local")
    router = RoutingTrainerTarget({"runpod": runpod, "local": local})
    handle = TargetHandle(provider="local")

    job_id = router.start_command(
        credentials=_credentials("local"),
        handle=handle,
        command="echo hi",
        workdir="/workspace",
    )

    assert job_id == "local-job"
    assert local.started_commands == ["echo hi"]
    assert runpod.started_commands == []


def test_unknown_provider_raises_non_retryable() -> None:
    router = RoutingTrainerTarget({"runpod": RecordingTarget("runpod")})

    with pytest.raises(TrainerTargetError) as excinfo:
        router.test_connection(credentials=_credentials("local"))

    assert "no trainer target for provider 'local'" in excinfo.value.detail
    assert excinfo.value.retryable is False
