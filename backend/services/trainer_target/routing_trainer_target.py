"""Provider-routing `TrainerTarget`.

Every `TrainerTarget` method takes a `TrainerCredentials` whose `provider`
field names the backend that owns the job. This wrapper holds one concrete
target per provider and dispatches each call to the right one based on
`credentials.provider`, so the runner can stay provider-agnostic and just
talk to a single `TrainerTarget`.
"""

from __future__ import annotations

from services.trainer_target.trainer_target import (
    AccountInfo,
    GpuTelemetry,
    NetworkVolume,
    PodInfo,
    ProgressCallback,
    RemoteCommandStatus,
    TrainerCredentials,
    TrainerTarget,
    TrainerTargetError,
    ValidationArtifact,
)
from state.lora_training_state import TargetHandle, TrainerProvider


class RoutingTrainerTarget:
    def __init__(self, targets: dict[TrainerProvider, TrainerTarget]) -> None:
        self._targets = targets

    def _target(self, credentials: TrainerCredentials) -> TrainerTarget:
        target = self._targets.get(credentials.provider)
        if target is None:
            raise TrainerTargetError(
                f"no trainer target for provider {credentials.provider!r}",
                retryable=False,
            )
        return target

    def test_connection(self, *, credentials: TrainerCredentials) -> None:
        return self._target(credentials).test_connection(credentials=credentials)

    def connect_account(self, *, credentials: TrainerCredentials) -> AccountInfo:
        return self._target(credentials).connect_account(credentials=credentials)

    def list_pods(self, *, credentials: TrainerCredentials) -> list[PodInfo]:
        return self._target(credentials).list_pods(credentials=credentials)

    def stop_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        return self._target(credentials).stop_pod(
            credentials=credentials, pod_id=pod_id
        )

    def resume_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        return self._target(credentials).resume_pod(
            credentials=credentials, pod_id=pod_id
        )

    def ensure_network_volume(
        self,
        *,
        credentials: TrainerCredentials,
        name: str,
        size_gb: int,
        datacenter_id: str | None = None,
    ) -> NetworkVolume:
        return self._target(credentials).ensure_network_volume(
            credentials=credentials,
            name=name,
            size_gb=size_gb,
            datacenter_id=datacenter_id,
        )

    def delete_network_volume(
        self, *, credentials: TrainerCredentials, volume_id: str
    ) -> None:
        return self._target(credentials).delete_network_volume(
            credentials=credentials, volume_id=volume_id
        )

    def ensure_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle | None
    ) -> TargetHandle:
        return self._target(credentials).ensure_workspace(
            credentials=credentials, handle=handle
        )

    def ensure_provisioned(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        progress: ProgressCallback | None = None,
    ) -> None:
        return self._target(credentials).ensure_provisioned(
            credentials=credentials, handle=handle, progress=progress
        )

    def upload_directory(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        local_dir: str,
        remote_dir: str,
    ) -> None:
        return self._target(credentials).upload_directory(
            credentials=credentials,
            handle=handle,
            local_dir=local_dir,
            remote_dir=remote_dir,
        )

    def download_file(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_path: str,
        local_path: str,
    ) -> None:
        return self._target(credentials).download_file(
            credentials=credentials,
            handle=handle,
            remote_path=remote_path,
            local_path=local_path,
        )

    def start_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        command: str,
        workdir: str,
    ) -> str:
        return self._target(credentials).start_command(
            credentials=credentials,
            handle=handle,
            command=command,
            workdir=workdir,
        )

    def poll_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> RemoteCommandStatus:
        return self._target(credentials).poll_command(
            credentials=credentials, handle=handle, remote_job_id=remote_job_id
        )

    def read_logs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
        tail: int,
    ) -> list[str]:
        return self._target(credentials).read_logs(
            credentials=credentials,
            handle=handle,
            remote_job_id=remote_job_id,
            tail=tail,
        )

    def terminate(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> None:
        return self._target(credentials).terminate(
            credentials=credentials, handle=handle, remote_job_id=remote_job_id
        )

    def release_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> None:
        return self._target(credentials).release_workspace(
            credentials=credentials, handle=handle
        )

    def query_gpu(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> GpuTelemetry:
        return self._target(credentials).query_gpu(
            credentials=credentials, handle=handle
        )

    def list_validation_outputs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
        since_step: int,
    ) -> list[ValidationArtifact]:
        return self._target(credentials).list_validation_outputs(
            credentials=credentials,
            handle=handle,
            remote_output_dir=remote_output_dir,
            since_step=since_step,
        )

    def list_checkpoints(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
    ) -> list[int]:
        return self._target(credentials).list_checkpoints(
            credentials=credentials,
            handle=handle,
            remote_output_dir=remote_output_dir,
        )

    def count_precomputed_source(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        precomputed_dir: str,
        source: str,
    ) -> int:
        return self._target(credentials).count_precomputed_source(
            credentials=credentials,
            handle=handle,
            precomputed_dir=precomputed_dir,
            source=source,
        )

    def delete_remote_paths(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        paths: list[str],
    ) -> None:
        return self._target(credentials).delete_remote_paths(
            credentials=credentials, handle=handle, paths=paths
        )
