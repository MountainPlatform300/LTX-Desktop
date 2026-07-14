"""Remote GPU execution backend for the LoRA trainer.

`TrainerTarget` is the side-effect boundary for everything that has to
happen on a remote CUDA box: provisioning compute, shipping the dataset
up, running the LTX-2 trainer scripts, tailing their logs, and pulling
the finished `lora_weights.safetensors` back down.

Design notes
------------
- **Generic verbs.** The protocol intentionally exposes plumbing
  (`start_command`, `poll_command`, `download_file`, ...) rather than
  trainer-specific verbs (`caption`, `preprocess`, `train`). The actual
  `caption_videos.py` / `process_dataset.py` / `train.py` command lines
  are built by `lora_command_builder` and handed to `start_command`, so a
  new trainer script never touches the remote-execution code.
- **Stateless w.r.t. secrets.** Every call takes a `TrainerCredentials`
  snapshot, mirroring `LTXAPIClient` (which takes `api_key=` per call).
  Credentials come from `AppSettings` at call time; the service holds
  no secret state.
- **Handles are durable.** `ensure_workspace` returns a `TargetHandle`
  the caller persists on the owning entity, so a reconciler can re-poll
  a remote job after an app restart instead of orphaning it.
- **One implementation.** `RunPodTrainerTarget` is the real backend wired
  into the service bundle; a fake stands in for it in tests.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from state.lora_training_state import TargetHandle, TrainerProvider

# Optional sink for live progress lines (remote log tails) during a long
# blocking step like provisioning, so the caller can surface them to the UI.
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class TrainerCredentials:
    """Everything a remote command needs, snapshotted from settings.

    `gemini_api_key` is forwarded so `caption_videos.py --captioner-type
    gemini_flash` can run without baking a key into the remote.
    """

    provider: TrainerProvider
    workspace_dir: str
    model_path: str
    text_encoder_path: str
    gemini_api_key: str = ""
    # RunPod
    runpod_api_key: str = ""
    runpod_gpu_type: str = ""
    runpod_image: str = ""
    runpod_network_volume_id: str = ""
    runpod_datacenter: str = ""
    # Size for a pod-local volume when no network volume is configured
    # (ephemeral / caching-off), so /workspace can hold the models + data.
    volume_size_gb: int = 0
    # Auto-provisioning (consumed by `ensure_provisioned`). When
    # `auto_provision` is False the target assumes a pre-baked image and
    # skips bootstrap entirely. `hf_token` lets the remote download gated
    # weights; empty = anonymous (public repos only).
    auto_provision: bool = True
    trainer_repo_url: str = ""
    trainer_repo_ref: str = "main"
    model_hf_repo: str = ""
    # When set, provisioning downloads just this file from `model_hf_repo`
    # (the LTX-2 repo is hundreds of GB; the trainer needs one checkpoint)
    # rather than the whole repo. Empty = whole-repo download (back-compat).
    model_filename: str = ""
    text_encoder_hf_repo: str = ""
    hf_token: str = ""


RemoteCommandState = Literal["running", "succeeded", "failed"]


@dataclass(frozen=True)
class RemoteCommandStatus:
    """Snapshot of a remote command's lifecycle.

    `exit_code` is populated once the process terminates (0 ->
    succeeded, non-zero -> failed). `error` carries a transport- or
    provider-level message when the status itself couldn't be
    determined cleanly.
    """

    state: RemoteCommandState
    exit_code: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class GpuTelemetry:
    """Live snapshot of the GPU the training job is running on.

    Used by the GPU-status panel. `vram_*` are in MiB (matching
    `nvidia-smi`'s `--format=csv,noheader,nounits` memory fields);
    `*_util_pct` are 0-100; `temp_c` is Celsius or ``None`` when the
    device didn't report a temperature.
    """

    name: str
    vram_total_mb: int
    vram_used_mb: int
    gpu_util_pct: int
    mem_util_pct: int
    temp_c: int | None = None


@dataclass(frozen=True)
class ValidationArtifact:
    """One validation sample the trainer wrote to the run's output dir.

    The trainer saves samples as
    ``{remote_output_dir}/samples/step_{NNNNNN}_{i}.{ext}`` (see
    ``ltx_trainer.validation_runner``); `remote_path` is the absolute
    path on the target, `step`/`sample_index` are parsed from the
    filename, and `ext` picks the media kind (``mp4``/``png``/``wav``).
    """

    step: int
    sample_index: int
    remote_path: str
    ext: str


@dataclass(frozen=True)
class GpuOffer:
    """A GPU type the provider can allocate, for the connect-flow picker.

    `id` is the provider's gpu-type id passed back to `ensure_workspace`
    via `runpod_gpu_type`. `price_per_hr` is on-demand $/hr (None if the
    provider didn't report one). `available` reflects current stock so the
    UI can grey out / fall back from sold-out types.
    """

    id: str
    label: str
    memory_gb: int
    price_per_hr: float | None = None
    available: bool = True
    active_region_available: bool | None = None
    available_elsewhere: bool | None = None
    best_available_region: str | None = None
    recommended: bool = False


@dataclass(frozen=True)
class NetworkVolume:
    """An existing persistent network volume on the account."""

    id: str
    name: str
    size_gb: int
    datacenter_id: str = ""
    created_by_app: bool = False


RegionHealthStatus = Literal["healthy", "no_stock", "unknown"]


@dataclass(frozen=True)
class RegionHealth:
    """Current qualifying (>=32 GB) GPU stock for one storage region."""

    datacenter_id: str
    status: RegionHealthStatus
    qualifying_gpu_available: bool
    available_gpu_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PodInfo:
    """A pod currently on the account (running or stopped), for the connect
    flow's "active pods" panel so the user can see and reclaim compute."""

    id: str
    name: str
    gpu: str
    status: str
    cost_per_hr: float | None = None
    # True when this pod looks like one the app created (so the UI can flag
    # strays the app can safely offer to reuse/terminate).
    created_by_app: bool = False
    # Normalized lifecycle so the UI can pick the right action without parsing
    # the raw `status` string. `desired_status` is the provider's canonical
    # state (e.g. "RUNNING"/"STOPPED"/"EXITED"); `running` is True only when the
    # pod is actually consuming GPU (billable).
    desired_status: str = ""
    running: bool = False
    # Provider-reported billable runtime. `uptime_seconds` is cumulative for the
    # current pod lifecycle; `last_started_at` lets the UI keep the meter moving
    # smoothly between API polls while a pod is running.
    uptime_seconds: int | None = None
    last_started_at: str | None = None


@dataclass(frozen=True)
class AccountInfo:
    """What the connect probe returns: allocatable GPUs, existing volumes, and
    any pods already on the account.

    Provider-management data for the RunPod backend, which allocates compute.
    """

    gpus: tuple[GpuOffer, ...]
    volumes: tuple[NetworkVolume, ...]
    pods: tuple[PodInfo, ...] = ()
    # The datacenter GPU availability was evaluated against (the network
    # volume's region, when one is configured). Empty = global availability.
    datacenter: str = ""
    region_health: tuple[RegionHealth, ...] = ()


class TrainerTargetError(RuntimeError):
    """Raised for provider/transport failures.

    `retryable` lets the runner distinguish a transient network blip
    (worth another reconciler tick) from a hard configuration error
    (surface to the user immediately).
    """

    def __init__(
        self,
        detail: str,
        *,
        retryable: bool = False,
        code: Literal[
            "capacity_unavailable", "ownership_violation", "provider_error"
        ] = "provider_error",
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retryable = retryable
        self.code = code


# Fixed `nvidia-smi` argv for `query_gpu`. Kept as a single constant string so
# NO user-controlled value is ever interpolated into a GPU query — every call
# runs exactly this command, on WSL (`wsl.exe`) or over SSH (RunPod). Output is
# one CSV line: name, memory.total(MiB), memory.used(MiB), util.gpu(%),
# util.memory(%), temp.gpu(C).
NVIDIA_SMI_GPU_QUERY = (
    "nvidia-smi --query-gpu=name,memory.total,memory.used,"
    "utilization.gpu,utilization.memory,temperature.gpu "
    "--format=csv,noheader,nounits"
)

# Filenames the trainer writes for validation samples:
# ``step_{NNNNNN}_{i}.{ext}`` (e.g. ``step_000050_1.mp4``).
_SAMPLE_FILENAME_RE = re.compile(r"step_(\d+)_(\d+)\.([A-Za-z0-9]+)$")


def parse_nvidia_smi_output(out: str) -> GpuTelemetry:
    """Parse one CSV line of `nvidia-smi` GPU query output into `GpuTelemetry`.

    Raises `TrainerTargetError` if there is no parseable line. Unparseable
    numeric fields fall back to ``0`` (or ``None`` for temperature) so a
    partial `nvidia-smi` response still yields a usable snapshot rather than
    killing the status panel.
    """
    first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    if not first:
        raise TrainerTargetError(
            "nvidia-smi returned no GPU output", retryable=False
        )
    parts = [p.strip() for p in first.split(",")]

    def _int(i: int) -> int:
        try:
            return int(parts[i])
        except (ValueError, IndexError):
            return 0

    name = parts[0] if parts and parts[0] else "Unknown"
    temp_c: int | None = None
    if len(parts) > 5 and parts[5] not in ("", "N/A", "[N/A]"):
        try:
            temp_c = int(parts[5])
        except ValueError:
            temp_c = None
    return GpuTelemetry(
        name=name,
        vram_total_mb=_int(1),
        vram_used_mb=_int(2),
        gpu_util_pct=_int(3),
        mem_util_pct=_int(4),
        temp_c=temp_c,
    )


def parse_samples_listing(out: str, remote_samples_dir: str) -> list[ValidationArtifact]:
    """Parse `ls -1 {remote_output_dir}/samples` output into artifacts.

    Lines that don't match the trainer's ``step_NNNNNN_i.ext`` naming are
    ignored (e.g. stray files, future subdirs). Sorted by ``(step, index)``.
    `remote_samples_dir` is the absolute path the listing was run against, used
    to reconstruct each artifact's absolute `remote_path`.
    """
    base = remote_samples_dir.rstrip("/")
    artifacts: list[ValidationArtifact] = []
    for raw in out.splitlines():
        name = raw.strip()
        if not name:
            continue
        match = _SAMPLE_FILENAME_RE.search(name)
        if not match:
            continue
        artifacts.append(
            ValidationArtifact(
                step=int(match.group(1)),
                sample_index=int(match.group(2)),
                remote_path=f"{base}/{name}",
                ext=match.group(3).lower(),
            )
        )
    artifacts.sort(key=lambda a: (a.step, a.sample_index))
    return artifacts


def samples_dir_for(remote_output_dir: str) -> str:
    """The remote subdir the trainer writes validation samples into."""
    return remote_output_dir.rstrip("/") + "/samples"


# Trainer adapter filenames: `lora_weights_step_NNNNN.safetensors` under
# `{output_dir}/checkpoints/` (one per checkpoint interval + a final save).
_CHECKPOINT_FILENAME_RE = re.compile(r"lora_weights_step_(\d+)\.safetensors$")


def checkpoints_dir_for(remote_output_dir: str) -> str:
    """The remote subdir the trainer writes adapter checkpoints into."""
    return remote_output_dir.rstrip("/") + "/checkpoints"


def parse_checkpoints_listing(out: str) -> list[int]:
    """Parse `ls -1 {remote_output_dir}/checkpoints` into sorted step numbers.

    Non-matching lines (stray files, subdirs) are ignored. Sorted ascending so
    callers can take `[-1]` for the highest existing checkpoint.
    """
    steps: list[int] = []
    for raw in out.splitlines():
        name = raw.strip()
        if not name:
            continue
        match = _CHECKPOINT_FILENAME_RE.search(name)
        if match is not None:
            steps.append(int(match.group(1)))
    steps.sort()
    return steps


class TrainerTarget(Protocol):
    def test_connection(self, *, credentials: TrainerCredentials) -> None:
        """Validate credentials/reachability. Raises on failure."""
        ...

    def connect_account(self, *, credentials: TrainerCredentials) -> AccountInfo:
        """Validate the key and return allocatable GPUs + existing volumes.

        Powers the one-click connect flow's GPU picker and cache toggle.
        """
        ...

    def list_pods(self, *, credentials: TrainerCredentials) -> list[PodInfo]:
        """Pods currently on the account (running or stopped).

        Standalone pod listing (no GPU/volume discovery) for the Trainer's
        compute panel so the user can see and control stray pods that would
        otherwise keep billing. RunPod-only concept; a local target returns [].
        """
        ...

    def stop_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        """Pause a running pod (stop GPU billing; keep the container disk).

        Reversible via `resume_pod`. Idempotent: stopping an already-stopped
        or gone pod is success. RunPod-only.
        """
        ...

    def resume_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        """Start a stopped pod (resume GPU billing). RunPod-only.

        Idempotent: resuming an already-running or gone pod is success.
        """
        ...

    def ensure_network_volume(
        self,
        *,
        credentials: TrainerCredentials,
        name: str,
        size_gb: int,
        datacenter_id: str | None = None,
    ) -> NetworkVolume:
        """Create (or reuse a same-named) persistent network volume.

        Returns the volume so the caller can persist its id in settings and
        mount it on future pods. RunPod-only.
        """
        ...

    def delete_network_volume(
        self, *, credentials: TrainerCredentials, volume_id: str
    ) -> None:
        """Delete an app-owned persistent volume after provider ownership checks."""
        ...

    def ensure_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle | None
    ) -> TargetHandle:
        """Make compute ready and return a durable handle.

        Create an on-demand RunPod pod (or reuse the one in `handle`).
        """
        ...

    def ensure_provisioned(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        progress: ProgressCallback | None = None,
    ) -> None:
        """Install the trainer + (optionally) base models on the workspace.

        `progress`, when given, is called periodically with the latest remote
        log line (git clone %, uv sync, the HF download bar) so the caller can
        surface live setup progress to the UI.

        Idempotent and marker-gated: the first call on a fresh pod runs
        the bootstrap (clone trainer, `uv sync`, download configured HF
        weights); later calls — including on a reused pod or mounted
        network volume — short-circuit on the marker. A no-op when
        `credentials.auto_provision` is False. Raises `TrainerTargetError`
        on failure.
        """
        ...

    def upload_directory(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        local_dir: str,
        remote_dir: str,
    ) -> None:
        """Upload a local directory tree to the remote (idempotent)."""
        ...

    def download_file(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Download a single remote file to a local path."""
        ...

    def start_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        command: str,
        workdir: str,
    ) -> str:
        """Launch a detached remote command; return its remote job id."""
        ...

    def poll_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> RemoteCommandStatus:
        """Return the current lifecycle status of a remote command."""
        ...

    def read_logs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
        tail: int,
    ) -> list[str]:
        """Return up to `tail` trailing log lines for a remote command."""
        ...

    def terminate(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> None:
        """Best-effort kill of a running remote command."""
        ...

    def release_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> None:
        """Release compute (delete the RunPod pod)."""
        ...

    def query_gpu(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> GpuTelemetry:
        """Live GPU telemetry for the training target.

        Runs the fixed `NVIDIA_SMI_GPU_QUERY` argv on the target (WSL or
        RunPod pod) — no caller-controlled value reaches the shell. Raises
        `TrainerTargetError` on transport/command failure; callers surface a
        degraded status panel rather than retrying in a tight loop.
        """
        ...

    def list_validation_outputs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
        since_step: int,
    ) -> list[ValidationArtifact]:
        """New validation samples the trainer has written under `remote_output_dir`.

        Lists ``{remote_output_dir}/samples`` and returns artifacts whose step
        is greater than `since_step`, so a reconciler can download just the
        samples it hasn't seen. The only interpolated value is the run's own
        output directory (derived server-side, not user-typed). Returns ``[]``
        when the directory doesn't exist yet (validation hasn't run).
        """
        ...

    def list_checkpoints(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
    ) -> list[int]:
        """Step numbers of every adapter checkpoint present under
        ``{remote_output_dir}/checkpoints`` (ascending).

        Used by the download/redownload path to pick the highest *existing*
        remote adapter when the training log is gone (e.g. a redownload to a
        fresh pod after the original container exited). Returns ``[]`` when the
        dir doesn't exist yet (training hasn't checkpointed).
        """
        ...

    def count_precomputed_source(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        precomputed_dir: str,
        source: str,
    ) -> int:
        """Number of ``.pt`` files under ``{precomputed_dir}/{source}`` (recursive).

        Used by the post-preprocess guard to verify ``process_dataset.py``
        actually wrote a given source — notably ``audio_latents`` when
        ``with_audio`` is on, since audio extraction can silently produce 0
        files while the command still exits 0 (e.g. the audio model failed to
        load), which otherwise only surfaces later as a cryptic
        ``No valid samples found`` at training start. Returns 0 when the dir
        doesn't exist. Best-effort: a transport error returns 0 rather than
        raising, so a transient SSH/WSL hiccup can't falsely fail a run.
        """
        ...

    def delete_remote_paths(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        paths: list[str],
    ) -> None:
        """Recursively delete one or more remote paths (files or directories).

        Used by the "reset" action to clear a stage's intermediate artifacts on
        the workspace before re-running it from scratch (e.g. a training run's
        ``checkpoints/`` + ``samples/`` output dir, or a dataset's cached
        ``.precomputed/`` latents). Best-effort per path: a missing path is not
        an error (reset must work even if a prior run died before writing
        anything). Raises ``TrainerTargetError`` only on a transport failure.
        """
        ...
