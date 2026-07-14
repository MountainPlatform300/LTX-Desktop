"""Local `TrainerTarget` (runs the trainer inside WSL2 on the user's machine).

A Windows user with a CUDA-capable GPU can train a LoRA without renting a
remote pod: the same Linux trainer commands built by `lora_command_builder`
run inside WSL2. This target mirrors `RunPodTrainerTarget` — same generic
"run this command" plumbing, same marker-gated provisioning — but every
operation goes through a `WslRemote` (`wsl.exe` subprocess) instead of SSH to
a remote pod.

There is no compute lifecycle to manage: the GPU is the local machine's, so
`ensure_workspace` just makes the workspace directory and `release_workspace`
is a no-op (the workspace is kept for reuse). Network volumes are a
RunPod-only concept and are rejected here.
"""

from __future__ import annotations

import logging
import math
import shlex
import time
from dataclasses import dataclass

from handlers import lora_command_builder as paths
from services.trainer_target.trainer_target import (
    AccountInfo,
    GpuOffer,
    GpuTelemetry,
    NetworkVolume,
    NVIDIA_SMI_GPU_QUERY,
    PodInfo,
    ProgressCallback,
    RemoteCommandStatus,
    TrainerCredentials,
    TrainerTargetError,
    ValidationArtifact,
    parse_nvidia_smi_output,
    parse_samples_listing,
    samples_dir_for,
    checkpoints_dir_for,
    parse_checkpoints_listing,
)
from services.trainer_target.wsl_remote import WslConnection, WslRemote
from state.lora_training_state import TargetHandle

# Logs flow to stdout (captured by Electron into the in-app log viewer).
# Prefix with "Local:" so WSL-backed training is easy to spot/filter there.
logger = logging.getLogger(__name__)

# Provisioning (clone trainer + uv sync + optional multi-GB model download) can
# be slow on a cold WSL distro; generous ceiling, gated by the marker so it only
# ever runs once per workspace.
_PROVISION_TIMEOUT_SECONDS = 3600
# Poll provisioning fairly often so the live download % visibly climbs.
_PROVISION_POLL_INTERVAL_SECONDS = 8

# Minimum VRAM (GB) required for local LTX-2 training. The LTX-2 low-VRAM
# config targets ~32GB (RTX 5090); the 5090 reports 32510 MiB which floors to
# 31 GB via integer-floor (memory_total_MiB // 1024), so 31 is the practical
# floor that still admits a 5090 while rejecting smaller cards.
LOCAL_MIN_VRAM_GB = 31


@dataclass(frozen=True)
class LocalTrainerEligibility:
    """Read-only capability snapshot for the local (WSL2) training backend.

    Produced by `LocalTrainerTarget.probe_eligibility`; the UI polls it to
    decide whether to offer "train locally". Always returned (never raised):
    every failure mode maps to ``eligible=False`` with a human-readable
    `reason`, plus the granular flags so the UI can tailor its guidance.
    """

    eligible: bool
    reason: str
    wsl_installed: bool
    cuda_in_wsl: bool
    gpu_name: str | None
    vram_gb: int | None


def _latest_progress_line(lines: list[str]) -> str:
    """Last meaningful line of a log tail, for live progress (truncated)."""
    for raw in reversed(lines):
        line = raw.strip()
        if line:
            return line if len(line) <= 140 else line[:139] + "…"
    return ""


class LocalTrainerTarget:
    def __init__(self, distro: str | None = None) -> None:
        # `distro=None` targets the user's default WSL distribution. Run every
        # command as root: the workspace is under /root/.ltx-desktop-lora and
        # the model-load overcommit fix needs `sysctl -w` (root-only). This
        # also makes setup independent of the distro's default user — a distro
        # installed outside the in-app wizard (first-launch user prompt) defaults
        # to a non-root user, which made `mkdir /root/.ltx-desktop-lora` fail
        # with "Permission denied".
        self._wsl = WslRemote(WslConnection(distro, user="root"))

    def probe_eligibility(self) -> LocalTrainerEligibility:
        """Report whether local (WSL2) LoRA training is possible on this box.

        A read-only capability probe the UI polls; it NEVER raises. Any
        failure (missing ``wsl.exe``, no distro, broken CUDA-on-WSL, too
        little VRAM) is reported as an ineligible result with a `reason`
        and the granular flags set accordingly. Checks, in order:

          1. WSL present (a distro responds) — else `wsl_installed=False`.
          2. CUDA works in WSL (``nvidia-smi`` runs) — else `cuda_in_wsl=False`.
          3. VRAM gate — the GPU has at least `LOCAL_MIN_VRAM_GB`.
        """
        # 1. Is WSL installed with at least one distro? `-l -q` lists distro
        # names; a missing `wsl.exe` raises FileNotFoundError inside the
        # WslRemote wrapper (surfaced as TrainerTargetError), which we treat
        # as "not installed". `_run_wsl` runs `bash -lc`, so a successful
        # `echo` proves a usable distro exists (an installed-but-distroless
        # WSL fails to launch bash).
        try:
            code, _, err = self._wsl.run("echo ok")
        except TrainerTargetError:
            return LocalTrainerEligibility(
                eligible=False,
                reason=(
                    "WSL2 is not installed. Run `wsl --install` (admin + "
                    "reboot), then install a Linux distro."
                ),
                wsl_installed=False,
                cuda_in_wsl=False,
                gpu_name=None,
                vram_gb=None,
            )
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            logger.warning("Local eligibility: unexpected WSL probe error: %s", exc)
            return LocalTrainerEligibility(
                eligible=False,
                reason=(
                    "WSL2 is not installed. Run `wsl --install` (admin + "
                    "reboot), then install a Linux distro."
                ),
                wsl_installed=False,
                cuda_in_wsl=False,
                gpu_name=None,
                vram_gb=None,
            )
        if code != 0:
            logger.info("Local eligibility: WSL probe returned %d: %s", code, err.strip())
            return LocalTrainerEligibility(
                eligible=False,
                reason=(
                    "WSL2 is not installed or has no Linux distro. Run "
                    "`wsl --install` (admin + reboot), then install a distro."
                ),
                wsl_installed=False,
                cuda_in_wsl=False,
                gpu_name=None,
                vram_gb=None,
            )

        # 2. Does CUDA work in WSL? Query the GPU name + total VRAM.
        try:
            code, out, err = self._wsl.run(
                "nvidia-smi --query-gpu=name,memory.total "
                "--format=csv,noheader,nounits"
            )
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            logger.warning("Local eligibility: nvidia-smi probe error: %s", exc)
            return LocalTrainerEligibility(
                eligible=False,
                reason=(
                    "WSL is installed but the NVIDIA CUDA-on-WSL driver isn't "
                    "working (nvidia-smi failed in WSL)."
                ),
                wsl_installed=True,
                cuda_in_wsl=False,
                gpu_name=None,
                vram_gb=None,
            )
        first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        if code != 0 or not first:
            return LocalTrainerEligibility(
                eligible=False,
                reason=(
                    "WSL is installed but the NVIDIA CUDA-on-WSL driver isn't "
                    "working (nvidia-smi failed in WSL)."
                ),
                wsl_installed=True,
                cuda_in_wsl=False,
                gpu_name=None,
                vram_gb=None,
            )
        name, _, mem = first.partition(",")
        gpu_name = name.strip() or None
        try:
            vram_gb = math.floor(int(mem.strip()) / 1024)
        except ValueError:
            # nvidia-smi ran but its VRAM output was unparseable — treat as a
            # broken CUDA-on-WSL setup rather than guessing.
            return LocalTrainerEligibility(
                eligible=False,
                reason=(
                    "WSL is installed but the NVIDIA CUDA-on-WSL driver isn't "
                    "working (nvidia-smi failed in WSL)."
                ),
                wsl_installed=True,
                cuda_in_wsl=False,
                gpu_name=gpu_name,
                vram_gb=None,
            )

        # 3. VRAM gate.
        if vram_gb < LOCAL_MIN_VRAM_GB:
            return LocalTrainerEligibility(
                eligible=False,
                reason=(
                    f"GPU has {vram_gb} GB VRAM; local LTX-2 training needs "
                    "~32 GB (RTX 5090 or better). Use RunPod instead."
                ),
                wsl_installed=True,
                cuda_in_wsl=True,
                gpu_name=gpu_name,
                vram_gb=vram_gb,
            )

        return LocalTrainerEligibility(
            eligible=True,
            reason="",
            wsl_installed=True,
            cuda_in_wsl=True,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
        )

    def test_connection(self, *, credentials: TrainerCredentials) -> None:
        del credentials
        code, _, err = self._wsl.run("nvidia-smi -L")
        if code != 0:
            raise TrainerTargetError(
                "WSL2 with CUDA is not available — install WSL2 and the NVIDIA "
                f"CUDA driver for WSL, then retry. Detail: {err.strip() or 'nvidia-smi failed'}",
                retryable=False,
            )

    def connect_account(self, *, credentials: TrainerCredentials) -> AccountInfo:
        del credentials
        code, out, err = self._wsl.run(
            "nvidia-smi --query-gpu=name,memory.total "
            "--format=csv,noheader,nounits"
        )
        if code != 0:
            raise TrainerTargetError(
                "Could not query the local GPU via nvidia-smi in WSL2: "
                f"{err.strip() or 'command failed'}",
                retryable=False,
            )
        first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        if not first:
            raise TrainerTargetError(
                "nvidia-smi returned no GPU — WSL2 has no CUDA device available",
                retryable=False,
            )
        name, _, mem = first.partition(",")
        try:
            memory_gb = math.floor(int(mem.strip()) / 1024)
        except ValueError:
            memory_gb = 0
        offer = GpuOffer(
            id="local",
            label=name.strip() or "Local GPU",
            memory_gb=memory_gb,
        )
        logger.info(
            "Local: connected — GPU '%s' (%d GB) in WSL2", offer.label, memory_gb
        )
        return AccountInfo(gpus=(offer,), volumes=(), pods=())

    def list_pods(self, *, credentials: TrainerCredentials) -> list[PodInfo]:
        # Local training runs on the user's own machine — there are no
        # provider pods to list. The compute panel is RunPod-only; returning []
        # keeps the routing target provider-agnostic.
        del credentials
        return []

    def stop_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        del credentials, pod_id
        raise TrainerTargetError(
            "local training has no provider pods to stop", retryable=False
        )

    def resume_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        del credentials, pod_id
        raise TrainerTargetError(
            "local training has no provider pods to resume", retryable=False
        )

    def ensure_network_volume(
        self,
        *,
        credentials: TrainerCredentials,
        name: str,
        size_gb: int,
        datacenter_id: str | None = None,
    ) -> NetworkVolume:
        del credentials, name, size_gb, datacenter_id
        raise TrainerTargetError(
            "network volumes are not used for local training", retryable=False
        )

    def delete_network_volume(
        self, *, credentials: TrainerCredentials, volume_id: str
    ) -> None:
        del credentials, volume_id
        raise TrainerTargetError(
            "network volumes are not used for local training", retryable=False
        )

    def ensure_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle | None
    ) -> TargetHandle:
        del handle
        code, _, err = self._wsl.run(
            "mkdir -p " + shlex.quote(credentials.workspace_dir)
        )
        if code != 0:
            raise TrainerTargetError(
                f"Failed to create local workspace {credentials.workspace_dir}: {err}",
                retryable=True,
            )
        return TargetHandle(provider="local")

    def ensure_provisioned(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        progress: ProgressCallback | None = None,
    ) -> None:
        del handle
        if not credentials.auto_provision:
            return
        if not credentials.trainer_repo_url:
            raise TrainerTargetError(
                "Auto-provisioning is enabled but no trainer repo URL is "
                "configured (set loraTrainerRepoUrl or disable loraAutoProvision)",
                retryable=False,
            )
        workspace = credentials.workspace_dir
        marker = paths.provision_marker_path(workspace)
        marker_value = paths.provision_marker_value(
            credentials.trainer_repo_url,
            credentials.trainer_repo_ref or "main",
        )
        uv_bin = paths.uv_bin_dir(workspace)
        # Match RunPod's exact-source marker check so a persisted local WSL
        # workspace cannot silently keep running a different trainer revision.
        code, _, _ = self._wsl.run(
            f'export PATH={shlex.quote(uv_bin)}:"$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; '
            f"test -f {shlex.quote(marker)} "
            f"&& test \"$(cat {shlex.quote(marker)})\" = {shlex.quote(marker_value)} "
            "&& command -v uv >/dev/null 2>&1"
        )
        if code != 0:
            logger.info(
                "Local: provisioning workspace at %s — cloning trainer (%s@%s), uv "
                "sync, downloading checkpoint '%s' + encoder (this can take a while "
                "on a cold distro)",
                workspace,
                credentials.trainer_repo_url,
                credentials.trainer_repo_ref or "main",
                credentials.model_filename or credentials.model_hf_repo,
            )
            command = paths.provision_command(
                workspace_dir=workspace,
                repo_url=credentials.trainer_repo_url,
                repo_ref=credentials.trainer_repo_ref or "main",
                marker_path=marker,
                model_hf_repo=credentials.model_hf_repo,
                model_filename=credentials.model_filename,
                model_path=credentials.model_path,
                text_encoder_hf_repo=credentials.text_encoder_hf_repo,
                text_encoder_path=credentials.text_encoder_path,
                hf_token=credentials.hf_token,
                # Local training targets a Blackwell GPU (RTX 5090, sm_120);
                # provision must resolve torch from the cu128 index, not PyPI.
                # Install gcc-14/g++-14 too: WSL's default gcc-15 is rejected by
                # nvcc, which blocks the optimum-quanto CUDA extension build and
                # surfaces as `device not ready` on sm_120.
                torch_cuda_index="cu128",
                install_host_compiler=True,
            )
            self._run_polled(
                command=command,
                workdir=workspace,
                progress=progress,
                timeout=_PROVISION_TIMEOUT_SECONDS,
                poll_interval=_PROVISION_POLL_INTERVAL_SECONDS,
                begin_message="Cloning trainer & installing dependencies…",
                step="workspace provisioning",
            )
        # In-place torch upgrade for workspaces provisioned before the Blackwell
        # fix (or any workspace whose `uv sync` didn't pin cu128). Gated on the
        # torch-index marker, so a fresh cu128 provision (which already touched
        # it) skips this entirely. Re-syncs torch to 2.9.1+cu128 (sm_120).
        torch_marker = paths.torch_index_marker_path(workspace)
        code, _, _ = self._wsl.run("test -f " + shlex.quote(torch_marker))
        if code != 0:
            logger.info("Local: upgrading trainer torch to cu128 (Blackwell sm_120)")
            self._run_polled(
                command=paths.ensure_torch_index_command(
                    workspace_dir=workspace,
                    marker_path=torch_marker,
                    install_host_compiler=True,
                ),
                workdir=workspace,
                progress=progress,
                timeout=_PROVISION_TIMEOUT_SECONDS,
                poll_interval=_PROVISION_POLL_INTERVAL_SECONDS,
                begin_message="Upgrading torch for RTX 5090 (cu128)…",
                step="torch cu128 upgrade",
            )

        # Ensure system ffmpeg for torchaudio's audio backend (demuxing audio
        # out of mp4/mov). A fresh provision already installs it, but a
        # workspace provisioned before this fix never got it. Idempotent no-op
        # where ffmpeg exists; never fatal — a missing ffmpeg only breaks audio
        # training, which the post-preprocess audio guard reports clearly.
        try:
            self._wsl.run(paths.ensure_ffmpeg_command())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local: ensure-ffmpeg skipped (%s)", exc)

        # Backfill the audio-fallback monkeypatch on `process_videos.py` (see
        # `patch_trainer_audio_fallback_command`): on WSL cu128, torchaudio 2.9+
        # routes `torchaudio.load` through torchcodec, whose bundled libav ABI
        # doesn't match system ffmpeg — so `torchaudio.load(<mp4>)` raises even
        # with ffmpeg installed, silently skipping every clip's audio and
        # emptying `audio_latents/`. The patch wraps `torchaudio.load` with an
        # ffmpeg-subprocess fallback (no-op where `torchaudio.load` succeeds).
        # Idempotent + best-effort.
        try:
            self._wsl.run(paths.patch_trainer_audio_fallback_command(workspace))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local: trainer audio-fallback patch skipped (%s)", exc)

    def _run_polled(
        self,
        *,
        command: str,
        workdir: str,
        progress: ProgressCallback | None,
        timeout: float,
        poll_interval: float,
        begin_message: str,
        step: str,
    ) -> None:
        """Run a long setup command detached in WSL and poll until it finishes.

        Streams the last meaningful log line to `progress` so a slow provision /
        upgrade visibly advances instead of appearing hung. Raises
        `TrainerTargetError` on failure or timeout (mirrors the RunPod target's
        detached-run handling). Shared by the provision and torch-upgrade steps.
        """
        job_id = self._wsl.run_detached(command=command, workdir=workdir)
        if progress is not None:
            progress(begin_message)
        deadline = time.monotonic() + timeout
        last_reported = ""
        while time.monotonic() < deadline:
            status = self._wsl.poll(job_id)
            if status.state == "succeeded":
                logger.info("Local: %s complete", step)
                if progress is not None:
                    progress("Setup complete")
                return
            if status.state == "failed":
                tail = self._wsl.read_logs(job_id, 40)
                detail = (
                    " | ".join(tail[-12:])
                    if tail
                    else (status.error or "unknown error")
                )
                logger.error("Local: %s failed: %s", step, detail)
                raise TrainerTargetError(
                    f"{step} failed: {detail}", retryable=False
                )
            line = _latest_progress_line(self._wsl.read_logs(job_id, 8))
            if line and line != last_reported:
                last_reported = line
                logger.info("Local setup: %s", line)
                if progress is not None:
                    progress(line)
            time.sleep(poll_interval)
        tail = self._wsl.read_logs(job_id, 30)
        where = " | ".join(t for t in tail[-8:] if t.strip()) or "no output"
        logger.error("Local: %s timed out — last: %s", step, where)
        raise TrainerTargetError(
            f"{step} timed out. Last output: {where}",
            retryable=True,
        )

    def upload_directory(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        local_dir: str,
        remote_dir: str,
    ) -> None:
        del credentials, handle
        self._wsl.upload_directory(local_dir=local_dir, remote_dir=remote_dir)

    def download_file(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_path: str,
        local_path: str,
    ) -> None:
        del credentials, handle
        self._wsl.download_file(remote_path=remote_path, local_path=local_path)

    # The trainer loads the multi-GB checkpoint by mmap'ing a file larger than
    # the box's RAM. Linux's default overcommit heuristic
    # (vm.overcommit_memory=0) rejects that mapping once torch/CUDA are already
    # resident — `mmap ... Cannot allocate memory`, which killed preprocess and
    # training at model-load. Mode 1 ("always overcommit") lets the read-only,
    # demand-paged weight mapping through; it's the standard setting for mmap'ing
    # large model weights and safe here (the pages are file-backed and evictable,
    # not anonymous commit). We run as root in the distro, so `sysctl -w` works;
    # `|| true` keeps it best-effort. Set per-command so it survives a distro
    # restart between runs without a persisted config.
    _OVERCOMMIT_PREFIX = "sysctl -w vm.overcommit_memory=1 >/dev/null 2>&1 || true\n"

    def start_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        command: str,
        workdir: str,
    ) -> str:
        del credentials, handle
        return self._wsl.run_detached(
            command=self._OVERCOMMIT_PREFIX + command, workdir=workdir
        )

    def poll_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> RemoteCommandStatus:
        del credentials, handle
        return self._wsl.poll(remote_job_id)

    def read_logs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
        tail: int,
    ) -> list[str]:
        del credentials, handle
        return self._wsl.read_logs(remote_job_id, tail)

    def terminate(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> None:
        del credentials, handle
        self._wsl.terminate(remote_job_id)

    def release_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> None:
        # No compute to release: the GPU is the local machine's. Keep the
        # workspace (trainer install, cached weights/latents) for reuse.
        del credentials, handle
        return

    def query_gpu(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> GpuTelemetry:
        del credentials, handle
        code, out, err = self._wsl.run(NVIDIA_SMI_GPU_QUERY)
        if code != 0:
            raise TrainerTargetError(
                "Local: nvidia-smi failed in WSL2: "
                f"{err.strip() or 'command failed'}",
                retryable=True,
            )
        return parse_nvidia_smi_output(out)

    def list_validation_outputs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
        since_step: int,
    ) -> list[ValidationArtifact]:
        del credentials, handle
        samples_dir = samples_dir_for(remote_output_dir)
        # Fixed `ls -1` against the run's own samples dir (server-derived, not
        # user-typed); `|| true` so a not-yet-created dir yields an empty list
        # rather than an error. shlex-quote the path defensively.
        code, out, _ = self._wsl.run(
            f"ls -1 {shlex.quote(samples_dir)} 2>/dev/null || true"
        )
        if code != 0:
            return []
        return [
            a for a in parse_samples_listing(out, samples_dir) if a.step > since_step
        ]

    def list_checkpoints(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
    ) -> list[int]:
        del credentials, handle
        ckpt_dir = checkpoints_dir_for(remote_output_dir)
        # Fixed `ls -1` against the run's own checkpoints dir (server-derived,
        # not user-typed); `|| true` so a not-yet-created dir yields an empty
        # list rather than an error. shlex-quote the path defensively.
        code, out, _ = self._wsl.run(
            f"ls -1 {shlex.quote(ckpt_dir)} 2>/dev/null || true"
        )
        if code != 0:
            return []
        return parse_checkpoints_listing(out)

    def count_precomputed_source(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        precomputed_dir: str,
        source: str,
    ) -> int:
        del credentials, handle
        src_dir = f"{precomputed_dir.rstrip('/')}/{source}"
        # Recursive .pt count under the source dir; `2>/dev/null` so a missing
        # source (process_dataset.py wrote nothing for it) yields 0 instead of
        # an error. shlex-quote the server-derived path defensively.
        code, out, _ = self._wsl.run(
            f"find {shlex.quote(src_dir)} -type f -name '*.pt' 2>/dev/null | wc -l"
        )
        if code != 0:
            return 0
        try:
            return max(0, int(out.strip()))
        except ValueError:
            return 0

    def delete_remote_paths(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        paths: list[str],
    ) -> None:
        del credentials, handle
        if not paths:
            return
        # `rm -rf` each quoted path; `2>/dev/null || true` per path so a missing
        # path (a prior run that died before writing anything) isn't an error —
        # reset must succeed regardless of how far the previous run got.
        quoted = " ".join(shlex.quote(p) for p in paths)
        code, _, err = self._wsl.run(f"rm -rf {quoted} 2>/dev/null || true")
        if code != 0:
            raise TrainerTargetError(
                f"Local: failed to delete remote paths: {err.strip() or 'rm failed'}",
                retryable=True,
            )
