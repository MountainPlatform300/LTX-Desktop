"""Background reconciler for the LoRA-trainer control plane.

Unlike the queue runner (which claims a pending item and runs it
synchronously to completion), remote training jobs are long-lived and
asynchronous, so this is a *reconciliation* loop: each tick it walks
the non-terminal entities, advances each one step toward its terminal
state, and persists. A step is either "submit the next remote command"
or "poll the command we already submitted" — never a blocking wait on
the whole job.

Per tick, in dependency order:

1. **Uploads** — for each dataset in `uploading`: provision compute,
   stage clips + (manual) captions locally, upload to the remote, mark
   `uploaded`. Uploads are blocking transfers; acceptable for a
   single-user desktop control plane (see note below).
2. **Preprocessing** — drive the `pending -> captioning -> preprocessing
   -> ready` machine. Captioning (optional) and `process_dataset.py`
   each run as a remote command; we submit, then poll on later ticks.
3. **Training** — drive `pending -> running -> completed`. Generate the
   trainer YAML, upload it, run `train.py`, poll, then download
   `lora_weights.safetensors` on success.

Compute affinity: a dataset's uploaded clips, its `.precomputed`
latents, and every training run derived from it all live on the *same*
remote workspace (the pod/workspace recorded in the dataset's
`TargetHandle`). Preprocessing and training reuse that handle rather
than allocating fresh compute, which is what makes "preprocess once,
train many" cheap. The pod is released only when the dataset is deleted.

Single-thread note: one reconciler thread does everything serially. A
long blocking upload therefore delays polling of an in-flight training
job by at most one upload. For one user that's fine; if this ever needs
to fan out, uploads should move onto the task runner. Documented rather
than over-engineered for v1.

Lock discipline mirrors the rest of the backend: read a snapshot from
the handler (which locks internally), do all remote/IO work *without*
holding the app lock, then call a handler transition (which locks) to
write the result.
"""

from __future__ import annotations

import logging
import hashlib
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from handlers import lora_command_builder as paths
from handlers import lora_dataset_prep as prep
from api_types import DEFAULT_VALIDATION_PROMPT
from handlers.lora_config_builder import (
    build_training_yaml,
    build_validation_sample_dicts,
    build_validation_sample_refs,
    preset_loads_text_encoder_in_8bit,
    preset_reference_downscale_factor,
    ValidationSampleSpec,
)
from handlers.lora_export import BundleError, build_dataset_rows, plan_clip_layout, refs_of
from handlers.lora_run_summary import build_run_summary_markdown
from handlers.lora_training_handler import (
    LoraTrainingHandler,
    LoraTransitionError,
    ReconcileEntityKind,
)
from logging_policy import log_background_exception
from secret_redaction import redact_text
from services.clip_processor.clip_processor import (
    ClipProcessor,
    ClipProcessorError,
    EditPlan,
    ScaleSpec,
)
from state.app_settings import AppSettingsPatch
from services.trainer_target.trainer_target import (
    AccountInfo,
    NetworkVolume,
    PodInfo,
    TrainerCredentials,
    TrainerTarget,
    TrainerTargetError,
    ValidationArtifact,
)
from state.lora_training_state import (
    GpuStatus,
    LoraClip,
    LoraDataset,
    LoraDatasetType,
    PreprocessedDataset,
    TargetHandle,
    TrainerProvider,
    TrainingJob,
    WorkspacePolicy,
    RunpodSelection,
    SavedModelReadiness,
    ValidationFeedItem,
    CheckpointArtifact,
)

if TYPE_CHECKING:
    from handlers.settings_handler import SettingsHandler
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_settings import AppSettings

logger = logging.getLogger(__name__)

_BACKGROUND_TASK_NAME = "lora-trainer-runner"
DEFAULT_POLL_SECONDS: float = 15.0
DEFAULT_STOP_JOIN_TIMEOUT_SECONDS: float = 120.0
_LOG_TAIL = 100
# Cap GPU telemetry queries at one per job per this many seconds. The
# reconciler already polls every ~15s; this just guards a faster cycle from
# hammering nvidia-smi while the GPU is busy training.
_GPU_QUERY_MIN_INTERVAL_SECONDS = 10.0

# Auto-created network volume for model caching. Sized for the LTX-2
# checkpoint (~44 GB) + Gemma encoder (~24 GB) + datasets/precomputed
# latents + output LoRAs, with headroom. Name is stable so reconnects
# reuse the same volume instead of creating duplicates.
RUNPOD_VOLUME_NAME = "ltx-desktop-lora"
DEFAULT_RUNPOD_VOLUME_SIZE_GB = 250
ESTIMATED_MODEL_DOWNLOAD_BYTES = 72 * 1024 * 1024 * 1024
# RunPod mounts a network volume at /workspace by default, and that's where
# auto-provisioning installs the trainer + downloads weights. When the app
# manages the pod (auto-provision on) this is the workspace, regardless of any
# stale user override.
RUNPOD_MANAGED_WORKSPACE_DIR = "/workspace"

# Standard-dataset upload guard: clips whose short side exceeds this are
# downscaled before upload. The trainer resizes/crops every clip to the training
# bucket anyway, but it decodes the *source* at full resolution first, so a 4K/8K
# import risks preprocessing OOM and a huge upload. 768 covers every practical
# LTX training bucket short side (448/512/544/768) so we never force the trainer
# to upscale; clips at or under it are copied byte-for-byte (no re-encode).
# IC-LoRA has its own (tighter) cap in `lora_dataset_prep`.
STANDARD_MAX_SHORT_SIDE = 768

# Canonical LTX-2 trainer weights (mirror the AppSettings defaults). Used as
# a fallback when a persisted setting is blank — older/legacy settings files
# stored empty repos, which would silently skip the model download. The model
# repo is ~314 GB, so only the checkpoint file is pulled (see command builder).
DEFAULT_MODEL_HF_REPO = "Lightricks/LTX-2.3"
DEFAULT_MODEL_CHECKPOINT_FILE = "ltx-2.3-22b-dev.safetensors"
DEFAULT_TEXT_ENCODER_HF_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


# Local (WSL2) training runs entirely inside the distro, on its NATIVE ext4
# filesystem — NOT the `/mnt/c` (9p/DrvFs) Windows mount. DrvFs cannot mmap large
# files, and `safetensors` mmaps the multi-GB checkpoint, so a workspace on
# `/mnt/c` fails to load the model (`mmap ... Cannot allocate memory`); ext4 also
# makes the heavy training I/O dramatically faster. The dataset is copied in from
# Windows and the trained LoRA copied back out, so nothing the user needs stays
# trapped in the distro. The setup wizard installs a root-default Ubuntu, so the
# workspace lives under /root.
LOCAL_WSL_WORKSPACE_DIR = "/root/.ltx-desktop-lora"

# Qwen3-Omni-30B (the qwen_omni captioner) loads ~31 GiB of FP8 weights onto
# the GPU via a vLLM server, so it needs a >=40 GiB card. Below this the server
# OOMs on startup; gate it upfront and point the user at Gemini Flash instead.
MIN_QWEN_CAPTIONER_VRAM_GB = 40

# Best-effort progress: matches "... step 123/2000 ..." in trainer logs.
_STEP_RE = re.compile(r"step[^\d]*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
# Optional loss capture from the same step line, e.g. "... loss 0.234 ...".
_LOSS_RE = re.compile(r"\bloss[^\d.]*([\d.]+)", re.IGNORECASE)


def _format_eta(seconds: int | None) -> str:
    """Human-readable ETA for a training-progress log line, or "—" if unknown."""
    if seconds is None or seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _latest_meaningful_line(lines: list[str]) -> str:
    """Last non-empty, stripped log line (truncated) — a pre-step heartbeat."""
    for raw in reversed(lines):
        line = raw.strip()
        if line:
            return line if len(line) <= 200 else line[:199] + "…"
    return ""


def _training_setup_phase(line: str) -> str:
    """Map noisy trainer startup logs to a stable user-facing phase."""
    low = line.lower()
    if "validat" in low or "sample" in low:
        return "Running initial validation…"
    if "quantiz" in low or "quanto" in low:
        return "Quantizing model…"
    if "load" in low or "checkpoint" in low or "safetensor" in low:
        return "Loading model…"
    if "dataset" in low or "dataloader" in low or "latent" in low:
        return "Loading training data…"
    return "Preparing training…"


_PCT_RE = re.compile(r"(\d{1,3})\s*%")
# tqdm renders the remaining time after "<" in "[elapsed<remaining, rate]".
_ETA_RE = re.compile(r"<\s*(\d+):(\d+)(?::(\d+))?")


def _eta_seconds(text: str) -> int | None:
    """Parse a tqdm ``[elapsed<remaining]`` ETA (``mm:ss`` or ``hh:mm:ss``) -> seconds.

    Shared by the setup and preprocess progress parsers so the bar-ETA decode
    lives in one place.
    """
    em = _ETA_RE.search(text)
    if not em:
        return None
    nums = [int(x) for x in em.groups() if x is not None]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


def _bar_percent(text: str) -> int | None:
    """Parse a tqdm/git bar percent (clamped to 100), or None if no bar present."""
    pm = _PCT_RE.search(text)
    return min(100, int(pm.group(1))) if pm else None


def _parse_setup_progress(line: str) -> tuple[str, int | None, int | None]:
    """Turn a remote setup log line into (phase label, percent, eta_seconds).

    Handles our own `[provision] <step>` markers (clean phase, no bar) and
    tqdm/git download bars (percent + ETA parsed from the bar itself).
    """
    text = line.strip()
    low = text.lower()
    # Our own step markers → clean phase labels (definitive).
    if "[provision]" in low:
        if "cloning" in low:
            return "Cloning trainer", None, None
        if "dependencies" in low or "uv sync" in low:
            return "Installing dependencies", None, None
        if "encoder" in low:
            return "Downloading text encoder", None, None
        if "model" in low:
            return "Downloading training model", None, None
        if "done" in low:
            return "Finishing setup", None, None
        return "Preparing environment", None, None
    # A progress bar with a percent → a file download (checkpoint/encoder) or
    # a git clone phase. Parse % and ETA from the bar.
    pct = _bar_percent(text)
    eta = _eta_seconds(text)
    if pct is not None:
        if "objects" in low or "updating files" in low or "cloning" in low:
            return "Cloning trainer", pct, None
        if any(k in low for k in ("text_encoder", "text-encoder", "gemma", "encoder")):
            return "Downloading text encoder", pct, eta
        return "Downloading training model", pct, eta
    # No percent: uv dependency install — surface the package name so the card
    # visibly ticks through them (otherwise it reads as stuck for minutes).
    dep = re.search(r"download(?:ed|ing)\s+([A-Za-z0-9._-]+)", low)
    if dep and ".safetensors" not in low:
        return f"Installing dependencies — {dep.group(1)}", None, None
    if any(k in low for k in ("resolved", "installed", "built", "installing", "uv sync")):
        return "Installing dependencies", None, None
    if "cloning" in low or "receiving objects" in low or "updating files" in low:
        return "Cloning trainer", None, None
    return "Setting up environment", None, None


_COUNT_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
# Trainer logs each save as "... saved in checkpoints/lora_weights_step_02000.safetensors".
_CKPT_RE = re.compile(r"lora_weights_step_(\d+)\.safetensors")


def _latest_checkpoint_step(log_lines: list[str]) -> int | None:
    """Highest `lora_weights_step_N` step mentioned in the training log, or None.

    The trainer saves one checkpoint per interval plus a final save at the last
    step, so the largest N is the most-trained adapter to download.
    """
    steps = [int(m.group(1)) for line in log_lines for m in _CKPT_RE.finditer(line)]
    return max(steps) if steps else None


def _parse_preprocess_progress(
    tail: list[str],
) -> tuple[str, int | None, int | None] | None:
    """Turn a `process_dataset.py` log tail into (phase, percent, eta_seconds).

    Scans from the newest line back for the first informative signal: a tqdm
    bar (percent + ETA, or an `n/m` count we convert to a percent) or a known
    phase keyword (model load / caching). Returns None if nothing is parseable
    yet (e.g. only blank lines), so the caller leaves the current detail as-is.
    """
    for line in reversed(tail):
        text = line.strip()
        if not text:
            continue
        low = text.lower()
        eta = _eta_seconds(text)
        percent = _bar_percent(text)
        if percent is None:
            cm = _COUNT_RE.search(text)
            if cm and int(cm.group(2)) > 0:
                percent = min(100, round(100 * int(cm.group(1)) / int(cm.group(2))))
        if any(k in low for k in ("caption", "captioning")):
            return ("Captioning clips", percent, eta)
        if any(k in low for k in ("loading", "checkpoint", "vae", "encoder")):
            return ("Loading models", percent, eta)
        if any(
            k in low for k in ("precompute", "caching", "latent", "encod", "process")
        ):
            return ("Caching latents", percent, eta)
        if percent is not None:
            return ("Caching latents", percent, eta)
    return None


def _even_dim(value: float) -> int:
    """Round to the nearest even pixel dimension (>=2), as h.264 requires."""
    return max(2, round(value / 2.0) * 2)


def _audio_skip_summary(tail: list[str]) -> str | None:
    """Pull the trainer's audio-skip self-report out of a `process_dataset` log tail.

    `compute_latents` logs, at the end of the latents phase:
        `Audio processing: <ok> videos with audio, <skipped> without audio (skipped)`
    and per-clip `Could not extract audio from <path>: <error>` at debug level.
    When every clip's audio extraction failed (the common case: torchaudio's
    ffmpeg backend isn't functional on the pod, so `torchaudio.load(<mp4>)`
    falls back to soundfile which can't demux video containers), `<ok>` is 0 and
    `<skipped>` equals the clip count — the smoking gun. Returns a short string
    surfacing that line (and the first per-clip error) so the prep-failed
    message says *why* audio latents came out empty instead of a generic
    "audio model may be missing".
    """
    summary = None
    for line in tail:
        m = re.search(r"Audio processing:\s*(\d+)\s*videos? with audio,\s*(\d+)\s*without", line)
        if m:
            ok, skipped = int(m.group(1)), int(m.group(2))
            summary = f"trainer reported {ok} clip(s) with audio, {skipped} skipped"
            break
    first_err = None
    for line in tail:
        m = re.search(r"Could not extract audio from [^:]+:\s*(.+)", line)
        if m:
            first_err = m.group(1).strip()
            break
    if summary and first_err:
        return f"{summary}; first error: {first_err}"
    return summary


def _lora_weights_filename(job: TrainingJob) -> str:
    """Descriptive adapter filename, e.g. `cleanplate-rank32-2000steps.safetensors`,
    so the file is identifiable in Finder instead of a bare UUID."""
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", job.name or "").strip("-._").lower()[:40]
    stem = clean or "lora"
    steps = job.total_steps or job.config.steps
    return f"{stem}-rank{job.config.rank}-{steps}steps.safetensors"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class LoraTrainingRunner:
    def __init__(
        self,
        *,
        handler: LoraTrainingHandler,
        trainer_target: TrainerTarget,
        settings_handler: "SettingsHandler",
        config: "RuntimeConfig",
        clip_processor: ClipProcessor,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        stop_join_timeout_seconds: float = DEFAULT_STOP_JOIN_TIMEOUT_SECONDS,
        free_inference_gpu: Callable[[], None] | None = None,
    ) -> None:
        self._handler = handler
        self._target = trainer_target
        self._settings = settings_handler
        self._config = config
        self._clip_processor = clip_processor
        self._poll_seconds = poll_seconds
        self._stop_join_timeout = stop_join_timeout_seconds
        # Local WSL2 training shares the single GPU with the Windows inference
        # server, which keeps the 22B model resident in VRAM after a generation
        # (~31 GB on a 5090). Without freeing it, the trainer's first CUDA op
        # starves on VRAM and crashes mid-run (dxgk fault / hard kill, no exit
        # code). This callback drops the resident pipeline + empties the CUDA
        # cache so the trainer gets the full card; the next inference request
        # reloads. Best-effort: a failure here must never block the run.
        self._free_inference_gpu = free_inference_gpu
        self._wakeup = handler.wakeup_event
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        # In-memory (step, monotonic_time, ema_steps_per_sec) sample per training
        # job for a smoothed step-rate ETA. Not persisted: best-effort, resets on
        # restart (eta just re-warms over the next couple of polls).
        self._train_rate_samples: dict[str, tuple[int, float, float]] = {}
        # Last step/line we already logged per job, so progress logging emits one
        # line per advance (the reconciler polls every ~15s) instead of
        # repeating the same line on every poll.
        self._last_logged_train_step: dict[str, int] = {}
        self._last_logged_train_line: dict[str, str] = {}
        # Last monotonic time we queried GPU telemetry per job, for throttling.
        # nvidia-smi has non-trivial overhead and the GPU is busy training, so
        # we cap queries at one per `_GPU_QUERY_MIN_INTERVAL_SECONDS` per job.
        self._last_gpu_query_at: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle (mirrors QueueRunner)
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name="lora-trainer-runner", daemon=True
            )
            self._thread.start()
            logger.info("LoRA trainer runner started")

    def stop(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._shutdown.set()
            self._wakeup.set()
            self._thread = None
        thread.join(timeout=self._stop_join_timeout)
        if thread.is_alive():
            logger.warning(
                "LoRA trainer runner did not stop within %.0fs; leaving "
                "daemon thread (next boot re-polls via crash recovery)",
                self._stop_join_timeout,
            )
        else:
            logger.info("LoRA trainer runner stopped")

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                self._tick()
            except Exception as exc:
                log_background_exception(_BACKGROUND_TASK_NAME, exc)
            # Always wake periodically: remote jobs need polling even
            # when no producer signals. Producers also set the event to
            # shorten latency for newly-submitted work.
            self._wakeup.wait(timeout=self._poll_seconds)
            self._wakeup.clear()

    def reconcile_once(self) -> None:
        """Run a single reconciliation pass synchronously.

        Exposed for tests (and a potential manual "refresh now" action)
        so the full state machine can be driven deterministically
        without the background thread.
        """
        self._tick()

    def _tick(self) -> None:
        settings = self._settings.get_settings_snapshot()
        for dataset in self._handler.list_datasets_to_upload():
            self._guard_entity(
                "dataset",
                dataset.id,
                lambda dataset=dataset: self._reconcile_upload(dataset, settings),
            )
        active_preprocessed = self._handler.list_active_preprocessed()
        for item in active_preprocessed:
            self._guard_entity(
                "preprocess",
                item.id,
                lambda item=item: self._reconcile_preprocess(item, settings),
            )
        active_training = self._handler.list_active_training()
        for job in active_training:
            self._guard_entity(
                "training",
                job.id,
                lambda job=job: self._reconcile_training(job, settings),
            )
        self._guard(
            "idle-stop",
            lambda: self._reconcile_idle_stops(
                settings, active_preprocessed, active_training
            ),
        )

    def _guard(self, label: str, fn: Callable[[], None]) -> None:
        """Non-entity safety net (e.g. idle-stop): log + swallow.

        Entity-scoped reconcile work goes through `_guard_entity` instead, which
        records the failure on the entity (status_detail + retry budget) rather
        than silently dropping it and leaving the card stuck on its old status.
        """
        try:
            fn()
        except Exception as exc:
            log_background_exception(f"{_BACKGROUND_TASK_NAME}:{label}", exc)

    def _prepare_local_gpu(self, provider: str) -> None:
        """Free the inference server's resident GPU memory before a local run.

        Local WSL2 training shares the single GPU with this process's inference
        server. The 22B model stays resident in VRAM after a generation, leaving
        almost nothing for the trainer — its first CUDA op then crashes mid-run
        (dxgk fault / hard kill, no exit code), which is the recurring "LoRA
        preprocess was killed" failure on a 32 GB card. Dropping the resident
        pipeline + emptying the CUDA cache hands the full card to the trainer;
        the next inference request reloads on demand. No-op for remote providers
        and when no callback was wired. Best-effort: never blocks the run.
        """
        if provider != "local":
            return
        if self._free_inference_gpu is not None:
            try:
                logger.info(
                    "LoRA trainer: freeing inference GPU memory before local run "
                    "(so the trainer gets the full card)"
                )
                self._free_inference_gpu()
            except Exception as exc:
                logger.warning(
                    "LoRA trainer: could not free inference GPU memory (continuing): %s",
                    exc,
                )

    def _guard_entity(
        self,
        kind: ReconcileEntityKind,
        entity_id: str,
        fn: Callable[[], None],
    ) -> None:
        """Entity-scoped reconcile guard with transient-failure accounting.

        On a clean tick, clears any prior retry surface (`status_detail` /
        `consecutive_failures`). On an exception, classifies it:

        - **Transient** (retryable `TrainerTargetError`, generic unexpected
          errors): recorded via `record_transient_failure` — the card shows
          "Retrying after error: …" and the next tick retries. After
          `_TRANSIENT_FAILURE_BUDGET` consecutive failures the handler escalates
          to `failed`/`upload_failed` so a stuck entity can't loop forever.
        - **Fatal** (`BundleError`, `LoraTransitionError`, non-retryable
          `TrainerTargetError`): the entity is marked failed immediately with
          the detail, rather than hanging on its pre-failure status.

        Either way the exception is still logged for diagnosis.
        """
        try:
            fn()
        except Exception as exc:
            transient, detail = self._classify_failure(exc)
            log_background_exception(
                f"{_BACKGROUND_TASK_NAME}:{kind}[{entity_id}]", exc
            )
            if transient:
                escalated = self._handler.record_transient_failure(
                    kind, entity_id, detail
                )
                if escalated:
                    logger.warning(
                        "LoRA %s %s: transient failures exceeded budget; "
                        "marked failed",
                        kind,
                        entity_id,
                    )
            else:
                self._fail_entity(kind, entity_id, detail)
            return
        self._handler.record_reconcile_success(kind, entity_id)

    @staticmethod
    def _classify_failure(exc: Exception) -> tuple[bool, str]:
        """Split an escaped exception into (transient, human-readable detail).

        Unknown exception types default to transient: a one-off surprise should
        not immediately doom a multi-hour run, and the retry budget still
        escalates if it keeps happening.
        """
        if isinstance(exc, TrainerTargetError):
            return exc.retryable, exc.detail
        if isinstance(exc, BundleError | LoraTransitionError):
            return False, (str(exc).strip() or repr(exc))
        # OSError / generic: retry with a budget. `str(exc)` can be empty for
        # bare OSErrors, so fall back to the type name + repr.
        detail = str(exc).strip() or f"{type(exc).__name__}: {exc!r}"
        return True, detail

    def _fail_entity(
        self, kind: ReconcileEntityKind, entity_id: str, detail: str
    ) -> None:
        if kind == "dataset":
            self._handler.fail_dataset_upload(entity_id, detail)
        elif kind == "preprocess":
            self._handler.fail_preprocess(entity_id, detail)
        else:
            self._handler.fail_training(entity_id, detail)

    @staticmethod
    def _remote_failure_detail(
        tail: list[str], exit_code: int | None, error: str | None, *, kind: str
    ) -> str:
        """Build a human-readable failure detail from a remote command's tail.

        Always surfaces the exit code, even when a log tail is present: a
        SIGKILL (exit 137 — system OOM killer or spot preemption) or a segfault
        (exit 139) kills the process before Python can print a traceback, so
        the last log line is often an innocuous warning (e.g. a `torch_dtype`
        deprecation) rather than the cause. The exit code is the only signal
        that distinguishes a hard kill from a normal Python exception (exit 1),
        so it must never be dropped just because there's a tail.
        """
        lines = [t for t in tail if t.strip()]
        body = " | ".join(lines[-12:]) if lines else (error or f"{kind} failed")
        body = redact_text(body)
        suffix = ""
        if exit_code is not None:
            # 137/139 are the common hard-kill codes; spell them out so the
            # card/Log self-diagnoses instead of looking like a trainer bug.
            hint = {
                137: " (SIGKILL — likely OOM killer or spot preemption)",
                139: " (SIGSEGV — native crash, often a CUDA/driver fault)",
            }.get(exit_code, "")
            suffix = f" [exit {exit_code}{hint}]"
        elif lines:
            # No exit code was recorded but the job produced output, so it
            # definitely ran — it was killed before the wrapper could write
            # `$?` to the status file (system OOM killer took the wrapper too,
            # the WSL distro shut down mid-run, or the pod was preempted).
            # Say so explicitly; otherwise the last log line (often an
            # innocuous `torch_dtype` deprecation) reads as the cause.
            suffix = " [no exit code recorded — process was killed mid-run (OOM / distro shutdown / spot preemption)]"
        return f"{body}{suffix}" if body else f"{kind} failed{suffix}"

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------

    def _credentials(
        self,
        settings: "AppSettings",
        provider: TrainerProvider = "runpod",
        *,
        workspace_policy: WorkspacePolicy | None = None,
        cache_volume_id: str | None = None,
        selection: RunpodSelection | None = None,
    ) -> TrainerCredentials:
        # Fall back to the canonical LTX-2 weights when a setting is blank so a
        # legacy/empty settings file doesn't silently skip the model download.
        checkpoint_file = settings.lora_model_checkpoint_file or DEFAULT_MODEL_CHECKPOINT_FILE
        model_hf_repo = settings.lora_model_hf_repo or DEFAULT_MODEL_HF_REPO
        text_encoder_hf_repo = (
            settings.lora_text_encoder_hf_repo or DEFAULT_TEXT_ENCODER_HF_REPO
        )
        if provider == "local":
            # Local (WSL2) training: workspace on the distro's native ext4 fs
            # (see LOCAL_WSL_WORKSPACE_DIR — must NOT be on /mnt/c, which can't
            # mmap the multi-GB checkpoint). Always app-managed (auto-provision)
            # and never uses any runpod_* fields. Reuse the same trainer repo +
            # HF source config the RunPod path reads, so both backends provision
            # the same trainer + weights.
            workspace = LOCAL_WSL_WORKSPACE_DIR
            return TrainerCredentials(
                provider="local",
                workspace_dir=workspace,
                model_path=paths.default_model_path(workspace, checkpoint_file),
                text_encoder_path=paths.default_text_encoder_path(workspace),
                gemini_api_key=settings.gemini_api_key,
                auto_provision=True,
                trainer_repo_url=settings.lora_trainer_repo_url,
                trainer_repo_ref=settings.lora_trainer_repo_ref,
                model_hf_repo=model_hf_repo,
                model_filename=checkpoint_file,
                text_encoder_hf_repo=text_encoder_hf_repo,
                hf_token=settings.hf_token or self._handler.current_hf_token() or "",
            )
        # With auto-provisioning the app owns the on-pod layout: the managed
        # network volume mounts at /workspace and provisioning downloads the
        # weights there. So we IGNORE any workspace/path overrides and always use
        # the managed locations. Overrides apply only to a pre-baked image
        # (auto-provision off) where the user knows where the weights live.
        app_managed = settings.lora_auto_provision
        if app_managed:
            workspace = RUNPOD_MANAGED_WORKSPACE_DIR
            model_path = paths.default_model_path(workspace, checkpoint_file)
            text_encoder_path = paths.default_text_encoder_path(workspace)
        else:
            workspace = settings.lora_remote_workspace_dir or "/workspace"
            model_path = settings.lora_remote_model_path or paths.default_model_path(
                workspace, checkpoint_file
            )
            text_encoder_path = (
                settings.lora_remote_text_encoder_path
                or paths.default_text_encoder_path(workspace)
            )
        return TrainerCredentials(
            provider=provider,
            workspace_dir=workspace,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            gemini_api_key=settings.gemini_api_key,
            runpod_api_key=settings.runpod_api_key,
            runpod_gpu_type=(
                selection.gpu_type if selection is not None else settings.runpod_gpu_type
            ),
            runpod_image=settings.runpod_image,
            # Only attach the network volume when caching is on. Otherwise a
            # persisted volume id would still region-lock the pod even though
            # the user opted into ephemeral/any-region pods.
            runpod_network_volume_id=(
                ""
                if (
                    selection is not None
                    and selection.workspace_policy == "ephemeral_any_region"
                )
                or workspace_policy == "ephemeral_any_region"
                else (
                    selection.volume_id or ""
                    if selection is not None
                    and selection.workspace_policy == "primary_cache"
                    else (
                        cache_volume_id
                        if workspace_policy == "primary_cache"
                        and cache_volume_id is not None
                        else (
                            settings.runpod_network_volume_id
                            if settings.runpod_keep_model_cached
                            else ""
                        )
                    )
                )
            ),
            runpod_datacenter=selection.datacenter if selection is not None else "",
            volume_size_gb=settings.runpod_volume_size_gb,
            auto_provision=settings.lora_auto_provision,
            trainer_repo_url=settings.lora_trainer_repo_url,
            trainer_repo_ref=settings.lora_trainer_repo_ref,
            model_hf_repo=model_hf_repo,
            model_filename=checkpoint_file,
            text_encoder_hf_repo=text_encoder_hf_repo,
            # Prefer the manually-pasted HF token (BYOK, works regardless of the
            # app's HF-OAuth gating); fall back to an OAuth token if present.
            hf_token=settings.hf_token or self._handler.current_hf_token() or "",
        )

    @staticmethod
    def _model_fingerprint(credentials: TrainerCredentials) -> str:
        material = "\n".join(
            (
                credentials.trainer_repo_url,
                credentials.trainer_repo_ref,
                credentials.model_hf_repo,
                credentials.model_filename,
                credentials.text_encoder_hf_repo,
                credentials.runpod_image,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _credentials_for_trainer_snapshot(
        self,
        settings: "AppSettings",
        provider: TrainerProvider,
        *,
        repo_url: str | None,
        repo_ref: str | None,
        workspace_policy: WorkspacePolicy | None = None,
        cache_volume_id: str | None = None,
        selection: RunpodSelection | None = None,
    ) -> TrainerCredentials:
        credentials = self._credentials(
            settings,
            provider,
            workspace_policy=workspace_policy,
            cache_volume_id=cache_volume_id,
            selection=selection,
        )
        if not repo_url or not repo_ref:
            return credentials
        return replace(
            credentials,
            trainer_repo_url=repo_url,
            trainer_repo_ref=repo_ref,
        )

    # ------------------------------------------------------------------
    # Idle auto-stop
    # ------------------------------------------------------------------

    def _reconcile_idle_stops(
        self,
        settings: "AppSettings",
        active_preprocessed: list[PreprocessedDataset],
        active_training: list[TrainingJob],
    ) -> None:
        """Tear down RunPod pods that have sat idle past the configured window.

        Cached workspaces preserve their network volume; ephemeral workspaces
        are terminated after the same grace period so they cannot bill forever.
        Datasets with work in flight are refreshed, never stopped.
        """
        idle_minutes = settings.runpod_idle_stop_minutes
        if idle_minutes <= 0:
            return
        # Map every in-flight item back to its dataset so we never tear down
        # a pod that's actively working, and reset its idle clock.
        busy: set[str] = {p.dataset_id for p in active_preprocessed}
        for job in active_training:
            pre = self._handler.get_preprocessed(job.preprocessed_id)
            if pre is not None:
                busy.add(pre.dataset_id)
        for dataset_id in busy:
            self._handler.touch_dataset_activity(dataset_id)
        busy_pod_ids = {
            target.pod_id
            for target in [
                *(item.target for item in active_preprocessed),
                *(job.target for job in active_training),
            ]
            if target is not None and target.pod_id is not None
        }
        active_datasets = self._handler.list_active_datasets()
        busy_workspace_keys = {
            (
                dataset.workspace_policy,
                dataset.cache_volume_id,
                dataset.runpod_selection.datacenter
                if dataset.runpod_selection is not None
                else "",
            )
            for dataset in active_datasets
            if dataset.id in busy
            or dataset.status in ("uploading", "gpu_selection_required")
        }
        cutoff = _utcnow() - timedelta(minutes=idle_minutes)
        released_pod_ids: set[str] = set()
        for dataset in self._handler.list_datasets_with_pod():
            if (
                dataset.id in busy
                or dataset.target is None
                or dataset.target.pod_id is None
                or dataset.target.pod_id in busy_pod_ids
                or dataset.target.pod_id in released_pod_ids
                or (
                    dataset.workspace_policy,
                    dataset.cache_volume_id,
                    dataset.runpod_selection.datacenter
                    if dataset.runpod_selection is not None
                    else "",
                )
                in busy_workspace_keys
            ):
                continue
            keep_alive_until = _parse_iso(dataset.keep_alive_until)
            if keep_alive_until is not None and keep_alive_until > _utcnow():
                continue
            last_active = _parse_iso(
                dataset.final_activity_at or dataset.last_active_at
            )
            if last_active is None or last_active > cutoff:
                continue
            creds = self._credentials(
                settings,
                "runpod",
                workspace_policy=dataset.workspace_policy,
                cache_volume_id=dataset.cache_volume_id,
                selection=dataset.runpod_selection,
            )
            try:
                self._handler.mark_workspace_release_attempt(
                    dataset.target.pod_id, error=None
                )
                self._target.release_workspace(
                    credentials=creds, handle=dataset.target
                )
            except TrainerTargetError as exc:
                self._handler.mark_workspace_release_attempt(
                    dataset.target.pod_id, error=exc.detail
                )
                logger.warning(
                    "LoRA idle-stop: failed to release pod for dataset %s: %s",
                    dataset.id,
                    exc.detail,
                )
                continue
            self._handler.mark_pod_stopped(dataset.target.pod_id)
            released_pod_ids.add(dataset.target.pod_id)
            logger.info(
                "LoRA idle auto-stop: released idle %s pod for dataset %s "
                "(idle > %d min)%s",
                dataset.workspace_policy,
                dataset.id,
                idle_minutes,
                (
                    "; network volume preserves its data"
                    if dataset.workspace_policy == "primary_cache"
                    else "; ephemeral workspace was discarded"
                ),
            )

    def _ensure_active_pod(
        self,
        dataset: LoraDataset,
        creds: TrainerCredentials,
        *,
        remote_job_id: str | None = None,
    ) -> TargetHandle:
        """Return a live pod handle for a dataset, re-creating one if it was
        idle-stopped.

        When a prior idle-stop cleared the pod id, this re-allocates a pod
        (re-mounting the network volume, so the marker-gated provisioning
        short-circuits and the cached weights + latents are already present)
        and persists the new handle.
        """
        handle = dataset.target
        if handle is None:
            raise TrainerTargetError(
                "Dataset has no remote workspace handle", retryable=False
            )
        previous_pod_id = handle.pod_id
        live_handle = self._target.ensure_workspace(
            credentials=creds, handle=handle
        )
        if (
            previous_pod_id is not None
            and live_handle.pod_id != previous_pod_id
            and remote_job_id is not None
        ):
            # The old pod vanished while work was in flight. Do not poll its
            # remote job id on a replacement pod or silently restart from a
            # different workspace. Reclaim the replacement immediately.
            self._handler.set_dataset_target(dataset.id, live_handle)
            self._target.release_workspace(
                credentials=creds, handle=live_handle
            )
            self._handler.mark_dataset_pod_stopped(dataset.id)
            raise TrainerTargetError(
                "The RunPod pod disappeared while remote work was running. "
                "Resume from the latest checkpoint or reset this phase.",
                retryable=False,
            )
        self._target.ensure_provisioned(
            credentials=creds, handle=live_handle
        )
        self._handler.set_dataset_target(dataset.id, live_handle)
        return live_handle

    # ------------------------------------------------------------------
    # Stage 1: upload
    # ------------------------------------------------------------------

    def _reusable_runpod_handle(
        self, dataset: LoraDataset, creds: TrainerCredentials
    ) -> tuple[TargetHandle | None, bool]:
        """Return a compatible idle app workspace, or whether one is busy.

        A new dataset previously always rented a second pod. The reconciler is
        serial, so dataset handles form a small lease ledger: active
        preprocess/training handles are busy; an otherwise-compatible handle can
        be transferred to the next upload and revalidated by
        ``ensure_provisioned``. A compatible busy pod makes the new upload wait,
        avoiding surprise duplicate billing.
        """
        if creds.provider != "runpod" or dataset.target is not None:
            return None, False
        active_preprocessed = self._handler.list_active_preprocessed()
        active_training = self._handler.list_active_training()
        busy_dataset_ids = {item.dataset_id for item in active_preprocessed}
        for job in active_training:
            preprocessed = self._handler.get_preprocessed(job.preprocessed_id)
            if preprocessed is not None:
                busy_dataset_ids.add(preprocessed.dataset_id)
        busy_pod_ids = {
            target.pod_id
            for target in [
                *(item.target for item in active_preprocessed),
                *(job.target for job in active_training),
            ]
            if target is not None and target.pod_id is not None
        }
        requested_gpu = (
            dataset.runpod_selection.gpu_type
            if dataset.runpod_selection is not None
            else creds.runpod_gpu_type
        )
        compatible_busy = False
        for candidate in self._handler.list_datasets_with_pod():
            handle = candidate.target
            if handle is None or handle.pod_id is None or handle.provider != "runpod":
                continue
            candidate_gpu = (
                candidate.runpod_selection.gpu_type
                if candidate.runpod_selection is not None
                else ""
            )
            if (
                candidate.workspace_policy != dataset.workspace_policy
                or candidate.cache_volume_id != dataset.cache_volume_id
                or (requested_gpu and requested_gpu != candidate_gpu)
            ):
                continue
            if candidate.id in busy_dataset_ids or handle.pod_id in busy_pod_ids:
                compatible_busy = True
                continue
            logger.info(
                "LoRA trainer: reusing idle RunPod workspace %s for dataset '%s'",
                handle.pod_id,
                dataset.name,
            )
            return handle.model_copy(deep=True), False
        return None, compatible_busy

    def _finalize_upload_cancel(
        self, dataset: LoraDataset, handle: TargetHandle, creds: TrainerCredentials
    ) -> bool:
        """Release the pod and finalize an upload cancel.

        Returns True if finalized (dataset is now `cancelled`); False if the
        release failed — in that case `cancel_requested` stays set so the next
        reconcile tick retries the release before flipping to `cancelled`
        (we don't want to mark cancelled while a pod may still be billing).
        Either way the caller must stop the upload: this never resumes work.
        """
        try:
            self._target.release_workspace(credentials=creds, handle=handle)
        except TrainerTargetError as exc:
            logger.warning(
                "LoRA upload %s cancel: release failed: %s", dataset.id, exc.detail
            )
            return False
        self._handler.mark_dataset_upload_cancelled(dataset.id)
        logger.info("LoRA trainer: upload '%s' cancelled", dataset.name)
        return True

    def _release_failed_upload_pod(
        self, dataset_id: str, creds: TrainerCredentials
    ) -> None:
        """Best-effort billing cleanup after a terminal upload failure."""
        dataset = self._handler.get_dataset(dataset_id)
        if (
            dataset is None
            or dataset.target is None
            or dataset.target.pod_id is None
        ):
            return
        try:
            self._target.release_workspace(
                credentials=creds, handle=dataset.target
            )
        except TrainerTargetError as exc:
            logger.error(
                "LoRA upload %s failed and its pod could not be released: %s",
                dataset_id,
                exc.detail,
            )
            return
        self._handler.mark_pod_stopped(dataset.target.pod_id)
        logger.info(
            "LoRA upload %s failed; released its workspace to stop billing",
            dataset_id,
        )

    def _reconcile_upload(self, dataset: LoraDataset, settings: "AppSettings") -> None:
        # Upload is the FIRST stage and provisions the workspace, so it can't read
        # the provider off `dataset.target` (which doesn't exist yet) — it reads
        # the provider the run was started with, persisted on the dataset.
        creds = self._credentials(
            settings,
            dataset.provider,
            workspace_policy=dataset.workspace_policy,
            cache_volume_id=dataset.cache_volume_id,
            selection=dataset.runpod_selection,
        )
        workspace = creds.workspace_dir
        # "pod" is RunPod-specific (a pod is its GPU container); local training
        # has no pod, so keep the user-facing status wording provider-appropriate.
        is_local = dataset.provider == "local"
        # Human-readable, collision-free remote leaf (e.g. datasets/xray-a3f8c2e1)
        # so the GPU workspace is browseable. Recorded in state and reused by
        # every later step (never recomputed from the id).
        remote_dir = paths.dataset_dir(workspace, dataset.id, dataset.name)
        logger.info(
            "LoRA trainer: uploading dataset '%s' (%s) to %s workspace",
            dataset.name,
            dataset.id,
            dataset.provider,
        )
        # Honor a cancel request BEFORE touching the pod: a cancel that landed
        # between ticks (or while we were blocked in a prior phase) should tear
        # down the already-provisioned pod and stop, not continue the upload.
        # `request_cancel_upload` already finalized target-less cancels, so here
        # a live target is the only thing to reclaim.
        if self._handler.is_dataset_cancel_requested(dataset.id):
            if dataset.target is not None and dataset.target.pod_id is not None:
                self._finalize_upload_cancel(dataset, dataset.target, creds)
            else:
                self._handler.mark_dataset_upload_cancelled(dataset.id)
                logger.info("LoRA trainer: upload '%s' cancelled", dataset.name)
            return
        try:
            self._handler.set_dataset_status_detail(
                dataset.id,
                "Preparing local GPU…" if is_local else "Acquiring GPU pod…",
            )
            seed_handle = dataset.target
            if not is_local and seed_handle is None:
                seed_handle, compatible_busy = self._reusable_runpod_handle(
                    dataset, creds
                )
                if seed_handle is None and compatible_busy:
                    self._handler.set_dataset_status_detail(
                        dataset.id,
                        "Waiting for the current RunPod workspace to become available…",
                    )
                    return
                if seed_handle is not None:
                    self._handler.set_dataset_status_detail(
                        dataset.id, "Reusing ready RunPod workspace…"
                    )
            handle = self._target.ensure_workspace(
                credentials=creds, handle=seed_handle
            )
            # Persist the handle immediately so a later failure (provisioning,
            # staging, upload) can't orphan a running, billing pod: the dataset
            # now owns it, so a retry reuses it and delete/idle-stop reclaim it.
            self._handler.set_dataset_target(dataset.id, handle)
            if dataset.auto_pipeline is not None and dataset.provider == "runpod":
                self._handler.set_pipeline_billing_start(
                    dataset.id,
                    started_at=_utcnow().isoformat(),
                    hourly_rate=self._pod_hourly_rate(creds, handle),
                )
            # A cancel that landed during pod acquisition: release the pod we
            # just provisioned and stop before the heavy install/transfer.
            if self._handler.is_dataset_cancel_requested(dataset.id):
                self._finalize_upload_cancel(dataset, handle, creds)
                return
            # Bootstrap the workspace (install trainer + optional model
            # download) before the first upload. Idempotent + marker-gated
            # inside the target, so a reused pod/volume short-circuits.
            # Blocking by design for this single-user control plane (same
            # stance as uploads); only ever pays the full cost once.
            self._handler.set_dataset_status_detail(
                dataset.id,
                "Setting up environment…" if is_local else "Setting up pod…",
            )
            # Surface live setup progress (clone %, uv sync, model download bar)
            # onto the card as a clean phase + %/ETA so the user sees it working.
            self._target.ensure_provisioned(
                credentials=creds,
                handle=handle,
                progress=lambda line, _id=dataset.id, _provider=creds.provider: self._report_setup_progress(
                    _id, line, _provider
                ),
            )
            if creds.runpod_network_volume_id:
                self._handler.mark_saved_model_ready(
                    volume_id=creds.runpod_network_volume_id,
                    fingerprint=self._model_fingerprint(creds),
                    estimated_download_bytes=ESTIMATED_MODEL_DOWNLOAD_BYTES,
                )
            # A cancel that landed during provisioning: the install already
            # completed (ensure_provisioned isn't cooperatively interruptible),
            # but we can still release the pod and stop before uploading.
            if self._handler.is_dataset_cancel_requested(dataset.id):
                self._finalize_upload_cancel(dataset, handle, creds)
                return
            self._handler.set_dataset_status_detail(dataset.id, "Preparing clips…")
            staging = self._build_staging(dataset, remote_dir)
            # A cancel that landed during staging: release before the transfer.
            if self._handler.is_dataset_cancel_requested(dataset.id):
                self._finalize_upload_cancel(dataset, handle, creds)
                return
            self._handler.set_dataset_status_detail(
                dataset.id,
                "Copying clips to local workspace…" if is_local else "Uploading clips…",
            )
            self._target.upload_directory(
                credentials=creds,
                handle=handle,
                local_dir=str(staging),
                remote_dir=remote_dir,
            )
            # A cancel that landed during the transfer: the (non-interruptible)
            # upload finished, but we honor the cancel by NOT auto-advancing into
            # preprocessing — release the pod and mark cancelled instead.
            if self._handler.is_dataset_cancel_requested(dataset.id):
                self._finalize_upload_cancel(dataset, handle, creds)
                return
        except TrainerTargetError as exc:
            logger.warning(
                "LoRA dataset %s upload failed: %s", dataset.id, exc.detail
            )
            if exc.code == "capacity_unavailable" and dataset.auto_pipeline is not None:
                self._handler.require_dataset_gpu_selection(dataset.id, exc.detail)
            else:
                self._handler.fail_dataset_upload(dataset.id, exc.detail)
                self._release_failed_upload_pod(dataset.id, creds)
            return
        except (BundleError, OSError) as exc:
            # Staging-side failure (missing clip file, copy error): fail the
            # upload with a clear message rather than retrying forever. Some
            # exceptions (e.g. bare OSError) stringify to "", so fall back to
            # the type name and emit the full traceback to the log for
            # diagnosis.
            detail = str(exc).strip() or repr(exc)
            logger.warning(
                "LoRA dataset %s staging failed (%s): %s",
                dataset.id,
                type(exc).__name__,
                detail,
            )
            log_background_exception(
                f"{_BACKGROUND_TASK_NAME}:staging[{dataset.id}]", exc
            )
            self._handler.fail_dataset_upload(
                dataset.id, f"{type(exc).__name__}: {detail}"
            )
            self._release_failed_upload_pod(dataset.id, creds)
            return
        self._handler.mark_dataset_uploaded(
            dataset.id, remote_dataset_dir=remote_dir, handle=handle
        )
        logger.info("LoRA trainer: dataset '%s' uploaded to %s", dataset.name, remote_dir)
        # One-click pipeline: auto-advance into preprocessing the instant upload
        # finishes, carrying the chosen training config forward to the run.
        spec = self._handler.consume_auto_pipeline(dataset.id)
        if spec is not None:
            self._handler.create_preprocessing(
                dataset_id=dataset.id,
                resolution_buckets=spec.resolution_buckets,
                with_audio=spec.with_audio,
                auto_caption=spec.auto_caption,
                captioner_type=spec.captioner_type,
                auto_training=spec.training,
                # The route already GPU-adjusted this preset via _apply_gpu_preset
                # before stashing it on the spec, so preprocess sees the same
                # final preset training will use (e.g. low_vram on a 32 GB GPU).
                preset=spec.training.config.preset,
            )
            logger.info(
                "LoRA trainer: one-click pipeline — auto-preprocessing '%s'",
                dataset.name,
            )

    def _report_setup_progress(
        self, dataset_id: str, line: str, provider: TrainerProvider
    ) -> None:
        """Parse a remote setup log line and push phase + %/ETA to the card."""
        phase, percent, eta = _parse_setup_progress(line)
        if phase in ("Downloading training model", "Downloading text encoder"):
            destination = (
                "RunPod workspace" if provider == "runpod" else "this computer"
            )
            phase = f"{phase} to {destination}"
        self._handler.set_dataset_status_detail(
            dataset_id, phase, percent=percent, eta_seconds=eta
        )

    def _report_preprocess_progress(
        self,
        item: PreprocessedDataset,
        dataset_id: str,
        creds: TrainerCredentials,
        handle: TargetHandle,
    ) -> None:
        """Tail the remote preprocess log and push phase + %/ETA to the card.

        Best-effort: a failed log read (transient SSH hiccup) just leaves the
        current detail in place rather than disrupting the running job.
        """
        if item.target is None or item.target.remote_job_id is None:
            return
        try:
            tail = self._target.read_logs(
                credentials=creds,
                handle=handle,
                remote_job_id=item.target.remote_job_id,
                tail=15,
            )
        except TrainerTargetError:
            return
        parsed = _parse_preprocess_progress(tail)
        if parsed is None:
            return
        phase, percent, eta = parsed
        # Captioning and latent-caching share this poll path. The state machine,
        # not incidental caption metadata in the log, owns the visible phase.
        if item.status == "captioning":
            phase = "Captioning clips"
        else:
            dataset = self._handler.get_dataset(dataset_id)
            phase = (
                "Encoding input/output pairs"
                if dataset is not None and dataset.type == "ic_lora"
                else "Caching training latents"
            )
        self._handler.set_dataset_status_detail(
            dataset_id, phase, percent=percent, eta_seconds=eta
        )

    def _build_staging(self, dataset: LoraDataset, remote_dataset_dir: str) -> Path:
        """Stage a dataset for upload.

        Kept/unreviewed, non-trashed clips ship; rejected or recycled ones are
        excluded. ``media_path``/``video``/``reference_video`` are absolute
        remote paths because the remote ``process_dataset.py`` runs with the
        trainer repo as its cwd.

        - **IC-LoRA** routes through `lora_dataset_prep`: each clip pair is
          re-encoded to a common fps/resolution/frame-count (rotation baked in,
          audio stripped) and validated, then shipped as one
          ``{caption, video, reference_video}`` record — no input-only rows.
          Unusable pairs are dropped and logged.
        - **Standard** keeps the legacy numbered ``media_path`` layout;
          ``dataset.json`` is omitted when it will be auto-captioned remotely.
        """
        staging = self._config.app_data_dir / "lora" / "staging" / dataset.id
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "clips").mkdir(parents=True, exist_ok=True)

        train_clips = [
            c
            for c in dataset.clips
            if c.triage not in ("reject", "holdout") and not c.deleted_at
        ]
        base = remote_dataset_dir.rstrip("/")
        prep_options = prep.PrepOptions(
            short_side=dataset.ic_staged_short_side,
            bucket_frames=dataset.ic_staged_bucket_frames,
            trigger_word=dataset.trigger_word,
        )

        if dataset.type == "ic_lora":
            report = prep.prepare_ic_lora_bundle(
                dataset=dataset,
                clips=train_clips,
                staging_dir=staging,
                processor=self._clip_processor,
                options=prep_options,
                render_media=lambda rel: f"{base}/{rel}",
            )
            logger.info(
                "LoRA dataset %s staged for upload: %s", dataset.id, report.summary()
            )
            if report.exported == 0:
                reason = report.dropped[0].reason if report.dropped else "no pairs"
                raise BundleError(
                    f"No training-ready pairs to upload ({reason}). "
                    "Check captions and clip pairing, then re-upload."
                )
        else:
            rel_by_local, order = plan_clip_layout(dataset, train_clips)
            for clip in order:
                self._stage_standard_clip(
                    Path(clip.local_path), staging / rel_by_local[clip.local_path]
                )
            rows = build_dataset_rows(
                dataset, train_clips, rel_by_local, render_media=lambda rel: f"{base}/{rel}"
            )
            if any(c.caption.strip() for c in train_clips):
                import json

                (staging / "dataset.json").write_text(
                    json.dumps(rows, indent=2), encoding="utf-8"
                )

        # Stage held-out clips' reference videos for the IC-LoRA validation
        # feed. No-op when no clips are marked `holdout` or when holdout clips
        # have no reference (text-to-video holdout is prompt-only). Staged
        # files land at `{remote_dataset_dir}/holdout/{clip.id}.mp4`, which
        # `_start_training` bakes into the validation sample's reference
        # condition. Best-effort: failures are logged, never fatal.
        holdout_report = prep.stage_holdout_references(
            dataset=dataset,
            staging_dir=staging,
            processor=self._clip_processor,
            options=prep_options,
            # IC-LoRA validation needs a reference video; auto-stage the first
            # training clip's reference when the user curated no holdout, so a
            # run still gets a validation feed. No-op for standard/t2v.
            auto_pick_when_empty=dataset.type == "ic_lora",
        )
        if holdout_report.dropped:
            logger.info(
                "LoRA dataset %s holdout staging dropped: %s",
                dataset.id,
                "; ".join(f"{d.name}: {d.reason}" for d in holdout_report.dropped),
            )
        if holdout_report.auto_picked is not None:
            logger.info(
                "LoRA dataset %s no holdout clips — auto-staged clip %s for validation",
                dataset.id,
                holdout_report.auto_picked,
            )
        return staging

    def _stage_standard_clip(self, src: Path, dst: Path) -> None:
        """Stage one standard-dataset clip, downscaling only if oversized.

        Clips with a short side at or under `STANDARD_MAX_SHORT_SIDE` are copied
        byte-for-byte (no re-encode, no quality loss). Oversized clips (4K/8K)
        are downscaled — aspect preserved, fps/frames/audio untouched, never
        upscaled — so the remote trainer doesn't have to decode huge frames
        (OOM risk) and the upload stays small. A failed probe ships the original
        as-is rather than blocking the upload.
        """
        try:
            probe = self._clip_processor.probe(video_path=str(src))
        except ClipProcessorError:
            shutil.copy2(src, dst)
            return
        short = min(probe.width, probe.height)
        if short <= 0 or short <= STANDARD_MAX_SHORT_SIDE:
            shutil.copy2(src, dst)
            return
        scale = STANDARD_MAX_SHORT_SIDE / short
        width = _even_dim(probe.width * scale)
        height = _even_dim(probe.height * scale)
        self._clip_processor.render(
            source_path=str(src),
            plan=EditPlan(scale=ScaleSpec(width=width, height=height)),
            out_path=str(dst),
        )
        logger.info(
            "LoRA trainer: downscaled oversized clip %s (%dx%d -> %dx%d) for upload",
            src.name, probe.width, probe.height, width, height,
        )

    # ------------------------------------------------------------------
    # Stage 2: preprocessing
    # ------------------------------------------------------------------

    def _reconcile_preprocess(
        self, item: PreprocessedDataset, settings: "AppSettings"
    ) -> None:
        dataset = self._handler.get_dataset(item.dataset_id)
        if dataset is None or dataset.target is None or dataset.remote_dataset_dir is None:
            self._handler.fail_preprocess(
                item.id, "Source dataset is missing or not uploaded"
            )
            return
        provider = dataset.target.provider
        creds = self._credentials_for_trainer_snapshot(
            settings,
            provider,
            repo_url=item.trainer_repo_url,
            repo_ref=item.trainer_repo_ref,
            workspace_policy=dataset.workspace_policy,
            cache_volume_id=dataset.cache_volume_id,
            selection=dataset.runpod_selection,
        )
        # Honor a cancel request BEFORE touching the pod/activity clock: if the
        # user cancelled, terminate the in-flight remote job (if any) and finalize
        # the cancel. We avoid re-provisioning an idle-stopped pod just to kill a
        # job that's already gone — if the pod is gone we cancel outright.
        if item.cancel_requested:
            remote_job_id = item.target.remote_job_id if item.target else None
            if (
                remote_job_id is not None
                and dataset.target.pod_id is not None
            ):
                try:
                    self._target.terminate(
                        credentials=creds,
                        handle=dataset.target,
                        remote_job_id=remote_job_id,
                    )
                except TrainerTargetError as exc:
                    if exc.retryable:
                        # Transient: leave `cancel_requested` set so the next
                        # tick retries the terminate, and keep status as-is so
                        # the card shows "Cancelling…" rather than a false
                        # "Cancelled".
                        logger.warning(
                            "LoRA preprocess %s cancel: terminate failed (will retry): %s",
                            item.id,
                            exc.detail,
                        )
                        return
                    # Non-retryable (e.g. "pod not found"): the pod and its
                    # remote job are already gone, so retrying can never succeed.
                    # Fall through to finalize the cancel instead of re-attempting
                    # every tick forever.
                    logger.warning(
                        "LoRA preprocess %s cancel: terminate unrecoverable (%s) — "
                        "pod appears gone, marking cancelled",
                        item.id,
                        exc.detail,
                    )
            self._handler.mark_preprocess_cancelled(item.id)
            logger.info("LoRA trainer: preprocessing '%s' cancelled", dataset.name)
            return
        workspace = creds.workspace_dir
        repo = paths.trainer_workdir(workspace)
        # This dataset has active work — keep its idle-stop clock fresh so a
        # long preprocess (latent caching can take many minutes) is never torn
        # down mid-run by idle auto-stop.
        self._handler.touch_dataset_activity(dataset.id)
        # Re-acquire the pod if a prior idle auto-stop released it (no-op when
        # it's still running). The network volume preserves the dataset clips.
        handle = self._ensure_active_pod(
            dataset,
            creds,
            remote_job_id=item.target.remote_job_id if item.target else None,
        )
        # Resolve dataset paths from the dir recorded at upload, not from the
        # id, so a post-upload rename can't point us at a non-existent folder.
        remote_dir = dataset.remote_dataset_dir

        if item.status == "pending":
            # A reset asks for a fresh start: wipe the remote `.precomputed`
            # latent cache before re-running so process_dataset rebuilds it
            # from scratch (resume, by contrast, keeps the cache and re-runs
            # over it). Best-effort transport errors retry on the next tick
            # (reset_requested stays set until the wipe succeeds).
            force_recaption = item.reset_requested
            if force_recaption:
                self._target.delete_remote_paths(
                    credentials=creds,
                    handle=handle,
                    paths=[paths.precomputed_run_dir_in(remote_dir, item.id)],
                )
                self._handler.clear_preprocess_reset_requested(item.id)
                logger.info(
                    "LoRA trainer: reset wiped remote .precomputed for dataset '%s'",
                    dataset.name,
                )
            self._start_preprocess_first_step(
                item,
                dataset,
                creds,
                handle,
                repo,
                remote_dir,
                force_recaption=force_recaption,
            )
            return

        # captioning / preprocessing: poll the in-flight command.
        if item.target is None or item.target.remote_job_id is None:
            self._handler.fail_preprocess(item.id, "Lost remote job handle")
            return
        status = self._target.poll_command(
            credentials=creds, handle=handle, remote_job_id=item.target.remote_job_id
        )
        if status.state == "running":
            # Push live progress to the card so a multi-minute latent-caching
            # run (model load + per-clip VAE encode) visibly ticks instead of
            # sitting on a static "Caching latents…".
            self._report_preprocess_progress(item, dataset.id, creds, handle)
            return
        if status.state == "failed":
            # Surface the remote log tail so "Prep failed" says WHY (e.g. clips
            # too short for the frame bucket, OOM, model-load error).
            tail = self._target.read_logs(
                credentials=creds,
                handle=handle,
                remote_job_id=item.target.remote_job_id,
                tail=40,
            )
            detail = self._remote_failure_detail(
                tail, status.exit_code, status.error, kind="Preprocessing"
            )
            # A no-exit-code failure on a local run means the WSL job's process
            # group was hard-killed (kernel OOM killer, `systemd-oomd`, a WSL
            # distro shutdown, or a native crash) before the wrapper could record
            # `$?`. Pull the failed unit's `systemctl show` + journal + meminfo +
            # dmesg so the failure self-diagnoses instead of looking like a
            # trainer bug. Best-effort; never blocks the fail.
            if status.exit_code is None and dataset.provider == "local":
                diag = self._wsl_oom_diagnosis(item.target.remote_job_id)
                if diag:
                    logger.warning(
                        "LoRA preprocess %s suspected OOM — WSL diagnostics:\n%s",
                        item.id,
                        diag,
                    )
                    detail = f"{detail}\n— WSL diagnostics —\n{diag}"
            logger.warning("LoRA preprocess %s failed: %s", item.id, detail)
            self._handler.fail_preprocess(item.id, detail)
            return
        # succeeded
        if item.status == "captioning":
            logger.info("LoRA trainer: captioning complete for dataset '%s'", dataset.name)
            self._start_process_dataset(item, dataset, creds, handle, repo, remote_dir)
        else:  # preprocessing succeeded
            logger.info(
                "LoRA trainer: preprocessing complete — latents cached for dataset '%s'",
                dataset.name,
            )
            precomputed = paths.precomputed_run_dir_in(remote_dir, item.id)
            # Fail fast when a source silently produced 0 files: process_dataset.py
            # can exit 0 yet write nothing for a source — notably audio_latents when
            # the audio model failed to load — which otherwise only surfaces later
            # at training start as a cryptic "No valid samples found" (the trainer
            # counts only samples present in ALL configured sources, so one empty
            # source zeros every per-source count). Catch it here with a clear msg.
            latents_n = self._target.count_precomputed_source(
                credentials=creds,
                handle=handle,
                precomputed_dir=precomputed,
                source="latents",
            )
            if latents_n == 0:
                detail = (
                    "Preprocessing produced no video latents — process_dataset.py "
                    "likely errored silently. Check the preprocess logs, then re-run."
                )
                logger.warning("LoRA preprocess %s failed: %s", item.id, detail)
                self._handler.fail_preprocess(item.id, detail)
                return
            conditions_n = self._target.count_precomputed_source(
                credentials=creds,
                handle=handle,
                precomputed_dir=precomputed,
                source="conditions",
            )
            if conditions_n != latents_n:
                detail = (
                    "Preprocessing produced an incomplete text-conditioning cache "
                    f"({conditions_n} conditions for {latents_n} video latents). "
                    "Reset preprocessing and try again."
                )
                logger.warning("LoRA preprocess %s failed: %s", item.id, detail)
                self._handler.fail_preprocess(item.id, detail)
                return
            if dataset.type == "ic_lora":
                references_n = self._target.count_precomputed_source(
                    credentials=creds,
                    handle=handle,
                    precomputed_dir=precomputed,
                    source="reference_latents",
                )
                if references_n != latents_n:
                    detail = (
                        "IC-LoRA preprocessing produced an incomplete reference cache "
                        f"({references_n} references for {latents_n} target latents). "
                        "Check clip pairing, then reset preprocessing."
                    )
                    logger.warning("LoRA preprocess %s failed: %s", item.id, detail)
                    self._handler.fail_preprocess(item.id, detail)
                    return
            if item.with_audio:
                audio_n = self._target.count_precomputed_source(
                    credentials=creds,
                    handle=handle,
                    precomputed_dir=precomputed,
                    source="audio_latents",
                )
                if audio_n != latents_n:
                    # Surface the trainer's own audio-skip report so the message
                    # says *why* (typically: torchaudio's ffmpeg/torchcodec
                    # backend isn't functional on the remote, so
                    # `torchaudio.load(<mp4>)` can't demux video containers and
                    # every clip is silently counted as "no audio track").
                    # Best-effort: a read failure just yields the generic msg.
                    # `item.target.remote_job_id` is guaranteed non-None here —
                    # the top of this reconcile already failed+returned when it
                    # was missing.
                    skip_detail = ""
                    try:
                        log_tail = self._target.read_logs(
                            credentials=creds,
                            handle=handle,
                            remote_job_id=item.target.remote_job_id,
                            tail=300,
                        )
                        summary = _audio_skip_summary(log_tail)
                        if summary:
                            skip_detail = f"\n— trainer audio report —\n{summary}"
                    except Exception:  # noqa: BLE001
                        pass
                    detail = (
                        "Train with audio was on, but preprocessing produced an "
                        f"incomplete audio cache ({audio_n} audio latents for "
                        f"{latents_n} video latents). The usual cause is that `torchaudio.load` "
                        "can't demux audio out of the mp4/mov clips and silently "
                        "skips every clip. On RunPod this is fixed by the system "
                        "ffmpeg provisioning installs; on WSL (cu128) torchaudio "
                        "2.9+ routes through torchcodec whose libav ABI doesn't "
                        "match system ffmpeg, so provisioning also patches the "
                        "trainer to fall back to an `ffmpeg` subprocess. Re-provision "
                        "the workspace (or start a fresh pod/volume) so the patch is "
                        "applied, then re-run — or turn off 'Train with audio'."
                        + skip_detail
                    )
                    logger.warning("LoRA preprocess %s failed: %s", item.id, detail)
                    self._handler.fail_preprocess(item.id, detail)
                    return
            self._handler.mark_preprocess_ready(
                item.id,
                remote_precomputed_dir=precomputed,
            )
            # One-click pipeline: auto-start the training run that was queued
            # when the user kicked off the pipeline (no second click needed).
            pending = self._handler.consume_preprocess_auto_training(item.id)
            if pending is not None:
                self._handler.start_training(
                    preprocessed_id=item.id,
                    name=pending.name,
                    config=pending.config,
                    description=pending.description,
                    # Inherit the provider the one-click pipeline chose so the
                    # auto-started run trains on the same backend as the upload
                    # and preprocess stages (not always RunPod).
                    provider=pending.provider,
                    gpu_type=pending.gpu_type,
                    gpu_vram_gb=pending.gpu_vram_gb,
                    runpod_selection=pending.runpod_selection,
                    workload_billing_started_at=pending.workload_billing_started_at,
                    captured_hourly_rate=pending.captured_hourly_rate,
                )
                logger.info(
                    "LoRA trainer: one-click pipeline — auto-starting training '%s'",
                    pending.name,
                )

    def _start_preprocess_first_step(
        self,
        item: PreprocessedDataset,
        dataset: LoraDataset,
        creds: TrainerCredentials,
        handle: TargetHandle,
        repo: str,
        remote_dir: str,
        *,
        force_recaption: bool = False,
    ) -> None:
        if dataset.type == "ic_lora":
            self._handler.set_dataset_status_detail(
                dataset.id, "Encoding input/output pairs…"
            )
            requested = prep.options_for_resolution_buckets(
                item.resolution_buckets, trigger_word=dataset.trigger_word
            )
            if (
                requested.short_side != dataset.ic_staged_short_side
                or requested.bucket_frames != dataset.ic_staged_bucket_frames
            ):
                logger.info(
                    "LoRA trainer: restaging IC-LoRA pairs for %spx/%s frames",
                    requested.short_side,
                    requested.bucket_frames,
                )
                restaged = dataset.model_copy(
                    deep=True,
                    update={
                        "ic_staged_short_side": requested.short_side,
                        "ic_staged_bucket_frames": requested.bucket_frames,
                    },
                )
                staging = self._build_staging(restaged, remote_dir)
                self._target.upload_directory(
                    credentials=creds,
                    handle=handle,
                    local_dir=str(staging),
                    remote_dir=remote_dir,
                )
                self._handler.set_ic_staging_envelope(
                    dataset.id,
                    short_side=requested.short_side,
                    bucket_frames=requested.bucket_frames,
                )
        # Local training shares the single GPU with this inference server;
        # free the resident 22B so the trainer's first CUDA op doesn't starve
        # and crash mid-run. No-op for remote providers.
        self._prepare_local_gpu(dataset.provider)
        # IC-LoRA never auto-captions remotely: `caption_videos.py` scans the
        # clips dir and rewrites dataset.json with only media_path+caption,
        # which would clobber the `reference_path` pairing we staged. IC-LoRA
        # ships its reference-aware dataset.json from the upload and goes
        # straight to preprocessing. A resumed standard run whose captioning
        # already completed (`captioning_completed`) also skips straight to
        # preprocessing so it doesn't re-caption (and risk clobbering edits).
        if item.auto_caption and dataset.type != "ic_lora" and not item.captioning_completed:
            # qwen_omni runs Qwen3-Omni-30B on the GPU via a vLLM server; FP8
            # weights (~31 GiB) need a >=40 GiB card. Below that it OOMs with a
            # confusing traceback, so gate it upfront with a clear message and
            # point the user at Gemini Flash. Best-effort: a transport glitch
            # in the VRAM probe never blocks the run.
            if item.captioner_type == "qwen_omni":
                vram_gb = self._probe_vram_gb(creds, handle)
                if vram_gb is not None and vram_gb < MIN_QWEN_CAPTIONER_VRAM_GB:
                    self._handler.fail_preprocess(
                        item.id,
                        f"Qwen captioning needs a GPU with at least "
                        f"{MIN_QWEN_CAPTIONER_VRAM_GB} GB VRAM (this GPU has "
                        f"{vram_gb:.0f} GB). Switch the captioner to Gemini Flash.",
                    )
                    return
            logger.info(
                "LoRA trainer: starting auto-captioning (%s) for dataset '%s'",
                item.captioner_type,
                dataset.name,
            )
            # Gemini key only matters for gemini_flash, and it's a prefix-assign
            # on the caption command — don't attach it to the qwen_omni script
            # (it'd bind to `set -e` and be discarded; qwen doesn't use it).
            gemini_prefix = (
                paths.gemini_key_env_prefix(creds.gemini_api_key)
                if item.captioner_type == "gemini_flash"
                else ""
            )
            command = (
                paths.cache_env_prefix(creds.workspace_dir)
                + gemini_prefix
                + paths.caption_command(
                    clips_dir=paths.dataset_clips_dir_in(remote_dir),
                    dataset_json=paths.dataset_json_path_in(remote_dir),
                    captioner_type=item.captioner_type,
                    override=force_recaption,
                )
            )
            job_id = self._target.start_command(
                credentials=creds, handle=handle, command=command, workdir=repo
            )
            self._handler.set_preprocess_captioning(
                item.id, handle=handle, remote_job_id=job_id
            )
        else:
            self._start_process_dataset(item, dataset, creds, handle, repo, remote_dir)

    def _probe_vram_gb(self, creds: "TrainerCredentials", handle: "TargetHandle") -> float | None:
        """Live GPU VRAM in GB, or None if the probe failed (never raises).

        Used to gate the Qwen3-Omni-30B captioner, which needs >=40 GiB. A
        transport/diagnostics failure degrades to "unknown" so a glitch never
        blocks a run — the captioner's own OOM surfacing still catches a
        genuinely-too-small card.
        """
        try:
            telemetry = self._target.query_gpu(credentials=creds, handle=handle)
        except TrainerTargetError as exc:
            logger.warning("LoRA trainer: VRAM probe failed (continuing): %s", exc.detail)
            return None
        return telemetry.vram_total_mb / 1024.0

    def _wsl_oom_diagnosis(self, job_id: str | None = None) -> str:
        """Post-mortem WSL snapshot for a local run that died with no exit code.
        If `job_id` is given, includes the failed systemd unit's `systemctl show`
        + journal (the definitive exit reason — oom-kill / signal / crash). Empty
        off-Windows / on probe failure.
        """
        try:
            from services.wsl_memory.wsl_memory import wsl_postmortem

            unit = f"ltxjob_{job_id}" if job_id else None
            return wsl_postmortem(unit=unit)
        except Exception as exc:
            logger.warning("LoRA trainer: WSL post-mortem failed: %s", exc)
            return ""

    def _start_process_dataset(
        self,
        item: PreprocessedDataset,
        dataset: LoraDataset,
        creds: TrainerCredentials,
        handle: TargetHandle,
        repo: str,
        remote_dir: str,
    ) -> None:
        # IC-LoRA concatenates clean reference + noised target tokens into one
        # sequence, so its backward recompute is ~4x the self-attention memory of
        # text-to-video. On a 32 GB card already carrying ~22 GB of int8 weights,
        # full-size references OOM the first backward (CUDA device-not-ready via
        # TDR). The `low_vram` preset halves reference spatial resolution via the
        # official `--reference-downscale-factor` lever; the `flexible`
        # strategy's `reference` condition then infers the factor from ref/target
        # dims at train time, so no training-YAML change is needed. Text-to-video
        # has no references, so
        # the flag is irrelevant there (factor stays 1).
        ref_downscale = (
            preset_reference_downscale_factor(item.preset)
            if dataset.type == "ic_lora"
            else 1
        )
        # process_dataset.py rejects multiple resolution buckets when reference
        # downscaling is on, so collapse to a single bucket for that mode. Take
        # the first configured bucket and surface the override so the user knows
        # their multi-bucket IC-LoRA low_vram run trains at one resolution.
        effective_buckets = item.resolution_buckets
        if ref_downscale > 1 and ";" in item.resolution_buckets:
            effective_buckets = item.resolution_buckets.split(";")[0].strip()
            self._handler.set_preprocess_effective_buckets(item.id, effective_buckets)
            logger.warning(
                "LoRA trainer: IC-LoRA low_vram downscales references by %dx, "
                "which requires a single resolution bucket — using '%s' from '%s' "
                "for dataset '%s'",
                ref_downscale,
                effective_buckets,
                item.resolution_buckets,
                dataset.name,
            )
        logger.info(
            "LoRA trainer: starting preprocessing (caching latents @ %s) for dataset '%s'",
            effective_buckets,
            dataset.name,
        )
        precomputed = paths.precomputed_run_dir_in(remote_dir, item.id)
        command = paths.cache_env_prefix(creds.workspace_dir) + paths.process_dataset_command(
            dataset_json=paths.dataset_json_path_in(remote_dir),
            resolution_buckets=effective_buckets,
            model_path=creds.model_path,
            text_encoder_path=creds.text_encoder_path,
            with_audio=item.with_audio,
            trigger_word=dataset.trigger_word,
            # Match the training stage's text-encoder precision for this preset
            # (low_vram -> 8-bit): Gemma3 12B is 23 GB in bf16 and OOMs a 32 GB
            # GPU under WSL2, so the low_vram preset loads it in 8-bit here too.
            load_text_encoder_in_8bit=preset_loads_text_encoder_in_8bit(item.preset),
            # IC-LoRA low_vram only: halve reference resolution (see above).
            reference_downscale_factor=ref_downscale,
            # Each preprocess record gets an immutable cache root. Reusing the
            # trainer's default `.precomputed` directory across dataset edits
            # can silently mix stale and current latents.
            output_dir=precomputed,
        )
        job_id = self._target.start_command(
            credentials=creds, handle=handle, command=command, workdir=repo
        )
        self._handler.set_preprocess_processing(
            item.id, handle=handle, remote_job_id=job_id
        )

    # ------------------------------------------------------------------
    # Stage 3: training
    # ------------------------------------------------------------------

    def _capture_training_compute_rate(
        self,
        job: TrainingJob,
        creds: TrainerCredentials,
        handle: TargetHandle,
    ) -> None:
        if (
            job.provider != "runpod"
            or job.compute_rate_per_hr is not None
            or handle.pod_id is None
        ):
            return
        try:
            pod = next(
                (
                    item
                    for item in self._target.list_pods(credentials=creds)
                    if item.id == handle.pod_id
                ),
                None,
            )
        except TrainerTargetError:
            return
        if pod is not None and pod.cost_per_hr is not None:
            self._handler.set_training_compute_rate(job.id, pod.cost_per_hr)

    def _pod_hourly_rate(
        self, creds: TrainerCredentials, handle: TargetHandle
    ) -> float | None:
        if handle.pod_id is None:
            return None
        try:
            pod = next(
                (
                    item
                    for item in self._target.list_pods(credentials=creds)
                    if item.id == handle.pod_id
                ),
                None,
            )
        except TrainerTargetError:
            return None
        return pod.cost_per_hr if pod is not None else None

    def _reconcile_training(self, job: TrainingJob, settings: "AppSettings") -> None:
        preprocessed = self._handler.get_preprocessed(job.preprocessed_id)
        if preprocessed is None or preprocessed.remote_precomputed_dir is None:
            self._handler.fail_training(job.id, "Preprocessed dataset unavailable")
            return
        dataset = self._handler.get_dataset(preprocessed.dataset_id)
        if dataset is None or dataset.target is None:
            self._handler.fail_training(job.id, "Source dataset workspace unavailable")
            return
        requested = job.runpod_selection
        current = dataset.runpod_selection
        if requested is not None and current is not None and requested != current:
            same_persistent_workspace = (
                requested.workspace_policy == "primary_cache"
                and current.workspace_policy == "primary_cache"
                and requested.volume_id is not None
                and requested.volume_id == current.volume_id
                and requested.datacenter == current.datacenter
            )
            if not same_persistent_workspace:
                self._handler.require_training_gpu_selection(
                    job.id,
                    "Choose a GPU in the preprocessed dataset's original cache "
                    "region, or start a new full pipeline.",
                )
                return
            old_creds = self._credentials(
                settings,
                dataset.provider,
                workspace_policy=dataset.workspace_policy,
                cache_volume_id=dataset.cache_volume_id,
                selection=current,
            )
            if dataset.target.pod_id is not None:
                released_pod_id = dataset.target.pod_id
                self._target.release_workspace(
                    credentials=old_creds, handle=dataset.target
                )
                self._handler.mark_pod_stopped(released_pod_id)
            self._handler.set_dataset_runpod_selection(dataset.id, requested)
            refreshed = self._handler.get_dataset(dataset.id)
            if refreshed is None or refreshed.target is None:
                self._handler.fail_training(job.id, "Source dataset workspace unavailable")
                return
            dataset = refreshed
        dataset_target = dataset.target
        if dataset_target is None:
            self._handler.fail_training(job.id, "Source dataset workspace unavailable")
            return
        creds = self._credentials_for_trainer_snapshot(
            settings,
            dataset_target.provider,
            repo_url=job.trainer_repo_url,
            repo_ref=job.trainer_repo_ref,
            workspace_policy=dataset.workspace_policy,
            cache_volume_id=dataset.cache_volume_id,
        )
        if requested is not None:
            creds = replace(
                creds,
                runpod_gpu_type=requested.gpu_type,
                runpod_network_volume_id=requested.volume_id or "",
                runpod_datacenter=requested.datacenter,
            )
        workspace = creds.workspace_dir
        if job.cancel_requested:
            remote_job_id = job.target.remote_job_id if job.target else None
            if remote_job_id is not None and dataset_target.pod_id is not None:
                try:
                    self._target.terminate(
                        credentials=creds,
                        handle=dataset_target,
                        remote_job_id=remote_job_id,
                    )
                except TrainerTargetError as exc:
                    if exc.retryable:
                        logger.warning(
                            "LoRA training %s cancel: terminate failed "
                            "(will retry): %s",
                            job.id,
                            exc.detail,
                        )
                        return
                    logger.warning(
                        "LoRA training %s cancel: terminate unrecoverable "
                        "(%s) — marking cancelled",
                        job.id,
                        exc.detail,
                    )
            self._handler.mark_training_cancelled(job.id)
            return
        # Keep the idle-stop clock fresh so a long training run is never torn
        # down mid-run by idle auto-stop.
        self._handler.touch_dataset_activity(dataset.id)
        # Re-acquire the pod if idle auto-stop released it (no-op while running);
        # the network volume preserves the preprocessed latents to train from.
        try:
            handle = self._ensure_active_pod(
                dataset,
                creds,
                remote_job_id=job.target.remote_job_id if job.target else None,
            )
        except TrainerTargetError as exc:
            if exc.code == "capacity_unavailable":
                self._handler.require_training_gpu_selection(job.id, exc.detail)
                return
            raise
        self._capture_training_compute_rate(job, creds, handle)
        if job.provider == "runpod" and job.workload_billing_started_at is None:
            self._handler.begin_training_billing(
                job.id,
                started_at=_utcnow().isoformat(),
                hourly_rate=self._pod_hourly_rate(creds, handle),
            )

        # Redownload retry: the run already finished training (weights are on the
        # network volume) but the download failed. Skip the poll — the original
        # remote job is gone — and go straight to fetching the existing adapter.
        if job.redownload_requested:
            self._download_and_complete(job, creds, handle, workspace)
            return

        if job.status == "pending":
            self._handler.mark_training_setup_started(job.id)
            self._handler.set_training_status_detail(job.id, "Preparing training…")
            # A reset asks for a fresh start: wipe the remote output dir
            # (checkpoints + samples) and the local run folder before
            # re-running train.py from step 0. Resume, by contrast, keeps the
            # existing checkpoints (config.load_checkpoint is set by the
            # handler) and continues from the last one.
            if job.reset_requested:
                remote_output = job.remote_output_dir or paths.output_dir(
                    workspace, job.id, job.name
                )
                self._target.delete_remote_paths(
                    credentials=creds,
                    handle=handle,
                    paths=[remote_output],
                )
                local_run = self._local_run_dir(job)
                if local_run.exists():
                    shutil.rmtree(local_run, ignore_errors=True)
                self._handler.clear_training_reset_requested(job.id)
                logger.info(
                    "LoRA trainer: reset wiped remote output + local run dir "
                    "for '%s'",
                    job.name,
                )
            self._start_training(
                job,
                preprocessed.remote_precomputed_dir,
                creds,
                handle,
                workspace,
                dataset.type,
                dataset,
                preprocessed,
            )
            return

        # running: poll + best-effort progress.
        if job.target is None or job.target.remote_job_id is None:
            self._handler.fail_training(job.id, "Lost remote job handle")
            return
        self._update_progress(job, creds, handle)
        self._poll_gpu_status(job, creds, handle)
        status = self._target.poll_command(
            credentials=creds, handle=handle, remote_job_id=job.target.remote_job_id
        )
        if status.state == "running":
            return
        if status.state == "failed":
            # Surface the remote log tail so a training failure says WHY (OOM,
            # NaN loss, config/schema error) instead of a bare exit code.
            tail = self._target.read_logs(
                credentials=creds,
                handle=handle,
                remote_job_id=job.target.remote_job_id,
                tail=40,
            )
            detail = self._remote_failure_detail(
                tail, status.exit_code, status.error, kind="Training"
            )
            logger.warning("LoRA training %s failed: %s", job.id, detail)
            self._handler.fail_training(job.id, detail)
            return
        # succeeded -> download the trained adapter and mark complete.
        self._download_and_complete(job, creds, handle, workspace)

    def _download_and_complete(
        self,
        job: TrainingJob,
        creds: TrainerCredentials,
        handle: TargetHandle,
        workspace: str,
    ) -> None:
        """Resolve the trained adapter on the remote, download it, mark complete.

        Shared by the normal post-training path and the redownload-retry path
        (recovering a finished run whose download failed — the weights still
        live on the network volume). On any download error the job is failed
        with a clear message so it can be retried again.
        """
        # Output dir recorded at start (slug-named, e.g. outputs/my-run-7c44d9a2).
        remote_output = job.remote_output_dir or paths.output_dir(
            workspace, job.id, job.name
        )
        # The trainer writes adapters to checkpoints/lora_weights_step_NNNNN.
        # Resolve the step to download in priority order:
        #   1. `list_checkpoints` — the actual remote `checkpoints/` dir. This is
        #      the only source that works on a *redownload to a fresh pod*, where
        #      the original training log is gone with the exited container. It
        #      also authoritative-picks the highest existing adapter.
        #   2. `job.latest_checkpoint_step` — persisted during polling, survives a
        #      reconciler restart. Used when the listing is unreachable but we
        #      still know how far training got.
        #   3. `read_logs` + `_latest_checkpoint_step` — the original path, only
        #      useful when the pod that ran training is still alive.
        #   4. `job.total_steps or job.config.steps` — the configured final step,
        #      the filename a completed run always produces. Last resort.
        step: int | None = None
        try:
            remote_steps = self._target.list_checkpoints(
                credentials=creds, handle=handle, remote_output_dir=remote_output
            )
            if remote_steps:
                step = remote_steps[-1]
        except TrainerTargetError:
            step = None
        if step is None and job.latest_checkpoint_step is not None:
            step = job.latest_checkpoint_step
        if step is None and job.target is not None and job.target.remote_job_id is not None:
            try:
                logs = self._target.read_logs(
                    credentials=creds,
                    handle=handle,
                    remote_job_id=job.target.remote_job_id,
                    tail=_LOG_TAIL,
                )
                step = _latest_checkpoint_step(logs)
            except TrainerTargetError:
                step = None
        step = step or job.total_steps or job.config.steps
        remote_weights = paths.lora_checkpoint_path_in(remote_output, step)
        local_path = self._local_lora_path(job)
        # Each run gets its own folder (descriptive, collision-free) holding the
        # weights + run-summary.md + training-config.json.
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "LoRA trainer: training run '%s' finished — downloading LoRA weights "
            "(step %s)",
            job.name,
            step,
        )
        self._handler.set_training_status_detail(
            job.id,
            (
                "Downloading trained LoRA to this computer…"
                if job.provider == "runpod"
                else "Copying trained LoRA to the app…"
            ),
        )
        try:
            self._target.download_file(
                credentials=creds,
                handle=handle,
                remote_path=remote_weights,
                local_path=str(local_path),
            )
        except TrainerTargetError as exc:
            self._handler.fail_training(job.id, f"Download failed: {exc.detail}")
            return
        self._handler.mark_training_completed(job.id, local_lora_path=str(local_path))
        self._write_run_summary(job, local_path)
        # Also fetch the exact YAML the trainer consumed (best-effort) — the most
        # authoritative, reproducible record of the run, with resolved paths.
        try:
            self._target.download_file(
                credentials=creds,
                handle=handle,
                remote_path=paths.config_path(workspace, job.id, job.name),
                local_path=str(local_path.parent / "training-config.yaml"),
            )
        except TrainerTargetError:
            logger.info(
                "LoRA trainer: remote training config unavailable for %s "
                "(training-config.json still written)",
                job.id,
            )
        logger.info(
            "LoRA trainer: training run '%s' complete — LoRA saved to %s",
            job.name,
            local_path,
        )

    def _write_run_summary(self, job: TrainingJob, local_path: Path) -> None:
        """Write `run-summary.md` + `training-config.json` into the run folder
        (best-effort).

        Reads the just-completed job (with its final timestamps) plus the source
        preprocessed/dataset records for resolution + clip counts. A failure here
        must never fail the run — the LoRA is already downloaded.
        """
        try:
            completed = self._handler.get_training(job.id) or job
            preprocessed = self._handler.get_preprocessed(job.preprocessed_id)
            dataset = (
                self._handler.get_dataset(preprocessed.dataset_id)
                if preprocessed is not None
                else None
            )
            run_dir = local_path.parent
            markdown = build_run_summary_markdown(
                job=completed,
                preprocessed=preprocessed,
                dataset=dataset,
                local_lora_path=str(local_path),
            )
            (run_dir / "run-summary.md").write_text(markdown, encoding="utf-8")
            # Machine-readable config of exactly what was trained (reproducible).
            (run_dir / "training-config.json").write_text(
                completed.config.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info("LoRA trainer: wrote run summary + config to %s", run_dir)
        except Exception:  # noqa: BLE001 - summary is best-effort, never fatal
            logger.warning(
                "LoRA trainer: failed to write run summary for %s", job.id,
                exc_info=True,
            )

    def _start_training(
        self,
        job: TrainingJob,
        precomputed_dir: str,
        creds: TrainerCredentials,
        handle: TargetHandle,
        workspace: str,
        dataset_type: LoraDatasetType,
        dataset: LoraDataset,
        preprocessed: PreprocessedDataset,
    ) -> None:
        # Free the resident inference model so local training gets the full GPU
        # (see _prepare_local_gpu). No-op for remote providers.
        self._prepare_local_gpu(dataset.provider)
        remote_output = paths.output_dir(workspace, job.id, job.name)
        remote_config = paths.config_path(workspace, job.id, job.name)
        # Build validation samples for the in-app training-results feed. The
        # feed cadence is the existing `validation_interval` (e.g. 50 -> a feed
        # entry at steps 50, 100, 150, ...). Two sources feed it:
        #   - prompt samples: `job.config.validation_prompts` (text-to-video
        #     only; IC-LoRA validation needs a reference video, so bare prompts
        #     are dropped by the builder);
        #   - holdout clips: clips the user marked `triage="holdout"`, excluded
        #     from training and staged at upload to
        #     `{remote_dataset_dir}/holdout/{clip.id}.mp4`. For IC-LoRA each
        #     becomes a reference-conditioned sample (prompt = caption,
        #     reference = the staged input video); for t2v a prompt-only sample
        #     (prompt = caption). This is what makes the feed cover IC-LoRA.
        # Preprocessing owns the cached reference-latent shape. Training is
        # validated to the same preset, but use the immutable preprocess
        # snapshot here as the final source of truth.
        ref_downscale = preset_reference_downscale_factor(preprocessed.preset)
        holdout_specs = self._holdout_validation_specs(dataset, dataset_type)
        # Auto-seed validation prompts from the dataset's captions when the
        # user kept the generic placeholder (or left it blank), so a run
        # validates against a real caption instead of "A high quality sample
        # from the trained concept." IC-LoRA ignores bare prompts (its
        # validation comes from holdout / auto-picked clips), but the seeded
        # prompts still land in the recorded config and YAML for the t2v case.
        effective_prompts = self._effective_validation_prompts(
            job.config.validation_prompts, dataset
        )
        # Snap validation video_dims to the training bucket's aspect when the
        # user kept the 576x576x49 default (sentinel for "auto"), so validation
        # stops composing at a 1:1 aspect the LoRA never trained on. A user who
        # set custom validation dims is left alone.
        resolved_dims = self._resolve_validation_dims(job, preprocessed)
        config_for_yaml = job.config
        if resolved_dims is not None:
            vw, vh, vf = resolved_dims
            config_for_yaml = job.config.model_copy(
                update={
                    "validation_video_width": vw,
                    "validation_video_height": vh,
                    "validation_video_frames": vf,
                }
            )
            logger.info(
                "LoRA trainer: validation video_dims auto-matched to training "
                "bucket %dx%dx%d (kept default 576x576x49 -> bucket aspect)",
                vw, vh, vf,
            )
        validation_samples = build_validation_sample_dicts(
            prompt_samples=effective_prompts,
            holdout=holdout_specs,
            dataset_type=dataset_type,
            reference_downscale_factor=ref_downscale,
        )
        validation_refs = build_validation_sample_refs(
            prompt_samples=effective_prompts,
            holdout=holdout_specs,
            dataset_type=dataset_type,
            reference_downscale_factor=ref_downscale,
        )
        yaml_text = build_training_yaml(
            config=config_for_yaml.model_copy(
                update={"validation_prompts": effective_prompts}
            ),
            dataset_type=dataset_type,
            model_path=creds.model_path,
            text_encoder_path=creds.text_encoder_path,
            preprocessed_data_root=precomputed_dir,
            output_dir=remote_output,
            validation_samples=validation_samples,
        )
        # Stage the config locally then upload its dir. The staged filename
        # must match the remote config path's basename so train.py finds it.
        staging = self._config.app_data_dir / "lora" / "configs" / job.id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        # Split the remote (POSIX) config path by hand — Path() would apply the
        # host OS's separator rules to a remote path.
        remote_config_dir, _, config_filename = remote_config.rpartition("/")
        (staging / config_filename).write_text(yaml_text, encoding="utf-8")
        self._target.upload_directory(
            credentials=creds,
            handle=handle,
            local_dir=str(staging),
            remote_dir=remote_config_dir,
        )
        command = paths.cache_env_prefix(workspace) + paths.train_command(
            config_yaml_path=remote_config
        )
        logger.info(
            "LoRA trainer: starting training run '%s' (%d steps) on the remote GPU",
            job.name,
            job.config.steps,
        )
        job_id = self._target.start_command(
            credentials=creds,
            handle=handle,
            command=command,
            workdir=paths.trainer_workdir(workspace),
        )
        self._handler.set_training_running(
            job.id,
            handle=handle,
            remote_job_id=job_id,
            remote_output_dir=remote_output,
        )
        # Record the validation sample refs so later polls can map downloaded
        # validation artifacts back to their prompt/source for the feed.
        self._handler.set_validation_sample_refs(job.id, validation_refs)

    def _holdout_validation_specs(
        self, dataset: LoraDataset, dataset_type: LoraDatasetType
    ) -> list[ValidationSampleSpec]:
        """Build validation sample specs from the dataset's held-out clips.

        A held-out clip (`triage="holdout"`, not soft-deleted) is reserved for
        the validation feed: it is excluded from training (see `_build_staging`)
        and, for IC-LoRA, its reference video was staged at upload to
        ``{remote_dataset_dir}/holdout/{clip.id}.mp4``. We turn each into a
        `ValidationSampleSpec`:

        - IC-LoRA: prompt = the clip's caption, reference = the staged input
          video's absolute remote path. Clips without a caption or without a
          staged reference are skipped (the builder drops reference-less IC-LoRA
          samples anyway, but filtering here keeps the feed refs honest).
        - text-to-video: prompt = the clip's caption, no reference.

        Order is stable (dataset clip order) so the 1-based ``sample_index`` the
        trainer writes maps deterministically to the recorded refs.
        """
        remote_base = (dataset.remote_dataset_dir or "").rstrip("/")
        specs: list[ValidationSampleSpec] = []
        for clip in dataset.clips:
            if clip.triage != "holdout" or clip.deleted_at:
                continue
            caption = clip.caption.strip()
            if not caption:
                continue
            if dataset_type == "ic_lora":
                if not remote_base:
                    continue
                reference_video_path = f"{remote_base}/{prep.holdout_reference_relpath(clip.id)}"
                specs.append(
                    ValidationSampleSpec(
                        prompt=caption, reference_video_path=reference_video_path
                    )
                )
            else:
                specs.append(ValidationSampleSpec(prompt=caption))
        # IC-LoRA validation requires a reference video, so a user who didn't
        # mark any holdout clip would otherwise get NO validation feed. Fall
        # back to the first training clip: its reference video was staged at
        # upload (see `_build_staging` → `stage_holdout_references`) to
        # `holdout/{clip.id}.mp4`, so we can condition a validation sample on
        # it. The clip is also in the training set, so this monitors progress
        # rather than generalization — surfaced in the modal as "auto-picked".
        if dataset_type == "ic_lora" and not specs and remote_base:
            auto = self._auto_pick_validation_clip(dataset)
            if auto is not None:
                specs.append(
                    ValidationSampleSpec(
                        prompt=auto.caption.strip(),
                        reference_video_path=f"{remote_base}/{prep.holdout_reference_relpath(auto.id)}",
                    )
                )
        return specs

    def _effective_validation_prompts(
        self, config_prompts: list[str], dataset: LoraDataset
    ) -> list[str]:
        """Resolve the validation prompts that actually ship in the YAML.

        Always auto-seeds up to three distinct training-clip captions
        (preferring ones that mention the trigger word) so validation runs
        against real, trigger-word-bearing prompts instead of a generic
        placeholder — the official low-vram config uses two rich descriptive
        prompts, and a single fixed prompt + fixed seed is why validations
        look near-identical across checkpoints. The user's explicit prompts
        (any non-empty value that isn't the generic placeholder) are kept and
        merged on top, deduped. When there are no captions to seed from, a
        richer trigger-word fallback replaces the placeholder so a
        not-yet-captioned run still validates on-concept. IC-LoRA ignores bare
        prompts (its validation uses reference videos), but the resolved list
        is still recorded on the config / YAML for the t2v case.
        """
        user_prompts = [
            p.strip()
            for p in config_prompts
            if p.strip() and p.strip() != DEFAULT_VALIDATION_PROMPT
        ]
        auto = self._auto_seed_from_captions(dataset)
        if not auto:
            auto = self._fallback_trigger_prompts(dataset)
        # User's explicit prompts first (honor intent), then auto-seeded
        # captions to top up toward trigger-word diversity. Dedup case-insensitively.
        merged: list[str] = []
        seen: set[str] = set()
        for prompt in user_prompts + auto:
            key = prompt.lower()
            if not prompt or key in seen:
                continue
            seen.add(key)
            merged.append(prompt)
        return merged

    def _fallback_trigger_prompts(self, dataset: LoraDataset) -> list[str]:
        """Richer trigger-word validation prompts when no caption is available
        to seed from (e.g. a captionless t2v run before auto-captioning). Falls
        back to nothing when there's no trigger word — a run with no trigger and
        no captions stays validation-less rather than emitting the generic
        placeholder, matching the prior "empty means no validation" contract.
        """
        trigger = (dataset.trigger_word or "").strip()
        if not trigger:
            return []
        return [
            f"A high quality video of {trigger}, detailed and cinematic.",
            f"{trigger} in a well-lit, realistic scene with natural motion.",
        ]

    def _auto_seed_from_captions(self, dataset: LoraDataset) -> list[str]:
        """Up to three distinct training-clip captions, preferring the trigger word."""
        trigger = (dataset.trigger_word or "").strip().lower()
        seeded: list[str] = []
        seen: set[str] = set()

        def consider(cap: str) -> bool:
            key = cap.lower()
            if not cap or key in seen:
                return False
            seen.add(key)
            seeded.append(cap)
            return True

        # First pass: captions mentioning the trigger word (most on-concept).
        if trigger:
            for clip in dataset.clips:
                if len(seeded) >= 3:
                    break
                if clip.deleted_at or clip.triage in ("reject", "holdout"):
                    continue
                cap = clip.caption.strip()
                if cap and trigger in cap.lower():
                    consider(cap)
        # Second pass: any other training caption, up to three total.
        for clip in dataset.clips:
            if len(seeded) >= 3:
                break
            if clip.deleted_at or clip.triage in ("reject", "holdout"):
                continue
            consider(clip.caption.strip())
        return seeded[:3]

    def _resolve_validation_dims(
        self, job: TrainingJob, preprocessed: PreprocessedDataset
    ) -> tuple[int, int, int] | None:
        """Match validation `video_dims` to the training bucket's aspect when
        the user kept the default 576x576x49.

        Training defaults to a 16:9 bucket (768x448) while validation defaults
        to 1:1 (576x576) — the official low-vram config's validation dims — so
        validation composes at an aspect ratio the LoRA never trained on, which
        is one reason validation looks worse than real inference. Treating
        576x576x49 as an "auto" sentinel, this snaps validation to the primary
        training bucket's dims (so low_vram at 512x512 validates at 512x512,
        and a 16:9 run validates at 16:9). A user who set custom validation
        dims (anything other than the sentinel) is left alone. Returns the
        (w, h, frames) to use, or None to keep the configured dims.
        """
        cfg = job.config
        if (
            cfg.validation_video_width,
            cfg.validation_video_height,
            cfg.validation_video_frames,
        ) != (576, 576, 49):
            return None
        buckets = (
            preprocessed.effective_resolution_buckets
            or preprocessed.resolution_buckets
        )
        primary = (buckets or "").split(";")[0].strip()
        try:
            w_s, h_s, f_s = primary.split("x")
            w, h, f = int(w_s), int(h_s), int(f_s)
        except ValueError:
            return None
        if w < 32 or h < 32 or f < 1 or w % 32 or h % 32 or f % 8 != 1:
            return None
        return (w, h, f)

    def _auto_pick_validation_clip(self, dataset: LoraDataset) -> LoraClip | None:
        """First training clip with a caption and a reference, for IC-LoRA fallback."""
        for clip in dataset.clips:
            if clip.deleted_at or clip.triage in ("reject", "holdout"):
                continue
            if not clip.caption.strip():
                continue
            if not refs_of(clip):
                continue
            return clip
        return None


    def _update_progress(
        self, job: TrainingJob, creds: TrainerCredentials, handle: TargetHandle
    ) -> None:
        if job.target is None or job.target.remote_job_id is None:
            return
        # Pull any new validation samples into the results feed (independent of
        # the step parse below — samples land at the validation interval, not
        # every step). Best-effort: a transport blip just skips this tick.
        self._poll_validation_feed(job, creds, handle)
        # Likewise pull any new adapter checkpoints so the user can reveal them
        # next to their validation samples as they train.
        self._poll_checkpoints(job, creds, handle)
        try:
            lines = self._target.read_logs(
                credentials=creds,
                handle=handle,
                remote_job_id=job.target.remote_job_id,
                tail=_LOG_TAIL,
            )
        except TrainerTargetError:
            return
        # Persist the highest checkpoint step seen so far. The trainer logs each
        # save as `lora_weights_step_NNNNN.safetensors`; persisting it here means
        # a redownload (after a restart, when the ephemeral container log is
        # gone) can still resolve a real checkpoint instead of guessing the
        # configured final step. Only writes when the step advances (handler).
        ckpt_step = _latest_checkpoint_step(lines)
        if ckpt_step is not None:
            self._handler.update_training_checkpoint_step(job.id, ckpt_step)
        last_step: int | None = None
        last_total: int | None = None
        last_step_line = ""
        for line in lines:
            match = _STEP_RE.search(line)
            if match is not None:
                last_step = int(match.group(1))
                last_total = int(match.group(2))
                last_step_line = line
        if last_step is not None:
            eta = self._estimate_training_eta(job.id, last_step, last_total)
            self._handler.update_training_progress(
                job.id,
                current_step=last_step,
                total_steps=last_total,
                eta_seconds=eta,
            )
            # Mirror the card's progress into the app log so a long run visibly
            # advances there too. One line per step advance — at the ~15s poll
            # cadence this is a steady heartbeat, not a flood, and stays silent
            # while the step is frozen (e.g. a checkpoint-save pause).
            if last_step != self._last_logged_train_step.get(job.id):
                self._last_logged_train_step[job.id] = last_step
                loss_match = _LOSS_RE.search(last_step_line)
                loss_txt = f", loss {float(loss_match.group(1)):.4g}" if loss_match else ""
                logger.info(
                    "LoRA training '%s': step %d/%d%s (ETA %s)",
                    job.name,
                    last_step,
                    last_total or 0,
                    loss_txt,
                    _format_eta(eta),
                )
            return
        # No step parsed yet — model load / int8 quantization can take minutes
        # before step 1 appears. Heartbeat the latest log line so the user sees
        # activity instead of a silent log. Stops once steps begin advancing.
        last_line = _latest_meaningful_line(lines)
        if last_line:
            self._handler.set_training_status_detail(
                job.id, _training_setup_phase(last_line)
            )
        if last_line and last_line != self._last_logged_train_line.get(job.id):
            self._last_logged_train_line[job.id] = last_line
            logger.info("LoRA training '%s': %s", job.name, last_line)

    def _poll_validation_feed(
        self, job: TrainingJob, creds: TrainerCredentials, handle: TargetHandle
    ) -> None:
        """Detect + download new validation samples into the training-results feed.

        Lists the run's remote ``samples/`` dir for artifacts newer than the
        newest feed entry, downloads each, and appends a `ValidationFeedItem`
        per artifact. Best-effort: a `TrainerTargetError` (transport blip, pod
        gone) skips this tick rather than failing the run. No-op when the run
        has no validation samples configured (validation disabled).
        """
        if not job.remote_output_dir or not job.validation_sample_refs:
            return
        since_step = max((i.step for i in job.validation_feed), default=0)
        try:
            artifacts = self._target.list_validation_outputs(
                credentials=creds,
                handle=handle,
                remote_output_dir=job.remote_output_dir,
                since_step=since_step,
            )
        except TrainerTargetError:
            return
        if not artifacts:
            return
        items = self._download_validation_artifacts(job, artifacts, creds, handle)
        if items:
            self._handler.append_validation_feed_items(job.id, items)

    def _download_validation_artifacts(
        self,
        job: TrainingJob,
        artifacts: list[ValidationArtifact],
        creds: TrainerCredentials,
        handle: TargetHandle,
    ) -> list[ValidationFeedItem]:
        """Download validation artifacts and map each to a feed item.

        `sample_index` is 1-based into the run's `validation_sample_refs`; an
        out-of-range index (trainer wrote more samples than we configured —
        e.g. a future trainer version) degrades to a prompt-less entry rather
        than crashing. A failed download skips that one artifact but keeps the
        rest (one bad transfer shouldn't drop the whole tick's feed).
        """
        feed_dir = self._local_run_dir(job) / "validations"
        feed_dir.mkdir(parents=True, exist_ok=True)
        refs = job.validation_sample_refs
        created_at = _utcnow().isoformat()
        items: list[ValidationFeedItem] = []
        for art in artifacts:
            ref = (
                refs[art.sample_index - 1]
                if 0 < art.sample_index <= len(refs)
                else None
            )
            local_path = (
                feed_dir / f"step_{art.step:06d}_{art.sample_index}.{art.ext}"
            )
            try:
                self._target.download_file(
                    credentials=creds,
                    handle=handle,
                    remote_path=art.remote_path,
                    local_path=str(local_path),
                )
            except TrainerTargetError:
                logger.warning(
                    "LoRA training '%s': failed to download validation sample "
                    "step %d index %d — skipping",
                    job.name,
                    art.step,
                    art.sample_index,
                )
                continue
            items.append(
                ValidationFeedItem(
                    step=art.step,
                    sample_index=art.sample_index,
                    local_path=str(local_path),
                    extension=art.ext,
                    source=ref.source if ref is not None else "prompt",
                    prompt=ref.prompt if ref is not None else "",
                    reference_local_path=(
                        ref.reference_local_path if ref is not None else None
                    ),
                    created_at=created_at,
                )
            )
        return items

    def _poll_checkpoints(
        self, job: TrainingJob, creds: TrainerCredentials, handle: TargetHandle
    ) -> None:
        """Detect + download new adapter checkpoints as the trainer saves them.

        Lists the run's remote ``checkpoints/`` dir for steps not yet
        downloaded, fetches each into the per-run ``checkpoints/`` folder, and
        appends a `CheckpointArtifact` per file. Best-effort: a
        `TrainerTargetError` (transport blip, pod gone) skips this tick rather
        than failing the run. No-op until the run has a remote output dir.
        """
        if not job.remote_output_dir:
            return
        have = {c.step for c in job.checkpoints}
        try:
            steps = self._target.list_checkpoints(
                credentials=creds,
                handle=handle,
                remote_output_dir=job.remote_output_dir,
            )
        except TrainerTargetError:
            return
        new_steps = sorted(s for s in steps if s not in have)
        if not new_steps:
            return
        items = self._download_checkpoints(job, new_steps, creds, handle)
        if items:
            self._handler.append_checkpoint_artifacts(job.id, items)

    def _download_checkpoints(
        self,
        job: TrainingJob,
        steps: list[int],
        creds: TrainerCredentials,
        handle: TargetHandle,
    ) -> list[CheckpointArtifact]:
        """Download adapter checkpoints for the given steps.

        A failed download skips that one checkpoint but keeps the rest (one bad
        transfer shouldn't drop the whole tick's set). `remote_path` is built
        from the run's own output dir (server-derived) via the shared
        `lora_command_builder` helper, so no user-controlled value reaches the
        remote path.
        """
        assert job.remote_output_dir is not None
        ckpt_dir = self._local_run_dir(job) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        created_at = _utcnow().isoformat()
        items: list[CheckpointArtifact] = []
        for step in steps:
            remote_path = paths.lora_checkpoint_path_in(job.remote_output_dir, step)
            local_path = ckpt_dir / f"lora_weights_step_{step:05d}.safetensors"
            try:
                self._target.download_file(
                    credentials=creds,
                    handle=handle,
                    remote_path=remote_path,
                    local_path=str(local_path),
                )
            except TrainerTargetError:
                logger.warning(
                    "LoRA training '%s': failed to download checkpoint step %d — skipping",
                    job.name,
                    step,
                )
                continue
            items.append(
                CheckpointArtifact(
                    step=step,
                    remote_path=remote_path,
                    local_path=str(local_path),
                    created_at=created_at,
                )
            )
        return items

    def _poll_gpu_status(
        self, job: TrainingJob, creds: TrainerCredentials, handle: TargetHandle
    ) -> None:
        """Refresh the run's GPU telemetry snapshot for the status panel.

        Throttled to one `nvidia-smi` query per job per
        `_GPU_QUERY_MIN_INTERVAL_SECONDS`. Best-effort: a `TrainerTargetError`
        (transport blip, pod gone) skips this tick and keeps the last known
        status rather than failing the run.
        """
        now = time.monotonic()
        last = self._last_gpu_query_at.get(job.id, 0.0)
        if now - last < _GPU_QUERY_MIN_INTERVAL_SECONDS:
            return
        self._last_gpu_query_at[job.id] = now
        try:
            telemetry = self._target.query_gpu(credentials=creds, handle=handle)
        except TrainerTargetError:
            return
        self._handler.set_training_gpu_status(
            job.id,
            GpuStatus(
                name=telemetry.name,
                vram_total_mb=telemetry.vram_total_mb,
                vram_used_mb=telemetry.vram_used_mb,
                gpu_util_pct=telemetry.gpu_util_pct,
                mem_util_pct=telemetry.mem_util_pct,
                temp_c=telemetry.temp_c,
                updated_at=_utcnow().isoformat(),
            ),
        )

    def _estimate_training_eta(
        self, job_id: str, step: int, total: int | None
    ) -> int | None:
        """Seconds remaining, from an EMA-smoothed step rate across polls.

        Returns None until two advancing samples exist (and ignores polls where
        the step didn't move, so a checkpoint-save pause doesn't spike the rate).
        """
        now = time.monotonic()
        prev = self._train_rate_samples.get(job_id)
        if prev is None:
            self._train_rate_samples[job_id] = (step, now, 0.0)
            return None
        prev_step, prev_time, prev_rate = prev
        if step <= prev_step or now <= prev_time:
            return None  # no advance yet — keep the prior sample/rate
        inst_rate = (step - prev_step) / (now - prev_time)  # steps/sec
        # EMA smooths the per-interval noise (a slow first interval after model
        # load, a fast burst after, occasional validation/checkpoint pauses).
        ema_rate = inst_rate if prev_rate <= 0 else 0.3 * inst_rate + 0.7 * prev_rate
        self._train_rate_samples[job_id] = (step, now, ema_rate)
        if total is None or ema_rate <= 0:
            return None
        return int(max(0, total - step) / ema_rate)

    def _local_run_dir(self, job: TrainingJob) -> Path:
        """Per-run output folder: `lora/trained/<name>-<shortid>/`.

        Holds the weights, run-summary.md and training-config.json so a finished
        run is self-describing on disk (vs. a wall of bare-UUID .safetensors).
        """
        return (
            self._config.app_data_dir
            / "lora"
            / "trained"
            / paths.remote_slug(job.name, job.id)
        )

    def _local_lora_path(self, job: TrainingJob) -> Path:
        return self._local_run_dir(job) / _lora_weights_filename(job)

    # ------------------------------------------------------------------
    # Workspace release (called by the delete-dataset route)
    # ------------------------------------------------------------------

    def read_job_logs(self, job: TrainingJob) -> list[str]:
        """Tail the remote training logs for the logs route.

        Returns an empty list when the job hasn't reached the remote
        yet. Resolves the dataset's workspace handle (where the job
        actually runs) the same way the reconciler does.
        """
        if job.target is None or job.target.remote_job_id is None:
            return []
        preprocessed = self._handler.get_preprocessed(job.preprocessed_id)
        if preprocessed is None:
            return []
        dataset = self._handler.get_dataset(preprocessed.dataset_id)
        if dataset is None or dataset.target is None:
            return []
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(
            settings,
            dataset.target.provider,
            workspace_policy=dataset.workspace_policy,
            cache_volume_id=dataset.cache_volume_id,
        )
        return self._target.read_logs(
            credentials=creds,
            handle=dataset.target,
            remote_job_id=job.target.remote_job_id,
            tail=_LOG_TAIL,
        )

    def test_connection(self) -> None:
        """Validate the configured RunPod credentials.

        Raises `TrainerTargetError` on failure; the route maps that to a
        user-facing ok/message payload.
        """
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(settings)
        self._target.test_connection(credentials=creds)

    def terminate_runpod_pod(self, pod_id: str) -> None:
        """Terminate a specific RunPod pod by id (connect UI action).

        Raises `TrainerTargetError` on failure; the route maps it to ok/message.
        """
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(
            settings, "runpod", workspace_policy="ephemeral_any_region"
        )
        self._target.release_workspace(
            credentials=creds, handle=TargetHandle(provider="runpod", pod_id=pod_id)
        )
        self._handler.mark_pod_stopped(pod_id)

    def list_runpod_pods(self) -> list[PodInfo]:
        """Pods currently on the account for the Trainer compute panel.

        Standalone list (no GPU/volume discovery) so the user can monitor and
        control stray pods. Raises `TrainerTargetError` (e.g. bad key); the
        route maps it to a 4xx.
        """
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(settings, "runpod")
        return self._target.list_pods(credentials=creds)

    def stop_runpod_pod(self, pod_id: str) -> None:
        """Pause a running RunPod pod (stop GPU billing, keep disk)."""
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(settings, "runpod")
        self._target.stop_pod(credentials=creds, pod_id=pod_id)
        self._handler.mark_pod_stopped(pod_id)

    def resume_runpod_pod(self, pod_id: str) -> None:
        """Start a stopped RunPod pod (resume GPU billing)."""
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(settings, "runpod")
        self._target.resume_pod(credentials=creds, pod_id=pod_id)

    def connect_runpod(
        self,
    ) -> tuple[
        AccountInfo,
        str | None,
        tuple[SavedModelReadiness, int | None],
        dict[str, tuple[SavedModelReadiness, int | None]],
    ]:
        """Read-only account discovery; never creates paid storage."""
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(settings, "runpod")
        account = self._target.connect_account(credentials=creds)
        self._handler.reconcile_saved_model_volumes({v.id for v in account.volumes})
        configured_volume_id = settings.runpod_network_volume_id or None
        active_volume_id: str | None = (
            configured_volume_id if settings.runpod_keep_model_cached else None
        )
        # Auto-heal a stale volume id: if the configured volume was deleted on
        # RunPod (no longer in the account's list), clear it so we don't keep
        # trying to mount a missing volume.
        if active_volume_id and all(v.id != active_volume_id for v in account.volumes):
            logger.warning(
                "RunPod: configured network volume %s no longer exists — clearing it",
                active_volume_id,
            )
            active_volume_id = None
            self._settings.update_settings(
                AppSettingsPatch.model_validate({"runpod_network_volume_id": ""})
            )
        fingerprint = self._model_fingerprint(creds)
        readiness_by_volume = {
            volume.id: self._handler.saved_model_readiness(
                volume_id=volume.id,
                fingerprint=fingerprint,
                estimated_download_bytes=ESTIMATED_MODEL_DOWNLOAD_BYTES,
            )
            for volume in account.volumes
        }
        active_readiness = (
            readiness_by_volume.get(active_volume_id)
            if active_volume_id is not None
            else None
        ) or ("missing", ESTIMATED_MODEL_DOWNLOAD_BYTES)
        readiness = (active_readiness[0], ESTIMATED_MODEL_DOWNLOAD_BYTES)
        return account, active_volume_id, readiness, readiness_by_volume

    def select_runpod_volume(self, volume_id: str) -> NetworkVolume:
        """Select one existing app-owned volume as the sole primary cache."""
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(settings, "runpod")
        account = self._target.connect_account(credentials=creds)
        volume = next((v for v in account.volumes if v.id == volume_id), None)
        if volume is None:
            raise TrainerTargetError(
                f"RunPod volume {volume_id} was not found. Refresh the account "
                "and choose an existing cache.",
                retryable=False,
            )
        if not volume.created_by_app:
            raise TrainerTargetError(
                f"Refusing to select RunPod volume {volume_id}: it was not "
                "created by LTX Desktop",
                retryable=False,
            )
        self._settings.update_settings(
            AppSettingsPatch.model_validate(
                {
                    "runpod_network_volume_id": volume.id,
                    "runpod_keep_model_cached": True,
                }
            )
        )
        return volume

    def disable_runpod_cache(self) -> None:
        """Detach the primary cache for new work without deleting paid storage."""
        self._settings.update_settings(
            AppSettingsPatch.model_validate({"runpod_keep_model_cached": False})
        )

    def create_runpod_volume(
        self, *, datacenter_id: str | None, size_gb: int | None
    ) -> NetworkVolume:
        """Explicitly create and select a new app-owned primary cache."""
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(settings, "runpod")
        volume = self._target.ensure_network_volume(
            credentials=creds,
            name=f"{RUNPOD_VOLUME_NAME}-{uuid.uuid4().hex[:8]}",
            size_gb=size_gb
            or settings.runpod_volume_size_gb
            or DEFAULT_RUNPOD_VOLUME_SIZE_GB,
            datacenter_id=datacenter_id,
        )
        self._settings.update_settings(
            AppSettingsPatch.model_validate(
                {
                    "runpod_network_volume_id": volume.id,
                    "runpod_keep_model_cached": True,
                }
            )
        )
        return volume

    def relocate_runpod_volume(
        self, *, datacenter_id: str, size_gb: int | None
    ) -> tuple[NetworkVolume, str | None]:
        """Stage a replacement cache and switch new work to it.

        Provisioning the ~68GB model set can take tens of minutes and requires a
        billable GPU pod, so it is intentionally not performed inside this HTTP
        operation.  The replacement is created empty, atomically selected for
        *new* pipelines, and provisioned by the normal first-work reconciler.
        The old volume is returned and is never deleted implicitly.
        """
        settings = self._settings.get_settings_snapshot()
        # The remembered volume can still own recoverable remote artifacts even
        # while caching is disabled.  Detaching it must not bypass relocation's
        # dependency guard.
        old_volume_id = settings.runpod_network_volume_id or None
        if old_volume_id:
            active, recovery = self._handler.cache_volume_dependencies(old_volume_id)
            if active:
                raise TrainerTargetError(
                    "Cannot relocate the primary cache while active jobs use it: "
                    + ", ".join(active[:5]),
                    retryable=False,
                )
            if recovery:
                raise TrainerTargetError(
                    "Cannot relocate the primary cache while remote recovery "
                    "artifacts depend on it. Finish/download or delete the "
                    "dependent datasets first: "
                    + ", ".join(recovery[:5]),
                    retryable=False,
                )
        replacement = self.create_runpod_volume(
            datacenter_id=datacenter_id, size_gb=size_gb
        )
        return replacement, old_volume_id

    def delete_runpod_volume(self, volume_id: str) -> None:
        """Delete an inactive app-owned volume after durable dependency checks."""
        settings = self._settings.get_settings_snapshot()
        active, recovery = self._handler.cache_volume_dependencies(volume_id)
        if active:
            raise TrainerTargetError(
                "Cannot delete a cache volume while active jobs use it: "
                + ", ".join(active[:5]),
                retryable=False,
            )
        if recovery:
            raise TrainerTargetError(
                "Cannot delete a cache volume while remote recovery artifacts "
                "depend on it: "
                + ", ".join(recovery[:5]),
                retryable=False,
            )
        creds = self._credentials(settings, "runpod")
        self._target.delete_network_volume(
            credentials=creds, volume_id=volume_id
        )
        self._handler.remove_saved_model_volume(volume_id)
        if settings.runpod_network_volume_id == volume_id:
            self._settings.update_settings(
                AppSettingsPatch.model_validate(
                    {
                        "runpod_network_volume_id": "",
                        "runpod_keep_model_cached": False,
                    }
                )
            )

    def release_workspace_for_dataset(self, dataset: LoraDataset) -> None:
        """Best-effort release of the dataset's remote compute.

        Called after the handler accepts a dataset deletion so rejected
        deletes cannot stop active work. The caller retains the dataset
        snapshot containing the target handle. No-op when the dataset never
        provisioned compute.
        """
        if dataset.target is None:
            return
        settings = self._settings.get_settings_snapshot()
        creds = self._credentials(
            settings,
            dataset.target.provider,
            workspace_policy=dataset.workspace_policy,
            cache_volume_id=dataset.cache_volume_id,
        )
        try:
            self._target.release_workspace(credentials=creds, handle=dataset.target)
        except TrainerTargetError as exc:
            logger.warning(
                "LoRA trainer: failed to release workspace for dataset %s: %s",
                dataset.id,
                exc.detail,
            )
