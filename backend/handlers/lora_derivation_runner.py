"""Background worker for target/variant derivation jobs.

The LoRA studio's "generate a target" pipeline is multi-stage and slow:
optionally edit a frame with Nano Banana (remote), then drive it into
motion either with the local IC-LoRA depth/canny pipeline (single-flight
GPU) or remote Kling motion-control. Running it inline would block the
request and freeze the UI, so jobs are persisted (`DerivationJobsState`)
and drained here, one at a time, on a dedicated thread.

Some jobs gate on a user review of the edited still (`require_review`): the
runner does the cheap Nano Banana edit, then pauses the job in `review` and
stops. The user approves it (-> `approved`) and the runner claims it again for
a motion-only second phase (it reuses the approved still, never re-editing),
or regenerates the edit (-> `pending`, re-runs the edit phase). This lets the
expensive motion drive fire only for stills the user has actually seen.

Bounded concurrency: a dispatcher thread launches up to N worker threads
(N = the user's `lora_fal_concurrency` setting, read live each dispatch
so changes apply without a restart). Remote Fal jobs (Nano Banana edits,
Kling / Kling O3) then run in parallel, which is the big win for bulk
generation. Local IC-LoRA drives stay effectively single-flight: they
wait for the one GPU slot and back off on a 409 ("already in progress"),
so even with many workers only one GPU drive runs at a time.

Transient Fal failures (HTTP 429 rate limits, timeouts, 5xx) auto-retry
with exponential backoff + jitter inside the worker; permanent errors
(e.g. 422 "video too large", bad input, missing API key) fail fast — no
point re-spending the call.

Lock discipline matches the rest of the backend: the handler owns all
state mutation (claim/advance/complete/fail under its lock); this runner
performs the slow edit/drive work outside the lock and reports results
back.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from typing import TYPE_CHECKING, Callable, TypeVar

from api_types import IcLoraGenerateRequest, IcLoraImageInput
from logging_policy import log_background_exception
from state.lora_derivation_jobs_state import DerivationJob

if TYPE_CHECKING:
    from handlers.generation_handler import GenerationHandler
    from handlers.ic_lora_handler import IcLoraHandler
    from handlers.lora_training_handler import LoraTrainingHandler

logger = logging.getLogger(__name__)

_BACKGROUND_TASK_NAME = "lora-derivation-runner"
DEFAULT_POLL_SECONDS: float = 30.0
DEFAULT_STOP_JOIN_TIMEOUT_SECONDS: float = 30.0
# How long to wait for the GPU to free before giving up on a local drive.
DEFAULT_GPU_WAIT_SECONDS: float = 1800.0
_GPU_POLL_SECONDS: float = 2.0

# Transient Fal failures (rate limits, timeouts, 5xx) retry with capped
# exponential backoff + jitter before the job is failed.
_MAX_FAL_ATTEMPTS: int = 4
_RETRY_BASE_SECONDS: float = 2.0
_RETRY_CAP_SECONDS: float = 30.0
# Fal HTTP statuses worth retrying (429 + server-side 5xx). Everything
# else (400/401/403/404/422 …) is a permanent client error.
_TRANSIENT_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

T = TypeVar("T")


class DerivationCancelled(Exception):
    """Raised internally when a job's cancel flag is observed mid-pipeline."""


class LoraDerivationRunner:
    def __init__(
        self,
        *,
        handler: "LoraTrainingHandler",
        ic_lora: "IcLoraHandler",
        generation: "GenerationHandler",
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        gpu_wait_seconds: float = DEFAULT_GPU_WAIT_SECONDS,
        stop_join_timeout_seconds: float = DEFAULT_STOP_JOIN_TIMEOUT_SECONDS,
        max_fal_attempts: int = _MAX_FAL_ATTEMPTS,
        retry_base_seconds: float = _RETRY_BASE_SECONDS,
        retry_cap_seconds: float = _RETRY_CAP_SECONDS,
    ) -> None:
        self._handler = handler
        self._ic_lora = ic_lora
        self._generation = generation
        self._poll_seconds = poll_seconds
        self._gpu_wait_seconds = gpu_wait_seconds
        self._stop_join_timeout = stop_join_timeout_seconds
        self._max_fal_attempts = max_fal_attempts
        self._retry_base_seconds = retry_base_seconds
        self._retry_cap_seconds = retry_cap_seconds
        self._wakeup = handler.derivation_wakeup_event
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        # In-flight worker accounting for the bounded pool. Only the
        # dispatcher launches workers (so the read-limit/claim/increment
        # sequence needs no extra synchronization); workers decrement on
        # exit under this lock and wake the dispatcher to top the pool up.
        self._active_lock = threading.Lock()
        self._active = 0
        self._worker_seq = 0

    # ------------------------------------------------------------------
    # Lifecycle (mirrors ClipJobsRunner)
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name=_BACKGROUND_TASK_NAME, daemon=True
            )
            self._thread.start()
            logger.info("LoRA derivation runner started")

    def stop(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._shutdown.set()
            self._wakeup.set()
            self._thread = None
        thread.join(timeout=self._stop_join_timeout)
        if not thread.is_alive():
            logger.info("LoRA derivation runner stopped")

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
                launched = self._dispatch()
            except Exception as exc:
                log_background_exception(_BACKGROUND_TASK_NAME, exc)
                launched = False
            if launched:
                # Capacity was used; loop to see if more can be launched.
                continue
            # Either the pool is full or there's nothing pending — sleep
            # until a job is enqueued (wakeup) or a worker frees a slot
            # (workers also set the wakeup on exit).
            self._wakeup.wait(timeout=self._poll_seconds)
            self._wakeup.clear()

    def _dispatch(self) -> bool:
        """Launch workers up to the live concurrency limit.

        Returns True if at least one worker was started this pass. Only the
        dispatcher thread claims jobs, so reading the limit, checking the
        active count, and claiming form a safe sequence without extra locks
        (workers only ever *decrement* the count).
        """
        launched = False
        while not self._shutdown.is_set():
            limit = self._handler.derivation_concurrency()
            with self._active_lock:
                if self._active >= limit:
                    return launched
            job = self._handler.claim_next_derivation_job()
            if job is None:
                return launched
            with self._active_lock:
                self._active += 1
                self._worker_seq += 1
                seq = self._worker_seq
            worker = threading.Thread(
                target=self._worker,
                args=(job,),
                name=f"{_BACKGROUND_TASK_NAME}-w{seq}",
                daemon=True,
            )
            worker.start()
            launched = True
        return launched

    def _worker(self, job: DerivationJob) -> None:
        try:
            self._process(job)
        except Exception as exc:  # defensive: _process already guards itself
            log_background_exception(f"{_BACKGROUND_TASK_NAME}:{job.id}", exc)
        finally:
            with self._active_lock:
                self._active -= 1
            # A slot freed — let the dispatcher top the pool back up.
            self._wakeup.set()

    def reconcile_once(self) -> None:
        """Process all currently-pending jobs synchronously (for tests)."""
        while self._drain():
            pass

    def _drain(self) -> bool:
        """Claim and process the next pending job inline. Returns True if one
        ran. Used by `reconcile_once` for deterministic, single-threaded
        testing; the live loop uses the bounded `_dispatch` pool instead."""
        job = self._handler.claim_next_derivation_job()
        if job is None:
            return False
        self._process(job)
        return True

    # ------------------------------------------------------------------
    # Work
    # ------------------------------------------------------------------

    def _process(self, job: DerivationJob) -> None:
        try:
            self._raise_if_cancelled(job.id)

            # Frame-edit-only job (from the frame-edit modal): edit the frame
            # and complete with the still — no animate step, no review. The
            # completed still folds into the gallery as a still entry that
            # remembers its driver (for a later motion-lock). GPU-wait for
            # Klein edits is handled inside _resolve_anchor, so a queued Klein
            # edit backs off while another generation is in flight instead of
            # failing with "already in progress".
            if job.direction == "frame_edit":
                anchor_path = self._resolve_anchor(job)
                if anchor_path is None:
                    # Paused for review — frame_edit jobs never set
                    # require_review, so this shouldn't happen; guard anyway.
                    return
                self._raise_if_cancelled(job.id)
                probe = self._handler.probe_clip(video_path=anchor_path)
                self._handler.complete_derivation_job(
                    job.id, derived_path=anchor_path, probe=probe
                )
                logger.info(
                    "lora.derivation frame_edit done job=%s out=%s",
                    job.id, anchor_path,
                )
                return

            anchor_path = self._resolve_anchor(job)
            if anchor_path is None:
                # Paused for user review — stop; the user resumes it.
                return
            self._raise_if_cancelled(job.id)

            # Stage 3: drive the still into motion.
            if job.engine == "kling":
                derived_path, probe = self._run_with_retry(
                    job,
                    lambda: self._handler.motion_edit_clip(
                        source_path=job.driver_path,
                        reference_image_path=anchor_path,
                        prompt=job.scene_prompt,
                        engine="kling_motion",
                        video_strength=0.5,
                        character_orientation=job.character_orientation,
                    ),
                )
            elif job.engine == "kling_o3":
                # Kling O3 v2v edit: the driver clip is the content; the prompt
                # drives the edit (required by the endpoint, so fall back to the
                # caption / a neutral default). The still is only sent as an
                # appearance reference (@Image1) when it's an actual Nano Banana
                # edit; otherwise it's a pure video + prompt re-render.
                prompt = (
                    job.scene_prompt.strip()
                    or job.caption.strip()
                    or "high quality video"
                )
                derived_path, probe = self._run_with_retry(
                    job,
                    lambda: self._handler.motion_edit_clip(
                        source_path=job.driver_path,
                        reference_image_path=anchor_path if job.frame_edited else None,
                        prompt=prompt,
                        engine="kling_o3",
                        video_strength=0.5,
                        character_orientation=job.character_orientation,
                        keep_audio=job.keep_audio,
                    ),
                )
            else:
                derived_path = self._drive_ltx_local(job, anchor_path)
                probe = self._handler.probe_clip(video_path=derived_path)

            self._handler.complete_derivation_job(
                job.id, derived_path=derived_path, probe=probe
            )
            logger.info(
                "lora.derivation done job=%s engine=%s out=%s", job.id, job.engine, derived_path
            )
        except DerivationCancelled:
            self._handler.fail_derivation_job(job.id, "Cancelled")
            logger.info("lora.derivation cancelled job=%s", job.id)
        except Exception as exc:
            log_background_exception(f"{_BACKGROUND_TASK_NAME}:{job.id}", exc)
            self._handler.fail_derivation_job(job.id, _error_message(exc))

    def _resolve_anchor(self, job: DerivationJob) -> str | None:
        """Build (or recover) the content-anchor still for the drive.

        Returns the anchor path, or `None` when the job is paused for review
        (the caller stops and the user resumes it).

        Two phases, keyed by the claimed status:
          - `generating`: motion-only (approved review) — reuse the already
            edited/approved still; never re-edit.
          - `editing`: build the anchor (extract frame + optional Nano Banana
            edit). If the job gates on review and an edit actually ran, pause
            in `review` instead of driving.
        """
        if job.status == "generating":
            anchor = job.edited_frame_path or job.frame_path
            if anchor is not None:
                return anchor
            # Defensive: rebuild from source with no edit (shouldn't happen,
            # an approved job always carries an edited still).
            result = self._run_with_retry(
                job,
                lambda: self._handler.prepare_content_anchor(
                    driver_path=job.driver_path,
                    frame_path=job.frame_path,
                    frame_time_seconds=job.frame_time_seconds,
                    edit_prompt="",
                    model=None,
                ),
            )
            return result.anchor_path

        # Edit phase (claimed from `pending` as `editing`).
        # A Klein edit runs on the single local GPU slot, so serialize it
        # behind any in-flight generation (cancel-aware) before editing —
        # Fal/Nano Banana edits are remote and need no such wait.
        if job.edit_engine == "klein" and job.edit_prompt.strip():
            self._wait_for_gpu(job.id)
        result = self._run_with_retry(
            job,
            lambda: self._handler.prepare_content_anchor(
                driver_path=job.driver_path,
                frame_path=job.frame_path,
                frame_time_seconds=job.frame_time_seconds,
                edit_prompt=job.edit_prompt,
                model=None,
                edit_engine=job.edit_engine,
            ),
        )
        anchor = result.anchor_path
        edited = bool(job.edit_prompt.strip())
        if job.require_review and edited:
            self._handler.mark_derivation_review(
                job.id,
                edited_frame_path=anchor,
                source_frame_path=result.source_frame_path,
            )
            logger.info("lora.derivation paused for review job=%s", job.id)
            return None
        self._handler.mark_derivation_generating(
            job.id,
            edited_frame_path=anchor if edited else None,
            source_frame_path=result.source_frame_path,
        )
        return anchor

    def _drive_ltx_local(self, job: DerivationJob, anchor_path: str) -> str:
        """Run the local IC-LoRA depth/canny drive, waiting for the GPU.

        The single GPU slot may be busy (the main queue runner mid-
        generation); wait for it to free, re-checking cancel, before
        invoking the IC-LoRA pipeline. The wait→generate window can still
        race the queue runner, so a 409 ("already in progress") just sends
        us back to waiting rather than failing the job.
        """
        # A non-empty prompt is required by the pipeline; fall back to the
        # caption, then a neutral default.
        prompt = job.scene_prompt.strip() or job.caption.strip() or "high quality video"
        req = IcLoraGenerateRequest(
            video_path=job.driver_path,
            conditioning_type=job.conditioning_type,
            prompt=prompt,
            conditioning_strength=job.conditioning_strength,
            images=[IcLoraImageInput(path=anchor_path, frame=0, strength=1.0)],
        )
        deadline = time.monotonic() + self._gpu_wait_seconds
        while True:
            self._wait_for_gpu(job.id)
            try:
                result = self._ic_lora.generate(req)
                break
            except Exception as exc:
                # Lost the GPU race to the queue runner — back off and retry
                # until the overall GPU-wait budget is exhausted.
                if getattr(exc, "status_code", None) == 409 and time.monotonic() < deadline:
                    time.sleep(_GPU_POLL_SECONDS)
                    continue
                raise
        if result.status == "cancelled":
            raise DerivationCancelled()
        return result.video_path

    def _wait_for_gpu(self, job_id: str) -> None:
        deadline = time.monotonic() + self._gpu_wait_seconds
        while self._generation.is_generation_running():
            self._raise_if_cancelled(job_id)
            if self._shutdown.is_set():
                raise DerivationCancelled()
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for the GPU to free up")
            time.sleep(_GPU_POLL_SECONDS)

    def _run_with_retry(self, job: DerivationJob, fn: Callable[[], T]) -> T:
        """Run a Fal-backed step, retrying transient failures with backoff.

        Permanent errors (and cancellation) propagate immediately; only
        rate-limit / timeout / 5xx failures are retried, up to
        `_MAX_FAL_ATTEMPTS`. Cancellation is honored during the backoff.
        """
        attempt = 0
        while True:
            self._raise_if_cancelled(job.id)
            try:
                return fn()
            except DerivationCancelled:
                raise
            except Exception as exc:
                attempt += 1
                if attempt >= self._max_fal_attempts or not _is_transient(exc):
                    raise
                base = min(
                    self._retry_cap_seconds,
                    self._retry_base_seconds * (2 ** (attempt - 1)),
                )
                delay = base + random.uniform(0, base * 0.25)
                logger.warning(
                    "lora.derivation transient fal error job=%s attempt=%d/%d "
                    "retrying in %.1fs: %s",
                    job.id, attempt, self._max_fal_attempts, delay, _error_message(exc),
                )
                self._sleep_with_cancel(job.id, delay)

    def _sleep_with_cancel(self, job_id: str, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._shutdown.is_set():
                raise DerivationCancelled()
            self._raise_if_cancelled(job_id)
            time.sleep(min(0.5, remaining))

    def _raise_if_cancelled(self, job_id: str) -> None:
        if self._handler.is_derivation_cancelled(job_id):
            raise DerivationCancelled()


def _error_message(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, str) and detail:
        return detail
    return str(exc) or exc.__class__.__name__


def _status_code_of(exc: Exception) -> int | None:
    """Best-effort HTTP status for a Fal failure.

    The Fal services bake the *real* upstream status into the message
    (e.g. "Fal video job failed (429): …"). We deliberately parse only the
    message and ignore any `status_code` attribute: those errors carry a
    generic local default (often 502) that would mis-flag every failure as
    a retryable server error. No parsable code → fall back to the
    network-wording heuristic in `_is_transient`.
    """
    match = re.search(r"\((\d{3})\)", _error_message(exc))
    return int(match.group(1)) if match else None


def _is_transient(exc: Exception) -> bool:
    """True if the failure is worth retrying (rate limit / timeout / 5xx)."""
    code = _status_code_of(exc)
    if code is not None:
        return code in _TRANSIENT_STATUSES
    # No status parsed — treat network-level wording as transient.
    text = _error_message(exc).lower()
    return any(
        kw in text for kw in ("timed out", "timeout", "connection", "temporarily")
    )
