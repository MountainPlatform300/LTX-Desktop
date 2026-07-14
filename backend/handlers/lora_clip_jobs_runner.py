"""Background worker for local clip-prep jobs (sprite/filmstrip gen).

Separate from `LoraTrainingRunner` on purpose: these are fast, local,
GPU-free ffmpeg tasks that should never queue behind a multi-minute
remote training poll. A single dispatcher thread drains the durable
clip-jobs ledger and farms each job out to a small bounded
`ThreadPoolExecutor`, so a 50-clip curation set generates previews with
throttled concurrency instead of spawning 50 ffmpeg processes at once.

Lock discipline matches the rest of the backend: the handler owns all
state mutation (claim/complete/fail under its lock); this runner only
does the ffmpeg work outside the lock and reports results back.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from logging_policy import log_background_exception
from services.clip_processor.clip_processor import ClipProcessorError
from state.lora_clip_jobs_state import ClipJob

if TYPE_CHECKING:
    from handlers.lora_training_handler import LoraTrainingHandler
    from services.clip_processor.clip_processor import ClipProcessor

logger = logging.getLogger(__name__)

_BACKGROUND_TASK_NAME = "lora-clip-jobs-runner"
DEFAULT_POLL_SECONDS: float = 30.0
DEFAULT_MAX_WORKERS: int = 4
DEFAULT_SPRITE_TILES: int = 12
DEFAULT_SPRITE_WIDTH: int = 160
DEFAULT_STOP_JOIN_TIMEOUT_SECONDS: float = 30.0


class ClipJobsRunner:
    def __init__(
        self,
        *,
        handler: "LoraTrainingHandler",
        clip_processor: "ClipProcessor",
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        max_workers: int = DEFAULT_MAX_WORKERS,
        sprite_tiles: int = DEFAULT_SPRITE_TILES,
        sprite_width: int = DEFAULT_SPRITE_WIDTH,
        stop_join_timeout_seconds: float = DEFAULT_STOP_JOIN_TIMEOUT_SECONDS,
    ) -> None:
        self._handler = handler
        self._clip_processor = clip_processor
        self._poll_seconds = poll_seconds
        self._max_workers = max(1, max_workers)
        self._sprite_tiles = sprite_tiles
        self._sprite_width = sprite_width
        self._stop_join_timeout = stop_join_timeout_seconds
        self._wakeup = handler.clip_jobs_wakeup_event
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._lifecycle_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle (mirrors LoraTrainingRunner)
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown.clear()
            self._pool = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="lora-clip-job"
            )
            self._thread = threading.Thread(
                target=self._run_loop, name="lora-clip-jobs-runner", daemon=True
            )
            self._thread.start()
            logger.info("LoRA clip-jobs runner started (workers=%d)", self._max_workers)

    def stop(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            pool = self._pool
            if thread is None:
                return
            self._shutdown.set()
            self._wakeup.set()
            self._thread = None
            self._pool = None
        thread.join(timeout=self._stop_join_timeout)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        if not thread.is_alive():
            logger.info("LoRA clip-jobs runner stopped")

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
                self._dispatch()
            except Exception as exc:
                log_background_exception(_BACKGROUND_TASK_NAME, exc)
            self._wakeup.wait(timeout=self._poll_seconds)
            self._wakeup.clear()

    def reconcile_once(self) -> None:
        """Drive a single dispatch pass synchronously (for tests)."""
        self._dispatch()

    def _dispatch(self) -> None:
        pool = self._pool
        claimed = self._handler.claim_pending_clip_jobs()
        for job in claimed:
            if pool is None:
                # No pool (test path or stopped): run inline.
                self._process(job)
            else:
                pool.submit(self._process, job)

    # ------------------------------------------------------------------
    # Work
    # ------------------------------------------------------------------

    def _process(self, job: ClipJob) -> None:
        try:
            if job.kind == "sprite":
                self._process_sprite(job)
        except ClipProcessorError as exc:
            self._handler.fail_clip_job(job.id, exc.detail)
        except Exception as exc:
            log_background_exception(f"{_BACKGROUND_TASK_NAME}:{job.id}", exc)
            self._handler.fail_clip_job(job.id, f"Sprite generation failed: {exc}")

    def _process_sprite(self, job: ClipJob) -> None:
        probe = self._clip_processor.probe(video_path=job.source_path)
        poster_at = probe.duration_seconds * 0.25 if probe.duration_seconds > 0 else 0.0
        poster_bytes = self._clip_processor.extract_frame(
            video_path=job.source_path, time_seconds=poster_at
        )
        poster_path = self._handler.allocate_thumb_path(suffix=".png")
        poster_path.write_bytes(poster_bytes)
        # Publish the poster immediately so the gallery card drops its spinner
        # this poll cycle, rather than after the (full-decode) sprite finishes.
        self._handler.set_clip_job_poster(job.id, poster_path=str(poster_path))

        sprite_path = self._handler.allocate_thumb_path(suffix=".jpg")
        tiles = self._clip_processor.generate_sprite(
            video_path=job.source_path,
            out_path=str(sprite_path),
            tile_count=self._sprite_tiles,
            tile_width=self._sprite_width,
        )
        self._handler.complete_sprite_job(
            job.id,
            poster_path=str(poster_path),
            sprite_path=str(sprite_path),
            sprite_tiles=tiles,
        )
