"""Durable ledger for local clip-preparation jobs.

The dataset-curation gallery runs background, GPU-free work over the
user's clips — currently sprite/filmstrip generation for hover-scrub,
with motion/perceptual-hash analysis to follow on the same machinery.
These are short ffmpeg tasks but a curation set can hold dozens of
clips, so they run async on a bounded worker pool (`ClipJobsRunner`)
rather than blocking the request that enqueued them.

The work is keyed by *source path*, not by a persisted dataset clip:
the studio curates clips in the browser before any dataset is saved
(mirroring the stateless `probe`/`apply-edits` endpoints). The frontend
enqueues jobs, polls this ledger, and merges the results
(`sprite_path`/`poster_path`) onto its in-memory clips.

Persistence + crash recovery mirror the other LoRA ledgers: atomic JSON
write, and an in-flight (`running`) job with no result is reset to
`pending` on load so the runner re-attempts it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from api_types import LoraClipJobApi, LoraClipJobsResponse

# Only sprite generation in M0; motion/phash/caption reuse this ledger
# in later milestones, so the kind is a union from the start.
ClipJobKind = Literal["sprite"]
ClipJobStatus = Literal["pending", "running", "completed", "failed"]


class ClipJob(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    kind: ClipJobKind
    source_path: str
    status: ClipJobStatus
    created_at: str
    # sprite results (null until the job completes)
    poster_path: str | None = None
    sprite_path: str | None = None
    sprite_tiles: int | None = None
    error: str | None = None
    updated_at: str | None = None


class ClipJobsState(BaseModel):
    model_config = ConfigDict(strict=True)

    schema_version: int = 1
    jobs: list[ClipJob] = Field(default_factory=list[ClipJob])


def clip_job_to_api(job: ClipJob) -> LoraClipJobApi:
    return LoraClipJobApi(
        id=job.id,
        kind=job.kind,
        sourcePath=job.source_path,
        status=job.status,
        posterPath=job.poster_path,
        spritePath=job.sprite_path,
        spriteTiles=job.sprite_tiles,
        error=job.error,
    )


def clip_jobs_state_to_api(state: ClipJobsState) -> LoraClipJobsResponse:
    return LoraClipJobsResponse(jobs=[clip_job_to_api(j) for j in state.jobs])
