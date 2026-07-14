"""Durable ledger for AI target/variant derivation jobs.

The LoRA studio's "generate a target" pipeline is multi-stage and can take
minutes (a Nano Banana frame edit, then a local IC-LoRA drive or a remote
Kling motion-control). Running it inline would block the request and the UI,
so jobs are persisted here and executed asynchronously by
`LoraDerivationRunner`, mirroring the clip-jobs ledger.

A job carries everything the pipeline needs (driver video, optional source
still, edit prompt, engine + control knobs) plus the bits the frontend needs
to fold the finished clip back into the gallery (`reference_path`,
`source_clip_id`, `caption`, and the result `derived_path` + `probe`).

Some jobs gate on a user review of the edited still before the (expensive)
motion drive (`require_review`): the runner pauses such a job in `review`
after the Nano Banana edit; the user then approves it (-> `approved`, claimed
for the motion-only phase) or regenerates the edit (-> back to `pending`).

Persistence + crash recovery mirror the other LoRA ledgers: atomic JSON
write. On load, a job caught mid-edit (`editing`) is reset to `pending`; a job
caught mid-drive (`generating`) is reset to `approved` if it already has an
approved edited still (so it isn't re-edited) else `pending`. `review` is a
stable paused state and survives untouched.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from api_types import (
    ConditioningType,
    LoraClipProbeApi,
    LoraDerivationDirectionApi,
    LoraDerivationEngineApi,
    LoraDerivationJobApi,
    LoraDerivationJobsResponse,
    LoraDerivationStatusApi,
    LoraFrameEditEngineApi,
)

DerivationEngine = LoraDerivationEngineApi
DerivationStatus = LoraDerivationStatusApi
DerivationDirection = LoraDerivationDirectionApi
FrameEditEngine = LoraFrameEditEngineApi


class DerivationJob(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    status: DerivationStatus
    engine: DerivationEngine
    # See `LoraDerivationDirectionApi`. Controls how the frontend folds the
    # finished clip back: "target" => result is the target (driver is its
    # reference); "reference" => result is a new reference for `source_clip_id`,
    # which becomes the target.
    direction: DerivationDirection = "target"
    label: str = ""
    driver_path: str
    # Pre-existing still (still entry): used as the content anchor directly
    # when no edit prompt is given. Null => extract a frame from the driver.
    frame_path: str | None = None
    reference_path: str | None = None
    dataset_id: str | None = None
    source_clip_id: str | None = None
    frame_time_seconds: float = 0.0
    edit_prompt: str = ""
    nano_banana_model: str | None = None
    # Engine for the edit stage: "fal" (Nano Banana, remote) or "klein"
    # (local FLUX.2 [klein] 9B). Independent of the motion-drive `engine`.
    edit_engine: FrameEditEngine = "fal"
    scene_prompt: str = ""
    conditioning_type: ConditioningType = "depth"
    conditioning_strength: float = 1.0
    character_orientation: str = "video"
    # Kling O3 ("kling_o3") only: keep the source clip's original audio.
    keep_audio: bool = True
    # Whether the resolved still is a Nano-Banana-edited frame (vs a verbatim
    # source frame). Kling O3 only sends the still as an appearance reference
    # (@Image1) when this is true; otherwise it's a pure video + prompt edit.
    frame_edited: bool = False
    caption: str = ""
    # When true and an edit runs, the job pauses in `review` after the edit so
    # the user can approve/regenerate the still before the motion drive.
    require_review: bool = False
    # Filled as the pipeline advances.
    # The exact still fed to the editor: the extracted source frame (at
    # `frame_time_seconds`) or the pre-existing `frame_path`. Persisted so the
    # review UI can show a true "before" that matches the edited "after".
    source_frame_path: str | None = None
    edited_frame_path: str | None = None
    derived_path: str | None = None
    probe: LoraClipProbeApi | None = None
    error: str | None = None
    cancel_requested: bool = False
    created_at: str
    updated_at: str | None = None


class DerivationJobsState(BaseModel):
    model_config = ConfigDict(strict=True)

    schema_version: int = 1
    jobs: list[DerivationJob] = Field(default_factory=list[DerivationJob])


# Terminal states keep results around for the UI; the frontend prunes its view
# once it has folded a completed clip in.
TERMINAL_STATUSES: frozenset[DerivationStatus] = frozenset(
    {"completed", "failed", "cancelled"}
)


def derivation_job_to_api(job: DerivationJob) -> LoraDerivationJobApi:
    return LoraDerivationJobApi(
        id=job.id,
        status=job.status,
        engine=job.engine,
        direction=job.direction,
        label=job.label,
        driverPath=job.driver_path,
        referencePath=job.reference_path,
        datasetId=job.dataset_id,
        sourceClipId=job.source_clip_id,
        caption=job.caption,
        requireReview=job.require_review,
        editEngine=job.edit_engine,
        sourceFramePath=job.source_frame_path,
        editedFramePath=job.edited_frame_path,
        derivedPath=job.derived_path,
        probe=job.probe,
        error=job.error,
        createdAt=job.created_at,
        updatedAt=job.updated_at,
    )


def derivation_jobs_state_to_api(state: DerivationJobsState) -> LoraDerivationJobsResponse:
    return LoraDerivationJobsResponse(
        jobs=[derivation_job_to_api(j) for j in state.jobs]
    )
