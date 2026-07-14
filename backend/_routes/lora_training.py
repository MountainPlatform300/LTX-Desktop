"""Routes for the LoRA trainer control plane.

Endpoints (all under /api/lora):

    GET    /api/lora/datasets                      — list datasets
    POST   /api/lora/datasets                      — create a draft dataset
    PATCH  /api/lora/datasets/{id}                 — edit a draft dataset
    DELETE /api/lora/datasets/{id}                 — delete + release compute
    POST   /api/lora/datasets/{id}/upload          — upload to remote
    POST   /api/lora/datasets/{id}/cancel          — cancel an in-progress upload
    POST   /api/lora/datasets/{id}/rename          — rename at any status
    POST   /api/lora/datasets/{id}/move            — move into a folder
    POST   /api/lora/folders                       — create folder
    PATCH  /api/lora/folders/{id}                  — rename folder
    DELETE /api/lora/folders/{id}?recursive=bool   — delete folder
    POST   /api/lora/folders/{id}/move             — reparent folder

    GET    /api/lora/preprocessed                  — list preprocessed datasets
    POST   /api/lora/preprocessed                  — start preprocessing
    POST   /api/lora/preprocessed/{id}/cancel      — cancel preprocessing
    DELETE /api/lora/preprocessed/{id}             — delete preprocessed

    GET    /api/lora/training                       — list training jobs
    POST   /api/lora/training                       — start a training run
    GET    /api/lora/training/{id}/logs             — tail remote logs
    POST   /api/lora/training/{id}/cancel           — cancel a training run
    POST   /api/lora/training/{id}/retry-download    — re-fetch a finished run's adapter
    DELETE /api/lora/training/{id}                   — delete a finished run

    POST   /api/lora/test-connection                — validate provider creds

Thin by design: ids/captions are minted at the boundary, snake/camel
conversion happens via the `*_to_api` helpers in
`state/lora_training_state.py`, and typed handler errors
(`LoraEntityNotFoundError` / `LoraTransitionError`) map to 404 / 409.
All remote work is the reconciler's job; these routes only mutate the
durable ledgers and wake it.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pathlib import Path

from _routes._errors import HTTPError
from secret_redaction import redact_lines, redact_text
from api_types import (
    CreateLoraDatasetRequest,
    LoraAnimateFrameRequest,
    LoraApplyEditsRequest,
    LoraApplyEditsResponse,
    LoraCaptionClipRequest,
    LoraCaptionClipResponse,
    LoraClipEditsApi,
    LoraClipInput,
    LoraClipJobsResponse,
    LoraClipProbeApi,
    LoraEnqueueClipJobsRequest,
    CreateLoraDerivationJobRequest,
    CancelAllLoraDerivationsRequest,
    RegenerateLoraDerivationEditRequest,
    LoraDatasetApi,
    LoraDatasetsResponse,
    LoraFolderApi,
    LoraDerivationJobApi,
    LoraDerivationJobsResponse,
    LoraDerivedClipResponse,
    LoraEditFrameRequest,
    LoraEditFrameResponse,
    ExportLoraDatasetRequest,
    ExportLoraDatasetResponse,
    ImportLoraDatasetRequest,
    PublicationExampleApi,
    PublicationMetaApi,
    PublishLoraExportRequest,
    PublishLoraExportResponse,
    PublishLoraPreviewRequest,
    PublishLoraPreviewResponse,
    LoraJobLogsResponse,
    LoraKeepAliveRequest,
    LoraMotionEditRequest,
    PexelsDownloadRequest,
    PexelsDownloadResponse,
    PexelsMediaItemApi,
    PexelsSearchRequest,
    PexelsSearchResponse,
    LoraPreprocessedApi,
    LoraPreprocessedResponse,
    LoraProbeClipRequest,
    LoraProbeClipResponse,
    LoraRestyleClipRequest,
    LoraSceneApi,
    LoraSceneSplitRequest,
    LoraSceneSplitResponse,
    LoraConnectRunpodResponse,
    LoraCostEstimateRequest,
    LoraCostEstimateResponse,
    LoraEstimatePhaseApi,
    LoraGpuOfferApi,
    LoraCreateNetworkVolumeRequest,
    LoraNetworkVolumeApi,
    LoraNetworkVolumeActionResponse,
    LoraRegionHealthApi,
    LoraRelocateNetworkVolumeRequest,
    LoraSelectNetworkVolumeRequest,
    LocalTrainerEligibilityResponse,
    LoraPodInfoApi,
    LoraPodActionResponse,
    LoraTerminatePodResponse,
    LoraTestConnectionRequest,
    LoraTestConnectionResponse,
    LoraTrainingJobApi,
    LoraProviderApi,
    LoraTrainingProfileApi,
    LoraTrainingProfilesResponse,
    LoraTrainingResponse,
    CreateTrainingProfileRequest,
    UpdateTrainingProfileRequest,
    RunPreprocessingRequest,
    LoraTrainingConfigApi,
    StartTrainingPipelineRequest,
    StartTrainingRequest,
    ReselectRunpodRequest,
    RunpodSelectionApi,
    UploadLoraDatasetRequest,
    UpdateLoraDatasetRequest,
    RenameLoraDatasetRequest,
    CreateLoraFolderRequest,
    RenameLoraFolderRequest,
    MoveLoraDatasetRequest,
    MoveLoraFolderRequest,
)
from app_handler import AppHandler
from state.app_settings import AppSettings
from handlers import lora_config_builder
from handlers.lora_cost_estimator import (
    EstimateInputs,
    HistoricalTiming,
    estimate_cost,
)
from handlers import lora_dataset_prep as LoraDatasetPrep
from handlers import lora_export, lora_publish
from handlers.lora_training_handler import (
    LoraEntityNotFoundError,
    LoraTransitionError,
)
from services.clip_processor.clip_processor import (
    ClipProbeResult,
    ClipProcessorError,
    CropSpec,
    EditPlan,
    ScaleSpec,
    TrimSpec,
)
from services.image_editor.image_editor import ImageEditorError, NanoBananaModel
from services.pexels_client.pexels_client import PexelsError, PexelsMediaResult
from services.trainer_target.local_trainer_target import LocalTrainerEligibility
from services.trainer_target.trainer_target import (
    NetworkVolume,
    PodInfo,
    RegionHealth,
    TrainerTargetError,
)
from services.video_restyler.video_restyler import VideoRestylerError
from services.video_captioner.video_captioner import VideoCaptionerError
from state import get_state_service
from state.lora_clip_jobs_state import clip_jobs_state_to_api
from state.lora_derivation_jobs_state import (
    derivation_job_to_api,
    derivation_jobs_state_to_api,
)
from state.lora_training_state import (
    AutoPipelineSpec,
    ClipCrop,
    ClipEdits,
    ClipProbe,
    ClipScale,
    ClipTrim,
    LoraClip,
    LoraDatasetType,
    PendingTraining,
    RunpodSelection,
    SavedModelReadiness,
    TrainingConfig,
    dataset_to_api,
    folder_to_api,
    datasets_state_to_api,
    preprocessed_state_to_api,
    preprocessed_to_api,
    profile_to_api,
    profiles_state_to_api,
    training_config_from_api,
    training_job_to_api,
    training_state_to_api,
    runpod_selection_from_api,
)

router = APIRouter(prefix="/api/lora", tags=["lora-trainer"])


def _not_found(exc: LoraEntityNotFoundError) -> HTTPError:
    return HTTPError(404, str(exc), code="LORA_ENTITY_NOT_FOUND")


def _conflict(exc: LoraTransitionError) -> HTTPError:
    return HTTPError(409, str(exc), code="LORA_INVALID_TRANSITION")


def _probe_from_input(probe: LoraClipProbeApi | None) -> ClipProbe | None:
    if probe is None:
        return None
    return ClipProbe(
        duration_seconds=probe.durationSeconds,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        frame_count=probe.frameCount,
        has_audio=probe.hasAudio,
        video_codec=probe.videoCodec,
    )


def _probe_to_api(probe: ClipProbeResult) -> LoraClipProbeApi:
    return LoraClipProbeApi(
        durationSeconds=probe.duration_seconds,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        frameCount=probe.frame_count,
        hasAudio=probe.has_audio,
        videoCodec=probe.video_codec,
    )


def _edits_from_input(edits: LoraClipEditsApi | None) -> ClipEdits | None:
    if edits is None:
        return None
    return ClipEdits(
        trim=None if edits.trim is None else ClipTrim(
            start_seconds=edits.trim.startSeconds,
            end_seconds=edits.trim.endSeconds,
        ),
        crop=None if edits.crop is None else ClipCrop(
            x=edits.crop.x, y=edits.crop.y, width=edits.crop.width, height=edits.crop.height
        ),
        scale=None if edits.scale is None else ClipScale(
            width=edits.scale.width, height=edits.scale.height
        ),
        fps=edits.fps,
        speed=edits.speed,
        mute=edits.mute,
        reverse=edits.reverse,
    )


def _edit_plan(edits: LoraClipEditsApi) -> EditPlan:
    return EditPlan(
        trim=None if edits.trim is None else TrimSpec(
            start_seconds=edits.trim.startSeconds, end_seconds=edits.trim.endSeconds
        ),
        crop=None if edits.crop is None else CropSpec(
            x=edits.crop.x, y=edits.crop.y, width=edits.crop.width, height=edits.crop.height
        ),
        scale=None if edits.scale is None else ScaleSpec(
            width=edits.scale.width, height=edits.scale.height
        ),
        fps=edits.fps,
        speed=edits.speed,
        mute=edits.mute,
        reverse=edits.reverse,
    )


def _clips_from_input(
    clips: list[LoraClipInput], *, preserve_ids: bool = False
) -> list[LoraClip]:
    return [
        LoraClip(
            id=clip.id if preserve_ids and clip.id is not None else uuid.uuid4().hex,
            local_path=clip.localPath,
            caption=clip.caption,
            duration_seconds=clip.durationSeconds,
            reference_path=clip.referencePath,
            reference_paths=clip.referencePaths,
            origin=clip.origin,
            probe=_probe_from_input(clip.probe),
            source_path=clip.sourcePath,
            edits=_edits_from_input(clip.edits),
            poster_path=clip.posterPath,
            sprite_path=clip.spritePath,
            sprite_tiles=clip.spriteTiles,
            triage=clip.triage,
            deleted_at=clip.deletedAt,
        )
        for clip in clips
    ]


# ----------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------


@router.get("/datasets", response_model=LoraDatasetsResponse)
def route_list_datasets(
    includeArchived: bool = Query(default=False),
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetsResponse:
    state = handler.lora_training.get_datasets_state()
    if not includeArchived:
        state.datasets = [d for d in state.datasets if d.archived_at is None]
    return datasets_state_to_api(state)


@router.post("/datasets", response_model=LoraDatasetApi)
def route_create_dataset(
    req: CreateLoraDatasetRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    dataset = handler.lora_training.create_dataset(
        name=req.name,
        dataset_type=req.type,
        trigger_word=req.triggerWord,
        clips=_clips_from_input(req.clips),
        originating_project_id=req.originatingProjectId,
    )
    return dataset_to_api(dataset)


@router.patch("/datasets/{dataset_id}", response_model=LoraDatasetApi)
def route_update_dataset(
    dataset_id: str,
    req: UpdateLoraDatasetRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    clips = (
        _clips_from_input(req.clips, preserve_ids=True)
        if req.clips is not None
        else None
    )
    try:
        dataset = handler.lora_training.update_dataset(
            dataset_id,
            name=req.name,
            dataset_type=req.type,
            trigger_word=req.triggerWord,
            clips=clips,
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return dataset_to_api(dataset)


@router.post("/datasets/{dataset_id}/rename", response_model=LoraDatasetApi)
def route_rename_dataset(
    dataset_id: str,
    req: RenameLoraDatasetRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    """Rename a dataset at any status (display-only; doesn't touch the remote)."""
    try:
        dataset = handler.lora_training.rename_dataset(dataset_id, req.name)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return dataset_to_api(dataset)


@router.post("/folders", response_model=LoraFolderApi)
def route_create_folder(
    req: CreateLoraFolderRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraFolderApi:
    """Create a collection folder, optionally nested under `parentId`."""
    try:
        folder = handler.lora_training.create_folder(req.name, req.parentId)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return folder_to_api(folder)


@router.patch("/folders/{folder_id}", response_model=LoraFolderApi)
def route_rename_folder(
    folder_id: str,
    req: RenameLoraFolderRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraFolderApi:
    try:
        folder = handler.lora_training.rename_folder(folder_id, req.name)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return folder_to_api(folder)


@router.delete("/folders/{folder_id}", status_code=204)
def route_delete_folder(
    folder_id: str,
    recursive: bool = False,
    handler: AppHandler = Depends(get_state_service),
) -> None:
    """Delete a folder. Non-recursive moves contents up to the deleted folder's
    parent; recursive deletes subfolders + contained datasets (compute-release)."""
    try:
        handler.lora_training.delete_folder(folder_id, recursive=recursive)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None


@router.post("/folders/{folder_id}/move", response_model=LoraFolderApi)
def route_move_folder(
    folder_id: str,
    req: MoveLoraFolderRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraFolderApi:
    """Reparent a folder (rejects cycles). `parentId` null = root."""
    try:
        folder = handler.lora_training.move_folder(folder_id, req.parentId)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return folder_to_api(folder)


@router.post("/datasets/{dataset_id}/move", response_model=LoraDatasetApi)
def route_move_dataset(
    dataset_id: str,
    req: MoveLoraDatasetRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    """Move a dataset into a folder. `folderId` null = root."""
    try:
        dataset = handler.lora_training.move_dataset(dataset_id, req.folderId)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return dataset_to_api(dataset)


@router.post("/datasets/import", response_model=LoraDatasetApi)
def route_import_dataset(
    req: ImportLoraDatasetRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    try:
        dataset = handler.lora_training.import_dataset(source_path=req.sourcePath)
    except LoraTransitionError as exc:
        raise HTTPError(400, str(exc), code="LORA_IMPORT_FAILED") from None
    return dataset_to_api(dataset)


@router.post("/datasets/{dataset_id}/export", response_model=ExportLoraDatasetResponse)
def route_export_dataset(
    dataset_id: str,
    req: ExportLoraDatasetRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ExportLoraDatasetResponse:
    options = LoraDatasetPrep.PrepOptions(
        fps=req.icLoraFps,
        short_side=req.icLoraShortSide,
        bucket_frames=req.icLoraBucketFrames,
        max_duration_seconds=req.icLoraMaxDurationSeconds,
        forbidden_words=tuple(w for w in req.forbiddenCaptionWords if w.strip()),
    )
    components = lora_export.BundleComponents(
        train_config=req.includeConfig,
        readme=req.includeReadme,
        manifest=req.includeManifest,
        model_card=req.includeModelCard,
    )
    try:
        export_path, clip_count, dropped = handler.lora_training.export_dataset(
            dataset_id,
            dest_path=req.destPath,
            export_format=req.format,
            include_rejected=req.includeRejected,
            profile_id=req.profileId,
            prep_options=options,
            components=components,
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise HTTPError(400, str(exc), code="LORA_EXPORT_FAILED") from None
    return ExportLoraDatasetResponse(
        exportPath=export_path, clipCount=clip_count, droppedPairs=dropped
    )


def _to_publication_meta(api: PublicationMetaApi) -> lora_publish.PublicationMeta:
    return lora_publish.PublicationMeta(
        title=api.title,
        summary=api.summary,
        description=api.description,
        author=api.author,
        license=api.license,
        tags=list(api.tags),
        base_model=api.baseModel,
    )


def _from_publication_meta(meta: lora_publish.PublicationMeta) -> PublicationMetaApi:
    return PublicationMetaApi(
        title=meta.title,
        summary=meta.summary,
        description=meta.description,
        author=meta.author,
        license=meta.license,
        tags=list(meta.tags),
        baseModel=meta.base_model,
    )


def _to_publication_examples(
    items: list[PublicationExampleApi],
) -> list[lora_publish.PublicationExample]:
    return [
        lora_publish.PublicationExample(media_path=e.mediaPath, caption=e.caption)
        for e in items
    ]


@router.post(
    "/training/{training_id}/publish/preview",
    response_model=PublishLoraPreviewResponse,
)
def route_publish_preview(
    training_id: str,
    req: PublishLoraPreviewRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PublishLoraPreviewResponse:
    try:
        meta, cards = handler.lora_training.publish_preview(
            training_id,
            platforms=list(req.platforms),
            meta=_to_publication_meta(req.meta) if req.meta is not None else None,
            examples=_to_publication_examples(req.examples),
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise HTTPError(400, str(exc), code="LORA_PUBLISH_FAILED") from None
    return PublishLoraPreviewResponse(meta=_from_publication_meta(meta), cards=cards)


@router.post(
    "/training/{training_id}/publish/export",
    response_model=PublishLoraExportResponse,
)
def route_publish_export(
    training_id: str,
    req: PublishLoraExportRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PublishLoraExportResponse:
    try:
        publication_path, manifest = handler.lora_training.publish_export(
            training_id,
            dest_path=req.destPath,
            platforms=list(req.platforms),
            meta=_to_publication_meta(req.meta),
            examples=_to_publication_examples(req.examples),
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise HTTPError(400, str(exc), code="LORA_PUBLISH_FAILED") from None
    return PublishLoraExportResponse(
        publicationPath=publication_path,
        exampleCount=int(manifest.get("exampleCount", 0)),
        files=list(manifest.get("files", [])),
        weightsFile=manifest.get("weightsFile"),
    )


@router.delete("/datasets/{dataset_id}", status_code=204)
def route_delete_dataset(
    dataset_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> None:
    dataset = handler.lora_training.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPError(404, f"Dataset not found: {dataset_id}", code="LORA_ENTITY_NOT_FOUND")
    try:
        handler.lora_training.delete_dataset(dataset_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    # Only release remote compute after every deletion precondition passes.
    # The captured snapshot retains the target handle after the ledger row is
    # removed. Release remains best-effort, matching the existing API contract.
    handler.lora_training_runner.release_workspace_for_dataset(dataset)


@router.post("/datasets/{dataset_id}/archive", response_model=LoraDatasetApi)
def route_archive_dataset(
    dataset_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    try:
        return dataset_to_api(handler.lora_training.archive_dataset(dataset_id))
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None


@router.post("/datasets/{dataset_id}/unarchive", response_model=LoraDatasetApi)
def route_unarchive_dataset(
    dataset_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    try:
        return dataset_to_api(handler.lora_training.unarchive_dataset(dataset_id))
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None


@router.post("/caption-clip", response_model=LoraCaptionClipResponse)
def route_caption_clip(
    req: LoraCaptionClipRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraCaptionClipResponse:
    try:
        caption = handler.lora_training.caption_clip(
            video_path=req.videoPath,
            with_audio=req.withAudio,
        )
    except VideoCaptionerError as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_CAPTION_ERROR") from None
    return LoraCaptionClipResponse(caption=caption)


@router.post("/probe-clip", response_model=LoraProbeClipResponse)
def route_probe_clip(
    req: LoraProbeClipRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraProbeClipResponse:
    try:
        probe = handler.lora_training.probe_clip(video_path=req.videoPath)
    except ClipProcessorError as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_PROBE_ERROR") from None
    return LoraProbeClipResponse(probe=_probe_to_api(probe))


@router.post("/apply-edits", response_model=LoraApplyEditsResponse)
def route_apply_edits(
    req: LoraApplyEditsRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraApplyEditsResponse:
    plan = _edit_plan(req.edits)
    if plan.is_empty:
        raise HTTPError(400, "No edits to apply", code="LORA_EDIT_ERROR")
    try:
        derived_path, probe = handler.lora_training.apply_clip_edits(
            source_path=req.sourcePath, plan=plan
        )
    except ClipProcessorError as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_EDIT_ERROR") from None
    return LoraApplyEditsResponse(derivedPath=derived_path, probe=_probe_to_api(probe))


@router.post("/scene-split", response_model=LoraSceneSplitResponse)
def route_scene_split(
    req: LoraSceneSplitRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraSceneSplitResponse:
    try:
        scenes = handler.lora_training.split_scenes(
            source_path=req.sourcePath, threshold=req.threshold
        )
    except ClipProcessorError as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_SCENE_ERROR") from None
    return LoraSceneSplitResponse(
        scenes=[
            LoraSceneApi(
                localPath=path,
                startSeconds=span.start_seconds,
                endSeconds=span.end_seconds,
                probe=_probe_to_api(probe),
            )
            for path, span, probe in scenes
        ]
    )


@router.post("/edit-frame", response_model=LoraEditFrameResponse)
def route_edit_frame(
    req: LoraEditFrameRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraEditFrameResponse:
    model: NanoBananaModel | None = req.model
    try:
        frame_path = handler.lora_training.edit_frame(
            source_path=req.sourcePath,
            time_seconds=req.timeSeconds,
            prompt=req.prompt,
            model=model,
            engine=req.engine,
        )
    except ClipProcessorError as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_FRAME_ERROR") from None
    except ImageEditorError as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_FRAME_ERROR") from None
    return LoraEditFrameResponse(framePath=frame_path)


@router.post("/animate-frame", response_model=LoraDerivedClipResponse)
def route_animate_frame(
    req: LoraAnimateFrameRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivedClipResponse:
    try:
        derived_path, probe = handler.lora_training.animate_image(
            image_path=req.imagePath, prompt=req.prompt
        )
    except (ClipProcessorError, VideoRestylerError) as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_ANIMATE_ERROR") from None
    return LoraDerivedClipResponse(derivedPath=derived_path, probe=_probe_to_api(probe))


@router.post("/restyle-clip", response_model=LoraDerivedClipResponse)
def route_restyle_clip(
    req: LoraRestyleClipRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivedClipResponse:
    try:
        derived_path, probe = handler.lora_training.restyle_clip(
            source_path=req.sourcePath, prompt=req.prompt
        )
    except (ClipProcessorError, VideoRestylerError) as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_RESTYLE_ERROR") from None
    return LoraDerivedClipResponse(derivedPath=derived_path, probe=_probe_to_api(probe))


@router.post("/motion-edit", response_model=LoraDerivedClipResponse)
def route_motion_edit(
    req: LoraMotionEditRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivedClipResponse:
    try:
        derived_path, probe = handler.lora_training.motion_edit_clip(
            source_path=req.sourcePath,
            reference_image_path=req.referenceImagePath,
            prompt=req.prompt,
            engine=req.engine,
            video_strength=req.videoStrength,
            character_orientation=req.characterOrientation,
            keep_audio=req.keepAudio,
        )
    except (ClipProcessorError, VideoRestylerError) as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_MOTION_EDIT_ERROR") from None
    return LoraDerivedClipResponse(derivedPath=derived_path, probe=_probe_to_api(probe))


@router.post("/pexels/search", response_model=PexelsSearchResponse)
def route_pexels_search(
    req: PexelsSearchRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PexelsSearchResponse:
    """Search Pexels stock media (BYOK) for the LoRA collection browser."""
    try:
        result = handler.lora_training.search_pexels(
            query=req.query,
            media=req.media,
            page=req.page,
            per_page=req.perPage,
            orientation=req.orientation,
        )
    except PexelsError as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_PEXELS_ERROR") from None
    return PexelsSearchResponse(
        items=[_pexels_item_to_api(it) for it in result.items],
        page=result.page,
        perPage=result.per_page,
        totalResults=result.total_results,
        hasNext=result.has_next,
    )


@router.post("/pexels/download", response_model=PexelsDownloadResponse)
def route_pexels_download(
    req: PexelsDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PexelsDownloadResponse:
    """Download a chosen Pexels asset into app storage for use as a clip."""
    try:
        local_path, probe = handler.lora_training.download_pexels_asset(
            url=req.url, kind=req.kind, ext=req.ext
        )
    except (PexelsError, ClipProcessorError) as exc:
        raise HTTPError(exc.status_code, exc.detail, code="LORA_PEXELS_ERROR") from None
    return PexelsDownloadResponse(
        localPath=local_path,
        probe=_probe_to_api(probe) if probe is not None else None,
    )


def _pexels_item_to_api(item: PexelsMediaResult) -> PexelsMediaItemApi:
    return PexelsMediaItemApi(
        id=item.id,
        kind=item.kind,
        width=item.width,
        height=item.height,
        durationSeconds=item.duration_seconds,
        previewUrl=item.preview_url,
        downloadUrl=item.download_url,
        downloadExt=item.download_ext,
        pexelsUrl=item.pexels_url,
        author=item.author,
        authorUrl=item.author_url,
        alt=item.alt,
    )


@router.post("/datasets/{dataset_id}/upload", response_model=LoraDatasetApi)
def route_upload_dataset(
    dataset_id: str,
    req: UploadLoraDatasetRequest | None = None,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    try:
        dataset = handler.lora_training.request_upload(
            dataset_id,
            provider=req.provider if req is not None else None,
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return dataset_to_api(dataset)


@router.post("/datasets/{dataset_id}/cancel", response_model=LoraDatasetApi)
def route_cancel_upload(
    dataset_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    """Cancel an in-progress upload: releases the provisioned GPU pod and stops
    the run before preprocessing. Only valid while `status == uploading`."""
    try:
        dataset = handler.lora_training.request_cancel_upload(dataset_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return dataset_to_api(dataset)


# ----------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------


@router.get("/preprocessed", response_model=LoraPreprocessedResponse)
def route_list_preprocessed(
    handler: AppHandler = Depends(get_state_service),
) -> LoraPreprocessedResponse:
    return preprocessed_state_to_api(handler.lora_training.get_preprocessed_state())


@router.post("/preprocessed", response_model=LoraPreprocessedApi)
def route_start_preprocessing(
    req: RunPreprocessingRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraPreprocessedApi:
    settings = handler.settings.get_settings_snapshot()
    # The dataset is already uploaded, so its target records the provider the
    # clips live on; pick the preset the same way the one-click pipeline does so
    # manual preprocess matches training's text-encoder precision (low_vram ->
    # 8-bit on a sub-80 GB GPU). Unlike the training/pipeline route, this only
    # DOWNGRADES standard -> low_vram — manual preprocess has no training intent,
    # so it doesn't enforce the training min-VRAM floor (a 24 GB pod can still
    # cache latents with the 8-bit text encoder).
    dataset = handler.lora_training.get_dataset(req.datasetId)
    if dataset is not None and dataset.target is not None:
        provider: LoraProviderApi = dataset.target.provider
    else:
        provider = settings.lora_provider
    if provider == "local":
        eligibility = handler.lora_training.local_trainer_eligibility()
        vram = eligibility.vram_gb if eligibility.eligible and eligibility.vram_gb else 0
    else:
        vram = settings.runpod_gpu_vram_gb
    recommended = lora_config_builder.recommended_preset_for_vram(vram)
    if provider == "runpod" and vram <= 0:
        recommended = "low_vram"
    preset: str = "low_vram" if recommended == "low_vram" else "standard"
    try:
        item = handler.lora_training.create_preprocessing(
            dataset_id=req.datasetId,
            resolution_buckets=req.resolutionBuckets,
            with_audio=req.withAudio if dataset is None or dataset.type != "ic_lora" else False,
            auto_caption=req.autoCaption,
            captioner_type=req.captionerType,
            preset=preset,
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return preprocessed_to_api(item)


@router.post("/preprocessed/{preprocessed_id}/cancel", response_model=LoraPreprocessedApi)
def route_cancel_preprocessing(
    preprocessed_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraPreprocessedApi:
    try:
        item = handler.lora_training.request_cancel_preprocessing(preprocessed_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return preprocessed_to_api(item)


@router.post("/preprocessed/{preprocessed_id}/resume", response_model=LoraPreprocessedApi)
def route_resume_preprocessing(
    preprocessed_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraPreprocessedApi:
    """Re-run a failed/cancelled preprocess, reusing the uploaded workspace +
    cached captions (skips re-captioning, re-runs the latent-caching step)."""
    try:
        item = handler.lora_training.request_preprocess_resume(preprocessed_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return preprocessed_to_api(item)


@router.post("/preprocessed/{preprocessed_id}/reset", response_model=LoraPreprocessedApi)
def route_reset_preprocessing(
    preprocessed_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraPreprocessedApi:
    """Clear a finished preprocess's cached latents + captions and re-run from
    scratch (a fresh start, vs. resume which reuses cached state)."""
    try:
        item = handler.lora_training.request_preprocess_reset(preprocessed_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return preprocessed_to_api(item)


@router.delete("/preprocessed/{preprocessed_id}", status_code=204)
def route_delete_preprocessed(
    preprocessed_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> None:
    try:
        handler.lora_training.delete_preprocessed(preprocessed_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------


@router.get("/training", response_model=LoraTrainingResponse)
def route_list_training(
    includeArchived: bool = Query(default=False),
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingResponse:
    state = handler.lora_training.get_training_state()
    if not includeArchived:
        state.items = [job for job in state.items if job.archived_at is None]
    return training_state_to_api(state)


def _resolve_run_config(
    handler: AppHandler,
    *,
    profile_id: str | None,
    config_api: LoraTrainingConfigApi | None,
    trigger_override: str | None,
    dataset_type: LoraDatasetType,
    vram_gb: int,
) -> tuple[TrainingConfig, bool]:
    """Resolve a run's config: a snapshotted profile, an inline config, or
    the dataset-specific Auto recommendation.

    Returns ``(config, hardware_adaptive)``. Curated built-ins are safe to
    materialize for the selected GPU; custom/inline configs stay exact and use
    the expert-override path.
    """
    if profile_id is not None:
        profile = handler.lora_training.get_profile(profile_id)
        if dataset_type not in profile.dataset_types:
            raise HTTPError(
                422,
                f'"{profile.name}" is not compatible with this dataset type.',
                code="LORA_PROFILE_INCOMPATIBLE",
            )
        if profile.min_vram_gb is not None and vram_gb < profile.min_vram_gb:
            raise HTTPError(
                422,
                f'"{profile.name}" requires at least {profile.min_vram_gb} GB of VRAM.',
                code="LORA_PROFILE_GPU_INCOMPATIBLE",
            )
        config = profile.config.model_copy(deep=True)
        hardware_adaptive = False
    elif config_api is not None:
        config = training_config_from_api(config_api)
        hardware_adaptive = False
    else:
        profiles = handler.lora_training.get_profiles_state().profiles
        profile = next(
            (
                candidate
                for candidate in profiles
                if candidate.builtin
                and candidate.auto_recommended
                and dataset_type in candidate.dataset_types
                and (
                    candidate.min_vram_gb is None
                    or vram_gb >= candidate.min_vram_gb
                )
            ),
            None,
        )
        config = (
            profile.config.model_copy(deep=True)
            if profile is not None
            else TrainingConfig()
        )
        hardware_adaptive = True
    if trigger_override is not None:
        config = config.model_copy(update={"trigger_word": trigger_override})
    return config, hardware_adaptive


def _apply_validation_prompts_override(
    config: TrainingConfig, prompts: list[str] | None
) -> TrainingConfig:
    """Apply an optional `validationPrompts` override from the start request.

    The training modal sends the user's edited/approved prompts here without
    forking the whole config. An empty list is honored (no prompt-only
    validation samples); ``None`` means "no override — keep the resolved
    config's prompts", which the runner auto-seeds from captions when default.
    """
    if prompts is None:
        return config
    return config.model_copy(update={"validation_prompts": list(prompts)})


def _apply_gpu_preset(
    config: TrainingConfig,
    settings: AppSettings,
    *,
    provider: LoraProviderApi,
    local_vram_gb: int | None,
    runpod_vram_gb: int | None = None,
    explicit_config: bool = False,
    allow_unsafe_override: bool = False,
) -> TrainingConfig:
    """Materialize a safe config for the run's GPU.

    Sub-80 GB cards use the quantized low-VRAM path and cap the adapter at the
    official low-VRAM profile's rank 16. Unknown RunPod VRAM is treated
    conservatively instead of silently retaining the full-precision rank-32
    defaults; selecting a known 80 GB+ GPU restores the standard profile.

    VRAM source is provider-specific — the selected RunPod GPU, or the probed
    local GPU. For RunPod we also reject a card too small to train at all (422).
    Local isn't re-rejected here: eligibility already enforces the minimum-VRAM
    floor before we get here, and the RTX 5090 reports ~31 GB (just under the
    32 GB RunPod cutoff) yet trains fine on the quantized path.
    """
    if provider == "local":
        vram = local_vram_gb or 0
    else:
        vram = (
            runpod_vram_gb
            if runpod_vram_gb is not None
            else settings.runpod_gpu_vram_gb
        )
        if 0 < vram < lora_config_builder.MIN_TRAINING_VRAM_GB:
            raise HTTPError(
                422,
                f"The selected GPU has {vram} GB of VRAM, but LoRA training needs "
                f"at least {lora_config_builder.MIN_TRAINING_VRAM_GB} GB. Pick a "
                "larger GPU in the training dialog.",
                code="LORA_GPU_TOO_SMALL",
            )
    recommended = lora_config_builder.recommended_preset_for_vram(vram)
    if provider == "runpod" and vram <= 0:
        recommended = "low_vram"
    if recommended == "low_vram":
        safe_low_vram = (
            config.preset == "low_vram"
            and config.rank <= 16
            and config.batch_size == 1
            and config.quantization in (None, "int8-quanto", "int4-quanto")
            and config.optimizer_type in (None, "adamw8bit")
            and config.load_text_encoder_in_8bit is not False
            and config.offload_optimizer_during_validation is not False
            and config.skip_initial_validation is not False
        )
        if explicit_config and not safe_low_vram:
            if not allow_unsafe_override:
                raise HTTPError(
                    422,
                    "This training profile exceeds the conservative settings for "
                    "the selected GPU and may run out of memory. Return to the "
                    "training dialog and enable the expert override to continue.",
                    code="LORA_UNSAFE_TRAINING_OVERRIDE",
                )
            return config
        rank = min(config.rank, 16)
        config = config.model_copy(
            update={
                "preset": "low_vram",
                "rank": rank,
                "alpha": min(config.alpha, rank),
                "batch_size": 1,
                "enable_gradient_checkpointing": True,
                "optimizer_type": "adamw8bit",
                "quantization": (
                    config.quantization
                    if config.quantization in ("int8-quanto", "int4-quanto")
                    else "int8-quanto"
                ),
                "load_text_encoder_in_8bit": True,
                "offload_optimizer_during_validation": True,
                "skip_initial_validation": True,
            }
        )
    return config


def _run_gpu_info(
    settings: AppSettings, eligibility: LocalTrainerEligibility | None
) -> tuple[str, int]:
    """GPU label + VRAM to stamp on the run summary, provider-aware: the probed
    local GPU when training locally, else the selected RunPod GPU."""
    if eligibility is not None:
        return (eligibility.gpu_name or "Local GPU", eligibility.vram_gb or 0)
    return (settings.runpod_gpu_type, settings.runpod_gpu_vram_gb)


def _selection_or_legacy(
    request_selection: RunpodSelectionApi | None, settings: AppSettings
) -> RunpodSelection | None:
    if request_selection is not None:
        return runpod_selection_from_api(request_selection)
    if not settings.runpod_gpu_type:
        return None
    has_cache = (
        settings.runpod_keep_model_cached
        and bool(settings.runpod_network_volume_id)
    )
    return RunpodSelection(
        gpu_type=settings.runpod_gpu_type,
        gpu_vram_gb=settings.runpod_gpu_vram_gb,
        workspace_policy="primary_cache" if has_cache else "ephemeral_any_region",
        volume_id=settings.runpod_network_volume_id if has_cache else None,
    )


def _buckets_fit_conservative_profile(value: str) -> bool:
    """Whether every bucket stays within the 32–60 GB profile envelope."""
    try:
        buckets = [part.strip() for part in value.split(";") if part.strip()]
        parsed = [tuple(int(v) for v in part.split("x")) for part in buckets]
    except ValueError:
        return False
    return bool(parsed) and all(
        len(bucket) == 3
        and bucket[0] * bucket[1] <= 512 * 512
        and bucket[2] <= 49
        for bucket in parsed
    )


def _require_provider_available(
    handler: AppHandler, provider: LoraProviderApi
) -> LocalTrainerEligibility | None:
    """Reject an impossible local run up front with a clear, actionable error.

    RunPod is always available (its creds are validated elsewhere at run time);
    "local" depends on the machine (WSL2 + CUDA + a big-enough GPU), so probe
    eligibility and 422 with the human-readable reason rather than letting the
    reconciler fail the run asynchronously. Returns the local eligibility (so the
    caller can read the probed GPU/VRAM) or None for RunPod.
    """
    if provider != "local":
        return None
    eligibility = handler.lora_training.local_trainer_eligibility()
    if not eligibility.eligible:
        raise HTTPError(422, eligibility.reason, code="LOCAL_TRAINER_UNAVAILABLE")
    return eligibility


@router.post("/training", response_model=LoraTrainingJobApi)
def route_start_training(
    req: StartTrainingRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingJobApi:
    settings = handler.settings.get_settings_snapshot()
    eligibility = _require_provider_available(handler, req.provider)
    selection = (
        _selection_or_legacy(req.runpodSelection, settings)
        if req.provider == "runpod"
        else None
    )
    if (
        req.provider == "runpod"
        and 0
        < (
            (selection.gpu_vram_gb or 0)
            if selection is not None
            else settings.runpod_gpu_vram_gb
        )
        < lora_config_builder.MIN_TRAINING_VRAM_GB
    ):
        rejected_vram = (
            selection.gpu_vram_gb
            if selection is not None
            else settings.runpod_gpu_vram_gb
        )
        raise HTTPError(
            422,
            f"The selected GPU has {rejected_vram} GB of VRAM, but LoRA training needs "
            f"at least {lora_config_builder.MIN_TRAINING_VRAM_GB} GB.",
            code="LORA_GPU_TOO_SMALL",
        )
    preprocessed = handler.lora_training.get_preprocessed(req.preprocessedId)
    if preprocessed is None:
        raise HTTPError(
            404,
            f"Preprocessed dataset {req.preprocessedId!r} not found",
            code="LORA_ENTITY_NOT_FOUND",
        )
    dataset = handler.lora_training.get_dataset(preprocessed.dataset_id)
    if dataset is None:
        raise HTTPError(
            404,
            f"Dataset {preprocessed.dataset_id!r} not found",
            code="LORA_ENTITY_NOT_FOUND",
        )
    selected_vram = (
        selection.gpu_vram_gb or 0
        if selection is not None
        else ((eligibility.vram_gb or 0) if eligibility is not None else 0)
    )
    try:
        config, hardware_adaptive = _resolve_run_config(
            handler,
            profile_id=req.profileId,
            config_api=req.config,
            trigger_override=req.triggerWordOverride,
            dataset_type=dataset.type,
            vram_gb=selected_vram,
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    config = _apply_gpu_preset(
        config,
        settings,
        provider=req.provider,
        local_vram_gb=eligibility.vram_gb if eligibility else None,
        runpod_vram_gb=selection.gpu_vram_gb if selection else None,
        explicit_config=not hardware_adaptive,
        allow_unsafe_override=req.allowUnsafeOverride,
    )
    config = _apply_validation_prompts_override(config, req.validationPrompts)
    gpu_type, gpu_vram_gb = (
        (selection.gpu_type, selection.gpu_vram_gb)
        if selection is not None
        else _run_gpu_info(settings, eligibility)
    )
    if (
        config.preset == "low_vram"
        and not _buckets_fit_conservative_profile(
            preprocessed.effective_resolution_buckets
            or preprocessed.resolution_buckets
        )
        and not req.allowUnsafeOverride
    ):
        raise HTTPError(
            422,
            "This dataset was prepared at a resolution above the conservative "
            "GPU profile. Preprocess it at 512x512x49, or enable the expert "
            "override in the training dialog.",
            code="LORA_UNSAFE_TRAINING_OVERRIDE",
        )
    try:
        job = handler.lora_training.start_training(
            preprocessed_id=req.preprocessedId,
            name=req.name,
            description=(req.description or "").strip() or None,
            config=config,
            provider=req.provider,
            # Snapshot the selected GPU so the run summary records what trained.
            gpu_type=gpu_type,
            gpu_vram_gb=gpu_vram_gb,
            runpod_selection=selection,
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return training_job_to_api(job)


@router.post("/training-pipeline", response_model=LoraDatasetApi)
def route_start_training_pipeline(
    req: StartTrainingPipelineRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    """One-click: upload → preprocess → train. Stores the intent on the dataset;
    the reconciler auto-advances through every stage."""
    settings = handler.settings.get_settings_snapshot()
    eligibility = _require_provider_available(handler, req.provider)
    selection = (
        _selection_or_legacy(req.runpodSelection, settings)
        if req.provider == "runpod"
        else None
    )
    dataset = handler.lora_training.get_dataset(req.datasetId)
    if dataset is None:
        raise HTTPError(404, f"Dataset {req.datasetId!r} not found", code="LORA_ENTITY_NOT_FOUND")
    selected_vram = (
        selection.gpu_vram_gb or 0
        if selection is not None
        else ((eligibility.vram_gb or 0) if eligibility is not None else 0)
    )
    try:
        config, hardware_adaptive = _resolve_run_config(
            handler,
            profile_id=req.profileId,
            config_api=req.config,
            trigger_override=req.triggerWordOverride,
            dataset_type=dataset.type,
            vram_gb=selected_vram,
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    config = _apply_gpu_preset(
        config,
        settings,
        provider=req.provider,
        local_vram_gb=eligibility.vram_gb if eligibility else None,
        runpod_vram_gb=selection.gpu_vram_gb if selection else None,
        explicit_config=not hardware_adaptive,
        allow_unsafe_override=req.allowUnsafeOverride,
    )
    config = _apply_validation_prompts_override(config, req.validationPrompts)
    gpu_type, gpu_vram_gb = (
        (selection.gpu_type, selection.gpu_vram_gb)
        if selection is not None
        else _run_gpu_info(settings, eligibility)
    )
    # IC-LoRA is video-only. Silently carrying the generic audio toggle into
    # preprocessing creates audio latents that the IC training strategy then
    # discards (or fails count validation on), so normalize it at the boundary.
    with_audio = req.withAudio if dataset.type != "ic_lora" else False
    auto_profile = req.profileId is None and req.config is None
    resolution_buckets = req.resolutionBuckets
    if (
        auto_profile
        and config.preset == "low_vram"
        and resolution_buckets == "768x448x49"
    ):
        resolution_buckets = "512x512x49"
    if (
        config.preset == "low_vram"
        and not _buckets_fit_conservative_profile(resolution_buckets)
        and not req.allowUnsafeOverride
    ):
        raise HTTPError(
            422,
            "This resolution exceeds the conservative profile for the selected "
            "GPU. Use 512x512x49, or enable the expert override.",
            code="LORA_UNSAFE_TRAINING_OVERRIDE",
        )
    spec = AutoPipelineSpec(
        resolution_buckets=resolution_buckets,
        with_audio=with_audio,
        auto_caption=req.autoCaption,
        captioner_type="gemini_flash" if req.captionerType == "gemini_flash" else "qwen_omni",
        training=PendingTraining(
            config=config,
            name=req.name,
            description=(req.description or "").strip() or None,
            gpu_type=gpu_type,
            gpu_vram_gb=gpu_vram_gb,
            runpod_selection=selection,
        ),
        runpod_selection=selection,
    )
    workspace_policy = (
        selection.workspace_policy if selection is not None else req.workspacePolicy
    )
    if workspace_policy is None:
        workspace_policy = (
            "primary_cache"
            if (
                req.provider == "runpod"
                and settings.runpod_keep_model_cached
                and settings.runpod_network_volume_id
            )
            else "ephemeral_any_region"
        )
    cache_volume_id = (
        selection.volume_id
        if selection is not None and workspace_policy == "primary_cache"
        else (
            settings.runpod_network_volume_id
            if req.provider == "runpod" and workspace_policy == "primary_cache"
            else None
        )
    )
    if req.provider == "runpod" and workspace_policy == "primary_cache" and not cache_volume_id:
        raise HTTPError(
            409,
            "No primary RunPod cache is selected. Connect RunPod and create or "
            "select a cache volume, or use the ephemeral-any-region policy.",
            code="LORA_CACHE_NOT_CONFIGURED",
        )
    try:
        dataset = handler.lora_training.start_training_pipeline(
            dataset_id=req.datasetId,
            spec=spec,
            provider=req.provider,
            workspace_policy=workspace_policy,
            cache_volume_id=cache_volume_id,
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return dataset_to_api(dataset)


@router.post(
    "/datasets/{dataset_id}/reselect-runpod", response_model=LoraDatasetApi
)
def route_reselect_dataset_runpod(
    dataset_id: str,
    req: ReselectRunpodRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    try:
        dataset = handler.lora_training.reselect_dataset(
            dataset_id, runpod_selection_from_api(req.selection)
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return dataset_to_api(dataset)


@router.post(
    "/training/{training_id}/reselect-runpod", response_model=LoraTrainingJobApi
)
def route_reselect_training_runpod(
    training_id: str,
    req: ReselectRunpodRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingJobApi:
    try:
        job = handler.lora_training.reselect_training(
            training_id, runpod_selection_from_api(req.selection)
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return training_job_to_api(job)


@router.post("/training/estimate", response_model=LoraCostEstimateResponse)
def route_estimate_training_cost(
    req: LoraCostEstimateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraCostEstimateResponse:
    jobs = handler.lora_training.get_training_state().items
    preprocessed_by_id = {
        item.id: item
        for item in handler.lora_training.get_preprocessed_state().items
    }
    datasets_by_id = {
        item.id: item for item in handler.lora_training.get_datasets_state().datasets
    }
    exact_gpu = bool(req.gpuType) and any(
        job.status == "completed" and job.gpu_type == req.gpuType for job in jobs
    )

    def vram_tier(value: int) -> int:
        return 80 if value >= 80 else 32 if value >= 32 else 0

    def elapsed(start: str | None, end: str | None) -> int | None:
        if start is None or end is None:
            return None
        try:
            return max(
                0,
                int(
                    (
                        datetime.fromisoformat(end) - datetime.fromisoformat(start)
                    ).total_seconds()
                ),
            )
        except ValueError:
            return None

    history: list[HistoricalTiming] = []
    for job in jobs:
        if (
            job.status != "completed"
            or job.started_at is None
            or job.completed_at is None
        ):
            continue
        if exact_gpu and job.gpu_type != req.gpuType:
            continue
        if (
            not exact_gpu
            and req.gpuVramGb > 0
            and job.gpu_vram_gb > 0
            and vram_tier(job.gpu_vram_gb) != vram_tier(req.gpuVramGb)
        ):
            continue
        if (
            job.config.preset != req.config.preset
            or job.config.with_audio != req.withAudio
        ):
            continue
        preprocessed = preprocessed_by_id.get(job.preprocessed_id)
        if preprocessed is None:
            continue
        effective_buckets = (
            preprocessed.effective_resolution_buckets
            or preprocessed.resolution_buckets
        )
        if effective_buckets != req.resolutionBuckets:
            continue
        dataset = datasets_by_id.get(preprocessed.dataset_id)
        if dataset is None or dataset.type != req.mode:
            continue
        try:
            started = datetime.fromisoformat(job.started_at)
            completed = datetime.fromisoformat(job.completed_at)
            first = (
                datetime.fromisoformat(job.first_step_at)
                if job.first_step_at is not None
                else started
            )
        except ValueError:
            continue
        history.append(
            HistoricalTiming(
                steps=job.total_steps or job.config.steps,
                setup_seconds=max(0, int((first - started).total_seconds())),
                train_seconds=max(1, int((completed - first).total_seconds())),
                upload_seconds=elapsed(
                    dataset.upload_started_at, dataset.upload_completed_at
                ),
                preprocess_seconds=elapsed(
                    preprocessed.started_at, preprocessed.completed_at
                ),
            )
        )
    estimate = estimate_cost(
        EstimateInputs(
            steps=req.config.steps,
            clip_count=req.clipCount,
            total_clip_seconds=req.totalClipSeconds,
            preprocessed=req.preprocessed,
            resolution_buckets=req.resolutionBuckets,
            mode=req.mode,
            with_audio=req.withAudio,
            gpu_price_per_hr=req.gpuPricePerHr,
            storage_readiness=req.storageReadiness,
            estimated_model_download_bytes=req.estimatedModelDownloadBytes,
            idle_timeout_minutes=req.idleTimeoutMinutes,
            storage_size_gb=req.storageSizeGb,
        ),
        tuple(history),
    )
    return LoraCostEstimateResponse(
        lowSeconds=estimate.low_seconds,
        highSeconds=estimate.high_seconds,
        lowGpuCost=estimate.low_gpu_cost,
        highGpuCost=estimate.high_gpu_cost,
        phases=[
            LoraEstimatePhaseApi(
                phase=phase.phase,
                lowSeconds=phase.low_seconds,
                highSeconds=phase.high_seconds,
            )
            for phase in estimate.phases
        ],
        confidence=estimate.confidence,
        matchedHistoryCount=estimate.matched_history_count,
        downloadBytes=estimate.download_bytes,
        storageMonthlyCost=estimate.storage_monthly_cost,
    )


# ---- Training profiles -------------------------------------------------


@router.get("/profiles", response_model=LoraTrainingProfilesResponse)
def route_list_profiles(
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingProfilesResponse:
    return profiles_state_to_api(handler.lora_training.get_profiles_state())


@router.post("/profiles", response_model=LoraTrainingProfileApi)
def route_create_profile(
    req: CreateTrainingProfileRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingProfileApi:
    profile = handler.lora_training.create_profile(
        name=req.name,
        config=training_config_from_api(req.config),
        description=req.description,
        dataset_types=list(req.datasetTypes),
    )
    return profile_to_api(profile)


@router.patch("/profiles/{profile_id}", response_model=LoraTrainingProfileApi)
def route_update_profile(
    profile_id: str,
    req: UpdateTrainingProfileRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingProfileApi:
    config = training_config_from_api(req.config) if req.config is not None else None
    try:
        profile = handler.lora_training.update_profile(
            profile_id,
            name=req.name,
            config=config,
            description=req.description,
            dataset_types=(
                list(req.datasetTypes) if req.datasetTypes is not None else None
            ),
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return profile_to_api(profile)


@router.delete("/profiles/{profile_id}", status_code=204)
def route_delete_profile(
    profile_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> None:
    try:
        handler.lora_training.delete_profile(profile_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None


@router.get("/training/{training_id}/logs", response_model=LoraJobLogsResponse)
def route_training_logs(
    training_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraJobLogsResponse:
    job = None
    for candidate in handler.lora_training.get_training_state().items:
        if candidate.id == training_id:
            job = candidate
            break
    if job is None:
        raise HTTPError(404, f"Training not found: {training_id}", code="LORA_ENTITY_NOT_FOUND")
    if job.target is None or job.target.remote_job_id is None:
        return LoraJobLogsResponse(lines=[])
    try:
        lines = handler.lora_training_runner.read_job_logs(job)
    except TrainerTargetError as exc:
        raise HTTPError(
            502,
            redact_text(exc.detail),
            code="LORA_REMOTE_ERROR",
        ) from None
    return LoraJobLogsResponse(lines=redact_lines(lines))


@router.get("/training/{training_id}/validation-media")
def route_training_validation_media(
    training_id: str,
    step: int = Query(ge=0),
    sampleIndex: int = Query(ge=1),
    extension: str = Query(default="mp4", pattern=r"^(mp4|wav|png)$"),
    handler: AppHandler = Depends(get_state_service),
) -> FileResponse:
    """Serve one downloaded validation sample's media file to the frontend.

    Secure by construction: the only request-supplied values are the run id and
    the (step, sampleIndex, extension) lookup key — the file path is resolved server-side
    from the job's `validation_feed` (written by the runner from a fixed remote
    `samples/` dir). No client-controlled path ever reaches the filesystem, so
    there's no path-traversal surface.
    """
    job = None
    for candidate in handler.lora_training.get_training_state().items:
        if candidate.id == training_id:
            job = candidate
            break
    if job is None:
        raise HTTPError(404, f"Training not found: {training_id}", code="LORA_ENTITY_NOT_FOUND")
    match = next(
        (
            i
            for i in job.validation_feed
            if i.step == step
            and i.sample_index == sampleIndex
            and i.extension == extension
        ),
        None,
    )
    if match is None or not match.local_path:
        raise HTTPError(
            404, "Validation sample not found", code="LORA_ENTITY_NOT_FOUND"
        )
    path = Path(match.local_path)
    if not path.is_file():
        raise HTTPError(
            404, "Validation sample file missing", code="LORA_ENTITY_NOT_FOUND"
        )
    # FileResponse infers the media type from the extension (mp4/png/wav).
    return FileResponse(str(path))


@router.post("/training/{training_id}/cancel", response_model=LoraTrainingJobApi)
def route_cancel_training(
    training_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingJobApi:
    try:
        job = handler.lora_training.request_cancel_training(training_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return training_job_to_api(job)


@router.post("/training/{training_id}/retry-download", response_model=LoraTrainingJobApi)
def route_retry_training_download(
    training_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingJobApi:
    """Re-fetch the trained adapter for a run that finished training but failed
    at the download step (the weights persist on the network volume)."""
    try:
        job = handler.lora_training.request_training_redownload(training_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return training_job_to_api(job)


@router.post("/training/{training_id}/resume", response_model=LoraTrainingJobApi)
def route_resume_training(
    training_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingJobApi:
    """Resume a failed/cancelled training run from its last saved checkpoint
    (re-runs ``train.py`` with ``load_checkpoint`` set to the highest step
    reached)."""
    try:
        job = handler.lora_training.request_training_resume(training_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return training_job_to_api(job)


@router.post("/training/{training_id}/reset", response_model=LoraTrainingJobApi)
def route_reset_training(
    training_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingJobApi:
    """Clear a finished training run's checkpoints/samples and re-train from
    scratch (a fresh start, vs. resume which continues from the last
    checkpoint)."""
    try:
        job = handler.lora_training.request_training_reset(training_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return training_job_to_api(job)


@router.delete("/training/{training_id}", status_code=204)
def route_delete_training(
    training_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> None:
    try:
        handler.lora_training.delete_training(training_id)
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None


@router.post("/training/{training_id}/archive", response_model=LoraTrainingJobApi)
def route_archive_training(
    training_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingJobApi:
    try:
        return training_job_to_api(handler.lora_training.archive_training(training_id))
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None


@router.post("/training/{training_id}/unarchive", response_model=LoraTrainingJobApi)
def route_unarchive_training(
    training_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTrainingJobApi:
    try:
        return training_job_to_api(handler.lora_training.unarchive_training(training_id))
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None


# ----------------------------------------------------------------
# Connection test
# ----------------------------------------------------------------


@router.post("/test-connection", response_model=LoraTestConnectionResponse)
def route_test_connection(
    req: LoraTestConnectionRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTestConnectionResponse:
    try:
        handler.lora_training_runner.test_connection()
    except TrainerTargetError as exc:
        return LoraTestConnectionResponse(ok=False, message=exc.detail)
    return LoraTestConnectionResponse(ok=True, message="Connection successful")


@router.get("/local-eligibility", response_model=LocalTrainerEligibilityResponse)
def route_local_eligibility(
    handler: AppHandler = Depends(get_state_service),
) -> LocalTrainerEligibilityResponse:
    """Report whether local (WSL2) LoRA training is possible on this machine.

    A read-only capability probe the UI polls to decide whether to offer the
    "train locally" provider; never errors (an unavailable setup is reported
    as `eligible=false` with a `reason`)."""
    result = handler.lora_training.local_trainer_eligibility()
    return LocalTrainerEligibilityResponse(
        eligible=result.eligible,
        reason=result.reason,
        wslInstalled=result.wsl_installed,
        cudaInWsl=result.cuda_in_wsl,
        gpuName=result.gpu_name,
        vramGb=result.vram_gb,
    )


@router.post("/runpod/connect", response_model=LoraConnectRunpodResponse)
def route_connect_runpod(
    handler: AppHandler = Depends(get_state_service),
) -> LoraConnectRunpodResponse:
    """Validate the saved key and discover GPUs, volumes, and readiness.

    This endpoint is read-only and never creates paid storage.
    """
    try:
        account, active_volume_id, readiness, readiness_by_volume = (
            handler.lora_training_runner.connect_runpod()
        )
    except TrainerTargetError as exc:
        return LoraConnectRunpodResponse(ok=False, message=exc.detail)
    health_by_dc = {health.datacenter_id: health for health in account.region_health}
    return LoraConnectRunpodResponse(
        ok=True,
        message="Connected",
        gpus=[
            LoraGpuOfferApi(
                id=g.id,
                label=g.label,
                memoryGb=g.memory_gb,
                pricePerHr=g.price_per_hr,
                available=g.available,
                activeRegionAvailable=g.active_region_available,
                availableElsewhere=g.available_elsewhere,
                bestAvailableRegion=g.best_available_region,
                recommended=g.recommended,
            )
            for g in account.gpus
        ],
        volumes=[
            _network_volume_to_api(
                v,
                active=v.id == active_volume_id,
                health=health_by_dc.get(v.datacenter_id),
                saved_model_readiness=readiness_by_volume[v.id][0],
            )
            for v in account.volumes
        ],
        pods=[_pod_info_to_api(p) for p in account.pods],
        activeVolumeId=active_volume_id,
        datacenter=account.datacenter,
        cacheEnabled=active_volume_id is not None,
        requiresVolumeSelection=False,
        regionHealth=[
            LoraRegionHealthApi(
                datacenterId=health.datacenter_id,
                status=health.status,
                qualifyingGpuAvailable=health.qualifying_gpu_available,
                availableGpuIds=list(health.available_gpu_ids),
            )
            for health in account.region_health
        ],
        savedModelReadiness=readiness[0],
        estimatedModelDownloadBytes=readiness[1],
    )


@router.post(
    "/runpod/volumes/create", response_model=LoraNetworkVolumeActionResponse
)
def route_create_runpod_volume(
    req: LoraCreateNetworkVolumeRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraNetworkVolumeActionResponse:
    """Explicitly create and select one paid app-owned cache volume."""
    try:
        volume = handler.lora_training_runner.create_runpod_volume(
            datacenter_id=req.datacenterId, size_gb=req.sizeGb
        )
    except TrainerTargetError as exc:
        raise HTTPError(409, exc.detail, code="LORA_CACHE_VOLUME_ERROR") from None
    return LoraNetworkVolumeActionResponse(
        ok=True,
        message="Cache volume created and selected",
        volume=_network_volume_to_api(volume, active=True),
        provisioningRequired=True,
    )


@router.post(
    "/runpod/volumes/select", response_model=LoraNetworkVolumeActionResponse
)
def route_select_runpod_volume(
    req: LoraSelectNetworkVolumeRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraNetworkVolumeActionResponse:
    try:
        volume = handler.lora_training_runner.select_runpod_volume(req.volumeId)
    except TrainerTargetError as exc:
        raise HTTPError(409, exc.detail, code="LORA_CACHE_VOLUME_ERROR") from None
    return LoraNetworkVolumeActionResponse(
        ok=True,
        message="Primary cache selected",
        volume=_network_volume_to_api(volume, active=True),
    )


@router.post(
    "/runpod/cache/disable", response_model=LoraNetworkVolumeActionResponse
)
def route_disable_runpod_cache(
    handler: AppHandler = Depends(get_state_service),
) -> LoraNetworkVolumeActionResponse:
    handler.lora_training_runner.disable_runpod_cache()
    return LoraNetworkVolumeActionResponse(
        ok=True,
        message="Cache detached; new pipelines use ephemeral any-region workspaces",
    )


@router.post(
    "/runpod/volumes/relocate", response_model=LoraNetworkVolumeActionResponse
)
def route_relocate_runpod_volume(
    req: LoraRelocateNetworkVolumeRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraNetworkVolumeActionResponse:
    try:
        volume, previous_id = handler.lora_training_runner.relocate_runpod_volume(
            datacenter_id=req.datacenterId, size_gb=req.sizeGb
        )
    except TrainerTargetError as exc:
        raise HTTPError(409, exc.detail, code="LORA_CACHE_VOLUME_ERROR") from None
    return LoraNetworkVolumeActionResponse(
        ok=True,
        message=(
            "Replacement cache created and selected. It will be provisioned by "
            "the next pipeline; the previous volume was retained."
        ),
        volume=_network_volume_to_api(volume, active=True),
        previousVolumeId=previous_id,
        provisioningRequired=True,
    )


@router.delete(
    "/runpod/volumes/{volume_id}", response_model=LoraNetworkVolumeActionResponse
)
def route_delete_runpod_volume(
    volume_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraNetworkVolumeActionResponse:
    try:
        handler.lora_training_runner.delete_runpod_volume(volume_id)
    except TrainerTargetError as exc:
        raise HTTPError(409, exc.detail, code="LORA_CACHE_VOLUME_ERROR") from None
    return LoraNetworkVolumeActionResponse(ok=True, message="Cache volume deleted")


@router.post("/runpod/pods/{pod_id}/terminate", response_model=LoraTerminatePodResponse)
def route_terminate_runpod_pod(
    pod_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraTerminatePodResponse:
    """Terminate a specific RunPod pod (from the connect UI's active-pods list)."""
    try:
        handler.lora_training_runner.terminate_runpod_pod(pod_id)
    except TrainerTargetError as exc:
        if exc.code == "ownership_violation":
            raise HTTPError(403, exc.detail, code="LORA_POD_NOT_OWNED") from None
        return LoraTerminatePodResponse(ok=False, message=exc.detail)
    return LoraTerminatePodResponse(ok=True, message="Pod terminated")


@router.get("/runpod/pods", response_model=list[LoraPodInfoApi])
def route_list_runpod_pods(
    handler: AppHandler = Depends(get_state_service),
) -> list[LoraPodInfoApi]:
    """List every RunPod pod on the account for the Trainer compute panel."""
    try:
        pods = handler.lora_training_runner.list_runpod_pods()
    except TrainerTargetError as exc:
        raise HTTPError(502, exc.detail, code="LORA_REMOTE_ERROR") from None
    return [_pod_info_to_api(p) for p in pods]


@router.post("/runpod/pods/{pod_id}/stop", response_model=LoraPodActionResponse)
def route_stop_runpod_pod(
    pod_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraPodActionResponse:
    """Pause a running RunPod pod (stops GPU billing, keeps the container disk)."""
    try:
        handler.lora_training_runner.stop_runpod_pod(pod_id)
    except TrainerTargetError as exc:
        if exc.code == "ownership_violation":
            raise HTTPError(403, exc.detail, code="LORA_POD_NOT_OWNED") from None
        return LoraPodActionResponse(ok=False, message=exc.detail)
    return LoraPodActionResponse(ok=True, message="Pod stopped")


@router.post("/runpod/pods/{pod_id}/resume", response_model=LoraPodActionResponse)
def route_resume_runpod_pod(
    pod_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraPodActionResponse:
    """Start a stopped RunPod pod (resumes GPU billing)."""
    try:
        handler.lora_training_runner.resume_runpod_pod(pod_id)
    except TrainerTargetError as exc:
        if exc.code == "ownership_violation":
            raise HTTPError(403, exc.detail, code="LORA_POD_NOT_OWNED") from None
        return LoraPodActionResponse(ok=False, message=exc.detail)
    return LoraPodActionResponse(ok=True, message="Pod resumed")


@router.post("/runpod/pods/{pod_id}/keep-alive", response_model=LoraDatasetApi)
def route_keep_runpod_pod_alive(
    pod_id: str,
    req: LoraKeepAliveRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDatasetApi:
    try:
        dataset = handler.lora_training.extend_workspace_keep_alive(
            pod_id, minutes=req.minutes
        )
    except LoraEntityNotFoundError as exc:
        raise _not_found(exc) from None
    except LoraTransitionError as exc:
        raise _conflict(exc) from None
    return dataset_to_api(dataset)


def _pod_info_to_api(p: PodInfo) -> LoraPodInfoApi:
    """Map a backend `PodInfo` to its API shape (shared by connect + list)."""
    return LoraPodInfoApi(
        id=p.id,
        name=p.name,
        gpu=p.gpu,
        status=p.status,
        costPerHr=p.cost_per_hr,
        createdByApp=p.created_by_app,
        desiredStatus=p.desired_status,
        running=p.running,
        uptimeSeconds=p.uptime_seconds,
        lastStartedAt=p.last_started_at,
    )


def _network_volume_to_api(
    volume: NetworkVolume,
    *,
    active: bool,
    health: RegionHealth | None = None,
    saved_model_readiness: SavedModelReadiness = "unknown",
) -> LoraNetworkVolumeApi:
    return LoraNetworkVolumeApi(
        id=volume.id,
        name=volume.name,
        sizeGb=volume.size_gb,
        datacenterId=volume.datacenter_id,
        createdByApp=volume.created_by_app,
        active=active,
        regionHealth=health.status if health is not None else "unknown",
        qualifyingGpuAvailable=(
            health.qualifying_gpu_available if health is not None else None
        ),
        availableGpuIds=(
            list(health.available_gpu_ids) if health is not None else []
        ),
        savedModelReadiness=saved_model_readiness,
    )


# ----------------------------------------------------------------
# Clip-prep jobs (local sprite/filmstrip generation)
# ----------------------------------------------------------------


@router.post("/clip-jobs", response_model=LoraClipJobsResponse)
def route_enqueue_clip_jobs(
    req: LoraEnqueueClipJobsRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraClipJobsResponse:
    """Enqueue local prep jobs for a batch of source clips.

    Returns the (possibly reused) jobs immediately; the actual ffmpeg
    work runs on the clip-jobs runner. Poll `GET /clip-jobs` for results.
    """
    handler.lora_training.enqueue_clip_jobs(
        source_paths=req.sourcePaths, kind=req.kind
    )
    return clip_jobs_state_to_api(handler.lora_training.get_clip_jobs_state())


@router.get("/clip-jobs", response_model=LoraClipJobsResponse)
def route_list_clip_jobs(
    handler: AppHandler = Depends(get_state_service),
) -> LoraClipJobsResponse:
    return clip_jobs_state_to_api(handler.lora_training.get_clip_jobs_state())


# --- Target/variant derivation jobs (background pipeline) ------------------


@router.post("/derivations", response_model=LoraDerivationJobApi)
def route_create_derivation(
    req: CreateLoraDerivationJobRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivationJobApi:
    """Enqueue a background 'generate target/variant' pipeline.

    Returns the created job immediately; the multi-stage work (frame edit
    -> local IC-LoRA drive or remote Kling) runs on the derivation runner.
    Poll `GET /derivations` for progress and results.
    """
    job = handler.lora_training.enqueue_derivation_job(req)
    return derivation_job_to_api(job)


@router.get("/derivations", response_model=LoraDerivationJobsResponse)
def route_list_derivations(
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivationJobsResponse:
    return derivation_jobs_state_to_api(handler.lora_training.get_derivation_jobs_state())


@router.post("/derivations/cancel-all", response_model=LoraDerivationJobsResponse)
def route_cancel_all_derivations(
    req: CancelAllLoraDerivationsRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivationJobsResponse:
    """Abort a whole bulk Fal generation run (optionally scoped to a dataset).

    Cancels every active job at once; the runner stops in-flight Fal calls
    between stages. Returns the updated jobs ledger.
    """
    handler.lora_training.cancel_all_derivation_jobs(dataset_id=req.datasetId)
    return derivation_jobs_state_to_api(handler.lora_training.get_derivation_jobs_state())


@router.post("/derivations/{job_id}/cancel", response_model=LoraDerivationJobApi)
def route_cancel_derivation(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivationJobApi:
    job = handler.lora_training.cancel_derivation_job(job_id)
    if job is None:
        raise HTTPError(404, "Derivation job not found", code="LORA_NOT_FOUND")
    return derivation_job_to_api(job)


@router.post("/derivations/{job_id}/approve", response_model=LoraDerivationJobApi)
def route_approve_derivation(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivationJobApi:
    """Approve a reviewed edit; the job proceeds to the motion drive."""
    job = handler.lora_training.approve_derivation_job(job_id)
    if job is None:
        raise HTTPError(409, "Job is not awaiting review", code="LORA_DERIVATION_ERROR")
    return derivation_job_to_api(job)


@router.post("/derivations/{job_id}/regenerate-edit", response_model=LoraDerivationJobApi)
def route_regenerate_derivation_edit(
    job_id: str,
    req: RegenerateLoraDerivationEditRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivationJobApi:
    """Re-run the Nano Banana edit for a reviewed job (optional new prompt)."""
    job = handler.lora_training.regenerate_derivation_edit(
        job_id, edit_prompt=req.editPrompt
    )
    if job is None:
        raise HTTPError(409, "Job is not awaiting review", code="LORA_DERIVATION_ERROR")
    return derivation_job_to_api(job)


@router.post("/derivations/{job_id}/retry", response_model=LoraDerivationJobApi)
def route_retry_derivation(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivationJobApi:
    job = handler.lora_training.retry_derivation_job(job_id)
    if job is None:
        raise HTTPError(409, "Job cannot be retried", code="LORA_DERIVATION_ERROR")
    return derivation_job_to_api(job)


@router.post("/derivations/{job_id}/dismiss", response_model=LoraDerivationJobsResponse)
def route_dismiss_derivation(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> LoraDerivationJobsResponse:
    handler.lora_training.dismiss_derivation_job(job_id)
    return derivation_jobs_state_to_api(handler.lora_training.get_derivation_jobs_state())
