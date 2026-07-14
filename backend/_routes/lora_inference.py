"""Routes for in-app LoRA inference (Gen Space "Apply LoRA") + the LoRA Library.

The registry endpoint lists every LoRA Gen Space can apply — the official
IC-LoRA union-control adapter, user-trained adapters from completed training
jobs, and imported adapters from the user's LoRA library. The generate
endpoint routes a registry-picked LoRA to the right pipeline by variant
(standard / union_control / video_input_ic_lora). The Gen Space frontend
enqueues LoRA generations through the durable queue (`kind="lora"`); this
synchronous route is the same handler call, exposed for direct use and
testing.

The import / update / reprofile / delete endpoints back the LoRA Library: an
imported LoRA's name / description / HuggingFace URL are editable after import,
its prompt profile can be re-derived, and a user-trained LoRA's display
metadata can be edited (and the whole training job deleted) without leaving the
library surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from api_types import (
    AutoPromptRequest,
    AutoPromptResponse,
    ImportLoraRequest,
    ImportLoraResponse,
    LoraGenerateRequest,
    LoraGenerateResponse,
    LoraInferenceRegistryResponseApi,
    ReprofileImportedLoraRequest,
    ReprofileImportedLoraResponse,
    SetLoraExampleRequest,
    SetLoraExampleResponse,
    UpdateImportedLoraRequest,
    UpdateImportedLoraResponse,
    UpdateLoraPromptTemplateRequest,
    UpdateLoraPromptTemplateResponse,
    UpdateTrainedLoraRequest,
    UpdateTrainedLoraResponse,
)
from app_handler import AppHandler
from handlers.lora_training_handler import LoraEntityNotFoundError, LoraTransitionError
from _routes._errors import HTTPError
from state import get_state_service

router = APIRouter(prefix="/api/lora-inference", tags=["lora-inference"])


@router.get("/registry", response_model=LoraInferenceRegistryResponseApi)
def route_lora_inference_registry(
    handler: AppHandler = Depends(get_state_service),
) -> LoraInferenceRegistryResponseApi:
    return LoraInferenceRegistryResponseApi(
        entries=handler.lora_inference_registry.list_entries()
    )


@router.post("/generate", response_model=LoraGenerateResponse)
def route_lora_inference_generate(
    req: LoraGenerateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> LoraGenerateResponse:
    return handler.lora_inference.generate(req)


@router.post("/import", response_model=ImportLoraResponse)
def route_lora_inference_import(
    req: ImportLoraRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ImportLoraResponse:
    entry, profile_status, profile_message = handler.lora_inference.import_lora_with_profile(
        source_path=req.sourcePath,
        name=req.name,
        variant=req.variant,
        description=req.description,
        trigger_word=req.triggerWord,
        huggingface_url=req.huggingfaceUrl,
        example_prompt=req.examplePrompt,
    )
    return ImportLoraResponse(
        entry=entry,
        profileStatus=profile_status,
        profileMessage=profile_message,
    )


@router.delete("/imported/{lora_id}", status_code=204)
def route_lora_inference_delete_imported(
    lora_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> None:
    handler.imported_lora_library.delete_imported(lora_id)


@router.patch("/imported/{lora_id}", response_model=UpdateImportedLoraResponse)
def route_lora_inference_update_imported(
    lora_id: str,
    req: UpdateImportedLoraRequest,
    handler: AppHandler = Depends(get_state_service),
) -> UpdateImportedLoraResponse:
    entry = handler.lora_inference.update_imported(
        lora_id=lora_id,
        name=req.name,
        description=req.description,
        huggingface_url=req.huggingfaceUrl,
    )
    return UpdateImportedLoraResponse(entry=entry)


@router.post(
    "/imported/{lora_id}/reprofile",
    response_model=ReprofileImportedLoraResponse,
)
def route_lora_inference_reprofile_imported(
    lora_id: str,
    req: ReprofileImportedLoraRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ReprofileImportedLoraResponse:
    entry, profile_status, profile_message = handler.lora_inference.reprofile_imported(
        lora_id=lora_id,
        huggingface_url=req.huggingfaceUrl,
        example_prompt=req.examplePrompt,
    )
    return ReprofileImportedLoraResponse(
        entry=entry,
        profileStatus=profile_status,
        profileMessage=profile_message,
    )


@router.patch("/trained/{lora_id}", response_model=UpdateTrainedLoraResponse)
def route_lora_inference_update_trained(
    lora_id: str,
    req: UpdateTrainedLoraRequest,
    handler: AppHandler = Depends(get_state_service),
) -> UpdateTrainedLoraResponse:
    try:
        entry = handler.lora_inference.update_trained(
            lora_id=lora_id,
            name=req.name,
            description=req.description,
        )
    except LoraEntityNotFoundError as exc:
        raise HTTPError(404, str(exc), code="LORA_ENTITY_NOT_FOUND") from None
    except LoraTransitionError as exc:
        raise HTTPError(409, str(exc), code="LORA_INVALID_TRANSITION") from None
    return UpdateTrainedLoraResponse(entry=entry)


@router.delete("/trained/{lora_id}", status_code=204)
def route_lora_inference_delete_trained(
    lora_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> None:
    try:
        handler.lora_inference.delete_trained(lora_id)
    except LoraEntityNotFoundError as exc:
        raise HTTPError(404, str(exc), code="LORA_ENTITY_NOT_FOUND") from None
    except LoraTransitionError as exc:
        raise HTTPError(409, str(exc), code="LORA_INVALID_TRANSITION") from None


# ------------------------------------------------------------
# Example media (CivitAI-style "what does this LoRA do?" preview)
# ------------------------------------------------------------
#
# Attach / replace / remove a user-supplied example image or video on any
# user-managed LoRA (imported or trained). The file is copied into app storage
# server-side; the bytes are served back through `example-media` so no
# filesystem path is ever exposed to the client.
@router.post("/entries/{lora_id}/example", response_model=SetLoraExampleResponse)
def route_lora_inference_set_example(
    lora_id: str,
    req: SetLoraExampleRequest,
    handler: AppHandler = Depends(get_state_service),
) -> SetLoraExampleResponse:
    entry = handler.lora_inference.set_example(lora_id=lora_id, source_path=req.sourcePath)
    return SetLoraExampleResponse(entry=entry)


@router.delete("/entries/{lora_id}/example", status_code=204)
def route_lora_inference_clear_example(
    lora_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> None:
    handler.lora_inference.clear_example(lora_id)


@router.get("/entries/{lora_id}/example-media")
def route_lora_inference_example_media(
    lora_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> FileResponse:
    """Serve a LoRA's example media bytes to the frontend.

    Secure by construction: only the entry id is client-supplied; the file path
    is resolved server-side from the imported ledger / training job. No
    client-controlled path reaches the filesystem, so there's no traversal
    surface. 404s cleanly when the entry has no example or the file is missing.
    """
    path = handler.lora_inference.get_example_local_path(lora_id)
    if path is None:
        raise HTTPError(404, "Example media not found", code="LORA_EXAMPLE_NOT_FOUND")
    # FileResponse infers the media type from the extension (mp4/png/webp/...).
    return FileResponse(path)


@router.post("/auto-prompt", response_model=AutoPromptResponse)
def route_lora_inference_auto_prompt(
    req: AutoPromptRequest,
    handler: AppHandler = Depends(get_state_service),
) -> AutoPromptResponse:
    prompt = handler.lora_inference.auto_prompt(lora_id=req.loraId, video_path=req.videoPath)
    return AutoPromptResponse(prompt=prompt)


@router.put(
    "/prompt-template/{lora_id}",
    response_model=UpdateLoraPromptTemplateResponse,
)
def route_lora_inference_update_prompt_template(
    lora_id: str,
    req: UpdateLoraPromptTemplateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> UpdateLoraPromptTemplateResponse:
    entry = handler.lora_inference.update_prompt_template(
        lora_id=lora_id,
        prompt_template=req.promptTemplate,
        trigger_word=req.triggerWord,
    )
    return UpdateLoraPromptTemplateResponse(entry=entry)
