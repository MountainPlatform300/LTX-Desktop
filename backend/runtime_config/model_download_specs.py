"""Canonical checkpoint specs and LTX model relationships."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never, cast, get_args

from api_types import (
    LTXLocalModelId,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenPipeline,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    ModelCheckpointID,
)

logger = logging.getLogger(__name__)


ALL_MODEL_CP_IDS = cast(tuple[ModelCheckpointID, ...], get_args(ModelCheckpointID))
ALL_LTX_LOCAL_MODEL_IDS = cast(tuple[LTXLocalModelId, ...], get_args(LTXLocalModelId))


@dataclass(frozen=True, slots=True)
class ModelCheckpointSpec:
    relative_path: Path
    expected_size_bytes: int
    is_folder: bool
    repo_id: str
    description: str
    # True when the HuggingFace repo is gated — the downloader must attach an
    # HF token and the user must have accepted the gate on the repo page before
    # the download will succeed. Surfaced in the UI so the picker can route to
    # the HF auth flow before attempting the download.
    is_gated: bool = False

    @property
    def name(self) -> str:
        return self.relative_path.name


@dataclass(frozen=True, slots=True)
class LTXLocalModelDeprecated:
    pass


@dataclass(frozen=True, slots=True)
class LTXLocalModelRelevant:
    upgrade_messages: dict[LTXLocalModelId, str]


LTXLocalModelRelevance = LTXLocalModelDeprecated | LTXLocalModelRelevant


@dataclass(frozen=True, slots=True)
class LtxIcLorasSpec:
    depth_cp: ModelCheckpointID
    canny_cp: ModelCheckpointID
    pose_cp: ModelCheckpointID


@dataclass(frozen=True, slots=True)
class LTXLocalModelSpec:
    model_cp: ModelCheckpointID
    upscale_cp: ModelCheckpointID
    text_encoder_cp: ModelCheckpointID
    ic_loras_spec: LtxIcLorasSpec
    relevance: LTXLocalModelRelevance
    supported_pipelines: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...]
    # Optional higher-quality "dev base" overlay for IC-LoRA: when the user opts
    # in (AppSettings.use_dev_quality_base) and both are downloaded, the IC-LoRA
    # pipeline loads `quality_base_cp` (the full dev checkpoint) with
    # `distilled_lora_cp` stacked @0.5 instead of the distilled checkpoint —
    # matching the ComfyUI dev + distilled-LoRA flow. None = no overlay.
    quality_base_cp: ModelCheckpointID | None = None
    distilled_lora_cp: ModelCheckpointID | None = None


def _local_resolution_spec(
    *,
    fps_to_durations: dict[LTXVideoGenFps, tuple[LTXVideoGenDuration, ...]],
) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            fps: list(durations)
            for fps, durations in fps_to_durations.items()
        },
    )


IMG_GEN_MODEL_CP_ID: ModelCheckpointID = "z-image-turbo"
DEPTH_PROCESSOR_CP_ID: ModelCheckpointID = "dpt-hybrid-midas"
PERSON_DETECTOR_CP_ID: ModelCheckpointID = "yolox-l-torchscript"
POSE_PROCESSOR_CP_ID: ModelCheckpointID = "dw-ll-ucoco-384-bs5"


def get_model_cp_spec(cp_id: ModelCheckpointID) -> ModelCheckpointSpec:
    match cp_id:
        case "ltx-2.3-22b-distilled":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-distilled.safetensors"),
                expected_size_bytes=43_000_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="Main transformer model",
            )
        case "ltx-2.3-22b-dev":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-dev.safetensors"),
                expected_size_bytes=46_100_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="Full dev transformer (opt-in IC-LoRA quality base)",
            )
        case "ltx-2.3-22b-distilled-lora-384-1.1":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
                expected_size_bytes=7_610_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="Distilled v1.1 LoRA for the dev base (opt-in IC-LoRA quality base)",
            )
        case "ltx-2.3-spatial-upscaler-x2-1.0":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
                expected_size_bytes=1_900_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="2x upscaler",
            )
        case "ltx-2.3-22b-ic-lora-union-control-ref0.5":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
                expected_size_bytes=654_465_352,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                description="Union IC-LoRA control model",
            )
        case "dpt-hybrid-midas":
            return ModelCheckpointSpec(
                relative_path=Path("dpt-hybrid-midas"),
                expected_size_bytes=500_000_000,
                is_folder=True,
                repo_id="Intel/dpt-hybrid-midas",
                description="DPT-Hybrid MiDaS depth processor",
            )
        case "yolox-l-torchscript":
            return ModelCheckpointSpec(
                relative_path=Path("yolox_l.torchscript.pt"),
                expected_size_bytes=217_697_649,
                is_folder=False,
                repo_id="hr16/yolox-onnx",
                description="YOLOX person detector for pose preprocessing",
            )
        case "dw-ll-ucoco-384-bs5":
            return ModelCheckpointSpec(
                relative_path=Path("dw-ll_ucoco_384_bs5.torchscript.pt"),
                expected_size_bytes=135_059_124,
                is_folder=False,
                repo_id="hr16/DWPose-TorchScript-BatchSize5",
                description="DW Pose TorchScript processor",
            )
        case "gemma-3-12b-it-qat-q4_0-unquantized":
            return ModelCheckpointSpec(
                relative_path=Path("gemma-3-12b-it-qat-q4_0-unquantized"),
                expected_size_bytes=25_000_000_000,
                is_folder=True,
                repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
                description="Gemma text encoder (bfloat16)",
            )
        case "z-image-turbo":
            return ModelCheckpointSpec(
                relative_path=Path("Z-Image-Turbo"),
                expected_size_bytes=31_000_000_000,
                is_folder=True,
                repo_id="Tongyi-MAI/Z-Image-Turbo",
                description="Z-Image-Turbo model for text-to-image generation",
            )
        case "flux-2-klein-9b":
            return ModelCheckpointSpec(
                relative_path=Path("FLUX.2-klein-9B"),
                # Estimate: 9B transformer (bf16) + 8B Qwen3 text encoder + VAE.
                # Tunable after a first real download — only affects progress-bar
                # accuracy, not functionality.
                expected_size_bytes=34_000_000_000,
                is_folder=True,
                repo_id="black-forest-labs/FLUX.2-klein-9B",
                description=(
                    "FLUX.2 [klein] 9B — unified text-to-image and multi-reference "
                    "image editing (gated on HuggingFace, non-commercial)"
                ),
                is_gated=True,
            )
        case _:
            assert_never(cp_id)


def get_ltx_model_spec(model_id: LTXLocalModelId) -> LTXLocalModelSpec:
    match model_id:
        case "ltx-2.3-22b-distilled":
            return LTXLocalModelSpec(
                model_cp="ltx-2.3-22b-distilled",
                upscale_cp="ltx-2.3-spatial-upscaler-x2-1.0",
                text_encoder_cp="gemma-3-12b-it-qat-q4_0-unquantized",
                ic_loras_spec=LtxIcLorasSpec(
                    depth_cp="ltx-2.3-22b-ic-lora-union-control-ref0.5",
                    canny_cp="ltx-2.3-22b-ic-lora-union-control-ref0.5",
                    pose_cp="ltx-2.3-22b-ic-lora-union-control-ref0.5",
                ),
                relevance=LTXLocalModelRelevant(upgrade_messages={}),
                # Opt-in dev quality base for IC-LoRA: dev checkpoint +
                # distilled v1.1 LoRA @0.5 (never applied to the distilled
                # checkpoint itself — only to dev).
                quality_base_cp="ltx-2.3-22b-dev",
                distilled_lora_cp="ltx-2.3-22b-distilled-lora-384-1.1",
                supported_pipelines=(
                    (
                        "fast",
                        LTXVideoGenerationSpec(
                            display_name="LTX 2.3 Fast",
                            supported_resolutions_durations={
                                "540p": _local_resolution_spec(
                                    fps_to_durations={
                                        24: (5, 6, 8, 10, 20),
                                    },
                                ),
                                "720p": _local_resolution_spec(
                                    fps_to_durations={
                                        24: (5, 6, 8, 10),
                                    },
                                ),
                                "1080p": _local_resolution_spec(
                                    fps_to_durations={
                                        24: (5,),
                                    },
                                ),
                            },
                        ),
                    ),
                ),
            )
        case _:
            assert_never(model_id)


def get_ltx_cps() -> set[ModelCheckpointID]:
    cp_ids: set[ModelCheckpointID] = set()
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        cp_ids.add(get_ltx_model_spec(model_id).model_cp)
    return cp_ids


def get_latest_ltx_model_id() -> LTXLocalModelId:
    relevant: list[LTXLocalModelId] = []
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if isinstance(get_ltx_model_spec(model_id).relevance, LTXLocalModelRelevant):
            relevant.append(model_id)
    if len(relevant) != 1:
        raise RuntimeError(f"Expected exactly one relevant LTX model, found {len(relevant)}")
    return relevant[0]


def get_ltx_model_id_for_cp(cp_id: ModelCheckpointID) -> LTXLocalModelId | None:
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if get_ltx_model_spec(model_id).model_cp == cp_id:
            return model_id
    return None


def get_ic_loras_cp_ids(ic_loras_spec: LtxIcLorasSpec) -> tuple[ModelCheckpointID, ...]:
    return tuple(dict.fromkeys((ic_loras_spec.depth_cp, ic_loras_spec.canny_cp, ic_loras_spec.pose_cp)))


def get_ltx_model_cp_ids(model_id: LTXLocalModelId) -> tuple[ModelCheckpointID, ...]:
    spec = get_ltx_model_spec(model_id)
    return (
        spec.model_cp,
        spec.upscale_cp,
        spec.text_encoder_cp,
        *get_ic_loras_cp_ids(spec.ic_loras_spec),
    )


def get_ltx_overlay_cp_ids(model_id: LTXLocalModelId) -> tuple[ModelCheckpointID, ...]:
    """Opt-in dev quality-base checkpoints (not part of the required bundle).

    Returned in download order (base, then LoRA). Empty when the model has no
    overlay configured. Callers gate on `AppSettings.use_dev_quality_base` —
    these are never forced on a default install.
    """
    spec = get_ltx_model_spec(model_id)
    overlay: list[ModelCheckpointID] = []
    if spec.quality_base_cp is not None:
        overlay.append(spec.quality_base_cp)
    if spec.distilled_lora_cp is not None:
        overlay.append(spec.distilled_lora_cp)
    return tuple(overlay)


def _normalized_relative_path(cp_id: ModelCheckpointID) -> Path:
    relative_path = get_model_cp_spec(cp_id).relative_path
    if relative_path.is_absolute():
        raise ValueError(f"Model path for {cp_id} must be relative: {relative_path}")

    normalized_parts = [part for part in relative_path.parts if part not in ("", ".")]
    if not normalized_parts:
        raise ValueError(f"Model path for {cp_id} cannot be empty: {relative_path}")
    if ".." in normalized_parts:
        raise ValueError(f"Model path for {cp_id} cannot traverse parents: {relative_path}")

    return Path(*normalized_parts)


def resolve_model_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    return models_dir / _normalized_relative_path(cp_id)


def resolve_downloading_dir(models_dir: Path) -> Path:
    return models_dir / ".downloading"


def resolve_downloading_target_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    return resolve_downloading_dir(models_dir) / _normalized_relative_path(cp_id)


def resolve_downloading_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    spec = get_model_cp_spec(cp_id)
    relative_path = _normalized_relative_path(cp_id)
    downloading_dir = resolve_downloading_dir(models_dir)
    if spec.is_folder:
        return downloading_dir / relative_path
    parent = relative_path.parent
    if parent == Path("."):
        return downloading_dir
    return downloading_dir / parent


def is_cp_downloaded(models_dir: Path, cp_id: ModelCheckpointID) -> bool:
    path = resolve_model_path(models_dir, cp_id)
    spec = get_model_cp_spec(cp_id)
    if spec.is_folder:
        return path.exists() and any(path.iterdir())
    return path.exists()


def get_existing_cp_path(models_dir: Path, cp_id: ModelCheckpointID) -> Path:
    path = resolve_model_path(models_dir, cp_id)
    if not is_cp_downloaded(models_dir, cp_id):
        raise FileNotFoundError(f"Checkpoint not found: {cp_id} at {path}")
    return path


def delete_cp_path(models_dir: Path, cp_id: ModelCheckpointID) -> None:
    path = resolve_model_path(models_dir, cp_id)
    spec = get_model_cp_spec(cp_id)
    if spec.is_folder:
        if path.exists():
            import shutil

            shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)


def get_downloaded_ltx_model_id(models_dir: Path) -> LTXLocalModelId | None:
    downloaded: list[LTXLocalModelId] = []
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if is_cp_downloaded(models_dir, get_ltx_model_spec(model_id).model_cp):
            downloaded.append(model_id)
    if not downloaded:
        return None
    if len(downloaded) == 1:
        return downloaded[0]

    logger.warning("Multiple LTX model checkpoints detected: %s", ", ".join(downloaded))
    relevant: list[LTXLocalModelId] = []
    for model_id in downloaded:
        if isinstance(get_ltx_model_spec(model_id).relevance, LTXLocalModelRelevant):
            relevant.append(model_id)
    if len(relevant) == 1:
        return relevant[0]
    if len(relevant) > 1:
        logger.warning("Multiple relevant LTX models detected; selecting the first available: %s", relevant[0])
        return relevant[0]
    logger.warning("Multiple deprecated LTX models detected; selecting the first available: %s", downloaded[0])
    return downloaded[0]


def _validate_model_cp_specs() -> None:
    relative_paths: dict[Path, ModelCheckpointID] = {}
    for cp_id in ALL_MODEL_CP_IDS:
        normalized = _normalized_relative_path(cp_id)
        existing = relative_paths.get(normalized)
        if existing is not None:
            raise RuntimeError(f"Duplicate checkpoint path mapping: {existing} and {cp_id} -> {normalized}")
        relative_paths[normalized] = cp_id


def _validate_ltx_specs() -> None:
    ltx_cps = get_ltx_cps()
    if len(ltx_cps) != len(ALL_LTX_LOCAL_MODEL_IDS):
        raise RuntimeError("LTX model primary checkpoints must map 1:1 with LTX model ids")
    _ = get_latest_ltx_model_id()


_validate_model_cp_specs()
_validate_ltx_specs()
