"""LTX IC-LoRA pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


class LTXIcLoraPipeline:
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        distilled_lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> "LTXIcLoraPipeline":
        return LTXIcLoraPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            lora_path=lora_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            distilled_lora_path=distilled_lora_path,
            lora_scale=lora_scale,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        distilled_lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.ic_lora import ICLoraPipeline

        self._streaming_prefetch_count = streaming_prefetch_count
        # Diagnostic: log each adapter's key format so a misformatted LoRA (e.g.
        # an LTX-native file run through the Comfy rename map, which silently
        # no-ops) is obvious in the backend log instead of looking like "my LoRA
        # has no effect". Reads only the safetensors header; never raises.
        from services.lora_diagnostics import log_lora_diagnostics

        log_lora_diagnostics(
            lora_path=distilled_lora_path,
            label="IC-LoRA distilled base",
            base_checkpoint_path=checkpoint_path,
            applying_comfy_rename=True,
        )
        log_lora_diagnostics(
            lora_path=lora_path,
            label="IC-LoRA",
            base_checkpoint_path=checkpoint_path,
            distilled_lora_path=distilled_lora_path,
            applying_comfy_rename=True,
        )
        # The requested IC-LoRA strength is baked into the pipeline at load.
        # When a distilled LoRA is supplied (the opt-in dev quality base), it is
        # stacked at 0.5 on top of the dev checkpoint — the distilled LoRA must
        # never be applied to an already-distilled checkpoint, so the handler
        # only passes it when the dev base is in use.
        loras: list[LoraPathStrengthAndSDOps] = []
        if distilled_lora_path is not None:
            loras.append(LoraPathStrengthAndSDOps(path=distilled_lora_path, strength=0.5, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP))
        loras.append(LoraPathStrengthAndSDOps(path=lora_path, strength=lora_scale, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP))
        self.pipeline = ICLoraPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=upsampler_path,
            gemma_root=cast(str, gemma_root),
            loras=loras,
            device=device,
            quantization=QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None,
        )

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        tiling_config: TilingConfigType,
        skip_stage_2: bool,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        return self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            video_conditioning=video_conditioning,
            tiling_config=tiling_config,
            streaming_prefetch_count=self._streaming_prefetch_count,
            skip_stage_2=skip_stage_2,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        output_path: str,
        skip_stage_2: bool = True,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            video_conditioning=video_conditioning,
            tiling_config=tiling_config,
            skip_stage_2=skip_stage_2,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)
