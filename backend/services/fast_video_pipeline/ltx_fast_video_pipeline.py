"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Final, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


class LTXFastVideoPipeline:
    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._device = device
        self._streaming_prefetch_count = streaming_prefetch_count
        self._quantization = QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None
        # A standard user-trained LoRA applied to t2v/i2v: built into the
        # DistilledPipeline at load time (same LoraPathStrengthAndSDOps mechanism
        # the IC-LoRA pipeline uses). None -> base model, no adapter.
        self._lora_path = lora_path
        self._lora_scale = lora_scale
        # Diagnostic: log the adapter's key format so a misformatted LoRA (e.g.
        # an LTX-native file run through the Comfy rename map, which silently
        # no-ops) is obvious in the backend log. Reads only the safetensors
        # header; never raises.
        from services.lora_diagnostics import log_lora_diagnostics

        log_lora_diagnostics(
            lora_path=lora_path,
            label="distilled t2v",
            base_checkpoint_path=checkpoint_path,
            applying_comfy_rename=True,
        )
        loras: list[LoraPathStrengthAndSDOps] = []
        if lora_path:
            loras = [
                LoraPathStrengthAndSDOps(
                    path=lora_path,
                    strength=lora_scale,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            gemma_root=cast(str, gemma_root),
            spatial_upsampler_path=upsampler_path,
            loras=loras,
            device=device,
            quantization=self._quantization,
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
        tiling_config: TilingConfigType,
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
            tiling_config=tiling_config,
            streaming_prefetch_count=self._streaming_prefetch_count,
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
        output_path: str,
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
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()

        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_pipelines.distilled import DistilledPipeline

        loras: list[LoraPathStrengthAndSDOps] = []
        if self._lora_path:
            loras = [
                LoraPathStrengthAndSDOps(
                    path=self._lora_path,
                    strength=self._lora_scale,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]
        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=self._checkpoint_path,
            gemma_root=cast(str, self._gemma_root),
            spatial_upsampler_path=self._upsampler_path,
            loras=loras,
            device=self._device,
            quantization=self._quantization,
            torch_compile=True,
        )
