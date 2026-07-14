"""Tests for the GPU-telemetry + validation-feed execution seams.

Covers the parsing helpers shared by `LocalTrainerTarget` and
`RunPodTrainerTarget` (so the fixed-`nvidia-smi` and `ls samples/` parsing is
verified once, independent of the WSL/SSH transports) plus the fake's
deterministic behavior used by the runner tests.
"""

from __future__ import annotations

import pytest

from services.trainer_target.trainer_target import (
    GpuTelemetry,
    TrainerCredentials,
    TrainerTargetError,
    ValidationArtifact,
    NVIDIA_SMI_GPU_QUERY,
    parse_nvidia_smi_output,
    parse_samples_listing,
    samples_dir_for,
)
from state.lora_training_state import TargetHandle
from tests.fakes.services import FakeTrainerTarget


class TestParseNvidiaSmi:
    def test_parses_full_csv_line(self) -> None:
        out = "NVIDIA RTX 5090, 32510, 12345, 87, 38, 65\n"
        tel = parse_nvidia_smi_output(out)
        assert tel == GpuTelemetry(
            name="NVIDIA RTX 5090",
            vram_total_mb=32510,
            vram_used_mb=12345,
            gpu_util_pct=87,
            mem_util_pct=38,
            temp_c=65,
        )

    def test_ignores_blank_lines_and_uses_first(self) -> None:
        out = "\n\n  NVIDIA A100, 81920, 1024, 0, 1, 33  \nNVIDIA A100, 0, 0, 0, 0, 0\n"
        tel = parse_nvidia_smi_output(out)
        assert tel.name == "NVIDIA A100"
        assert tel.vram_total_mb == 81920
        assert tel.temp_c == 33

    def test_empty_output_raises(self) -> None:
        with pytest.raises(TrainerTargetError):
            parse_nvidia_smi_output("")

    def test_whitespace_only_output_raises(self) -> None:
        with pytest.raises(TrainerTargetError):
            parse_nvidia_smi_output("\n   \n")

    def test_unparseable_numeric_fields_fall_to_zero_not_raise(self) -> None:
        # Name present but numeric fields garbled — degraded snapshot, not a crash.
        tel = parse_nvidia_smi_output("Some GPU, oops, nope, x, y, z")
        assert tel.name == "Some GPU"
        assert tel.vram_total_mb == 0
        assert tel.vram_used_mb == 0
        assert tel.gpu_util_pct == 0
        assert tel.mem_util_pct == 0
        assert tel.temp_c is None

    def test_na_temperature_becomes_none(self) -> None:
        tel = parse_nvidia_smi_output("GPU, 8192, 1024, 50, 12, [N/A]")
        assert tel.temp_c is None
        assert tel.vram_total_mb == 8192

    def test_query_constant_is_fixed_argv_no_user_input(self) -> None:
        # The query must be a literal constant (no f-string / interpolation site)
        # so no caller-controlled value can ever reach the nvidia-smi shell line.
        assert "nvidia-smi --query-gpu=" in NVIDIA_SMI_GPU_QUERY
        assert "--format=csv,noheader,nounits" in NVIDIA_SMI_GPU_QUERY


class TestParseSamplesListing:
    def test_parses_step_and_index_and_ext(self) -> None:
        out = "step_000050_1.mp4\nstep_000050_2.mp4\nstep_000100_1.png\n"
        arts = parse_samples_listing(out, "/run/out/samples")
        assert arts == [
            ValidationArtifact(50, 1, "/run/out/samples/step_000050_1.mp4", "mp4"),
            ValidationArtifact(50, 2, "/run/out/samples/step_000050_2.mp4", "mp4"),
            ValidationArtifact(100, 1, "/run/out/samples/step_000100_1.png", "png"),
        ]

    def test_sorted_by_step_then_index(self) -> None:
        out = "step_000150_2.mp4\nstep_000150_1.mp4\nstep_000050_1.mp4\n"
        arts = parse_samples_listing(out, "/o/samples")
        assert [a.step for a in arts] == [50, 150, 150]
        assert arts[1].sample_index == 1
        assert arts[2].sample_index == 2

    def test_ignores_non_matching_filenames(self) -> None:
        out = ".\n..\nconfig.yaml\nstep_000050_1.mp4\nfinal_video.mp4\n"
        arts = parse_samples_listing(out, "/o/samples")
        assert len(arts) == 1
        assert arts[0].step == 50

    def test_empty_listing_returns_empty(self) -> None:
        assert parse_samples_listing("", "/o/samples") == []

    def test_supports_wav_extension(self) -> None:
        arts = parse_samples_listing("step_000075_1.wav\n", "/o/samples")
        assert arts[0].ext == "wav"
        assert arts[0].remote_path == "/o/samples/step_000075_1.wav"

    def test_trailing_slash_on_dir_normalized(self) -> None:
        arts = parse_samples_listing("step_000010_1.mp4\n", "/o/samples/")
        assert arts[0].remote_path == "/o/samples/step_000010_1.mp4"


class TestSamplesDirFor:
    def test_appends_samples(self) -> None:
        assert samples_dir_for("/run/out") == "/run/out/samples"

    def test_normalizes_trailing_slash(self) -> None:
        assert samples_dir_for("/run/out/") == "/run/out/samples"


class TestFakeTrainerTargetTelemetry:
    def test_query_gpu_returns_configured_telemetry_and_counts(self) -> None:
        fake = FakeTrainerTarget()
        tel = fake.query_gpu(credentials=_creds(), handle=_handle())
        assert tel is fake.gpu_telemetry
        assert fake.query_gpu_calls == 1

    def test_query_gpu_propagates_configured_error(self) -> None:
        fake = FakeTrainerTarget()
        fake.raise_on_query_gpu = TrainerTargetError("boom", retryable=True)
        with pytest.raises(TrainerTargetError):
            fake.query_gpu(credentials=_creds(), handle=_handle())

    def test_list_validation_filters_since_step(self) -> None:
        fake = FakeTrainerTarget()
        fake.validation_artifacts = [
            ValidationArtifact(50, 1, "/o/samples/step_000050_1.mp4", "mp4"),
            ValidationArtifact(100, 1, "/o/samples/step_000100_1.mp4", "mp4"),
            ValidationArtifact(100, 2, "/o/samples/step_000100_2.mp4", "mp4"),
            ValidationArtifact(150, 1, "/o/samples/step_000150_1.mp4", "mp4"),
        ]
        out = fake.list_validation_outputs(
            credentials=_creds(),
            handle=_handle(),
            remote_output_dir="/o",
            since_step=50,
        )
        assert [a.step for a in out] == [100, 100, 150]
        # Records the call args so runner tests can assert cadence.
        assert fake.list_validation_calls == [("/o", 50)]

    def test_list_validation_since_zero_returns_all(self) -> None:
        fake = FakeTrainerTarget()
        fake.validation_artifacts = [
            ValidationArtifact(50, 1, "/o/samples/step_000050_1.mp4", "mp4"),
        ]
        out = fake.list_validation_outputs(
            credentials=_creds(), handle=_handle(), remote_output_dir="/o", since_step=0
        )
        assert len(out) == 1


def _creds() -> TrainerCredentials:
    return TrainerCredentials(
        provider="runpod", workspace_dir="/w", model_path="/m", text_encoder_path="/t"
    )


def _handle() -> TargetHandle:
    return TargetHandle(provider="runpod", pod_id="fake-pod-1")
