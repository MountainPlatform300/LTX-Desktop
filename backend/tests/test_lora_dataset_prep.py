"""Unit tests for the training-ready IC-LoRA dataset-prep pipeline.

Covers pairing (pairs-only, drop input-only rows), caption validation
(empty/truncated/trigger/forbidden), per-pair consistency on the normalized
outputs (fps/WxH/frame-count/8k+1/bucket), the `{caption, video,
reference_video}` schema, and the drop report. Media work is exercised through
the `FakeClipProcessor`, whose per-path probe overrides let us simulate the
normalized outputs without ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

from handlers import lora_dataset_prep as prep
def test_options_for_resolution_buckets_cover_every_bucket() -> None:
    options = prep.options_for_resolution_buckets(
        "768x448x49;512x512x81", trigger_word="TOK"
    )
    assert options.short_side == 512
    assert options.bucket_frames == 81
    assert options.trigger_word == "TOK"


from services.clip_processor.clip_processor import ClipProbeResult
from state.lora_training_state import LoraClip, LoraDataset
from tests.fakes.services import FakeClipProcessor


def _clip(path: str, *, caption: str = "A clean-shaven man smiles.", ref: str | None = None) -> LoraClip:
    return LoraClip(
        id=path,
        local_path=path,
        caption=caption,
        reference_path=ref,
        reference_paths=[ref] if ref else [],
    )


def _dataset(clips: list[LoraClip], *, trigger: str | None = None) -> LoraDataset:
    return LoraDataset(
        id="ds1",
        name="Beards",
        created_at="2024-01-01T00:00:00Z",
        status="draft",
        type="ic_lora",
        trigger_word=trigger,
        clips=clips,
    )


def _good_probe(frames: int = 49) -> ClipProbeResult:
    return ClipProbeResult(
        duration_seconds=2.0, width=1024, height=576, fps=25.0,
        frame_count=frames, has_audio=False, video_codec="h264",
    )


def _make_files(tmp_path: Path, *names: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"\x00media")
        out[n] = str(p)
    return out


class TestPure:
    def test_is_8k_plus_1(self) -> None:
        assert [n for n in range(1, 60) if prep.is_8k_plus_1(n)] == [1, 9, 17, 25, 33, 41, 49, 57]

    def test_caption_empty(self) -> None:
        assert prep.caption_problem("  ", trigger_word=None, forbidden_words=()) == "empty caption"

    def test_caption_truncated(self) -> None:
        problem = prep.caption_problem("A man stands near foliage in", trigger_word=None, forbidden_words=())
        assert problem is not None and "truncated" in problem

    def test_caption_truncated_single_dangling_word(self) -> None:
        # A bare 1-2 word ending on a function word is still a truncation signal.
        problem = prep.caption_problem("foliage in", trigger_word=None, forbidden_words=())
        assert problem is not None and "truncated" in problem

    def test_caption_without_trailing_punctuation_accepted(self) -> None:
        # Relaxed: a multi-word caption ending on a content word (no period) is
        # a normal stylistic choice, not a truncation.
        assert prep.caption_problem("A clean-shaven man smiles", trigger_word=None, forbidden_words=()) is None
        assert prep.caption_problem("Man smiling at camera", trigger_word=None, forbidden_words=()) is None

    def test_caption_trigger_word_rejected(self) -> None:
        problem = prep.caption_problem("A TOK man smiles.", trigger_word="TOK", forbidden_words=())
        assert problem is not None and "trigger word" in problem

    def test_caption_forbidden_word_rejected(self) -> None:
        problem = prep.caption_problem("A man with a beard.", trigger_word=None, forbidden_words=("beard",))
        assert problem is not None and "forbidden word" in problem

    def test_caption_ok(self) -> None:
        assert prep.caption_problem("A clean-shaven man smiles.", trigger_word="TOK", forbidden_words=("beard",)) is None


class TestPairing:
    def test_drops_unpaired_keeps_pairs(self, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "in.mp4", "out.mp4", "lonely.mp4")
        target = _clip(files["out.mp4"], ref=files["in.mp4"])
        reference = _clip(files["in.mp4"], caption="bearded")
        lonely = _clip(files["lonely.mp4"], caption="no pair")
        ds = _dataset([target, reference, lonely])

        pairs, drops = prep.collect_pairs(ds, ds.clips)
        assert len(pairs) == 1
        assert pairs[0].target.local_path == files["out.mp4"]
        assert pairs[0].reference.local_path == files["in.mp4"]
        # The reference clip is NOT separately dropped; only the truly lonely one.
        reasons = {d.name: d.reason for d in drops}
        assert "lonely.mp4" in reasons
        assert "in.mp4" not in reasons

    def test_missing_reference_file_drops_pair(self, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "out.mp4")
        target = _clip(files["out.mp4"], ref=str(tmp_path / "gone.mp4"))
        ds = _dataset([target])
        pairs, drops = prep.collect_pairs(ds, ds.clips)
        assert pairs == []
        # No resolvable reference at all → treated as unpaired.
        assert drops and "unpaired" in drops[0].reason


class TestPrepareBundle:
    def _setup(self, tmp_path: Path, *, caption: str = "A clean-shaven man smiles.", trigger=None):
        files = _make_files(tmp_path, "in.mp4", "out.mp4")
        target = _clip(files["out.mp4"], caption=caption, ref=files["in.mp4"])
        reference = _clip(files["in.mp4"], caption="bearded")
        ds = _dataset([target, reference], trigger=trigger)
        proc = FakeClipProcessor()
        return files, ds, proc

    def test_emits_pair_schema(self, tmp_path: Path) -> None:
        files, ds, proc = self._setup(tmp_path)
        proc.result = _good_probe(49)
        staging = tmp_path / "stage"
        report = prep.prepare_ic_lora_bundle(
            dataset=ds, clips=ds.clips, staging_dir=staging,
            processor=proc, options=prep.PrepOptions(), render_media=lambda r: r,
        )
        assert report.exported == 1 and report.dropped == []
        import json
        rows = json.loads((staging / "dataset.json").read_text())
        assert len(rows) == 1
        row = rows[0]
        assert set(row.keys()) == {"caption", "video", "reference_video"}
        assert row["video"].startswith("clips/0001_output_")
        assert row["reference_video"].startswith("clips/0001_reference_")
        assert (staging / row["video"]).is_file()
        assert (staging / row["reference_video"]).is_file()
        # Both clips normalized with the dataset-wide knobs.
        assert len(proc.normalize_calls) == 2
        assert all(c["fps"] == 25.0 and c["frames"] == 49 for c in proc.normalize_calls)

    def test_image_reference_is_exported_as_video(self, tmp_path: Path) -> None:
        # An image input (e.g. a still used as the IC-LoRA reference) must be
        # shipped, not dropped: it's normalized into a full-length .mp4 matched
        # to the target's geometry so the trainer reads it as a paired video.
        files = _make_files(tmp_path, "clouds.png", "out.mp4")
        target = _clip(files["out.mp4"], ref=files["clouds.png"])
        reference = _clip(files["clouds.png"], caption="clouds")
        ds = _dataset([target, reference])
        proc = FakeClipProcessor()
        proc.result = _good_probe(49)
        staging = tmp_path / "stage"

        report = prep.prepare_ic_lora_bundle(
            dataset=ds, clips=ds.clips, staging_dir=staging,
            processor=proc, options=prep.PrepOptions(), render_media=lambda r: r,
        )

        assert report.exported == 1 and report.dropped == []
        import json
        rows = json.loads((staging / "dataset.json").read_text())
        ref = rows[0]["reference_video"]
        assert ref.endswith(".mp4") and "_reference_" in ref
        assert (staging / ref).is_file()
        # The image was scaled to the target's exact dims (single_dims path),
        # not aligned via the video pair_dims path.
        img_call = next(c for c in proc.normalize_calls if c["source_path"] == files["clouds.png"])
        assert img_call["exact_width"] == 1024 and img_call["exact_height"] == 576
        assert img_call["frames"] == 49

    def test_drops_fps_mismatch_within_pair(self, tmp_path: Path) -> None:
        files, ds, proc = self._setup(tmp_path)
        staging = tmp_path / "stage"
        # The reference output comes back at 30 fps (issue #3).
        proc.result = _good_probe(49)
        proc.results_by_path[str(staging / ".prep_tmp" / "r.mp4")] = ClipProbeResult(
            duration_seconds=2.0, width=1024, height=576, fps=30.0,
            frame_count=49, has_audio=False, video_codec="h264",
        )
        report = prep.prepare_ic_lora_bundle(
            dataset=ds, clips=ds.clips, staging_dir=staging,
            processor=proc, options=prep.PrepOptions(), render_media=lambda r: r,
        )
        assert report.exported == 0
        assert report.dropped and "fps mismatch" in report.dropped[0].reason

    def test_drops_too_few_frames(self, tmp_path: Path) -> None:
        files, ds, proc = self._setup(tmp_path)
        proc.result = _good_probe(33)  # both clips only reach 33 frames < 49
        staging = tmp_path / "stage"
        report = prep.prepare_ic_lora_bundle(
            dataset=ds, clips=ds.clips, staging_dir=staging,
            processor=proc, options=prep.PrepOptions(bucket_frames=49), render_media=lambda r: r,
        )
        assert report.exported == 0
        assert report.dropped and "frames" in report.dropped[0].reason

    def test_drops_truncated_caption(self, tmp_path: Path) -> None:
        files, ds, proc = self._setup(tmp_path, caption="A man stands near foliage in")
        proc.result = _good_probe(49)
        staging = tmp_path / "stage"
        report = prep.prepare_ic_lora_bundle(
            dataset=ds, clips=ds.clips, staging_dir=staging,
            processor=proc, options=prep.PrepOptions(), render_media=lambda r: r,
        )
        assert report.exported == 0
        assert report.dropped and "truncated" in report.dropped[0].reason

    def test_remote_render_media_makes_absolute_paths(self, tmp_path: Path) -> None:
        files, ds, proc = self._setup(tmp_path)
        proc.result = _good_probe(49)
        staging = tmp_path / "stage"
        report = prep.prepare_ic_lora_bundle(
            dataset=ds, clips=ds.clips, staging_dir=staging,
            processor=proc, options=prep.PrepOptions(),
            render_media=lambda r: f"/workspace/ds/{r}",
        )
        assert report.exported == 1
        import json
        rows = json.loads((staging / "dataset.json").read_text())
        assert rows[0]["video"].startswith("/workspace/ds/clips/0001_output_")


class TestStageHoldoutReferences:
    """Held-out clips' reference videos are staged for the IC-LoRA validation feed."""

    def test_stages_ic_lora_holdout_reference(self, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "in.mp4", "out.mp4", "kept_in.mp4", "kept_out.mp4")
        holdout_target = _clip(files["out.mp4"], caption="A clean-shaven man smiles.", ref=files["in.mp4"])
        holdout_target.triage = "holdout"
        holdout_ref = _clip(files["in.mp4"], caption="bearded")
        kept_target = _clip(files["kept_out.mp4"], ref=files["kept_in.mp4"])
        kept_ref = _clip(files["kept_in.mp4"], caption="bearded")
        ds = _dataset([holdout_target, holdout_ref, kept_target, kept_ref])
        proc = FakeClipProcessor()
        proc.result = _good_probe(49)
        staging = tmp_path / "stage"
        (staging / "clips").mkdir(parents=True, exist_ok=True)

        report = prep.stage_holdout_references(
            dataset=ds, staging_dir=staging, processor=proc, options=prep.PrepOptions()
        )

        # Only the holdout target's reference is staged, under holdout/{id}.mp4.
        assert report.staged == [holdout_target.id]
        assert (staging / "holdout" / f"{holdout_target.id}.mp4").is_file()
        # The kept clip's reference is NOT staged here (it ships via the bundle).
        assert not (staging / "holdout" / f"{kept_target.id}.mp4").exists()
        # The reference (input) video was normalized, not the target.
        assert any(c["source_path"] == files["in.mp4"] for c in proc.normalize_calls)

    def test_t2v_holdout_without_reference_is_noop(self, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "clip.mp4")
        holdout = _clip(files["clip.mp4"], caption="A cat jumps.")
        holdout.triage = "holdout"
        ds = LoraDataset(
            id="ds1", name="Cats", created_at="2024-01-01T00:00:00Z",
            status="draft", type="standard", clips=[holdout],
        )
        proc = FakeClipProcessor()
        staging = tmp_path / "stage"
        (staging / "clips").mkdir(parents=True, exist_ok=True)

        report = prep.stage_holdout_references(
            dataset=ds, staging_dir=staging, processor=proc, options=prep.PrepOptions()
        )

        # t2v holdout has no reference — nothing staged, nothing dropped.
        assert report.staged == [] and report.dropped == []
        assert not (staging / "holdout").exists()
        assert proc.normalize_calls == []

    def test_drops_holdout_when_reference_file_missing(self, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "out.mp4")
        # The reference clip exists in the dataset but its file is absent on disk.
        missing_ref = str(tmp_path / "gone.mp4")
        holdout = _clip(files["out.mp4"], caption="A clean-shaven man smiles.", ref=missing_ref)
        holdout.triage = "holdout"
        reference = _clip(missing_ref, caption="bearded")
        ds = _dataset([holdout, reference])
        proc = FakeClipProcessor()
        staging = tmp_path / "stage"
        (staging / "clips").mkdir(parents=True, exist_ok=True)

        report = prep.stage_holdout_references(
            dataset=ds, staging_dir=staging, processor=proc, options=prep.PrepOptions()
        )

        assert report.staged == []
        assert report.dropped and "missing" in report.dropped[0].reason

    def test_auto_picks_training_clip_when_no_holdout(self, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "in.mp4", "out.mp4")
        target = _clip(files["out.mp4"], caption="A clean-shaven man smiles.", ref=files["in.mp4"])
        reference = _clip(files["in.mp4"], caption="bearded")
        ds = _dataset([target, reference])  # no holdout clips
        proc = FakeClipProcessor()
        proc.result = _good_probe(49)
        staging = tmp_path / "stage"
        (staging / "clips").mkdir(parents=True, exist_ok=True)

        report = prep.stage_holdout_references(
            dataset=ds, staging_dir=staging, processor=proc,
            options=prep.PrepOptions(), auto_pick_when_empty=True,
        )

        # No curated holdout -> the first training clip's reference is auto-
        # staged so IC-LoRA still gets a validation feed.
        assert report.auto_picked == target.id
        assert report.staged == [target.id]
        assert (staging / "holdout" / f"{target.id}.mp4").is_file()
        assert any(c["source_path"] == files["in.mp4"] for c in proc.normalize_calls)

    def test_no_auto_pick_when_holdout_present(self, tmp_path: Path) -> None:
        files = _make_files(tmp_path, "in.mp4", "out.mp4", "kept_out.mp4")
        holdout = _clip(files["out.mp4"], caption="held out", ref=files["in.mp4"])
        holdout.triage = "holdout"
        holdout_ref = _clip(files["in.mp4"], caption="bearded")
        kept = _clip(files["kept_out.mp4"], caption="kept", ref=files["in.mp4"])
        ds = _dataset([holdout, holdout_ref, kept])
        proc = FakeClipProcessor()
        proc.result = _good_probe(49)
        staging = tmp_path / "stage"
        (staging / "clips").mkdir(parents=True, exist_ok=True)

        report = prep.stage_holdout_references(
            dataset=ds, staging_dir=staging, processor=proc,
            options=prep.PrepOptions(), auto_pick_when_empty=True,
        )

        # A curated holdout wins; no auto-pick fallback.
        assert report.auto_picked is None
        assert report.staged == [holdout.id]

