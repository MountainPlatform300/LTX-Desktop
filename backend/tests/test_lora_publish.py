"""Unit + integration tests for the LoRA publication (model-card) flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from handlers import lora_publish
from handlers.lora_export import BundleError
from state.lora_training_state import (
    LoraClip,
    LoraDataset,
    PreprocessedDataset,
    TrainingConfig,
    TrainingJob,
)


# --- Fixtures-as-builders -------------------------------------------------


def _dataset(*, dtype: str = "standard", trigger: str | None = "MYTOK") -> LoraDataset:
    return LoraDataset(
        id="ds1",
        name="My LoRA",
        created_at="2026-01-01T00:00:00Z",
        status="uploaded",
        type=dtype,  # type: ignore[arg-type]
        trigger_word=trigger,
        clips=[
            LoraClip(id="c1", local_path="/tmp/a.mp4", caption="a cat", duration_seconds=3.0),
            LoraClip(id="c2", local_path="/tmp/b.mp4", caption="a dog", duration_seconds=4.0),
        ],
    )


def _preprocessed() -> PreprocessedDataset:
    return PreprocessedDataset(
        id="pre1",
        dataset_id="ds1",
        created_at="2026-01-01T00:00:00Z",
        status="ready",
        resolution_buckets="768x448x89",
    )


def _job() -> TrainingJob:
    return TrainingJob(
        id="job1",
        preprocessed_id="pre1",
        name="My LoRA run",
        created_at="2026-01-01T00:00:00Z",
        status="completed",
        config=TrainingConfig(rank=64, alpha=48, steps=1500, learning_rate=2e-4),
        provider="runpod",
    )


# --- Pure card generation -------------------------------------------------


def test_suggest_meta_prefills_from_run() -> None:
    meta = lora_publish.suggest_meta(_job(), _dataset())
    assert meta.title == "My LoRA"
    assert "MYTOK" in meta.summary
    assert "lora" in meta.tags and "ltx" in meta.tags
    assert "MYTOK" in meta.tags  # trigger word becomes a tag


def test_huggingface_card_has_front_matter_and_recipe() -> None:
    meta = lora_publish.suggest_meta(_job(), _dataset())
    card = lora_publish.build_model_card(
        platform="huggingface",
        job=_job(),
        preprocessed=_preprocessed(),
        dataset=_dataset(),
        examples=[],
        meta=meta,
    )
    assert card.startswith("---\n")
    assert "base_model: Lightricks/LTX-Video" in card
    assert "pipeline_tag: text-to-video" in card
    assert "## Trigger word" in card and "`MYTOK`" in card
    # Recipe table reflects the snapshotted config.
    assert "| LoRA rank / alpha | 64 / 48 |" in card
    assert "| Resolution buckets | 768x448x89 |" in card
    assert "## Intended use & limitations" in card


def test_portable_card_has_no_front_matter() -> None:
    meta = lora_publish.suggest_meta(_job(), _dataset())
    card = lora_publish.build_model_card(
        platform="portable",
        job=_job(),
        preprocessed=_preprocessed(),
        dataset=_dataset(),
        examples=[],
        meta=meta,
    )
    assert not card.startswith("---")
    assert card.startswith("# My LoRA")


def test_card_without_trigger_word_reads_naturally() -> None:
    ds = _dataset(trigger=None)
    meta = lora_publish.suggest_meta(_job(), ds)
    card = lora_publish.build_model_card(
        platform="portable",
        job=_job(),
        preprocessed=_preprocessed(),
        dataset=ds,
        examples=[],
        meta=meta,
    )
    assert "no dedicated trigger word" in card


def test_civitai_metadata_has_trained_words_and_recipe() -> None:
    meta = lora_publish.suggest_meta(_job(), _dataset())
    data = lora_publish.civitai_metadata(
        job=_job(), preprocessed=_preprocessed(), dataset=_dataset(), meta=meta
    )
    assert data["type"] == "LORA"
    assert data["trainedWords"] == ["MYTOK"]
    assert data["trainingDetails"]["LoRA rank / alpha"] == "64 / 48"


# --- Bundle ---------------------------------------------------------------


def _make_media(tmp_path: Path, name: str, data: bytes = b"\x00\x01") -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_build_bundle_writes_cards_examples_and_weights(tmp_path: Path) -> None:
    video = _make_media(tmp_path, "walk.mp4")
    image = _make_media(tmp_path, "still.png")
    weights = tmp_path / "lora_weights.safetensors"
    weights.write_bytes(b"\x00" * 16)
    meta = lora_publish.suggest_meta(_job(), _dataset())
    staging = tmp_path / "out"

    manifest = lora_publish.build_publication_bundle(
        platforms=["huggingface", "civitai", "portable"],
        job=_job(),
        preprocessed=_preprocessed(),
        dataset=_dataset(),
        examples=[
            lora_publish.PublicationExample(media_path=video, caption="a walking cat"),
            lora_publish.PublicationExample(media_path=image, caption="a sitting cat"),
        ],
        meta=meta,
        lora_path=str(weights),
        staging_dir=staging,
    )

    # All chosen cards + metadata + manifest exist.
    for name in ("README.md", "MODEL_CARD.md", "civitai_description.md", "civitai.json", "publication.json"):
        assert (staging / name).is_file(), name
    # Examples copied under stable names; weights copied beside the cards.
    examples = sorted(p.name for p in (staging / "examples").iterdir())
    assert examples == ["0001_walk.mp4", "0002_still.png"]
    assert (staging / "My_LoRA.safetensors").is_file()

    # The HF card renders a video tag + an image embed + the captions.
    readme = (staging / "README.md").read_text()
    assert "<video" in readme and "examples/0001_walk.mp4" in readme
    assert "![example](examples/0002_still.png)" in readme
    assert "a walking cat" in readme

    assert manifest["exampleCount"] == 2
    assert manifest["weightsFile"] == "My_LoRA.safetensors"
    assert "README.md" in manifest["files"]


def test_build_bundle_without_weights_omits_them(tmp_path: Path) -> None:
    meta = lora_publish.suggest_meta(_job(), _dataset())
    staging = tmp_path / "out"
    manifest = lora_publish.build_publication_bundle(
        platforms=["portable"],
        job=_job(),
        preprocessed=_preprocessed(),
        dataset=_dataset(),
        examples=[],
        meta=meta,
        lora_path=None,
        staging_dir=staging,
    )
    assert manifest["weightsFile"] is None
    assert not any(p.suffix == ".safetensors" for p in staging.iterdir())


def test_build_bundle_missing_example_raises(tmp_path: Path) -> None:
    meta = lora_publish.suggest_meta(_job(), _dataset())
    with pytest.raises(BundleError):
        lora_publish.build_publication_bundle(
            platforms=["portable"],
            job=_job(),
            preprocessed=_preprocessed(),
            dataset=_dataset(),
            examples=[lora_publish.PublicationExample(media_path=str(tmp_path / "nope.mp4"))],
            meta=meta,
            lora_path=None,
            staging_dir=tmp_path / "out",
        )


def test_build_bundle_requires_a_platform(tmp_path: Path) -> None:
    meta = lora_publish.suggest_meta(_job(), _dataset())
    with pytest.raises(BundleError):
        lora_publish.build_publication_bundle(
            platforms=[],
            job=_job(),
            preprocessed=_preprocessed(),
            dataset=_dataset(),
            examples=[],
            meta=meta,
            lora_path=None,
            staging_dir=tmp_path / "out",
        )


# --- Routes ---------------------------------------------------------------


def _inject_completed_run(handler, tmp_path: Path, *, status: str = "completed") -> tuple[str, str]:
    """Inject a dataset/preprocessed/run already in terminal states straight
    into the ledgers. The background reconciler (started by the ``client``
    fixture) only acts on in-flight states, so this keeps route tests
    deterministic — unlike driving a fresh run, which briefly sits in
    ``pending`` and races the reconciler thread. Returns (training_id, clip)."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x01")
    weights = tmp_path / "lora_weights.safetensors"
    weights.write_bytes(b"\x00" * 8)
    ds = LoraDataset(
        id="ds-route",
        name="My LoRA",
        created_at="2026-01-01T00:00:00Z",
        status="uploaded",
        trigger_word="MYTOK",
        remote_dataset_dir="/w/datasets/x",
        clips=[LoraClip(id="c1", local_path=str(clip), caption="a cat", duration_seconds=3.0)],
    )
    pre = PreprocessedDataset(
        id="pre-route",
        dataset_id=ds.id,
        created_at="2026-01-01T00:00:00Z",
        status="ready",
        resolution_buckets="768x448x89",
        remote_precomputed_dir="/w/.precomputed",
    )
    job = TrainingJob(
        id="job-route",
        preprocessed_id=pre.id,
        name="My LoRA run",
        created_at="2026-01-01T00:00:00Z",
        status=status,  # type: ignore[arg-type]
        config=TrainingConfig(rank=64),
        provider="runpod",
        local_lora_path=str(weights) if status == "completed" else None,
    )
    handler._datasets.datasets.append(ds)
    handler._preprocessed.items.append(pre)
    handler._training.items.append(job)
    return job.id, str(clip)


class TestPublishRoutes:
    def test_preview_returns_suggested_meta_and_cards(self, client, test_state, tmp_path) -> None:
        job_id, _ = _inject_completed_run(test_state.lora_training, tmp_path)
        r = client.post(
            f"/api/lora/training/{job_id}/publish/preview",
            json={"platforms": ["huggingface", "portable"]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["meta"]["title"] == "My LoRA"
        assert "MYTOK" in body["meta"]["summary"]
        assert set(body["cards"]) == {"huggingface", "portable"}
        assert "## Trigger word" in body["cards"]["huggingface"]
        assert body["cards"]["huggingface"].startswith("---\n")

    def test_export_writes_bundle(self, client, test_state, tmp_path) -> None:
        job_id, clip_path = _inject_completed_run(test_state.lora_training, tmp_path)
        dest = tmp_path / "publications"
        dest.mkdir()
        r = client.post(
            f"/api/lora/training/{job_id}/publish/export",
            json={
                "destPath": str(dest),
                "platforms": ["huggingface", "civitai"],
                "meta": {"title": "My LoRA", "summary": "great", "tags": ["lora", "ltx"]},
                "examples": [{"mediaPath": clip_path, "caption": "demo clip"}],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["exampleCount"] == 1
        assert body["weightsFile"] == "My_LoRA.safetensors"
        assert "README.md" in body["files"]
        out = Path(body["publicationPath"])
        assert (out / "README.md").is_file()
        readme = (out / "README.md").read_text()
        assert "demo clip" in readme
        assert json.loads((out / "civitai.json").read_text())["trainedWords"] == ["MYTOK"]

    def test_preview_rejects_unfinished_run(self, client, test_state, tmp_path) -> None:
        # A run that hasn't produced weights yet can't be published.
        job_id, _ = _inject_completed_run(test_state.lora_training, tmp_path, status="failed")
        r = client.post(
            f"/api/lora/training/{job_id}/publish/preview", json={"platforms": ["portable"]}
        )
        assert r.status_code == 400, r.text

    def test_preview_unknown_training_404(self, client) -> None:
        r = client.post("/api/lora/training/nope/publish/preview", json={"platforms": ["portable"]})
        assert r.status_code == 404, r.text
