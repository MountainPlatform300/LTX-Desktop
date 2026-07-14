"""Tests for the per-LoRA Gemini auto-prompt assistant.

Covers the three pieces:
  * `lora_prompt_template` default synthesis from explicit behavior metadata
  * the registry overlaying the (default or overridden) template on entries
  * the `auto-prompt` and `prompt-template` HTTP routes on `lora_inference`
"""

from __future__ import annotations

from pathlib import Path

from handlers.lora_prompt_template import build_default_prompt_template
from state.lora_training_state import (
    LoraClip,
    LoraDataset,
    PreprocessedDataset,
    TrainingConfig,
    TrainingJob,
)


def _inject_completed_run(
    handler,
    tmp_path: Path,
    *,
    job_id: str,
    dataset_type: str,
    name: str | None = None,
    description: str | None = None,
) -> TrainingJob:
    clip = tmp_path / f"{job_id}-clip.mp4"
    clip.write_bytes(b"\x00\x01")
    weights = tmp_path / f"{job_id}.safetensors"
    weights.write_bytes(b"\x00" * 8)
    ds = LoraDataset(
        id=f"ds-{job_id}",
        name=f"Dataset {job_id}",
        created_at="2026-01-01T00:00:00Z",
        status="uploaded",
        trigger_word="MYTOK",
        remote_dataset_dir=f"/w/datasets/{job_id}",
        type=dataset_type,  # type: ignore[arg-type]
        clips=[LoraClip(id="c1", local_path=str(clip), caption="a cat", duration_seconds=3.0)],
    )
    pre = PreprocessedDataset(
        id=f"pre-{job_id}",
        dataset_id=ds.id,
        created_at="2026-01-01T00:00:00Z",
        status="ready",
        resolution_buckets="768x448x89",
        remote_precomputed_dir=f"/w/.precomputed-{job_id}",
    )
    job = TrainingJob(
        id=job_id,
        preprocessed_id=pre.id,
        name=name or f"Run {job_id}",
        description=description,
        created_at="2026-01-01T00:00:00Z",
        status="completed",
        config=TrainingConfig(rank=64, trigger_word=ds.trigger_word),
        provider="runpod",
        local_lora_path=str(weights),
    )
    handler._datasets.datasets.append(ds)
    handler._preprocessed.items.append(pre)
    handler._training.items.append(job)
    return job


# ---------------------------------------------------------------------
# Default synthesis
# ---------------------------------------------------------------------


def test_ic_lora_default_template_starts_with_trigger() -> None:
    template = build_default_prompt_template(
        description="Changes the subject into a cone-headed character",
        variant="video_input_ic_lora",
        trigger_word="conehead",
    )
    assert template is not None
    # The trigger must appear as the first-token instruction.
    assert 'verified training trigger is "conehead"' in template
    assert "Do not assume the LoRA preserves a person" in template
    assert "identity remain unchanged" not in template


def test_clean_plate_template_removes_subject_without_preserving_identity() -> None:
    template = build_default_prompt_template(
        description="Removes foreground subjects and reconstructs the hidden background",
        variant="video_input_ic_lora",
        trigger_word="cleanplate",
    )
    assert template is not None
    assert "foreground elements have been removed" in template
    assert "occluded background has been reconstructed" in template
    assert "Do not describe or preserve the removed subject's identity" in template
    assert "identity remain unchanged" not in template


def test_standard_variant_has_no_default_template() -> None:
    assert (
        build_default_prompt_template(
            description="Applies a painterly style", variant="standard", trigger_word="style"
        )
        is None
    )


def test_union_control_default_template_mentions_conditioning() -> None:
    template = build_default_prompt_template(
        description="Preserves structure while changing the visual style",
        variant="union_control",
        trigger_word="union",
        conditioning_types=("canny", "depth", "pose"),
    )
    assert template is not None
    assert "canny / depth / pose" in template


def test_triggerless_template_never_invents_trigger_from_name() -> None:
    template = build_default_prompt_template(
        description=None, variant="video_input_ic_lora", trigger_word=None
    )
    assert template is not None
    assert "Do not invent" in template
    assert "<trigger>" not in template


# ---------------------------------------------------------------------
# Registry overlay
# ---------------------------------------------------------------------


def test_registry_overlays_default_template_on_ic_lora(test_state, tmp_path) -> None:
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="ic-tmpl", dataset_type="ic_lora", name="Conehead"
    )
    entries = test_state.lora_inference_registry.list_entries()
    entry = next(e for e in entries if e.id == "user-ic-tmpl")
    assert entry.promptTemplate is not None
    assert entry.triggerWord == "MYTOK"
    assert "MYTOK" in entry.promptTemplate


def test_registry_uses_description_not_display_name_for_behavior(test_state, tmp_path) -> None:
    _inject_completed_run(
        test_state.lora_training,
        tmp_path,
        job_id="description-source",
        dataset_type="ic_lora",
        name="Clean Plate TEST 750",
        description="Adds a neatly trimmed beard to the subject",
    )
    entry = next(
        item
        for item in test_state.lora_inference_registry.list_entries()
        if item.id == "user-description-source"
    )
    assert entry.promptTemplate is not None
    assert "Adds a neatly trimmed beard to the subject" in entry.promptTemplate
    assert "clean-plate IC-LoRA" not in entry.promptTemplate
    assert "Clean Plate TEST 750" not in entry.promptTemplate


def test_registry_leaves_standard_template_none(test_state, tmp_path) -> None:
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="std-tmpl", dataset_type="standard", name="My Style"
    )
    entries = test_state.lora_inference_registry.list_entries()
    entry = next(e for e in entries if e.id == "user-std-tmpl")
    assert entry.promptTemplate is None


def test_user_override_wins_over_default(test_state, tmp_path) -> None:
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="ic-ov", dataset_type="ic_lora", name="Conehead"
    )
    test_state.lora_inference.update_prompt_template(
        lora_id="user-ic-ov",
        prompt_template="CUSTOM SYSTEM PROMPT",
        trigger_word="customtok",
    )
    entries = test_state.lora_inference_registry.list_entries()
    entry = next(e for e in entries if e.id == "user-ic-ov")
    assert entry.promptTemplate == "CUSTOM SYSTEM PROMPT"
    assert entry.promptTemplateCustomized is True
    assert entry.triggerWord == "customtok"


def test_reset_override_falls_back_to_default(test_state, tmp_path) -> None:
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="ic-rs", dataset_type="ic_lora", name="Conehead"
    )
    test_state.lora_inference.update_prompt_template(
        lora_id="user-ic-rs", prompt_template="CUSTOM", trigger_word="customtok"
    )
    test_state.lora_inference.update_prompt_template(
        lora_id="user-ic-rs", prompt_template=None, trigger_word=None
    )
    entries = test_state.lora_inference_registry.list_entries()
    entry = next(e for e in entries if e.id == "user-ic-rs")
    assert entry.promptTemplate is not None
    assert entry.promptTemplate != "CUSTOM"
    assert entry.promptTemplateCustomized is False
    assert entry.triggerWord == "MYTOK"


def test_trigger_only_override_keeps_generated_template(test_state, tmp_path) -> None:
    _inject_completed_run(
        test_state.lora_training,
        tmp_path,
        job_id="ic-trigger-only",
        dataset_type="ic_lora",
        description="Changes the subject into a marble statue",
    )
    test_state.lora_inference.update_prompt_template(
        lora_id="user-ic-trigger-only",
        prompt_template=None,
        trigger_word="MARBLE",
    )
    entry = next(
        item
        for item in test_state.lora_inference_registry.list_entries()
        if item.id == "user-ic-trigger-only"
    )
    assert entry.promptTemplate is not None
    assert "MARBLE" in entry.promptTemplate
    assert entry.promptTemplateCustomized is False


# ---------------------------------------------------------------------
# Auto-prompt route
# ---------------------------------------------------------------------


def test_auto_prompt_requires_gemini_key(test_state, tmp_path, client) -> None:
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="ic-key", dataset_type="ic_lora", name="Conehead"
    )
    video = tmp_path / "ref.mp4"
    video.write_bytes(b"\x00")
    resp = client.post(
        "/api/lora-inference/auto-prompt",
        json={"loraId": "user-ic-key", "videoPath": str(video)},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "GEMINI_API_KEY_MISSING"


def test_auto_prompt_returns_caption_using_entry_template(test_state, tmp_path, client) -> None:
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="ic-ap", dataset_type="ic_lora", name="Conehead"
    )
    test_state.state.app_settings.gemini_api_key = "test-key"
    fake_captioner = test_state.video_captioner
    fake_captioner.caption_text = "conehead A close-up portrait of a person."
    video = tmp_path / "ref.mp4"
    video.write_bytes(b"\x00")

    resp = client.post(
        "/api/lora-inference/auto-prompt",
        json={"loraId": "user-ic-ap", "videoPath": str(video)},
    )
    assert resp.status_code == 200
    assert resp.json()["prompt"] == "conehead A close-up portrait of a person."

    # The captioner was called with the entry's (default) template as the
    # custom instructions and the configured API key.
    assert len(fake_captioner.calls) == 1
    call = fake_captioner.calls[0]
    assert call["api_key"] == "test-key"
    assert call["video_path"] == str(video)
    instructions = call["instructions"]
    assert isinstance(instructions, str)
    assert "MYTOK" in instructions


def test_auto_prompt_proxies_oversized_reference(test_state, tmp_path, client, monkeypatch) -> None:
    """A reference larger than Gemini's inline captioning ceiling is transcoded
    to a small caption-only proxy before being sent — auto-prompt must not 413
    on a big imported IC-LoRA reference."""
    import services.clip_processor.caption_proxy as caption_proxy

    # Shrink the budget so a tiny fake file triggers the proxy path without
    # writing 12MB of zeros; the real budget is exercised the same way.
    monkeypatch.setattr(caption_proxy, "CAPTION_PROXY_BUDGET_BYTES", 100)
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="ic-px", dataset_type="ic_lora", name="Conehead"
    )
    test_state.state.app_settings.gemini_api_key = "test-key"
    fake_captioner = test_state.video_captioner
    fake_clip = test_state.clip_processor
    video = tmp_path / "ref.mp4"
    video.write_bytes(b"\x00" * 200)  # over the monkeypatched budget

    resp = client.post(
        "/api/lora-inference/auto-prompt",
        json={"loraId": "user-ic-px", "videoPath": str(video)},
    )
    assert resp.status_code == 200, resp.text

    # A proxy was transcoded and the captioner received the proxy path, not the
    # oversized original. Auto-prompt is video-only so the proxy is muted.
    assert fake_clip.render_calls, "expected a proxy render for the oversized reference"
    assert len(fake_captioner.calls) == 1
    assert fake_captioner.calls[0]["video_path"] != str(video)
    assert fake_captioner.calls[0]["video_path"].endswith("proxy.mp4")


def test_auto_prompt_captions_small_reference_without_proxy(test_state, tmp_path, client, monkeypatch) -> None:
    """A small reference is captioned directly — no proxy transcode."""
    import services.clip_processor.caption_proxy as caption_proxy

    monkeypatch.setattr(caption_proxy, "CAPTION_PROXY_BUDGET_BYTES", 100)
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="ic-sm", dataset_type="ic_lora", name="Conehead"
    )
    test_state.state.app_settings.gemini_api_key = "test-key"
    fake_captioner = test_state.video_captioner
    fake_clip = test_state.clip_processor
    video = tmp_path / "ref.mp4"
    video.write_bytes(b"\x00" * 10)  # under the budget

    resp = client.post(
        "/api/lora-inference/auto-prompt",
        json={"loraId": "user-ic-sm", "videoPath": str(video)},
    )
    assert resp.status_code == 200, resp.text
    assert not fake_clip.render_calls, "small reference must not be proxied"
    assert fake_captioner.calls[0]["video_path"] == str(video)


def test_auto_prompt_rejects_standard_lora_without_template(test_state, tmp_path, client) -> None:
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="std-ap", dataset_type="standard", name="My Style"
    )
    test_state.state.app_settings.gemini_api_key = "test-key"
    video = tmp_path / "ref.mp4"
    video.write_bytes(b"\x00")
    resp = client.post(
        "/api/lora-inference/auto-prompt",
        json={"loraId": "user-std-ap", "videoPath": str(video)},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "LORA_NO_PROMPT_TEMPLATE"


def test_auto_prompt_unknown_lora_404(test_state, tmp_path, client) -> None:
    test_state.state.app_settings.gemini_api_key = "test-key"
    video = tmp_path / "ref.mp4"
    video.write_bytes(b"\x00")
    resp = client.post(
        "/api/lora-inference/auto-prompt",
        json={"loraId": "user-does-not-exist", "videoPath": str(video)},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------
# Update-template route
# ---------------------------------------------------------------------


def test_update_prompt_template_route_persists_and_returns_entry(
    test_state, tmp_path, client
) -> None:
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="ic-up", dataset_type="ic_lora", name="Conehead"
    )
    resp = client.put(
        "/api/lora-inference/prompt-template/user-ic-up",
        json={"promptTemplate": "MY CUSTOM TEMPLATE", "triggerWord": "conetok"},
    )
    assert resp.status_code == 200
    entry = resp.json()["entry"]
    assert entry["promptTemplate"] == "MY CUSTOM TEMPLATE"
    assert entry["triggerWord"] == "conetok"

    # Persists across a registry re-list.
    entries = test_state.lora_inference_registry.list_entries()
    again = next(e for e in entries if e.id == "user-ic-up")
    assert again.promptTemplate == "MY CUSTOM TEMPLATE"


def test_update_prompt_template_unknown_lora_404(test_state, client) -> None:
    resp = client.put(
        "/api/lora-inference/prompt-template/user-nope",
        json={"promptTemplate": "X", "triggerWord": None},
    )
    assert resp.status_code == 404


def test_regenerate_route_returns_generated_prompt_with_trigger_override(
    test_state, tmp_path, client
) -> None:
    _inject_completed_run(
        test_state.lora_training,
        tmp_path,
        job_id="ic-regenerate",
        dataset_type="ic_lora",
        description="Removes people and vehicles to create a clean plate",
    )
    resp = client.put(
        "/api/lora-inference/prompt-template/user-ic-regenerate",
        json={"promptTemplate": None, "triggerWord": "cleanplate"},
    )
    assert resp.status_code == 200, resp.text
    entry = resp.json()["entry"]
    assert entry["promptTemplate"]
    assert "cleanplate" in entry["promptTemplate"]
    assert entry["promptTemplateCustomized"] is False
