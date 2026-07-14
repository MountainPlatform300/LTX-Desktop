"""Tests for the imported LoRA library (user-supplied adapter weights).

Covers the import → registry → generate → delete lifecycle through the real
FastAPI app: a LoRA file the user got from outside the app is copied into app
storage, listed in the inference registry tagged with the variant the user
picked, routed through the right pipeline on generate, and removable. Uses
fake services (no mocks) per the backend test boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _routes._errors import HTTPError
from api_types import LoraInferenceRegistryResponseApi
from api_types import LoraInferenceEntryApi
from handlers.lora_inference_handler import LoraInferenceHandler


def _write_source_lora(tmp_path: Path, *, name: str = "external.safetensors") -> Path:
    src = tmp_path / name
    src.write_bytes(b"\x00" * 64)
    return src


def test_legacy_library_entry_cannot_be_applied(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pt"
    legacy.write_bytes(b"pickle-backed")
    entry = LoraInferenceEntryApi(
        id="imported-legacy",
        kind="imported",
        variant="standard",
        name="Legacy",
        conditioningTypes=[],
        localPath=str(legacy),
        available=True,
    )

    with pytest.raises(HTTPError) as exc:
        LoraInferenceHandler._require_local_path(entry)

    assert exc.value.code == "LORA_UNSAFE_WEIGHT_FORMAT"


def _import(
    client,
    *,
    source_path: str,
    name: str,
    variant: str,
    description: str | None = None,
    trigger_word: str | None = None,
    huggingface_url: str | None = None,
    example_prompt: str | None = None,
):
    body: dict = {"sourcePath": source_path, "name": name, "variant": variant}
    if description is not None:
        body["description"] = description
    if trigger_word is not None:
        body["triggerWord"] = trigger_word
    if huggingface_url is not None:
        body["huggingfaceUrl"] = huggingface_url
    if example_prompt is not None:
        body["examplePrompt"] = example_prompt
    return client.post("/api/lora-inference/import", json=body)


# ------------------------------------------------------------------
# Import
# ------------------------------------------------------------------


class TestImport:
    def test_import_standard_lora_lists_in_registry(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path)
        resp = _import(
            client, source_path=str(src), name="My Style LoRA", variant="standard"
        )
        assert resp.status_code == 200, resp.text
        entry = resp.json()["entry"]
        assert entry["kind"] == "imported"
        assert entry["variant"] == "standard"
        assert entry["available"] is True
        assert entry["conditioningTypes"] == []
        assert entry["sourceTrainingId"] is None
        assert entry["id"].startswith("imported-")
        assert Path(entry["localPath"]).is_file()

        reg = client.get("/api/lora-inference/registry")
        assert reg.status_code == 200
        parsed = LoraInferenceRegistryResponseApi.model_validate(reg.json())
        assert any(e.id == entry["id"] and e.kind == "imported" for e in parsed.entries)

    def test_import_copies_into_app_storage_independent_of_source(
        self, client, test_state, tmp_path
    ):
        src = _write_source_lora(tmp_path)
        resp = _import(client, source_path=str(src), name="Copied", variant="standard")
        assert resp.status_code == 200, resp.text
        local = resp.json()["entry"]["localPath"]
        assert Path(local).is_file()
        # The import must survive the original source disappearing.
        src.unlink()
        assert Path(local).is_file()

    def test_import_video_input_variant(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path, name="ref.safetensors")
        resp = _import(
            client, source_path=str(src), name="Ref IC-LoRA", variant="video_input_ic_lora"
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["entry"]["variant"] == "video_input_ic_lora"

    def test_import_rejects_unsupported_suffix(self, client, test_state, tmp_path):
        src = tmp_path / "not-a-lora.txt"
        src.write_bytes(b"nope")
        resp = _import(client, source_path=str(src), name="Bad", variant="standard")
        assert resp.status_code == 400
        assert resp.json()["code"] == "IMPORT_LORA_UNSUPPORTED_TYPE"

    @pytest.mark.parametrize("suffix", [".pt", ".bin", ".ckpt"])
    def test_import_rejects_pickle_backed_weight_formats(
        self, client, test_state, tmp_path, suffix
    ):
        src = _write_source_lora(tmp_path, name=f"unsafe{suffix}")
        resp = _import(client, source_path=str(src), name="Unsafe", variant="standard")
        assert resp.status_code == 400
        assert resp.json()["code"] == "IMPORT_LORA_UNSUPPORTED_TYPE"
        assert "safetensors" in resp.text

    def test_import_rejects_missing_file(self, client, test_state, tmp_path):
        resp = _import(
            client,
            source_path=str(tmp_path / "missing.safetensors"),
            name="Missing",
            variant="standard",
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "IMPORT_LORA_FILE_NOT_FOUND"

    def test_import_rejects_empty_name(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path)
        resp = _import(client, source_path=str(src), name="   ", variant="standard")
        assert resp.status_code == 400
        assert resp.json()["code"] == "IMPORT_LORA_NAME_REQUIRED"

    def test_import_trigger_word_surfaces_on_entry(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path)
        resp = _import(
            client,
            source_path=str(src),
            name="Instant Shave",
            variant="video_input_ic_lora",
            trigger_word="instant shave",
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["entry"]["triggerWord"] == "instant shave"

    def test_import_trigger_word_drives_system_prompt(self, client, test_state, tmp_path):
        # A name like "ltx-2.3-22b-ic-lora-instant-shave-0.9" derives a nonsense
        # trigger ("2 3 22b"); the user-supplied trigger word must win and be
        # embedded in the auto-generated system prompt instead.
        src = _write_source_lora(tmp_path)
        _import(
            client,
            source_path=str(src),
            name="ltx-2.3-22b-ic-lora-instant-shave-0.9",
            variant="video_input_ic_lora",
            trigger_word="instant shave",
        )
        reg = client.get("/api/lora-inference/registry")
        parsed = LoraInferenceRegistryResponseApi.model_validate(reg.json())
        entry = next(e for e in parsed.entries if e.kind == "imported")
        assert entry.triggerWord == "instant shave"
        assert entry.promptTemplate is not None
        assert "instant shave" in entry.promptTemplate
        assert "2 3 22b" not in entry.promptTemplate

    def test_import_without_trigger_word_does_not_guess_from_name(
        self, client, test_state, tmp_path
    ):
        src = _write_source_lora(tmp_path)
        _import(
            client,
            source_path=str(src),
            name="Conehead",
            variant="video_input_ic_lora",
        )
        reg = client.get("/api/lora-inference/registry")
        parsed = LoraInferenceRegistryResponseApi.model_validate(reg.json())
        entry = next(e for e in parsed.entries if e.kind == "imported")
        assert entry.triggerWord is None
        assert entry.promptTemplate is not None
        assert "Do not invent" in entry.promptTemplate

    def test_import_applies_profiled_prompt_template(
        self, client, test_state, fake_services, tmp_path
    ):
        # When the profiler returns a configured profile (e.g. derived from a
        # pasted HF URL), the import persists it as an override so the registry
        # serves the configured trigger + system prompt immediately, and the
        # response surfaces the profiling outcome.
        from services.interfaces import LoraPromptProfile, LoraPromptProfileResult

        fake_services.lora_prompt_profiler.result = LoraPromptProfileResult(
            status="configured",
            message="System prompt configured from the HuggingFace page via Gemini.",
            profile=LoraPromptProfile(
                trigger_word="MYTRIG", system_prompt="Use MYTRIG then describe the scene."
            ),
        )
        src = _write_source_lora(tmp_path)
        resp = _import(
            client,
            source_path=str(src),
            name="Custom",
            variant="video_input_ic_lora",
            trigger_word="ignored-by-profile",
            huggingface_url="https://huggingface.co/org/repo",
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        entry = body["entry"]
        # The profile's trigger wins over the user-supplied trigger word.
        assert entry["triggerWord"] == "MYTRIG"
        assert entry["promptTemplate"] == "Use MYTRIG then describe the scene."
        # The profiling outcome is surfaced (no longer silent).
        assert body["profileStatus"] == "configured"
        assert "HuggingFace" in body["profileMessage"]
        # The profiler was handed the HF URL + filename.
        call = fake_services.lora_prompt_profiler.calls[-1]
        assert call["huggingface_url"] == "https://huggingface.co/org/repo"
        assert call["filename"] == "external.safetensors"

        # Survives a registry re-read (override is persisted in the template store).
        reg = client.get("/api/lora-inference/registry")
        parsed = LoraInferenceRegistryResponseApi.model_validate(reg.json())
        re = next(e for e in parsed.entries if e.id == entry["id"])
        assert re.triggerWord == "MYTRIG"
        assert re.promptTemplate == "Use MYTRIG then describe the scene."

    def test_import_profiling_failure_falls_back_to_default(
        self, client, test_state, fake_services, tmp_path
    ):
        # A profiling error must never break the import — the entry is still
        # created with a safe triggerless default template, and the failure is
        # surfaced in the response so it isn't silent.
        fake_services.lora_prompt_profiler.raise_on_profile = RuntimeError("boom")
        src = _write_source_lora(tmp_path)
        resp = _import(
            client,
            source_path=str(src),
            name="Conehead",
            variant="video_input_ic_lora",
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["profileStatus"] == "failed"
        assert body["profileMessage"] is not None
        reg = client.get("/api/lora-inference/registry")
        parsed = LoraInferenceRegistryResponseApi.model_validate(reg.json())
        entry = next(e for e in parsed.entries if e.kind == "imported")
        assert entry.triggerWord is None
        assert "Do not invent" in (entry.promptTemplate or "")


# ------------------------------------------------------------------
# Delete
# ------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_entry_and_weights(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path)
        resp = _import(client, source_path=str(src), name="To Delete", variant="standard")
        body = resp.json()["entry"]
        entry_id = body["id"]
        local = body["localPath"]

        del_resp = client.delete(f"/api/lora-inference/imported/{entry_id}")
        assert del_resp.status_code == 204
        assert not Path(local).exists()

        reg = client.get("/api/lora-inference/registry")
        parsed = LoraInferenceRegistryResponseApi.model_validate(reg.json())
        assert all(e.id != entry_id for e in parsed.entries)

    def test_delete_unknown_returns_404(self, client, test_state):
        resp = client.delete("/api/lora-inference/imported/imported-nope")
        assert resp.status_code == 404
        assert resp.json()["code"] == "IMPORT_LORA_NOT_FOUND"


# ------------------------------------------------------------------
# Rename
# ------------------------------------------------------------------


class TestRename:
    def test_rename_updates_name_in_registry(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path)
        resp = _import(client, source_path=str(src), name="Old Name", variant="standard")
        entry_id = resp.json()["entry"]["id"]
        local = resp.json()["entry"]["localPath"]

        renamed = client.patch(
            f"/api/lora-inference/imported/{entry_id}", json={"name": "New Name"}
        )
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["entry"]["name"] == "New Name"
        # The weights file is untouched (display-only metadata change).
        assert Path(local).is_file()

        reg = client.get("/api/lora-inference/registry")
        parsed = LoraInferenceRegistryResponseApi.model_validate(reg.json())
        entry = next(e for e in parsed.entries if e.id == entry_id)
        assert entry.name == "New Name"

    def test_rename_trims_and_rejects_blank(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path)
        resp = _import(client, source_path=str(src), name="Old", variant="standard")
        entry_id = resp.json()["entry"]["id"]

        blank = client.patch(f"/api/lora-inference/imported/{entry_id}", json={"name": "   "})
        assert blank.status_code == 400
        assert blank.json()["code"] == "IMPORT_LORA_NAME_REQUIRED"

        # Whitespace-only trailing is trimmed to the inner value.
        renamed = client.patch(
            f"/api/lora-inference/imported/{entry_id}", json={"name": "  Trimmed  "}
        )
        assert renamed.status_code == 200
        assert renamed.json()["entry"]["name"] == "Trimmed"

    def test_rename_unknown_returns_404(self, client, test_state):
        resp = client.patch(
            "/api/lora-inference/imported/imported-nope", json={"name": "x"}
        )
        assert resp.status_code == 404
        assert resp.json()["code"] == "IMPORT_LORA_NOT_FOUND"


# ------------------------------------------------------------------
# Generate routing for an imported LoRA
# ------------------------------------------------------------------


class TestImportedGenerate:
    def test_imported_standard_lora_routes_through_fast_pipeline(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        src = _write_source_lora(tmp_path)
        imp = _import(
            client, source_path=str(src), name="Imported Style", variant="standard"
        )
        lora_id = imp.json()["entry"]["id"]
        local_path = imp.json()["entry"]["localPath"]
        test_state.state.app_settings.use_local_text_encoder = True

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "standard",
                "loraId": lora_id,
                "loraScale": 0.9,
                "request": {
                    "prompt": "a cinematic shot",
                    "resolution": "540p",
                    "duration": 5,
                    "fps": 24,
                    "aspectRatio": "16:9",
                },
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "complete"
        # The imported adapter must be built into the fast pipeline, same as a
        # trained standard LoRA.
        assert fake_services.fast_video_pipeline.last_lora_path == local_path
        assert fake_services.fast_video_pipeline.last_lora_scale == 0.9


# ------------------------------------------------------------------
# Library metadata enrichment + edit (imported)
# ------------------------------------------------------------------


class TestImportedMetadata:
    def test_import_surfaces_created_at_file_size_and_huggingface_url(
        self, client, test_state, tmp_path
    ):
        src = _write_source_lora(tmp_path)
        resp = _import(
            client,
            source_path=str(src),
            name="With HF",
            variant="standard",
            huggingface_url="https://huggingface.co/u/r",
        )
        assert resp.status_code == 200, resp.text
        entry = resp.json()["entry"]
        assert entry["createdAt"] is not None
        assert entry["fileSizeBytes"] == 64
        assert entry["huggingfaceUrl"] == "https://huggingface.co/u/r"

    def test_update_description_and_huggingface_url(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path)
        entry_id = _import(
            client, source_path=str(src), name="Edit Me", variant="standard"
        ).json()["entry"]["id"]

        resp = client.patch(
            f"/api/lora-inference/imported/{entry_id}",
            json={"description": "a nice style", "huggingfaceUrl": "https://huggingface.co/u/r2"},
        )
        assert resp.status_code == 200, resp.text
        entry = resp.json()["entry"]
        assert entry["description"] == "a nice style"
        assert entry["huggingfaceUrl"] == "https://huggingface.co/u/r2"
        # Name untouched when not supplied.
        assert entry["name"] == "Edit Me"

    def test_update_with_no_fields_returns_400(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path)
        entry_id = _import(
            client, source_path=str(src), name="No Fields", variant="standard"
        ).json()["entry"]["id"]
        resp = client.patch(f"/api/lora-inference/imported/{entry_id}", json={})
        assert resp.status_code == 400
        assert resp.json()["code"] == "IMPORT_LORA_NO_FIELDS"

    def test_update_blank_description_clears_it(self, client, test_state, tmp_path):
        src = _write_source_lora(tmp_path)
        entry_id = _import(
            client,
            source_path=str(src),
            name="Clear Desc",
            variant="standard",
            description="orig",
        ).json()["entry"]["id"]
        resp = client.patch(
            f"/api/lora-inference/imported/{entry_id}", json={"description": "   "}
        )
        assert resp.status_code == 200
        assert resp.json()["entry"]["description"] is None


# ------------------------------------------------------------------
# Reprofile (re-derive system prompt + trigger for an imported LoRA)
# ------------------------------------------------------------------


class TestReprofile:
    def test_reprofile_applies_configured_profile_as_override(
        self, client, test_state, fake_services, tmp_path
    ):
        from services.interfaces import LoraPromptProfile, LoraPromptProfileResult

        fake_services.lora_prompt_profiler.result = LoraPromptProfileResult(
            status="configured",
            message="ok",
            profile=LoraPromptProfile(
                trigger_word="mytok",
                system_prompt="Use mytok to activate.",
            ),
        )
        src = _write_source_lora(tmp_path)
        entry_id = _import(
            client, source_path=str(src), name="Reprofile Me", variant="standard"
        ).json()["entry"]["id"]
        fake_services.lora_prompt_profiler.calls.clear()

        resp = client.post(
            f"/api/lora-inference/imported/{entry_id}/reprofile",
            json={"examplePrompt": "mytok closeup"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["profileStatus"] == "configured"
        assert body["entry"]["triggerWord"] == "mytok"
        assert body["entry"]["promptTemplate"] == "Use mytok to activate."
        # The example prompt was forwarded to the profiler.
        assert fake_services.lora_prompt_profiler.calls[-1]["example_prompt"] == "mytok closeup"

    def test_reprofile_falls_back_to_stored_huggingface_url(
        self, client, test_state, fake_services, tmp_path
    ):
        fake_services.lora_prompt_profiler.calls.clear()
        src = _write_source_lora(tmp_path)
        entry_id = _import(
            client,
            source_path=str(src),
            name="HF Stored",
            variant="standard",
            huggingface_url="https://huggingface.co/u/orig",
        ).json()["entry"]["id"]

        resp = client.post(f"/api/lora-inference/imported/{entry_id}/reprofile", json={})
        assert resp.status_code == 200, resp.text
        assert (
            fake_services.lora_prompt_profiler.calls[-1]["huggingface_url"]
            == "https://huggingface.co/u/orig"
        )

    def test_reprofile_unknown_returns_404(self, client, test_state):
        resp = client.post("/api/lora-inference/imported/imported-nope/reprofile", json={})
        assert resp.status_code == 404
        assert resp.json()["code"] == "IMPORT_LORA_NOT_FOUND"

    def test_reprofile_failed_profiler_returns_failed_status(
        self, client, test_state, fake_services, tmp_path
    ):
        src = _write_source_lora(tmp_path)
        entry_id = _import(
            client, source_path=str(src), name="Boom", variant="standard"
        ).json()["entry"]["id"]
        fake_services.lora_prompt_profiler.raise_on_profile = RuntimeError("boom")
        resp = client.post(f"/api/lora-inference/imported/{entry_id}/reprofile", json={})
        assert resp.status_code == 200
        assert resp.json()["profileStatus"] == "failed"
