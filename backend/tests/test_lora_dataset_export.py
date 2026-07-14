"""Integration tests for portable LoRA dataset export + import.

Covers the trainer-ready bundle shape (relative `dataset.json` + `clips/`),
the reject filter, folder vs zip packaging, and a lossless export→import
roundtrip (captions, trigger word, IC-LoRA references, triage).
"""

from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

import pytest

from services.clip_processor.clip_processor import ClipProbeResult

# What the FakeClipProcessor reports for every NORMALIZED output clip: a valid,
# training-ready 25fps / 49-frame (8k+1) / 1024x576 clip so IC-LoRA pairs pass
# the consistency validation. Tests that want a drop override per-path.
_NORMALIZED_PROBE = ClipProbeResult(
    duration_seconds=2.0, width=1024, height=576, fps=25.0,
    frame_count=49, has_audio=False, video_codec="h264",
)


def _make_clip(tmp_path: Path, name: str) -> str:
    p = tmp_path / name
    p.write_bytes(b"\x00fake-media")
    return str(p)


def _create(client, clips, *, name: str = "My Set", type_: str = "standard"):
    r = client.post(
        "/api/lora/datasets",
        json={"name": name, "type": type_, "triggerWord": "TOK", "clips": clips},
    )
    assert r.status_code == 200, r.text
    return r.json()


class TestExport:
    def test_folder_bundle_is_trainer_ready(self, client, tmp_path) -> None:
        ds = _create(
            client,
            [
                {"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "a cat"},
                {"localPath": _make_clip(tmp_path, "b.mp4"), "caption": "a dog"},
            ],
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder", "includeRejected": False},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["clipCount"] == 2
        root = Path(body["exportPath"])
        assert root.is_dir() and root.name == "My_Set"

        rows = json.loads((root / "dataset.json").read_text())
        assert len(rows) == 2
        assert all(row["media_path"].startswith("clips/") for row in rows)
        assert {row["caption"] for row in rows} == {"a cat", "a dog"}
        assert len(list((root / "clips").iterdir())) == 2

        manifest = json.loads((root / "ltxdesktop.json").read_text())
        assert manifest["kind"] == "ltx-desktop-lora-dataset"
        assert manifest["triggerWord"] == "TOK"
        assert (root / "README.md").is_file()
        assert (root / "train_config.yaml").is_file()

    def test_excludes_rejected_by_default(self, client, tmp_path) -> None:
        ds = _create(
            client,
            [
                {"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "keep", "triage": "keep"},
                {"localPath": _make_clip(tmp_path, "b.mp4"), "caption": "drop", "triage": "reject"},
            ],
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["clipCount"] == 1
        rows = json.loads((Path(r.json()["exportPath"]) / "dataset.json").read_text())
        assert [row["caption"] for row in rows] == ["keep"]

    def test_include_rejected_flag(self, client, tmp_path) -> None:
        ds = _create(
            client,
            [
                {"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "keep", "triage": "keep"},
                {"localPath": _make_clip(tmp_path, "b.mp4"), "caption": "drop", "triage": "reject"},
            ],
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder", "includeRejected": True},
        )
        assert r.status_code == 200, r.text
        assert r.json()["clipCount"] == 2

    def test_zip_package(self, client, tmp_path) -> None:
        ds = _create(client, [{"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "a cat"}])
        zip_path = tmp_path / "bundle.zip"
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(zip_path), "format": "zip"},
        )
        assert r.status_code == 200, r.text
        out = Path(r.json()["exportPath"])
        assert out.suffix == ".zip" and out.is_file()
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert any(n.endswith("My_Set/dataset.json") for n in names)
        assert any("/clips/" in n for n in names)

    def test_excludes_trashed_clips(self, client, tmp_path) -> None:
        ds = _create(
            client,
            [
                {"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "keep"},
                {
                    "localPath": _make_clip(tmp_path, "b.mp4"),
                    "caption": "trashed",
                    "deletedAt": "2026-01-01T00:00:00Z",
                },
            ],
        )
        # The trashed clip round-trips through the dataset API.
        trashed = next(c for c in ds["clips"] if c["caption"] == "trashed")
        assert trashed["deletedAt"] == "2026-01-01T00:00:00Z"

        dest = tmp_path / "out"
        dest.mkdir()
        # Even with includeRejected the trashed clip never ships.
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder", "includeRejected": True},
        )
        assert r.status_code == 200, r.text
        assert r.json()["clipCount"] == 1
        rows = json.loads((Path(r.json()["exportPath"]) / "dataset.json").read_text())
        assert [row["caption"] for row in rows] == ["keep"]

    def test_all_trashed_is_400(self, client, tmp_path) -> None:
        ds = _create(
            client,
            [
                {
                    "localPath": _make_clip(tmp_path, "a.mp4"),
                    "caption": "x",
                    "deletedAt": "2026-01-01T00:00:00Z",
                }
            ],
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder", "includeRejected": True},
        )
        assert r.status_code == 400

    def test_all_rejected_is_400(self, client, tmp_path) -> None:
        ds = _create(
            client,
            [{"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "x", "triage": "reject"}],
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 400

    def test_ic_lora_emits_pairs_only_with_video_schema(
        self, client, tmp_path, fake_services
    ) -> None:
        # Normalized outputs all probe as a valid 25fps / 49-frame / 1024x576 clip.
        fake_services.clip_processor.result = _NORMALIZED_PROBE
        # Two bearded inputs + two clean-shaven targets that reference them; the
        # inputs are also standalone clips (the ~"input-only rows" the trainer
        # rejected). Only the two true PAIRS should ship.
        in_a = _make_clip(tmp_path, "in_a.mp4")
        out_a = _make_clip(tmp_path, "out_a.mp4")
        in_b = _make_clip(tmp_path, "in_b.mp4")
        out_b = _make_clip(tmp_path, "out_b.mp4")
        ds = _create(
            client,
            [
                {"localPath": in_a, "caption": "A bearded man."},
                {"localPath": out_a, "caption": "A clean-shaven man smiles.", "referencePath": in_a, "referencePaths": [in_a]},
                {"localPath": in_b, "caption": "A bearded man outdoors."},
                {"localPath": out_b, "caption": "A clean-shaven man outdoors.", "referencePath": in_b, "referencePaths": [in_b]},
            ],
            type_="ic_lora",
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["clipCount"] == 2
        root = Path(body["exportPath"])
        # Exactly two pairs → four files (output + reference each), no input-only.
        names = sorted(p.name for p in (root / "clips").iterdir())
        assert names == [
            "0001_output_out_a.mp4",
            "0001_reference_in_a.mp4",
            "0002_output_out_b.mp4",
            "0002_reference_in_b.mp4",
        ]
        rows = json.loads((root / "dataset.json").read_text())
        assert len(rows) == 2
        # Exactly the trainer's three keys, nothing else.
        assert all(set(row) == {"caption", "video", "reference_video"} for row in rows)
        by_caption = {row["caption"]: row for row in rows}
        assert by_caption["A clean-shaven man smiles."]["video"] == "clips/0001_output_out_a.mp4"
        assert by_caption["A clean-shaven man smiles."]["reference_video"] == "clips/0001_reference_in_a.mp4"

    def test_ic_lora_export_reports_dropped_pairs(
        self, client, tmp_path, fake_services
    ) -> None:
        fake_services.clip_processor.result = _NORMALIZED_PROBE
        good_in = _make_clip(tmp_path, "good_in.mp4")
        good_out = _make_clip(tmp_path, "good_out.mp4")
        bad_in = _make_clip(tmp_path, "bad_in.mp4")
        bad_out = _make_clip(tmp_path, "bad_out.mp4")
        ds = _create(
            client,
            [
                {"localPath": good_in, "caption": "A bearded man."},
                {"localPath": good_out, "caption": "A clean-shaven man smiles.", "referencePath": good_in, "referencePaths": [good_in]},
                {"localPath": bad_in, "caption": "A bearded man."},
                # Truncated caption → this pair must be dropped + reported.
                {"localPath": bad_out, "caption": "A man standing near foliage in", "referencePath": bad_in, "referencePaths": [bad_in]},
            ],
            type_="ic_lora",
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["clipCount"] == 1
        assert len(body["droppedPairs"]) == 1
        assert "truncated" in body["droppedPairs"][0]

    def test_ic_lora_aligns_near_aspect_pair_to_shared_dims(
        self, client, tmp_path, fake_services
    ) -> None:
        # A 1264x720 target paired with a 1920x1080 reference (both landscape,
        # ~1% aspect drift) used to be dropped because each kept its own width.
        # Now both are forced to one resolution via exact_width/exact_height.
        fake_services.clip_processor.result = _NORMALIZED_PROBE
        ref = _make_clip(tmp_path, "ref.mp4")
        tgt = _make_clip(tmp_path, "tgt.mp4")
        fake_services.clip_processor.results_by_path[ref] = ClipProbeResult(
            duration_seconds=5.0, width=1920, height=1080, fps=30.0,
            frame_count=150, has_audio=True, video_codec="h264",
        )
        fake_services.clip_processor.results_by_path[tgt] = ClipProbeResult(
            duration_seconds=5.0, width=1264, height=720, fps=25.0,
            frame_count=125, has_audio=False, video_codec="h264",
        )
        ds = _create(
            client,
            [
                {"localPath": ref, "caption": "A bearded man."},
                {"localPath": tgt, "caption": "A clean-shaven man.", "referencePath": ref, "referencePaths": [ref]},
            ],
            type_="ic_lora",
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["clipCount"] == 1  # not dropped
        # Both clips of the pair were normalized to the SAME exact resolution.
        sized = [c for c in fake_services.clip_processor.normalize_calls if c["exact_width"]]
        assert len(sized) == 2
        assert {(c["exact_width"], c["exact_height"]) for c in sized} == {(1012, 576)}

    def test_ic_lora_drops_pair_with_different_orientation(
        self, client, tmp_path, fake_services
    ) -> None:
        fake_services.clip_processor.result = _NORMALIZED_PROBE
        ref = _make_clip(tmp_path, "ref.mp4")
        tgt = _make_clip(tmp_path, "tgt.mp4")
        # Landscape reference, portrait target — can't force-align without bad
        # distortion, so the pair must be dropped with an orientation reason.
        fake_services.clip_processor.results_by_path[ref] = ClipProbeResult(
            duration_seconds=5.0, width=1920, height=1080, fps=25.0,
            frame_count=125, has_audio=False, video_codec="h264",
        )
        fake_services.clip_processor.results_by_path[tgt] = ClipProbeResult(
            duration_seconds=5.0, width=1080, height=1920, fps=25.0,
            frame_count=125, has_audio=False, video_codec="h264",
        )
        ds = _create(
            client,
            [
                {"localPath": ref, "caption": "A bearded man."},
                {"localPath": tgt, "caption": "A clean-shaven man.", "referencePath": ref, "referencePaths": [ref]},
            ],
            type_="ic_lora",
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 400, r.text  # nothing usable
        assert "orientation" in r.json()["message"]

    def test_ic_lora_export_fails_loudly_when_nothing_usable(
        self, client, tmp_path, fake_services
    ) -> None:
        fake_services.clip_processor.result = _NORMALIZED_PROBE
        ref = _make_clip(tmp_path, "ref.mp4")
        tgt = _make_clip(tmp_path, "tgt.mp4")
        ds = _create(
            client,
            [
                {"localPath": ref, "caption": "A bearded man."},
                # Trigger word leaks into the target caption → dropped.
                {"localPath": tgt, "caption": "A TOK clean-shaven man.", "referencePath": ref, "referencePaths": [ref]},
            ],
            type_="ic_lora",
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 400, r.text

    def test_standard_lora_keeps_numbered_clips(self, client, tmp_path) -> None:
        ds = _create(
            client,
            [
                {"localPath": _make_clip(tmp_path, "first.mp4"), "caption": "a"},
                {"localPath": _make_clip(tmp_path, "second.mp4"), "caption": "b"},
            ],
        )
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 200, r.text
        root = Path(r.json()["exportPath"])
        names = sorted(p.name for p in (root / "clips").iterdir())
        assert names == ["0001_first.mp4", "0002_second.mp4"]

    def test_bundle_includes_model_card_by_default(self, client, tmp_path) -> None:
        ds = _create(client, [{"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "a cat"}])
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 200, r.text
        card = (Path(r.json()["exportPath"]) / "MODEL_CARD.md").read_text()
        # HF front matter + pre-filled, dataset-tailored values.
        assert card.startswith("---")
        assert "base_model_relation: adapter" in card
        assert "**Training Type:** LoRA" in card
        assert "My Set" in card  # dataset name in the title
        assert "`TOK`" in card  # trigger word pre-filled

    def test_components_can_be_excluded(self, client, tmp_path) -> None:
        ds = _create(client, [{"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "a cat"}])
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={
                "destPath": str(dest),
                "format": "folder",
                "includeConfig": False,
                "includeReadme": False,
                "includeManifest": False,
                "includeModelCard": False,
            },
        )
        assert r.status_code == 200, r.text
        root = Path(r.json()["exportPath"])
        # Core dataset is always written; the opted-out extras are absent.
        assert (root / "dataset.json").is_file()
        assert list((root / "clips").iterdir())
        assert not (root / "train_config.yaml").exists()
        assert not (root / "README.md").exists()
        assert not (root / "ltxdesktop.json").exists()
        assert not (root / "MODEL_CARD.md").exists()

    def test_config_uses_selected_profile(self, client, tmp_path) -> None:
        ds = _create(client, [{"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "a cat"}])
        # The official Low VRAM profile carries rank 16 (default is 32).
        profiles = client.get("/api/lora/profiles").json()["profiles"]
        low_vram = next(p for p in profiles if p["name"] == "Low VRAM")

        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={
                "destPath": str(dest),
                "format": "folder",
                "includeRejected": False,
                "profileId": low_vram["id"],
            },
        )
        assert r.status_code == 200, r.text
        yaml = (Path(r.json()["exportPath"]) / "train_config.yaml").read_text()
        assert "rank: 16" in yaml

    def test_config_defaults_without_profile(self, client, tmp_path) -> None:
        ds = _create(client, [{"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "a cat"}])
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder", "includeRejected": False},
        )
        assert r.status_code == 200, r.text
        yaml = (Path(r.json()["exportPath"]) / "train_config.yaml").read_text()
        assert "rank: 32" in yaml

    def test_unknown_profile_is_404(self, client, tmp_path) -> None:
        ds = _create(client, [{"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "a cat"}])
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={
                "destPath": str(dest),
                "format": "folder",
                "includeRejected": False,
                "profileId": "does-not-exist",
            },
        )
        assert r.status_code == 404, r.text

    def test_missing_dataset_is_404(self, client, tmp_path) -> None:
        dest = tmp_path / "out"
        dest.mkdir()
        r = client.post(
            "/api/lora/datasets/nope/export",
            json={"destPath": str(dest), "format": "folder"},
        )
        assert r.status_code == 404


class TestImportRoundtrip:
    def test_folder_roundtrip_preserves_clips(self, client, tmp_path) -> None:
        ds = _create(
            client,
            [
                {"localPath": _make_clip(tmp_path, "a.mp4"), "caption": "a cat"},
                {"localPath": _make_clip(tmp_path, "b.mp4"), "caption": "a dog"},
            ],
        )
        dest = tmp_path / "out"
        dest.mkdir()
        exported = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(dest), "format": "folder"},
        ).json()["exportPath"]

        r = client.post("/api/lora/datasets/import", json={"sourcePath": exported})
        assert r.status_code == 200, r.text
        imp = r.json()
        assert imp["id"] != ds["id"]
        assert imp["name"] == "My Set"
        assert imp["triggerWord"] == "TOK"
        assert {c["caption"] for c in imp["clips"]} == {"a cat", "a dog"}
        for c in imp["clips"]:
            assert Path(c["localPath"]).is_file()

    def test_zip_roundtrip_preserves_ic_lora_references(
        self, client, tmp_path, fake_services
    ) -> None:
        fake_services.clip_processor.result = _NORMALIZED_PROBE
        ref = _make_clip(tmp_path, "ref.mp4")
        tgt = _make_clip(tmp_path, "tgt.mp4")
        ds = _create(
            client,
            [
                {"localPath": ref, "caption": "A bearded man."},
                {
                    "localPath": tgt,
                    "caption": "A clean-shaven man.",
                    "referencePath": ref,
                    "referencePaths": [ref],
                },
            ],
            type_="ic_lora",
        )
        zip_path = tmp_path / "ic.zip"
        exported = client.post(
            f"/api/lora/datasets/{ds['id']}/export",
            json={"destPath": str(zip_path), "format": "zip"},
        ).json()["exportPath"]

        r = client.post("/api/lora/datasets/import", json={"sourcePath": exported})
        assert r.status_code == 200, r.text
        imp = r.json()
        assert imp["type"] == "ic_lora"
        target = next(c for c in imp["clips"] if c["caption"] == "A clean-shaven man.")
        assert target["referencePath"]
        assert Path(target["referencePath"]).is_file()

    def test_invalid_source_is_400(self, client, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        r = client.post("/api/lora/datasets/import", json={"sourcePath": str(empty)})
        assert r.status_code == 400

    def test_manifest_file_reference_cannot_escape_bundle_root(
        self, client, tmp_path
    ) -> None:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        secret = tmp_path / "secret.mp4"
        secret.write_bytes(b"must-not-be-imported")
        (bundle / "ltxdesktop.json").write_text(
            json.dumps(
                {
                    "kind": "ltx-desktop-lora-dataset",
                    "schemaVersion": 1,
                    "name": "hostile",
                    "clips": [
                        {
                            "file": "../secret.mp4",
                            "caption": "steal a sibling file",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        response = client.post(
            "/api/lora/datasets/import",
            json={"sourcePath": str(bundle)},
        )

        assert response.status_code == 400
        assert "outside the import folder" in response.text


class TestSafeExtractall:
    """Zip-Slip guard: a hostile bundle must not write outside the import root."""

    def test_clean_zip_extracts(self, tmp_path: Path) -> None:
        from handlers import lora_export

        dest = tmp_path / "out"
        dest.mkdir()
        zip_path = tmp_path / "ok.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("clips/a.mp4", b"data")
            zf.writestr("ltxdesktop.json", b"{}")

        with zipfile.ZipFile(zip_path) as zf:
            lora_export.safe_extractall(zf, dest)

        assert (dest / "clips" / "a.mp4").is_file()
        assert (dest / "ltxdesktop.json").is_file()

    def test_traversal_entry_refused(self, tmp_path: Path) -> None:
        from handlers import lora_export

        dest = tmp_path / "out"
        dest.mkdir()
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../escape.mp4", b"pwned")

        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(lora_export.BundleError):
                lora_export.safe_extractall(zf, dest)
        # Nothing escaped.
        assert not (tmp_path / "escape.mp4").exists()

    def test_absolute_entry_refused(self, tmp_path: Path) -> None:
        from handlers import lora_export

        dest = tmp_path / "out"
        dest.mkdir()
        zip_path = tmp_path / "abs.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("/etc/hostile.txt", b"pwned")

        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(lora_export.BundleError):
                lora_export.safe_extractall(zf, dest)

    def test_manifest_member_traversal_refused(self, tmp_path: Path) -> None:
        from handlers import lora_export

        root = tmp_path / "bundle"
        root.mkdir()
        with pytest.raises(lora_export.BundleError):
            lora_export.resolve_bundle_member(root, "../outside.mp4")

    def test_symlink_entry_refused(self, tmp_path: Path) -> None:
        from handlers import lora_export

        dest = tmp_path / "out"
        dest.mkdir()
        zip_path = tmp_path / "link.zip"
        link = zipfile.ZipInfo("clips/link.mp4")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(link, "../outside.mp4")

        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(lora_export.BundleError, match="link or device"):
                lora_export.safe_extractall(zf, dest)

    def test_too_many_entries_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from handlers import lora_export

        monkeypatch.setattr(lora_export, "MAX_BUNDLE_FILES", 1)
        zip_path = tmp_path / "many.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("one.mp4", b"1")
            zf.writestr("two.mp4", b"2")

        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(lora_export.BundleError, match="too many entries"):
                lora_export.safe_extractall(zf, tmp_path / "out")

    def test_expanded_size_limit_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from handlers import lora_export

        monkeypatch.setattr(lora_export, "MAX_BUNDLE_UNCOMPRESSED_BYTES", 3)
        zip_path = tmp_path / "large.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("clip.mp4", b"four")

        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(lora_export.BundleError, match="allowed size"):
                lora_export.safe_extractall(zf, tmp_path / "out")

