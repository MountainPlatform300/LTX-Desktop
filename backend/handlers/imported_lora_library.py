"""Imported LoRA library — user-supplied adapter weights usable from Gen Space.

The library is a durable ledger of LoRAs the user imported from outside the app
(e.g. an adapter they downloaded from someone else), with the weights copied
into app storage at ``<app_data_dir>/lora/library/<id>/<filename>`` so the
import is independent of the original source path. Each import is tagged with
a variant the user picks at import time — ``standard`` (a t2v/i2v style
adapter) or ``video_input_ic_lora`` (a reference-video IC-LoRA) — the same
variants the inference registry / generate flow routes on, so an imported LoRA
flows through Gen Space exactly like a trained one.

Persistence mirrors the training handler: a single ``lora_library.json``
ledger written atomically (tmp + ``os.replace``), loaded once at startup with
crash-style pruning of entries whose weights file has vanished on disk.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from _routes._errors import HTTPError
from api_types import (
    ImportedLoraVariantApi,
    LoraInferenceEntryApi,
)
from handlers.base import StateHandlerBase

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)

# Safetensors is a data-only container. Pickle-backed `.pt`, `.bin`, and `.ckpt`
# files can execute code when loaded and are therefore unsuitable for direct
# import into a desktop app.
_ACCEPTED_SUFFIXES: tuple[str, ...] = (".safetensors",)

# Example-media extensions a user can attach to a library entry (CivitAI-style
# "what does this LoRA do?" preview). Images render as the card thumbnail;
# videos play inline. Everything else is rejected at attach time.
_EXAMPLE_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_EXAMPLE_VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".webm", ".mov", ".m4v")


def example_media_type_for(path: str) -> Literal["image", "video"] | None:
    """Classify an example file as "image" / "video" by suffix, else None."""
    suffix = Path(path).suffix.lower()
    if suffix in _EXAMPLE_IMAGE_SUFFIXES:
        return "image"
    if suffix in _EXAMPLE_VIDEO_SUFFIXES:
        return "video"
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ImportedLora(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    name: str
    variant: ImportedLoraVariantApi
    local_path: str
    created_at: str
    description: str | None = None
    file_size_bytes: int
    # The trigger word the user supplied at import (or None to fall back to a
    # name-derived default). Surfaces on the registry entry so the auto-prompt
    # system prompt is generated with the correct trigger.
    trigger_word: str | None = None
    # HuggingFace model-card URL the LoRA was imported from / profiled against,
    # so the Library can link back to the source. None for direct-file imports.
    huggingface_url: str | None = None
    # Optional user-supplied example image/video showing what the LoRA does,
    # copied into the per-LoRA dir as `example.<ext>`. None when no example is
    # attached. The media kind is inferred from the suffix at attach time.
    example_path: str | None = None


class ImportedLorasState(BaseModel):
    model_config = ConfigDict(strict=True)

    items: list[ImportedLora] = Field(default_factory=list[ImportedLora])


class ImportedLoraLibrary(StateHandlerBase):
    """Owns the imported-LoRA ledger + on-disk weights, read-only to the registry."""

    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)
        base = config.app_data_dir
        self._library_dir: Path = base / "lora" / "library"
        self._ledger_file: Path = base / "lora_library.json"
        self._items: list[ImportedLora] = []
        self._loaded: bool = False

    def load_state(self) -> None:
        """Load the ledger once at startup, pruning entries whose weights vanished."""
        with self.lock:
            if self._loaded:
                return
            self._items = self._load_file_unlocked().items
            before = len(self._items)
            self._items = [i for i in self._items if Path(i.local_path).is_file()]
            if len(self._items) != before:
                self._persist_unlocked()
            self._loaded = True

    def list_entries(self) -> list[LoraInferenceEntryApi]:
        with self.lock:
            return [self._to_entry(i) for i in self._items]

    def import_lora(
        self,
        *,
        source_path: str,
        name: str,
        variant: ImportedLoraVariantApi,
        description: str | None,
        trigger_word: str | None,
        huggingface_url: str | None = None,
    ) -> LoraInferenceEntryApi:
        src = Path(source_path)
        if not src.is_file():
            raise HTTPError(
                400, f"LoRA file not found: {source_path}", code="IMPORT_LORA_FILE_NOT_FOUND"
            )
        if src.suffix.lower() not in _ACCEPTED_SUFFIXES:
            raise HTTPError(
                400,
                "Only .safetensors LoRA files can be imported safely",
                code="IMPORT_LORA_UNSUPPORTED_TYPE",
            )
        clean_name = name.strip()
        if not clean_name:
            raise HTTPError(400, "LoRA name is required", code="IMPORT_LORA_NAME_REQUIRED")
        clean_trigger = trigger_word.strip() if trigger_word else None
        clean_hf = huggingface_url.strip() if huggingface_url else None

        with self.lock:
            new_id = uuid.uuid4().hex
            dest_dir = self._library_dir / new_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            size = dest.stat().st_size
            item = ImportedLora(
                id=new_id,
                name=clean_name,
                variant=variant,
                local_path=str(dest),
                created_at=_now_iso(),
                description=description,
                file_size_bytes=size,
                trigger_word=clean_trigger,
                huggingface_url=clean_hf,
            )
            self._items.append(item)
            self._persist_unlocked()
            return self._to_entry(item)

    def delete_imported(self, entry_id: str) -> None:
        """Delete by registry entry id (``imported-<id>``)."""
        with self.lock:
            target = next(
                (i for i in self._items if f"imported-{i.id}" == entry_id),
                None,
            )
            if target is None:
                raise HTTPError(
                    404, f"Unknown imported LoRA: {entry_id}", code="IMPORT_LORA_NOT_FOUND"
                )
            self._items = [i for i in self._items if i.id != target.id]
            self._persist_unlocked()
            weights_dir = self._library_dir / target.id
        # Filesystem teardown outside the lock.
        shutil.rmtree(weights_dir, ignore_errors=True)

    def update_imported(
        self,
        entry_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        huggingface_url: str | None = None,
    ) -> LoraInferenceEntryApi:
        """Patch editable metadata on an imported LoRA by registry entry id.

        Any subset of ``name`` / ``description`` / ``huggingface_url`` may be
        supplied; at least one must be set. ``name`` is validated non-blank when
        present. The weights file is untouched (display-only metadata). Returns
        the updated registry entry.
        """
        if name is None and description is None and huggingface_url is None:
            raise HTTPError(
                400, "Provide at least one field to update", code="IMPORT_LORA_NO_FIELDS"
            )
        if name is not None and not name.strip():
            raise HTTPError(400, "LoRA name is required", code="IMPORT_LORA_NAME_REQUIRED")
        with self.lock:
            target = next(
                (i for i in self._items if f"imported-{i.id}" == entry_id),
                None,
            )
            if target is None:
                raise HTTPError(
                    404, f"Unknown imported LoRA: {entry_id}", code="IMPORT_LORA_NOT_FOUND"
                )
            updates: dict[str, object] = {}
            if name is not None:
                updates["name"] = name.strip()
            if description is not None:
                updates["description"] = description.strip() or None
            if huggingface_url is not None:
                updates["huggingface_url"] = huggingface_url.strip() or None
            updated = target.model_copy(update=updates)
            self._items = [updated if i.id == target.id else i for i in self._items]
            self._persist_unlocked()
            return self._to_entry(updated)

    def find_imported(self, entry_id: str) -> ImportedLora | None:
        """Look up a ledger item by registry entry id (``imported-<id>``)."""
        with self.lock:
            return next(
                (i for i in self._items if f"imported-{i.id}" == entry_id),
                None,
            )

    def set_example(self, entry_id: str, *, source_path: str) -> LoraInferenceEntryApi:
        """Attach (or replace) an example image/video for an imported LoRA.

        Copies the user-supplied file into the per-LoRA dir as a stable
        ``example.<ext>`` so it survives the source moving. Validates the file
        exists and is a supported image/video type. Returns the updated entry.
        """
        src = Path(source_path)
        if not src.is_file():
            raise HTTPError(
                400, f"Example file not found: {source_path}", code="LORA_EXAMPLE_NOT_FOUND"
            )
        media_type = example_media_type_for(source_path)
        if media_type is None:
            raise HTTPError(
                400,
                f"Unsupported example media type: {src.suffix}",
                code="LORA_EXAMPLE_UNSUPPORTED_TYPE",
            )
        with self.lock:
            target = next(
                (i for i in self._items if f"imported-{i.id}" == entry_id),
                None,
            )
            if target is None:
                raise HTTPError(
                    404, f"Unknown imported LoRA: {entry_id}", code="IMPORT_LORA_NOT_FOUND"
                )
            dest_dir = self._library_dir / target.id
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Remove any prior example (different extension) before writing the
            # new one so there's never more than one example per LoRA.
            if target.example_path:
                try:
                    Path(target.example_path).unlink(missing_ok=True)
                except OSError:
                    pass
            dest = dest_dir / f"example{src.suffix.lower()}"
            shutil.copy2(src, dest)
            updated = target.model_copy(update={"example_path": str(dest)})
            self._items = [updated if i.id == target.id else i for i in self._items]
            self._persist_unlocked()
            return self._to_entry(updated)

    def clear_example(self, entry_id: str) -> None:
        """Remove an imported LoRA's example media (file + ledger field)."""
        with self.lock:
            target = next(
                (i for i in self._items if f"imported-{i.id}" == entry_id),
                None,
            )
            if target is None:
                raise HTTPError(
                    404, f"Unknown imported LoRA: {entry_id}", code="IMPORT_LORA_NOT_FOUND"
                )
            if not target.example_path:
                return
            example_path = target.example_path
            updated = target.model_copy(update={"example_path": None})
            self._items = [updated if i.id == target.id else i for i in self._items]
            self._persist_unlocked()
        try:
            Path(example_path).unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_entry(item: ImportedLora) -> LoraInferenceEntryApi:
        return LoraInferenceEntryApi(
            id=f"imported-{item.id}",
            kind="imported",
            variant=item.variant,
            name=item.name,
            conditioningTypes=[],
            localPath=item.local_path,
            available=True,
            sourceTrainingId=None,
            description=item.description,
            createdAt=item.created_at,
            fileSizeBytes=item.file_size_bytes,
            huggingfaceUrl=item.huggingface_url,
            triggerWord=item.trigger_word,
            exampleMediaType=example_media_type_for(item.example_path)
            if item.example_path
            else None,
        )

    def _load_file_unlocked(self) -> ImportedLorasState:
        if not self._ledger_file.exists():
            return ImportedLorasState()
        try:
            with open(self._ledger_file, "r", encoding="utf-8") as f:
                return ImportedLorasState.model_validate(json.load(f))
        except Exception as exc:
            backup = self._ledger_file.with_name(
                f"{self._ledger_file.stem}.corrupt-{int(time.time())}.json"
            )
            try:
                shutil.copy2(self._ledger_file, backup)
            except Exception:
                pass
            logger.warning(
                "%s could not be parsed; starting empty: %s", self._ledger_file.name, exc
            )
            return ImportedLorasState()

    def _persist_unlocked(self) -> None:
        path = self._ledger_file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                ImportedLorasState(items=self._items).model_dump(mode="json"),
                f,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
