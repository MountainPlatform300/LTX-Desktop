"""LoRA inference registry — the list of LoRAs usable from Gen Space.

The registry is a read-only view derived from two sources:

* **Official adapters** — the LTX-2 IC-LoRA union-control checkpoint, which
  accepts canny / depth / pose control-signal conditioning. Its availability is
  whether the checkpoint file exists in the models dir.
* **User-trained adapters** — every completed `TrainingJob` whose
  `local_lora_path` is set and still on disk. The dataset type (standard vs
  IC-LoRA) is resolved by joining job → preprocessed → dataset, and drives the
  `variant` ("standard" for a t2v/i2v LoRA, "video_input_ic_lora" for a
  reference-conditioned IC-LoRA).

There is no separate persistence: a freshly completed training job appears here
automatically on the next `list_entries` call (the post-training bridge is the
`mark_training_completed` → `local_lora_path` assignment in the training
handler). Keeping the registry derived avoids drift between the training ledger
and what Gen Space can pick.
"""

from __future__ import annotations

from pathlib import Path

from api_types import (
    ControlConditioningType,
    LoraInferenceEntryApi,
    LoraInferenceVariantApi,
)
from handlers.imported_lora_library import ImportedLoraLibrary, example_media_type_for
from handlers.lora_prompt_template import (
    LoraPromptTemplateStore,
    build_default_prompt_template,
)
from handlers.lora_training_handler import LoraTrainingHandler
from runtime_config.model_download_specs import (
    ModelCheckpointID,
    is_cp_downloaded,
    resolve_model_path,
)
from state.lora_training_state import (
    LoraDataset,
    LoraDatasetType,
    PreprocessedDataset,
    TrainingJob,
)

# The single official union-control IC-LoRA checkpoint. canny/depth/pose all
# resolve to this file (the runtime difference is preprocessing, not weights).
_OFFICIAL_UNION_CP_ID: ModelCheckpointID = "ltx-2.3-22b-ic-lora-union-control-ref0.5"
_OFFICIAL_UNION_NAME = "LTX-2 IC-LoRA Union Control"
_OFFICIAL_UNION_DESCRIPTION = (
    "Official LTX-2 IC-LoRA for structural control via canny, depth, or pose."
)
_UNION_CONDITIONING: tuple[ControlConditioningType, ...] = ("canny", "depth", "pose")


class LoraInferenceRegistry:
    """Builds the list of LoRAs Gen Space can apply, derived from live state."""

    def __init__(
        self,
        *,
        training_handler: LoraTrainingHandler,
        imported_library: ImportedLoraLibrary,
        template_store: LoraPromptTemplateStore,
        models_dir: Path,
    ) -> None:
        self._training = training_handler
        self._imported = imported_library
        self._templates = template_store
        self._models_dir = models_dir

    def list_entries(self) -> list[LoraInferenceEntryApi]:
        return [
            self._with_template(self._official_union_entry()),
            *(self._with_template(e) for e in self._imported.list_entries()),
            *(self._with_template(e) for e in self._user_trained_entries()),
        ]

    def _with_template(self, entry: LoraInferenceEntryApi) -> LoraInferenceEntryApi:
        """Overlay the per-LoRA prompt template + trigger word onto an entry.

        A user-edited override wins; otherwise the trigger word comes from the
        entry itself (e.g. an imported LoRA's stored trigger word) and the
        template is synthesized from its behavior description / variant. `standard`
        style LoRAs get no template (the auto-prompt assistant needs a
        reference video, which they don't have).
        """
        override = self._templates.get_override(entry.id)
        # Trigger words are training metadata, not something that can be safely
        # guessed from a display name. An explicit override wins; otherwise use
        # only the exact token recorded by the import/training flow.
        trigger_word = (
            override.trigger_word
            if override is not None and override.trigger_word
            else entry.triggerWord
        )
        default_template = build_default_prompt_template(
            description=entry.description,
            variant=entry.variant,
            trigger_word=trigger_word,
            conditioning_types=tuple(entry.conditioningTypes),
        )
        prompt_template = (
            override.prompt_template
            if override is not None and override.prompt_template is not None
            else default_template
        )
        return entry.model_copy(
            update={
                "triggerWord": trigger_word,
                "promptTemplate": prompt_template,
                "promptTemplateCustomized": bool(
                    override is not None and override.prompt_template is not None
                ),
            }
        )

    def _official_union_entry(self) -> LoraInferenceEntryApi:
        available = is_cp_downloaded(self._models_dir, _OFFICIAL_UNION_CP_ID)
        local_path = (
            str(resolve_model_path(self._models_dir, _OFFICIAL_UNION_CP_ID))
            if available
            else None
        )
        file_size: int | None = None
        if available and local_path:
            try:
                file_size = Path(local_path).stat().st_size
            except OSError:
                file_size = None
        return LoraInferenceEntryApi(
            id="official-ic-lora-union",
            kind="official_union",
            variant="union_control",
            name=_OFFICIAL_UNION_NAME,
            conditioningTypes=list(_UNION_CONDITIONING),
            localPath=local_path,
            available=available,
            sourceTrainingId=None,
            description=_OFFICIAL_UNION_DESCRIPTION,
            createdAt=None,
            fileSizeBytes=file_size,
            huggingfaceUrl=None,
        )

    def _user_trained_entries(self) -> list[LoraInferenceEntryApi]:
        training_state = self._training.get_training_state()
        preprocessed_by_id: dict[str, PreprocessedDataset] = {
            p.id: p for p in self._training.get_preprocessed_state().items
        }
        datasets_by_id: dict[str, LoraDataset] = {
            d.id: d for d in self._training.get_datasets_state().datasets
        }
        entries: list[LoraInferenceEntryApi] = []
        for job in training_state.items:
            if job.status != "completed" or not job.local_lora_path:
                continue
            weights_path = Path(job.local_lora_path)
            if not weights_path.is_file():
                continue
            try:
                file_size = weights_path.stat().st_size
            except OSError:
                file_size = None
            dataset_type = self._dataset_type_for_job(
                job, preprocessed_by_id, datasets_by_id
            )
            entries.append(
                LoraInferenceEntryApi(
                    id=f"user-{job.id}",
                    kind="user_trained",
                    variant=_variant_for_dataset_type(dataset_type),
                    name=job.name,
                    conditioningTypes=[],
                    localPath=job.local_lora_path,
                    available=True,
                    sourceTrainingId=job.id,
                    description=job.description,
                    createdAt=job.created_at,
                    fileSizeBytes=file_size,
                    huggingfaceUrl=None,
                    triggerWord=job.config.trigger_word,
                    exampleMediaType=example_media_type_for(job.example_path)
                    if job.example_path
                    else None,
                )
            )
        return entries

    @staticmethod
    def _dataset_type_for_job(
        job: TrainingJob,
        preprocessed_by_id: dict[str, PreprocessedDataset],
        datasets_by_id: dict[str, LoraDataset],
    ) -> LoraDatasetType:
        pre = preprocessed_by_id.get(job.preprocessed_id)
        if pre is None:
            return "standard"
        dataset = datasets_by_id.get(pre.dataset_id)
        if dataset is None:
            return "standard"
        return dataset.type


def _variant_for_dataset_type(dataset_type: LoraDatasetType) -> LoraInferenceVariantApi:
    if dataset_type == "ic_lora":
        return "video_input_ic_lora"
    return "standard"
