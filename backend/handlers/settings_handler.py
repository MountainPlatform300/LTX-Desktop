"""Settings state mutations and persistence."""

from __future__ import annotations

import json
import logging
from threading import RLock
from typing import TYPE_CHECKING

from secret_vault import SecretVault, SecretVaultError
from state.app_settings import AppSettings, UpdateSettingsRequest
from handlers._settings_utils import (
    collect_changed_paths,
    deep_merge_dicts,
    ensure_json_object,
    migrate_legacy_settings,
    strip_none_values,
)
from handlers.base import StateHandlerBase, with_state_lock
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

SECRET_SETTING_FIELDS = frozenset(
    {
        "ltx_api_key",
        "gemini_api_key",
        "fal_api_key",
        "pexels_api_key",
        "runpod_api_key",
        "hf_token",
    }
)


class SettingsHandler(StateHandlerBase):
    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)
        self._secret_vault = SecretVault.from_environment(
            config.app_data_dir / "credentials.vault"
        )

    @with_state_lock
    def load_settings(self, default_settings: AppSettings) -> AppSettings:
        settings_file = self.config.settings_file
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                migrated = migrate_legacy_settings(ensure_json_object(payload))
                legacy_secrets = {
                    field: value
                    for field in SECRET_SETTING_FIELDS
                    if isinstance((value := migrated.pop(field, None)), str) and value
                }
                if legacy_secrets and self._secret_vault is None:
                    raise SecretVaultError(
                        "Cannot migrate plaintext credentials without OS-backed key storage"
                    )
                vault = self._secret_vault
                vault_secrets = (
                    vault.load() if vault is not None else {}
                )
                if legacy_secrets:
                    vault_secrets.update(legacy_secrets)
                    if vault is None:
                        raise SecretVaultError("Credential vault is unavailable")
                    vault.save(vault_secrets)
                merged = deep_merge_dicts(
                    ensure_json_object(default_settings.model_dump(by_alias=False)),
                    migrated,
                )
                merged.update(
                    {
                        field: value
                        for field, value in vault_secrets.items()
                        if field in SECRET_SETTING_FIELDS
                    }
                )
                loaded = AppSettings.model_validate(merged)
                logger.info("Settings loaded from %s", settings_file)
                self.state.app_settings = loaded
                if legacy_secrets:
                    self.save_settings()
                return loaded
            except SecretVaultError:
                raise
            except Exception as exc:
                logger.warning("Could not load settings: %s", exc, exc_info=True)

        fallback = default_settings.model_copy(deep=True)
        if self._secret_vault is not None:
            vault_secrets = self._secret_vault.load()
            fallback = fallback.model_copy(
                update={
                    field: value
                    for field, value in vault_secrets.items()
                    if field in SECRET_SETTING_FIELDS
                }
            )
        self.state.app_settings = fallback
        return self.state.app_settings

    def save_settings(self) -> None:
        try:
            payload = self.get_settings_snapshot().model_dump(by_alias=False)
            secrets = {
                field: value
                for field in SECRET_SETTING_FIELDS
                if isinstance((value := payload.pop(field, "")), str) and value
            }
            if secrets and self._secret_vault is None:
                raise SecretVaultError(
                    "Credential storage is unavailable; refusing plaintext persistence"
                )
            if self._secret_vault is not None:
                self._secret_vault.save(secrets)
            temporary = self.config.settings_file.with_suffix(
                f"{self.config.settings_file.suffix}.tmp"
            )
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(self.config.settings_file)
        except Exception as exc:
            logger.warning("Could not save settings: %s", exc, exc_info=True)
            raise

    @with_state_lock
    def get_settings_snapshot(self) -> AppSettings:
        return self.state.app_settings.model_copy(deep=True)

    @with_state_lock
    def update_settings(self, patch: UpdateSettingsRequest) -> tuple[AppSettings, AppSettings, set[str]]:
        patch_payload = strip_none_values(ensure_json_object(patch.model_dump(by_alias=False, exclude_unset=True)))

        for key_field in (
            "ltx_api_key",
            "gemini_api_key",
            "fal_api_key",
            "pexels_api_key",
            "runpod_api_key",
            "hf_token",
        ):
            if key_field in patch_payload and patch_payload[key_field] == "":
                del patch_payload[key_field]
        if (
            self._secret_vault is None
            and any(field in patch_payload for field in SECRET_SETTING_FIELDS)
        ):
            raise SecretVaultError(
                "Credential storage is unavailable; restart through the desktop app"
            )

        before = self.state.app_settings.model_copy(deep=True)
        before_payload = ensure_json_object(before.model_dump(by_alias=False))

        if patch_payload:
            merged_payload = deep_merge_dicts(before_payload, patch_payload)
            self.state.app_settings = AppSettings.model_validate(merged_payload)

        after = self.state.app_settings.model_copy(deep=True)
        after_payload = ensure_json_object(after.model_dump(by_alias=False))

        if "prompt_cache_size" in patch_payload and self.state.text_encoder is not None:
            self._trim_prompt_cache()

        changed_paths = collect_changed_paths(before_payload, after_payload)
        self.save_settings()
        return before, after, changed_paths

    @with_state_lock
    def clear_secret(self, field: str) -> None:
        """Explicitly remove one stored credential.

        Empty strings in the general settings patch remain a backwards-
        compatible "leave masked secret unchanged" signal. Credential removal
        therefore uses this dedicated operation instead of overloading that
        ambiguous patch value.
        """
        if field not in SECRET_SETTING_FIELDS:
            raise ValueError(f"Unsupported secret setting: {field}")
        self.state.app_settings = self.state.app_settings.model_copy(
            update={field: ""}
        )
        self.save_settings()

    def _trim_prompt_cache(self) -> None:
        te = self.state.text_encoder
        if te is None:
            return

        max_size = self.state.app_settings.prompt_cache_size
        while len(te.prompt_cache) > max_size:
            oldest = next(iter(te.prompt_cache))
            del te.prompt_cache[oldest]
