"""Authenticated at-rest storage for application credentials."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, cast

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_ENV_KEY = "LTX_SETTINGS_SECRET_KEY"
_VERSION = 1


class SecretVaultError(RuntimeError):
    pass


def _decode_key(encoded: str) -> bytes:
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        key = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeError) as exc:
        raise SecretVaultError("Settings secret key is not valid base64") from exc
    if len(key) != 32:
        raise SecretVaultError("Settings secret key must contain 32 random bytes")
    return key


class SecretVault:
    def __init__(self, path: Path, encoded_key: str) -> None:
        self._path = path
        self._cipher = AESGCM(_decode_key(encoded_key))

    @classmethod
    def from_environment(cls, path: Path) -> "SecretVault | None":
        encoded_key = os.environ.get(_ENV_KEY, "")
        return cls(path, encoded_key) if encoded_key else None

    def load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            raw_value = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw_value, dict):
                raise SecretVaultError("Credential vault is malformed")
            raw = cast(dict[str, Any], raw_value)
            if raw.get("version") != _VERSION:
                raise SecretVaultError("Credential vault has an unsupported format")
            encrypted_value = raw.get("secrets")
            if not isinstance(encrypted_value, dict):
                raise SecretVaultError("Credential vault is malformed")
            encrypted = cast(dict[Any, Any], encrypted_value)
            values: dict[str, str] = {}
            for field, token in encrypted.items():
                if not isinstance(field, str) or not isinstance(token, str):
                    raise SecretVaultError("Credential vault contains an invalid entry")
                packed = base64.urlsafe_b64decode(
                    (token + "=" * (-len(token) % 4)).encode("ascii")
                )
                if len(packed) < 13:
                    raise SecretVaultError("Credential vault entry is truncated")
                plaintext = self._cipher.decrypt(
                    packed[:12],
                    packed[12:],
                    field.encode("utf-8"),
                )
                values[field] = plaintext.decode("utf-8")
            return values
        except SecretVaultError:
            raise
        except Exception as exc:
            raise SecretVaultError("Credential vault could not be decrypted") from exc

    def save(self, values: dict[str, str]) -> None:
        encrypted: dict[str, str] = {}
        for field, value in values.items():
            if not value:
                continue
            nonce = os.urandom(12)
            ciphertext = self._cipher.encrypt(
                nonce,
                value.encode("utf-8"),
                field.encode("utf-8"),
            )
            encrypted[field] = base64.urlsafe_b64encode(
                nonce + ciphertext
            ).decode("ascii").rstrip("=")
        payload = {"version": _VERSION, "secrets": encrypted}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, self._path)
