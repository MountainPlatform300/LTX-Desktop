"""Local SSH keypair management for auto-provisioned RunPod pods.

RunPod's templates add the `PUBLIC_KEY` env var to the pod's
`authorized_keys` on boot, so the cleanest hands-off auth story is: the
desktop app owns a dedicated keypair, injects the public half when it
creates a pod, and connects with the private half. This module owns that
keypair — generated once under the app data dir and reused thereafter —
so the user never has to register a key with RunPod by hand.

`paramiko` is imported lazily (like `ssh_remote`) so users who never
touch a remote trainer don't pay for it at startup. This file is in the
pyright exclude set alongside the other SSH adapters because the
paramiko surface is untyped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from services.trainer_target.trainer_target import TrainerTargetError

_KEY_BITS = 2048
_PRIVATE_KEY_NAME = "id_rsa"
_PUBLIC_KEY_NAME = "id_rsa.pub"
_KEY_COMMENT = "ltx-desktop-lora"


def _import_paramiko() -> Any:
    try:
        import paramiko  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise TrainerTargetError(
            "paramiko is required to generate the provisioning SSH key but is "
            "not installed in the backend environment",
            retryable=False,
        ) from exc
    return paramiko


class SshKeyManager:
    """Generates/loads a single RSA keypair under ``key_dir``."""

    def __init__(self, key_dir: Path) -> None:
        self._key_dir = key_dir
        self._private_path = key_dir / _PRIVATE_KEY_NAME
        self._public_path = key_dir / _PUBLIC_KEY_NAME

    @property
    def private_key_path(self) -> str:
        return str(self._private_path)

    def ensure_keypair(self) -> tuple[str, str]:
        """Return ``(private_key_path, public_key_openssh)``.

        Generates the keypair on first use (0600 private key); reuses the
        existing pair on every later call. The public string is in the
        ``ssh-rsa <base64> <comment>`` form RunPod's `PUBLIC_KEY` expects.
        """
        if self._private_path.exists() and self._public_path.exists():
            return self.private_key_path, self._public_path.read_text(
                encoding="utf-8"
            ).strip()

        paramiko = _import_paramiko()
        self._key_dir.mkdir(parents=True, exist_ok=True)
        key = paramiko.RSAKey.generate(_KEY_BITS)
        key.write_private_key_file(self.private_key_path)
        try:
            os.chmod(self.private_key_path, 0o600)
        except OSError:  # pragma: no cover - platform-dependent (Windows)
            pass
        public_line = f"ssh-rsa {key.get_base64()} {_KEY_COMMENT}"
        self._public_path.write_text(public_line + "\n", encoding="utf-8")
        return self.private_key_path, public_line
