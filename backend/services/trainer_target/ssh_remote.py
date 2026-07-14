"""Paramiko-based SSH exec + file transport for the RunPod trainer backend.

RunPod pods are reachable over SSH, so the "run a detached command, poll it,
ship files" mechanics live here, separate from how the pod's SSH connection
details are obtained. `paramiko` is imported lazily so a user who never trains
doesn't pay for it at startup.

Detached command model (no long-lived SSH session needed between
poll ticks): each `run_detached` creates a remote job dir
``~/.ltx_jobs/<job_id>`` and launches a wrapper via ``nohup`` that runs
the command, tees stdout/stderr to ``log``, and writes the integer exit
code to ``status`` when done. Polling reads ``status`` (absent ->
running). Logs are the tail of ``log``. Terminate kills the recorded
process group. This survives the SSH connection dropping between ticks
and an app restart (the job keeps running on the remote).
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import shlex
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from services.trainer_target.trainer_target import (
    RemoteCommandStatus,
    TrainerTargetError,
)

if TYPE_CHECKING:
    import paramiko

_JOBS_ROOT = "~/.ltx_jobs"
_CONNECT_TIMEOUT = 30
_MAX_TRUSTED_HOSTS = 256

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSHConnection:
    host: str
    port: int
    username: str
    password: str | None = None
    key_filename: str | None = None


class SshHostTrustStore:
    """Persist trust-on-first-use SSH host keys by stable workload identity."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = RLock()

    def verify_or_record(
        self,
        *,
        identity: str,
        hostname: str,
        port: int,
        key_type: str,
        key_base64: str,
    ) -> None:
        with self._lock:
            entries = self._load_unlocked()
            existing = entries.get(identity)
            if existing is not None:
                if (
                    existing.get("keyType") != key_type
                    or existing.get("keyBase64") != key_base64
                ):
                    raise TrainerTargetError(
                        "SSH host identity changed for "
                        f"{identity}. Refusing to connect because this may indicate "
                        "an intercepted connection or a rebuilt pod. If the pod "
                        "was intentionally rebuilt, terminate it and retry with a "
                        "new pod.",
                        retryable=False,
                    )
                return

            entries[identity] = {
                "keyType": key_type,
                "keyBase64": key_base64,
                "firstSeenAt": datetime.now(timezone.utc).isoformat(),
                "endpoint": f"{hostname}:{port}",
            }
            if len(entries) > _MAX_TRUSTED_HOSTS:
                oldest = min(
                    entries,
                    key=lambda item: str(entries[item].get("firstSeenAt", "")),
                )
                if oldest != identity:
                    entries.pop(oldest, None)
            self._persist_unlocked(entries)
            logger.info("Trusted first SSH host key for %s", identity)

    def _load_unlocked(self) -> dict[str, dict[str, object]]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainerTargetError(
                "Could not read the SSH host trust store; refusing an "
                "unverified remote connection.",
                retryable=False,
            ) from exc
        entries = raw.get("hosts") if isinstance(raw, dict) else None
        if not isinstance(entries, dict):
            raise TrainerTargetError(
                "The SSH host trust store is invalid; refusing an unverified "
                "remote connection.",
                retryable=False,
            )
        return {
            str(identity): value
            for identity, value in entries.items()
            if isinstance(value, dict)
        }

    def _persist_unlocked(self, entries: dict[str, dict[str, object]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(f"{self._path.suffix}.tmp")
        tmp.write_text(
            json.dumps({"schemaVersion": 1, "hosts": entries}, indent=2),
            encoding="utf-8",
        )
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp, self._path)


def _import_paramiko() -> Any:
    try:
        import paramiko  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise TrainerTargetError(
            "paramiko is required for SSH-based training targets but is not "
            "installed in the backend environment",
            retryable=False,
        ) from exc
    return paramiko


class SSHRemote:
    """Stateless-ish SSH helper: opens a fresh client per operation.

    Opening a connection per call (rather than holding one open across
    the reconciler's multi-minute poll cadence) keeps things robust to
    idle disconnects and pod restarts at the cost of a little handshake
    overhead — negligible against the job durations here.
    """

    def __init__(
        self,
        connection: SSHConnection,
        *,
        trust_store: SshHostTrustStore | None = None,
        trust_identity: str | None = None,
    ) -> None:
        self._conn = connection
        self._trust_store = trust_store
        self._trust_identity = trust_identity

    def _client(self) -> "paramiko.SSHClient":
        paramiko = _import_paramiko()
        client = paramiko.SSHClient()
        if self._trust_store is not None and self._trust_identity is not None:
            store = self._trust_store
            identity = self._trust_identity
            port = self._conn.port

            class _TrustOnFirstUsePolicy(paramiko.MissingHostKeyPolicy):
                def missing_host_key(self, _client: object, hostname: str, key: Any) -> None:
                    store.verify_or_record(
                        identity=identity,
                        hostname=hostname,
                        port=port,
                        key_type=str(key.get_name()),
                        key_base64=str(key.get_base64()),
                    )

            client.set_missing_host_key_policy(_TrustOnFirstUsePolicy())
        else:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(
                hostname=self._conn.host,
                port=self._conn.port,
                username=self._conn.username,
                password=self._conn.password,
                key_filename=self._conn.key_filename,
                timeout=_CONNECT_TIMEOUT,
                allow_agent=self._conn.key_filename is None,
                look_for_keys=self._conn.key_filename is None,
            )
        except TrainerTargetError:
            raise
        except Exception as exc:
            raise TrainerTargetError(
                f"SSH connection to {self._conn.host}:{self._conn.port} failed: {exc}",
                retryable=True,
            ) from exc
        return client

    def _exec(self, client: "paramiko.SSHClient", command: str) -> tuple[int, str, str]:
        _, stdout, stderr = client.exec_command(command, timeout=_CONNECT_TIMEOUT)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        return code, out, err

    def test_connection(self) -> None:
        client = self._client()
        try:
            code, _, err = self._exec(client, "echo ok")
            if code != 0:
                raise TrainerTargetError(f"Remote echo failed: {err}", retryable=True)
        finally:
            client.close()

    def run(self, command: str) -> tuple[int, str, str]:
        """Run a command synchronously; return (exit_code, stdout, stderr).

        For short, blocking probes (e.g. a marker-file check). Long jobs
        should still use `run_detached` + `poll`.
        """
        client = self._client()
        try:
            return self._exec(client, command)
        finally:
            client.close()

    def run_detached(self, *, command: str, workdir: str) -> str:
        job_id = uuid.uuid4().hex
        job_dir = f"{_JOBS_ROOT}/{job_id}"
        # Launch the user command fully detached and return immediately, even
        # though the command itself may run for many minutes (provisioning,
        # training). Two things are essential:
        #   1. `mkdir`/`cd` run in the FOREGROUND (before the `&`), so the job
        #      dir exists and the cwd is set before we record the pid. Wrapping
        #      the launch + pid-write in a `{ ...; }` group keeps the `&` from
        #      backgrounding the whole chain (which raced the pid write and
        #      captured the wrong PID).
        #   2. The backgrounded `setsid` redirects ALL of stdin/stdout/stderr
        #      (to /dev/null; the real command's output already goes to `log`),
        #      so it doesn't keep the SSH channel's fds open. Without this,
        #      `exec_command(...).read()` blocks until the long command exits
        #      and trips the channel timeout (the TimeoutError we hit).
        #   3. The command is wrapped in a SUBSHELL `( ... )` so the `> log`
        #      redirect covers EVERY line of a multi-line script (not just the
        #      last one — that bug sent all provisioning output to /dev/null).
        #      A subshell (not a `{ }` group) means an internal `set -e` abort
        #      exits only the subshell, so the outer shell still writes the
        #      real exit code to `status` instead of hanging until timeout.
        inner = (
            "(\n"
            + command
            + f"\n) > {job_dir}/log 2>&1\n"
            + f"echo $? > {job_dir}/status"
        )
        wrapper = (
            f"mkdir -p {job_dir} && cd {shlex.quote(workdir)} && "
            f"{{ setsid bash -c {shlex.quote(inner)} < /dev/null "
            f"> {job_dir}/launcher.log 2>&1 & "
            f"echo $! > {job_dir}/pid; }}"
        )
        client = self._client()
        try:
            code, _, err = self._exec(client, wrapper)
            if code != 0:
                raise TrainerTargetError(
                    f"Failed to launch remote command: {err}", retryable=True
                )
        finally:
            client.close()
        return job_id

    def poll(self, job_id: str) -> RemoteCommandStatus:
        job_dir = f"{_JOBS_ROOT}/{job_id}"
        client = self._client()
        try:
            code, out, _ = self._exec(
                client,
                f"if [ -f {job_dir}/status ]; then cat {job_dir}/status; "
                f"elif PID=$(cat {job_dir}/pid 2>/dev/null) "
                f"&& [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; "
                "then echo __RUNNING__; "
                f"else echo __DEAD__; tail -c 1200 {job_dir}/launcher.log "
                "2>/dev/null || true; fi",
            )
            text = out.strip()
            if code != 0 or text == "__RUNNING__":
                return RemoteCommandStatus(state="running")
            if text.startswith("__DEAD__"):
                detail = text.removeprefix("__DEAD__").strip()
                return RemoteCommandStatus(
                    state="failed",
                    error=detail
                    or "Remote command exited before recording its status",
                )
            try:
                exit_code = int(text)
            except ValueError:
                return RemoteCommandStatus(state="running")
            if exit_code == 0:
                return RemoteCommandStatus(state="succeeded", exit_code=0)
            return RemoteCommandStatus(
                state="failed",
                exit_code=exit_code,
                error=f"Remote command exited with code {exit_code}",
            )
        finally:
            client.close()

    def read_logs(self, job_id: str, tail: int) -> list[str]:
        job_dir = f"{_JOBS_ROOT}/{job_id}"
        client = self._client()
        try:
            _, out, _ = self._exec(
                client, f"tail -n {int(tail)} {job_dir}/log 2>/dev/null || true"
            )
            return out.splitlines()
        finally:
            client.close()

    def terminate(self, job_id: str) -> None:
        job_dir = f"{_JOBS_ROOT}/{job_id}"
        client = self._client()
        try:
            # Negative PID -> kill the whole process group started by setsid.
            self._exec(
                client,
                f"PID=$(cat {job_dir}/pid 2>/dev/null); "
                f'[ -n "$PID" ] && kill -TERM -"$PID" 2>/dev/null || true',
            )
        finally:
            client.close()

    def upload_directory(self, *, local_dir: str, remote_dir: str) -> None:
        client = self._client()
        try:
            self._exec(client, f"mkdir -p {shlex.quote(remote_dir)}")
            sftp = client.open_sftp()
            try:
                self._sftp_put_dir(sftp, Path(local_dir), remote_dir)
            finally:
                sftp.close()
        finally:
            client.close()

    def _sftp_put_dir(self, sftp: Any, local_dir: Path, remote_dir: str) -> None:
        self._sftp_mkdirs(sftp, remote_dir)
        for entry in sorted(local_dir.iterdir()):
            remote_path = posixpath.join(remote_dir, entry.name)
            if entry.is_dir():
                self._sftp_put_dir(sftp, entry, remote_path)
            else:
                sftp.put(str(entry), remote_path)

    def _sftp_mkdirs(self, sftp: Any, remote_dir: str) -> None:
        try:
            sftp.stat(remote_dir)
            return
        except IOError:
            pass
        parent = posixpath.dirname(remote_dir.rstrip("/"))
        if parent and parent != remote_dir:
            self._sftp_mkdirs(sftp, parent)
        try:
            sftp.mkdir(remote_dir)
        except IOError:
            # Raced or exists; tolerate.
            pass

    def download_file(self, *, remote_path: str, local_path: str) -> None:
        client = self._client()
        try:
            sftp = client.open_sftp()
            try:
                try:
                    attrs = sftp.stat(remote_path)
                except IOError as exc:
                    raise TrainerTargetError(
                        f"Remote artifact not found: {remote_path}", retryable=False
                    ) from exc
                if attrs.st_mode is not None and stat.S_ISDIR(attrs.st_mode):
                    raise TrainerTargetError(
                        f"Expected a file but found a directory: {remote_path}",
                        retryable=False,
                    )
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote_path, local_path)
            finally:
                sftp.close()
        finally:
            client.close()
