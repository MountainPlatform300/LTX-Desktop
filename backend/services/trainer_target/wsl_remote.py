"""`wsl.exe`-based command exec + file transport for the local trainer backend.

The LoRA trainer is a Linux toolchain, but a Windows user with a CUDA-capable
GPU can run it locally inside WSL2 — no remote pod needed. This module mirrors
`SSHRemote` exactly (the same detached-job / poll / logs / terminate model) but
runs every command through `wsl.exe ... bash -lc <command>` via `subprocess`
instead of paramiko over SSH.

Detached command model (identical to the SSH backend): each `run_detached`
creates a job dir ``~/.ltx_jobs/<job_id>`` inside the WSL filesystem and
launches a wrapper via ``setsid``/``nohup`` that runs the command, tees
stdout/stderr to ``log``, and writes the integer exit code to ``status`` when
done. Polling reads ``status`` (absent -> running). Logs are the tail of
``log``. Terminate kills the recorded process group. The job keeps running
inside WSL across separate `wsl.exe` invocations and an app restart.

The bash strings handed in by `lora_command_builder` (with `set -e`, pipes,
subshells, exports) run verbatim: each is passed to `bash -lc` as a SINGLE
argv element, so the shell — not Python — parses it, and no `shell=True` is
used on the Windows side.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass

from services.trainer_target.trainer_target import (
    RemoteCommandStatus,
    TrainerTargetError,
)

logger = logging.getLogger(__name__)

_JOBS_ROOT = "~/.ltx_jobs"
# Short-op timeout (seconds). `run`/poll/launch are quick; long jobs go through
# `run_detached` + `poll`, so they never block this `subprocess.run` ceiling.
# A directory copy in `upload_directory` can take a while, so keep it generous.
_RUN_TIMEOUT = 600

# Detached-launch hardening. Unlike the SSH/RunPod backends this mirrors, WSL
# has no persistent daemon, and — critically — when `wsl.exe` is launched from a
# subprocess with piped stdio (as the backend does), the interop relay reaps any
# plain backgrounded job the moment that `wsl.exe` exits. The symptom is an empty
# pid file, no log, no status — and the poller then waits forever on a job that
# never ran. So the primary launch uses `systemd-run` to start the job as a
# transient unit owned by PID 1 (immune to the relay teardown); distros without
# systemd fall back to `setsid -f` plus a brief settle so the daemon fully
# detaches before `wsl.exe` returns. Either way we boot the distro first and then
# VERIFY the job actually started, retrying a few times.
_DETACH_LAUNCH_ATTEMPTS = 3
_LAUNCH_VERIFY_POLLS = 8
_LAUNCH_VERIFY_INTERVAL_SECONDS = 0.5
# How long the fallback launcher keeps the launching shell alive after
# `setsid -f` so the daemon reparents to init before the relay tears down.
_DETACH_SETTLE_SECONDS = 1


@dataclass(frozen=True)
class WslConnection:
    """Which WSL distribution to target. `None` = the user's default distro.

    `user` overrides the distro's default login user for every command. The
    local trainer runs as ``root`` (see `LocalTrainerTarget`): the workspace
    lives under ``/root/.ltx-desktop-lora`` and the model-load fix sets
    ``vm.overcommit_memory`` via `sysctl -w`, both of which require root.
    Forcing ``-u root`` also makes the app independent of whether the distro
    was installed via the in-app wizard (``--no-launch`` → default user root)
    or manually (first-launch prompt → a non-root user, which makes ``/root``
    unwritable and breaks workspace creation).
    """

    distro: str | None = None
    user: str | None = None


class WslRemote:
    """Stateless WSL helper: spawns a fresh `wsl.exe` process per operation.

    Mirrors `SSHRemote` so the trainer-execution model is identical across
    backends; the only difference is the transport (`wsl.exe` subprocess vs an
    SSH channel). Each operation is a short-lived process, which keeps the
    detached-job model robust across the reconciler's multi-minute poll cadence.
    """

    def __init__(self, connection: WslConnection) -> None:
        self._conn = connection
        # Resolved lazily and cached (a fresh WslRemote is cheap, but these are
        # one wsl.exe call each, so don't repeat them per operation).
        self._jobs_root: str | None = None
        self._systemd_ok: bool | None = None
        self._login_env: dict[str, str] | None = None
        # Per-job keepalive `wsl.exe` processes. WSL2 shuts down the VM a few
        # seconds after the LAST `wsl.exe` process exits (its `vmIdleTimeout`),
        # and a systemd unit running inside does NOT count as activity — so the
        # runner's 15s reconcile gap (no `wsl.exe` calls between the launch-verify
        # and the first poll) lets the VM shut down mid-run, killing the job with
        # no exit code (the recurring "preprocess killed mid-run" failure). Each
        # `run_detached` starts a keepalive that blocks until the job is observed
        # finished by `poll`/`terminate`, holding the VM open for the whole run.
        # `cat` reads from stdin and blocks; closing stdin (in `_stop_keepalive`)
        # makes it EOF -> exit, so the wsl.exe ends cleanly with no orphan inside
        # the VM. Guarded by a lock: poll/terminate/launch can touch the dict.
        self._keepalive: dict[str, subprocess.Popen[bytes]] = {}
        self._keepalive_lock = threading.Lock()

    def _run_wsl(self, command: str) -> tuple[int, str, str]:
        """Run `command` in a login bash inside WSL; return (code, out, err).

        The script is fed on STDIN (`bash -l -s`) rather than as a `bash -lc
        <cmd>` argv argument. Passing a complex shell script as a Windows argv to
        `wsl.exe` mangles shell specials — most damagingly `$?`, which silently
        expands to empty there, so exit codes captured with `echo $? > status`
        were always wrong (and a failed job looked successful). Read from stdin,
        bash parses the script verbatim (set -e, pipes, subshells, `$?` all work)
        and no shell is involved on the Windows side.
        """
        argv: list[str] = ["wsl.exe"]
        if self._conn.distro:
            argv += ["-d", self._conn.distro]
        if self._conn.user:
            # Run as the configured user (root for the local trainer) rather
            # than the distro's default — see `WslConnection.user`.
            argv += ["-u", self._conn.user]
        # `-l` = login shell (sources profile, so PATH tweaks are honored, like
        # the SSH backend); `-s` = read the script from stdin.
        argv += ["bash", "-l", "-s"]
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, no shell
                argv,
                input=(command + "\n").encode("utf-8"),
                capture_output=True,
                timeout=_RUN_TIMEOUT,
            )
        except FileNotFoundError as exc:
            raise TrainerTargetError(
                "wsl.exe was not found — WSL2 is not installed or not on PATH",
                retryable=False,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TrainerTargetError(
                f"WSL command timed out after {_RUN_TIMEOUT}s", retryable=True
            ) from exc
        # `wsl.exe`'s own manager messages (e.g. the "not installed" notice it
        # prints when WSL or a distro is missing) are UTF-16LE; the bash child
        # process's output is UTF-8. A NUL byte signals UTF-16LE (ASCII there
        # has a NUL every other byte); otherwise it's the UTF-8 bash output.
        def _decode(raw: bytes) -> str:
            if b"\x00" in raw:
                try:
                    return raw.decode("utf-16-le")
                except UnicodeDecodeError:
                    pass
            return raw.decode("utf-8", "replace")

        out = _decode(completed.stdout)
        err = _decode(completed.stderr)
        return completed.returncode, out, err

    def test_connection(self) -> None:
        code, _, err = self._run_wsl("echo ok")
        if code != 0:
            raise TrainerTargetError(
                f"WSL echo failed: {err}", retryable=True
            )

    def run(self, command: str) -> tuple[int, str, str]:
        """Run a command synchronously; return (exit_code, stdout, stderr).

        For short, blocking probes (e.g. a marker-file check). Long jobs
        should still use `run_detached` + `poll`.
        """
        return self._run_wsl(command)

    def _jobs_root_path(self) -> str:
        """Absolute jobs root (`$HOME/.ltx_jobs`), resolved once.

        Resolving to an absolute path (no `~`) keeps the paths identical whether
        referenced from a login shell or from inside a systemd unit, whose HOME
        may differ.
        """
        if self._jobs_root is None:
            _, out, _ = self._run_wsl('printf %s "$HOME/.ltx_jobs"')
            self._jobs_root = out.strip() or "/root/.ltx_jobs"
        return self._jobs_root

    def _systemd_available(self) -> bool:
        """Whether `systemd-run` can be used (systemd is PID 1 and present).

        The modern Ubuntu-on-WSL default. When true, detached jobs become
        transient units owned by PID 1 — the only launch that reliably survives
        the wsl.exe/interop teardown.
        """
        if self._systemd_ok is None:
            _, out, _ = self._run_wsl(
                "[ -d /run/systemd/system ] && command -v systemd-run "
                ">/dev/null 2>&1 && echo yes || echo no"
            )
            self._systemd_ok = out.strip().endswith("yes")
        return self._systemd_ok

    def _login_env_args(self) -> str:
        """`--setenv=` args forwarding the login env into a systemd unit.

        A systemd transient unit does NOT inherit the interactive login
        environment, so HOME/USER/PATH are unset inside it — and the trainer
        scripts run under `set -u` and need `$HOME` (for `~`, uv, and HF caches)
        plus a PATH that includes the user's tools. We resolve these once from a
        login shell (where wsl.exe set HOME/USER and /etc/profile built PATH) and
        forward them explicitly.
        """
        if self._login_env is None:
            _, out, _ = self._run_wsl('printf "%s\\n%s\\n%s" "$HOME" "$USER" "$PATH"')
            lines = out.split("\n")
            keys = ("HOME", "USER", "PATH")
            env: dict[str, str] = {}
            for key, value in zip(keys, lines):
                value = value.strip()
                if value:
                    env[key] = value
            self._login_env = env
        return " ".join(
            f"--setenv={shlex.quote(f'{k}={v}')}" for k, v in self._login_env.items()
        )

    def _ensure_distro_warm(self) -> None:
        """Boot the distro with a trivial synchronous call.

        So the detached launch that follows isn't the cold first-contact, where
        WSL can shut the VM down the moment the launching process exits.
        """
        try:
            self._run_wsl("true")
        except TrainerTargetError:
            # If the distro genuinely can't start, the launch below surfaces the
            # real error; don't mask it here.
            pass

    def _detached_started(self, job_dir: str) -> bool:
        """Return True once the detached job has demonstrably taken hold.

        Within a short grace window, the job counts as started if the `status`
        file appears (it ran — even if instantly), the systemd unit is active,
        or the recorded process group is alive. Each probe is a `wsl.exe` call,
        which also keeps the distro warm across the fragile startup window.
        """
        check = (
            f"if [ -s {job_dir}/status ]; then echo done; "
            f'elif [ -s {job_dir}/unit ] && systemctl is-active --quiet "$(cat {job_dir}/unit)"; then echo alive; '
            f'elif P=$(cat {job_dir}/pid 2>/dev/null); [ -n "$P" ] && kill -0 -"$P" 2>/dev/null; then echo alive; '
            f"else echo no; fi"
        )
        for _ in range(_LAUNCH_VERIFY_POLLS):
            _, out, _ = self._run_wsl(check)
            token = out.strip()
            if token in ("done", "alive"):
                return True
            time.sleep(_LAUNCH_VERIFY_INTERVAL_SECONDS)
        return False

    def run_detached(self, *, command: str, workdir: str) -> str:
        # The user command is wrapped in a SUBSHELL so the `> log` redirect
        # covers every line and an internal `set -e` abort still records the real
        # exit code in `status`; the job also writes its own pid (for the
        # non-systemd fallback's group-kill/liveness).
        #
        # Boot the distro first, then VERIFY the job actually started, retrying a
        # few times — see the module note on why a plain `&` background job is
        # reaped here.
        self._ensure_distro_warm()
        root = self._jobs_root_path()
        use_systemd = self._systemd_available()

        last_err = "unknown error"
        for _attempt in range(_DETACH_LAUNCH_ATTEMPTS):
            job_id = uuid.uuid4().hex
            job_dir = f"{root}/{job_id}"
            inner = (
                f"echo $$ > {job_dir}/pid\n"
                f"cd {shlex.quote(workdir)}\n"
                "(\n"
                + command
                + f"\n) > {job_dir}/log 2>&1\n"
                + f"echo $? > {job_dir}/status"
            )
            if use_systemd:
                unit = f"ltxjob_{job_id}"
                # Transient unit owned by PID 1; `--collect` reaps it on exit so
                # finished units don't accumulate. `--setenv` forwards the login
                # env (HOME/USER/PATH) the unit would otherwise lack.
                # `LimitNOFILE` raises the unit's open-file-descriptor ceiling so
                # the trainer's dataloader/tensor-sharing FDs don't overrun the
                # default 1024 ("Too many open files"). The `unit` marker lets
                # poll / terminate find it later, statelessly.
                wrapper = (
                    f"mkdir -p {job_dir} && : > {job_dir}/log && "
                    f"printf %s {shlex.quote(unit)} > {job_dir}/unit && "
                    f"systemd-run --quiet --collect --property=LimitNOFILE=1048576 "
                    f"{self._login_env_args()} --unit={unit} "
                    f"bash -lc {shlex.quote(inner)}"
                )
            else:
                # No systemd: daemonize with `setsid -f` and hold the launching
                # shell open briefly so the daemon reparents to init before the
                # relay tears down. `setsid` makes the pid a group leader, so the
                # recorded pid doubles as the process-group id.
                wrapper = (
                    f"mkdir -p {job_dir} && : > {job_dir}/log && "
                    f"setsid -f bash -c {shlex.quote(inner)} < /dev/null > /dev/null 2>&1; "
                    f"sleep {_DETACH_SETTLE_SECONDS}"
                )
            code, _, err = self._run_wsl(wrapper)
            if code == 0 and self._detached_started(job_dir):
                self._start_keepalive(job_id)
                return job_id
            last_err = (
                err.strip()
                or "the background job did not start (no live unit, pid, or exit status)"
            )
            # Clear the dead job dir before retrying with a fresh id.
            self._run_wsl(f"rm -rf {job_dir}")
        raise TrainerTargetError(
            f"Failed to launch WSL command: {last_err}", retryable=True
        )

    def _start_keepalive(self, job_id: str) -> None:
        """Start a blocking `wsl.exe` that holds the VM open for this job's run.

        WSL2 reaps the VM a few seconds after the last `wsl.exe` exits; without a
        keepalive the runner's multi-second poll gap lets the VM (and the job)
        die. `bash -lc cat` blocks reading stdin, so the `wsl.exe` stays alive
        until `_stop_keepalive` closes its stdin. Best-effort: a failure to start
        the keepalive is logged but never blocks the run (the job still runs; it
        just risks the idle-shutdown if a poll gap is long).
        """
        argv: list[str] = ["wsl.exe"]
        if self._conn.distro:
            argv += ["-d", self._conn.distro]
        if self._conn.user:
            argv += ["-u", self._conn.user]
        argv += ["bash", "-lc", "cat"]
        try:
            proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError) as exc:
            logger.warning("WslRemote keepalive failed to start for %s: %s", job_id, exc)
            return
        with self._keepalive_lock:
            self._keepalive[job_id] = proc
        logger.info("WslRemote keepalive started for job_id=%s (pid=%s)", job_id, proc.pid)

    def _stop_keepalive(self, job_id: str) -> None:
        """Stop the keepalive for a finished/terminated job (idempotent)."""
        with self._keepalive_lock:
            proc = self._keepalive.pop(job_id, None)
        if proc is None:
            return
        # Closing stdin makes the `cat` inside WSL hit EOF -> exit -> wsl.exe
        # exits, so no orphaned process is left inside the VM.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup
            logger.warning("WslRemote keepalive cleanup error for %s: %s", job_id, exc)
        logger.info("WslRemote keepalive stopped for job_id=%s", job_id)

    def poll(self, job_id: str) -> RemoteCommandStatus:
        job_dir = f"{self._jobs_root_path()}/{job_id}"
        # Read the exit-status file (source of truth — always written when the
        # job finishes, even on a non-zero command) AND a liveness signal, in one
        # call. Status is checked FIRST, so a job that just finished is reported
        # from its real exit code regardless of liveness. If there's no status
        # and the job is not alive, it died without recording a result (it likely
        # failed to start) — report failed rather than "running" forever.
        _, out, _ = self._run_wsl(
            f"cat {job_dir}/status 2>/dev/null; echo '|'; "
            f'if [ -s {job_dir}/unit ]; then systemctl is-active "$(cat {job_dir}/unit)" 2>/dev/null || true; '
            f'else P=$(cat {job_dir}/pid 2>/dev/null); '
            f'{{ [ -n "$P" ] && kill -0 -"$P" 2>/dev/null && echo active; }} || echo inactive; fi'
        )
        status_text, _, live_text = out.partition("|")
        status_text = status_text.strip()
        live = live_text.strip()

        if status_text != "":
            try:
                exit_code = int(status_text)
            except ValueError:
                return RemoteCommandStatus(state="running")
            if exit_code == 0:
                result = RemoteCommandStatus(state="succeeded", exit_code=0)
            else:
                result = RemoteCommandStatus(
                    state="failed",
                    exit_code=exit_code,
                    error=f"WSL command exited with code {exit_code}",
                )
        elif live in ("active", "activating", "reloading"):
            result = RemoteCommandStatus(state="running")
        elif live in ("inactive", "failed"):
            # Explicit dead signal: the unit/process is gone and wrote no exit
            # status. It either failed to start OR was killed mid-run (system OOM
            # killer, the WSL distro shutting down, or a native crash) before the
            # wrapper could record `$?`.
            result = RemoteCommandStatus(
                state="failed",
                error=(
                    "WSL job is no longer running but wrote no exit status. It "
                    "either failed to start OR was killed mid-run (system OOM "
                    "killer, the WSL distro shutting down, or a native crash) "
                    "before the wrapper could record `$?`. Check the log tail for "
                    "how far it got, then retry or resume."
                ),
            )
        else:
            # Empty/unknown liveness: the `wsl.exe` probe returned no signal at all
            # (a transient transport hiccup, or the VM is mid-restart). Do NOT
            # false-fail a job that may still be running — re-check on the next tick.
            result = RemoteCommandStatus(state="running")
        # A terminal state means the job is done — release the VM keepalive so the
        # VM can idle-shutdown once no other job needs it. If the job is still
        # running but has no keepalive (e.g. the backend restarted mid-run and the
        # handle was lost), self-heal by starting one so the VM stays up.
        if result.state != "running":
            self._stop_keepalive(job_id)
        else:
            with self._keepalive_lock:
                has_keepalive = job_id in self._keepalive
            if not has_keepalive:
                self._start_keepalive(job_id)
        return result

    def read_logs(self, job_id: str, tail: int) -> list[str]:
        job_dir = f"{self._jobs_root_path()}/{job_id}"
        _, out, _ = self._run_wsl(
            f"tail -n {int(tail)} {job_dir}/log 2>/dev/null || true"
        )
        return out.splitlines()

    def terminate(self, job_id: str) -> None:
        # Instrumented: a recurring "preprocess killed mid-run, no exit code" bug
        # turned out to be the WSL2 VM idle-shutting-down between polls (fixed by
        # the keepalive). The stack trace is retained to confirm any terminate is
        # an explicit cancel, never a spurious kill.
        logger.warning(
            "WslRemote.terminate called for job_id=%s — stack:\n%s",
            job_id,
            "".join(traceback.format_stack()),
        )
        job_dir = f"{self._jobs_root_path()}/{job_id}"
        # Stop the systemd unit if there is one; otherwise kill the whole process
        # group (negative PID) started by `setsid`.
        self._run_wsl(
            f'if [ -s {job_dir}/unit ]; then systemctl stop "$(cat {job_dir}/unit)" 2>/dev/null || true; '
            f'else PID=$(cat {job_dir}/pid 2>/dev/null); '
            f'[ -n "$PID" ] && kill -TERM -"$PID" 2>/dev/null || true; fi'
        )
        # The job is gone — release its VM keepalive.
        self._stop_keepalive(job_id)

    def _wslpath(self, windows_path: str) -> str:
        """Translate a Windows path to its WSL form via `wslpath -a`."""
        code, out, err = self._run_wsl(f"wslpath -a {shlex.quote(windows_path)}")
        translated = out.strip()
        if code != 0 or not translated:
            raise TrainerTargetError(
                f"Failed to translate Windows path {windows_path!r} to a WSL "
                f"path: {err or 'wslpath produced no output'}",
                retryable=False,
            )
        return translated

    def upload_directory(self, *, local_dir: str, remote_dir: str) -> None:
        """Copy a Windows directory tree into a WSL path (idempotent).

        `local_dir` is a Windows path; it's translated to its `/mnt/...` form
        with `wslpath`, then copied into `remote_dir` inside WSL. There's no
        network hop — this is a same-machine filesystem copy across the WSL
        boundary.
        """
        wsl_src = self._wslpath(local_dir)
        quoted_dst = shlex.quote(remote_dir)
        # `cp -a <src>/.` copies the directory CONTENTS (preserving attrs) into
        # the destination, matching the SSH backend's "merge into remote_dir".
        code, _, err = self._run_wsl(
            f"mkdir -p {quoted_dst} && cp -a {shlex.quote(wsl_src)}/. {quoted_dst}/"
        )
        if code != 0:
            raise TrainerTargetError(
                f"Failed to copy {local_dir} into WSL path {remote_dir}: {err}",
                retryable=True,
            )

    def download_file(self, *, remote_path: str, local_path: str) -> None:
        """Copy a single WSL file out to a Windows path.

        `remote_path` is a WSL path; `local_path` is a Windows path translated
        with `wslpath`. Mirrors `SSHRemote.download_file`'s not-found handling.
        """
        wsl_dst = self._wslpath(local_path)
        quoted_src = shlex.quote(remote_path)
        code, _, err = self._run_wsl(
            f"test -f {quoted_src}"
        )
        if code != 0:
            raise TrainerTargetError(
                f"WSL artifact not found: {remote_path}", retryable=False
            )
        quoted_dst = shlex.quote(wsl_dst)
        code, _, err = self._run_wsl(
            f"mkdir -p \"$(dirname {quoted_dst})\" && cp {quoted_src} {quoted_dst}"
        )
        if code != 0:
            raise TrainerTargetError(
                f"Failed to copy WSL file {remote_path} to {local_path}: {err}",
                retryable=True,
            )
