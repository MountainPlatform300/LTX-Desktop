"""Poll robustness tests for `WslRemote`.

The Windows `wsl.exe` shell-out can't run in CI, so we drive `poll` directly
with a stubbed `_run_wsl` and assert the liveness semantics — in particular
that a transient empty `wsl.exe` response (no status, no liveness signal) is
treated as "running" rather than false-failing a still-running job.
"""

from __future__ import annotations

from services.trainer_target.wsl_remote import WslConnection, WslRemote


def _make_remote() -> WslRemote:
    r = WslRemote(WslConnection())
    # Pin the jobs root so `poll` doesn't probe `$HOME` via `_run_wsl`.
    r._jobs_root = "/root/.ltx_jobs"  # type: ignore[attr-defined]
    return r


def test_poll_transient_empty_output_keeps_running() -> None:
    # A transient `wsl.exe` hiccup returns empty output — no status, no
    # liveness signal. This must NOT be read as "failed"; the job may still be
    # running. Re-check on the next tick.
    r = _make_remote()
    r._run_wsl = lambda command: (0, "", "")  # type: ignore[assignment]
    status = r.poll("job-1")
    assert status.state == "running"
    assert status.exit_code is None


def test_poll_explicit_inactive_is_failed() -> None:
    # An explicit dead signal (`inactive` from `systemctl is-active` / the
    # `kill -0` fallback) with no status → genuinely dead → failed, no exit code.
    r = _make_remote()
    r._run_wsl = lambda command: (0, "|inactive", "")  # type: ignore[assignment]
    status = r.poll("job-1")
    assert status.state == "failed"
    assert status.exit_code is None


def test_poll_explicit_failed_unit_is_failed() -> None:
    r = _make_remote()
    r._run_wsl = lambda command: (0, "|failed", "")  # type: ignore[assignment]
    status = r.poll("job-1")
    assert status.state == "failed"
    assert status.exit_code is None


def test_poll_active_unit_is_running() -> None:
    r = _make_remote()
    r._run_wsl = lambda command: (0, "|active", "")  # type: ignore[assignment]
    status = r.poll("job-1")
    assert status.state == "running"


def test_poll_recorded_exit_code_wins() -> None:
    # Status file present → its exit code is authoritative, regardless of
    # liveness. 0 → succeeded.
    r = _make_remote()
    r._run_wsl = lambda command: (0, "0|inactive", "")  # type: ignore[assignment]
    status = r.poll("job-1")
    assert status.state == "succeeded"
    assert status.exit_code == 0


def test_poll_recorded_nonzero_exit_code_is_failed() -> None:
    r = _make_remote()
    r._run_wsl = lambda command: (0, "137|inactive", "")  # type: ignore[assignment]
    status = r.poll("job-1")
    assert status.state == "failed"
    assert status.exit_code == 137
