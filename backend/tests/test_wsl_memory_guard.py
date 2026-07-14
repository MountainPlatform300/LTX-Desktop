"""Tests for the WSL2 post-mortem diagnostic.

The Windows `wsl.exe` shell-out can't run in CI (no WSL), so we assert the
off-Windows no-op contract and drive the output-parsing logic with a stubbed
`subprocess.run` (so the section formatting is verified without a VM).
"""

from __future__ import annotations

import platform

from services.wsl_memory.wsl_memory import wsl_postmortem


def test_wsl_postmortem_off_windows_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert wsl_postmortem() == ""
    assert wsl_postmortem(unit="ltxjob_abc") == ""


class _FakeCompleted:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_wsl_postmortem_parses_unit_and_meminfo_sections(monkeypatch) -> None:
    # Stub wsl.exe: the script emits `systemctl show`, the unit journal,
    # meminfo, and dmesg — separated by the sentinel markers the function
    # splits on. Verify each section is labeled and surfaced.
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    canned = (
        "Result=oom-kill\nExecMainStatus=0\nExecMainCode=0\n___JOURNAL___\n"
        "systemd-oomd[123]: Killed ltxjob_abc due to memory pressure\n"
        "ltxjob_abc: Main process exited, code=killed, status=9 SIGKILL\n"
        "___SECTION___\n"
        "MemTotal:       58617492 kB\nMemAvailable:    50000000 kB\nSwapTotal:       0 kB\n"
        "___DMESG___\n"
        "[ 12.3] Out of memory: Killed process 4321 (python)\n"
    )

    def fake_run(argv, capture_output, timeout, check):  # noqa: ANN001
        return _FakeCompleted(canned.encode("utf-8"))

    monkeypatch.setattr("services.wsl_memory.wsl_memory.subprocess.run", fake_run)
    out = wsl_postmortem(unit="ltxjob_abc")
    assert "[systemctl show]" in out
    assert "Result=oom-kill" in out
    assert "[unit journal]" in out
    assert "systemd-oomd" in out
    assert "SIGKILL" in out
    assert "[meminfo]" in out
    assert "MemTotal" in out
    assert "MemAvailable" in out
    assert "[dmesg oom]" in out
    assert "Killed process 4321" in out


def test_wsl_postmortem_no_unit_omits_unit_sections(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    canned = (
        "MemTotal:       58617492 kB\n___DMESG___\n"
        "[ 12.3] Out of memory: Killed process 4321 (python)\n"
    )

    def fake_run(argv, capture_output, timeout, check):  # noqa: ANN001
        return _FakeCompleted(canned.encode("utf-8"))

    monkeypatch.setattr("services.wsl_memory.wsl_memory.subprocess.run", fake_run)
    out = wsl_postmortem()
    assert "[systemctl show]" not in out
    assert "[unit journal]" not in out
    assert "[meminfo]" in out
    assert "MemTotal" in out
    assert "[dmesg oom]" in out
