"""WSL2 post-mortem diagnostic for a local trainer run that died with no exit code.

A detached WSL command that is hard-killed (system OOM killer, `systemd-oomd`,
the WSL distro shutting down, or a native crash) dies before its wrapper can
record `$?`, so the runner sees "failed, no exit code". That message alone
doesn't say *why*. `wsl_postmortem()` grabs, for the failed job's systemd unit:

  * `systemctl show` — the unit's `Result` / `ExecMainStatus` / `ExecMainCode`
    (e.g. `Result=oom-kill`, `code=killed status=9 SIGKILL`).
  * `journalctl -u <unit>` — systemd's own messages about the unit, which is
    where `systemd-oomd` kills and native-crash exits are logged. The kernel
    `dmesg` ring buffer does NOT record `systemd-oomd` kills, so `dmesg` alone
    was a false negative.
  * `MemTotal` + `MemAvailable` + `SwapTotal` from `/proc/meminfo`.
  * Recent OOM / memory lines from `dmesg` (kernel OOM killer).

Best-effort: never raises, returns "" off-Windows. Quote-free command — see the
inline note on `wsl.exe` argv mangling.
"""

from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


def _decode_wsl(buf: bytes) -> str:
    # wsl.exe manager output is UTF-16LE (NUL bytes); distro output is UTF-8.
    if b"\x00" in buf:
        return buf.decode("utf-16le", errors="replace")
    return buf.decode("utf-8", errors="replace")


def _format_section(title: str, body: str) -> str:
    body = body.strip()
    if not body:
        return ""
    return f"[{title}]\n{body}"


def wsl_postmortem(unit: str | None = None) -> str:
    """Post-mortem snapshot for a no-exit-code local failure.

    If `unit` is given (the systemd unit name, e.g. `ltxjob_<job_id>`), includes
    `systemctl show <unit>` + `journalctl -u <unit>` — the definitive record of
    how the unit's main process ended (oom-kill / signal / crash). Always
    includes `MemTotal` + recent `dmesg` OOM lines. Empty off-Windows; never
    raises.
    """
    if platform.system() != "Windows":
        return ""
    # Quote-free command: `wsl.exe` re-tokenizes the argv after `--` and drops
    # embedded single/double quotes. The unit name is `ltxjob_<hex>` (safe, no
    # shell metacharacters), so it can be interpolated unquoted. `grep -e word`
    # terms are single words. Parse the sections in Python by sentinel markers.
    sections: list[str] = []
    if unit:
        sections.append(
            f"systemctl show {unit} -p Result -p ExecMainStatus -p ExecMainCode "
            f"-p ExecMainPID --no-pager; echo ___JOURNAL___; "
            f"journalctl -u {unit} --no-pager -n 40"
        )
    sections.append(
        "cat /proc/meminfo; echo ___DMESG___; "
        "dmesg 2>/dev/null | grep -i -e killed -e oom -e memory -e cgroup | tail -8"
    )
    script = "; echo ___SECTION___; ".join(sections)
    try:
        res = subprocess.run(
            ["wsl.exe", "-u", "root", "--", "bash", "-lc", script],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return f"postmortem failed: {exc}"
    text = _decode_wsl(res.stdout)

    out_parts: list[str] = []
    for raw in text.split("___SECTION___"):
        raw = raw.strip()
        if not raw:
            continue
        if unit and "___JOURNAL___" in raw:
            show_part, _, journal_part = raw.partition("___JOURNAL___")
            show_part = show_part.strip()
            journal_part = journal_part.strip()
            if show_part:
                out_parts.append(_format_section("systemctl show", show_part))
            if journal_part:
                out_parts.append(_format_section("unit journal", journal_part))
            continue
        if "___DMESG___" in raw:
            mem_part, _, dmesg_part = raw.partition("___DMESG___")
            mem_lines = [
                ln for ln in mem_part.splitlines()
                if any(k in ln for k in ("MemTotal", "MemAvailable", "SwapTotal"))
            ]
            mem_body = "\n".join(ln.strip() for ln in mem_lines if ln.strip())
            dmesg_body = dmesg_part.strip()
            if mem_body:
                out_parts.append(_format_section("meminfo", mem_body))
            if dmesg_body:
                out_parts.append(_format_section("dmesg oom", dmesg_body))
            continue
        out_parts.append(raw)

    if out_parts:
        return "\n".join(out_parts)
    err = _decode_wsl(res.stderr).strip()
    return f"postmortem: no output (exit {res.returncode}){(' — ' + err[:200]) if err else ''}"
