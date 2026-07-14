"""Static type checking gate for backend sources."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


def _format_diagnostics(payload: dict[str, Any], limit: int = 15) -> str:
    diagnostics = payload.get("generalDiagnostics", [])
    if not isinstance(diagnostics, list):
        return "No diagnostics available."

    lines: list[str] = []
    for item in diagnostics[:limit]:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file", "<unknown>"))
        message = str(item.get("message", "<no message>")).splitlines()[0]
        range_data = item.get("range", {})
        start = range_data.get("start", {}) if isinstance(range_data, dict) else {}
        line = int(start.get("line", 0)) + 1 if isinstance(start, dict) else 0
        severity = str(item.get("severity", "unknown"))
        lines.append(f"{severity}: {file_name}:{line}: {message}")

    if not lines:
        return "No diagnostics available."
    return "\n".join(lines)


def test_pyright_has_no_errors_or_warnings() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["pyright", "--outputjson"],
            cwd=backend_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # `pyright` isn't on PATH for this invocation (e.g. running
        # `.venv\Scripts\python -m pytest` directly instead of `uv run pytest`,
        # which is what puts the venv's `pyright` shim on PATH). The gate is
        # still enforced under `pnpm backend:test` / CI, where `uv run` exposes
        # it — skip rather than error when the binary simply isn't launchable.
        pytest.skip("pyright is not on PATH for this invocation; run via `uv run pytest` to enforce the type-check gate")

    output = result.stdout.strip() or result.stderr.strip()
    assert output, "pyright produced no output"

    try:
        payload: dict[str, Any] = json.loads(output)
    except json.JSONDecodeError:
        normalized = output.lower()
        if "command not found" in normalized or "not found" in normalized:
            pytest.skip("pyright is not available on PATH; skipping type-check gate")
        raise AssertionError(
            "pyright did not produce valid JSON output.\n"
            f"returnCode={result.returncode}\n"
            f"{output}"
        ) from None
    summary = payload.get("summary", {})
    assert isinstance(summary, dict), "pyright JSON output missing summary"

    error_count = int(summary.get("errorCount", -1))
    warning_count = int(summary.get("warningCount", -1))

    assert error_count == 0 and warning_count == 0, (
        "pyright reported issues:\n"
        f"errorCount={error_count}, warningCount={warning_count}, returnCode={result.returncode}\n"
        f"{_format_diagnostics(payload)}"
    )
