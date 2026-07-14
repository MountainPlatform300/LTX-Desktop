"""Best-effort secret scrubbing for user-visible and persisted logs.

This is a defense-in-depth boundary, not a substitute for keeping secrets out
of command strings and exception messages. Keep the patterns deliberately
focused on credential-shaped fields so ordinary trainer output remains useful.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_REDACTED = "[REDACTED]"

_ENV_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?P<prefix>[A-Z][A-Z0-9_]*(?:_KEY|_TOKEN|_SECRET|_PASSWORD|_CREDENTIAL)"
    r"\s*=\s*)(?:'[^'\r\n]*'|\"[^\"\r\n]*\"|[^\s;|]+)"
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)\b(?P<prefix>authorization\s*[:=]\s*(?:bearer|basic|key)\s+)"
    r"[^\s,;]+"
)
_SECRET_HEADER_RE = re.compile(
    r"(?i)\b(?P<prefix>(?:x-goog-api-key|x-api-key)\s*[:=]\s*)[^\s,;]+"
)
_QUOTED_FIELD_RE = re.compile(
    r"""(?ix)
    (?P<prefix>
        ["']?(?:access[_-]?token|refresh[_-]?token|api[_-]?key|password|secret)["']?
        \s*[:=]\s*
    )
    (?P<quote>["'])
    .*?
    (?P=quote)
    """
)
_UNQUOTED_FIELD_RE = re.compile(
    r"(?i)\b(?P<prefix>(?:access[_-]?token|refresh[_-]?token|api[_-]?key|password|secret)"
    r"\s*[:=]\s*)[^\s,;}\]]+"
)
_SECRET_QUERY_RE = re.compile(
    r"(?i)(?P<prefix>[?&](?:access_token|refresh_token|api_key|apikey|key|token|secret|password)=)"
    r"[^&#\s]+"
)


def redact_text(text: str, *, known_secrets: Iterable[str] = ()) -> str:
    """Return ``text`` with credential-shaped values replaced.

    Explicit secrets are also removed wherever they occur. Values shorter than
    four characters are ignored to avoid destroying ordinary log words.
    """

    redacted = text
    for pattern in (
        _ENV_ASSIGNMENT_RE,
        _AUTHORIZATION_RE,
        _SECRET_HEADER_RE,
        _QUOTED_FIELD_RE,
        _UNQUOTED_FIELD_RE,
        _SECRET_QUERY_RE,
    ):
        redacted = pattern.sub(rf"\g<prefix>{_REDACTED}", redacted)

    for secret in sorted(
        {value for value in known_secrets if len(value) >= 4},
        key=len,
        reverse=True,
    ):
        redacted = redacted.replace(secret, _REDACTED)
    return redacted


def redact_lines(lines: Iterable[str], *, known_secrets: Iterable[str] = ()) -> list[str]:
    """Redact a sequence of log lines while preserving its order."""

    secrets = tuple(known_secrets)
    return [redact_text(line, known_secrets=secrets) for line in lines]
