"""Tests for defense-in-depth secret scrubbing."""

from secret_redaction import redact_lines, redact_text


def test_redacts_environment_assignments() -> None:
    text = (
        "HF_TOKEN=hf_example "
        "GEMINI_API_KEY='gemini secret' "
        'RUNPOD_API_KEY="runpod-secret" command'
    )

    redacted = redact_text(text)

    assert redacted == (
        "HF_TOKEN=[REDACTED] "
        "GEMINI_API_KEY=[REDACTED] "
        "RUNPOD_API_KEY=[REDACTED] command"
    )


def test_redacts_headers_fields_and_query_parameters() -> None:
    text = (
        "Authorization: Bearer bearer-secret "
        "x-goog-api-key=google-secret "
        '{"access_token": "oauth-secret"} '
        "https://example.test/callback?code=ok&token=query-secret"
    )

    redacted = redact_text(text)

    for secret in ("bearer-secret", "google-secret", "oauth-secret", "query-secret"):
        assert secret not in redacted
    assert redacted.count("[REDACTED]") == 4
    assert "code=ok" in redacted


def test_redacts_explicit_known_secrets_longest_first() -> None:
    redacted = redact_text(
        "transport failed for opaque-value-long",
        known_secrets=("opaque-value", "opaque-value-long"),
    )

    assert redacted == "transport failed for [REDACTED]"


def test_ignores_short_known_values_and_preserves_normal_output() -> None:
    text = "step 12/100 loss=0.42 token count=128"

    assert redact_text(text, known_secrets=("key", "")) == text


def test_redact_lines_preserves_order() -> None:
    assert redact_lines(["first", "HF_TOKEN=secret-value", "last"]) == [
        "first",
        "HF_TOKEN=[REDACTED]",
        "last",
    ]
