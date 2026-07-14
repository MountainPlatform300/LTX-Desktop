"""Integration test for the local-training eligibility probe endpoint.

`GET /api/lora/local-eligibility` is a read-only capability probe the UI
polls to decide whether to offer "train locally". It hits the REAL
`LocalTrainerTarget` (no mocks): the probe is side-effect-safe and never
raises, so on a machine without WSL2 (the CI/test machine) it reports
`eligible=false` with `wslInstalled=false` and a non-empty `reason`.
"""

from __future__ import annotations

import pytest


def _wsl_installed(client) -> bool:
    """True if the real eligibility probe reports WSL2 as installed.

    The local-eligibility endpoint hits the real `LocalTrainerTarget` (no
    mocks), so this reflects the actual host. Used to skip tests whose premise
    is "no WSL2" on machines that have it — the no-mock policy means we can't
    fake WSL's absence.
    """
    r = client.get("/api/lora/local-eligibility")
    assert r.status_code == 200, r.text
    return bool(r.json()["wslInstalled"])


class TestLocalEligibility:
    def test_response_shape(self, client) -> None:
        r = client.get("/api/lora/local-eligibility")
        assert r.status_code == 200, r.text
        body = r.json()
        # All mirrored fields present (camelCase API surface).
        assert set(body) == {
            "eligible",
            "reason",
            "wslInstalled",
            "cudaInWsl",
            "gpuName",
            "vramGb",
        }
        assert isinstance(body["eligible"], bool)
        assert isinstance(body["reason"], str)
        assert isinstance(body["wslInstalled"], bool)
        assert isinstance(body["cudaInWsl"], bool)
        assert body["gpuName"] is None or isinstance(body["gpuName"], str)
        assert body["vramGb"] is None or isinstance(body["vramGb"], int)

    def test_ineligible_without_wsl(self, client) -> None:
        # Only meaningful where WSL2 is absent; skip on machines that have it
        # (the no-mock policy means we can't fake WSL's absence).
        if _wsl_installed(client):
            pytest.skip("WSL2 is installed on this machine; the without-WSL path can't be exercised without mocking.")
        r = client.get("/api/lora/local-eligibility")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["eligible"] is False
        assert body["wslInstalled"] is False
        assert body["cudaInWsl"] is False
        assert body["reason"].strip() != ""
