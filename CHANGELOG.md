# Changelog

All notable fork changes are documented here.

## Unreleased

No unreleased changes yet.

## 1.1.0-beta.2 - 2026-07-13

This beta adds unsigned installers for Windows x64, macOS arm64, and
Linux x64. Automatic installer updates remain disabled until signed
distribution is available.

### Added

- A signed-tag-gated, native cross-platform installer workflow with draft
  publication approval.
- Verified Windows/Linux runtime artifacts, installer-inclusive SHA-256
  checksums, SPDX SBOM, dependency evidence, and provenance attestations.
- Beginner download and installation instructions for SmartScreen, Gatekeeper,
  Debian packages, and AppImage.
- Privacy-reviewed LoRA workflow screenshots in the README and trainer guide.

### Security

- Installer publication fails closed on missing or oversized assets, package
  inventory violations, failed CI/security checks, or runtime integrity errors.
- Base Python archives and registry dependencies are verified against reviewed
  SHA-256 digests before packaging.
- Auto-update checks are disabled while installer binaries are unsigned.

## 1.1.0-beta.1 - 2026-07-13

This is the first source-only public beta of LTX Desktop. Signed
installers are not included.

### Added

- End-to-end standard and IC-LoRA dataset, training, validation, recovery,
  library, publishing, and export workflows.
- Local WSL2 and RunPod compute selection with live availability, estimated
  cost, billing visibility, ownership-scoped lifecycle controls, and recovery.
- App-native confirmations, LoRA help, archive management, and expanded tests.

### Security

- Replaced executable remote embedding deserialization with validated
  safetensors.
- Added encrypted OS-backed credential persistence and plaintext migration.
- Hardened archive import, filesystem containment, RunPod ownership, external
  downloads, trainer pinning, backend authentication, and secret redaction.
- Added a signed-tag-gated source-release workflow with SPDX SBOM, license
  evidence, SHA-256 checksums, and provenance attestations.
- Bound downloadable Python runtimes to a manifest embedded in the app and
  verified every archive part before extraction.
