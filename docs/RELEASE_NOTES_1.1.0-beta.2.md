# LTX Desktop 1.1.0-beta.2

This is the first installer beta of this unofficial LTX Desktop fork.
It is based on the public Apache-2.0 Lightricks/LTX-Desktop project and is not
an official Lightricks release.

## Download

Download assets only from this release on
`https://github.com/MountainPlatform300/LTX-Desktop/releases`.

- Windows 10/11 x64: `LTX-Desktop-Setup.exe`
- macOS 13+ on Apple Silicon: `LTX-Desktop-arm64.dmg`
- Ubuntu/Debian x64: `LTX-Desktop-x64.deb`
- Other compatible x64 Linux: `LTX-Desktop-x64.AppImage`

These installers are **unsigned**. Windows SmartScreen and macOS Gatekeeper
warnings are expected. Follow `docs/DOWNLOAD.md` for step-by-step installation
and optional SHA-256 verification. Automatic installer updates remain disabled;
download each beta from the official Releases page.

Windows and Linux fetch a matching, verified Python runtime on first launch.
The release includes its runtime manifests and numbered parts. macOS bundles
the runtime in the DMG and operates in API mode.

## Highlights

- End-to-end Standard and IC-LoRA dataset preparation, training, validation,
  recovery, library, publishing, and export workflows.
- Local WSL2 and RunPod training with live GPU availability, estimated cost,
  billing visibility, app-owned lifecycle controls, and recovery.
- Official LTX-2 training profiles plus user-created custom profiles.
- LoRA prompt templates, examples, archive management, Gen Space integration,
  and app-native confirmations.
- Security hardening for credentials, local backend authentication, dataset
  archives, external downloads, model formats, filesystem paths, trainer
  revisions, and RunPod ownership.
- Cross-platform installers built from the signed tag by public GitHub Actions,
  with SHA-256 checksums, an SPDX SBOM, dependency evidence, and GitHub
  provenance attestations.

## Privacy and external services

Optional features can contact LTX API services, Hugging Face, RunPod, Gemini,
fal.ai, and Pexels with user-selected data and user-supplied credentials.
Pseudonymous app-launch telemetry is sent to Lightricks by default and can be
disabled under **Settings > General > Usage Analytics**. See
`docs/NETWORK_SERVICES.md` and `docs/TELEMETRY.md`.

## Beta limitations

- Breaking changes remain possible.
- Fork-specific Hugging Face OAuth is not configured; use a Hugging Face
  token in Settings for gated models.
- The installers do not have Windows Authenticode or Apple Developer ID
  signatures and are not Apple-notarized.
- Automatic installer updates are disabled until signed distribution is
  qualified.
- Windows local LoRA training requires WSL2. macOS is API-only.
- RunPod creates billable resources. Confirm pods and storage are stopped or
  terminated after training and failures.

Please report bugs through GitHub Issues and security vulnerabilities through
the repository's private vulnerability reporting form.
