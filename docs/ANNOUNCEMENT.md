# Announcement draft

## Suggested title

LTX Desktop 1.1.0-beta.2: LoRA training installer beta

## Post

I have been working on an unofficial fork of
[LTX Desktop](https://github.com/Lightricks/LTX-Desktop) focused on making
LTX-2 LoRA training usable from the desktop app.

The first beta adds:

- Standard and IC-LoRA dataset preparation and training;
- local WSL2 or RunPod compute selection;
- official LTX-2 training profiles and optional custom profiles;
- live RunPod availability, cost estimates, billing visibility, recovery, and
  app-owned stop/terminate controls;
- validation samples, a LoRA library, prompt templates, Hugging Face publishing,
  portable dataset export, and ComfyUI-oriented outputs;
- security hardening for credentials, archives, remote downloads, model
  formats, local backend access, trainer revisions, and cloud-resource
  ownership.

This is an unofficial fork and is not an official Lightricks release.
Lightricks and the upstream contributors deserve credit for LTX Desktop and the
LTX models.

The beta includes **unsigned installers** for Windows x64, macOS
Apple Silicon, and Linux x64 while project-owned Windows signing and Apple
notarization are still being arranged. Windows SmartScreen and macOS
Gatekeeper warnings are expected. Download only from the repository's Releases
page and follow the linked step-by-step installation guide; do not use copies
from comments or mirrors.

Repository:
https://github.com/MountainPlatform300/LTX-Desktop

LoRA guide:
https://github.com/MountainPlatform300/LTX-Desktop/blob/main/docs/LORA_TRAINER.md

Installation guide:
https://github.com/MountainPlatform300/LTX-Desktop/blob/main/docs/DOWNLOAD.md

RunPod is optional and billable. The app shows active compute and lifecycle
controls, but users should still confirm that pods and storage are stopped or
terminated after training or failures.

The app retains upstream pseudonymous launch analytics to Lightricks, with a
visible opt-out under **Settings > General > Usage Analytics**. Prompts, media,
credentials, and paths are not included in the event payload. External-service
details are documented in the repository.

A video walkthrough will be linked from the GitHub release after its final
privacy review.

Feedback is welcome—especially clean setup reports, dataset UX issues, RunPod
recovery cases, and exported LoRA compatibility. Please use GitHub Issues for
bugs and private vulnerability reporting for security concerns.

## Publication checklist

- Replace branch links with the immutable release tag where practical.
- Add the privacy-reviewed tutorial URL.
- Confirm installers, runtime parts, checksums, SBOM, and attestations are
  attached to the release.
- Re-read the post for claims that changed after final release verification.
- Do not include API keys, pod/volume IDs, usernames, local paths, or private
  project names in attached screenshots or video.
