# LTX Desktop 1.1.0-beta.1

This is the first source-only beta of the unofficial LTX Desktop
fork. It is based on the public Apache-2.0 Lightricks/LTX-Desktop project and is
not an official Lightricks release.

## Highlights

- End-to-end Standard and IC-LoRA dataset preparation, training, validation,
  recovery, library, publishing, and export workflows.
- Local WSL2 and RunPod training with live GPU availability, estimated cost,
  billing visibility, app-owned lifecycle controls, and recovery.
- Official LTX-2 training profiles plus user-created custom profiles.
- LoRA prompt templates, examples, archive management, GenSpace integration,
  and app-native confirmations.
- Security hardening for credentials, local backend authentication, dataset
  archives, external downloads, model formats, filesystem paths, trainer
  revisions, and RunPod ownership.

## Distribution

This beta contains source code only. It does not include a trusted Windows or
macOS installer, installer update, or Windows/Linux Python runtime archive.
Do not download unsigned installers or Python runtime archives from comments,
mirrors, or unofficial links.

Developers can run the tagged source with Node.js 24, Python 3.13.12, `uv`,
Git, and pnpm:

```bash
pnpm setup:dev
pnpm dev
```

See the repository README and `docs/LORA_TRAINER.md` for requirements and
workflow guidance.

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
- Signed installers, automatic installer updates, and installer clean-machine
  qualification are deferred to a later release.
- RunPod creates billable resources. Confirm pods and storage are stopped or
  terminated after training and failures.

Please report bugs through GitHub Issues and security vulnerabilities through
the repository's private vulnerability reporting form.
