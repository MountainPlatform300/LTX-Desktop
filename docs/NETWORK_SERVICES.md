# External network services

LTX Desktop is a local desktop application, but selected features
contact external services. Provider terms, retention, security, and billing
policies apply independently of this project.

| Service | Purpose and data sent | Credential and control |
| --- | --- | --- |
| LTX API (`api.ltx.video`) | Text encoding, prompt enhancement, and optional cloud generation. Requests can include prompts and user-selected media. | User-supplied LTX API key. Cloud features are selected in settings or the generation flow. |
| Hugging Face (`huggingface.co`) | Model licenses, model metadata, README files, model/runtime downloads, and optional LoRA publishing. Uploaded LoRAs include files and model-card metadata chosen by the user. | User-supplied read/write token as required. Fork-specific OAuth is disabled until a project-owned client is registered. |
| GitHub (`github.com` and release asset hosts under `githubusercontent.com`) | Official installer downloads and matching Windows/Linux Python runtimes. Windows/Linux first run downloads multi-gigabyte numbered runtime parts from the same versioned release. Automatic installer updates are disabled during the unsigned beta. | No credential for public releases. Production runtime downloads reject other hosts and verify release manifests, part sizes, and SHA-256 digests before extraction. |
| RunPod (`rest.runpod.io`, `api.runpod.io`, rented pod SSH endpoints) | Optional remote LoRA preprocessing/training, pod and volume lifecycle, logs, and artifacts. Training datasets and captions are uploaded to the selected pod. | User-supplied RunPod API key and app-managed SSH material. Creates billable resources; users must verify stop/termination. |
| Google Gemini (`generativelanguage.googleapis.com`) | Optional captioning, prompt suggestions, and LoRA prompt profiling. Selected prompts, captions, metadata, or media can be sent. | User-supplied Gemini API key; feature is unused when not configured/selected. |
| fal.ai (`fal.run`, `rest.alpha.fal.ai`) | Optional image/video editing and restyling. Selected prompts and media are uploaded. | User-supplied fal.ai key; feature is unused when not configured/selected. |
| Pexels (`api.pexels.com` and allowlisted Pexels media hosts) | Optional stock-media search and download. Search queries and selected asset requests are sent. | User-supplied Pexels key. Downloads validate every redirect. |
| Lightricks telemetry (`ltx-desktop.lightricks.com`) | Allowlisted pseudonymous app-launch telemetry only: random installation ID, app version, OS family, distribution, and repository identifier. | No API credential. Enabled by default; disable in **Settings > General > Usage Analytics**. |
| Official LTX trainer and package sources (`github.com/Lightricks/LTX-2`, `download.pytorch.org`, `astral.sh`) | Pinned trainer checkout and dependency/bootstrap installation on a user-selected RunPod training machine. | Public downloads. Trainer repository and revision are pinned by the app. |

Model and API providers may return pre-signed upload or download URLs on their
own storage/CDN domains. The app uses those URLs only as part of the
provider-requested operation.

The app also opens provider account, API-key, billing, model-license, and model
pages in the system browser when the user clicks the corresponding link.

See [`TELEMETRY.md`](TELEMETRY.md) for the telemetry event policy and
[`SECURITY.md`](../SECURITY.md) for credential and local-data handling.
