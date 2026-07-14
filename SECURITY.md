# Security Policy

## Supported versions

LTX Desktop is currently beta software. Security fixes are applied
to the latest release and the default development branch; older builds may not
receive backports.

## Reporting a vulnerability

Do not report suspected vulnerabilities in a public issue.

Use this repository's
[private vulnerability reporting](https://github.com/MountainPlatform300/LTX-Desktop/security/advisories/new).
Include the
affected version, operating system, reproduction steps, impact, and any
suggested mitigation. Do not include real API keys, private media, or other
users' data.

## Security model

LTX Desktop is a single-user desktop application, not a multi-tenant
service. The Electron main process starts a FastAPI backend bound to the
loopback interface. Managed backend sessions use random authentication and
admin tokens. The renderer receives only the ordinary backend token; privileged
operations remain in the Electron main process.

Loopback binding does not make an untrusted local process safe. Software
running under the same operating-system account may be able to read app data,
inspect process memory, access local media, or interact with user-owned GPU and
cloud resources. Do not expose the backend port outside the local machine.

Development and test configurations may explicitly disable authentication.
Never use an insecure development backend on an untrusted or network-accessible
host.

## Installing releases

The installer beta is unsigned. Download it only from this repository's
[official Releases page](https://github.com/MountainPlatform300/LTX-Desktop/releases)
and follow [`docs/DOWNLOAD.md`](docs/DOWNLOAD.md). Do not trust copies posted in
issues, comments, chat, mirrors, or third-party download sites.

Each release includes SHA-256 checksums, an SPDX SBOM, dependency evidence, and
GitHub build-provenance attestations tied to a verified signed source tag.
These controls make tampering and build origin auditable, but they do not
provide Windows Authenticode reputation or Apple Developer ID notarization.
Automatic installer updates remain disabled until signed distribution is
qualified.

## Credentials and local data

API credentials and OAuth tokens are encrypted at rest using an authenticated
vault whose key is protected by the operating system's Electron `safeStorage`
facility. Legacy plaintext settings are migrated on load. RunPod SSH material
and generated artifacts remain local files. Protect the operating-system
account and app-data directory, use scoped/limited keys where available, do not
commit these files, and rotate credentials if the machine or a diagnostic
bundle may have been compromised.

Log redaction is defense in depth, not a guarantee. Backend, trainer, and
third-party tools can emit unexpected text. Review logs before sharing them and
remove prompts, paths, media names, tokens, and remote host details.

Generated media, imported datasets, model weights, training artifacts, and
captions remain on disk until the user deletes them or their containing project
or app-data directory.

## External services

Features are local unless their UI and settings select an API-backed provider.
Depending on the enabled feature, prompts, captions, media, model identifiers,
or training data may be sent to LTX API services, fal.ai, Google Gemini,
Pexels, Hugging Face, or RunPod. Those services have their own security,
privacy, retention, billing, and license terms.

Use least-privilege API keys where the provider supports scopes and budgets.
RunPod jobs can create billable compute and persistent storage; verify that
remote resources are stopped or terminated after failures and cancellations.
RunPod SSH uses trust on first use keyed by pod ID: a new pod is accepted
without user interaction, while later connections to that pod must present the
same host key. This protects reconnects but does not independently authenticate
the first connection to a newly rented pod.

### Remote prompt embeddings

Remote prompt embeddings must use safetensors and pass strict tensor type,
shape, size, and finite-value validation. Legacy pickle responses are rejected.
Do not point the client at an untrusted or unofficial prompt-embedding endpoint.

## Models, datasets, and trainer code

Treat imported model files, trainer repositories, datasets, symlinks, and
externally supplied paths as untrusted. Direct LoRA import and inference support
only `.safetensors`; legacy pickle-backed `.pt`, `.bin`, and `.ckpt` adapters
are blocked because loading them may execute code. Do not run custom trainer
forks or load executable model formats unless you have reviewed their code and
provenance.

Model licenses are separate from the application license. Review
[`NOTICES.md`](NOTICES.md) and the provider/model license before downloading,
training, redistributing, or publishing outputs.

## Telemetry

Pseudonymous product telemetry can be disabled in **Settings > General > Usage
Analytics**. See [`docs/TELEMETRY.md`](docs/TELEMETRY.md) for the
current event and storage policy.
