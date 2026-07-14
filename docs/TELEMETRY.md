# Telemetry

This unofficial fork retains LTX Desktop's minimal, pseudonymous telemetry so
Lightricks can understand aggregate app usage. The app does not include names,
emails, generated content, prompts, credentials, file paths, or location in the
payload.

The endpoint necessarily receives normal connection metadata such as an IP
address. Lightricks controls server-side logging, retention, and any processing
of that metadata; this fork cannot independently guarantee those
server-side practices. Review the upstream provider's current privacy terms
before opting in.

The current allowlist sends only an app-launch event with a random installation
identifier (a random UUID persisted on the device), app version,
operating-system family, and the
`MountainPlatform300/LTX-Desktop` unofficial-fork identifier. Events are sent to
`https://ltx-desktop.lightricks.com/v2/ingest`.

## Opting out

Analytics is enabled by default. You can disable it at any time in **Settings >
General > Usage Analytics**. When disabled, no events are sent.

To disable telemetry before the first launch, create an `app_state.json` file in the app data folder with the following content:

```json
{ "analyticsEnabled": false }
```

App data folder locations:

- **Windows:** `%LOCALAPPDATA%\LTXDesktop\`
- **macOS:** `~/Library/Application Support/LTXDesktop/`
- **Linux:** `$XDG_DATA_HOME/LTXDesktop/` (default: `~/.local/share/LTXDesktop/`)

Your preference is respected immediately — no restart required.

## Implementation

The telemetry implementation is fully contained in [`electron/analytics.ts`](../electron/analytics.ts). Events are sent to an ingestion endpoint over HTTPS. No third-party analytics SDKs are used.
