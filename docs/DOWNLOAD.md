# Download and install

LTX Desktop is an unofficial fork currently in beta. Installers
are currently **unsigned**: Windows SmartScreen and macOS Gatekeeper will warn
because the project does not yet have commercial Windows or Apple signing
credentials.

Only download releases from:

**https://github.com/MountainPlatform300/LTX-Desktop/releases**

Do not use installers reposted in comments, chat messages, mirrors, or other
websites. Code signing can be added later without changing the app's data
folders.

## Choose a download

| Computer | Release asset |
| --- | --- |
| Windows 10/11, 64-bit | `LTX-Desktop-Setup.exe` |
| Mac with Apple Silicon (M1 or newer), macOS 13+ | `LTX-Desktop-arm64.dmg` |
| Ubuntu/Debian Linux, 64-bit | `LTX-Desktop-x64.deb` |
| Other compatible 64-bit Linux | `LTX-Desktop-x64.AppImage` |

Intel Macs, Windows on ARM, and ARM Linux packages are not currently provided.

## Windows

1. Open the official [Releases page](https://github.com/MountainPlatform300/LTX-Desktop/releases)
   and expand **Assets** under the newest release.
2. Download `LTX-Desktop-Setup.exe`.
3. Double-click the downloaded file.
4. If SmartScreen says **Windows protected your PC**, confirm the filename,
   click **More info**, then click **Run anyway**.
5. Follow the installer and launch **LTX Desktop**.

The first launch downloads and verifies a multi-gigabyte Python runtime from
the same GitHub release. Keep the app open and connected to the internet until
setup finishes.

## macOS

The macOS beta supports Apple Silicon only and uses API mode for generation.

1. Open the official [Releases page](https://github.com/MountainPlatform300/LTX-Desktop/releases)
   and download `LTX-Desktop-arm64.dmg`.
2. Open the DMG and drag **LTX Desktop** into **Applications**.
3. In Finder, open **Applications**, Control-click or right-click the app, and
   choose **Open**.
4. Confirm **Open** in the warning dialog.
5. If macOS still blocks it, open **System Settings → Privacy & Security**,
   find the blocked-app message, click **Open Anyway**, and authenticate.

Do not disable Gatekeeper globally. The app's Python runtime is included in the
DMG, so the download is substantially larger than the Windows/Linux installer.

## Ubuntu or Debian Linux

1. Download `LTX-Desktop-x64.deb` from the official Releases page.
2. Open the downloaded file with **App Center**, **Software Install**, or your
   distribution's package installer.
3. Click **Install** and authenticate when prompted.
4. Launch **LTX Desktop** from the applications menu.

The first launch downloads and verifies the matching Python runtime. A desktop
secret-service/keyring must be available so credentials can be stored safely.

## AppImage on Linux

1. Download `LTX-Desktop-x64.AppImage`.
2. Right-click the file, open **Properties → Permissions**, and enable
   **Allow executing file as program**.
3. Double-click the AppImage.

Some distributions require the FUSE 2 compatibility package (`libfuse2` or
`libfuse2t64`). Installing system packages is distribution-specific; use the
`.deb` package when available for the simplest setup.

## First-run downloads and disk space

- Windows/Linux download a verified Python runtime from the matching GitHub
  release automatically. Each numbered runtime part is checked against the
  bundled manifest before extraction.
- Local generation downloads model weights from Hugging Face and can require
  **160GB or more** of free disk space.
- macOS and computers without a supported NVIDIA GPU use API mode and require
  an LTX API key for video generation.
- Prompts or media leave the computer only when an API-backed feature is used.
- Pseudonymous usage analytics are enabled by default and can be disabled in
  **Settings → General → Usage Analytics**.

## Optional: verify a download

Every release includes `SHA256SUMS`. Compare the downloaded file's SHA-256
value with the line for that exact filename.

Windows PowerShell:

```powershell
Get-FileHash "$HOME\Downloads\LTX-Desktop-Setup.exe" -Algorithm SHA256
```

macOS:

```bash
shasum -a 256 ~/Downloads/LTX-Desktop-arm64.dmg
```

Linux:

```bash
sha256sum ~/Downloads/LTX-Desktop-x64.deb
```

Checksums detect corruption or replacement relative to the release manifest.
GitHub build-provenance attestations provide an additional link to the public
workflow that built each asset. Neither is a substitute for operating-system
code signing; this limitation is why these packages are labeled unsigned.

## Uninstall

- **Windows:** Settings → Apps → Installed apps → LTX Desktop →
  **Uninstall**.
- **macOS:** move LTX Desktop from Applications to Trash.
- **Linux `.deb`:** remove it using the same package manager used to install it.
- **Linux AppImage:** delete the AppImage file.

Uninstalling the app does not automatically delete projects, models, settings,
or credentials. Their locations are listed in the
[README](../README.md#first-run--data-locations).
