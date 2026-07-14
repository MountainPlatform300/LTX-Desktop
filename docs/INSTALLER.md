# LTX Desktop - Installer Build Guide

This guide explains how to build a distributable installer for
**LTX Desktop**.

- For running from source and debugging: see [`README.md`](../README.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).
- For downloading and installing a published build: see [`DOWNLOAD.md`](DOWNLOAD.md).

> **Unsigned installer beta:** Only packages built by the gated
> `installer-release.yml` workflow and attached to this repository's official
> GitHub release are end-user builds. Local packages are development artifacts
> and must not be redistributed as official releases.

## What Gets Bundled

Every installer includes:
- **Electron app** (React frontend + Electron shell)
- **Backend Python code**

On macOS, the installer also includes embedded Python (version from
[`backend/.python-version`](../backend/.python-version)) with all dependencies:
  - PyTorch (CUDA on Windows/Linux, MPS on macOS)
  - FastAPI, Diffusers, Transformers
  - LTX-2 inference packages
  - All other required libraries

**NOT bundled** (downloaded at runtime):
- Model weights (downloaded on first run; can be large) from Hugging Face
- On **Linux** and **Windows**: the matching, integrity-checked Python
  environment from this fork's GitHub release

Python is isolated from the target system. On macOS it lives inside
`{install_dir}/resources/python/`; on Windows and Linux it lives in the
app-data directory.

## Prerequisites

Before building, ensure you have:

1. **Node.js 24** - https://nodejs.org/
2. **uv** - https://docs.astral.sh/uv/ (Python package manager)
3. **git** - needed for git-based Python packages
4. **Internet connection** (for downloading Python and packages)
5. **~15GB free space** (for Python environment + build artifacts; does not include model weights — see [README](../README.md) for full disk space requirements)

### Platform-Specific

- **Windows**: PowerShell 5.1+ (comes with Windows 10/11)
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)
- **Linux**: `build-essential` (or equivalent) for native extensions

## Quick Build

```bash
pnpm build
```

This auto-detects your platform and will:
1. Download a standalone Python distribution (version from [`backend/.python-version`](../backend/.python-version))
2. Install all Python dependencies for runtime verification and artifact creation
3. Build the frontend
4. Package everything with electron-builder
5. Create a local DMG (macOS), AppImage + deb (Linux), or NSIS installer
   (Windows) in the `release/` folder

Local packages are unsigned development artifacts. Published unsigned betas
still require the signed-tag, CI, checksum, SBOM, provenance, and clean-machine
gates in [`RELEASE.md`](RELEASE.md).

## Build Options

```bash
# Full build
pnpm build

# Skip Python setup (if already prepared)
pnpm build:skip-python

# Fast rebuild (unpacked, skip Python + pnpm install)
pnpm build:fast

# Just prepare Python environment
pnpm prepare:python
```

All commands auto-detect the current platform (macOS, Linux, or Windows).

### Build Script Options

The underlying `local-build.sh` / `local-build.ps1` scripts also accept:
- `--platform mac|linux|win` — Target platform (auto-detected if omitted)
- `--skip-python` — Use existing `python-embed/` directory
- `--clean` — Remove build artifacts before starting
- `--unpack` — Build unpacked app only (faster, no installer/DMG)

## Build Output

### macOS
```
release/
  └── LTX-Desktop-arm64.dmg
```

### Linux
```
release/
  ├── LTX-Desktop-x64.AppImage
  └── LTX-Desktop-x64.deb
```

### Windows
```
release/
  └── LTX-Desktop-Setup.exe
```

## Application Icon

Place icon files in `resources/` before building:
- `icon.ico` — Windows (multi-size ICO: 256x256, 128x128, 64x64, 48x48, 32x32, 16x16)
- `icon.png` — macOS and Linux (1024x1024 recommended)

## Troubleshooting

### "Python not found" during build
Ensure you have internet access. The script downloads Python automatically.

### Build fails with CUDA errors
The build doesn't require a GPU. CUDA packages are pre-built binaries.

### macOS: Gatekeeper blocks an unsigned build

Published users should follow the GUI-first instructions in
[`DOWNLOAD.md`](DOWNLOAD.md). For a local build from source, quarantine can be
removed for development:
```bash
xattr -dr com.apple.quarantine /Applications/LTX\ Desktop.app
```

### Installer is too large
Expected installer sizes (does not include model weights):
- **Windows/Linux**: the desktop package is comparatively small; the Python
  runtime is a separate multi-gigabyte first-run download
- **macOS**: ~2-3GB (PyTorch MPS is much smaller than CUDA variant)

## Package the Windows/Linux Python runtime

After preparing `python-embed`, create the dependency identity and verified
release parts:

```bash
pnpm python:hash
pnpm python:package
```

The output under `release/python/` contains a manifest, `python-deps-hash.txt`,
and numbered parts suitable for the matching GitHub release. Do not publish an
installer unless all runtime parts are present under the same version tag.

The official installer workflow builds these parts on the same native runner as
the corresponding Windows/Linux installer, attests them, and validates the
complete asset contract before staging a draft release.

### Runtime / first-run issues
End-user topics like OS warnings, system requirements, first-run setup, and
model download behavior are documented in [`DOWNLOAD.md`](DOWNLOAD.md).

## Advanced: Manual Build Steps

### macOS
```bash
# 1. Prepare Python environment
bash scripts/prepare-python.sh

# 2. Install dependencies
pnpm install

# 3. Build frontend
pnpm build:frontend

# 4. Build DMG
pnpm exec electron-builder --mac

# Or build unpacked app (faster, for testing)
pnpm exec electron-builder --mac --dir
```

### Linux
```bash
# 1. Prepare Python environment
bash scripts/prepare-python.sh

# 2. Install dependencies
pnpm install

# 3. Build frontend
pnpm build:frontend

# 4. Build AppImage + deb
pnpm exec electron-builder --linux

# Or build unpacked app (faster, for testing)
pnpm exec electron-builder --linux --dir
```

### Windows
```powershell
# 1. Prepare Python environment
./scripts/prepare-python.ps1

# 2. Install dependencies
pnpm install

# 3. Build frontend
pnpm build:frontend

# 4. Build installer
pnpm exec electron-builder --win
```
