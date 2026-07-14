#!/usr/bin/env bash
# create-installer.sh
# Runs electron-builder to produce the platform installer.
# Release beta.2 is unsigned; a later signed release will supply signing here.
#
# Expects the frontend to be built and python-embed to be ready.
# See local-build.sh for the convenience wrapper that runs all stages.
#
# Usage:
#   bash scripts/create-installer.sh [options]
#
# Options:
#   --platform mac|linux|win   Target platform (auto-detected if omitted)
#   --publish <mode>     Publish mode for electron-builder (always|never|onTag)
#   --unpack             Build unpacked app only (faster, no installer/dmg)

set -euo pipefail

# ============================================================
# Parse arguments
# ============================================================
UNPACK=false
PLATFORM=""
PUBLISH="never"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unpack) UNPACK=true ;;
    --publish)
      PUBLISH="$2"
      shift
      ;;
    --platform)
      PLATFORM="$2"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--platform mac|linux|win] [--publish always|never|onTag] [--unpack]"
      exit 1
      ;;
  esac
  shift
done

# Auto-detect platform if not specified
if [ -z "$PLATFORM" ]; then
  case "$(uname -s)" in
    Darwin)          PLATFORM="mac" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="win" ;;
    Linux)           PLATFORM="linux" ;;
    *)               echo "ERROR: Could not detect platform. Use --platform mac|linux|win"; exit 1 ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RELEASE_DIR="$PROJECT_DIR/release"

cd "$PROJECT_DIR"

# ============================================================
# Verify prerequisites
# ============================================================
if [ ! -d "dist" ] || [ ! -d "dist-electron" ]; then
  echo "ERROR: Frontend not built. Run local-build.sh or 'npm run build:frontend' first."
  exit 1
fi

if [ "$PLATFORM" = "mac" ] && [ ! -d "python-embed" ]; then
  echo "ERROR: Python environment not found. Run local-build.sh or prepare-python.sh first."
  exit 1
fi
if [ ! -f "python-deps-hash.txt" ]; then
  echo "ERROR: Python dependency hash not found. Run local-build.sh first."
  exit 1
fi
if [ "$UNPACK" = false ] && [ "$PLATFORM" != "mac" ] && [ ! -f "python-runtime-manifest.json" ]; then
  echo "ERROR: Verified Python runtime manifest not found. Run 'pnpm python:package' first."
  exit 1
fi

# ============================================================
# Build with electron-builder
# ============================================================
BUILDER_ARGS=""
case "$PLATFORM" in
  mac)   BUILDER_ARGS="--mac" ;;
  win)   BUILDER_ARGS="--win" ;;
  linux) BUILDER_ARGS="--linux" ;;
esac

if [ "$UNPACK" = true ]; then
  echo "Packaging unpacked app (fast mode)..."
  pnpm exec electron-builder $BUILDER_ARGS --dir
else
  PUBLISH_ARGS=""
  if [ -n "$PUBLISH" ]; then
    PUBLISH_ARGS="--publish $PUBLISH"
  fi
  echo "Packaging installer..."
  pnpm exec electron-builder $BUILDER_ARGS $PUBLISH_ARGS
fi
echo ""

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "  Build Complete!"
echo "========================================"

if [ "$UNPACK" = true ]; then
  case "$PLATFORM" in
    mac)
      echo ""
      echo "Unpacked app ready!"
      echo "Run: open \"$RELEASE_DIR/mac-arm64/LTX Desktop.app\""
      ;;
    win)
      echo ""
      echo "Unpacked app ready!"
      echo "Run: $RELEASE_DIR/win-unpacked/LTX Desktop.exe"
      ;;
    linux)
      echo ""
      echo "Unpacked app ready!"
      LINUX_UNPACKED="$RELEASE_DIR/linux-unpacked"
      [ -d "$RELEASE_DIR/linux-arm64-unpacked" ] && LINUX_UNPACKED="$RELEASE_DIR/linux-arm64-unpacked"
      echo "Run: $LINUX_UNPACKED/ltx-desktop"
      ;;
  esac
else
  echo ""
  echo "Output: $RELEASE_DIR/"
  ls -1 "$RELEASE_DIR/" 2>/dev/null | head -10
fi

echo ""
echo "Note: AI models (~150GB) will be downloaded on first run."
