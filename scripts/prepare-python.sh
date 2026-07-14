#!/usr/bin/env bash
# prepare-python.sh
# Downloads a standalone Python and installs all dependencies for macOS/Linux distribution.
#
# Dependencies are read from uv.lock (via `uv export`) — pyproject.toml is the
# single source of truth. No hardcoded dependency lists.
#
# Uses python-build-standalone (https://github.com/astral-sh/python-build-standalone)
# which provides relocatable Python builds for macOS and Linux.
#
# Prerequisites:
#   - uv must be installed (https://docs.astral.sh/uv/)
#   - curl must be available
#   - git must be available (for git-based Python packages)

set -euo pipefail

# ============================================================
# Configuration
# ============================================================
PYTHON_VERSION="${PYTHON_VERSION:-$(cat "$(dirname "$0")/../backend/.python-version" | tr -d '[:space:]')}"
PBS_TAG="${PBS_TAG:-20260211}"
OUTPUT_DIR="python-embed"
ARCH="${ARCH:-$(uname -m)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
OUTPUT_PATH="$PROJECT_DIR/$OUTPUT_DIR"
TEMP_DIR="$(mktemp -d)"

# Map architecture names for python-build-standalone
case "$ARCH" in
  arm64|aarch64) PBS_ARCH="aarch64" ;;
  x86_64|amd64)  PBS_ARCH="x86_64" ;;
  *) echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Detect OS for python-build-standalone target triple
case "$(uname -s)" in
  Darwin) PBS_OS="apple-darwin" ;;
  Linux)  PBS_OS="unknown-linux-gnu" ;;
  *)      echo "ERROR: Unsupported OS: $(uname -s)"; exit 1 ;;
esac

PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/cpython-${PYTHON_VERSION}+${PBS_TAG}-${PBS_ARCH}-${PBS_OS}-install_only_stripped.tar.gz"

if [ "$PYTHON_VERSION" != "3.13.12" ] || [ "$PBS_TAG" != "20260211" ]; then
  echo "ERROR: Python ${PYTHON_VERSION}+${PBS_TAG} is not pinned for release packaging."
  exit 1
fi

# Digests reported by the GitHub release asset API for PBS_TAG 20260211.
case "${PBS_ARCH}-${PBS_OS}" in
  aarch64-apple-darwin)
    PBS_SHA256="fdf98aad59d6fb99a0fc42a0392956bd7f246793003c1789b798f20e38b8ae75"
    ;;
  x86_64-apple-darwin)
    PBS_SHA256="a43f811568fba89d788fa822d85b7f18822f695d76f96709dca017dc9b86b263"
    ;;
  aarch64-unknown-linux-gnu)
    PBS_SHA256="180918f7fe5384bc09cf836103a5318a94a15725fa64df2005caedd56c36d64c"
    ;;
  x86_64-unknown-linux-gnu)
    PBS_SHA256="bab0e2aeec8a32a7f5cb62240d088d50ea468ef6d7522681bc171d527a5ba6f8"
    ;;
  *)
    echo "ERROR: No reviewed Python archive digest for ${PBS_ARCH}-${PBS_OS}."
    exit 1
    ;;
esac

PLATFORM_LABEL="$(uname -s) ($ARCH)"

echo "========================================"
echo "  LTX Video - Python Environment Setup"
echo "  Platform: $PLATFORM_LABEL"
echo "  Python: $PYTHON_VERSION"
echo "========================================"

# ============================================================
# Step 1: Verify prerequisites
# ============================================================
echo ""
echo "Step 1: Verifying prerequisites..."

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it: https://docs.astral.sh/uv/"
    exit 1
fi
UV_VERSION="$(uv --version)"
if [[ ! "$UV_VERSION" =~ ^uv\ 0\.11\.25([[:space:]]|$) ]]; then
    echo "ERROR: uv 0.11.25 is required; found '$UV_VERSION'."
    exit 1
fi
echo "  uv: $(command -v uv) ($UV_VERSION)"

if ! command -v curl &>/dev/null; then
    echo "ERROR: curl not found."
    exit 1
fi
echo "  curl: $(command -v curl)"

if ! command -v git &>/dev/null; then
    echo "ERROR: git not found (needed for git-based Python packages)."
    exit 1
fi
echo "  git: $(command -v git)"

# ============================================================
# Step 2: Export a direct-source pylock from uv.lock
# ============================================================
echo ""
echo "Step 2: Exporting direct-source pylock from uv.lock..."

PYLOCK_FILE="$BACKEND_DIR/pylock.runtime.toml"

# Preserve each artifact's locked source URL and digest so PyPI and CUDA
# packages cannot be confused across indexes.
uv export --frozen --format pylock.toml --no-editable --no-emit-project \
    --project "$BACKEND_DIR" \
    --output-file "$PYLOCK_FILE" >/dev/null

echo "  Exported direct-source runtime lock from uv.lock"

# ============================================================
# Step 3: Prepare directories
# ============================================================
echo ""
echo "Step 3: Preparing directories..."

if [ -d "$OUTPUT_PATH" ]; then
    echo "  Removing existing $OUTPUT_DIR directory..."
    rm -rf "$OUTPUT_PATH"
fi

mkdir -p "$OUTPUT_PATH"

# ============================================================
# Step 4: Download and extract standalone Python
# ============================================================
echo ""
echo "Step 4: Downloading Python $PYTHON_VERSION standalone ($PBS_ARCH)..."
echo "  URL: $PBS_URL"

PYTHON_TAR="$TEMP_DIR/python-standalone.tar.gz"
curl -L --fail --progress-bar -o "$PYTHON_TAR" "$PBS_URL"
ACTUAL_PBS_SHA256="$(shasum -a 256 "$PYTHON_TAR" | awk '{print $1}')"
if [ "$ACTUAL_PBS_SHA256" != "$PBS_SHA256" ]; then
    echo "ERROR: Standalone Python archive digest mismatch: $ACTUAL_PBS_SHA256"
    exit 1
fi
echo "  Downloaded Python standalone package"

# python-build-standalone extracts to a `python/` directory
echo "  Extracting..."
tar -xzf "$PYTHON_TAR" -C "$TEMP_DIR"

# Move contents from python/ into our output path
mv "$TEMP_DIR/python/"* "$OUTPUT_PATH/"
echo "  Extracted to $OUTPUT_PATH"

# Verify the Python binary exists
PYTHON_EXE="$OUTPUT_PATH/bin/python3"
if [ ! -f "$PYTHON_EXE" ]; then
    echo "ERROR: Python binary not found at $PYTHON_EXE"
    exit 1
fi

echo "  Python binary: $PYTHON_EXE"
"$PYTHON_EXE" --version

# ============================================================
# Step 5: Verify the embedded interpreter
# ============================================================
echo ""
echo "Step 5: Embedded interpreter verified."

# ============================================================
# Step 6: Install all dependencies from the direct-source lock
# ============================================================
echo ""
echo "Step 6: Installing dependencies from direct-source lock..."
echo "  (This may take a while — PyTorch + ML libraries are large)"

# Registry artifacts use their exact locked URLs and hashes. Commit-pinned VCS
# requirements remain bound to their reviewed revisions.
uv pip install -r "$PYLOCK_FILE" \
    --build-constraint "$SCRIPT_DIR/python-build-constraints.txt" \
    --python "$PYTHON_EXE"

echo "  All dependencies installed"

# ============================================================
# Step 7: Clean up
# ============================================================
echo ""
echo "Step 7: Cleaning up..."

# Remove __pycache__ and .pyc files
find "$OUTPUT_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pyc" -delete 2>/dev/null || true

# Remove pip cache and pip itself (not needed at runtime)
rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/pip 2>/dev/null || true
rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/pip-*.dist-info 2>/dev/null || true
rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/setuptools 2>/dev/null || true
rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/setuptools-*.dist-info 2>/dev/null || true

# Remove test directories to save space
find "$OUTPUT_PATH/lib" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "$OUTPUT_PATH/lib" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Remove files only needed for building native extensions, not at runtime.
# This cuts ~14k files and speeds up macOS codesigning dramatically.
# NOTE: Linux needs .h files for sageattention/triton JIT compilation.
if [ "$PBS_OS" = "apple-darwin" ]; then
  rm -rf "$OUTPUT_PATH/include" "$OUTPUT_PATH/share" 2>/dev/null || true
  find "$OUTPUT_PATH/lib" -type d -name "include" -exec rm -rf {} + 2>/dev/null || true
  find "$OUTPUT_PATH" -name "*.h" -delete 2>/dev/null || true
  find "$OUTPUT_PATH" -name "*.cuh" -delete 2>/dev/null || true
  find "$OUTPUT_PATH" -name "*.cu" -delete 2>/dev/null || true
else
  # Linux: keep .h/.cuh/.cu files for triton/sageattention JIT, but remove other build artifacts
  rm -rf "$OUTPUT_PATH/share" 2>/dev/null || true
fi
find "$OUTPUT_PATH" -name "*.pyi" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pxd" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pyx" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.hpp" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.cpp" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.cmake" -delete 2>/dev/null || true

# Remove temp directory and generated lock file
rm -rf "$TEMP_DIR"
rm -f "$PYLOCK_FILE"

echo "  Cleanup complete"

# ============================================================
# Step 8: Verify installation
# ============================================================
echo ""
echo "Step 8: Verifying installation..."

"$PYTHON_EXE" -c "
import sys
import platform
print(f'  Python: {sys.version}')
try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
    if platform.system() == 'Darwin':
        mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        print(f'  MPS available: {mps}')
    elif platform.system() == 'Linux':
        cuda = torch.cuda.is_available()
        print(f'  CUDA available: {cuda}')
        if cuda:
            print(f'  CUDA version: {torch.version.cuda}')
except ImportError as e:
    print(f'  PyTorch import FAILED: {e}')
    sys.exit(1)
try:
    import fastapi
    print(f'  FastAPI: {fastapi.__version__}')
except ImportError as e:
    print(f'  FastAPI import FAILED: {e}')
    sys.exit(1)
try:
    import diffusers
    print(f'  Diffusers: {diffusers.__version__}')
except ImportError as e:
    print(f'  Diffusers import FAILED: {e}')
    sys.exit(1)
try:
    from ltx_pipelines import distilled
    print(f'  ltx-pipelines: OK')
except ImportError as e:
    print(f'  ltx-pipelines: FAILED - {e}')
    sys.exit(1)
try:
    import runpod
    import paramiko
    print(f'  RunPod control dependencies: OK')
except ImportError as e:
    print(f'  RunPod control dependencies: FAILED - {e}')
    sys.exit(1)
"

node "$SCRIPT_DIR/generate-python-deps-hash.mjs"
cp "$PROJECT_DIR/python-deps-hash.txt" "$OUTPUT_PATH/deps-hash.txt"

# Calculate size
SIZE_BYTES=$(du -sb "$OUTPUT_PATH" 2>/dev/null | cut -f1 || du -sk "$OUTPUT_PATH" | awk '{print $1 * 1024}')
SIZE_GB=$(awk "BEGIN {printf \"%.2f\", $SIZE_BYTES / 1073741824}")

echo ""
echo "========================================"
echo "  Python environment ready!"
echo "  Location: $OUTPUT_PATH"
echo "  Size: ${SIZE_GB} GB"
echo "========================================"
