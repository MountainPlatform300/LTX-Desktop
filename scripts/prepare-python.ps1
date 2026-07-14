# prepare-python.ps1
# Downloads embedded Python and installs all dependencies for distribution.
#
# Dependencies are read from uv.lock (via `uv export`) — pyproject.toml is the
# single source of truth. No hardcoded dependency lists.
#
# Prerequisites:
#   - uv must be installed (https://docs.astral.sh/uv/)

param(
    [string]$PythonVersion = (Get-Content "$PSScriptRoot\..\backend\.python-version" -Raw).Trim(),
    [string]$OutputDir = "python-embed"
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LTX Video - Python Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $ProjectDir "backend"
$OutputPath = Join-Path $ProjectDir $OutputDir
$TempDir = Join-Path $env:TEMP "ltx-python-build"

# Python embed URL and digest from the python.org 3.13.12 release manifest.
$PinnedPythonVersion = "3.13.12"
$PythonArchiveSha256 = "76f238f606250c87c6beac75dccd35ee99070a13490555936abb6cb64ecce3d0"
if ($PythonVersion -ne $PinnedPythonVersion) {
    throw "Python $PythonVersion is not pinned for release packaging. Update the reviewed archive digest first."
}
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"

# ============================================================
# Step 1: Verify prerequisites
# ============================================================
Write-Host "`nStep 1: Verifying prerequisites..." -ForegroundColor Yellow

# Check uv is available
$UvExe = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvExe) {
    # Fall back to uv in the dev venv
    $VenvUv = Join-Path $BackendDir ".venv\Scripts\uv.exe"
    if (Test-Path $VenvUv) {
        $UvExe = $VenvUv
    } else {
        Write-Host "ERROR: uv not found. Install it: https://docs.astral.sh/uv/" -ForegroundColor Red
        exit 1
    }
}
$UvCommand = if ($UvExe -is [System.Management.Automation.CommandInfo]) { $UvExe.Source } else { [string]$UvExe }
$UvVersion = (& $UvCommand --version).Trim()
if ($UvVersion -notmatch "^uv 0\.11\.25(?:\s|$)") {
    Write-Host "ERROR: uv 0.11.25 is required; found '$UvVersion'." -ForegroundColor Red
    exit 1
}
Write-Host "uv: $UvCommand ($UvVersion)" -ForegroundColor Green

# ============================================================
# Step 2: Export a direct-source pylock from uv.lock
# ============================================================
Write-Host "`nStep 2: Exporting direct-source pylock from uv.lock..." -ForegroundColor Yellow

$PylockFile = Join-Path $BackendDir "pylock.runtime.toml"

# Preserve each artifact's locked source URL and digest so PyPI and CUDA
# packages cannot be confused across indexes.
& $UvCommand export --frozen --format pylock.toml --no-editable --no-emit-project `
    --project $BackendDir `
    --output-file $PylockFile | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: uv export failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Exported direct-source runtime lock from uv.lock" -ForegroundColor Green

# ============================================================
# Step 3: Prepare directories
# ============================================================
Write-Host "`nStep 3: Preparing directories..." -ForegroundColor Yellow

if (Test-Path $OutputPath) {
    Write-Host "Removing existing python-embed directory..."
    Remove-Item -Recurse -Force $OutputPath
}

if (Test-Path $TempDir) {
    Remove-Item -Recurse -Force $TempDir
}

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

# ============================================================
# Step 4: Download and extract embedded Python
# ============================================================
Write-Host "`nStep 4: Downloading Python $PythonVersion embeddable..." -ForegroundColor Yellow

$PythonZip = Join-Path $TempDir "python-embed.zip"
Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonZip -UseBasicParsing
$PythonArchiveStream = [System.IO.File]::OpenRead($PythonZip)
try {
    $Sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $PythonArchiveDigest = $Sha256.ComputeHash($PythonArchiveStream)
        $ActualPythonArchiveSha256 = (
            [System.BitConverter]::ToString($PythonArchiveDigest)
        ).Replace("-", "").ToLowerInvariant()
    } finally {
        $Sha256.Dispose()
    }
} finally {
    $PythonArchiveStream.Dispose()
}
if ($ActualPythonArchiveSha256 -ne $PythonArchiveSha256) {
    throw "Embedded Python archive digest mismatch: $ActualPythonArchiveSha256"
}
Write-Host "Downloaded Python embeddable package"

Expand-Archive -Path $PythonZip -DestinationPath $OutputPath -Force
Write-Host "Extracted to $OutputPath"

# ============================================================
# Step 5: Enable site-packages in embedded Python
# ============================================================
Write-Host "`nStep 5: Enabling site-packages in embedded Python..." -ForegroundColor Yellow

$PthFile = Get-ChildItem -Path $OutputPath -Filter "python*._pth" | Select-Object -First 1
if ($PthFile) {
    $PthContent = Get-Content $PthFile.FullName
    $PthContent = $PthContent -replace "^#import site", "import site"
    $PthContent += "`nLib\site-packages"
    Set-Content -Path $PthFile.FullName -Value $PthContent
    Write-Host "Modified $($PthFile.Name) to enable site-packages"
}

$PythonExe = Join-Path $OutputPath "python.exe"

# ============================================================
# Step 6: Install all dependencies from the direct-source lock
# ============================================================
Write-Host "`nStep 6: Installing dependencies from direct-source lock..." -ForegroundColor Yellow

# Registry artifacts use their exact locked URLs and hashes. Commit-pinned VCS
# requirements remain bound to their reviewed revisions.
& $UvCommand pip install -r $PylockFile `
    --build-constraint (Join-Path $ScriptDir "python-build-constraints.txt") `
    --python $PythonExe

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed!" -ForegroundColor Red
    exit 1
}
Write-Host "All dependencies installed" -ForegroundColor Green

# ============================================================
# Step 7: Copy Python headers for Triton/SageAttention JIT
# ============================================================
Write-Host "`nStep 7: Copying Python development files for Triton JIT..." -ForegroundColor Yellow

# Ensure the exact Python version is available via uv, then copy headers/libs
& $UvCommand python install "$PythonVersion" --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: uv Python installation failed!" -ForegroundColor Red
    exit 1
}
$UvPython = & $UvCommand python find "$PythonVersion" 2>$null
if ($UvPython) {
    # Ask Python itself for its prefix — reliable regardless of install layout
    $UvPrefix = & $UvPython -c "import sys; print(sys.prefix)"
    if ($LASTEXITCODE -ne 0 -or -not $UvPrefix) {
        Write-Host "ERROR: Failed to inspect uv-managed Python." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Using uv-managed Python at: $UvPrefix"

    $IncludeSrc = Join-Path $UvPrefix "Include"
    if (-not (Test-Path $IncludeSrc)) { $IncludeSrc = Join-Path $UvPrefix "include" }
    $IncludeDst = Join-Path $OutputPath "Include"
    if (Test-Path $IncludeSrc) {
        Copy-Item -Path $IncludeSrc -Destination $IncludeDst -Recurse -Force
        Write-Host "  Copied Include folder (Python headers)"
    } else {
        Write-Host "ERROR: No Include folder found at $UvPrefix" -ForegroundColor Red
        exit 1
    }

    $LibsSrc = Join-Path $UvPrefix "libs"
    $LibsDst = Join-Path $OutputPath "libs"
    if (Test-Path $LibsSrc) {
        Copy-Item -Path $LibsSrc -Destination $LibsDst -Recurse -Force
        Write-Host "  Copied libs folder (Python libraries)"
    } else {
        Write-Host "ERROR: No libs folder found at $UvPrefix" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "ERROR: Could not find Python $PythonVersion via uv" -ForegroundColor Red
    exit 1
}

# ============================================================
# Step 8: Clean up
# ============================================================
Write-Host "`nStep 8: Cleaning up..." -ForegroundColor Yellow

# Remove pip cache
$PipCachePaths = @(
    (Join-Path $OutputPath "Lib\site-packages\pip"),
    (Join-Path $OutputPath "Lib\site-packages\pip\_vendor\cachecontrol\caches"),
    (Join-Path $OutputPath "Lib\site-packages\pip\cache"),
    (Join-Path $OutputPath "Lib\site-packages\setuptools"),
    (Join-Path $OutputPath "Scripts\pip*")
)
foreach ($cachePath in $PipCachePaths) {
    if (Test-Path $cachePath) {
        Remove-Item -Recurse -Force $cachePath -ErrorAction SilentlyContinue
        Write-Host "  Removed cache: $cachePath"
    }
}
Get-ChildItem -Path (Join-Path $OutputPath "Lib\site-packages") -Directory -Filter "pip-*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path (Join-Path $OutputPath "Lib\site-packages") -Directory -Filter "setuptools-*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Remove __pycache__ and .pyc
Get-ChildItem -Path $OutputPath -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $OutputPath -Filter "*.pyc" | Remove-Item -Force

# Clean up temp directory and generated lock file
Remove-Item -Recurse -Force $TempDir
Remove-Item -Force $PylockFile -ErrorAction SilentlyContinue

# ============================================================
# Step 9: Verify
# ============================================================
Write-Host "`nStep 9: Verifying installation..." -ForegroundColor Yellow

$TestScript = @"
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'PyTorch import failed: {e}')
    raise
try:
    import fastapi
    print(f'FastAPI: {fastapi.__version__}')
except ImportError as e:
    print(f'FastAPI import failed: {e}')
    raise
try:
    import diffusers
    print(f'Diffusers: {diffusers.__version__}')
except ImportError as e:
    print(f'Diffusers import failed: {e}')
    raise
try:
    from ltx_pipelines import distilled
    print(f'ltx-pipelines: OK')
except ImportError as e:
    print(f'ltx-pipelines: FAILED - {e}')
    raise
try:
    import runpod
    import paramiko
    print(f'RunPod control dependencies: OK')
except ImportError as e:
    print(f'RunPod control dependencies: FAILED - {e}')
    raise
"@

$TestScript | & $PythonExe -
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python environment verification failed!" -ForegroundColor Red
    exit 1
}

& node (Join-Path $ScriptDir "generate-python-deps-hash.mjs") --platform=win32 --arch=x64
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to generate Python dependency hash!" -ForegroundColor Red
    exit 1
}
Copy-Item (Join-Path $ProjectDir "python-deps-hash.txt") (Join-Path $OutputPath "deps-hash.txt") -Force

# Calculate size
$Size = (Get-ChildItem -Path $OutputPath -Recurse | Measure-Object -Property Length -Sum).Sum
$SizeGB = [math]::Round($Size / 1GB, 2)

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Python environment ready!" -ForegroundColor Green
Write-Host "  Location: $OutputPath" -ForegroundColor Green
Write-Host "  Size: $SizeGB GB" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
