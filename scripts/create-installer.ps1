# create-installer.ps1
# Runs electron-builder to produce the Windows installer.
# Release beta.2 is unsigned; a later signed release will supply signing here.
#
# Expects the frontend to be built and python-embed to be ready.
# See local-build.ps1 for the convenience wrapper that runs all stages.

param(
    [switch]$Unpack,
    [string]$Publish = "never"
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$ReleaseDir = Join-Path $ProjectDir "release"

Set-Location $ProjectDir

# Verify prerequisites
if (-not (Test-Path "dist") -or -not (Test-Path "dist-electron")) {
    Write-Host "ERROR: Frontend not built. Run local-build.ps1 or 'npm run build:frontend' first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "python-deps-hash.txt")) {
    Write-Host "ERROR: Python dependency hash not found. Run local-build.ps1 first." -ForegroundColor Red
    exit 1
}
if (-not $Unpack -and -not (Test-Path "python-runtime-manifest.json")) {
    Write-Host "ERROR: Verified Python runtime manifest not found. Run 'pnpm python:package' first." -ForegroundColor Red
    exit 1
}

# Build with electron-builder
if ($Unpack) {
    Write-Host "Packaging unpacked app (fast mode)..." -ForegroundColor Yellow
    pnpm exec electron-builder --win --dir
} else {
    Write-Host "Packaging installer..." -ForegroundColor Yellow
    $PublishArgs = @()
    if ($Publish -ne "") {
        $PublishArgs = @("--publish", $Publish)
    }
    pnpm exec electron-builder --win @PublishArgs
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build!" -ForegroundColor Red
    exit 1
}

# Summary
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if ($Unpack) {
    $UnpackedDir = Join-Path $ReleaseDir "win-unpacked"
    $ExePath = Join-Path $UnpackedDir "LTX Desktop.exe"
    Write-Host "`nUnpacked app ready!" -ForegroundColor Cyan
    Write-Host "Run: $ExePath" -ForegroundColor Cyan
    Write-Host "`nTip: Just restart the app after code changes - no rebuild needed!" -ForegroundColor Green
} else {
    $InstallerPath = Join-Path $ReleaseDir "LTX-Desktop-Setup.exe"
    if (-not (Test-Path $InstallerPath)) {
        Write-Host "ERROR: Expected installer was not created: $InstallerPath" -ForegroundColor Red
        exit 1
    }
    $Installer = Get-Item $InstallerPath
    $InstallerSize = [math]::Round($Installer.Length / 1MB, 2)
    Write-Host "`nInstaller: $($Installer.Name)" -ForegroundColor Cyan
    Write-Host "Size: $InstallerSize MB" -ForegroundColor Cyan
    Write-Host "Location: $($Installer.FullName)" -ForegroundColor Cyan
}

Write-Host "`nNote: AI models (~150GB) will be downloaded on first run." -ForegroundColor Yellow
