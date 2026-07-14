import { spawn, spawnSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { logger } from './logger'
import { readAppState, writeAppState } from './app-state'
import { getMainWindow } from './window'

// In-app "Set up local training" wizard, Windows side.
//
// Local LoRA training runs the (Linux-only) LTX-2 trainer inside WSL2 with CUDA
// passthrough. Getting there on a fresh Windows machine means enabling the WSL
// optional features (needs admin), rebooting, then having a Linux distro with a
// working `nvidia-smi`. None of that can be done from the Python backend, so the
// orchestration lives here in the Electron main process.
//
// Design notes:
//  - We never trust parsed output of the elevated installer. Every decision is
//    driven by re-probing actual state (`probeWsl`), which makes the flow
//    idempotent and safe to resume after a reboot or a crash.
//  - The whole flow is best-effort and shows the exact manual command in the UI
//    as a fallback, because elevation + reboot behaviour varies across Windows
//    builds and can't be exercised in CI.

export type WslSetupStage =
  | 'idle'
  | 'installing'
  | 'reboot-required'
  | 'verifying'
  | 'complete'
  | 'error'

export interface WslSetupState {
  stage: WslSetupStage
  distro: string
  startedAt?: string
  updatedAt?: string
  error?: string | null
}

export interface WslProbeResult {
  platformSupported: boolean
  windowsBuild: number
  wslCommandPresent: boolean
  // The WSL2 engine itself is installed & functional (`wsl --version` works).
  // This is true even before any Linux distro is installed, and crucially tells
  // us whether enabling WSL (which needs admin + a reboot) is still required.
  wslEnginePresent: boolean
  wslInstalled: boolean
  defaultDistro: string | null
  cudaInWsl: boolean
  reason: string
}

// WSL2 memory auto-configuration. The LTX-2 trainer's preprocess step loads the
// Gemma3 12B text encoder into system RAM (the bf16 weights are ~23 GB even
// when 8-bit quantization is requested, because the load path pulls the full
// weights before quantizing). WSL2's DEFAULT memory limit is ~half the host
// RAM, which on a 32 GB-RAM machine gives WSL ~16 GB — not enough to hold that
// peak, so the Linux OOM killer SIGKILLs the process mid-load (no exit code).
// We read/merge the user's `~/.wslconfig` to raise the `memory` + `swap` limits
// based on the actual host RAM, so the user never has to edit the file by hand.
export interface WslMemoryProbe {
  hostRamGb: number
  configuredMemoryGb: number | null
  configuredSwapGb: number | null
  recommendedMemoryGb: number
  recommendedSwapGb: number
  needsUpdate: boolean
  path: string
  reason: string | null
}

export interface WslMemoryConfigResult {
  applied: boolean
  memoryGb: number
  swapGb: number
  needsRestart: boolean
  alreadyConfigured: boolean
  error?: string | null
}

// Result of `ensureWslMemoryReady`: tells the frontend whether a local run may
// proceed now, or whether WSL must be restarted first so a freshly-raised
// `.wslconfig` takes effect. `liveMemoryGb` is the RAM the *running* WSL VM
// actually reports (via `/proc/meminfo`), or null when WSL isn't running yet —
// in which case the next `wsl.exe` launch reads `.wslconfig` automatically, so
// no restart is needed.
export interface WslMemoryReadiness {
  ready: boolean
  needsRestart: boolean
  appliedNow: boolean
  recommendedMemoryGb: number
  liveMemoryGb: number | null
  configuredMemoryGb: number | null
  error: string | null
}

const GB = 1024 * 1024 * 1024

function wslConfigPath(): string {
  return path.join(os.homedir(), '.wslconfig')
}

// Parse a `.wslconfig` memory/swap value like "48GB", "48 GB", "16384MB" → GB.
function parseSizeGb(raw: string | undefined): number | null {
  if (!raw) return null
  const m = raw.match(/([0-9]+)\s*(gb|mb|tb)?/i)
  if (!m) return null
  const n = parseInt(m[1], 10)
  if (Number.isNaN(n)) return null
  const unit = (m[2] ?? 'gb').toLowerCase()
  if (unit === 'mb') return n / 1024
  if (unit === 'tb') return n * 1024
  return n
}

// Recommended WSL2 memory for the LTX-2 trainer, given the host's total RAM.
// Rule of thumb: give WSL everything except ~6 GB for Windows itself, but at
// least 24 GB (the floor that holds the ~23 GB text-encoder load peak). Swap
// covers the spillover so the OOM killer never fires.
function recommendedMemoryFor(hostRamGb: number): { memory: number; swap: number } {
  const memory = Math.max(24, hostRamGb - 6)
  const swap = hostRamGb >= 48 ? 16 : 24
  return { memory, swap }
}

export function probeWslMemory(): WslMemoryProbe {
  const hostRamGb = Math.max(1, Math.floor(os.totalmem() / GB))
  const { memory: recMemory, swap: recSwap } = recommendedMemoryFor(hostRamGb)
  const cfgPath = wslConfigPath()
  let configuredMemoryGb: number | null = null
  let configuredSwapGb: number | null = null
  try {
    if (fs.existsSync(cfgPath)) {
      const text = fs.readFileSync(cfgPath, 'utf-8')
      configuredMemoryGb = readWsl2Value(text, 'memory')
      configuredSwapGb = readWsl2Value(text, 'swap')
    }
  } catch (e) {
    logger.warn(`[wsl-setup] failed to read .wslconfig: ${String(e)}`)
  }
  // Unset memory → WSL2 defaults to ~50% of host RAM, which is usually too
  // small for the text-encoder load peak. Treat that as "needs update".
  const needsUpdate =
    configuredMemoryGb === null ? hostRamGb - 6 > hostRamGb / 2 : configuredMemoryGb < recMemory
  const reason = configuredMemoryGb === null
    ? `WSL2 is using its default (~half your RAM, ~${Math.floor(hostRamGb / 2)} GB). The trainer needs ~${recMemory} GB to load the text encoder without running out of memory.`
    : configuredMemoryGb < recMemory
      ? `Currently set to ${configuredMemoryGb} GB, but ~${recMemory} GB is recommended for this machine.`
      : null
  return {
    hostRamGb,
    configuredMemoryGb,
    configuredSwapGb,
    recommendedMemoryGb: recMemory,
    recommendedSwapGb: recSwap,
    needsUpdate,
    path: cfgPath,
    reason,
  }
}

// Read a key from the [wsl2] section of an INI-style string. Returns the value
// parsed to GB, or null if absent / unparseable. Only the [wsl2] section is
// consulted because that is where memory/swap live.
function readWsl2Value(text: string, key: string): number | null {
  const lines = text.split(/\r?\n/)
  let inWsl2 = false
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      inWsl2 = trimmed.toLowerCase() === '[wsl2]'
      continue
    }
    if (!inWsl2) continue
    const eq = trimmed.indexOf('=')
    if (eq < 0) continue
    const k = trimmed.slice(0, eq).trim().toLowerCase()
    if (k === key.toLowerCase()) {
      return parseSizeGb(trimmed.slice(eq + 1).trim())
    }
  }
  return null
}

// Merge `memory` and `swap` into the [wsl2] section of an existing `.wslconfig`,
// preserving every other section and key. If no [wsl2] section exists, append
// one. Returns the new file text.
function mergeWslConfig(existing: string, memoryGb: number, swapGb: number): string {
  const lines = existing.split(/\r?\n/)
  let out: string[] = []
  let inWsl2 = false
  let sawWsl2 = false
  let wroteMemory = false
  let wroteSwap = false
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      // Leaving the [wsl2] section: backfill any keys we still need to write.
      if (inWsl2 && (!wroteMemory || !wroteSwap)) {
        if (!wroteMemory) out.push(`memory=${memoryGb}GB`)
        if (!wroteSwap) out.push(`swap=${swapGb}GB`)
      }
      inWsl2 = trimmed.toLowerCase() === '[wsl2]'
      if (inWsl2) sawWsl2 = true
      out.push(line)
      continue
    }
    if (inWsl2) {
      const eq = trimmed.indexOf('=')
      if (eq >= 0) {
        const k = trimmed.slice(0, eq).trim().toLowerCase()
        if (k === 'memory') {
          out.push(`memory=${memoryGb}GB`)
          wroteMemory = true
          continue
        }
        if (k === 'swap') {
          out.push(`swap=${swapGb}GB`)
          wroteSwap = true
          continue
        }
      }
    }
    out.push(line)
  }
  if (inWsl2 && (!wroteMemory || !wroteSwap)) {
    if (!wroteMemory) out.push(`memory=${memoryGb}GB`)
    if (!wroteSwap) out.push(`swap=${swapGb}GB`)
  }
  if (!sawWsl2) {
    if (out.length > 0 && out[out.length - 1].trim() !== '') out.push('')
    out.push('[wsl2]', `memory=${memoryGb}GB`, `swap=${swapGb}GB`)
  }
  return out.join('\n')
}

export function configureWslMemory(): WslMemoryConfigResult {
  const probe = probeWslMemory()
  const memoryGb = probe.recommendedMemoryGb
  const swapGb = probe.recommendedSwapGb
  // Already adequately configured — don't touch the file or force a restart.
  if (
    probe.configuredMemoryGb !== null &&
    probe.configuredMemoryGb >= memoryGb &&
    probe.configuredSwapGb !== null &&
    probe.configuredSwapGb >= swapGb
  ) {
    return {
      applied: false,
      memoryGb: probe.configuredMemoryGb,
      swapGb: probe.configuredSwapGb,
      needsRestart: false,
      alreadyConfigured: true,
    }
  }
  try {
    let existing = ''
    if (fs.existsSync(probe.path)) {
      existing = fs.readFileSync(probe.path, 'utf-8')
    }
    // Back up the user's existing file once so they can restore it if needed.
    if (existing && !fs.existsSync(`${probe.path}.ltx-backup`)) {
      fs.writeFileSync(`${probe.path}.ltx-backup`, existing, 'utf-8')
    }
    const next = mergeWslConfig(existing, memoryGb, swapGb)
    fs.writeFileSync(probe.path, next, 'utf-8')
    logger.info(`[wsl-setup] wrote .wslconfig memory=${memoryGb}GB swap=${swapGb}GB`)
    return {
      applied: true,
      memoryGb,
      swapGb,
      needsRestart: true, // .wslconfig only applies after WSL restarts
      alreadyConfigured: false,
    }
  } catch (e) {
    return {
      applied: false,
      memoryGb,
      swapGb,
      needsRestart: false,
      alreadyConfigured: false,
      error: String((e as Error).message ?? e),
    }
  }
}

// RAM the *running* WSL2 VM actually has, probed via `/proc/meminfo` inside
// the default distro. Returns GB rounded down, or null when WSL isn't running
// (the probe command fails) — in that case the next `wsl.exe` launch reads
// `.wslconfig` fresh, so the caller treats it as "no restart needed".
function probeLiveWslMemoryGb(): number | null {
  const res = runWslCapture([
    '-u',
    'root',
    '--',
    'bash',
    '-lc',
    "awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null",
  ])
  if (res.code !== 0) return null
  const memKb = parseInt(res.out.trim(), 10)
  if (Number.isNaN(memKb) || memKb <= 0) return null
  return Math.floor(memKb / 1024 / 1024)
}

// One-shot, app-driven WSL2 memory readiness for a local training run. This is
// the fix for the recurring "LoRA preprocess was killed mid-run (OOM)" failure:
// the trainer loads the Gemma3 12B text encoder into WSL2 system RAM (~23 GB
// peak, even in 8-bit, because the full bf16 weights load before quantizing),
// which blows past WSL2's default ~half-host-RAM cap and gets the process
// SIGKILL'd. Here we:
//   1. write `.wslconfig` (memory+swap) if it's still too small — safe,
//      idempotent, backed up, only ever raises limits (see `configureWslMemory`);
//   2. probe the LIVE VM's actual RAM. If WSL is running with the old low cap,
//      the file change won't take effect until `wsl --shutdown`, so we surface
//      `needsRestart` and the frontend prompts once (concise). If WSL isn't
//      running yet, the next trainer command launches it with the new config —
//      `ready`, no prompt.
// Never throws: an IPC/probe failure degrades to `ready=true` so a run is never
// blocked on a diagnostics glitch (the trainer still has its own VRAM unload).
export function ensureWslMemoryReady(): WslMemoryReadiness {
  const probe = probeWslMemory()
  const recommended = probe.recommendedMemoryGb
  let appliedNow = false
  let error: string | null = null
  if (probe.needsUpdate) {
    const cfg = configureWslMemory()
    appliedNow = cfg.applied
    if (cfg.error) error = cfg.error
  }
  const configured = probeWslMemory().configuredMemoryGb
  const live = probeLiveWslMemoryGb()
  // 1 GB slack so a VM at recommended-1 (rounding/overhead) isn't flagged stale.
  const needsRestart = live !== null && live < recommended - 1
  return {
    ready: !needsRestart && error === null,
    needsRestart,
    appliedNow,
    recommendedMemoryGb: recommended,
    liveMemoryGb: live,
    configuredMemoryGb: configured,
    error,
  }
}

// App-startup auto-apply: write `.wslconfig` (memory+swap) if it's still at the
// too-small default, BEFORE anything probes WSL and inadvertently launches the
// VM with the old cap. No `wsl --shutdown` here (that would kill a user's
// unrelated WSL session) — just the file write, so the first WSL launch after
// startup picks up the raised limit. Best-effort, never throws.
export function autoApplyWslMemoryIfNeeded(): { applied: boolean; memoryGb: number; error: string | null } {
  try {
    const probe = probeWslMemory()
    if (!probe.needsUpdate) {
      return { applied: false, memoryGb: probe.recommendedMemoryGb, error: null }
    }
    const cfg = configureWslMemory()
    return {
      applied: cfg.applied,
      memoryGb: cfg.memoryGb,
      error: cfg.error ?? null,
    }
  } catch (e) {
    return { applied: false, memoryGb: 0, error: String((e as Error).message ?? e) }
  }
}

// `wsl --shutdown` stops the WSL2 VM so the next command restarts it with the
// new `.wslconfig`. Much lighter than a Windows reboot, but it DOES kill any
// running WSL process (including an in-flight training job) — only call when
// the user confirms.
export function restartWsl(): { success: boolean; error?: string } {
  // Instrumented: a recurring "preprocess killed mid-run, no exit code" bug was
  // traced to a `wsl --shutdown` stopping a healthy running job. Log loudly so a
  // mid-run shutdown is visible in the session log (it kills every WSL process,
  // including an in-flight training job).
  logger.warn('[wsl-setup] restartWsl: calling `wsl --shutdown` — this kills ALL running WSL processes')
  try {
    const res = spawnSync('wsl.exe', ['--shutdown'], { windowsHide: true, timeout: 30000 })
    if (res.error) return { success: false, error: String(res.error.message ?? res.error) }
    if ((res.status ?? -1) !== 0) {
      return { success: false, error: decodeWsl(res.stderr) || `wsl --shutdown exited ${res.status}` }
    }
    return { success: true }
  } catch (e) {
    return { success: false, error: String(e) }
  }
}

const DEFAULT_DISTRO = 'Ubuntu'
const MIN_WINDOWS_BUILD = 19041 // Windows 10 2004 — first build with WSL2

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

// wsl.exe emits its own manager messages as UTF-16LE, but output from programs
// run *inside* the distro is UTF-8. Detect the NUL bytes that UTF-16LE produces
// for ASCII text and decode accordingly.
function decodeWsl(buf: Buffer | null | undefined): string {
  if (!buf || buf.length === 0) return ''
  if (buf.includes(0x00)) return buf.toString('utf16le')
  return buf.toString('utf8')
}

interface WslCapture {
  code: number
  out: string
  err: string
  missing: boolean
}

function runWslCapture(args: string[], timeoutMs = 60000): WslCapture {
  try {
    const res = spawnSync('wsl.exe', args, {
      windowsHide: true,
      timeout: timeoutMs,
      maxBuffer: 4 * 1024 * 1024,
    })
    if (res.error) {
      const missing = (res.error as NodeJS.ErrnoException).code === 'ENOENT'
      return { code: -1, out: '', err: String(res.error.message ?? res.error), missing }
    }
    return {
      code: res.status ?? -1,
      out: decodeWsl(res.stdout),
      err: decodeWsl(res.stderr),
      missing: false,
    }
  } catch (e) {
    return { code: -1, out: '', err: String(e), missing: false }
  }
}

function getWindowsBuild(): number {
  // os.release() looks like "10.0.26300"
  const parts = os.release().split('.')
  return parts.length >= 3 ? parseInt(parts[2], 10) || 0 : 0
}

// Whether the WSL2 engine is installed (regardless of any distro). `wsl
// --version` prints "WSL version: x" only when the engine is present; on a
// machine where WSL has never been enabled it errors / prints install guidance.
function isWslEnginePresent(): boolean {
  const res = runWslCapture(['--version'])
  return !res.missing && /wsl version/i.test(res.out)
}

// ---------------------------------------------------------------------------
// Probe (no elevation)
// ---------------------------------------------------------------------------

export function probeWsl(): WslProbeResult {
  const windowsBuild = getWindowsBuild()
  const platformSupported = process.platform === 'win32' && windowsBuild >= MIN_WINDOWS_BUILD

  if (!platformSupported) {
    return {
      platformSupported: false,
      windowsBuild,
      wslCommandPresent: false,
      wslEnginePresent: false,
      wslInstalled: false,
      defaultDistro: null,
      cudaInWsl: false,
      reason:
        process.platform !== 'win32'
          ? 'Local GPU training requires Windows with WSL2.'
          : `This Windows build (${windowsBuild}) is too old for WSL2. Update to Windows 10 2004 (build ${MIN_WINDOWS_BUILD}) or newer.`,
    }
  }

  // List registered distros. `-l -q` prints one name per line (UTF-16LE), and
  // exits 0 with empty output when none are installed.
  const list = runWslCapture(['-l', '-q'])
  if (list.missing) {
    return {
      platformSupported: true,
      windowsBuild,
      wslCommandPresent: false,
      wslEnginePresent: false,
      wslInstalled: false,
      defaultDistro: null,
      cudaInWsl: false,
      reason: 'WSL is not installed yet. Run the setup to install it.',
    }
  }

  const wslEnginePresent = isWslEnginePresent()
  const distros = list.out
    .split(/\r?\n/)
    .map((l) => l.replace(/\0/g, '').trim())
    .filter((l) => l.length > 0)
  const wslInstalled = distros.length > 0
  const defaultDistro = distros[0] ?? null

  if (!wslInstalled) {
    return {
      platformSupported: true,
      windowsBuild,
      wslCommandPresent: true,
      wslEnginePresent,
      wslInstalled: false,
      defaultDistro: null,
      cudaInWsl: false,
      reason: wslEnginePresent
        ? 'WSL2 is ready, but no Linux distribution is installed yet. Run the setup to install Ubuntu — no restart needed.'
        : 'WSL is not enabled yet. Run the setup to install it (needs admin + a restart).',
    }
  }

  // Check CUDA visibility inside the distro. Running as root avoids the
  // first-launch interactive user-creation prompt for an --no-launch distro.
  const cuda = runWslCapture([
    '-u',
    'root',
    '--',
    'bash',
    '-lc',
    'command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L',
  ])
  const cudaInWsl = cuda.code === 0 && /GPU \d+:/.test(cuda.out)

  return {
    platformSupported: true,
    windowsBuild,
    wslCommandPresent: true,
    wslEnginePresent: true,
    wslInstalled: true,
    defaultDistro,
    cudaInWsl,
    reason: cudaInWsl
      ? 'WSL2 is ready with GPU access.'
      : 'WSL is installed but the GPU is not visible inside it yet. Make sure the latest NVIDIA Windows driver is installed, then reboot.',
  }
}

// ---------------------------------------------------------------------------
// Persisted state
// ---------------------------------------------------------------------------

function defaultState(): WslSetupState {
  return { stage: 'idle', distro: DEFAULT_DISTRO }
}

export function getWslSetupState(): WslSetupState {
  const persisted = readAppState().wslSetup as Partial<WslSetupState> | undefined
  if (!persisted || typeof persisted.stage !== 'string') return defaultState()
  return {
    stage: persisted.stage as WslSetupStage,
    distro: persisted.distro ?? DEFAULT_DISTRO,
    startedAt: persisted.startedAt,
    updatedAt: persisted.updatedAt,
    error: persisted.error ?? null,
  }
}

function updateState(patch: Partial<WslSetupState>): WslSetupState {
  const current = getWslSetupState()
  const next: WslSetupState = {
    ...current,
    ...patch,
    updatedAt: new Date().toISOString(),
  }
  const state = readAppState()
  state.wslSetup = next
  writeAppState(state)
  return next
}

function emit(): void {
  getMainWindow()?.webContents.send('wsl-setup-progress', getWslSetupState())
}

// ---------------------------------------------------------------------------
// Elevated install
// ---------------------------------------------------------------------------

// Fresh machine: enable the WSL optional features. This needs admin and a
// reboot before any distro can run. A single UAC prompt via PowerShell.
function runElevatedFeatureEnable(): Promise<{ code: number; error?: string }> {
  return new Promise((resolve) => {
    const psCommand =
      "$ErrorActionPreference='Stop'; " +
      "try { $p = Start-Process -FilePath 'wsl.exe' -ArgumentList '--install','--no-launch' -Verb RunAs -Wait -PassThru; exit $p.ExitCode } " +
      'catch { Write-Error $_; exit 1223 }'
    const child = spawn(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command', psCommand],
      { windowsHide: true },
    )
    let stderr = ''
    child.stderr?.on('data', (d) => {
      stderr += d.toString()
    })
    child.on('error', (e) => resolve({ code: -1, error: String(e) }))
    child.on('close', (code) => resolve({ code: code ?? -1, error: stderr.trim() || undefined }))
  })
}

// Engine already present: install the distro. This is a per-user store install
// that needs no admin and no reboot, so run it un-elevated (running elevated
// would register the distro under the elevated profile instead of the user's).
// `--no-launch` skips the interactive first-run user-creation window; we run
// trainer commands as root, which an --no-launch distro allows by default.
function installDistro(distro: string): Promise<{ code: number; error?: string }> {
  return new Promise((resolve) => {
    const child = spawn('wsl.exe', ['--install', '-d', distro, '--no-launch'], { windowsHide: true })
    const chunks: Buffer[] = []
    child.stderr?.on('data', (d: Buffer) => chunks.push(d))
    child.on('error', (e) => resolve({ code: -1, error: String(e) }))
    child.on('close', (code) =>
      resolve({ code: code ?? -1, error: decodeWsl(Buffer.concat(chunks)).trim() || undefined }),
    )
  })
}

// Map a finished install attempt to a persisted stage by RE-PROBING actual
// state — never by trusting the installer's output. `enginePresentBefore` tells
// us whether a reboot can still be pending (only on a fresh feature-enable).
function settleStageFromProbe(enginePresentBefore: boolean): void {
  const after = probeWsl()
  if (after.wslInstalled && after.cudaInWsl) {
    updateState({ stage: 'complete', error: null })
    logger.info('[wsl-setup] WSL + GPU ready; setup complete')
  } else if (after.wslInstalled) {
    updateState({ stage: 'verifying', error: null })
    logger.info('[wsl-setup] distro installed; verifying GPU visibility')
  } else if (after.wslEnginePresent || enginePresentBefore) {
    // Engine is there but no distro registered — no reboot will fix that; let
    // the user re-check / retry rather than sending them on a needless restart.
    updateState({ stage: 'verifying', error: null })
    logger.info('[wsl-setup] engine present but no distro yet; awaiting re-check')
  } else {
    updateState({ stage: 'reboot-required', error: null })
    logger.info('[wsl-setup] WSL features enabled; a reboot is required to finish')
  }
}

export async function startWslInstall(): Promise<void> {
  const current = getWslSetupState()
  if (current.stage === 'installing') {
    logger.info('[wsl-setup] install already in progress; ignoring re-entry')
    return
  }

  const before = probeWsl()
  if (before.wslInstalled && before.cudaInWsl) {
    updateState({ stage: 'complete', error: null })
    emit()
    return
  }
  if (!before.platformSupported) {
    updateState({ stage: 'error', error: before.reason })
    emit()
    return
  }

  updateState({ stage: 'installing', startedAt: new Date().toISOString(), error: null })
  emit()

  let res: { code: number; error?: string }
  if (before.wslEnginePresent) {
    // Common case on modern Windows: WSL2 is already there, just add a distro.
    logger.info(`[wsl-setup] engine present; installing distro ${current.distro} (no elevation, no reboot)`)
    res = await installDistro(current.distro)
  } else {
    logger.info('[wsl-setup] engine absent; enabling WSL features (elevated)')
    res = await runElevatedFeatureEnable()
  }

  if (res.code === 0) {
    settleStageFromProbe(before.wslEnginePresent)
  } else if (res.code === 1223) {
    updateState({
      stage: 'error',
      error: 'Administrator approval was declined. Enabling WSL needs admin rights — try again and accept the prompt.',
    })
    logger.warn('[wsl-setup] UAC declined (exit 1223)')
  } else {
    updateState({
      stage: 'error',
      error:
        res.error ||
        `WSL setup failed (exit ${res.code}). You can run "wsl --install" in an admin terminal instead.`,
    })
    logger.error(`[wsl-setup] install failed: exit=${res.code} err=${res.error ?? ''}`)
  }
  emit()
}

// "Restart now" — schedule a Windows restart with a short grace period so the
// user can cancel with `shutdown /a`. Best-effort; the UI always also instructs
// the user to reboot manually.
export function restartWindows(): { success: boolean; error?: string } {
  if (process.platform !== 'win32') return { success: false, error: 'Only supported on Windows.' }
  try {
    const res = spawnSync(
      'shutdown.exe',
      ['/r', '/t', '15', '/c', 'Restarting to finish WSL setup for LTX Desktop local training.'],
      { windowsHide: true },
    )
    if (res.error) return { success: false, error: String(res.error.message ?? res.error) }
    if ((res.status ?? -1) !== 0) {
      return { success: false, error: decodeWsl(res.stderr) || `shutdown exited ${res.status}` }
    }
    return { success: true }
  } catch (e) {
    return { success: false, error: String(e) }
  }
}

// Called once on app startup. If we were mid-setup before a reboot, re-probe and
// advance the persisted stage so the wizard can pick up where it left off.
export function resumeWslSetupIfNeeded(): void {
  const st = getWslSetupState()
  if (st.stage !== 'reboot-required' && st.stage !== 'installing' && st.stage !== 'verifying') {
    return
  }
  // Re-probe and settle the persisted stage to match reality. `false` because a
  // pending reboot is only real when the engine is genuinely still absent.
  settleStageFromProbe(false)
}
