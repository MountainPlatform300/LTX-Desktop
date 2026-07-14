import { useCallback, useEffect, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Cpu,
  Loader2,
  MemoryStick,
  Power,
  RefreshCw,
  Shield,
  X,
} from 'lucide-react'
import type { WslProbeResultApi, WslSetupStateApi } from '../../../shared/electron-api-schema'
import { WslMemoryOptimizer } from './WslMemoryOptimizer'

// Guided, in-app setup for local (WSL2 + CUDA) LoRA training. The heavy lifting
// lives in the Electron main process (electron/wsl-setup.ts): an elevated
// `wsl --install`, a reboot, then a re-probe that flips the GPU eligibility.
// This modal just drives that flow and resumes it after the reboot.

type Stage = WslSetupStateApi['stage']

// What the user is actually looking at — a blend of the persisted setup stage
// and a fresh capability probe (the probe wins when it says the GPU is ready).
type View = 'unsupported' | 'install' | 'installing' | 'reboot' | 'verifying' | 'ready' | 'error'

// Primarily probe-driven (it reflects real machine state); the persisted stage
// only decides the transient "installing" spinner and the genuine mid-reboot
// wait (which is real only while the WSL engine is still absent).
function deriveView(probe: WslProbeResultApi | null, state: WslSetupStateApi | null): View {
  if (!probe) return 'install'
  if (!probe.platformSupported) return 'unsupported'
  if (probe.cudaInWsl) return 'ready'
  const stage: Stage = state?.stage ?? 'idle'
  if (stage === 'installing') return 'installing'
  if (probe.wslInstalled) return 'verifying' // distro present, GPU not visible yet
  if (stage === 'reboot-required' && !probe.wslEnginePresent) return 'reboot'
  if (stage === 'error') return 'error'
  return 'install'
}

function StepRow({ done, busy, label }: { done: boolean; busy?: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2.5 text-sm">
      {busy ? (
        <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-400" />
      ) : done ? (
        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
      ) : (
        <span className="h-4 w-4 shrink-0 rounded-full border border-zinc-600" />
      )}
      <span className={done ? 'text-zinc-300' : 'text-zinc-400'}>{label}</span>
    </div>
  )
}

export function LocalTrainingSetupWizard({
  onClose,
  onReady,
}: {
  onClose: () => void
  /** Called when the user confirms a ready setup — parent switches to Local GPU. */
  onReady: () => void
}) {
  const [probe, setProbe] = useState<WslProbeResultApi | null>(null)
  const [state, setState] = useState<WslSetupStateApi | null>(null)
  const [busy, setBusy] = useState(false)
  const [restartNote, setRestartNote] = useState<string | null>(null)
  const [showMemory, setShowMemory] = useState(false)

  const refresh = useCallback(async () => {
    const [p, s] = await Promise.all([
      window.electronAPI.probeWsl(),
      window.electronAPI.getWslSetupState(),
    ])
    setProbe(p)
    setState(s)
  }, [])

  // Initial load + live progress from the main process while installing.
  useEffect(() => {
    void refresh()
    const unsub = window.electronAPI.onWslSetupProgress((next) => {
      setState(next)
      // The terminal stages change real capability — re-probe to confirm.
      if (next.stage === 'reboot-required' || next.stage === 'complete' || next.stage === 'error') {
        void window.electronAPI.probeWsl().then(setProbe)
      }
    })
    return unsub
  }, [refresh])

  const view = deriveView(probe, state)
  const engineReady = probe?.wslEnginePresent ?? false

  const handleInstall = useCallback(async () => {
    setBusy(true)
    try {
      await window.electronAPI.startWslInstall()
      await refresh()
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const handleRestart = useCallback(async () => {
    setBusy(true)
    setRestartNote(null)
    try {
      const res = await window.electronAPI.restartWindows()
      if (res.success) {
        setRestartNote('Restarting in ~15 seconds. Save your work. (Run "shutdown /a" to cancel.)')
      } else {
        setRestartNote(`Couldn't restart automatically (${res.error}). Please restart Windows yourself.`)
      }
    } finally {
      setBusy(false)
    }
  }, [])

  const handleRecheck = useCallback(async () => {
    setBusy(true)
    try {
      await refresh()
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const copyCommand = useCallback(() => {
    void navigator.clipboard?.writeText('wsl --install')
  }, [])

  const error = state?.error ?? null

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative flex max-h-[85vh] w-full max-w-lg mx-4 flex-col bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl">
        <div className="flex shrink-0 items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="flex items-center gap-2 text-base font-semibold text-white">
            <Cpu className="h-4 w-4 text-blue-300" /> Set up local GPU training
          </h2>
          <button
            onClick={onClose}
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-4">
          <p className="text-xs text-zinc-500 leading-relaxed">
            Local training runs the LTX-2 trainer on your own GPU inside WSL2 (a lightweight Linux layer
            built into Windows). This is a one-time setup. RunPod stays available either way.
          </p>

          {view === 'unsupported' && (
            <div className="flex items-start gap-2.5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2.5">
              <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0 text-amber-400" />
              <p className="text-xs text-amber-200/90">{probe?.reason ?? 'Local GPU training is not available on this machine.'}</p>
            </div>
          )}

          {view === 'install' && (
            <>
              <div className="space-y-2.5 rounded-lg border border-zinc-800 bg-zinc-800/40 px-3.5 py-3">
                <StepRow done label={`Windows build ${probe?.windowsBuild ?? ''} supports WSL2`} />
                <StepRow done={engineReady} label={engineReady ? 'WSL2 is enabled' : 'Enable WSL2 (needs admin + restart)'} />
                <StepRow done={false} label="Install the Ubuntu Linux distribution" />
                <StepRow done={false} label="Verify the GPU is visible inside WSL" />
              </div>
              <div className="flex items-start gap-2.5 rounded-lg border border-zinc-800 bg-zinc-800/40 px-3 py-2.5">
                <Shield className="h-4 w-4 mt-0.5 shrink-0 text-blue-300" />
                <p className="text-xs text-zinc-400">
                  {engineReady
                    ? 'WSL2 is already enabled on this PC, so this just installs Ubuntu — no admin prompt or restart needed.'
                    : 'Windows will ask for administrator approval, then a restart is needed. After you restart and reopen the app, setup picks up automatically.'}
                </p>
              </div>
            </>
          )}

          {view === 'installing' && (
            <div className="space-y-2.5 rounded-lg border border-zinc-800 bg-zinc-800/40 px-3.5 py-3">
              <StepRow
                busy
                done={false}
                label={engineReady ? 'Installing Ubuntu…' : 'Installing WSL2 — approve the administrator prompt…'}
              />
              <p className="text-[11px] text-zinc-500">
                {engineReady
                  ? 'Downloading and registering the Ubuntu distribution. This can take a few minutes.'
                  : 'A User Account Control window should have appeared. This can take a few minutes.'}
              </p>
            </div>
          )}

          {view === 'reboot' && (
            <>
              <div className="flex items-start gap-2.5 rounded-lg border border-blue-500/30 bg-blue-500/10 px-3 py-2.5">
                <Power className="h-4 w-4 mt-0.5 shrink-0 text-blue-300" />
                <p className="text-xs text-blue-100/90">
                  WSL2 is installed. <span className="font-semibold">Windows needs to restart</span> to finish.
                  After it reboots, reopen LTX Desktop — setup resumes on its own.
                </p>
              </div>
              {restartNote && <p className="text-[11px] text-amber-300/90">{restartNote}</p>}
            </>
          )}

          {view === 'verifying' && (
            <div className="space-y-2.5 rounded-lg border border-zinc-800 bg-zinc-800/40 px-3.5 py-3">
              <StepRow done label="WSL2 installed" />
              <StepRow busy={busy} done={false} label="Checking that the GPU is visible inside WSL…" />
              <p className="text-[11px] text-zinc-500">
                {probe?.reason ??
                  'If this persists, install the latest NVIDIA Windows driver (it includes CUDA-on-WSL) and restart.'}
              </p>
            </div>
          )}

          {view === 'ready' && (
            <>
              <div className="flex items-start gap-2.5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2.5">
                <CheckCircle2 className="h-4 w-4 mt-0.5 shrink-0 text-emerald-400" />
                <p className="text-xs text-emerald-100/90">
                  Local GPU training is ready{probe?.defaultDistro ? ` (${probe.defaultDistro})` : ''}. You can now train
                  on this machine.
                </p>
              </div>
              <button
                onClick={() => setShowMemory(true)}
                className="w-full flex items-center gap-2.5 rounded-lg border border-zinc-800 bg-zinc-800/40 px-3 py-2.5 text-left hover:bg-zinc-800/70"
              >
                <MemoryStick className="h-4 w-4 shrink-0 text-blue-300" />
                <span className="flex-1">
                  <span className="block text-sm text-zinc-200">Optimize WSL2 memory</span>
                  <span className="block text-[11px] text-zinc-500 mt-0.5">
                    Recommended before your first run — raises WSL2's RAM limit so preprocess doesn't run out of memory.
                  </span>
                </span>
              </button>
            </>
          )}

          {view === 'error' && (
            <div className="flex items-start gap-2.5 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2.5">
              <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0 text-red-400" />
              <p className="text-xs text-red-200/90">{error ?? 'Setup failed.'}</p>
            </div>
          )}

          {/* Manual fallback — always available. */}
          {view !== 'unsupported' && view !== 'ready' && (
            <details className="text-xs text-zinc-500">
              <summary className="cursor-pointer hover:text-zinc-300">Prefer to do it manually?</summary>
              <div className="mt-2 space-y-1.5">
                <p>Open an Administrator terminal and run, then restart Windows:</p>
                <div className="flex items-center gap-2 rounded-md border border-zinc-700 bg-black/40 px-2.5 py-1.5 font-mono text-zinc-300">
                  <span className="flex-1">wsl --install</span>
                  <button onClick={copyCommand} className="text-zinc-500 hover:text-white" title="Copy">
                    <Copy className="h-3.5 w-3.5" />
                  </button>
                </div>
                <p>Then reopen this app and click &ldquo;Re-check&rdquo;.</p>
              </div>
            </details>
          )}
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex justify-end gap-2 shrink-0">
          <button
            onClick={onClose}
            className="text-xs px-3 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600"
          >
            Close
          </button>

          {view === 'install' && (
            <button
              onClick={handleInstall}
              disabled={busy}
              className="text-xs px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white flex items-center gap-1.5"
            >
              {busy ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : engineReady ? (
                <Cpu className="h-3.5 w-3.5" />
              ) : (
                <Shield className="h-3.5 w-3.5" />
              )}
              {engineReady ? 'Install Ubuntu' : 'Install WSL2'}
            </button>
          )}

          {view === 'reboot' && (
            <button
              onClick={handleRestart}
              disabled={busy}
              className="text-xs px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white flex items-center gap-1.5"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Power className="h-3.5 w-3.5" />}
              Restart Windows now
            </button>
          )}

          {(view === 'verifying' || view === 'error') && (
            <button
              onClick={handleRecheck}
              disabled={busy}
              className="text-xs px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white flex items-center gap-1.5"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              Re-check
            </button>
          )}

          {view === 'ready' && (
            <button
              onClick={onReady}
              className="text-xs px-3.5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5"
            >
              <Cpu className="h-3.5 w-3.5" /> Use Local GPU
            </button>
          )}
        </div>
      </div>

      {showMemory && <WslMemoryOptimizer onClose={() => setShowMemory(false)} />}
    </div>
  )
}
