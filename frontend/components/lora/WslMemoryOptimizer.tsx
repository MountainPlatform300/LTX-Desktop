import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, CheckCircle2, Info, Loader2, MemoryStick, RefreshCw, X } from 'lucide-react'
import type { WslMemoryProbeApi } from '../../../shared/electron-api-schema'
import { Tooltip } from '../ui/tooltip'

// One-click WSL2 memory optimizer for local LoRA training. The LTX-2 trainer
// loads the Gemma3 12B text encoder into system RAM (~23 GB peak even in 8-bit),
// which blows past WSL2's default ~half-host-RAM cap and gets the process
// SIGKILL'd mid-preprocess (the "no exit code" OOM). This reads the user's
// `~/.wslconfig`, raises `memory` + `swap` to a value computed from the actual
// host RAM, then restarts WSL so the change takes effect — no manual file
// editing. The Electron main process does the work; this modal just drives it.

type Phase = 'idle' | 'applied' | 'restarted' | 'error'

export function WslMemoryOptimizer({ onClose }: { onClose: () => void }) {
  const [probe, setProbe] = useState<WslMemoryProbeApi | null>(null)
  const [busy, setBusy] = useState(false)
  const [phase, setPhase] = useState<Phase>('idle')
  const [message, setMessage] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const p = await window.electronAPI.probeWslMemory()
      setProbe(p)
    } catch (e) {
      setMessage(String((e as Error).message ?? e))
      setPhase('error')
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const handleApply = useCallback(async () => {
    setBusy(true)
    setMessage(null)
    try {
      const res = await window.electronAPI.configureWslMemory()
      if (res.error) {
        setMessage(res.error)
        setPhase('error')
        return
      }
      if (res.alreadyConfigured) {
        setPhase('idle')
        setMessage(null)
      } else {
        setPhase('applied')
      }
      await refresh()
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const handleRestartWsl = useCallback(async () => {
    setBusy(true)
    setMessage(null)
    try {
      const res = await window.electronAPI.restartWsl()
      if (!res.success) {
        setMessage(res.error)
        setPhase('error')
        return
      }
      setPhase('restarted')
    } finally {
      setBusy(false)
    }
  }, [])

  const hostRam = probe?.hostRamGb ?? null
  const currentMem = probe?.configuredMemoryGb ?? null
  const recMem = probe?.recommendedMemoryGb ?? null
  const recSwap = probe?.recommendedSwapGb ?? null
  const needsUpdate = probe?.needsUpdate ?? false

  return (
    <div className="fixed inset-0 z-[85] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative flex max-h-[85vh] w-full max-w-lg mx-4 flex-col bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl">
        <div className="flex shrink-0 items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="flex items-center gap-2 text-base font-semibold text-white">
            <MemoryStick className="h-4 w-4 text-blue-300" /> Optimize WSL2 memory
          </h2>
          <button
            onClick={onClose}
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-4">
          <p className="text-xs text-zinc-500 leading-relaxed flex items-start gap-1.5">
            <span>Raise WSL2's memory limit to fit your PC — computed automatically from your RAM.</span>
            <Tooltip
              wide
              side="bottom"
              content="Local training loads the LTX-2 text encoder into WSL2's system memory (~23 GB peak). WSL2's default cap is often too small for that, which kills the preprocess mid-run. This raises it for you so you don't edit config files by hand."
            >
              <Info className="h-3.5 w-3.5 shrink-0 text-zinc-600 hover:text-zinc-400 mt-0.5" />
            </Tooltip>
          </p>

          {probe && (
            <div className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-800/40 px-3.5 py-2.5 text-xs">
              <span className="text-zinc-500">WSL2 memory limit</span>
              <span
                className={
                  needsUpdate ? 'text-amber-300 font-medium' : 'text-emerald-300 font-medium'
                }
              >
                {currentMem !== null
                  ? `${currentMem} GB`
                  : `Default (~${hostRam !== null ? Math.floor(hostRam / 2) : '?'} GB)`}
                <Tooltip
                  wide
                  side="left"
                  content={
                    <div className="space-y-0.5">
                      <div>Your PC&apos;s RAM: {hostRam ?? '—'} GB</div>
                      <div>Recommended for training: {recMem ?? '—'} GB</div>
                      <div>Recommended swap: {recSwap ?? '—'} GB</div>
                    </div>
                  }
                >
                  <Info className="inline-block h-3 w-3 ml-1.5 text-zinc-600 hover:text-zinc-400 align-middle" />
                </Tooltip>
              </span>
            </div>
          )}

          {probe?.reason && needsUpdate && (
            <div className="flex items-start gap-2.5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2.5">
              <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0 text-amber-400" />
              <p className="text-xs text-amber-200/90">{probe.reason}</p>
            </div>
          )}

          {phase === 'applied' && (
            <div className="flex items-start gap-2.5 rounded-lg border border-blue-500/30 bg-blue-500/10 px-3 py-2.5">
              <RefreshCw className="h-4 w-4 mt-0.5 shrink-0 text-blue-300" />
              <p className="text-xs text-blue-100/90 flex items-start gap-1.5">
                <span>Settings applied. Restart WSL so the new limit takes effect.</span>
                <Tooltip
                  wide
                  side="bottom"
                  content="Restarting WSL closes any running WSL process, including an in-flight training job. Make sure nothing is training right now."
                >
                  <Info className="h-3.5 w-3.5 shrink-0 text-blue-300/70 hover:text-blue-200 mt-0.5" />
                </Tooltip>
              </p>
            </div>
          )}

          {phase === 'restarted' && (
            <div className="flex items-start gap-2.5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2.5">
              <CheckCircle2 className="h-4 w-4 mt-0.5 shrink-0 text-emerald-400" />
              <p className="text-xs text-emerald-100/90">
                WSL2 restarted. Re-run preprocessing — Resume skips the captioning you already did.
              </p>
            </div>
          )}

          {phase === 'error' && message && (
            <div className="flex items-start gap-2.5 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2.5">
              <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0 text-red-400" />
              <p className="text-xs text-red-200/90">{message}</p>
            </div>
          )}

          {phase === 'idle' && !needsUpdate && probe && (
            <div className="flex items-start gap-2.5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2.5">
              <CheckCircle2 className="h-4 w-4 mt-0.5 shrink-0 text-emerald-400" />
              <p className="text-xs text-emerald-100/90">
                WSL2's memory limit already looks sufficient for training. Nothing to change.
              </p>
            </div>
          )}
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex justify-end gap-2 shrink-0">
          <button
            onClick={onClose}
            className="text-xs px-3 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600"
          >
            Close
          </button>

          {needsUpdate && phase !== 'applied' && phase !== 'restarted' && (
            <button
              onClick={handleApply}
              disabled={busy || !probe}
              className="text-xs px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white flex items-center gap-1.5"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <MemoryStick className="h-3.5 w-3.5" />}
              Apply recommended settings
            </button>
          )}

          {phase === 'applied' && (
            <button
              onClick={handleRestartWsl}
              disabled={busy}
              className="text-xs px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white flex items-center gap-1.5"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              Restart WSL to apply
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
