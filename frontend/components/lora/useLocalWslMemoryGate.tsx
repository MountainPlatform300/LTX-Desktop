import { useCallback, useRef, useState } from 'react'
import { AlertTriangle, Info, Loader2, RefreshCw } from 'lucide-react'
import type { LoraProvider } from '../../contexts/LoraTrainingContext'
import type { WslMemoryReadinessApi } from '../../../shared/electron-api-schema'
import { Tooltip } from '../ui/tooltip'

// Auto gate for local LoRA runs: makes sure WSL2's memory limit is raised
// (written to `~/.wslconfig` by the Electron main process) AND that the running
// WSL VM actually has that limit, so the trainer's ~23 GB text-encoder load
// doesn't get SIGKILL'd mid-preprocess (the recurring "no exit code" OOM).
//
// Usage at a run trigger site:
//   const gate = useLocalWslMemoryGate()
//   await gate.ensureForRun(provider, async () => { await startTraining(...) })
//   {gate.dialog}
//
// - provider !== 'local': no-op, calls `startFn` immediately.
// - WSL memory already sufficient: calls `startFn` immediately.
// - `.wslconfig` was raised but the live VM is stale: shows ONE concise prompt
//   ("Restart WSL now?"). On confirm, restarts WSL, re-probes, and auto-invokes
//   the pending `startFn` once the VM reports enough RAM — so the user still
//   gets a one-click flow.
// - Any IPC/probe failure: degrades to proceeding (never blocks a run on a
//   diagnostics glitch; the trainer still unloads the inference GPU itself).

type PendingStart = (() => Promise<void>) | null

export function useLocalWslMemoryGate() {
  const [readiness, setReadiness] = useState<WslMemoryReadinessApi | null>(null)
  const [phase, setPhase] = useState<'idle' | 'restarting' | 'restarted' | 'error'>('idle')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const pendingStartRef = useRef<PendingStart>(null)

  const ensureForRun = useCallback(
    async (provider: LoraProvider, startFn: () => Promise<void>): Promise<void> => {
      if (provider !== 'local') {
        await startFn()
        return
      }
      let r: WslMemoryReadinessApi
      try {
        r = await window.electronAPI.ensureWslMemoryReady()
      } catch {
        // Never block a run on a diagnostics/IPC failure.
        await startFn()
        return
      }
      if (r.ready) {
        await startFn()
        return
      }
      if (r.needsRestart) {
        pendingStartRef.current = startFn
        setReadiness(r)
        setPhase('idle')
        setErrorMsg(null)
        return
      }
      // Config error / unknown — best-effort proceed.
      await startFn()
    },
    [],
  )

  const restartNow = useCallback(async () => {
    setPhase('restarting')
    setErrorMsg(null)
    try {
      const res = await window.electronAPI.restartWsl()
      if (!res.success) {
        setPhase('error')
        setErrorMsg(res.error || 'Could not restart WSL.')
        return
      }
      const r = await window.electronAPI.ensureWslMemoryReady()
      if (r.ready) {
        const fn = pendingStartRef.current
        pendingStartRef.current = null
        setReadiness(null)
        setPhase('idle')
        if (fn) await fn()
      } else {
        setPhase('error')
        setErrorMsg(
          r.error ||
            'WSL restarted but its memory is still below the recommended limit. Close other WSL apps and try again, or restart your computer.',
        )
      }
    } catch (e) {
      setPhase('error')
      setErrorMsg(String((e as Error).message ?? e))
    }
  }, [])

  const cancel = useCallback(() => {
    pendingStartRef.current = null
    setReadiness(null)
    setPhase('idle')
    setErrorMsg(null)
  }, [])

  const dialog = readiness ? (
    <div className="fixed inset-0 z-[90] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={cancel} />
      <div className="relative w-full max-w-md mx-4 bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl">
        <div className="px-5 py-4 space-y-3">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0 text-amber-400" />
            <p className="text-sm text-zinc-200 leading-relaxed">
              LTX Desktop raised WSL2&apos;s memory limit so training fits your PC. Restart WSL once to apply it.
              <Tooltip
                wide
                side="bottom"
                content={`Local training loads the LTX-2 text encoder into WSL2's system memory (~${readiness.recommendedMemoryGb} GB peak). WSL2 is currently running with a smaller cap, so it would be killed mid-run. Restarting WSL takes a few seconds and closes anything running in WSL.`}
              >
                <Info className="inline-block h-3.5 w-3.5 ml-1 text-zinc-500 hover:text-zinc-300 align-middle" />
              </Tooltip>
            </p>
          </div>

          {phase === 'error' && errorMsg && (
            <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2">
              <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0 text-red-400" />
              <p className="text-xs text-red-200/90">{errorMsg}</p>
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-zinc-800 flex justify-end gap-2">
          <button
            onClick={cancel}
            className="text-xs px-3 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600"
          >
            Later
          </button>
          <button
            onClick={() => void restartNow()}
            disabled={phase === 'restarting'}
            className="text-xs px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white flex items-center gap-1.5"
          >
            {phase === 'restarting' ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Restart WSL now
          </button>
        </div>
      </div>
    </div>
  ) : null

  return { ensureForRun, dialog }
}
