import { useState } from 'react'
import {
  AlertTriangle,
  Info,
  Loader2,
  MemoryStick,
  RotateCcw,
  X,
} from 'lucide-react'
import { Tooltip } from '../../components/ui/tooltip'
import type { LoraPreprocessed } from '../../contexts/LoraTrainingContext'

export function PrepRecoveryModal({
  preprocessed,
  datasetName,
  busy,
  onResume,
  onReset,
  onOptimizeMemory,
  onClose,
}: {
  preprocessed: LoraPreprocessed
  datasetName: string
  busy: boolean
  onResume: () => void
  onReset: () => void
  onOptimizeMemory: () => void
  onClose: () => void
}) {
  const [confirmReset, setConfirmReset] = useState(false)
  const failure = preprocessed.error?.trim() || null
  const looksLikeOom = !!failure && /killed mid-run|no exit code|OOM/i.test(failure)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="w-[min(560px,92vw)] rounded-2xl border border-zinc-700 bg-zinc-900 p-5 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <RotateCcw className="h-5 w-5 text-blue-400" />
            <h2 className="text-sm font-semibold text-zinc-100">Resume or reset preprocessing</h2>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300" disabled={busy}>
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="mt-3 text-xs text-zinc-400 flex items-start gap-1.5">
          <span>
            Preprocessing for <span className="text-zinc-200 font-medium">{datasetName}</span> stopped before finishing.
            You can usually pick up where it left off without re-doing the whole dataset.
          </span>
          <Tooltip
            wide
            side="bottom"
            content="The latent-caching step is the one that typically runs out of memory and gets killed. Resume reuses your captions and just re-runs that step; Reset wipes the cached latents and starts over."
          >
            <Info className="h-3.5 w-3.5 shrink-0 text-zinc-600 hover:text-zinc-400 mt-0.5" />
          </Tooltip>
        </p>

        {failure && (
          <pre className="mt-3 max-h-32 overflow-auto whitespace-pre-wrap rounded-lg border border-red-500/30 bg-red-500/[0.07] p-2.5 text-[11px] leading-relaxed text-red-200">
            {failure}
          </pre>
        )}

        {looksLikeOom && (
          <button
            onClick={onOptimizeMemory}
            disabled={busy}
            className="mt-3 flex w-full items-center justify-center gap-1.5 rounded-lg border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-xs font-medium text-blue-200 hover:bg-blue-500/20 disabled:opacity-40"
          >
            <MemoryStick className="h-3.5 w-3.5" />
            Check WSL2 memory
            <Tooltip
              wide
              side="bottom"
              content="A no-exit-code kill is usually system-RAM OOM (fixable here) or GPU-VRAM exhaustion (fix by lowering the resolution bucket or using fewer/shorter clips). This checks your WSL2 RAM and raises it only if it's too small."
            >
              <Info className="h-3.5 w-3.5 text-blue-300/70 hover:text-blue-200" />
            </Tooltip>
          </button>
        )}

        <div className="mt-4 space-y-2.5">
          <button
            onClick={onResume}
            disabled={busy}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40"
          >
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            Resume — keep captions, re-cache latents
          </button>
          <p className="text-center text-[11px] text-zinc-500">
            Reuses your uploaded workspace and the captions already generated. Nothing is deleted.
          </p>

          <div className="my-2 flex items-center gap-2 text-[11px] text-zinc-600">
            <span className="h-px flex-1 bg-zinc-800" /> or <span className="h-px flex-1 bg-zinc-800" />
          </div>

          {!confirmReset ? (
            <button
              onClick={() => setConfirmReset(true)}
              disabled={busy}
              className="flex w-full items-center justify-center gap-2 rounded-lg border border-red-500/40 px-3 py-2 text-xs font-medium text-red-300 hover:bg-red-500/10 disabled:opacity-40"
            >
              <AlertTriangle className="h-3.5 w-3.5" />
              Reset training setup
            </button>
          ) : (
            <div className="rounded-lg border border-red-500/40 bg-red-500/[0.07] p-3">
              <p className="text-[11px] font-semibold text-red-200">This will reset the training setup:</p>
              <ul className="mt-1.5 list-disc pl-4 text-[11px] text-red-200/90 space-y-0.5">
                <li>The cached latents and captions for this run are discarded.</li>
                <li>The training configuration (captioner, resolution, etc.) is cleared.</li>
                <li>You go back to <strong className="text-red-100">Train LoRA</strong> and pick all settings again.</li>
              </ul>
              <p className="mt-2 text-[11px] text-zinc-400">
                Your uploaded dataset clips are kept — only the training setup is reset.
              </p>
              <div className="mt-3 flex gap-2">
                <button
                  onClick={onReset}
                  disabled={busy}
                  className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-red-600 px-3 py-2 text-xs font-medium text-white hover:bg-red-500 disabled:opacity-40"
                >
                  {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  Yes, reset and re-configure
                </button>
                <button
                  onClick={() => setConfirmReset(false)}
                  disabled={busy}
                  className="rounded-lg border border-zinc-700 px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
