import { useMemo, useState } from 'react'
import { ArrowRight, Link2, X } from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import type { StudioClip } from '../studio/studio-store'

function clipName(clip: StudioClip): string {
  return clip.caption.trim() || clip.localPath.split('/').pop() || clip.localPath
}

function Thumb({ clip, role }: { clip: StudioClip; role: 'target' | 'ref' }) {
  const url = clip.posterPath ? pathToFileUrl(clip.posterPath) : null
  return (
    <div
      className="h-12 w-20 shrink-0 rounded-md bg-zinc-950 bg-cover bg-center border border-zinc-700 relative"
      style={url ? { backgroundImage: `url("${url}")` } : {}}
    >
      <span
        className={`absolute top-0.5 left-0.5 text-[8px] px-1 rounded uppercase tracking-wide ${
          role === 'target' ? 'bg-blue-500/80 text-white' : 'bg-zinc-700 text-zinc-200'
        }`}
      >
        {role === 'target' ? 'Output' : 'Input'}
      </span>
    </div>
  )
}

/**
 * Manually group selected clips into an IC-LoRA example: the user picks the
 * single *target* (the edited "after" / output) and the other clip becomes its
 * conditioning *input* (the "before" reference). The released LTX-2 trainer
 * conditions on a single reference, so an example is exactly one input plus one
 * output; selecting more than two clips is rejected here.
 */
export function GroupPairModal({
  clips,
  onClose,
  onConfirm,
}: {
  clips: StudioClip[]
  onClose: () => void
  onConfirm: (targetId: string, referenceIds: string[]) => void
}) {
  // Default the target to the only AI-derived clip if there is exactly one
  // (the typical "edited output"), else the last selected clip.
  const defaultTarget = useMemo(() => {
    const derived = clips.filter((c) => c.origin === 'ai_derived')
    if (derived.length === 1) return derived[0].id
    return clips[clips.length - 1]?.id ?? null
  }, [clips])

  const [targetId, setTargetId] = useState<string | null>(defaultTarget)
  const references = clips.filter((c) => c.id !== targetId)
  // An example is exactly one input conditioning one output. More than two
  // selected clips can't be expressed against the single-reference trainer.
  const tooMany = clips.length > 2
  const canConfirm = !!targetId && references.length === 1

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="w-full max-w-lg rounded-xl bg-zinc-900 border border-zinc-700 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
          <h2 className="flex items-center gap-2 text-sm font-medium text-zinc-100">
            <Link2 className="h-4 w-4 text-blue-400" /> Group into an example
          </h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-4 py-3 space-y-3">
          <p className="text-[12px] text-zinc-400">
            Pick the <span className="text-blue-300 font-medium">output</span> (the edited
            result the model should produce). The other clip becomes its{' '}
            <span className="text-zinc-300 font-medium">input</span> (the conditioning clip).
          </p>

          {tooMany && (
            <p className="rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-2 text-[11px] text-amber-300">
              An example is exactly one input and one output. The trainer
              conditions on a single reference, so select just two clips
              (you have {clips.length} selected).
            </p>
          )}

          <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
            {clips.map((clip) => {
              const isTarget = clip.id === targetId
              return (
                <label
                  key={clip.id}
                  className={`flex items-center gap-3 rounded-lg border px-2.5 py-2 cursor-pointer transition-colors ${
                    isTarget ? 'border-blue-500/60 bg-blue-500/10' : 'border-zinc-800 hover:border-zinc-700'
                  }`}
                >
                  <input
                    type="radio"
                    name="pair-target"
                    checked={isTarget}
                    onChange={() => setTargetId(clip.id)}
                    className="accent-blue-500"
                  />
                  <Thumb clip={clip} role={isTarget ? 'target' : 'ref'} />
                  <span className="flex-1 min-w-0 text-[12px] text-zinc-300 truncate">{clipName(clip)}</span>
                  <span className="text-[10px] text-zinc-500 shrink-0">{isTarget ? 'Output' : 'Input'}</span>
                </label>
              )
            })}
          </div>

          <div className="flex items-center gap-2 text-[11px] text-zinc-500">
            <span>{references.length} input{references.length === 1 ? '' : 's'}</span>
            <ArrowRight className="h-3 w-3" />
            <span>1 output</span>
          </div>
        </div>

        <div className="flex justify-end gap-2 px-4 py-3 border-t border-zinc-800">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-[12px] rounded-md text-zinc-300 hover:bg-zinc-800"
          >
            Cancel
          </button>
          <button
            disabled={!canConfirm}
            onClick={() => {
              if (!canConfirm || !targetId) return
              onConfirm(
                targetId,
                references.map((c) => c.id),
              )
              onClose()
            }}
            className="px-3 py-1.5 text-[12px] rounded-md bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Create example
          </button>
        </div>
      </div>
    </div>
  )
}
