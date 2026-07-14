import { useMemo, useState } from 'react'
import { Loader2, Scissors, X } from 'lucide-react'
import { formatDuration } from '../../lib/lora-quality'
import type { StudioClip } from '../studio/studio-store'

/**
 * Bulk trim recipe applied (non-destructively) to every selected clip:
 *   - `cap`: keep the first `seconds` of each clip (clips already shorter are
 *     left untouched).
 *   - `ends`: remove `head` seconds from the start and `tail` from the end of
 *     each clip.
 */
export type TrimPlan =
  | { mode: 'cap'; seconds: number }
  | { mode: 'ends'; head: number; tail: number }

const EPS = 0.05

function clipDuration(c: StudioClip): number {
  return c.probe?.durationSeconds ?? c.durationSeconds ?? 0
}

// A small labeled seconds input (min 0, one-decimal step).
function SecondsField({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (n: number) => void
}) {
  return (
    <label className="flex-1 space-y-1">
      <span className="text-[11px] text-zinc-400">{label}</span>
      <div className="flex items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 focus-within:border-blue-500/50">
        <input
          type="number"
          min={0}
          step={0.5}
          value={Number.isFinite(value) ? value : 0}
          onChange={(e) => onChange(Math.max(0, Number(e.target.value) || 0))}
          className="w-full bg-transparent text-sm text-zinc-100 focus:outline-none"
        />
        <span className="text-[11px] text-zinc-500">sec</span>
      </div>
    </label>
  )
}

// Dialog for trimming the start/end of many clips at once. Trim is rendered as
// a non-destructive edit per clip (revertible), so clips of different lengths
// each get a sensible window from one shared recipe.
export function BulkTrimModal({
  clips,
  busy,
  onClose,
  onApply,
}: {
  clips: StudioClip[]
  busy: boolean
  onClose: () => void
  onApply: (plan: TrimPlan) => void
}) {
  const [mode, setMode] = useState<'cap' | 'ends'>('cap')
  const [cap, setCap] = useState(5)
  const [head, setHead] = useState(0)
  const [tail, setTail] = useState(0)

  const plan: TrimPlan = mode === 'cap' ? { mode, seconds: cap } : { mode, head, tail }

  // Preview the effect across the selection so the user knows what will change
  // before committing (and which clips are skipped).
  const summary = useMemo(() => {
    let affected = 0
    let skippedShort = 0
    let emptied = 0
    for (const c of clips) {
      const dur = clipDuration(c)
      if (dur <= 0) continue
      if (mode === 'cap') {
        if (dur > cap + EPS) affected += 1
        else skippedShort += 1
      } else {
        const keep = dur - head - tail
        if (head <= 0 && tail <= 0) continue
        if (keep <= EPS) emptied += 1
        else affected += 1
      }
    }
    return { affected, skippedShort, emptied }
  }, [clips, mode, cap, head, tail])

  const nothingToDo =
    summary.affected === 0 ||
    (mode === 'ends' && head <= 0 && tail <= 0) ||
    (mode === 'cap' && cap <= 0)

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-md mx-4">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Scissors className="h-4 w-4 text-blue-300" />
            <h2 className="text-base font-semibold text-white">Trim {clips.length} clip{clips.length === 1 ? '' : 's'}</h2>
          </div>
          <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4">
          {/* Mode */}
          <div className="flex rounded-lg border border-zinc-800 bg-zinc-950 p-0.5 text-xs">
            {([
              ['cap', 'Keep first'],
              ['ends', 'Trim ends'],
            ] as const).map(([value, label]) => (
              <button
                key={value}
                onClick={() => setMode(value)}
                className={`flex-1 rounded-md px-2 py-1.5 font-medium transition-colors ${
                  mode === value ? 'bg-blue-500/15 text-white ring-1 ring-blue-500/30' : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {mode === 'cap' ? (
            <div className="space-y-2">
              <div className="flex items-end gap-2">
                <SecondsField label="Keep the first" value={cap} onChange={setCap} />
              </div>
              <div className="flex flex-wrap gap-1.5">
                {[2, 3, 5, 10].map((s) => (
                  <button
                    key={s}
                    onClick={() => setCap(s)}
                    className={`text-xs px-2.5 py-1 rounded-md border transition-colors ${
                      cap === s ? 'bg-blue-500/15 border-blue-500/40 text-white' : 'border-zinc-800 text-zinc-300 hover:border-zinc-700'
                    }`}
                  >
                    {s}s
                  </button>
                ))}
              </div>
              <p className="text-[11px] text-zinc-500">Each clip is cut to its first {formatDuration(cap)} from the start. Shorter clips are left as-is.</p>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-end gap-2">
                <SecondsField label="Trim from start" value={head} onChange={setHead} />
                <SecondsField label="Trim from end" value={tail} onChange={setTail} />
              </div>
              <p className="text-[11px] text-zinc-500">Removes {formatDuration(head)} from the start and {formatDuration(tail)} from the end of each clip.</p>
            </div>
          )}

          {/* Effect summary */}
          <div className="rounded-md bg-zinc-950 border border-zinc-800 px-3 py-2 text-[11px] text-zinc-400 space-y-0.5">
            <div>
              <span className="text-zinc-200 font-medium">{summary.affected}</span> of {clips.length} clip{clips.length === 1 ? '' : 's'} will be trimmed.
            </div>
            {mode === 'cap' && summary.skippedShort > 0 && (
              <div className="text-zinc-500">{summary.skippedShort} already shorter — skipped.</div>
            )}
            {mode === 'ends' && summary.emptied > 0 && (
              <div className="text-amber-400">{summary.emptied} too short for this trim — skipped.</div>
            )}
          </div>
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex items-center justify-end gap-2">
          <button onClick={onClose} className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white">Cancel</button>
          <button
            onClick={() => onApply(plan)}
            disabled={busy || nothingToDo}
            className="text-xs px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
          >
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />} Trim {summary.affected || ''}
          </button>
        </div>
      </div>
    </div>
  )
}
