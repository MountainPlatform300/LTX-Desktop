import { CheckCircle2, Circle, XCircle } from 'lucide-react'
import {
  formatDuration,
  preflightChecks,
  type ClipLike,
  type DatasetHealth,
} from '../../lib/lora-quality'

function scoreTone(score: number): { bar: string; text: string; label: string } {
  if (score >= 75) return { bar: 'bg-green-500', text: 'text-green-400', label: 'Looks great' }
  if (score >= 45) return { bar: 'bg-amber-500', text: 'text-amber-400', label: 'Usable' }
  return { bar: 'bg-red-500', text: 'text-red-400', label: 'Needs work' }
}

// Compact readiness meter shown under the clip list. Summarizes the
// dataset's training-readiness without blocking anything.
export function DatasetHealthMeter({ health }: { health: DatasetHealth }) {
  const tone = scoreTone(health.score)
  const chips: string[] = [
    `${health.clipCount} clip${health.clipCount === 1 ? '' : 's'}`,
    `${health.captionedCount}/${health.clipCount} captioned`,
  ]
  if (health.totalDurationSeconds > 0) chips.push(`${formatDuration(health.totalDurationSeconds)} total`)
  if (health.aspectRatios.length > 0) {
    chips.push(health.aspectConsistent ? health.aspectRatios[0] : `mixed (${health.aspectRatios.length})`)
  }

  return (
    <div className="bg-zinc-800/40 rounded-lg px-3 py-2.5 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-300">Dataset health</span>
        <span className={`text-xs font-medium ${tone.text}`}>{tone.label}</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-zinc-900 overflow-hidden">
        <div className={`h-full ${tone.bar} transition-all duration-300`} style={{ width: `${health.score}%` }} />
      </div>
      <div className="flex flex-wrap gap-1">
        {chips.map((c) => (
          <span key={c} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-400">{c}</span>
        ))}
        {health.errorCount > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400">{health.errorCount} error{health.errorCount === 1 ? '' : 's'}</span>
        )}
        {health.warningCount > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400">{health.warningCount} warning{health.warningCount === 1 ? '' : 's'}</span>
        )}
      </div>
    </div>
  )
}

// Pre-upload checklist. Blocker checks that fail render in red; soft checks
// render amber/neutral. Returns nothing about gating — the caller reads
// `preflightChecks` to decide whether to confirm.
export function PreflightChecklist({
  clips,
  triggerWord,
  requireAudio,
}: {
  clips: ClipLike[]
  triggerWord?: string | null
  requireAudio?: boolean
}) {
  const checks = preflightChecks(clips, { triggerWord, requireAudio })
  return (
    <div className="space-y-1.5">
      {checks.map((c) => (
        <div key={c.label} className="flex items-center justify-between gap-2 text-xs">
          <span className="flex items-center gap-1.5">
            {c.ok ? (
              <CheckCircle2 className="h-3.5 w-3.5 text-green-400" />
            ) : c.blocker ? (
              <XCircle className="h-3.5 w-3.5 text-red-400" />
            ) : (
              <Circle className="h-3.5 w-3.5 text-amber-400" />
            )}
            <span className={c.ok ? 'text-zinc-300' : c.blocker ? 'text-red-300' : 'text-amber-300'}>{c.label}</span>
          </span>
          {c.detail && <span className="text-[10px] text-zinc-500 font-mono">{c.detail}</span>}
        </div>
      ))}
    </div>
  )
}
