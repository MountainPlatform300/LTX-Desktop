import { AlertTriangle, Info } from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import {
  clipWarnings,
  datasetHealth,
  formatDuration,
  probeBadges,
} from '../../lib/lora-quality'
import type { StudioClip } from './studio-store'

/**
 * Selection-aware right panel.
 *  - 0 selected → dataset overview + readiness score.
 *  - 1 selected → full clip detail (preview, probe badges, warnings).
 *  - N selected → batch summary (bulk editing lands in M2).
 */
export function Inspector({
  clips,
  selectedIds,
}: {
  clips: StudioClip[]
  selectedIds: Set<string>
}) {
  const selected = clips.filter((c) => selectedIds.has(c.id))

  if (selected.length === 0) return <DatasetOverview clips={clips} />
  if (selected.length === 1) return <ClipDetail clip={selected[0]} />
  return <BatchSummary clips={selected} />
}

function ScoreRing({ score }: { score: number }) {
  const tone = score >= 75 ? 'text-emerald-400' : score >= 50 ? 'text-amber-400' : 'text-red-400'
  return (
    <div className="flex items-baseline gap-1">
      <span className={`text-2xl font-semibold ${tone}`}>{score}</span>
      <span className="text-xs text-zinc-500">/100 ready</span>
    </div>
  )
}

function DatasetOverview({ clips }: { clips: StudioClip[] }) {
  const health = datasetHealth(clips.map((c) => ({ caption: c.caption, probe: c.probe })))
  return (
    <div className="p-4 space-y-4">
      <div>
        <p className="text-xs font-medium text-zinc-400 mb-1">Dataset readiness</p>
        <ScoreRing score={health.score} />
      </div>
      <dl className="space-y-1.5 text-xs">
        <Row label="Clips" value={`${health.clipCount}`} />
        <Row label="Captioned" value={`${health.captionedCount}/${health.clipCount}`} />
        <Row label="Total length" value={formatDuration(health.totalDurationSeconds)} />
        <Row label="Aspect ratios" value={health.aspectRatios.length ? health.aspectRatios.join(', ') : '—'} />
        <Row label="Quality errors" value={`${health.errorCount}`} tone={health.errorCount > 0 ? 'error' : undefined} />
      </dl>
      <p className="text-[11px] text-zinc-600 leading-relaxed flex gap-1.5">
        <Info className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
        Select clips to edit them, or click a filter to act on a whole category.
      </p>
    </div>
  )
}

function ClipDetail({ clip }: { clip: StudioClip }) {
  const warnings = clipWarnings({ caption: clip.caption, probe: clip.probe })
  const posterUrl = clip.posterPath ? pathToFileUrl(clip.posterPath) : null
  return (
    <div className="p-4 space-y-3">
      <div className="aspect-video w-full rounded-md bg-zinc-900 bg-cover bg-center" style={posterUrl ? { backgroundImage: `url("${posterUrl}")` } : undefined} />
      <div>
        <p className="text-xs font-medium text-zinc-400 mb-1">Caption</p>
        <p className="text-xs text-zinc-200 leading-relaxed">
          {clip.caption || <span className="text-zinc-600 italic">No caption yet</span>}
        </p>
      </div>
      {clip.probe && (
        <div className="flex flex-wrap gap-1">
          {probeBadges(clip.probe).map((b) => (
            <span key={b} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300">{b}</span>
          ))}
        </div>
      )}
      {warnings.length > 0 && (
        <div className="space-y-1">
          {warnings.map((w, i) => (
            <p
              key={i}
              className={`text-[11px] flex items-start gap-1.5 ${w.level === 'error' ? 'text-red-400' : 'text-amber-400'}`}
            >
              <AlertTriangle className="h-3 w-3 flex-shrink-0 mt-0.5" />
              {w.text}
            </p>
          ))}
        </div>
      )}
      <p className="text-[10px] text-zinc-600 font-mono break-all">{clip.localPath}</p>
    </div>
  )
}

function BatchSummary({ clips }: { clips: StudioClip[] }) {
  const totalSeconds = clips.reduce((acc, c) => acc + (c.probe?.durationSeconds ?? c.durationSeconds ?? 0), 0)
  const uncaptioned = clips.filter((c) => !c.caption.trim()).length
  return (
    <div className="p-4 space-y-3">
      <p className="text-sm font-medium text-white">{clips.length} clips selected</p>
      <dl className="space-y-1.5 text-xs">
        <Row label="Combined length" value={formatDuration(totalSeconds)} />
        <Row label="Uncaptioned" value={`${uncaptioned}`} />
      </dl>
      <p className="text-[11px] text-zinc-600 leading-relaxed">
        Bulk trim, crop, caption, and restyle for the selection arrive in the next milestone.
      </p>
    </div>
  )
}

function Row({ label, value, tone }: { label: string; value: string; tone?: 'error' }) {
  return (
    <div className="flex items-center justify-between">
      <dt className="text-zinc-500">{label}</dt>
      <dd className={tone === 'error' ? 'text-red-400' : 'text-zinc-200'}>{value}</dd>
    </div>
  )
}
