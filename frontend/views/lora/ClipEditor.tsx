import { useMemo, useRef, useState } from 'react'
import { Film, Loader2, Play, RotateCcw, Volume2, VolumeX, X } from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import { centeredCrop, formatDuration, snapDown32 } from '../../lib/lora-quality'
import type { ClipEdits } from '../../contexts/LoraTrainingContext'
import type { StudioClip } from '../studio/studio-store'

type AspectId = 'original' | '16:9' | '9:16' | '1:1' | '4:3'

const ASPECTS: Array<{ id: AspectId; label: string; ratio: [number, number] | null }> = [
  { id: 'original', label: 'Original', ratio: null },
  { id: '16:9', label: '16:9', ratio: [16, 9] },
  { id: '9:16', label: '9:16', ratio: [9, 16] },
  { id: '1:1', label: '1:1', ratio: [1, 1] },
  { id: '4:3', label: '4:3', ratio: [4, 3] },
]

function aspectFromEdits(clip: StudioClip): AspectId {
  const crop = clip.edits?.crop
  if (!crop || !clip.probe) return 'original'
  const r = crop.width / crop.height
  let best: AspectId = 'original'
  let bestErr = Infinity
  for (const a of ASPECTS) {
    if (!a.ratio) continue
    const err = Math.abs(a.ratio[0] / a.ratio[1] - r)
    if (err < bestErr) {
      bestErr = err
      best = a.id
    }
  }
  return bestErr < 0.02 ? best : 'original'
}

// Focused trim + crop editor for one clip. Trim uses the pre-rendered sprite
// filmstrip for scrubbing; crop is preset-based (centered, bucket-legal). On
// apply, the parent renders the derivative via `applyClipEdits`.
export function ClipEditor({
  clip,
  busy,
  onClose,
  onApply,
  onRevert,
}: {
  clip: StudioClip
  busy: boolean
  onClose: () => void
  onApply: (edits: ClipEdits) => void
  onRevert: () => void
}) {
  const duration = clip.probe?.durationSeconds ?? clip.durationSeconds ?? 0
  const tiles = clip.spriteTiles ?? 0
  const spriteUrl = clip.spritePath ? pathToFileUrl(clip.spritePath) : null
  const posterUrl = clip.posterPath ? pathToFileUrl(clip.posterPath) : null

  const initialTrim = clip.edits?.trim
  const [startFrac, setStartFrac] = useState(() =>
    duration > 0 && initialTrim ? Math.max(0, initialTrim.startSeconds / duration) : 0,
  )
  const [endFrac, setEndFrac] = useState(() =>
    duration > 0 && initialTrim ? Math.min(1, initialTrim.endSeconds / duration) : 1,
  )
  const [aspect, setAspect] = useState<AspectId>(() => aspectFromEdits(clip))
  const [scrub, setScrub] = useState<number | null>(null)

  const baseSpeed = clip.edits?.speed ?? 1
  const baseFps = clip.edits?.fps ?? null
  const baseMute = clip.edits?.mute ?? false
  const baseReverse = clip.edits?.reverse ?? false
  const [speed, setSpeed] = useState(baseSpeed)
  const [fps, setFps] = useState<number | null>(baseFps)
  const [mute, setMute] = useState(baseMute)
  const [reverse, setReverse] = useState(baseReverse)
  const [showVideo, setShowVideo] = useState(false)
  const localUrl = pathToFileUrl(clip.localPath)

  const barRef = useRef<HTMLDivElement | null>(null)
  const dragging = useRef<'start' | 'end' | null>(null)

  const tileIndex = (frac: number) => (tiles > 1 ? Math.round(frac * (tiles - 1)) : 0)
  const previewFrac = scrub ?? startFrac

  const startSeconds = startFrac * duration
  const endSeconds = endFrac * duration
  const trimmed = startFrac > 0.001 || endFrac < 0.999
  const aspectDef = ASPECTS.find((a) => a.id === aspect) ?? ASPECTS[0]
  const cropRect = useMemo(() => {
    if (!aspectDef.ratio || !clip.probe) return null
    return centeredCrop(clip.probe.width, clip.probe.height, aspectDef.ratio[0], aspectDef.ratio[1])
  }, [aspectDef, clip.probe])

  const timingChanged =
    speed !== baseSpeed || fps !== baseFps || mute !== baseMute || reverse !== baseReverse
  const hasChange = trimmed || aspect !== aspectFromEdits(clip) || timingChanged
  const cropDims = cropRect ? `${snapDown32(cropRect.width)}×${snapDown32(cropRect.height)}` : null

  const pointerFrac = (clientX: number): number => {
    const el = barRef.current
    if (!el) return 0
    const rect = el.getBoundingClientRect()
    return Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const frac = pointerFrac(e.clientX)
    setScrub(frac)
    if (!dragging.current) return
    if (dragging.current === 'start') setStartFrac(Math.min(frac, endFrac - 0.02))
    else setEndFrac(Math.max(frac, startFrac + 0.02))
  }

  const handleStyle = (frac: number): React.CSSProperties =>
    spriteUrl && tiles > 0
      ? {
          backgroundImage: `url("${spriteUrl}")`,
          backgroundSize: `${tiles * 100}% 100%`,
          backgroundPosition: `${tiles > 1 ? (tileIndex(frac) / (tiles - 1)) * 100 : 0}% 50%`,
        }
      : posterUrl
        ? { backgroundImage: `url("${posterUrl}")`, backgroundSize: 'cover', backgroundPosition: 'center' }
        : {}

  const apply = () => {
    const edits: ClipEdits = {
      trim: trimmed && duration > 0 ? { startSeconds, endSeconds } : null,
      crop: cropRect ?? null,
      fps: fps ?? null,
      speed: speed !== 1 ? speed : null,
      mute,
      reverse,
    }
    onApply(edits)
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-2xl mx-4">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="text-base font-semibold text-white">Edit clip</h2>
          <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-5">
          {/* Preview with crop overlay */}
          <div className="relative w-full aspect-video rounded-lg overflow-hidden bg-zinc-950" style={showVideo ? undefined : handleStyle(previewFrac)}>
            {showVideo && (
              <video src={localUrl} controls muted={mute} loop className="w-full h-full object-contain" />
            )}
            <button
              onClick={() => setShowVideo((v) => !v)}
              className="absolute top-2 right-2 z-10 h-7 w-7 flex items-center justify-center rounded-md bg-black/60 text-zinc-200 hover:text-white"
              title={showVideo ? 'Show filmstrip' : 'Play video'}
            >
              {showVideo ? <Film className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            </button>
            {!showVideo && cropRect && clip.probe && (
              <>
                <div className="absolute inset-0 bg-black/50" />
                <div
                  className="absolute border-2 border-blue-400 bg-transparent shadow-[0_0_0_9999px_rgba(0,0,0,0.5)]"
                  style={{
                    left: `${(cropRect.x / clip.probe.width) * 100}%`,
                    top: `${(cropRect.y / clip.probe.height) * 100}%`,
                    width: `${(cropRect.width / clip.probe.width) * 100}%`,
                    height: `${(cropRect.height / clip.probe.height) * 100}%`,
                  }}
                />
              </>
            )}
          </div>

          {/* Trim filmstrip */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-zinc-400">
              <span>Trim</span>
              <span className="font-mono">
                {formatDuration(startSeconds)} → {formatDuration(endSeconds)} ({formatDuration(Math.max(0, endSeconds - startSeconds))})
              </span>
            </div>
            <div
              ref={barRef}
              className="relative h-14 rounded-md overflow-hidden bg-zinc-950 cursor-ew-resize select-none"
              style={
                spriteUrl && tiles > 0
                  ? { backgroundImage: `url("${spriteUrl}")`, backgroundSize: `${tiles * 100}% 100%`, backgroundRepeat: 'no-repeat' }
                  : posterUrl
                    ? { backgroundImage: `url("${posterUrl}")`, backgroundSize: 'cover' }
                    : undefined
              }
              onPointerMove={onPointerMove}
              onPointerLeave={() => {
                if (!dragging.current) setScrub(null)
              }}
              onPointerUp={() => {
                dragging.current = null
              }}
            >
              {/* dimmed regions outside the trim window */}
              <div className="absolute inset-y-0 left-0 bg-black/60" style={{ width: `${startFrac * 100}%` }} />
              <div className="absolute inset-y-0 right-0 bg-black/60" style={{ width: `${(1 - endFrac) * 100}%` }} />
              {/* handles */}
              {(['start', 'end'] as const).map((which) => {
                const frac = which === 'start' ? startFrac : endFrac
                return (
                  <div
                    key={which}
                    onPointerDown={(e) => {
                      e.stopPropagation()
                      ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
                      dragging.current = which
                    }}
                    className="absolute top-0 bottom-0 w-2 -ml-1 bg-blue-400 cursor-ew-resize"
                    style={{ left: `${frac * 100}%` }}
                  />
                )
              })}
            </div>
          </div>

          {/* Crop */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-zinc-400">
              <span>Crop</span>
              {cropDims && <span className="font-mono">{cropDims}</span>}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {ASPECTS.map((a) => (
                <button
                  key={a.id}
                  onClick={() => setAspect(a.id)}
                  className={`text-xs px-3 py-1.5 rounded-md border transition-colors ${
                    aspect === a.id ? 'bg-blue-500/15 border-blue-500/40 text-white' : 'border-zinc-800 text-zinc-300 hover:border-zinc-700'
                  }`}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>

          {/* Timing & audio */}
          <div className="space-y-2.5 border-t border-zinc-800 pt-4">
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[11px] text-zinc-400">
                <span>Speed</span>
                <span className="font-mono">{speed.toFixed(2)}×</span>
              </div>
              <input
                type="range"
                min={0.25}
                max={4}
                step={0.05}
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[11px] text-zinc-400">Frame rate</span>
              <div className="flex gap-1.5">
                {([
                  [null, 'Keep'],
                  [24, '24'],
                  [25, '25'],
                  [30, '30'],
                ] as const).map(([val, label]) => (
                  <button
                    key={label}
                    onClick={() => setFps(val)}
                    className={`text-xs px-2.5 py-1 rounded-md border transition-colors ${
                      fps === val ? 'bg-blue-500/15 border-blue-500/40 text-white' : 'border-zinc-800 text-zinc-300 hover:border-zinc-700'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setMute((m) => !m)}
                className={`flex-1 text-xs px-3 py-1.5 rounded-md border flex items-center justify-center gap-1.5 transition-colors ${
                  mute ? 'bg-blue-500/15 border-blue-500/40 text-white' : 'border-zinc-800 text-zinc-300 hover:border-zinc-700'
                }`}
              >
                {mute ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
                {mute ? 'Muted' : 'Mute audio'}
              </button>
              <button
                onClick={() => setReverse((r) => !r)}
                className={`flex-1 text-xs px-3 py-1.5 rounded-md border flex items-center justify-center gap-1.5 transition-colors ${
                  reverse ? 'bg-blue-500/15 border-blue-500/40 text-white' : 'border-zinc-800 text-zinc-300 hover:border-zinc-700'
                }`}
              >
                <RotateCcw className="h-3.5 w-3.5" /> Reverse
              </button>
            </div>
          </div>
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex items-center justify-between gap-2">
          <button
            onClick={onRevert}
            disabled={busy || (!clip.edits && clip.localPath === clip.sourcePath)}
            className="text-xs px-3 py-1.5 rounded-lg text-zinc-400 hover:text-white disabled:opacity-30 flex items-center gap-1.5"
          >
            <RotateCcw className="h-3.5 w-3.5" /> Revert to original
          </button>
          <div className="flex items-center gap-2">
            <button onClick={onClose} className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white">Cancel</button>
            <button
              onClick={apply}
              disabled={busy || !hasChange}
              className="text-xs px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
            >
              {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />} Apply edit
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
