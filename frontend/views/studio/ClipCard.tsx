import { useRef, useState } from 'react'
import { AlertTriangle, Check, FlaskConical, Image as ImageIcon, Link2, Loader2, Maximize2, MessageSquare, ThumbsDown, ThumbsUp, Volume2 } from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import { clipWarnings, formatDuration } from '../../lib/lora-quality'
import type { ReadinessTone } from '../../lib/lora-pairs'
import type { ClipTriage, StudioClip } from './studio-store'

/** Inline pair marker shown in the "flat" gallery layout, where paired clips
 *  render as normal cards but stay visually linked by a numbered, color-coded
 *  badge instead of being grouped into one PairCard. */
export interface PairBadge {
  index: number
  tone: ReadinessTone
  role: 'ref' | 'target'
}

const PAIR_BADGE_BG: Record<ReadinessTone, string> = {
  ready: 'bg-emerald-500/85',
  warn: 'bg-amber-500/85',
  error: 'bg-red-500/85',
}

type Readiness = 'ready' | 'warn' | 'error'

function readiness(clip: StudioClip): Readiness {
  const warnings = clipWarnings({ caption: clip.caption, probe: clip.probe })
  if (warnings.some((w) => w.level === 'error')) return 'error'
  if (warnings.length > 0 || !clip.caption.trim()) return 'warn'
  return 'ready'
}

// Borderless Gen-Space-style tiles: a clean card by default, with a subtle ring
// drawn only by exception (selection, triage, or a quality problem) so the grid
// stays calm and good clips read as "fine" without a permanent colored frame.
const PROBLEM_RING: Record<Readiness, string> = {
  ready: '',
  warn: 'ring-1 ring-amber-500/40',
  error: 'ring-1 ring-red-500/55',
}

/**
 * A single gallery card. The headline interaction is hover-scrub: moving the
 * cursor across the card sweeps through a pre-rendered filmstrip sprite, so
 * the whole clip is legible without playing it. Falls back to the poster
 * frame (or a loading shimmer) until the sprite job finishes. A status ring
 * and badges surface training readiness at a glance.
 */
/** How a click should mutate the selection.
 *  - `additive`: toggle this clip in/out of the current set (checkbox or ⌘/Ctrl)
 *  - `range`: extend from the anchor to this clip (Shift) */
export interface SelectIntent {
  additive: boolean
  range: boolean
}

export function ClipCard({
  clip,
  selected,
  onSelect,
  onOpen,
  onContextMenu,
  pairBadge,
  highlighted,
  onHoverSet,
  onTriage,
}: {
  clip: StudioClip
  selected: boolean
  onSelect: (id: string, intent: SelectIntent) => void
  onOpen?: (id: string) => void
  onContextMenu?: (id: string, x: number, y: number) => void
  pairBadge?: PairBadge | null
  /** True when another member of this clip's set is hovered (flat layout). */
  highlighted?: boolean
  /** Report hover over a set member so siblings can highlight together. */
  onHoverSet?: (index: number | null) => void
  /** Toggle the clip's keep/reject curation flag (null = clear). Omit to hide
   *  the inline triage controls. */
  onTriage?: (id: string, triage: ClipTriage | null) => void
}) {
  const frameRef = useRef<HTMLDivElement | null>(null)
  const [tileIndex, setTileIndex] = useState(0)
  const [hovering, setHovering] = useState(false)

  const isImage = clip.kind === 'image'
  const tiles = clip.spriteTiles ?? 0
  // Stills have no filmstrip — never hover-scrub them.
  const hasSprite = !isImage && clip.spritePath != null && tiles > 0
  const posterUrl = clip.posterPath ? pathToFileUrl(clip.posterPath) : null
  const spriteUrl = clip.spritePath ? pathToFileUrl(clip.spritePath) : null

  const handleMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!hasSprite) return
    const el = frameRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
    setTileIndex(Math.min(tiles - 1, Math.floor(ratio * tiles)))
  }

  const showStrip = hovering && hasSprite
  const frameStyle: React.CSSProperties = showStrip
    ? {
        backgroundImage: `url("${spriteUrl}")`,
        backgroundSize: `${tiles * 100}% 100%`,
        backgroundPosition: `${tiles > 1 ? (tileIndex / (tiles - 1)) * 100 : 0}% 50%`,
      }
    : posterUrl
      ? { backgroundImage: `url("${posterUrl}")`, backgroundSize: 'cover', backgroundPosition: 'center' }
      : {}

  const duration = clip.probe ? formatDuration(clip.probe.durationSeconds) : clip.durationSeconds != null ? formatDuration(clip.durationSeconds) : null
  const resolution = clip.probe ? `${clip.probe.width}×${clip.probe.height}` : null
  const state = readiness(clip)
  const uncaptioned = !clip.caption.trim()
  const triage = clip.triage
  const ringClass = selected
    ? 'ring-2 ring-blue-500'
    : highlighted
      ? 'ring-2 ring-blue-400/60'
      : triage === 'keep'
        ? 'ring-2 ring-emerald-500/60'
        : triage === 'reject'
          ? 'ring-2 ring-red-500/45'
          : triage === 'holdout'
            ? 'ring-2 ring-amber-500/55'
            : PROBLEM_RING[state]

  return (
    // `select-none` + the Shift-aware mousedown guard stop the browser from
    // turning a Shift+click into a native text selection (which only highlights
    // captions); the click still fires, so range-select runs as intended.
    <div
      className={`group relative rounded-lg overflow-hidden bg-zinc-900 ring-inset transition-shadow cursor-pointer select-none ${ringClass}`}
      onMouseDown={(e) => { if (e.shiftKey) e.preventDefault() }}
      onClick={(e) => onSelect(clip.id, { additive: e.metaKey || e.ctrlKey, range: e.shiftKey })}
      onDoubleClick={() => onOpen?.(clip.id)}
      onMouseEnter={() => pairBadge && onHoverSet?.(pairBadge.index)}
      onMouseLeave={() => pairBadge && onHoverSet?.(null)}
      onContextMenu={(e) => {
        if (!onContextMenu) return
        e.preventDefault()
        e.stopPropagation()
        onContextMenu(clip.id, e.clientX, e.clientY)
      }}
    >
      <div
        ref={frameRef}
        className="aspect-video w-full bg-zinc-900 bg-no-repeat relative"
        style={frameStyle}
        onMouseEnter={() => setHovering(true)}
        onMouseLeave={() => setHovering(false)}
        onMouseMove={handleMove}
      >
        {!posterUrl && !hasSprite && (
          <div className="absolute inset-0 flex items-center justify-center text-zinc-600">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        )}

        {/* Rejected clips are dimmed so the eye skips them while curating. */}
        {triage === 'reject' && <div className="absolute inset-0 bg-zinc-950/55 pointer-events-none" />}

        {/* Selection checkbox — clicking it toggles this clip in/out of a
            multi-selection without disturbing the rest (additive), so the user
            can keep ticking boxes. Stops propagation so it never triggers the
            card's single-select. */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            onSelect(clip.id, { additive: true, range: false })
          }}
          aria-label={selected ? 'Deselect clip' : 'Select clip'}
          aria-pressed={selected}
          className={`absolute top-2 left-2 h-5 w-5 rounded-md flex items-center justify-center transition-opacity hover:scale-110 ${
            selected
              ? 'bg-blue-500 text-white opacity-100'
              : 'bg-black/50 text-white/80 opacity-0 group-hover:opacity-100 ring-1 ring-white/40'
          }`}
        >
          <Check className="h-3.5 w-3.5" />
        </button>

        {/* Top-right: keep/reject triage, expand-to-open on hover, status markers */}
        <div className="absolute top-2 right-2 flex gap-1 items-center">
          {onTriage && (
            <>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onTriage(clip.id, triage === 'keep' ? null : 'keep') }}
                title={triage === 'keep' ? 'Keeper — click to clear' : 'Mark as keeper'}
                aria-pressed={triage === 'keep'}
                className={`h-5 w-5 rounded-md items-center justify-center hover:scale-110 ${
                  triage === 'keep' ? 'bg-emerald-500 text-white flex' : 'bg-black/55 text-white/80 ring-1 ring-white/30 hidden group-hover:flex'
                }`}
              >
                <ThumbsUp className="h-3 w-3" />
              </button>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onTriage(clip.id, triage === 'reject' ? null : 'reject') }}
                title={triage === 'reject' ? 'Rejected — click to clear' : 'Mark as rejected'}
                aria-pressed={triage === 'reject'}
                className={`h-5 w-5 rounded-md items-center justify-center hover:scale-110 ${
                  triage === 'reject' ? 'bg-red-500 text-white flex' : 'bg-black/55 text-white/80 ring-1 ring-white/30 hidden group-hover:flex'
                }`}
              >
                <ThumbsDown className="h-3 w-3" />
              </button>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onTriage(clip.id, triage === 'holdout' ? null : 'holdout') }}
                title={triage === 'holdout' ? 'Holdout (validation only) — click to clear' : 'Mark as holdout (validation only)'}
                aria-pressed={triage === 'holdout'}
                className={`h-5 w-5 rounded-md items-center justify-center hover:scale-110 ${
                  triage === 'holdout' ? 'bg-amber-500 text-white flex' : 'bg-black/55 text-white/80 ring-1 ring-white/30 hidden group-hover:flex'
                }`}
              >
                <FlaskConical className="h-3 w-3" />
              </button>
            </>
          )}
          {onOpen && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onOpen(clip.id)
              }}
              title="Open"
              className="h-6 w-6 rounded-md bg-black/60 text-white items-center justify-center hidden group-hover:flex hover:bg-black/80"
            >
              <Maximize2 className="h-3 w-3" />
            </button>
          )}
          {state === 'error' && (
            <span className="h-5 w-5 rounded-md bg-red-500/80 text-white flex group-hover:hidden items-center justify-center" title="Quality error">
              <AlertTriangle className="h-3 w-3" />
            </span>
          )}
          {uncaptioned && (
            <span className="h-5 w-5 rounded-md bg-black/60 text-amber-300 flex group-hover:hidden items-center justify-center" title="No caption">
              <MessageSquare className="h-3 w-3" />
            </span>
          )}
        </div>

        {/* Scrub progress underline */}
        {showStrip && tiles > 1 && (
          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-black/40">
            <div className="h-full bg-blue-400" style={{ width: `${((tileIndex + 1) / tiles) * 100}%` }} />
          </div>
        )}

        {/* Paired (control→target) marker. In the flat layout a numbered,
            color-coded badge keeps a pair's members visually linked; otherwise
            a target carrying a referencePath shows a generic "Pair" tag. */}
        {pairBadge ? (
          <span
            className={`absolute bottom-2 left-2 text-[10px] px-1.5 py-0.5 rounded text-white flex items-center gap-1 ${PAIR_BADGE_BG[pairBadge.tone]}`}
            title={`Example ${pairBadge.index} — ${pairBadge.role === 'target' ? 'the output the model should produce' : 'an input the model conditions on'}`}
          >
            <Link2 className="h-2.5 w-2.5" /> Example {pairBadge.index} · {pairBadge.role === 'target' ? 'output' : 'input'}
          </span>
        ) : clip.referencePath ? (
          <span
            className="absolute bottom-2 left-2 text-[10px] px-1.5 py-0.5 rounded bg-blue-500/80 text-white flex items-center gap-1"
            title="Output clip — generated from an input"
          >
            <Link2 className="h-2.5 w-2.5" /> Example
          </span>
        ) : null}

        {/* Still-image marker */}
        {isImage && (
          <span
            className="absolute top-2 left-9 text-[10px] px-1.5 py-0.5 rounded bg-sky-500/80 text-white flex items-center gap-1"
            title="Still image — use Animate or Generate example to turn it into a clip"
          >
            <ImageIcon className="h-2.5 w-2.5" /> Still
          </span>
        )}

        {/* Badges */}
        <div className="absolute bottom-2 right-2 flex gap-1 items-center">
          {clip.probe?.hasAudio && (
            <span className="h-5 px-1 rounded bg-black/60 text-zinc-200 flex items-center" title="Has audio">
              <Volume2 className="h-3 w-3" />
            </span>
          )}
          {resolution && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/60 text-zinc-200">{resolution}</span>
          )}
          {duration && !isImage && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/60 text-zinc-200">{duration}</span>
          )}
        </div>
      </div>

      <div className="px-2 py-1.5 bg-zinc-900">
        <p className="text-[11px] text-zinc-400 truncate">
          {clip.caption || <span className="text-zinc-600 italic">No caption</span>}
        </p>
      </div>
    </div>
  )
}
