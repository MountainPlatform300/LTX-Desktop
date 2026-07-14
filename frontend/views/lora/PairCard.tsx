import { useRef, useState } from 'react'
import { ArrowRight, Check, Link2, Loader2, Maximize2 } from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import { formatDuration } from '../../lib/lora-quality'
import { groupMemberIds, pairReadiness, type PairGroup, type ReadinessTone } from '../../lib/lora-pairs'
import type { SelectIntent } from '../studio/ClipCard'
import type { StudioClip } from '../studio/studio-store'

export type PairLayout = 'combined' | 'sideBySide'

const DOT: Record<ReadinessTone, string> = {
  ready: 'bg-emerald-500',
  warn: 'bg-amber-500',
  error: 'bg-red-500',
}
// Calm, ClipCard-style framing: no permanent colored frame for ready sets; a
// subtle ring is drawn only when something needs attention.
const TONE_RING: Record<ReadinessTone, string> = {
  ready: '',
  warn: 'ring-1 ring-amber-500/40',
  error: 'ring-1 ring-red-500/55',
}
// Readiness label colors itself only when action is needed, so "Ready" stays
// quiet and warnings draw the eye.
const TONE_TEXT: Record<ReadinessTone, string> = {
  ready: 'text-zinc-400',
  warn: 'text-amber-300',
  error: 'text-red-300',
}
const READY_LABEL: Record<ReadinessTone, string> = {
  ready: 'Ready',
  warn: 'Needs caption',
  error: 'Incomplete',
}

/** Human label for the set, honest about input count (never says "pair"). */
export function setLabel(group: PairGroup): string {
  const n = group.referencePaths.length
  return n === 1 ? 'Example · 1 input' : `Example · ${n} inputs`
}

/** One reference/result thumbnail. Fills its parent's height (the media row
 *  sets a fixed height), so every frame in a set lines up regardless of how
 *  many references there are. Keeps ClipCard's hover-scrub behaviour. */
function PairThumb({
  clip,
  label,
  fill = false,
}: {
  clip: StudioClip | null
  label: string
  fill?: boolean
}) {
  const frameRef = useRef<HTMLDivElement | null>(null)
  const [tileIndex, setTileIndex] = useState(0)
  const [hovering, setHovering] = useState(false)

  // `fill` lets a thumb flex to share its row's width while keeping a true 16:9
  // box (width-driven height), so it scales cleanly with the card/grid size and
  // never stretches. Otherwise it keeps a fixed-height 16:9 box.
  const sizing = fill ? 'flex-1 min-w-0 aspect-video' : 'h-full aspect-video shrink-0'

  if (!clip) {
    return (
      <div className={`${sizing} rounded-md bg-zinc-950 border border-dashed border-zinc-700 flex flex-col items-center justify-center gap-1 text-zinc-600`}>
        <Link2 className="h-4 w-4" />
        <span className="text-[10px]">Missing</span>
      </div>
    )
  }

  const isImage = clip.kind === 'image'
  const tiles = clip.spriteTiles ?? 0
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

  const duration = clip.probe ? formatDuration(clip.probe.durationSeconds) : null

  return (
    <div
      ref={frameRef}
      className={`${sizing} rounded-md bg-zinc-950 bg-no-repeat relative overflow-hidden`}
      style={frameStyle}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
      onMouseMove={handleMove}
    >
      {!posterUrl && !hasSprite && (
        <div className="absolute inset-0 flex items-center justify-center text-zinc-600">
          <Loader2 className="h-4 w-4 animate-spin" />
        </div>
      )}
      <span className="absolute top-1 left-1 text-[8.5px] font-medium px-1.5 py-0.5 rounded-md bg-black/55 backdrop-blur-sm text-zinc-200 uppercase tracking-wide">
        {label}
      </span>
      {duration && !isImage && (
        <span className="absolute bottom-1 right-1 text-[8.5px] px-1.5 py-0.5 rounded-md bg-black/55 backdrop-blur-sm text-zinc-200">{duration}</span>
      )}
    </div>
  )
}

/** The reference (before) side of a set: every reference at equal height, with
 *  dashed placeholders for references whose clip is missing, capped with "+N". */
function ReferenceSlot({ group, max, fill = false }: { group: PairGroup; max: number; fill?: boolean }) {
  const single = group.referencePaths.length <= 1
  const shown = group.controls.slice(0, max)
  const missing = Math.max(0, group.referencePaths.length - group.controls.length)
  const missingShown = Math.min(missing, Math.max(0, max - shown.length))
  const overflow = group.referencePaths.length - shown.length - missingShown

  return (
    <div className={`flex items-center gap-1 min-w-0 ${fill ? 'flex-1' : 'h-full'}`}>
      {shown.map((c, i) => (
        <PairThumb key={c.id} clip={c} fill={fill} label={single ? 'Input' : `Input ${i + 1}`} />
      ))}
      {Array.from({ length: missingShown }).map((_, i) => (
        <PairThumb key={`missing-${i}`} clip={null} fill={fill} label="Input" />
      ))}
      {overflow > 0 && (
        <span className="shrink-0 self-stretch px-1.5 flex items-center text-[10px] text-zinc-400 bg-zinc-950 rounded-md">
          +{overflow}
        </span>
      )}
    </div>
  )
}

/**
 * A training Set (one result conditioned on one-or-more references) rendered as
 * a single card so the reference → result relationship is unmistakable, with a
 * readiness chip. Interactions mirror ClipCard but operate on the whole set:
 * the checkbox toggles every member, a plain click selects the set, and
 * open/right-click target the result clip.
 */
export function PairCard({
  group,
  selectedIds,
  onSelect,
  onOpen,
  onContextMenu,
  layout = 'combined',
}: {
  group: PairGroup
  selectedIds: Set<string>
  onSelect: (ids: string[], intent: SelectIntent) => void
  onOpen?: (id: string) => void
  onContextMenu?: (id: string, x: number, y: number) => void
  layout?: PairLayout
}) {
  const result = group.targets[0]
  const memberIds = groupMemberIds(group)
  const selected = memberIds.length > 0 && memberIds.every((id) => selectedIds.has(id))
  const { tone, reasons } = pairReadiness(group)
  const extraResults = group.targets.length - 1
  const refCount = group.referencePaths.length
  const countLabel = `${refCount} ${refCount === 1 ? 'input' : 'inputs'}${
    extraResults > 0 ? ` · +${extraResults} output${extraResults === 1 ? '' : 's'}` : ''
  }`

  const tag = (
    <span
      className="flex items-center gap-1.5 text-[11px] font-medium text-blue-200"
      title="Training example — the input(s) the model sees, and the output it should produce"
    >
      <Link2 className="h-3 w-3" />
      {setLabel(group)}
      {extraResults > 0 ? ` · +${extraResults} outputs` : ''}
    </span>
  )
  const readinessChip = (
    <span className={`flex items-center gap-1.5 text-[10px] ${TONE_TEXT[tone]}`} title={reasons.join(' · ')}>
      <span className={`h-1.5 w-1.5 rounded-full ${DOT[tone]}`} />
      {READY_LABEL[tone]}
    </span>
  )

  // A defined tile (always-on subtle ring) + a violet left "spine" brands every
  // example as one linked unit, clearly distinct from loose clips, without a
  // loud colored frame. Status is carried by the readiness chip, not the ring.
  const containerBase = `group relative rounded-xl overflow-hidden bg-zinc-900 transition-shadow cursor-pointer select-none before:content-[''] before:absolute before:inset-y-0 before:left-0 before:w-[3px] before:bg-blue-500/50 ${
    selected ? 'ring-2 ring-blue-500' : `ring-1 ring-zinc-800 hover:ring-zinc-700 ${TONE_RING[tone]}`
  }`
  const handleClick = (e: React.MouseEvent) =>
    onSelect(memberIds, { additive: e.metaKey || e.ctrlKey, range: e.shiftKey })
  // Block Shift+click from starting a native text selection so range-select
  // (handled on click) runs instead of just highlighting the example's labels.
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.shiftKey) e.preventDefault()
  }
  const handleContext = (e: React.MouseEvent) => {
    if (!onContextMenu) return
    e.preventDefault()
    e.stopPropagation()
    onContextMenu(result.id, e.clientX, e.clientY)
  }
  const checkbox = (extra: string) => (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation()
        onSelect(memberIds, { additive: true, range: false })
      }}
      aria-label={selected ? 'Deselect set' : 'Select set'}
      aria-pressed={selected}
      className={`h-5 w-5 rounded-md flex items-center justify-center transition-opacity hover:scale-110 ${extra} ${
        selected
          ? 'bg-blue-500 text-white opacity-100'
          : 'bg-black/50 text-white/80 opacity-0 group-hover:opacity-100 ring-1 ring-white/40'
      }`}
    >
      <Check className="h-3.5 w-3.5" />
    </button>
  )
  const openBtn = (extra: string) =>
    onOpen && (
      <button
        onClick={(e) => {
          e.stopPropagation()
          onOpen(result.id)
        }}
        title="Open output clip"
        className={`h-6 w-6 rounded-md bg-black/60 text-white items-center justify-center hidden group-hover:flex hover:bg-black/80 ${extra}`}
      >
        <Maximize2 className="h-3 w-3" />
      </button>
    )

  if (layout === 'sideBySide') {
    // Compact, fixed-height list row: scannable when there are many sets.
    return (
      <div
        className={`col-span-full flex items-center gap-3 px-3 ${containerBase}`}
        style={{ height: 88 }}
        onMouseDown={handleMouseDown}
        onClick={handleClick}
        onDoubleClick={() => onOpen?.(result.id)}
        onContextMenu={handleContext}
      >
        {checkbox('shrink-0')}
        <div className="flex items-center gap-2 h-16 shrink-0 rounded-lg bg-zinc-950/60 ring-1 ring-zinc-800/70 px-2">
        <ReferenceSlot group={group} max={3} />
        <ArrowRight className="h-4 w-4 shrink-0 text-blue-400/80" />
        <PairThumb clip={result} label="Output" />
        </div>
        <div className="flex-1 min-w-0 flex flex-col justify-center gap-1">
          <div className="flex items-center gap-2">
            {tag}
            {readinessChip}
          </div>
          <p className="text-[11px] text-zinc-400 truncate">
            {result.caption || <span className="text-zinc-600 italic">No caption</span>}
          </p>
        </div>
        {openBtn('shrink-0')}
      </div>
    )
  }

  // Combined card: a slim meta row, a fixed-height media row (references and
  // result share one height), and a caption footer — mirroring ClipCard's calm
  // framing so examples sit cleanly next to loose clips in the grid.
  return (
    <div
      className={`col-span-2 ${containerBase}`}
      onMouseDown={handleMouseDown}
      onClick={handleClick}
      onDoubleClick={() => onOpen?.(result.id)}
      onContextMenu={handleContext}
    >
      <div className="flex items-center gap-2 px-2.5 pt-2 pb-1">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            onSelect(memberIds, { additive: true, range: false })
          }}
          aria-label={selected ? 'Deselect example' : 'Select example'}
          aria-pressed={selected}
          className={`h-4 w-4 shrink-0 rounded flex items-center justify-center transition-colors ${
            selected
              ? 'bg-blue-500 text-white'
              : 'text-transparent ring-1 ring-zinc-700 group-hover:ring-zinc-500 hover:text-zinc-300'
          }`}
        >
          <Check className="h-3 w-3" />
        </button>
        <Link2 className="h-3 w-3 shrink-0 text-blue-400/80" />
        <span
          className="flex-1 min-w-0 truncate text-[11px] text-zinc-400"
          title="Training example — the input(s) the model sees, and the output it should produce"
        >
          {countLabel}
        </span>
        {readinessChip}
      </div>

      {/* Recessed track unifies input(s) → output as one connected strip. */}
      <div className="relative mx-2.5 mb-0.5 flex items-center gap-1.5 rounded-lg bg-zinc-950/60 ring-1 ring-zinc-800/70 px-2 py-2">
        <ReferenceSlot group={group} max={3} fill />
        <ArrowRight className="h-4 w-4 shrink-0 text-blue-400/80" />
        <PairThumb clip={result} label="Output" fill />
        {openBtn('absolute top-1.5 right-1.5 z-10')}
      </div>

      <div className="px-2.5 pt-1.5 pb-2">
        <p className="text-[11px] text-zinc-400 truncate">
          {result.caption || <span className="text-zinc-600 italic">No caption</span>}
        </p>
      </div>
    </div>
  )
}
