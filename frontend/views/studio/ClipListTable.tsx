import { useEffect, useState } from 'react'
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  ChevronsUpDown,
  Image as ImageIcon,
  Link2,
  Maximize2,
  Volume2,
} from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import { clipWarnings, formatDuration } from '../../lib/lora-quality'
import { pairReadiness, type PairGroup } from '../../lib/lora-pairs'
import type { PairBadge, SelectIntent } from './ClipCard'
import type { StudioClip } from './studio-store'

export type ListColumnKey = 'caption' | 'size' | 'duration' | 'fps' | 'status'
export interface ListSort {
  key: ListColumnKey | null
  dir: 'asc' | 'desc'
}

/**
 * Shared column template so the header and every row line up regardless of
 * grouping. Columns: select · thumbnail · caption · size · length · fps ·
 * status · open.
 */
const GRID =
  'grid grid-cols-[1.5rem_5rem_minmax(8rem,1fr)_6rem_4.5rem_4rem_5.5rem_1.75rem] items-center gap-2'

type Readiness = 'ready' | 'warn' | 'error'
const STATUS_RANK: Record<Readiness, number> = { error: 0, warn: 1, ready: 2 }
const STATUS_LABEL: Record<Readiness, string> = { error: 'Needs fix', warn: 'Review', ready: 'Ready' }
const STATUS_DOT: Record<Readiness, string> = {
  ready: 'bg-emerald-400',
  warn: 'bg-amber-400',
  error: 'bg-red-400',
}

const PAIR_BADGE_BG = {
  ready: 'bg-emerald-500/85',
  warn: 'bg-amber-500/85',
  error: 'bg-red-500/85',
} as const

const TONE_ACCENT = {
  ready: 'border-emerald-500/50',
  warn: 'border-amber-500/50',
  error: 'border-red-500/50',
} as const

function readiness(clip: StudioClip): Readiness {
  const warnings = clipWarnings({ caption: clip.caption, probe: clip.probe })
  if (warnings.some((w) => w.level === 'error')) return 'error'
  if (warnings.length > 0 || !clip.caption.trim()) return 'warn'
  return 'ready'
}

function durationSeconds(clip: StudioClip): number {
  return clip.probe?.durationSeconds ?? clip.durationSeconds ?? 0
}

function sortValue(clip: StudioClip, key: ListColumnKey): number | string {
  switch (key) {
    case 'caption':
      return clip.caption.trim().toLowerCase()
    case 'size':
      return clip.probe ? clip.probe.width * clip.probe.height : 0
    case 'duration':
      return durationSeconds(clip)
    case 'fps':
      return clip.probe?.fps ?? 0
    case 'status':
      return STATUS_RANK[readiness(clip)]
  }
}

function compareBy(a: StudioClip, b: StudioClip, sort: ListSort): number {
  if (!sort.key) return 0
  const va = sortValue(a, sort.key)
  const vb = sortValue(b, sort.key)
  const c = typeof va === 'string' ? va.localeCompare(vb as string) : (va as number) - (vb as number)
  return sort.dir === 'asc' ? c : -c
}

function sortClips(clips: StudioClip[], sort: ListSort): StudioClip[] {
  return sort.key ? [...clips].sort((a, b) => compareBy(a, b, sort)) : clips
}

function groupRep(group: PairGroup): StudioClip | null {
  return group.targets[0] ?? group.controls[0] ?? null
}

function HeaderCell({
  label,
  colKey,
  sort,
  onSort,
  align = 'left',
}: {
  label: string
  colKey: ListColumnKey
  sort: ListSort
  onSort: (key: ListColumnKey) => void
  align?: 'left' | 'center'
}) {
  const active = sort.key === colKey
  return (
    <button
      onClick={() => onSort(colKey)}
      className={`flex items-center gap-1 text-[10px] uppercase tracking-wide transition-colors ${
        align === 'center' ? 'justify-center' : ''
      } ${active ? 'text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'}`}
    >
      {label}
      {active ? (
        sort.dir === 'asc' ? (
          <ArrowUp className="h-3 w-3" />
        ) : (
          <ArrowDown className="h-3 w-3" />
        )
      ) : (
        <ChevronsUpDown className="h-3 w-3 opacity-40" />
      )}
    </button>
  )
}

function Row({
  clip,
  selected,
  editable,
  pairBadge,
  onSelect,
  onOpen,
  onContextMenu,
  onCaptionChange,
  accent,
}: {
  clip: StudioClip
  selected: boolean
  editable: boolean
  pairBadge?: PairBadge | null
  onSelect: (id: string, intent: SelectIntent) => void
  onOpen?: (id: string) => void
  onContextMenu?: (id: string, x: number, y: number) => void
  onCaptionChange: (id: string, caption: string) => void
  /** Tailwind border-color class for the example accent stripe, or null. */
  accent: string | null
}) {
  const [draft, setDraft] = useState(clip.caption)
  useEffect(() => setDraft(clip.caption), [clip.caption])

  const isImage = clip.kind === 'image'
  const posterUrl = clip.posterPath ? pathToFileUrl(clip.posterPath) : null
  const state = readiness(clip)
  const warnings = clipWarnings({ caption: clip.caption, probe: clip.probe })
  const duration = clip.probe || clip.durationSeconds != null ? formatDuration(durationSeconds(clip)) : null

  const commit = () => {
    if (draft !== clip.caption) onCaptionChange(clip.id, draft)
  }

  return (
    <div
      className={`group ${GRID} h-14 px-2 border-l-2 rounded-r-md transition-colors cursor-pointer select-none ${
        accent ?? 'border-transparent'
      } ${selected ? 'bg-blue-500/10' : 'hover:bg-zinc-800/50'}`}
      onClick={(e) => onSelect(clip.id, { additive: e.metaKey || e.ctrlKey, range: e.shiftKey })}
      onDoubleClick={() => onOpen?.(clip.id)}
      onContextMenu={(e) => {
        if (!onContextMenu) return
        e.preventDefault()
        e.stopPropagation()
        onContextMenu(clip.id, e.clientX, e.clientY)
      }}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          onSelect(clip.id, { additive: true, range: false })
        }}
        aria-label={selected ? 'Deselect clip' : 'Select clip'}
        aria-pressed={selected}
        className={`h-4 w-4 shrink-0 rounded border flex items-center justify-center ${
          selected ? 'bg-blue-500 border-blue-500' : 'border-zinc-600 hover:border-zinc-400'
        }`}
      >
        {selected && <span className="h-2 w-2 rounded-sm bg-white" />}
      </button>

      <div
        className="h-10 w-[4.5rem] shrink-0 rounded bg-zinc-900 bg-cover bg-center relative overflow-hidden"
        style={posterUrl ? { backgroundImage: `url("${posterUrl}")` } : undefined}
      >
        {isImage && (
          <span className="absolute top-0.5 left-0.5 text-sky-300" title="Still image">
            <ImageIcon className="h-3 w-3" />
          </span>
        )}
        {pairBadge && (
          <span
            className={`absolute bottom-0 left-0 right-0 text-[9px] leading-tight text-white text-center ${PAIR_BADGE_BG[pairBadge.tone]}`}
            title={`Example ${pairBadge.index} — ${pairBadge.role === 'target' ? 'target' : 'reference'}`}
          >
            {pairBadge.role === 'target' ? 'target' : 'ref'}
          </span>
        )}
      </div>

      <div className="min-w-0" onClick={(e) => e.stopPropagation()}>
        {editable ? (
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur()
              if (e.key === 'Escape') {
                setDraft(clip.caption)
                e.currentTarget.blur()
              }
            }}
            placeholder="Describe this clip…"
            spellCheck={false}
            className="w-full bg-transparent text-xs text-zinc-200 placeholder:text-zinc-600 px-2 py-1.5 rounded-md border border-transparent hover:border-zinc-700 focus:border-blue-500/60 focus:bg-zinc-950/60 focus:outline-none"
          />
        ) : (
          <p className="text-xs text-zinc-300 truncate px-2">
            {clip.caption || <span className="text-zinc-600 italic">No caption</span>}
          </p>
        )}
      </div>

      <div className="text-[10px] text-zinc-400 text-center tabular-nums">
        {clip.probe ? `${clip.probe.width}×${clip.probe.height}` : '—'}
      </div>
      <div className="text-[10px] text-zinc-400 text-center tabular-nums">
        {!isImage && duration ? duration : '—'}
      </div>
      <div className="text-[10px] text-zinc-400 text-center tabular-nums flex items-center justify-center gap-1">
        {!isImage && clip.probe && clip.probe.fps > 0 ? `${Math.round(clip.probe.fps)}` : '—'}
        {clip.probe?.hasAudio && <Volume2 className="h-3 w-3 text-zinc-500" />}
      </div>
      <div className="flex items-center justify-center">
        {warnings.length > 0 ? (
          <span
            className={`flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] ${
              state === 'error' ? 'bg-red-500/15 text-red-300' : 'bg-amber-500/15 text-amber-300'
            }`}
            title={warnings.map((w) => w.text).join('\n')}
          >
            <AlertTriangle className="h-3 w-3" /> {warnings.length}
          </span>
        ) : (
          <span className="flex items-center gap-1 text-[10px] text-zinc-500">
            <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[state]}`} />
            {STATUS_LABEL[state]}
          </span>
        )}
      </div>

      <div className="flex items-center justify-center">
        {onOpen && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              onOpen(clip.id)
            }}
            title="Open"
            className="h-7 w-7 rounded-md text-zinc-500 hover:text-white hover:bg-zinc-800 hidden group-hover:flex items-center justify-center"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  )
}

/**
 * The "List" gallery layout: a sortable, column-aligned table. Clicking a
 * column header sorts the whole table by that field; for IC-LoRA datasets the
 * reference/target members of each example stay grouped under a labeled block
 * (sorting orders rows within a group and orders the groups by their target).
 */
export function ClipListTable({
  groups,
  loose,
  grouped,
  pairBadges,
  selectedIds,
  editable,
  sort,
  onSortChange,
  onSelect,
  onSelectGroup,
  onOpen,
  onContextMenu,
  onCaptionChange,
  onSelectAll,
  onClearSelection,
}: {
  groups: PairGroup[]
  loose: StudioClip[]
  grouped: boolean
  pairBadges: Map<string, PairBadge>
  selectedIds: Set<string>
  editable: boolean
  sort: ListSort
  onSortChange: (key: ListColumnKey) => void
  onSelect: (id: string, intent: SelectIntent) => void
  onSelectGroup: (ids: string[], intent: SelectIntent) => void
  onOpen?: (id: string) => void
  onContextMenu?: (id: string, x: number, y: number) => void
  onCaptionChange: (id: string, caption: string) => void
  onSelectAll: () => void
  onClearSelection: () => void
}) {
  const sortedLoose = sortClips(loose, sort)
  const orderedGroups = sort.key
    ? [...groups].sort((a, b) => {
        const ra = groupRep(a)
        const rb = groupRep(b)
        if (!ra || !rb) return 0
        return compareBy(ra, rb, sort)
      })
    : groups

  const allIds = [
    ...groups.flatMap((g) => [...g.controls, ...g.targets].map((c) => c.id)),
    ...loose.map((c) => c.id),
  ]
  const allSelected = allIds.length > 0 && allIds.every((id) => selectedIds.has(id))

  return (
    <div className="text-xs">
      {/* Sticky header — covers the main's top padding so rows scroll under it. */}
      <div
        className={`sticky -top-4 z-10 -mx-4 -mt-4 px-4 pt-4 pb-2 bg-zinc-950/95 backdrop-blur border-b border-zinc-800 ${GRID}`}
      >
        <button
          type="button"
          onClick={allSelected ? onClearSelection : onSelectAll}
          aria-label={allSelected ? 'Deselect all' : 'Select all'}
          className={`h-4 w-4 shrink-0 rounded border flex items-center justify-center ${
            allSelected ? 'bg-blue-500 border-blue-500' : 'border-zinc-600 hover:border-zinc-400'
          }`}
        >
          {allSelected && <span className="h-2 w-2 rounded-sm bg-white" />}
        </button>
        <span className="text-[10px] uppercase tracking-wide text-zinc-600 pl-1">Clip</span>
        <HeaderCell label="Caption" colKey="caption" sort={sort} onSort={onSortChange} />
        <HeaderCell label="Size" colKey="size" sort={sort} onSort={onSortChange} align="center" />
        <HeaderCell label="Length" colKey="duration" sort={sort} onSort={onSortChange} align="center" />
        <HeaderCell label="FPS" colKey="fps" sort={sort} onSort={onSortChange} align="center" />
        <HeaderCell label="Status" colKey="status" sort={sort} onSort={onSortChange} align="center" />
        <span />
      </div>

      <div className="flex flex-col gap-1 pt-2">
        {grouped &&
          orderedGroups.map((group, i) => {
            const tone = pairReadiness(group).tone
            const accent = TONE_ACCENT[tone]
            const ids = [...group.controls, ...group.targets].map((c) => c.id)
            const refs = sortClips(group.controls, sort)
            const tgts = sortClips(group.targets, sort)
            const groupSelected = ids.length > 0 && ids.every((id) => selectedIds.has(id))
            return (
              <div key={`set:${group.id}`} className="rounded-md">
                <div
                  className={`flex items-center gap-2 h-7 px-2 border-l-2 ${accent} bg-zinc-900/40 rounded-r-md cursor-pointer select-none`}
                  onMouseDown={(e) => { if (e.shiftKey) e.preventDefault() }}
                  onClick={(e) =>
                    onSelectGroup(ids, { additive: e.metaKey || e.ctrlKey, range: e.shiftKey })
                  }
                >
                  <span
                    className={`h-3.5 w-3.5 shrink-0 rounded border flex items-center justify-center ${
                      groupSelected ? 'bg-blue-500 border-blue-500' : 'border-zinc-600'
                    }`}
                  >
                    {groupSelected && <span className="h-1.5 w-1.5 rounded-sm bg-white" />}
                  </span>
                  <Link2 className="h-3 w-3 text-blue-300" />
                  <span className="text-[11px] font-medium text-zinc-200">Example {i + 1}</span>
                  <span className="text-[10px] text-zinc-500">
                    {group.controls.length} {group.controls.length === 1 ? 'reference' : 'references'} →{' '}
                    {group.targets.length} target{group.targets.length === 1 ? '' : 's'}
                  </span>
                  <span className={`ml-auto h-1.5 w-1.5 rounded-full ${STATUS_DOT[tone]}`} title={STATUS_LABEL[tone]} />
                </div>
                {[...refs, ...tgts].map((clip) => (
                  <Row
                    key={clip.id}
                    clip={clip}
                    selected={selectedIds.has(clip.id)}
                    editable={editable}
                    pairBadge={pairBadges.get(clip.id) ?? null}
                    onSelect={onSelect}
                    onOpen={onOpen}
                    onContextMenu={onContextMenu}
                    onCaptionChange={onCaptionChange}
                    accent={accent}
                  />
                ))}
              </div>
            )
          })}

        {sortedLoose.length > 0 && (
          <>
            {grouped && groups.length > 0 && (
              <p className="text-[10px] uppercase tracking-wide text-zinc-600 px-2 pt-2 pb-0.5">Ungrouped clips</p>
            )}
            {sortedLoose.map((clip) => (
              <Row
                key={clip.id}
                clip={clip}
                selected={selectedIds.has(clip.id)}
                editable={editable}
                pairBadge={pairBadges.get(clip.id) ?? null}
                onSelect={onSelect}
                onOpen={onOpen}
                onContextMenu={onContextMenu}
                onCaptionChange={onCaptionChange}
                accent={null}
              />
            ))}
          </>
        )}
      </div>
    </div>
  )
}
