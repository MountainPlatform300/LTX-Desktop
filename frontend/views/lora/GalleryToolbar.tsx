import { useEffect, useRef, useState } from 'react'
import { Check, CheckSquare, ChevronDown, Filter, Loader2, Plus, Search, SlidersHorizontal, X } from 'lucide-react'
import type { Facet, FacetTone } from '../studio/studio-facets'
import type { LifecycleAction } from './lifecycle'

export type Density = 'large' | 'medium' | 'small'
/** Grid of thumbnails vs. a compact scannable list of rows. */
export type GalleryLayout = 'grid' | 'list'
export type SortKey = 'added' | 'duration' | 'attention' | 'pairs'
/** How to render IC-LoRA sets in the gallery. */
export type PairView = 'combined' | 'sideBySide' | 'flat'
/** Which clips to show with respect to pairing. */
export type PairFilter = 'all' | 'pairsOnly' | 'looseOnly' | 'incomplete'

const SORT_LABELS: Record<SortKey, string> = {
  added: 'Order added',
  duration: 'Longest first',
  attention: 'Needs attention',
  pairs: 'Group examples together',
}

const VIEW_HINTS: Record<PairView, string> = {
  combined: 'Each example as one card: input(s) → output',
  sideBySide: 'Each example as a compact list row',
  flat: 'Every clip separately, tagged by example',
}
const PAIR_VIEW_LABELS: Record<PairView, string> = {
  combined: 'Combined',
  sideBySide: 'Rows',
  flat: 'Flat',
}

const PAIR_FILTER_LABELS: Record<PairFilter, string> = {
  all: 'All clips',
  pairsOnly: 'Examples only',
  looseOnly: 'Ungrouped only',
  incomplete: 'Incomplete examples',
}

const FACET_TONE: Record<FacetTone, string> = {
  neutral: 'text-zinc-300',
  warn: 'text-amber-400',
  error: 'text-red-400',
}

const DENSITY_OPTIONS: Density[] = ['large', 'medium', 'small']
const DENSITY_LABELS: Record<Density, string> = {
  large: 'Large',
  medium: 'Medium',
  small: 'Small',
}

/** Lightweight click-outside + Escape popover anchored to a trigger button. */
function Popover({
  label,
  icon,
  badge,
  highlight,
  align = 'right',
  children,
}: {
  label: string
  icon: React.ReactNode
  badge?: number
  highlight?: boolean
  align?: 'left' | 'right'
  children: (close: () => void) => React.ReactNode
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])
  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        className={`text-[11px] px-2.5 py-1.5 rounded-md border flex items-center gap-1.5 transition-colors ${
          highlight || open
            ? 'border-blue-500/40 bg-blue-500/10 text-white'
            : 'border-zinc-800 text-zinc-300 hover:text-white hover:border-zinc-700'
        }`}
      >
        {icon}
        {label}
        {badge ? (
          <span className="ml-0.5 text-[10px] px-1 rounded-full bg-blue-500/30 text-blue-100">{badge}</span>
        ) : null}
        <ChevronDown className={`h-3 w-3 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div
          className={`absolute z-30 mt-1.5 min-w-[15rem] rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl shadow-black/40 p-2 ${
            align === 'right' ? 'right-0' : 'left-0'
          }`}
        >
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  )
}

function Segmented<T extends string>({
  options,
  value,
  onChange,
  labels,
  hints,
}: {
  options: readonly T[]
  value: T
  onChange: (v: T) => void
  labels: Record<T, string>
  hints?: Record<T, string>
}) {
  return (
    <div className="flex items-center rounded-md border border-zinc-800 overflow-hidden">
      {options.map((o) => (
        <button
          key={o}
          onClick={() => onChange(o)}
          title={hints?.[o]}
          className={`flex-1 px-2 py-1 text-[11px] ${
            value === o ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          {labels[o]}
        </button>
      ))}
    </div>
  )
}

function MenuSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="px-1 py-1.5">
      <p className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">{title}</p>
      {children}
    </div>
  )
}

export function GalleryToolbar({
  search,
  onSearchChange,
  facets,
  activeFacets,
  onFacetClick,
  onClearFacets,
  density,
  onDensityChange,
  layout,
  onLayoutChange,
  sort,
  onSortChange,
  onAddClips,
  pairCount,
  pairView,
  onPairViewChange,
  pairFilter,
  onPairFilterChange,
  totalCount,
  visibleCount,
  selectedCount,
  onSelectAll,
  onClearSelection,
  readiness,
  primary,
  primaryBusy,
  primaryDisabled,
  onPrimary,
}: {
  search: string
  onSearchChange: (value: string) => void
  facets: Facet[]
  activeFacets: string[]
  onFacetClick: (id: string) => void
  onClearFacets: () => void
  density: Density
  onDensityChange: (d: Density) => void
  layout: GalleryLayout
  onLayoutChange: (l: GalleryLayout) => void
  sort: SortKey
  onSortChange: (s: SortKey) => void
  /** When provided (editable collections), shows an "Add clips" button. */
  onAddClips?: () => void
  pairCount: number
  pairView: PairView
  onPairViewChange: (v: PairView) => void
  pairFilter: PairFilter
  onPairFilterChange: (f: PairFilter) => void
  totalCount: number
  visibleCount: number
  selectedCount: number
  onSelectAll: () => void
  onClearSelection: () => void
  readiness: { score: number; tone: 'ready' | 'warn' | 'error'; label: string; pairsLabel?: string | null } | null
  primary: LifecycleAction | null
  primaryBusy: boolean
  primaryDisabled: boolean
  onPrimary: () => void
}) {
  const READINESS_DOT: Record<'ready' | 'warn' | 'error', string> = {
    ready: 'bg-emerald-400',
    warn: 'bg-amber-400',
    error: 'bg-red-400',
  }
  const activeSet = new Set(activeFacets)
  const activeFacetDefs = facets.filter((f) => activeSet.has(f.id))
  return (
    <div className="border-b border-zinc-800">
      <div className="flex items-center gap-2 px-4 py-2.5">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
          <input
            data-tour="captions"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search captions..."
            className="w-full pl-8 pr-3 py-1.5 rounded-md bg-zinc-900 border border-zinc-800 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600"
          />
        </div>

        <span className="text-[11px] text-zinc-500 whitespace-nowrap">
          {visibleCount === totalCount ? `${totalCount}` : `${visibleCount}/${totalCount}`} clip
          {totalCount === 1 ? '' : 's'}
        </span>

        {onAddClips && (
          <button
            data-tour="import"
            onClick={onAddClips}
            title="Import videos or images into this collection"
            className="text-[11px] px-2.5 py-1.5 rounded-md border border-zinc-700 text-zinc-200 hover:text-white hover:border-zinc-600 flex items-center gap-1.5"
          >
            <Plus className="h-3.5 w-3.5" /> Import
          </button>
        )}

        <div className="flex-1" />

        <button
          onClick={selectedCount > 0 ? onClearSelection : onSelectAll}
          className="text-[11px] px-2.5 py-1.5 rounded-md bg-zinc-800 text-zinc-300 hover:bg-zinc-700 flex items-center gap-1.5"
        >
          <CheckSquare className="h-3.5 w-3.5" />
          {selectedCount > 0 ? `Clear (${selectedCount})` : 'Select all'}
        </button>

        {/* Filter popover — multi-select smart-facet tags. */}
        {facets.length > 0 && (
          <Popover
            label="Filter"
            icon={<Filter className="h-3.5 w-3.5" />}
            badge={activeFacetDefs.length}
            highlight={activeFacetDefs.length > 0}
          >
            {(close) => (
              <div>
                <div className="flex items-center justify-between px-1 pb-1.5 mb-1 border-b border-zinc-800">
                  <span className="text-[10px] uppercase tracking-wide text-zinc-500">Tags</span>
                  {activeFacetDefs.length > 0 && (
                    <button
                      onClick={onClearFacets}
                      className="text-[10px] text-zinc-400 hover:text-zinc-200 flex items-center gap-1"
                    >
                      <X className="h-3 w-3" /> Clear all
                    </button>
                  )}
                </div>
                <div className="px-0.5 max-h-64 overflow-y-auto">
                  {facets.map((facet) => {
                    const active = activeSet.has(facet.id)
                    return (
                      <button
                        key={facet.id}
                        onClick={() => onFacetClick(facet.id)}
                        className={`w-full text-left text-[11px] px-2 py-1 rounded-md flex items-center gap-2 ${
                          active ? 'bg-blue-500/15 text-white' : `hover:bg-zinc-800 ${FACET_TONE[facet.tone]}`
                        }`}
                      >
                        <span
                          className={`h-3.5 w-3.5 shrink-0 rounded border flex items-center justify-center ${
                            active ? 'bg-blue-500 border-blue-500' : 'border-zinc-600'
                          }`}
                        >
                          {active && <Check className="h-2.5 w-2.5 text-white" />}
                        </span>
                        <span className="flex-1 truncate">{facet.label}</span>
                        <span className="text-[10px] text-zinc-500">{facet.ids.length}</span>
                      </button>
                    )
                  })}
                </div>
                <div className="px-1 pt-1.5 mt-1 border-t border-zinc-800">
                  <button
                    onClick={() => {
                      onSelectAll()
                      close()
                    }}
                    className="w-full text-[11px] px-2 py-1 rounded-md bg-zinc-800 text-zinc-200 hover:bg-zinc-700 flex items-center justify-center gap-1.5"
                  >
                    <CheckSquare className="h-3.5 w-3.5" /> Select all shown ({visibleCount})
                  </button>
                </div>
              </div>
            )}
          </Popover>
        )}

        {/* View popover — sort, example grouping, layout & thumbnail size. */}
        <Popover label="View" icon={<SlidersHorizontal className="h-3.5 w-3.5" />}>
          {() => (
            <div className="divide-y divide-zinc-800">
              <MenuSection title="Sort by">
                <select
                  value={sort}
                  onChange={(e) => onSortChange(e.target.value as SortKey)}
                  className="w-full text-[11px] px-2 py-1.5 rounded-md bg-zinc-950 border border-zinc-800 text-zinc-300 focus:outline-none focus:border-zinc-600"
                >
                  {(Object.keys(SORT_LABELS) as SortKey[]).map((k) => (
                    <option key={k} value={k}>
                      {SORT_LABELS[k]}
                    </option>
                  ))}
                </select>
              </MenuSection>

              {pairCount > 0 && (
                <MenuSection title="Show">
                  <select
                    value={pairFilter}
                    onChange={(e) => onPairFilterChange(e.target.value as PairFilter)}
                    className="w-full text-[11px] px-2 py-1.5 rounded-md bg-zinc-950 border border-zinc-800 text-zinc-300 focus:outline-none focus:border-zinc-600"
                  >
                    {(Object.keys(PAIR_FILTER_LABELS) as PairFilter[]).map((k) => (
                      <option key={k} value={k}>
                        {PAIR_FILTER_LABELS[k]}
                      </option>
                    ))}
                  </select>
                </MenuSection>
              )}

              {layout === 'grid' && pairCount > 0 && (
                <MenuSection title="Examples as">
                  <Segmented
                    options={Object.keys(PAIR_VIEW_LABELS) as PairView[]}
                    value={pairView}
                    onChange={onPairViewChange}
                    labels={PAIR_VIEW_LABELS}
                    hints={VIEW_HINTS}
                  />
                </MenuSection>
              )}

              <MenuSection title="Layout">
                <Segmented
                  options={['grid', 'list'] as GalleryLayout[]}
                  value={layout}
                  onChange={onLayoutChange}
                  labels={{ grid: 'Grid', list: 'List' }}
                />
              </MenuSection>

              {layout === 'grid' && (
                <MenuSection title="Thumbnail size">
                  <Segmented
                    options={DENSITY_OPTIONS}
                    value={density}
                    onChange={onDensityChange}
                    labels={DENSITY_LABELS}
                  />
                </MenuSection>
              )}
            </div>
          )}
        </Popover>

        {readiness && (
          <span
            data-tour="readiness"
            className="text-[11px] px-2 py-1 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-300 flex items-center gap-1.5 whitespace-nowrap"
            title={readiness.label}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${READINESS_DOT[readiness.tone]}`} />
            {readiness.score}%
            {readiness.pairsLabel && <span className="text-zinc-500">· {readiness.pairsLabel}</span>}
          </span>
        )}

        {primary && (
          <button
            data-tour="train"
            onClick={onPrimary}
            disabled={primaryDisabled || primaryBusy}
            className="text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 whitespace-nowrap"
          >
            {primaryBusy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {primary.label}
          </button>
        )}
      </div>

      {/* Active tags are summarized inline as removable chips so they stay
          discoverable while the picker is collapsed. */}
      {activeFacetDefs.length > 0 && (
        <div className="flex items-center gap-1.5 px-4 pb-2 flex-wrap">
          {activeFacetDefs.map((facet) => (
            <button
              key={facet.id}
              onClick={() => onFacetClick(facet.id)}
              title="Remove tag"
              className="text-[11px] px-2 py-0.5 rounded-full border border-blue-500/40 bg-blue-500/15 text-white flex items-center gap-1"
            >
              {facet.label}
              <X className="h-3 w-3" />
            </button>
          ))}
          {activeFacetDefs.length > 1 && (
            <button
              onClick={onClearFacets}
              className="text-[11px] px-2 py-0.5 rounded-full text-zinc-400 hover:text-zinc-200"
            >
              Clear all
            </button>
          )}
        </div>
      )}
    </div>
  )
}
