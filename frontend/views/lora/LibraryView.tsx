import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  Check,
  ExternalLink,
  FolderOpen,
  ImagePlus,
  LayoutGrid,
  Library as LibraryIcon,
  List as ListIcon,
  Loader2,
  Pencil,
  RotateCw,
  Sparkles,
  Trash2,
  Download,
  Wand2,
  X,
} from 'lucide-react'
import { ApiClient } from '../../lib/api-client'
import { logger } from '../../lib/logger'
import { useBackendMediaUrl } from '../../lib/backend-media'
import type { LoraInferenceRegistryState, LoraInferenceEntry } from '../../hooks/use-lora-inference-registry'
import { InfoHint } from '../../components/lora/trainingFormParts'
import { confirmAction } from '../../components/ui/confirm-dialog'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const VARIANT_LABEL: Record<LoraInferenceEntry['variant'], string> = {
  standard: 'Standard',
  union_control: 'Union Control',
  video_input_ic_lora: 'IC-LoRA',
}

const KIND_LABEL: Record<LoraInferenceEntry['kind'], string> = {
  official_union: 'Official',
  imported: 'Imported',
  user_trained: 'Trained',
}

function formatSize(bytes: number | null | undefined): string {
  if (bytes == null) return '—'
  if (bytes < 1024) return `${bytes} B`
  const mb = bytes / (1024 * 1024)
  if (mb < 1024) return `${mb.toFixed(1)} MB`
  return `${(mb / 1024).toFixed(2)} GB`
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

// A CivitAI-style "what does this LoRA do?" thumbnail. Renders the user-
// attached example (image or video) as a blob URL from the secure example-media
// route; falls back to a gradient placeholder when no example is attached.
// `rounded` / `className` let the gallery card, list row, and detail panel
// reuse the same chrome.
export function ExampleThumb({
  entry,
  className = '',
  rounded = 'rounded-md',
  controls = false,
  intrinsic = false,
}: {
  entry: LoraInferenceEntry
  className?: string
  rounded?: string
  controls?: boolean
  intrinsic?: boolean
}) {
  const mediaPath = entry.exampleMediaType ? ApiClient.loraExampleMediaPath(entry.id) : null
  const { url, error } = useBackendMediaUrl(mediaPath)
  const isVideo = entry.exampleMediaType === 'video'

  if (!entry.exampleMediaType || error) {
    return (
      <div
        className={`${rounded} bg-gradient-to-br from-zinc-700 to-zinc-800 flex items-center justify-center ${className}`}
      >
        <Wand2 className="h-5 w-5 text-zinc-500" />
      </div>
    )
  }
  if (!url) {
    return (
      <div
        className={`${rounded} bg-gradient-to-br from-zinc-700 to-zinc-800 flex items-center justify-center ${className}`}
      >
        <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />
      </div>
    )
  }
  if (isVideo) {
    // autoPlay + muted + loop plays silently inline (CivitAI-style thumbnail
    // motion, matching AssetCard). `controls` is passed for the detail-panel
    // preview so the user can scrub / unmute. preload="metadata" surfaces the
    // first frame even before playback starts.
    return (
      <video
        src={url}
        autoPlay
        muted
        loop
        playsInline
        preload="metadata"
        controls={controls}
        className={`${rounded} bg-black object-contain ${intrinsic ? 'block h-auto max-h-[26rem] w-full' : ''} ${className}`}
      />
    )
  }
  return (
    <img
      src={url}
      alt={entry.name}
      className={`${rounded} bg-black object-contain ${intrinsic ? 'block h-auto max-h-[26rem] w-full' : ''} ${className}`}
    />
  )
}

type KindFilter = 'all' | LoraInferenceEntry['kind']
type SortKey = 'name' | 'created' | 'size'
type ViewMode = 'gallery' | 'list'

const VIEW_MODES: { id: ViewMode; label: string; icon: typeof LayoutGrid }[] = [
  { id: 'gallery', label: 'Gallery', icon: LayoutGrid },
  { id: 'list', label: 'List', icon: ListIcon },
]

const VIEW_MODE_STORAGE_KEY = 'lora-library-view-mode'

function loadViewMode(): ViewMode {
  try {
    const v = localStorage.getItem(VIEW_MODE_STORAGE_KEY)
    return v === 'list' ? 'list' : 'gallery'
  } catch {
    return 'gallery'
  }
}

const FILTERS: { id: KindFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'imported', label: 'Imported' },
  { id: 'user_trained', label: 'Trained' },
  { id: 'official_union', label: 'Official' },
]

// ---------------------------------------------------------------------------
// LibraryView
// ---------------------------------------------------------------------------

export function LibraryView({
  registry,
  onTryInGenSpace,
  onOpenTrainingRun,
}: {
  registry: LoraInferenceRegistryState
  onTryInGenSpace: (loraId: string) => void
  onOpenTrainingRun: (jobId: string) => void
}) {
  const { entries, loading, error, refresh } = registry
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [filter, setFilter] = useState<KindFilter>('all')
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState<SortKey>('name')
  const [viewMode, setViewMode] = useState<ViewMode>(() => loadViewMode())
  const [importOpen, setImportOpen] = useState(false)

  const onViewMode = useCallback((m: ViewMode) => {
    setViewMode(m)
    try {
      localStorage.setItem(VIEW_MODE_STORAGE_KEY, m)
    } catch {
      /* ignore */
    }
  }, [])

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = entries.filter((e) => {
      if (filter !== 'all' && e.kind !== filter) return false
      if (!q) return true
      return (
        e.name.toLowerCase().includes(q) ||
        (e.triggerWord ?? '').toLowerCase().includes(q) ||
        (e.description ?? '').toLowerCase().includes(q)
      )
    })
    const sorted = [...filtered].sort((a, b) => {
      switch (sortBy) {
        case 'created':
          return (b.createdAt ?? '').localeCompare(a.createdAt ?? '')
        case 'size':
          return (b.fileSizeBytes ?? 0) - (a.fileSizeBytes ?? 0)
        default:
          return a.name.localeCompare(b.name)
      }
    })
    // Official first within any sort (it's the baseline adapter), then the rest.
    return [
      ...sorted.filter((e) => e.kind === 'official_union'),
      ...sorted.filter((e) => e.kind !== 'official_union'),
    ]
  }, [entries, filter, query, sortBy])

  const selected = useMemo(
    () => visible.find((e) => e.id === selectedId) ?? entries.find((e) => e.id === selectedId) ?? null,
    [visible, entries, selectedId],
  )

  // Clear selection if it disappears from the registry (e.g. after delete).
  useEffect(() => {
    if (selectedId && !entries.some((e) => e.id === selectedId)) {
      setSelectedId(null)
    }
  }, [entries, selectedId])

  return (
    <div className="flex-1 flex min-h-0">
      <div className="flex-1 flex flex-col min-h-0">
        <LibraryHeader
          count={visible.length}
          total={entries.length}
          query={query}
          onQuery={setQuery}
          filter={filter}
          onFilter={setFilter}
          sortBy={sortBy}
          onSort={setSortBy}
          viewMode={viewMode}
          onViewMode={onViewMode}
          onImport={() => setImportOpen(true)}
        />
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex items-center justify-center h-full text-zinc-500">
              <Loader2 className="h-5 w-5 animate-spin" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-full text-sm text-red-400">{error}</div>
          ) : visible.length === 0 ? (
            <EmptyState hasEntries={entries.length > 0} onImport={() => setImportOpen(true)} />
          ) : viewMode === 'list' ? (
            <div className="flex flex-col divide-y divide-zinc-800/80 border border-zinc-800 rounded-lg overflow-hidden">
              {visible.map((entry) => (
                <LoraListRow
                  key={entry.id}
                  entry={entry}
                  active={entry.id === selectedId}
                  onClick={() => setSelectedId(entry.id)}
                />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-3">
              {visible.map((entry) => (
                <LoraCard
                  key={entry.id}
                  entry={entry}
                  active={entry.id === selectedId}
                  onClick={() => setSelectedId(entry.id)}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {selected && (
        <DetailPanel
          key={selected.id}
          entry={selected}
          onClose={() => setSelectedId(null)}
          onChanged={refresh}
          onTryInGenSpace={onTryInGenSpace}
          onOpenTrainingRun={onOpenTrainingRun}
        />
      )}

      {importOpen && (
        <ImportLoraModal onClose={() => setImportOpen(false)} onImported={refresh} />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

function LibraryHeader({
  count,
  total,
  query,
  onQuery,
  filter,
  onFilter,
  sortBy,
  onSort,
  viewMode,
  onViewMode,
  onImport,
}: {
  count: number
  total: number
  query: string
  onQuery: (v: string) => void
  filter: KindFilter
  onFilter: (f: KindFilter) => void
  sortBy: SortKey
  onSort: (s: SortKey) => void
  viewMode: ViewMode
  onViewMode: (m: ViewMode) => void
  onImport: () => void
}) {
  return (
    <div className="px-4 py-3 border-b border-zinc-800 flex flex-col gap-2.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <LibraryIcon className="h-4 w-4 text-violet-300" />
          <h2 className="text-sm font-semibold text-white">LoRA Library</h2>
          <span className="text-[11px] text-zinc-500">
            {count} shown · {total} total
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-0.5 rounded-md border border-zinc-700 bg-zinc-800/60 p-0.5">
            {VIEW_MODES.map((m) => {
              const Icon = m.icon
              return (
                <button
                  key={m.id}
                  onClick={() => onViewMode(m.id)}
                  title={m.label}
                  className={`px-1.5 py-1 rounded text-[11px] transition-colors ${
                    viewMode === m.id ? 'bg-blue-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                </button>
              )
            })}
          </div>
          <button
            onClick={onImport}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            Import LoRA
          </button>
        </div>
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <input
          value={query}
          onChange={(e) => onQuery(e.target.value)}
          placeholder="Search by name, trigger, or description…"
          spellCheck={false}
          className="flex-1 min-w-[200px] px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
        />
        <div className="flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-800/60 p-0.5">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => onFilter(f.id)}
              className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                filter === f.id ? 'bg-blue-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <select
          value={sortBy}
          onChange={(e) => onSort(e.target.value as SortKey)}
          className="px-2 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-200 focus:outline-none focus:border-blue-500"
        >
          <option value="name">Sort: Name</option>
          <option value="created">Sort: Newest</option>
          <option value="size">Sort: Size</option>
        </select>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

function EmptyState({ hasEntries, onImport }: { hasEntries: boolean; onImport: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center gap-3">
      <div className="h-12 w-12 rounded-full bg-zinc-800 flex items-center justify-center">
        <LibraryIcon className="h-6 w-6 text-zinc-600" />
      </div>
      <div>
        <p className="text-sm font-medium text-zinc-300">
          {hasEntries ? 'No LoRAs match your filters' : 'Your library is empty'}
        </p>
        <p className="text-xs text-zinc-500 mt-1 max-w-sm">
          {hasEntries
            ? 'Try clearing the search or switching the filter.'
            : 'Import a LoRA from a file or HuggingFace, or finish a training run — trained adapters appear here automatically.'}
        </p>
      </div>
      {!hasEntries && (
        <button
          onClick={onImport}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium"
        >
          <Download className="h-3.5 w-3.5" />
          Import a LoRA
        </button>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Card
// ---------------------------------------------------------------------------

function KindBadge({ kind }: { kind: LoraInferenceEntry['kind'] }) {
  const styles: Record<LoraInferenceEntry['kind'], string> = {
    official_union: 'border-blue-500/40 text-blue-300 bg-blue-500/10',
    imported: 'border-violet-500/40 text-violet-300 bg-violet-500/10',
    user_trained: 'border-emerald-500/40 text-emerald-300 bg-emerald-500/10',
  }
  return (
    <span className={`shrink-0 text-[9px] leading-none px-1.5 py-0.5 rounded border ${styles[kind]}`}>
      {KIND_LABEL[kind]}
    </span>
  )
}

function LoraCard({
  entry,
  active,
  onClick,
}: {
  entry: LoraInferenceEntry
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`text-left rounded-lg border transition-colors flex flex-col overflow-hidden ${
        active
          ? 'border-blue-500 bg-blue-500/10'
          : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-800/40'
      } ${!entry.available ? 'opacity-60' : ''}`}
    >
      <div className="relative aspect-video bg-zinc-800">
        <ExampleThumb entry={entry} className="h-full w-full" rounded="rounded-none" />
        <div className="absolute top-1.5 right-1.5">
          <KindBadge kind={entry.kind} />
        </div>
      </div>
      <div className="p-3 flex flex-col gap-1.5">
        <p className="text-xs font-semibold text-zinc-100 truncate">{entry.name}</p>
        <p className="text-[10px] text-zinc-500 truncate">{VARIANT_LABEL[entry.variant]}</p>
        <div className="flex items-center gap-1.5 flex-wrap text-[10px] text-zinc-400">
          {entry.triggerWord && (
            <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700 truncate max-w-full">
              “{entry.triggerWord}”
            </span>
          )}
          {!entry.available && (
            <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30">
              Not downloaded
            </span>
          )}
        </div>
        <div className="flex items-center justify-between text-[10px] text-zinc-500 pt-1 border-t border-zinc-800">
          <span>{formatSize(entry.fileSizeBytes)}</span>
          <span>{formatDate(entry.createdAt)}</span>
        </div>
      </div>
    </button>
  )
}

function LoraListRow({
  entry,
  active,
  onClick,
}: {
  entry: LoraInferenceEntry
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-3 px-3 py-2 text-left transition-colors ${
        active
          ? 'bg-blue-500/10'
          : 'bg-zinc-900/30 hover:bg-zinc-800/40'
      } ${!entry.available ? 'opacity-60' : ''}`}
    >
      <ExampleThumb entry={entry} className="h-12 w-20 shrink-0" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <p className="text-xs font-semibold text-zinc-100 truncate">{entry.name}</p>
          <KindBadge kind={entry.kind} />
        </div>
        <p className="text-[10px] text-zinc-500 truncate">
          {VARIANT_LABEL[entry.variant]}
          {entry.triggerWord ? ` · “${entry.triggerWord}”` : ''}
        </p>
      </div>
      <div className="shrink-0 text-right text-[10px] text-zinc-500">
        <p>{formatSize(entry.fileSizeBytes)}</p>
        <p>{formatDate(entry.createdAt)}</p>
      </div>
    </button>
  )
}

// ---------------------------------------------------------------------------
// Detail / edit panel
// ---------------------------------------------------------------------------

function DetailPanel({
  entry,
  onClose,
  onChanged,
  onTryInGenSpace,
  onOpenTrainingRun,
}: {
  entry: LoraInferenceEntry
  onClose: () => void
  onChanged: () => Promise<void> | void
  onTryInGenSpace: (loraId: string) => void
  onOpenTrainingRun: (jobId: string) => void
}) {
  const isImported = entry.kind === 'imported'
  const isTrained = entry.kind === 'user_trained'
  const isOfficial = entry.kind === 'official_union'
  const editableMeta = isImported || isTrained

  const [name, setName] = useState(entry.name)
  const [description, setDescription] = useState(entry.description ?? '')
  const [huggingfaceUrl, setHuggingfaceUrl] = useState(entry.huggingfaceUrl ?? '')
  const [triggerWord, setTriggerWord] = useState(entry.triggerWord ?? '')
  const [promptTemplate, setPromptTemplate] = useState(entry.promptTemplate ?? '')

  // Re-bind local edits when switching to a different LoRA (keyed remount also
  // works, but this keeps the panel mounted across card clicks).
  useEffect(() => {
    setName(entry.name)
    setDescription(entry.description ?? '')
    setHuggingfaceUrl(entry.huggingfaceUrl ?? '')
    setTriggerWord(entry.triggerWord ?? '')
    setPromptTemplate(entry.promptTemplate ?? '')
  }, [entry.id, entry.name, entry.description, entry.huggingfaceUrl, entry.triggerWord, entry.promptTemplate])

  const [saving, setSaving] = useState(false)
  const [savingPrompt, setSavingPrompt] = useState(false)
  const [reprofiling, setReprofiling] = useState(false)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ tone: 'ok' | 'err'; text: string } | null>(null)

  const nameChanged = name.trim() !== entry.name
  const descChanged = (description.trim() || null) !== (entry.description ?? null)
  const hfChanged = (huggingfaceUrl.trim() || null) !== (entry.huggingfaceUrl ?? null)
  const metaChanged = nameChanged || descChanged || hfChanged
  const triggerChanged = (triggerWord.trim() || null) !== (entry.triggerWord ?? null)
  const templateChanged = (promptTemplate ?? null) !== (entry.promptTemplate ?? null)
  const promptChanged = triggerChanged || templateChanged

  const saveMeta = useCallback(async () => {
    if (!metaChanged || !editableMeta) return
    setSaving(true)
    setMsg(null)
    const body = {
      ...(nameChanged ? { name: name.trim() } : {}),
      ...(isImported ? { description: description.trim() || null, huggingfaceUrl: huggingfaceUrl.trim() || null } : { description: description.trim() || null }),
    }
    const result = isImported
      ? await ApiClient.updateImportedLora(entry.id, body as Parameters<typeof ApiClient.updateImportedLora>[1])
      : await ApiClient.updateTrainedLora(entry.id, body as Parameters<typeof ApiClient.updateTrainedLora>[1])
    setSaving(false)
    if (!result.ok) {
      const text = (result.error as { message?: string })?.message ?? 'Save failed'
      setMsg({ tone: 'err', text })
      logger.warn(`LoRA meta save failed (${text})`)
      return
    }
    await onChanged()
    setMsg({ tone: 'ok', text: 'Saved' })
  }, [metaChanged, editableMeta, nameChanged, isImported, description, huggingfaceUrl, name, entry.id, onChanged])

  const savePrompt = useCallback(async () => {
    if (!promptChanged) return
    setSavingPrompt(true)
    setMsg(null)
    const result = await ApiClient.updateLoraPromptTemplate(entry.id, {
      promptTemplate: promptTemplate.trim() || null,
      triggerWord: triggerWord.trim() || null,
    })
    setSavingPrompt(false)
    if (!result.ok) {
      const text = (result.error as { message?: string })?.message ?? 'Save failed'
      setMsg({ tone: 'err', text })
      logger.warn(`LoRA prompt save failed (${text})`)
      return
    }
    await onChanged()
    setMsg({ tone: 'ok', text: 'System prompt saved' })
  }, [promptChanged, promptTemplate, triggerWord, entry.id, onChanged])

  const regeneratePrompt = useCallback(async () => {
    if (savingPrompt || metaChanged) return
    if (!await confirmAction({
      title: 'Regenerate system prompt?',
      message: 'The current custom prompt will be replaced using “What this LoRA does” and the trigger word.',
      confirmLabel: 'Regenerate',
    })) return
    setSavingPrompt(true)
    setMsg(null)
    const result = await ApiClient.updateLoraPromptTemplate(entry.id, {
      promptTemplate: null,
      triggerWord: triggerWord.trim() || null,
    })
    setSavingPrompt(false)
    if (!result.ok) {
      const text = (result.error as { message?: string })?.message ?? 'Regeneration failed'
      setMsg({ tone: 'err', text })
      return
    }
    setPromptTemplate(result.data.entry.promptTemplate ?? '')
    await onChanged()
    setMsg({ tone: 'ok', text: 'System prompt regenerated' })
  }, [entry.id, metaChanged, onChanged, savingPrompt, triggerWord])

  const reprofile = useCallback(async () => {
    if (!isImported) return
    setReprofiling(true)
    setMsg(null)
    const result = await ApiClient.reprofileImportedLora(entry.id, {
      huggingfaceUrl: huggingfaceUrl.trim() || null,
    })
    setReprofiling(false)
    if (!result.ok) {
      const text = (result.error as { message?: string })?.message ?? 'Re-profile failed'
      setMsg({ tone: 'err', text })
      return
    }
    await onChanged()
    const status = result.data.profileStatus
    if (status === 'failed') {
      setMsg({ tone: 'err', text: result.data.profileMessage ?? 'Could not auto-configure the prompt.' })
    } else if (status === 'skipped') {
      setMsg({ tone: 'err', text: 'No source to profile from — add a HuggingFace URL or example prompt.' })
    } else {
      setMsg({ tone: 'ok', text: result.data.profileMessage ?? 'System prompt re-configured.' })
    }
  }, [isImported, entry.id, huggingfaceUrl, onChanged])

  const remove = useCallback(async () => {
    if (!await confirmAction({
      title: `Delete “${entry.name}”?`,
      message: 'The adapter will be removed from the library. This cannot be undone.',
      confirmLabel: 'Delete adapter',
      variant: 'destructive',
    })) return
    setBusy(true)
    setMsg(null)
    const result = isImported
      ? await ApiClient.deleteImportedLora(entry.id)
      : isTrained
        ? await ApiClient.deleteTrainedLora(entry.id)
        : null
    setBusy(false)
    if (result && !result.ok) {
      const text = (result.error as { message?: string })?.message ?? 'Delete failed'
      setMsg({ tone: 'err', text })
      return
    }
    await onChanged()
    onClose()
  }, [entry.name, entry.id, isImported, isTrained, onChanged, onClose])

  const reveal = useCallback(() => {
    if (entry.localPath) {
      void window.electronAPI.showItemInFolder({ filePath: entry.localPath })
    }
  }, [entry.localPath])

  return (
    <aside className="w-96 shrink-0 border-l border-zinc-800 flex flex-col min-h-0 bg-zinc-900/40">
      <div className="flex items-center justify-between px-3 py-3 border-b border-zinc-800">
        <div className="flex items-center gap-2 min-w-0">
          <KindBadge kind={entry.kind} />
          <span className="text-[10px] text-zinc-500 truncate">{VARIANT_LABEL[entry.variant]}</span>
        </div>
        <button onClick={onClose} className="h-6 w-6 flex items-center justify-center rounded text-zinc-400 hover:text-white hover:bg-zinc-800">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {/* Identity */}
        <section className="space-y-2">
          <FieldLabel>Name</FieldLabel>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={!editableMeta}
            className="w-full px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 disabled:opacity-60 focus:outline-none focus:border-blue-500"
          />
          <div className="flex items-center gap-1.5">
            <FieldLabel>What this LoRA does</FieldLabel>
            <InfoHint content="A short description of the learned transformation. The default prompt assistant uses this instead of guessing from the LoRA name." />
          </div>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            disabled={!editableMeta}
            rows={3}
            maxLength={500}
            placeholder={editableMeta ? 'e.g. Removes foreground people and reconstructs the hidden background' : 'No behavior description'}
            className="w-full px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 disabled:opacity-60 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500 resize-none"
          />
          {isImported && (
            <>
              <FieldLabel>HuggingFace URL</FieldLabel>
              <input
                type="url"
                value={huggingfaceUrl}
                onChange={(e) => setHuggingfaceUrl(e.target.value)}
                placeholder="https://huggingface.co/<org>/<repo>"
                spellCheck={false}
                className="w-full px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
              />
            </>
          )}
          {metaChanged && editableMeta && (
            <button
              onClick={saveMeta}
              disabled={saving}
              className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium disabled:opacity-50"
            >
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
              Save metadata
            </button>
          )}
        </section>

        {/* Example media (CivitAI-style preview) */}
        {(isImported || isTrained) && (
          <ExampleSection entry={entry} onChanged={onChanged} />
        )}

        {/* System prompt / trigger */}
        <section className="space-y-2">
          <div className="flex items-center gap-1.5">
            <FieldLabel>Trigger word</FieldLabel>
            <InfoHint content="The token or phrase that activates the LoRA. Include it in your prompt when generating." />
          </div>
          <input
            value={triggerWord}
            onChange={(e) => setTriggerWord(e.target.value)}
            placeholder="Exact token from the model card"
            className="w-full px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
          />
          {entry.variant !== 'union_control' && !triggerWord.trim() && (
            <p className="text-[10px] leading-relaxed text-amber-300">
              No verified trigger is recorded. The app will not guess one from the LoRA name; add the exact training token if this LoRA requires one.
            </p>
          )}
          <div className="flex items-center gap-1.5">
            <FieldLabel>System prompt</FieldLabel>
            <InfoHint content="The system prompt Gemini uses to write a tailored generation prompt from a reference video. Edit it to match the LoRA's exact style and trigger." />
          </div>
          <textarea
            value={promptTemplate ?? ''}
            onChange={(e) => setPromptTemplate(e.target.value)}
            rows={6}
            spellCheck={false}
            placeholder="The auto-generated system prompt will appear here…"
            className="w-full px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500 resize-none font-mono"
          />
          {promptChanged && (
            <button
              onClick={savePrompt}
              disabled={savingPrompt}
              className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium disabled:opacity-50"
            >
              {savingPrompt ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
              Save system prompt
            </button>
          )}
          <button
            onClick={() => void regeneratePrompt()}
            disabled={savingPrompt || metaChanged}
            title={metaChanged ? 'Save metadata before regenerating' : 'Regenerate from behavior description and trigger'}
            className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/40 hover:bg-zinc-800 text-zinc-200 text-xs font-medium disabled:opacity-50"
          >
            {savingPrompt ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCw className="h-3.5 w-3.5" />}
            Regenerate from fields
          </button>
          {isImported && (
            <button
              onClick={reprofile}
              disabled={reprofiling}
              className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/40 hover:bg-zinc-800 text-zinc-200 text-xs font-medium disabled:opacity-50"
            >
              {reprofiling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              Re-profile from HuggingFace
            </button>
          )}
        </section>

        {/* Provenance */}
        <section className="space-y-1.5 text-[11px] text-zinc-400">
          <FieldLabel>Details</FieldLabel>
          <DetailRow label="Kind" value={KIND_LABEL[entry.kind]} />
          <DetailRow label="Variant" value={VARIANT_LABEL[entry.variant]} />
          <DetailRow label="File size" value={formatSize(entry.fileSizeBytes)} />
          <DetailRow label="Created" value={formatDate(entry.createdAt)} />
          <DetailRow
            label="Status"
            value={entry.available ? 'Available' : 'Not downloaded'}
          />
          {entry.conditioningTypes.length > 0 && (
            <DetailRow label="Control" value={entry.conditioningTypes.join(', ')} />
          )}
          {isTrained && entry.sourceTrainingId && (
            <div className="pt-1">
              <button
                onClick={() => onOpenTrainingRun(entry.sourceTrainingId!)}
                className="inline-flex items-center gap-1.5 text-blue-300 hover:text-blue-200"
              >
                <Pencil className="h-3 w-3" />
                Open training run
              </button>
            </div>
          )}
          {entry.huggingfaceUrl && (
            <div className="pt-1">
              <a
                href={entry.huggingfaceUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 text-blue-300 hover:text-blue-200"
              >
                <ExternalLink className="h-3 w-3" />
                HuggingFace page
              </a>
            </div>
          )}
        </section>

        {msg && (
          <div
            className={`text-[11px] flex items-start gap-1.5 rounded-md border p-2 ${
              msg.tone === 'err'
                ? 'text-amber-300 border-amber-500/40 bg-amber-500/10'
                : 'text-emerald-300 border-emerald-500/40 bg-emerald-500/10'
            }`}
          >
            {msg.tone === 'err' ? (
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            ) : (
              <Check className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            )}
            <span>{msg.text}</span>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="px-3 py-3 border-t border-zinc-800 space-y-2">
        <button
          onClick={() => onTryInGenSpace(entry.id)}
          disabled={!entry.available}
          className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium disabled:opacity-50"
        >
          <Wand2 className="h-3.5 w-3.5" />
          Try in Gen Space
        </button>
        <div className="flex items-center gap-2">
          <button
            onClick={reveal}
            disabled={!entry.localPath}
            className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/40 hover:bg-zinc-800 text-zinc-200 text-xs font-medium disabled:opacity-50"
          >
            <FolderOpen className="h-3.5 w-3.5" />
            Reveal file
          </button>
          {(isImported || isTrained) && (
            <button
              onClick={remove}
              disabled={busy}
              className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md border border-red-500/40 bg-red-500/10 hover:bg-red-500/20 text-red-300 text-xs font-medium disabled:opacity-50"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
              Delete
            </button>
          )}
        </div>
        {isOfficial && (
          <p className="text-[10px] text-zinc-600 leading-relaxed text-center">
            The official union adapter is managed by the model download flow and can't be edited or deleted here.
          </p>
        )}
      </div>
    </aside>
  )
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <p className="text-[10px] uppercase tracking-wider text-zinc-500">{children}</p>
}

function ExampleSection({
  entry,
  onChanged,
}: {
  entry: LoraInferenceEntry
  onChanged: () => Promise<void> | void
}) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const pickAndUpload = useCallback(async () => {
    if (busy) return
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Choose an example image or video',
      filters: [
        { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp', 'gif'] },
        { name: 'Videos', extensions: ['mp4', 'webm', 'mov', 'm4v'] },
      ],
    })
    const filePath = paths?.[0]
    if (!filePath) return
    setBusy(true)
    setErr(null)
    const result = await ApiClient.setLoraExample(entry.id, { sourcePath: filePath })
    setBusy(false)
    if (!result.ok) {
      const text = (result.error as { message?: string })?.message ?? 'Could not attach example'
      setErr(text)
      logger.warn(`LoRA example attach failed (${text})`)
      return
    }
    await onChanged()
  }, [busy, entry.id, onChanged])

  const remove = useCallback(async () => {
    if (busy || !entry.exampleMediaType) return
    if (!await confirmAction({
      title: 'Remove example media?',
      message: 'The attached example will be removed from this LoRA.',
      confirmLabel: 'Remove example',
      variant: 'destructive',
    })) return
    setBusy(true)
    setErr(null)
    const result = await ApiClient.clearLoraExample(entry.id)
    setBusy(false)
    if (!result.ok) {
      const text = (result.error as { message?: string })?.message ?? 'Could not remove example'
      setErr(text)
      logger.warn(`LoRA example remove failed (${text})`)
      return
    }
    await onChanged()
  }, [busy, entry.exampleMediaType, entry.id, onChanged])

  return (
    <section className="space-y-2">
      <div className="flex items-center gap-1.5">
        <FieldLabel>Example</FieldLabel>
        <InfoHint content="Attach an image or video showing what this LoRA does. The complete original frame is preserved, including ultrawide before/after examples; gallery thumbnails fit it without cropping." />
      </div>
      <div className="rounded-md border border-zinc-800 bg-zinc-900/40 overflow-hidden">
        {entry.exampleMediaType ? (
          <div className="relative flex min-h-24 items-center justify-center bg-black">
            <ExampleThumb entry={entry} rounded="rounded-none" controls intrinsic />
            {busy && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <Loader2 className="h-5 w-5 animate-spin text-white" />
              </div>
            )}
          </div>
        ) : (
          <div className="aspect-video flex flex-col items-center justify-center gap-1.5 text-zinc-600">
            {busy ? <Loader2 className="h-5 w-5 animate-spin" /> : <ImagePlus className="h-5 w-5" />}
            <p className="text-[10px]">No example yet</p>
          </div>
        )}
      </div>
      {err && (
        <p className="text-[11px] text-red-400 flex items-start gap-1.5">
          <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
          <span>{err}</span>
        </p>
      )}
      <div className="flex items-center gap-2">
        <button
          onClick={pickAndUpload}
          disabled={busy}
          className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/40 hover:bg-zinc-800 text-zinc-200 text-xs font-medium disabled:opacity-50"
        >
          <ImagePlus className="h-3.5 w-3.5" />
          {entry.exampleMediaType ? 'Replace' : 'Add example'}
        </button>
        {entry.exampleMediaType && (
          <button
            onClick={remove}
            disabled={busy}
            className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md border border-red-500/40 bg-red-500/10 hover:bg-red-500/20 text-red-300 text-xs font-medium disabled:opacity-50"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Remove
          </button>
        )}
      </div>
    </section>
  )
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-zinc-500">{label}</span>
      <span className="text-zinc-300 truncate text-right">{value}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Import modal
// ---------------------------------------------------------------------------

const IMPORT_VARIANTS: { value: 'standard' | 'video_input_ic_lora'; label: string; hint: string }[] = [
  { value: 'standard', label: 'Standard', hint: 'Text/image-to-video style adapter' },
  { value: 'video_input_ic_lora', label: 'IC-LoRA', hint: 'Reference-video identity adapter' },
]

function ImportLoraModal({ onClose, onImported }: { onClose: () => void; onImported: () => Promise<void> | void }) {
  const [path, setPath] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [variant, setVariant] = useState<'standard' | 'video_input_ic_lora'>('standard')
  const [triggerWord, setTriggerWord] = useState('')
  const [hfUrl, setHfUrl] = useState('')
  const [examplePrompt, setExamplePrompt] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [importing, setImporting] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [profile, setProfile] = useState<{ status: string; message: string | null } | null>(null)

  const pickFile = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select LoRA weights',
      filters: [{ name: 'Safe LoRA weights', extensions: ['safetensors'] }],
    })
    if (paths && paths.length > 0) {
      const filePath = paths[0]
      setPath(filePath)
      const derived = filePath.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') ?? ''
      setName((prev) => prev || derived)
      setErr(null)
    }
  }, [])

  const submit = useCallback(async () => {
    if (!path || !name.trim() || importing) return
    setImporting(true)
    setErr(null)
    setProfile(null)
    const result = await ApiClient.importLora({
      sourcePath: path,
      name: name.trim(),
      variant,
      triggerWord: triggerWord.trim() || null,
      huggingfaceUrl: hfUrl.trim() || null,
      examplePrompt: examplePrompt.trim() || null,
    })
    setImporting(false)
    if (!result.ok) {
      const text = (result.error as { message?: string })?.message ?? 'Import failed'
      setErr(text)
      logger.warn(`LoRA import failed (${text})`)
      return
    }
    const data = result.data
    if (data.profileStatus === 'failed') {
      setProfile({ status: data.profileStatus, message: data.profileMessage ?? null })
      await onImported()
      return
    }
    await onImported()
    onClose()
  }, [path, name, importing, variant, triggerWord, hfUrl, examplePrompt, onImported, onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="w-full max-w-md rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
          <h3 className="text-sm font-semibold text-white">Import a LoRA</h3>
          <button onClick={onClose} className="h-6 w-6 flex items-center justify-center rounded text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="p-4 space-y-3">
          <button
            onClick={pickFile}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg border border-dashed border-zinc-700 bg-zinc-800/40 hover:border-blue-500 text-xs text-zinc-300"
          >
            <FolderOpen className="h-4 w-4 text-zinc-400" />
            {path ? path.split(/[\\/]/).pop() : 'Choose a .safetensors file'}
          </button>
          <div className="space-y-1">
            <FieldLabel>Name</FieldLabel>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Instant Shave"
              className="w-full px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="space-y-1">
            <FieldLabel>Variant</FieldLabel>
            <div className="grid grid-cols-2 gap-2">
              {IMPORT_VARIANTS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setVariant(opt.value)}
                  className={`text-left px-2.5 py-2 rounded-md border text-xs transition-colors ${
                    variant === opt.value
                      ? 'border-blue-500 bg-blue-500/10 text-white'
                      : 'border-zinc-700 bg-zinc-800/40 text-zinc-300 hover:border-zinc-600'
                  }`}
                >
                  <p className="font-medium">{opt.label}</p>
                  <p className="text-[10px] text-zinc-500">{opt.hint}</p>
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-1">
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="w-full flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-zinc-400 hover:text-zinc-200"
            >
              {showAdvanced ? <RotateCw className="h-3 w-3" /> : <Sparkles className="h-3 w-3" />}
              Auto-configure the system prompt (optional)
            </button>
            {showAdvanced && (
              <div className="space-y-2.5 rounded-md border border-zinc-700/70 bg-zinc-800/30 p-2.5">
                <p className="text-[10px] text-zinc-500 leading-relaxed">
                  A LoRA only activates if the prompt matches the structure and trigger it was trained on. Paste a
                  HuggingFace page or an example prompt and Gemini will configure an accurate system prompt + trigger.
                </p>
                <input
                  type="url"
                  value={hfUrl}
                  onChange={(e) => setHfUrl(e.target.value)}
                  placeholder="https://huggingface.co/<org>/<repo>"
                  spellCheck={false}
                  className="w-full px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
                />
                <textarea
                  value={examplePrompt}
                  onChange={(e) => setExamplePrompt(e.target.value)}
                  placeholder="Paste a prompt the LoRA was trained on…"
                  rows={3}
                  spellCheck={false}
                  className="w-full px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500 resize-none"
                />
                <div className="space-y-1">
                  <FieldLabel>Trigger word (optional)</FieldLabel>
                  <input
                    value={triggerWord}
                    onChange={(e) => setTriggerWord(e.target.value)}
                    placeholder="Exact token from the model card"
                    className="w-full px-2.5 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
                  />
                  <p className="text-[10px] leading-relaxed text-zinc-500">
                    Leave blank only if the author confirms no trigger is required. The app does not infer triggers from filenames.
                  </p>
                </div>
              </div>
            )}
          </div>
          {err && (
            <div className="text-[11px] text-red-400 flex items-start gap-1.5">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span>{err}</span>
            </div>
          )}
          {profile && (
            <div className="text-[11px] text-amber-300 flex items-start gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/10 p-2">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span>{profile.message ?? 'Could not auto-configure the system prompt. You can edit it manually after import.'}</span>
            </div>
          )}
          <button
            onClick={submit}
            disabled={!path || !name.trim() || importing}
            className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium disabled:opacity-50"
          >
            {importing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
            Import LoRA
          </button>
        </div>
      </div>
    </div>
  )
}
