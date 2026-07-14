import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Archive,
  Boxes,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Combine,
  Copy,
  Download,
  FileArchive,
  Folder,
  FolderInput,
  FolderOpen,
  FolderPlus,
  Library,
  Loader2,
  Move,
  Pencil,
  Plus,
  Trash2,
  XCircle,
} from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import type {
  LoraDataset,
  LoraFolder,
  LoraPreprocessed,
  LoraTrainingJob,
} from '../../contexts/LoraTrainingContext'
import { STAGE_DOT, deriveLifecycle } from './lifecycle'
import { ClipContextMenu, type ContextMenuItem } from './ClipContextMenu'
import {
  ComputePanel,
  type PodLifecycleInfo,
  type PodWorkTarget,
} from './ComputePanel'
import type { Selection } from './selection'
import { ArchiveManager } from './ArchiveManager'
import {
  loadLoraUiPreferences,
  saveLoraUiPreferences,
  type LoraSidebarSection,
  type LoraSidebarSectionSizes,
} from '../../lib/lora-ui-persistence'

function datasetPoster(dataset: LoraDataset): string | null {
  for (const clip of dataset.clips) {
    if (clip.posterPath) return clip.posterPath
  }
  return null
}

function GroupLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="px-2 pt-3 pb-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-600">
      {children}
    </p>
  )
}

function PaneHeader({
  title,
  count,
  collapsed,
  onToggle,
  actions,
}: {
  title: string
  count?: number
  collapsed: boolean
  onToggle: () => void
  actions?: React.ReactNode
}) {
  return (
    <div className="flex h-9 shrink-0 items-center justify-between border-b border-zinc-800 px-2">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={!collapsed}
        className="flex min-w-0 flex-1 items-center gap-1.5 rounded px-1 py-1 text-left hover:bg-zinc-800/70"
      >
        {collapsed
          ? <ChevronRight className="h-3.5 w-3.5 text-zinc-500" />
          : <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />}
        <span className="truncate text-[11px] font-semibold text-zinc-300">{title}</span>
        {count != null && <span className="text-[10px] text-zinc-600">{count}</span>}
      </button>
      {actions && <div className="ml-1 flex items-center gap-1">{actions}</div>}
    </div>
  )
}

function RenameInput({
  initial,
  onCommit,
  onCancel,
}: {
  initial: string
  onCommit: (next: string) => void
  onCancel: () => void
}) {
  const [draft, setDraft] = useState(initial)
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => {
    const el = ref.current
    if (el) {
      el.focus()
      el.select()
    }
  }, [])
  const commit = () => {
    const next = draft.trim()
    if (next && next !== initial) onCommit(next)
    else onCancel()
  }
  return (
    <input
      ref={ref}
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault()
          commit()
        } else if (e.key === 'Escape') {
          e.preventDefault()
          onCancel()
        }
      }}
      onBlur={commit}
      className="w-full bg-zinc-900 border border-blue-500/60 rounded text-xs text-white px-1 py-0.5 outline-none"
    />
  )
}

function NewFolderInput({
  onCommit,
  onCancel,
}: {
  onCommit: (name: string) => void
  onCancel: () => void
}) {
  const [draft, setDraft] = useState('')
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => {
    const el = ref.current
    if (el) {
      el.focus()
      // No select() — empty initial; just place the caret at the end.
    }
  }, [])
  const commit = () => {
    const next = draft.trim()
    if (next) onCommit(next)
    else onCancel()
  }
  return (
    <input
      ref={ref}
      placeholder="Folder name"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault()
          commit()
        } else if (e.key === 'Escape') {
          e.preventDefault()
          onCancel()
        }
      }}
      onBlur={commit}
      className="w-full bg-zinc-900 border border-blue-500/60 rounded text-xs text-white px-1 py-0.5 outline-none placeholder:text-zinc-600"
    />
  )
}

function CollectionRow({
  dataset,
  preprocessed,
  trainingJobs,
  active,
  marked,
  renaming,
  depth,
  onClick,
  onContextMenu,
  onRename,
  onRenameCancel,
  onDragStart,
}: {
  dataset: LoraDataset
  preprocessed: LoraPreprocessed[]
  trainingJobs: LoraTrainingJob[]
  active: boolean
  marked: boolean
  renaming: boolean
  depth: number
  onClick: (e: React.MouseEvent) => void
  onContextMenu: (e: React.MouseEvent) => void
  onRename: (next: string) => void
  onRenameCancel: () => void
  onDragStart: (e: React.DragEvent) => void
}) {
  const life = useMemo(
    () => deriveLifecycle(dataset, preprocessed, trainingJobs),
    [dataset, preprocessed, trainingJobs],
  )
  const poster = datasetPoster(dataset)
  const posterUrl = poster ? pathToFileUrl(poster) : null

  return (
    <button
      draggable={!renaming}
      onDragStart={onDragStart}
      onClick={onClick}
      onContextMenu={onContextMenu}
      style={{ paddingLeft: 8 + depth * 12 }}
      className={`w-full flex items-center gap-2.5 pr-2 py-1.5 rounded-md text-left transition-colors ${
        active
          ? 'bg-blue-500/15'
          : marked
            ? 'bg-blue-500/10 ring-1 ring-inset ring-blue-400/40'
            : 'hover:bg-zinc-800/70'
      }`}
    >
      <div
        className="h-9 w-12 flex-shrink-0 rounded bg-zinc-800 bg-cover bg-center"
        style={posterUrl ? { backgroundImage: `url("${posterUrl}")` } : undefined}
      >
        {!posterUrl && (
          <div className="h-full w-full flex items-center justify-center text-zinc-600">
            <Boxes className="h-4 w-4" />
          </div>
        )}
      </div>
      <div className="min-w-0 flex-1">
        {renaming ? (
          <RenameInput
            initial={dataset.name}
            onCommit={onRename}
            onCancel={onRenameCancel}
          />
        ) : (
          <>
            <p className={`text-xs font-medium truncate flex items-center gap-1.5 ${active ? 'text-white' : 'text-zinc-200'}`}>
              <span className="truncate">{dataset.name}</span>
              {dataset.type === 'ic_lora' && (
                <span
                  className="shrink-0 text-[9px] leading-none px-1 py-0.5 rounded border border-blue-500/40 text-blue-300 bg-blue-500/10"
                  title="In-Context LoRA: input → output"
                >
                  IC
                </span>
              )}
            </p>
            <p className="text-[10px] text-zinc-500 flex items-center gap-1.5">
              <span className={`h-1.5 w-1.5 rounded-full ${STAGE_DOT[life.tone]}`} />
              {(() => {
                const n = dataset.clips.filter((c) => !c.deletedAt).length
                return `${n} clip${n === 1 ? '' : 's'}`
              })()}{' '}
              · {life.label}
            </p>
          </>
        )}
      </div>
    </button>
  )
}

function FolderRow({
  folder,
  expanded,
  depth,
  renaming,
  isDropTarget,
  onToggle,
  onContextMenu,
  onRename,
  onRenameCancel,
  onDragStart,
  onDragOver,
  onDragLeave,
  onDrop,
}: {
  folder: LoraFolder
  expanded: boolean
  depth: number
  renaming: boolean
  isDropTarget: boolean
  onToggle: () => void
  onContextMenu: (e: React.MouseEvent) => void
  onRename: (next: string) => void
  onRenameCancel: () => void
  onDragStart: (e: React.DragEvent) => void
  onDragOver: (e: React.DragEvent) => void
  onDragLeave: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent) => void
}) {
  const Chevron = expanded ? ChevronDown : ChevronRight
  return (
    <div
      draggable={!renaming}
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onToggle}
      onContextMenu={onContextMenu}
      style={{ paddingLeft: 4 + depth * 12 }}
      className={`w-full flex items-center gap-1.5 pr-2 py-1.5 rounded-md cursor-pointer text-left transition-colors ${
        isDropTarget ? 'bg-blue-500/20 ring-1 ring-inset ring-blue-400/60' : 'hover:bg-zinc-800/70'
      }`}
    >
      <Chevron className="h-3.5 w-3.5 flex-shrink-0 text-zinc-500" />
      <Folder className="h-4 w-4 flex-shrink-0 text-amber-300/80" />
      <div className="min-w-0 flex-1">
        {renaming ? (
          <RenameInput
            initial={folder.name}
            onCommit={onRename}
            onCancel={onRenameCancel}
          />
        ) : (
          <p className="text-xs font-medium truncate text-zinc-200">{folder.name}</p>
        )}
      </div>
    </div>
  )
}

function RunRow({
  job,
  active,
  onClick,
  onArchive,
}: {
  job: LoraTrainingJob
  active: boolean
  onClick: () => void
  onArchive?: () => void
}) {
  const icon =
    job.status === 'running' || job.status === 'pending' ? (
      <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />
    ) : job.status === 'completed' ? (
      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
    ) : (
      <XCircle className="h-3.5 w-3.5 text-red-400" />
    )
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onClick()
        }
      }}
      className={`group w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-colors ${
        active ? 'bg-blue-500/15' : 'hover:bg-zinc-800/70'
      }`}
    >
      <span className="flex-shrink-0">{icon}</span>
      <span className={`text-xs truncate flex-1 ${active ? 'text-white' : 'text-zinc-300'}`}>{job.name}</span>
      {job.status === 'running' && job.gpuStatus ? (
        <span className="text-[10px] text-zinc-500 flex-shrink-0 tabular-nums" title="VRAM in use">
          {Math.round((job.gpuStatus.vramUsedMb / Math.max(1, job.gpuStatus.vramTotalMb)) * 100)}%
        </span>
      ) : job.status === 'running' && job.totalSteps ? (
        <span className="text-[10px] text-zinc-500 flex-shrink-0">
          {Math.round(((job.currentStep ?? 0) / job.totalSteps) * 100)}%
        </span>
      ) : null}
      {onArchive && (
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation()
            onArchive()
          }}
          title="Archive run"
          className="rounded p-1 text-zinc-600 opacity-0 transition-opacity hover:bg-zinc-700 hover:text-zinc-300 group-hover:opacity-100 focus:opacity-100"
        >
          <Archive className="h-3 w-3" />
        </button>
      )}
    </div>
  )
}

const COLLAPSE_KEY = 'lora.sidebar.collapsedFolders'

function loadCollapsed(): Set<string> {
  try {
    const raw = localStorage.getItem(COLLAPSE_KEY)
    if (!raw) return new Set()
    const arr = JSON.parse(raw) as string[]
    return new Set(arr)
  } catch {
    return new Set()
  }
}

function saveCollapsed(ids: Set<string>) {
  try {
    localStorage.setItem(COLLAPSE_KEY, JSON.stringify([...ids]))
  } catch {
    // ignore storage failures
  }
}

// Client-side descendant check mirroring the backend cycle guard, so
// drag-and-drop / Move-to can't even attempt an illegal reparent.
function isDescendantFolder(
  folders: LoraFolder[],
  maybeDescendantId: string,
  ancestorId: string,
): boolean {
  if (maybeDescendantId === ancestorId) return true
  let cur: string | null | undefined = maybeDescendantId
  const seen = new Set<string>()
  while (cur != null && !seen.has(cur)) {
    seen.add(cur)
    if (cur === ancestorId) return true
    const f = folders.find((x) => x.id === cur)
    cur = f?.parentId
  }
  return false
}

export function CollectionsSidebar({
  datasets,
  archivedDatasets,
  folders,
  preprocessed,
  trainingJobs,
  archivedTrainingJobs,
  selection,
  libraryCount,
  onSelectDataset,
  onSelectRun,
  onSelectLibrary,
  onNewDataset,
  onImportDataset,
  onDuplicate,
  onMerge,
  onDelete,
  onArchive,
  onArchiveRun,
  onRestoreDataset,
  onRestoreRun,
  onDeleteArchivedDataset,
  onDeleteArchivedRun,
  onRename,
  onCreateFolder,
  onRenameFolder,
  onMoveFolder,
  onDeleteFolder,
  onMoveDataset,
  activePodIds,
  workByPodId,
  lifecycleByPodId,
  onOpenPodWork,
}: {
  datasets: LoraDataset[]
  archivedDatasets: LoraDataset[]
  folders: LoraFolder[]
  preprocessed: LoraPreprocessed[]
  trainingJobs: LoraTrainingJob[]
  archivedTrainingJobs: LoraTrainingJob[]
  selection: Selection
  libraryCount: number
  onSelectDataset: (id: string) => void
  onSelectRun: (id: string) => void
  onSelectLibrary: () => void
  onNewDataset: () => void
  onImportDataset: (source: 'folder' | 'zip') => void
  onDuplicate: (id: string) => void
  onMerge: (ids: string[]) => void
  onDelete: (ids: string[]) => void
  onArchive: (ids: string[]) => void
  onArchiveRun: (id: string) => void
  onRestoreDataset: (id: string) => Promise<void>
  onRestoreRun: (id: string) => Promise<void>
  onDeleteArchivedDataset: (id: string) => Promise<void>
  onDeleteArchivedRun: (id: string) => Promise<void>
  onRename: (id: string, name: string) => void
  onCreateFolder: (name: string, parentId: string | null) => void
  onRenameFolder: (id: string, name: string) => void
  onMoveFolder: (id: string, parentId: string | null) => void
  onDeleteFolder: (id: string, recursive: boolean) => void
  onMoveDataset: (id: string, folderId: string | null) => void
  // Pod ids with an in-progress training job, so the ComputePanel can warn
  // before stopping/terminating a pod that's actively running a job.
  activePodIds: Set<string>
  workByPodId: ReadonlyMap<string, PodWorkTarget>
  lifecycleByPodId: ReadonlyMap<string, PodLifecycleInfo>
  onOpenPodWork: (target: PodWorkTarget) => void
}) {
  const runs = trainingJobs.filter((j) => j.status !== 'completed')
  const trained = trainingJobs.filter((j) => j.status === 'completed')

  const [markedIds, setMarkedIds] = useState<Set<string>>(new Set())
  const [menu, setMenu] = useState<
    { x: number; y: number; kind: 'dataset' | 'folder'; id: string } | null
  >(null)
  const [importMenu, setImportMenu] = useState<{ x: number; y: number } | null>(null)
  // Move-to picker: lists all folders + Root.
  const [moveMenu, setMoveMenu] = useState<
    { x: number; y: number; kind: 'dataset' | 'folder'; id: string } | null
  >(null)
  // Folder delete: choose "move contents up" vs "delete contents".
  const [folderDeleteMenu, setFolderDeleteMenu] = useState<{ x: number; y: number; id: string } | null>(null)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renamingFolderId, setRenamingFolderId] = useState<string | null>(null)
  // Inline "new folder" input. `undefined` = inactive; otherwise the parent
  // folder id to create under (`null` = root). Electron doesn't implement
  // window.prompt, so folder creation uses an inline input row instead.
  const [creatingFolderParent, setCreatingFolderParent] = useState<
    string | null | undefined
  >(undefined)
  const [collapsed, setCollapsed] = useState<Set<string>>(loadCollapsed)
  const [dropTargetId, setDropTargetId] = useState<string | null>(null)
  const dragItemRef = useRef<{ kind: 'dataset' | 'folder'; id: string } | null>(null)
  const expandTimerRef = useRef<number | null>(null)
  const anchorRef = useRef<string | null>(null)
  const sidebarRef = useRef<HTMLElement>(null)
  const [sectionSizes, setSectionSizes] = useState<LoraSidebarSectionSizes>(
    () => loadLoraUiPreferences().sidebarSectionSizes,
  )
  const [dragSectionPixels, setDragSectionPixels] =
    useState<LoraSidebarSectionSizes | null>(null)
  const [collapsedSections, setCollapsedSections] = useState<Set<LoraSidebarSection>>(
    () => new Set(loadLoraUiPreferences().collapsedSidebarSections),
  )
  const [archiveOpen, setArchiveOpen] = useState<'dataset' | 'run' | null>(null)

  const updateSectionSizes = (next: LoraSidebarSectionSizes) => {
    setSectionSizes(next)
    saveLoraUiPreferences({
      sidebarSectionSizes: next,
      computePanePercent: next.compute,
    })
  }

  const toggleSection = (section: LoraSidebarSection) => {
    setCollapsedSections((previous) => {
      const next = new Set(previous)
      if (next.has(section)) next.delete(section)
      else next.add(section)
      saveLoraUiPreferences({ collapsedSidebarSections: [...next] })
      return next
    })
  }

  const expandSections = (...sections: LoraSidebarSection[]) => {
    setCollapsedSections((previous) => {
      if (!sections.some((section) => previous.has(section))) return previous
      const next = new Set(previous)
      for (const section of sections) next.delete(section)
      saveLoraUiPreferences({ collapsedSidebarSections: [...next] })
      return next
    })
  }

  const resizeSections = (
    upper: LoraSidebarSection,
    lower: LoraSidebarSection,
    deltaPercent: number,
  ) => {
    const total = sectionSizes[upper] + sectionSizes[lower]
    const nextUpper = Math.min(total - 8, Math.max(8, sectionSizes[upper] + deltaPercent))
    updateSectionSizes({
      ...sectionSizes,
      [upper]: nextUpper,
      [lower]: total - nextUpper,
    })
  }

  const startResize = (
    event: React.PointerEvent<HTMLDivElement>,
    upper: LoraSidebarSection,
    lower: LoraSidebarSection,
  ) => {
    event.preventDefault()
    // Dragging a divider is an explicit request for space. Re-open either
    // collapsed neighbor so the same gesture can resize it immediately.
    expandSections(upper, lower)
    const sidebar = sidebarRef.current
    if (!sidebar) return
    const measured = (['datasets', 'runs', 'compute'] as const).reduce(
      (sizes, section) => {
        const pane = sidebar.querySelector<HTMLElement>(
          `[data-sidebar-section="${section}"]`,
        )
        sizes[section] = pane?.getBoundingClientRect().height ?? 36
        return sizes
      },
      { datasets: 36, runs: 36, compute: 36 } as LoraSidebarSectionSizes,
    )
    setDragSectionPixels(measured)
    const startY = event.clientY
    const pairPixels = measured[upper] + measured[lower]
    const pairWeights = sectionSizes[upper] + sectionSizes[lower]
    let latest = measured
    const onMove = (moveEvent: PointerEvent) => {
      const requestedUpper = measured[upper] + moveEvent.clientY - startY
      const nextUpper = Math.min(
        pairPixels - 36,
        Math.max(36, requestedUpper),
      )
      latest = {
        ...measured,
        [upper]: nextUpper,
        [lower]: pairPixels - nextUpper,
      }
      setDragSectionPixels(latest)
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      setDragSectionPixels(null)
      updateSectionSizes({
        ...sectionSizes,
        [upper]: pairWeights * (latest[upper] / pairPixels),
        [lower]: pairWeights * (latest[lower] / pairPixels),
      })
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  // Flat DFS order of datasets as rendered, so shift-range selection stays
  // sensible across the tree.
  const flatOrder = useMemo(() => {
    const order: string[] = []
    const walkFolder = (folderId: string) => {
      for (const d of datasets) if (d.folderId === folderId) order.push(d.id)
      for (const f of folders) if (f.parentId === folderId) walkFolder(f.id)
    }
    for (const d of datasets) if (d.folderId == null) order.push(d.id)
    for (const f of folders) if (f.parentId == null) walkFolder(f.id)
    return order
  }, [datasets, folders])

  const toggleFolder = (id: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      saveCollapsed(next)
      return next
    })
  }

  const clearExpandTimer = () => {
    if (expandTimerRef.current != null) {
      window.clearTimeout(expandTimerRef.current)
      expandTimerRef.current = null
    }
  }

  const handleDragStart = (kind: 'dataset' | 'folder', id: string, e: React.DragEvent) => {
    dragItemRef.current = { kind, id }
    e.dataTransfer.effectAllowed = 'move'
    // Some browsers require data to be set for DnD to initiate.
    e.dataTransfer.setData('text/plain', id)
  }

  const handleFolderDragOver = (folderId: string, e: React.DragEvent) => {
    const drag = dragItemRef.current
    // Reject dropping a folder onto itself or one of its descendants.
    if (drag?.kind === 'folder' && isDescendantFolder(folders, folderId, drag.id)) {
      e.dataTransfer.dropEffect = 'none'
      return
    }
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setDropTargetId(folderId)
    // Expand-on-hover: if the hovered folder is collapsed, expand it after a
    // short dwell so nested drops are possible.
    if (collapsed.has(folderId) && expandTimerRef.current == null) {
      expandTimerRef.current = window.setTimeout(() => {
        setCollapsed((prev) => {
          const next = new Set(prev)
          next.delete(folderId)
          saveCollapsed(next)
          return next
        })
        expandTimerRef.current = null
      }, 500)
    }
  }

  const handleFolderDragLeave = (folderId: string) => {
    if (dropTargetId === folderId) setDropTargetId(null)
    clearExpandTimer()
  }

  const handleFolderDrop = (folderId: string, e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    clearExpandTimer()
    setDropTargetId(null)
    const drag = dragItemRef.current
    dragItemRef.current = null
    if (!drag) return
    if (drag.kind === 'dataset') {
      onMoveDataset(drag.id, folderId)
    } else {
      if (drag.id === folderId) return
      if (isDescendantFolder(folders, folderId, drag.id)) return
      onMoveFolder(drag.id, folderId)
    }
  }

  const handleRootDrop = (e: React.DragEvent) => {
    e.preventDefault()
    clearExpandTimer()
    setDropTargetId(null)
    const drag = dragItemRef.current
    dragItemRef.current = null
    if (!drag) return
    if (drag.kind === 'dataset') onMoveDataset(drag.id, null)
    else onMoveFolder(drag.id, null)
  }

  const handleRootDragOver = (e: React.DragEvent) => {
    if (dragItemRef.current) {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'move'
    }
  }

  const handleRowClick = (id: string, e: React.MouseEvent) => {
    if (e.shiftKey && anchorRef.current) {
      const a = flatOrder.indexOf(anchorRef.current)
      const b = flatOrder.indexOf(id)
      if (a !== -1 && b !== -1) {
        const [lo, hi] = a < b ? [a, b] : [b, a]
        setMarkedIds(new Set(flatOrder.slice(lo, hi + 1)))
        return
      }
    }
    if (e.metaKey || e.ctrlKey) {
      setMarkedIds((prev) => {
        const next = new Set(prev)
        if (next.has(id)) next.delete(id)
        else next.add(id)
        return next
      })
      anchorRef.current = id
      return
    }
    anchorRef.current = id
    setMarkedIds(new Set([id]))
    onSelectDataset(id)
  }

  const openDatasetMenu = (id: string, e: React.MouseEvent) => {
    e.preventDefault()
    if (!markedIds.has(id)) {
      setMarkedIds(new Set([id]))
      anchorRef.current = id
    }
    setMenu({ x: e.clientX, y: e.clientY, kind: 'dataset', id })
  }

  const openFolderMenu = (id: string, e: React.MouseEvent) => {
    e.preventDefault()
    setMenu({ x: e.clientX, y: e.clientY, kind: 'folder', id })
  }

  const marked = [...markedIds].filter((id) => flatOrder.includes(id))

  // --- Move-to picker items (shared by dataset + folder Move to…) ---
  const moveToItems = (kind: 'dataset' | 'folder', sourceId: string): ContextMenuItem[] => {
    const items: ContextMenuItem[] = []
    const sourceFolder = folders.find((f) => f.id === sourceId)
    items.push({
      label: 'Root',
      icon: FolderInput,
      onClick: () => {
        if (kind === 'dataset') onMoveDataset(sourceId, null)
        else onMoveFolder(sourceId, null)
      },
    })
    for (const f of folders) {
      if (kind === 'folder') {
        // Can't move a folder into itself or one of its descendants.
        if (isDescendantFolder(folders, f.id, sourceId)) continue
        // No-op if already the parent.
        if (sourceFolder?.parentId === f.id) continue
      }
      items.push({
        label: f.name,
        icon: Folder,
        onClick: () => {
          if (kind === 'dataset') onMoveDataset(sourceId, f.id)
          else onMoveFolder(sourceId, f.id)
        },
      })
    }
    return items
  }

  // --- Context menu items ---
  const menuItems: ContextMenuItem[] = []
  if (menu) {
    if (menu.kind === 'dataset') {
      if (marked.length <= 1) {
        const id = menu.id
        menuItems.push({ label: 'Open', icon: FolderOpen, onClick: () => onSelectDataset(id) })
        menuItems.push({ label: 'Rename', icon: Pencil, onClick: () => setRenamingId(id) })
        menuItems.push({
          label: 'Move to…',
          icon: Move,
          onClick: () =>
            setMoveMenu({ x: menu.x, y: menu.y, kind: 'dataset', id }),
        })
        menuItems.push({ label: 'Duplicate', icon: Copy, onClick: () => onDuplicate(id) })
      } else {
        menuItems.push({ label: `Merge ${marked.length} into new collection…`, icon: Combine, onClick: () => onMerge(marked) })
      }
      menuItems.push({ type: 'separator' })
      menuItems.push({
        label: marked.length > 1 ? `Archive ${marked.length} collections` : 'Archive',
        icon: Archive,
        onClick: () => onArchive(marked),
      })
      menuItems.push({
        label: marked.length > 1 ? `Delete ${marked.length} collections` : 'Delete',
        icon: Trash2,
        danger: true,
        onClick: () => onDelete(marked),
      })
    } else {
      // Folder menu
      const id = menu.id
      menuItems.push({ label: 'New subfolder', icon: FolderPlus, onClick: () => promptNewFolder(id) })
      menuItems.push({ label: 'Rename', icon: Pencil, onClick: () => setRenamingFolderId(id) })
      menuItems.push({
        label: 'Move to…',
        icon: Move,
        onClick: () =>
          setMoveMenu({ x: menu.x, y: menu.y, kind: 'folder', id }),
      })
      menuItems.push({ type: 'separator' })
      menuItems.push({
        label: 'Delete folder',
        icon: Trash2,
        danger: true,
        onClick: () => setFolderDeleteMenu({ x: menu.x, y: menu.y, id }),
      })
    }
  }

  // Show an inline "new folder" input row. Electron doesn't implement
  // window.prompt, so we render an input inline (at root or inside the
  // right-clicked folder) instead of a native prompt. Auto-expands a collapsed
  // parent so the input is visible.
  const promptNewFolder = (parentId: string | null) => {
    if (parentId !== null) {
      setCollapsed((prev) => {
        if (!prev.has(parentId)) return prev
        const next = new Set(prev)
        next.delete(parentId)
        return next
      })
    }
    setCreatingFolderParent(parentId)
  }
  const commitNewFolder = (name: string) => {
    const parent = creatingFolderParent
    setCreatingFolderParent(undefined)
    if (parent !== undefined) onCreateFolder(name, parent)
  }
  const cancelNewFolder = () => setCreatingFolderParent(undefined)

  // --- Tree rendering ---
  const renderFolder = (folder: LoraFolder, depth: number): React.ReactNode => {
    const isExpanded = !collapsed.has(folder.id)
    const childFolders = folders.filter((f) => f.parentId === folder.id)
    const childDatasets = datasets.filter((d) => d.folderId === folder.id)
    return (
      <div key={folder.id} className="space-y-0.5">
        <FolderRow
          folder={folder}
          expanded={isExpanded}
          depth={depth}
          renaming={renamingFolderId === folder.id}
          isDropTarget={dropTargetId === folder.id}
          onToggle={() => toggleFolder(folder.id)}
          onContextMenu={(e) => openFolderMenu(folder.id, e)}
          onRename={(next) => {
            setRenamingFolderId(null)
            onRenameFolder(folder.id, next)
          }}
          onRenameCancel={() => setRenamingFolderId(null)}
          onDragStart={(e) => handleDragStart('folder', folder.id, e)}
          onDragOver={(e) => handleFolderDragOver(folder.id, e)}
          onDragLeave={() => handleFolderDragLeave(folder.id)}
          onDrop={(e) => handleFolderDrop(folder.id, e)}
        />
        {isExpanded && (
          <div className="space-y-0.5">
            {creatingFolderParent === folder.id && (
              <div style={{ paddingLeft: 4 + (depth + 1) * 12 }} className="pr-2">
                <NewFolderInput onCommit={commitNewFolder} onCancel={cancelNewFolder} />
              </div>
            )}
            {childFolders.map((f) => renderFolder(f, depth + 1))}
            {childDatasets.map((d) => (
              <CollectionRow
                key={d.id}
                dataset={d}
                preprocessed={preprocessed}
                trainingJobs={trainingJobs}
                active={selection?.kind === 'dataset' && selection.id === d.id}
                marked={markedIds.has(d.id)}
                renaming={renamingId === d.id}
                depth={depth + 1}
                onClick={(e) => handleRowClick(d.id, e)}
                onContextMenu={(e) => openDatasetMenu(d.id, e)}
                onRename={(next) => {
                  setRenamingId(null)
                  onRename(d.id, next)
                }}
                onRenameCancel={() => setRenamingId(null)}
                onDragStart={(e) => handleDragStart('dataset', d.id, e)}
              />
            ))}
          </div>
        )}
      </div>
    )
  }

  const rootFolders = folders.filter((f) => f.parentId == null)
  const rootDatasets = datasets.filter((d) => d.folderId == null)
  const paneRow = (section: LoraSidebarSection) =>
    collapsedSections.has(section)
      ? '36px'
      : dragSectionPixels
        ? `${dragSectionPixels[section]}px`
      : `minmax(36px, ${sectionSizes[section]}fr)`
  const resizeSeparator = (
    upper: LoraSidebarSection,
    lower: LoraSidebarSection,
  ) => {
    return (
      <div
        role="separator"
        aria-label={`Resize ${upper} and ${lower} panes`}
        aria-orientation="horizontal"
        tabIndex={0}
        onPointerDown={(event) => startResize(event, upper, lower)}
        onDoubleClick={() => {
          expandSections(upper, lower)
          updateSectionSizes({
            datasets: 42,
            runs: 24,
            compute: 34,
          })
        }}
        onKeyDown={(event) => {
          if (event.key === 'ArrowUp') {
            event.preventDefault()
            expandSections(upper, lower)
            resizeSections(upper, lower, -2)
          } else if (event.key === 'ArrowDown') {
            event.preventDefault()
            expandSections(upper, lower)
            resizeSections(upper, lower, 2)
          }
        }}
        className="block touch-none cursor-row-resize border-y border-zinc-800 bg-zinc-950 hover:bg-blue-500/20 focus:outline-none focus:ring-1 focus:ring-blue-500"
      />
    )
  }

  return (
    <aside ref={sidebarRef} data-tour="sidebar" aria-label="LoRA collections" className="flex h-full max-h-full min-h-0 w-full flex-1 flex-col overflow-hidden border-r border-zinc-800 lg:flex-none">
      <div
        className="grid h-full min-h-0 flex-1 overflow-hidden"
        style={{
          gridTemplateRows: `${paneRow('datasets')} 6px ${paneRow('runs')} 6px ${paneRow('compute')}`,
        }}
      >
      <section data-sidebar-section="datasets" className="flex min-h-0 flex-col overflow-hidden" aria-label="Datasets">
        <PaneHeader
          title="Datasets"
          count={datasets.length}
          collapsed={collapsedSections.has('datasets')}
          onToggle={() => toggleSection('datasets')}
          actions={
            <>
              <button
                onClick={() => promptNewFolder(null)}
                title="New folder"
                className="flex h-6 w-6 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-white"
              >
                <FolderPlus className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={(event) => {
                  const rect = event.currentTarget.getBoundingClientRect()
                  setImportMenu({ x: rect.left, y: rect.bottom + 4 })
                }}
                title="Import a dataset bundle"
                className="flex h-6 w-6 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-white"
              >
                <Download className="h-3.5 w-3.5" />
              </button>
              <button
                data-tour="new-collection"
                onClick={onNewDataset}
                title="New dataset"
                className="flex h-6 w-6 items-center justify-center rounded-md bg-blue-600 text-white hover:bg-blue-500"
              >
                <Plus className="h-3.5 w-3.5" />
              </button>
            </>
          }
        />
        {!collapsedSections.has('datasets') && (
        <div className="min-h-0 flex-1 overflow-y-auto px-1.5 pb-3 pt-1.5">
        <button
          data-tour="lora-library"
          onClick={onSelectLibrary}
          className={`w-full flex items-center gap-2.5 px-2 py-2 mb-2 rounded-md text-left transition-colors ${
            selection?.kind === 'library' ? 'bg-blue-500/15' : 'hover:bg-zinc-800/70'
          }`}
        >
          <div className="h-8 w-8 flex-shrink-0 rounded-md bg-gradient-to-br from-violet-500/30 to-blue-500/20 flex items-center justify-center">
            <Library className="h-4 w-4 text-violet-300" />
          </div>
          <div className="min-w-0 flex-1">
            <p className={`text-xs font-semibold truncate ${selection?.kind === 'library' ? 'text-white' : 'text-zinc-200'}`}>
              LoRA Library
            </p>
            <p className="text-[10px] text-zinc-500">
              {libraryCount} adapter{libraryCount === 1 ? '' : 's'} · browse &amp; manage
            </p>
          </div>
        </button>

        {datasets.length === 0 && folders.length === 0 && creatingFolderParent === undefined ? (
          <p className="px-2 py-2 text-[11px] text-zinc-600 leading-relaxed">
            No datasets yet. Create one, or send clips from Gen Space with &ldquo;To LoRA&rdquo;.
          </p>
        ) : (
          <>
            {marked.length > 1 && (
              <p className="px-2 pb-1 text-[10px] text-blue-300">
                {marked.length} selected · right-click to merge or delete
              </p>
            )}
            <div
              className="space-y-0.5"
              onDragOver={handleRootDragOver}
              onDrop={handleRootDrop}
            >
              {creatingFolderParent === null && (
                <div style={{ paddingLeft: 4 }} className="pr-2">
                  <NewFolderInput onCommit={commitNewFolder} onCancel={cancelNewFolder} />
                </div>
              )}
              {rootFolders.map((f) => renderFolder(f, 0))}
              {rootDatasets.map((d) => (
                <CollectionRow
                  key={d.id}
                  dataset={d}
                  preprocessed={preprocessed}
                  trainingJobs={trainingJobs}
                  active={selection?.kind === 'dataset' && selection.id === d.id}
                  marked={markedIds.has(d.id)}
                  renaming={renamingId === d.id}
                  depth={0}
                  onClick={(e) => handleRowClick(d.id, e)}
                  onContextMenu={(e) => openDatasetMenu(d.id, e)}
                  onRename={(next) => {
                    setRenamingId(null)
                    onRename(d.id, next)
                  }}
                  onRenameCancel={() => setRenamingId(null)}
                  onDragStart={(e) => handleDragStart('dataset', d.id, e)}
                />
              ))}
            </div>
          </>
        )}
        </div>
        )}
        {!collapsedSections.has('datasets') && archivedDatasets.length > 0 && (
          <button
            type="button"
            onClick={() => setArchiveOpen('dataset')}
            className="flex h-7 shrink-0 items-center gap-1.5 border-t border-zinc-800 px-3 text-[10px] text-zinc-500 hover:bg-zinc-800/60 hover:text-zinc-300"
          >
            <Archive className="h-3 w-3" />
            Archived datasets ({archivedDatasets.length})
          </button>
        )}
      </section>

      {resizeSeparator('datasets', 'runs')}

      <section data-sidebar-section="runs" className="flex min-h-0 flex-col overflow-hidden" aria-label="Training and runs">
        <PaneHeader
          title="Training / Runs"
          count={trainingJobs.length}
          collapsed={collapsedSections.has('runs')}
          onToggle={() => toggleSection('runs')}
        />
        {!collapsedSections.has('runs') && (
          <div className="min-h-0 flex-1 overflow-y-auto px-1.5 pb-3">
          {runs.length > 0 ? (
            <>
            <GroupLabel>Active and recent</GroupLabel>
            <div className="space-y-0.5">
              {runs.map((j) => (
                <RunRow
                  key={j.id}
                  job={j}
                  active={selection?.kind === 'run' && selection.id === j.id}
                  onClick={() => onSelectRun(j.id)}
                  onArchive={
                    j.status === 'pending' || j.status === 'running' || j.status === 'gpu_selection_required'
                      ? undefined
                      : () => onArchiveRun(j.id)
                  }
                />
              ))}
            </div>
            </>
          ) : (
            <p className="px-2 py-2 text-[11px] leading-relaxed text-zinc-600">
              No active training runs.
            </p>
          )}

        {trained.length > 0 && (
          <>
            <GroupLabel>Trained LoRAs</GroupLabel>
            <div className="space-y-0.5">
              {trained.map((j) => (
                <RunRow
                  key={j.id}
                  job={j}
                  active={selection?.kind === 'run' && selection.id === j.id}
                  onClick={() => onSelectRun(j.id)}
                  onArchive={() => onArchiveRun(j.id)}
                />
              ))}
            </div>
          </>
        )}
          </div>
        )}
        {!collapsedSections.has('runs') && archivedTrainingJobs.length > 0 && (
          <button
            type="button"
            onClick={() => setArchiveOpen('run')}
            className="flex h-7 shrink-0 items-center gap-1.5 border-t border-zinc-800 px-3 text-[10px] text-zinc-500 hover:bg-zinc-800/60 hover:text-zinc-300"
          >
            <Archive className="h-3 w-3" />
            Archived runs ({archivedTrainingJobs.length})
          </button>
        )}
      </section>

      {resizeSeparator('runs', 'compute')}

      <div data-sidebar-section="compute" className="min-h-0 overflow-hidden">
        <ComputePanel
          activePodIds={activePodIds}
          workByPodId={workByPodId}
          lifecycleByPodId={lifecycleByPodId}
          onOpenWork={onOpenPodWork}
          collapsed={collapsedSections.has('compute')}
          onToggleCollapsed={() => toggleSection('compute')}
        />
      </div>
      </div>

      {archiveOpen && (
        <ArchiveManager
          kind={archiveOpen}
          datasets={archivedDatasets}
          runs={archivedTrainingJobs}
          datasetNameForRun={(run) => {
            const prep = preprocessed.find((item) => item.id === run.preprocessedId)
            if (!prep) return null
            return [...datasets, ...archivedDatasets].find((dataset) => dataset.id === prep.datasetId)?.name ?? null
          }}
          onClose={() => setArchiveOpen(null)}
          onRestoreDataset={onRestoreDataset}
          onRestoreRun={onRestoreRun}
          onDeleteDataset={onDeleteArchivedDataset}
          onDeleteRun={onDeleteArchivedRun}
        />
      )}

      {menu && menuItems.length > 0 && (
        <ClipContextMenu x={menu.x} y={menu.y} items={menuItems} onClose={() => setMenu(null)} />
      )}
      {moveMenu && (
        <ClipContextMenu
          x={moveMenu.x}
          y={moveMenu.y}
          items={moveToItems(moveMenu.kind, moveMenu.id)}
          onClose={() => setMoveMenu(null)}
        />
      )}
      {folderDeleteMenu && (
        <ClipContextMenu
          x={folderDeleteMenu.x}
          y={folderDeleteMenu.y}
          items={[
            {
              label: 'Move contents up, then delete folder',
              icon: FolderInput,
              onClick: () => onDeleteFolder(folderDeleteMenu.id, false),
            },
            {
              label: 'Delete folder and all contents',
              icon: Trash2,
              danger: true,
              onClick: () => onDeleteFolder(folderDeleteMenu.id, true),
            },
          ]}
          onClose={() => setFolderDeleteMenu(null)}
        />
      )}
      {importMenu && (
        <ClipContextMenu
          x={importMenu.x}
          y={importMenu.y}
          items={[
            { label: 'Import from folder…', icon: FolderInput, onClick: () => onImportDataset('folder') },
            { label: 'Import from .zip…', icon: FileArchive, onClick: () => onImportDataset('zip') },
          ]}
          onClose={() => setImportMenu(null)}
        />
      )}

    </aside>
  )
}
