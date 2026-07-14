import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, ArrowLeftRight, BookOpen, CheckSquare, Crop, FlaskConical, FolderPlus, Image as ImageIcon, Images, Info, Link2, Maximize2, MessageSquare, Pencil, Plus, Scissors, SlidersHorizontal, Sparkles, ThumbsDown, ThumbsUp, Trash2, Unlink, X } from 'lucide-react'
import { ClipCard, type PairBadge } from '../studio/ClipCard'
import { ClipListTable, type ListColumnKey, type ListSort } from '../studio/ClipListTable'
import { PairCard } from './PairCard'
import { computeFacets } from '../studio/studio-facets'
import { toClipInput, useStudioActions, useStudioStore, type ClipTriage, type StudioClip } from '../studio/studio-store'
import { clipWarnings, datasetHealth, preflightChecks, type ClipProbeLike } from '../../lib/lora-quality'
import { countReadyPairs, derivePairs, groupMemberIds, pairReadiness } from '../../lib/lora-pairs'
import { GalleryToolbar, type Density, type GalleryLayout, type PairFilter, type PairView, type SortKey } from './GalleryToolbar'
import { DatasetInspector } from './DatasetInspector'
import { ClipEditor } from './ClipEditor'
import { ClipDetailModal } from './ClipDetailModal'
import { BulkActionsPill } from './BulkActionsPill'
import { BulkTrimModal, type TrimPlan } from './BulkTrimModal'
import { CreateTargetWizard } from './CreateTargetWizard'
import { EditReviewModal } from './EditReviewModal'
import { GroupPairModal } from './GroupPairModal'
import { FrameEditModal } from './FrameEditModal'
import { DatasetExportModal, type DatasetExportOptions } from './DatasetExportModal'
import { TrashModal } from './TrashModal'
import { TrainingSettingsModal } from './TrainingSettingsModal'
import { useLoraTraining } from '../../contexts/LoraTrainingContext'
import { useToast } from '../../contexts/ToastContext'
import { BulkCaptionModal } from './BulkCaptionModal'
import { ClipContextMenu, type ContextMenuItem } from './ClipContextMenu'
import { Tooltip } from '../../components/ui/tooltip'
import type { Lifecycle } from './lifecycle'
import type { ClipEdits, ClipInput, DerivationJob, LoraDataset, LoraProvider } from '../../contexts/LoraTrainingContext'
import { loadLoraUiPreferences, saveLoraUiPreferences } from '../../lib/lora-ui-persistence'

const DENSITY_GRID: Record<Density, string> = {
  large: 'grid-cols-2 lg:grid-cols-3',
  medium: 'grid-cols-3 lg:grid-cols-4 xl:grid-cols-5',
  small: 'grid-cols-4 lg:grid-cols-6 xl:grid-cols-7',
}

// Below this duration, splitting a clip is a no-op, so the Split actions stay
// hidden to keep the menus uncluttered.
const SPLIT_MIN_SECONDS = 12

const CROP_RATIOS: Array<{ label: string; ratio: [number, number] }> = [
  { label: '16:9', ratio: [16, 9] },
  { label: '9:16', ratio: [9, 16] },
  { label: '1:1', ratio: [1, 1] },
]

const NORMALIZE_CHOICES: Array<{ label: string; fps: number | null }> = [
  { label: 'Snap size + 24fps', fps: 24 },
  { label: 'Snap size + 25fps', fps: 25 },
  { label: 'Snap size + 30fps', fps: 30 },
  { label: 'Snap size only', fps: null },
]

export function userFacingDatasetError(error: string | null | undefined): string | null {
  if (!error) return null
  if (!/Pick one in Settings\s*→\s*LoRA Trainer/i.test(error)) return error
  const gpu = error.match(/GPU '([^']+)'/)?.[1]
  return `${gpu ? `GPU '${gpu}'` : 'The selected GPU'} is unavailable right now. `
    + 'Return to GPU selection and choose another available GPU, or refresh and '
    + 'retry later. Your dataset and progress are preserved.'
}

/**
 * Which "axis" a tag belongs to, for group-aware multi-filtering: tags in the
 * same dimension combine with OR (a clip can't be both 16:9 and 21:9), while
 * tags across dimensions combine with AND (e.g. 16:9 AND uncaptioned). Each
 * standalone quality flag is its own dimension so they intersect.
 */
function facetDimension(id: string): string {
  if (id.startsWith('aspect:')) return 'aspect'
  if (id === 'kept' || id === 'rejected' || id === 'holdout') return 'triage'
  if (id === 'pairs' || id === 'incomplete-pairs' || id === 'ungrouped') return 'pairing'
  return id
}

function SectionHeader({ label, count, hint }: { label: string; count: number; hint?: string }) {
  return (
    <div className="flex items-baseline gap-2 mb-2 px-0.5">
      <h3 className="text-[11px] font-semibold uppercase tracking-wide text-zinc-400">{label}</h3>
      <span className="text-[11px] text-zinc-600">{count}</span>
      {hint && <span className="text-[11px] text-zinc-600 truncate">· {hint}</span>}
    </div>
  )
}

export function CollectionView({
  dataset,
  life,
  provider,
  credentialsReady,
  busyAction,
  onPrimary,
  onRename,
  onSetTrigger,
  onSetType,
  onDelete,
  onArchive,
  onCaptionSelected,
  onCaptionChange,
  onApplyCaptions,
  onSetTriage,
  onRemoveSelected,
  onRestoreClips,
  onPurgeClips,
  onCropSelected,
  onTrimSelected,
  onApplyEdit,
  onRevertEdit,
  onCreatePair,
  onCreateReference,
  onNormalizeSelected,
  onSceneSplitSelected,
  onSegmentSelected,
  onAddStill,
  onGroupPair,
  onUngroup,
  onDropFiles,
  onAddClips,
  onBrowsePexels,
  onCreateCollection,
  onOpenRecipes,
  onCancelPreprocess,
  onCancelUpload,
}: {
  dataset: LoraDataset
  life: Lifecycle
  provider: LoraProvider
  credentialsReady: boolean
  busyAction: boolean
  onPrimary: () => void
  onRename: (next: string) => void
  onSetTrigger: (next: string) => void
  onSetType: (type: 'standard' | 'ic_lora') => void
  onDelete: () => void
  onArchive: () => void
  onCaptionSelected: (ids: string[]) => void
  onCaptionChange: (clipId: string, caption: string) => void
  onApplyCaptions: (updates: Array<{ id: string; caption: string }>) => void
  onSetTriage: (ids: string[], triage: ClipTriage | null) => void
  onRemoveSelected: (ids: string[]) => void
  onRestoreClips: (ids: string[]) => void
  onPurgeClips: (ids: string[]) => void
  onCropSelected: (ids: string[], ratio: [number, number]) => void
  onTrimSelected: (ids: string[], plan: TrimPlan) => void
  onApplyEdit: (clipId: string, edits: ClipEdits) => void
  onRevertEdit: (clipId: string) => void
  onCreatePair: (input: ClipInput) => Promise<boolean>
  onCreateReference: (sourceClipId: string, input: ClipInput) => Promise<boolean>
  onCreateMany: (inputs: ClipInput[]) => void
  onNormalizeSelected: (ids: string[], targetFps: number | null) => void
  onSceneSplitSelected: (ids: string[]) => void
  onSegmentSelected: (ids: string[], seconds: number) => void
  onAddStill: (opts: { id?: string; framePath: string; caption: string; driverPath: string; probe?: ClipProbeLike | null }) => Promise<boolean>
  onGroupPair: (targetId: string, referenceIds: string[]) => void
  onUngroup: (ids: string[]) => void
  onDropFiles: (paths: string[]) => void
  onAddClips: () => void
  onBrowsePexels: () => void
  onCreateCollection: (clips: ClipInput[]) => void
  onOpenRecipes: () => void
  onCancelPreprocess: (preprocessedId: string) => void
  onCancelUpload: () => void
}) {
  const allClips = useStudioStore((s) => s.clips)
  // Trashed clips live in the recycle bin: hidden from the gallery and excluded
  // from every selector below (pairing, facets, readiness). `clips` therefore
  // means "active clips" everywhere downstream.
  const clips = useMemo(() => allClips.filter((c) => !c.deletedAt), [allClips])
  const trashedClips = useMemo(() => allClips.filter((c) => c.deletedAt), [allClips])
  const selectedIds = useStudioStore((s) => s.selectedIds)
  const triggerWord = useStudioStore((s) => s.triggerWord)
  const { clearSelection, setSelection, setEditPreview } = useStudioActions()

  // Anchor for Shift-range selection (the last clip clicked without Shift),
  // mirroring file-explorer behaviour. Lives in a ref so re-ranging from the
  // same anchor doesn't depend on render order.
  const selectionAnchor = useRef<string | null>(null)
  // The gallery's exact top-to-bottom render order, captured per render so a
  // Shift-range matches what the user sees. Examples render first (each
  // contributing all its member ids as one contiguous unit), then loose clips —
  // which is *not* `visibleClips` order, so ranges must use this instead.
  // `groupOf` maps every example member to its full membership so a range can be
  // expanded to whole examples.
  const displayOrderRef = useRef<{ order: string[]; groupOf: Map<string, string[]> }>({
    order: [],
    groupOf: new Map(),
  })

  const [search, setSearch] = useState('')
  const [density, setDensity] = useState<Density>('medium')
  const [sort, setSort] = useState<SortKey>('added')
  const [pairView, setPairView] = useState<PairView>(
    () => loadLoraUiPreferences().pairView,
  )
  const [pairFilter, setPairFilter] = useState<PairFilter>(
    () => loadLoraUiPreferences().pairFilter,
  )
  const [layout, setLayout] = useState<GalleryLayout>(
    () => loadLoraUiPreferences().galleryLayout,
  )
  // Column sort for the List layout (independent of the grid's coarse Sort).
  const [listSort, setListSort] = useState<ListSort>({ key: null, dir: 'asc' })
  const [groupModalOpen, setGroupModalOpen] = useState(false)
  const [hoveredSet, setHoveredSet] = useState<number | null>(null)
  const [activeFacets, setActiveFacets] = useState<string[]>([])
  const [editorClipId, setEditorClipId] = useState<string | null>(null)
  const [detailClipId, setDetailClipId] = useState<string | null>(null)
  // Unified "generate target/variant" wizard: a single clip, or a batch over
  // the current selection.
  const [wizardClipId, setWizardClipId] = useState<string | null>(null)
  const [wizardMode, setWizardMode] = useState<'target' | 'variant'>('target')
  const [wizardBatchOpen, setWizardBatchOpen] = useState(false)
  const [frameEditClipId, setFrameEditClipId] = useState<string | null>(null)
  const [bulkFrameOpen, setBulkFrameOpen] = useState(false)
  const [bulkCaptionOpen, setBulkCaptionOpen] = useState(false)
  const [bulkTrimOpen, setBulkTrimOpen] = useState(false)
  const [menu, setMenu] = useState<{ x: number; y: number; clipId: string } | null>(null)
  const [bgMenu, setBgMenu] = useState<{ x: number; y: number } | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const dragDepth = useRef(0)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const inspectorToggleRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!inspectorOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopPropagation()
      setInspectorOpen(false)
      inspectorToggleRef.current?.focus()
    }
    window.addEventListener('keydown', onKeyDown, true)
    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [inspectorOpen])

  const onPairViewChange = useCallback((v: PairView) => {
    setPairView(v)
    saveLoraUiPreferences({ pairView: v })
  }, [])
  const onPairFilterChange = useCallback((f: PairFilter) => {
    setPairFilter(f)
    saveLoraUiPreferences({ pairFilter: f })
  }, [])
  const onListSort = useCallback((key: ListColumnKey) => {
    setListSort((prev) =>
      prev.key === key ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'asc' },
    )
  }, [])
  const onLayoutChange = useCallback((l: GalleryLayout) => {
    setLayout(l)
    saveLoraUiPreferences({ galleryLayout: l })
  }, [])

  const facets = useMemo(() => computeFacets(clips), [clips])

  // Compact always-visible readiness signal next to the primary CTA, so the
  // user knows whether the set is upload-ready while curating.
  const readiness = useMemo(() => {
    if (clips.length === 0) return null
    // Rejected and holdout clips are excluded from training (holdout is
    // validation-only), so they shouldn't count toward (or drag down) readiness
    // — only the kept/unreviewed clips ship.
    const trainClips = clips.filter((c) => c.triage !== 'reject' && c.triage !== 'holdout')
    const likeClips = trainClips.map((c) => ({ caption: c.caption, probe: c.probe }))
    const health = datasetHealth(likeClips)
    const blocked = preflightChecks(likeClips, { triggerWord }).some((c) => c.blocker && !c.ok)
    // Dataset-level pair rollup (over all clips, not just the filtered view).
    const { pairs } = derivePairs(trainClips)
    const readyPairs = countReadyPairs(pairs)
    const pairsLabel = pairs.length > 0 ? `${readyPairs}/${pairs.length} examples ready` : null
    const tone: 'ready' | 'warn' | 'error' = blocked
      ? 'error'
      : health.errorCount > 0 || health.captionedCount < health.clipCount || readyPairs < pairs.length
        ? 'warn'
        : 'ready'
    const label = blocked
      ? 'Not ready — needs more clips'
      : tone === 'warn'
        ? pairsLabel && readyPairs < pairs.length
          ? `Almost ready — ${pairsLabel}`
          : 'Almost ready — captions / quality warnings remain'
        : 'Ready to upload'
    return { score: health.score, tone, label, pairsLabel }
  }, [clips, triggerWord])
  // Drop any active tags that no longer exist (e.g. the last uncaptioned clip
  // got a caption, so the facet disappears).
  useEffect(() => {
    setActiveFacets((prev) => {
      const valid = prev.filter((id) => facets.some((f) => f.id === id))
      return valid.length === prev.length ? prev : valid
    })
  }, [facets])
  const activeFacetDefs = useMemo(
    () => facets.filter((f) => activeFacets.includes(f.id)),
    [facets, activeFacets],
  )

  const query = search.trim().toLowerCase()
  const visibleClips = useMemo(() => {
    let list = clips
    if (activeFacetDefs.length) {
      // Group-aware combine: OR within a dimension (aspect / triage / pairing),
      // AND across dimensions. A clip must satisfy every active dimension.
      const byDim = new Map<string, Set<string>>()
      for (const f of activeFacetDefs) {
        const dim = facetDimension(f.id)
        const set = byDim.get(dim) ?? new Set<string>()
        for (const id of f.ids) set.add(id)
        byDim.set(dim, set)
      }
      const dims = [...byDim.values()]
      list = list.filter((c) => dims.every((set) => set.has(c.id)))
    }
    if (query) list = list.filter((c) => c.caption.toLowerCase().includes(query))
    if (sort === 'duration') {
      list = [...list].sort(
        (a, b) => (b.probe?.durationSeconds ?? 0) - (a.probe?.durationSeconds ?? 0),
      )
    } else if (sort === 'attention') {
      const severity = (c: (typeof list)[number]) => {
        const w = clipWarnings({ caption: c.caption, probe: c.probe })
        if (w.some((x) => x.level === 'error')) return 2
        if (w.length > 0 || !c.caption.trim()) return 1
        return 0
      }
      list = [...list].sort((a, b) => severity(b) - severity(a))
    } else if (sort === 'pairs') {
      // Keep set members contiguous and ordered (references then result),
      // sets first by their derivation order, loose clips last. This is what
      // makes the "flat" layout legible and groups everything else neatly.
      const { pairs } = derivePairs(clips)
      const rank = new Map<string, number>()
      pairs.forEach((g, gi) => {
        g.controls.forEach((c, ci) => rank.set(c.id, gi * 1000 + ci))
        g.targets.forEach((t, ti) => rank.set(t.id, gi * 1000 + 500 + ti))
      })
      const keyOf = (c: (typeof list)[number]) => rank.get(c.id) ?? Number.MAX_SAFE_INTEGER
      list = [...list].sort((a, b) => keyOf(a) - keyOf(b))
    }
    return list
  }, [clips, activeFacetDefs, query, sort])

  // Selection model (file-explorer semantics):
  //  - plain click       → select only this clip
  //  - checkbox / ⌘/Ctrl → toggle this clip in/out of the set (additive)
  //  - Shift+click        → select the contiguous range from the anchor, in the
  //                         order currently shown (so it respects sort/filter)
  const handleSelect = useCallback(
    (id: string, intent: { additive: boolean; range: boolean }) => {
      const { order, groupOf } = displayOrderRef.current
      const anchor = selectionAnchor.current
      const a = anchor ? order.indexOf(anchor) : -1
      const b = order.indexOf(id)
      if (intent.range && a !== -1 && b !== -1) {
        const [lo, hi] = a < b ? [a, b] : [b, a]
        // Expand any example the range touches to its *whole* membership: range
        // endpoints land on a single group member, and an example card only
        // reads as selected when every member is, so a raw slice would clip
        // boundary examples and leave them looking unselected.
        const range = new Set<string>()
        for (const cid of order.slice(lo, hi + 1)) {
          const members = groupOf.get(cid)
          if (members) for (const m of members) range.add(m)
          else range.add(cid)
        }
        // Shift alone replaces with the range; ⌘/Ctrl+Shift extends the set.
        setSelection(intent.additive ? [...new Set([...selectedIds, ...range])] : [...range])
        return
      }
      if (intent.additive) {
        const next = new Set(selectedIds)
        if (next.has(id)) next.delete(id)
        else next.add(id)
        setSelection([...next])
        selectionAnchor.current = id
        return
      }
      setSelection([id])
      selectionAnchor.current = id
    },
    [selectedIds, setSelection],
  )

  // Selecting a pair card operates on all its member ids at once.
  const handleSelectGroup = useCallback(
    (ids: string[], intent: { additive: boolean; range: boolean }) => {
      if (ids.length === 0) return
      const anchorId = ids[ids.length - 1]
      if (intent.range) {
        // Range from anchor to the group's edited clip, in displayed order.
        handleSelect(anchorId, intent)
        return
      }
      if (intent.additive) {
        const allSelected = ids.every((id) => selectedIds.has(id))
        const next = new Set(selectedIds)
        for (const id of ids) {
          if (allSelected) next.delete(id)
          else next.add(id)
        }
        setSelection([...next])
        selectionAnchor.current = anchorId
        return
      }
      setSelection(ids)
      selectionAnchor.current = anchorId
    },
    [selectedIds, setSelection, handleSelect],
  )

  // Group the visible clips into IC-LoRA sets (one target + N references) vs
  // loose clips. Pure derivation from each target's references.
  const { pairGroups, looseClips } = useMemo(() => {
    const { pairs, looseClipIds } = derivePairs(visibleClips)
    return { pairGroups: pairs, looseClips: visibleClips.filter((c) => looseClipIds.has(c.id)) }
  }, [visibleClips])

  // clipId → its pair marker (index, role, readiness) for the "flat" layout.
  const pairBadges = useMemo(() => {
    const map = new Map<string, PairBadge>()
    pairGroups.forEach((g, i) => {
      const tone = pairReadiness(g).tone
      for (const c of g.controls) map.set(c.id, { index: i + 1, tone, role: 'ref' })
      for (const t of g.targets) map.set(t.id, { index: i + 1, tone, role: 'target' })
    })
    return map
  }, [pairGroups])

  // Apply the pairing filter to what actually renders (selection/facets still
  // operate on the full visible set).
  const incompletePairs = useMemo(
    () => pairGroups.filter((g) => pairReadiness(g).tone !== 'ready'),
    [pairGroups],
  )
  const shownPairs =
    pairFilter === 'looseOnly' ? [] : pairFilter === 'incomplete' ? incompletePairs : pairGroups
  const shownLoose = pairFilter === 'pairsOnly' || pairFilter === 'incomplete' ? [] : looseClips
  const nothingShown = shownPairs.length === 0 && shownLoose.length === 0

  // Keep the Shift-range order in sync with what actually renders (examples
  // first as contiguous units, then loose clips), so range selection is WYSIWYG
  // and a range that spans examples selects each example in full.
  const displayOrder = useMemo(() => {
    const order: string[] = []
    const groupOf = new Map<string, string[]>()
    for (const g of shownPairs) {
      const members = groupMemberIds(g)
      for (const memberId of members) {
        order.push(memberId)
        groupOf.set(memberId, members)
      }
    }
    for (const c of shownLoose) order.push(c.id)
    return { order, groupOf }
  }, [shownPairs, shownLoose])
  useEffect(() => {
    displayOrderRef.current = displayOrder
  }, [displayOrder])


  // Only draft / retryable datasets accept new or edited clips; once uploaded
  // the remote copy is in flight, so curation is locked (matches SendToLora).
  const editable =
    dataset.status === 'draft' ||
    dataset.status === 'upload_failed' ||
    dataset.status === 'cancelled'
  // IC-LoRA collections train reference → target transformations, so the
  // example-grouping UI (sections, set views, Generate target / Group) only
  // appears here. Standard LoRAs are a plain grid of clips + captions.
  const isIcLora = dataset.type === 'ic_lora'

  // Tags only narrow the view (toggle in/out); they no longer hijack the
  // selection. Use "Select all shown" to act on the filtered set.
  const onFacetClick = (id: string) => {
    setActiveFacets((prev) => (prev.includes(id) ? prev.filter((f) => f !== id) : [...prev, id]))
  }
  const onClearFacets = () => setActiveFacets([])
  const selectAllShown = useCallback(
    () => setSelection(visibleClips.map((c) => c.id)),
    [visibleClips, setSelection],
  )

  const primaryDisabled = (life.primary?.needsCredentials ?? false) && !credentialsReady
  const selectedCount = selectedIds.size
  const selectedList = [...selectedIds]
  const editorClip = editorClipId ? clips.find((c) => c.id === editorClipId) ?? null : null
  const wizardClip = wizardClipId ? clips.find((c) => c.id === wizardClipId) ?? null : null
  const frameEditClip = frameEditClipId ? clips.find((c) => c.id === frameEditClipId) ?? null : null
  const detailClip = detailClipId ? clips.find((c) => c.id === detailClipId) ?? null : null
  const selectedClips = selectedList
    .map((id) => clips.find((c) => c.id === id))
    .filter((c): c is StudioClip => !!c)
  const singleClip = selectedClips.length === 1 ? selectedClips[0] : null
  // Edit ops (crop/trim/normalize/split) only make sense for video clips.
  const selHasVideo = selectedClips.some((c) => c.kind === 'video')
  // Split only helps when a clip is long enough to break up; surfacing it for
  // already-short clips just adds noise (it'd be a no-op).
  const canSplit = selectedClips.some(
    (c) => c.kind === 'video' && (c.probe?.durationSeconds ?? c.durationSeconds ?? 0) > SPLIT_MIN_SECONDS,
  )

  // All pair groups across the whole dataset (not just the filtered view), so
  // ungroup works regardless of the active facet/filter. Same derivation the
  // gallery cards use, so detection and rendering can never disagree.
  const datasetPairs = useMemo(() => derivePairs(clips).pairs, [clips])
  // Split the gallery into "Examples" / "Ungrouped clips" sections once an
  // IC-LoRA collection has any examples; standard LoRAs stay a plain grid.
  const sectioned = isIcLora && datasetPairs.length > 0

  // Pair groups that any selected clip belongs to (as a target or reference).
  const selectedPairGroups = useMemo(() => {
    if (selectedIds.size === 0) return []
    return datasetPairs.filter((g) => groupMemberIds(g).some((id) => selectedIds.has(id)))
  }, [datasetPairs, selectedIds])
  const hasPairedSelection = selectedPairGroups.length > 0
  // Examples that can actually be reversed: both sides resolved (else there's
  // nothing to promote into a target).
  const flippablePairCount = selectedPairGroups.filter(
    (g) => g.controls.length > 0 && g.targets.length > 0,
  ).length

  const handleCreateCollection = useCallback(() => {
    if (selectedClips.length === 0) return
    onCreateCollection(selectedClips.map(toClipInput))
  }, [selectedClips, onCreateCollection])

  const handleGroup = useCallback(() => {
    if (selectedClips.length > 1) setGroupModalOpen(true)
  }, [selectedClips.length])

  const handleUngroup = useCallback(() => {
    // Dissolve every group the selection touches by clearing references on all
    // of their targets — so selecting any member (or the whole pair card)
    // fully ungroups it, and the card disappears.
    const toClear = [...new Set(selectedPairGroups.flatMap((g) => g.targets.map((t) => t.id)))]
    if (toClear.length) onUngroup(toClear)
  }, [selectedPairGroups, onUngroup])

  // Candidate driving videos for motion-locking a still into a pair.
  const driverChoices = useMemo(
    () =>
      clips
        .filter((c) => c.kind === 'video')
        .map((c) => ({ path: c.localPath, label: c.caption?.trim() || c.localPath.split('/').pop() || c.localPath })),
    [clips],
  )

  // Background target/variant generation for this collection. Completed jobs
  // are folded into the gallery (and dismissed); in-flight/failed ones show in
  // the jobs tray.
  const { derivationJobs, cancelDerivation, cancelAllDerivations, dismissDerivation, approveDerivation, regenerateDerivationEdit, exportDataset, captionClip } = useLoraTraining()

  // Single-clip auto-caption for the detail modal. Mirrors the bulk captioner
  // but writes through onCaptionChange so the modal reflects it immediately.
  const onAutoCaption = useCallback(
    async (id: string): Promise<{ ok: boolean; error?: string }> => {
      const clip = clips.find((c) => c.id === id)
      if (!clip) return { ok: false, error: 'Clip not found' }
      const res = await captionClip(clip.localPath, false)
      if (res.ok) {
        onCaptionChange(id, res.data)
        return { ok: true }
      }
      return { ok: false, error: res.error }
    },
    [clips, captionClip, onCaptionChange],
  )

  // Rebuild an example with a new reference/target split: references become
  // loose, then each target re-references the new reference set.
  const onReassignRoles = useCallback(
    async (referenceIds: string[], targetIds: string[]) => {
      if (referenceIds.length === 0 || targetIds.length === 0) return
      await onUngroup(referenceIds)
      for (const targetId of targetIds) await onGroupPair(targetId, referenceIds)
    },
    [onUngroup, onGroupPair],
  )
  const { addToast } = useToast()

  // Bulk flip: in every selected example, swap which clips are references and
  // which are targets (old targets become references, old references become
  // targets). Groups whose references aren't all present are skipped — there's
  // nothing resolved to promote into a target.
  const handleReverseRoles = useCallback(() => {
    const flippable = selectedPairGroups.filter(
      (g) => g.controls.length > 0 && g.targets.length > 0,
    )
    if (flippable.length === 0) return
    // Snapshot ids up front; each flip is independent (groups never share a
    // clip), so applying them sequentially against the live store is safe.
    void (async () => {
      for (const g of flippable) {
        const newRefs = g.targets.map((t) => t.id)
        const newTargets = g.controls.map((c) => c.id)
        await onReassignRoles(newRefs, newTargets)
      }
    })()
    const skipped = selectedPairGroups.length - flippable.length
    addToast({
      title: `Reversed ${flippable.length} example${flippable.length === 1 ? '' : 's'}`,
      description: skipped > 0 ? `${skipped} skipped (missing inputs)` : 'Inputs ↔ outputs swapped',
      variant: 'success',
    })
  }, [selectedPairGroups, onReassignRoles, addToast])
  const [exportOpen, setExportOpen] = useState(false)
  const [exportBusy, setExportBusy] = useState(false)
  const [trashOpen, setTrashOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const rejectedCount = useMemo(() => clips.filter((c) => c.triage === 'reject').length, [clips])

  // User-facing "remove" soft-deletes into the recycle bin and offers a quick
  // Undo so an accidental delete is one click to recover.
  const handleRemove = useCallback(
    (ids: string[]) => {
      if (!ids.length) return
      const removed = [...ids]
      onRemoveSelected(removed)
      addToast({
        title: `Moved ${removed.length} clip${removed.length === 1 ? '' : 's'} to Trash`,
        description: 'Restore anytime from the Trash.',
        actionLabel: 'Undo',
        onAction: () => onRestoreClips(removed),
      })
    },
    [onRemoveSelected, onRestoreClips, addToast],
  )

  const handleExport = useCallback(
    async ({ format, includeRejected, profileId, components, icLora }: DatasetExportOptions) => {
      const electron = window.electronAPI
      let destPath: string | null = null
      if (format === 'zip') {
        const safe = dataset.name.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._]+|[._]+$/g, '') || 'lora-dataset'
        destPath = (await electron?.showSaveDialog?.({
          title: 'Export dataset',
          defaultPath: `${safe}.zip`,
          filters: [{ name: 'Zip archive', extensions: ['zip'] }],
        })) ?? null
      } else {
        destPath = (await electron?.showOpenDirectoryDialog?.({ title: 'Choose a folder to export into' })) ?? null
      }
      if (!destPath) return
      setExportBusy(true)
      const result = await exportDataset(dataset.id, {
        destPath,
        format,
        includeRejected,
        profileId,
        includeConfig: components.config,
        includeReadme: components.readme,
        includeManifest: components.manifest,
        includeModelCard: components.modelCard,
        ...(icLora
          ? {
              icLoraFps: icLora.fps,
              icLoraShortSide: icLora.shortSide,
              icLoraBucketFrames: icLora.bucketFrames,
              forbiddenCaptionWords: icLora.forbiddenWords,
            }
          : {}),
      })
      setExportBusy(false)
      if (!result.ok) {
        addToast({ title: 'Export failed', description: result.error, variant: 'warning' })
        return
      }
      setExportOpen(false)
      const { exportPath, clipCount, droppedPairs } = result.data
      const unit = isIcLora ? 'pair' : 'clip'
      const droppedNote = droppedPairs.length > 0 ? ` · ${droppedPairs.length} dropped` : ''
      if (droppedPairs.length > 0) {
        // Surface the first few drop reasons so the user knows what to fix.
        addToast({
          title: `${droppedPairs.length} ${unit}${droppedPairs.length === 1 ? '' : 's'} dropped on export`,
          description: droppedPairs.slice(0, 4).join('\n') + (droppedPairs.length > 4 ? '\n…' : ''),
          variant: 'warning',
        })
      }
      addToast({
        title: 'Dataset exported',
        description: `${clipCount} ${unit}${clipCount === 1 ? '' : 's'}${droppedNote} → ${exportPath}`,
        variant: 'success',
        actionLabel: 'Reveal',
        onAction: () => window.electronAPI?.showItemInFolder?.({ filePath: exportPath }),
      })
    },
    [dataset.id, dataset.name, exportDataset, addToast, isIcLora],
  )
  const foldedJobsRef = useRef<Set<string>>(new Set())
  const [reviewOpen, setReviewOpen] = useState(false)
  const datasetJobs = useMemo(
    () => derivationJobs.filter((j) => j.datasetId === dataset.id),
    [derivationJobs, dataset.id],
  )
  // The edit phase the review modal streams through (queued -> editing -> ready).
  const reviewSessionJobs = useMemo(
    () => datasetJobs.filter((j) => j.requireReview && ['pending', 'editing', 'review'].includes(j.status)),
    [datasetJobs],
  )
  const reviewReadyCount = useMemo(
    () => reviewSessionJobs.filter((j) => j.status === 'review').length,
    [reviewSessionJobs],
  )
  // "Before" thumbnail for the review modal: the exact frame that was edited
  // (persisted on the job), so it matches the "after". Falls back to the
  // source clip's poster for older jobs that predate source-frame capture.
  const sourcePosterFor = useCallback(
    (job: DerivationJob) => {
      if (job.sourceFramePath) return job.sourceFramePath
      const src = job.sourceClipId ? clips.find((c) => c.id === job.sourceClipId) : null
      return src?.posterPath ?? null
    },
    [clips],
  )

  useEffect(() => {
    let cancelled = false
    const foldCompleted = async () => {
      for (const job of datasetJobs) {
        if (cancelled || job.status !== 'completed' || !job.derivedPath || foldedJobsRef.current.has(job.id)) continue
        foldedJobsRef.current.add(job.id)
        const base = {
          id: job.id,
          localPath: job.derivedPath,
          sourcePath: job.derivedPath,
          origin: 'ai_derived' as const,
          caption: job.caption,
          probe: job.probe ?? null,
          durationSeconds: job.probe?.durationSeconds ?? null,
        }
        let persisted = false
        if (job.direction === 'frame_edit') {
          persisted = await onAddStill({
            id: job.id,
            framePath: job.derivedPath,
            caption: job.caption,
            driverPath: job.driverPath,
            probe: job.probe,
          })
        } else if (job.direction === 'reference' && job.sourceClipId) {
          persisted = await onCreateReference(job.sourceClipId, { ...base, referencePath: null })
        } else if (job.direction === 'variant') {
          persisted = await onCreatePair({ ...base, referencePath: null })
        } else {
          const sourceClip = job.sourceClipId ? clips.find((c) => c.id === job.sourceClipId) : null
          const refPath = isIcLora ? (job.referencePath || sourceClip?.localPath || null) : null
          persisted = await onCreatePair({ ...base, referencePath: refPath })
        }
        if (persisted) {
          await dismissDerivation(job.id)
        } else {
          // Keep the durable completion record so a later poll/reopen retries
          // the handoff; deterministic clip ids make that retry idempotent.
          foldedJobsRef.current.delete(job.id)
        }
      }
    }
    void foldCompleted()
    return () => { cancelled = true }
  }, [datasetJobs, onCreatePair, onCreateReference, onAddStill, dismissDerivation, clips, isIcLora])

  // Right-click selects the clip (unless it's already part of the selection,
  // so bulk actions keep operating on the whole set) and opens the menu.
  const openContextMenu = (id: string, x: number, y: number) => {
    if (!selectedIds.has(id)) setSelection([id])
    setBgMenu(null)
    setMenu({ x, y, clipId: id })
  }

  // Right-click on empty gallery space → import / selection actions.
  const openBgMenu = (e: React.MouseEvent) => {
    e.preventDefault()
    setMenu(null)
    setBgMenu({ x: e.clientX, y: e.clientY })
  }
  const bgMenuItems: ContextMenuItem[] = []
  if (bgMenu) {
    if (editable) bgMenuItems.push({ label: 'Import media…', icon: Plus, onClick: onAddClips })
    if (editable) bgMenuItems.push({ label: 'Browse Pexels…', icon: Images, onClick: onBrowsePexels })
    bgMenuItems.push({ label: 'Select all shown', icon: CheckSquare, onClick: selectAllShown, disabled: visibleClips.length === 0 })
    if (selectedCount > 0) bgMenuItems.push({ label: 'Clear selection', icon: X, onClick: clearSelection })
  }

  // Single-clip actions shared by the detail view, inspector, and context
  // menu. Each closes the large detail view (if open) before launching its
  // dedicated modal, so we never stack two big surfaces.
  const startEdit = (id: string) => { setDetailClipId(null); setEditorClipId(id) }
  const startFrameEdit = (id: string) => { setDetailClipId(null); setFrameEditClipId(id) }
  // Both "generate target" (a paired IC-LoRA example) and "variant" (a
  // standalone, ungrouped clip) flow through the single staged wizard; the
  // `mode` decides whether the result is grouped.
  const startGenerate = (id: string, mode: 'target' | 'variant' = 'target') => {
    setDetailClipId(null)
    setWizardMode(mode)
    setWizardClipId(id)
  }
  const startTarget = (id: string) => startGenerate(id, 'target')
  const startVariant = (id: string) => startGenerate(id, 'variant')
  const removeOne = (id: string) => { setDetailClipId(null); handleRemove([id]) }

  const menuMulti = selectedCount > 1
  const menuClip = menu ? clips.find((c) => c.id === menu.clipId) ?? null : null
  const menuIsImage = menuClip?.kind === 'image'
  const menuItems: ContextMenuItem[] = []
  if (menu && menuClip) {
    if (!menuMulti) {
      menuItems.push({ label: 'Open', icon: Maximize2, onClick: () => setDetailClipId(menu.clipId) })
    }
    if (editable && !menuMulti && !menuIsImage) {
      menuItems.push({ label: 'Edit (trim & crop)…', icon: Pencil, onClick: () => setEditorClipId(menu.clipId) })
      menuItems.push({ label: 'Frame edit (AI)…', icon: ImageIcon, onClick: () => setFrameEditClipId(menu.clipId) })
    }
    menuItems.push({
      label: menuMulti ? `Auto-caption ${selectedCount} clips` : 'Auto-caption',
      icon: MessageSquare,
      onClick: () => onCaptionSelected(selectedList),
      disabled: busyAction,
    })
    if (menuMulti) {
      menuItems.push({ label: `Edit captions (${selectedCount})…`, icon: MessageSquare, onClick: () => setBulkCaptionOpen(true) })
    }
    if (editable) {
      menuItems.push({ type: 'separator' })
      const genVerb = isIcLora ? 'example' : 'variant'
      if (menuMulti) {
        menuItems.push({ label: `Frame edit ${selectedCount} clips (AI)…`, icon: ImageIcon, onClick: () => setBulkFrameOpen(true) })
        menuItems.push({
          label: `Generate ${selectedCount} ${genVerb}s…`,
          icon: Sparkles,
          onClick: () => setWizardBatchOpen(true),
          disabled: menuIsImage && driverChoices.length === 0,
        })
        if (isIcLora) {
          menuItems.push({ label: `Group into an example (${selectedCount})…`, icon: Link2, onClick: () => setGroupModalOpen(true) })
        }
      } else {
        menuItems.push({
          label: `Generate ${genVerb}…`,
          icon: Sparkles,
          onClick: () => startGenerate(menu.clipId),
          disabled: menuIsImage && driverChoices.length === 0,
        })
      }
      if (hasPairedSelection) {
        menuItems.push({
          label: selectedPairGroups.length > 1 ? `Ungroup ${selectedPairGroups.length} examples` : 'Ungroup',
          icon: Unlink,
          onClick: handleUngroup,
        })
        if (flippablePairCount > 0) {
          menuItems.push({
            label:
              selectedPairGroups.length > 1
                ? `Reverse ${selectedPairGroups.length} examples (input ↔ output)`
                : 'Reverse (input ↔ output)',
            icon: ArrowLeftRight,
            onClick: handleReverseRoles,
            disabled: busyAction,
          })
        }
      }
      if (selHasVideo) {
        menuItems.push({ type: 'separator' })
        menuItems.push({
          label: 'Crop',
          icon: Crop,
          children: CROP_RATIOS.map((a) => ({
            label: a.label,
            onClick: () => onCropSelected(selectedList, a.ratio),
          })),
        })
        menuItems.push({ label: 'Trim…', icon: Scissors, onClick: () => setBulkTrimOpen(true), disabled: busyAction })
        menuItems.push({
          label: 'Normalize',
          icon: Maximize2,
          children: NORMALIZE_CHOICES.map((n) => ({
            label: n.label,
            onClick: () => onNormalizeSelected(selectedList, n.fps),
          })),
        })
        if (canSplit) {
          menuItems.push({ label: 'Split into scenes', icon: Scissors, onClick: () => onSceneSplitSelected(selectedList), disabled: busyAction })
        }
      }
      menuItems.push({ type: 'separator' })
      const selTriage = selectedClips.map((c) => c.triage)
      const allKeep = selTriage.length > 0 && selTriage.every((t) => t === 'keep')
      const allReject = selTriage.length > 0 && selTriage.every((t) => t === 'reject')
      const allHoldout = selTriage.length > 0 && selTriage.every((t) => t === 'holdout')
      menuItems.push({
        label: allKeep ? 'Clear keep flag' : menuMulti ? `Mark ${selectedCount} as keep` : 'Mark as keep',
        icon: ThumbsUp,
        onClick: () => onSetTriage(selectedList, allKeep ? null : 'keep'),
      })
      menuItems.push({
        label: allReject ? 'Clear reject flag' : menuMulti ? `Reject ${selectedCount} clips` : 'Reject',
        icon: ThumbsDown,
        onClick: () => onSetTriage(selectedList, allReject ? null : 'reject'),
      })
      menuItems.push({
        label: allHoldout ? 'Clear holdout flag' : menuMulti ? `Hold out ${selectedCount} clips` : 'Hold out (validation only)',
        icon: FlaskConical,
        onClick: () => onSetTriage(selectedList, allHoldout ? null : 'holdout'),
      })
      menuItems.push({ type: 'separator' })
      menuItems.push({
        label: menuMulti ? `Move ${selectedCount} clips to Trash` : 'Move to Trash',
        icon: Trash2,
        danger: true,
        onClick: () => handleRemove(selectedList),
      })
    }
    // Global actions (independent of which clip was clicked).
    menuItems.push({ type: 'separator' })
    if (editable) {
      menuItems.push({ label: 'Import media…', icon: Plus, onClick: onAddClips })
    }
    // Copying clips into a new draft never touches the source collection, so
    // it's available regardless of editability.
    menuItems.push({
      label: menuMulti ? `New collection from ${selectedCount} clips` : 'New collection from this clip',
      icon: FolderPlus,
      onClick: handleCreateCollection,
    })
  }

  // Gallery keyboard shortcuts. Ignored while typing in a field so they don't
  // fight the inspector inputs or open modals.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) {
        return
      }
      if (e.key === 'Escape') {
        if (selectedIds.size > 0) {
          clearSelection()
          selectionAnchor.current = null
        }
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'a') {
        if (visibleClips.length > 0) {
          e.preventDefault()
          selectAllShown()
        }
      } else if ((e.key === 'Backspace' || e.key === 'Delete') && selectedIds.size > 0 && editable) {
        e.preventDefault()
        handleRemove([...selectedIds])
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedIds, visibleClips.length, clearSelection, selectAllShown, handleRemove, editable])

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    dragDepth.current = 0
    setDragOver(false)
    if (!editable) return
    const paths: string[] = []
    for (const file of Array.from(e.dataTransfer.files)) {
      if (!file.type.startsWith('video/') && !file.type.startsWith('image/')) continue
      const p = window.electronAPI?.getPathForFile(file)
      if (p) paths.push(p)
    }
    if (paths.length) onDropFiles(paths)
  }

  return (
    <div className="flex-1 flex min-w-0">
      <div className="flex-1 flex flex-col min-w-0 relative">
        <div className="flex items-center justify-between gap-3 border-b border-zinc-800 px-3 py-2 lg:hidden">
          <span className="min-w-0 truncate text-xs font-medium text-zinc-300">{dataset.name}</span>
          <button
            ref={inspectorToggleRef}
            type="button"
            onClick={() => setInspectorOpen(true)}
            aria-label="Open collection inspector"
            aria-expanded={inspectorOpen}
            aria-controls="lora-collection-inspector"
            className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-zinc-700 px-2.5 py-1.5 text-xs text-zinc-300 hover:border-zinc-600 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          >
            <SlidersHorizontal className="h-3.5 w-3.5" />
            Details
          </button>
        </div>
        <GalleryToolbar
          search={search}
          onSearchChange={setSearch}
          facets={facets}
          activeFacets={activeFacets}
          onFacetClick={onFacetClick}
          onClearFacets={onClearFacets}
          density={density}
          onDensityChange={setDensity}
          layout={layout}
          onLayoutChange={onLayoutChange}
          sort={sort}
          onSortChange={setSort}
          onAddClips={editable ? onAddClips : undefined}
          pairCount={isIcLora ? datasetPairs.length : 0}
          pairView={pairView}
          onPairViewChange={onPairViewChange}
          pairFilter={pairFilter}
          onPairFilterChange={onPairFilterChange}
          totalCount={clips.length}
          visibleCount={visibleClips.length}
          selectedCount={selectedCount}
          onSelectAll={selectAllShown}
          onClearSelection={clearSelection}
          readiness={readiness}
          primary={life.primary}
          // `view-run` is a navigation action, not a stage-advancing one — it
          // must stay clickable while a run is in progress (that's the whole
          // point), so exclude it from the `life.busy` gate. `busyAction`
          // (an in-flight clip edit) still disables it.
          primaryBusy={busyAction || (life.busy && life.primary?.kind !== 'view-run')}
          primaryDisabled={primaryDisabled}
          onPrimary={onPrimary}
        />

        {isIcLora ? (
          <div className="px-4 py-1.5 border-b border-blue-500/25 bg-blue-500/[0.06] flex items-center gap-2">
            <Link2 className="h-3.5 w-3.5 text-blue-300 shrink-0" />
            <p className="text-[11px] font-medium text-blue-100 flex items-center gap-1.5 min-w-0">
              <span className="truncate">
                IC-LoRA · teach a transformation <span className="text-blue-300/70 font-normal">input → output</span>
              </span>
              <Tooltip
                side="bottom"
                content={
                  <span className="block max-w-[18rem] whitespace-normal">
                    Group clips into examples (one or more inputs → one output), or Generate examples with AI. The model learns to produce the output.
                  </span>
                }
              >
                <span className="shrink-0 text-blue-300/60 hover:text-blue-200 cursor-help">
                  <Info className="h-3.5 w-3.5" />
                </span>
              </Tooltip>
            </p>
            <div className="flex-1" />
            {editable && selectedCount > 1 && (
              <button
                onClick={() => setGroupModalOpen(true)}
                className="text-[11px] shrink-0 px-2.5 py-1 rounded-md bg-blue-600 hover:bg-blue-500 text-white flex items-center gap-1.5"
              >
                <Link2 className="h-3 w-3" /> Group into example
              </button>
            )}
          </div>
        ) : (
          <div className="px-4 py-1.5 border-b border-zinc-800 bg-zinc-900/40 flex items-center gap-2">
            <Images className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
            <p className="text-[11px] font-medium text-zinc-300 flex items-center gap-1.5 min-w-0">
              <span className="truncate">Standard LoRA · learn a look or subject</span>
              <Tooltip
                side="bottom"
                content={
                  <span className="block max-w-[18rem] whitespace-normal">
                    Add varied clips + captions of the same concept. No before/after pairing needed.
                  </span>
                }
              >
                <span className="shrink-0 text-zinc-500 hover:text-zinc-300 cursor-help">
                  <Info className="h-3.5 w-3.5" />
                </span>
              </Tooltip>
            </p>
          </div>
        )}

        {/* Single slim alerts strip — only shown when something is actionable. */}
        {((!credentialsReady && life.primary?.needsCredentials) || reviewSessionJobs.length > 0) && (
          <div className="px-4 py-1.5 border-b border-amber-500/20 bg-amber-500/[0.06] flex items-center gap-3 flex-wrap">
            {reviewSessionJobs.length > 0 && (
              <div className="flex items-center gap-2 min-w-0">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400/70" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-400" />
                </span>
                <p className="text-[11px] text-amber-100 min-w-0">
                  {reviewReadyCount > 0 ? (
                    <>
                      <span className="font-medium">{reviewReadyCount} {reviewReadyCount === 1 ? 'edit' : 'edits'} ready to review</span>
                      <span className="text-amber-300/60"> — approve before the videos generate</span>
                    </>
                  ) : (
                    <span className="font-medium">Editing {reviewSessionJobs.length} {reviewSessionJobs.length === 1 ? 'frame' : 'frames'}…</span>
                  )}
                </p>
                <button
                  onClick={() => setReviewOpen(true)}
                  className="text-[11px] shrink-0 px-2.5 py-0.5 rounded-md bg-amber-500/90 hover:bg-amber-400 text-amber-950 font-medium"
                >
                  Review edits
                </button>
              </div>
            )}
            {reviewSessionJobs.length > 0 && !credentialsReady && life.primary?.needsCredentials && (
              <span className="h-3 w-px bg-amber-500/25" />
            )}
            {!credentialsReady && life.primary?.needsCredentials && (
              <span className="text-[11px] text-amber-300/90 flex items-center gap-1.5">
                <AlertTriangle className="h-3.5 w-3.5" />
                {provider === 'local'
                  ? 'Finish local GPU setup (provider menu, top right) to train on this machine.'
                  : 'Connect a cloud GPU in Settings to upload and train.'}
              </span>
            )}
          </div>
        )}

        <main
          className={`flex-1 overflow-y-auto p-4 relative ${dragOver ? 'ring-2 ring-inset ring-blue-500/60' : ''}`}
          onDragEnter={(e) => {
            if (!editable) return
            e.preventDefault()
            dragDepth.current += 1
            setDragOver(true)
          }}
          onDragOver={(e) => {
            if (editable) e.preventDefault()
          }}
          onDragLeave={() => {
            dragDepth.current = Math.max(0, dragDepth.current - 1)
            if (dragDepth.current === 0) setDragOver(false)
          }}
          onDrop={handleDrop}
          onContextMenu={openBgMenu}
        >
          {dragOver && (
            <div className="absolute inset-2 z-10 rounded-lg border-2 border-dashed border-blue-500/60 bg-blue-500/5 flex items-center justify-center pointer-events-none">
              <span className="text-sm text-blue-300">Drop videos or images to import them</span>
            </div>
          )}
          {clips.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center gap-4 px-6">
              <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-blue-500/20 to-blue-500/20 border border-blue-500/30 flex items-center justify-center">
                <ImageIcon className="h-6 w-6 text-blue-300" />
              </div>
              <div>
                <p className="text-sm font-medium text-zinc-300">This collection is empty</p>
                <p className="text-xs text-zinc-500 mt-1 max-w-sm leading-relaxed">
                  Import videos or images, drag-and-drop files here, or send assets from Gen Space with &ldquo;To LoRA&rdquo;.
                  Not sure what to gather? Check the recipes.
                </p>
              </div>
              {editable && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={onAddClips}
                    className="text-xs px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white flex items-center gap-1.5"
                  >
                    <Plus className="h-3.5 w-3.5" /> Import media
                  </button>
                  <button
                    onClick={onBrowsePexels}
                    className="text-xs px-3.5 py-2 rounded-lg border border-zinc-700 text-zinc-200 hover:bg-zinc-800 flex items-center gap-1.5"
                  >
                    <Images className="h-3.5 w-3.5" /> Browse Pexels
                  </button>
                  <button
                    onClick={onOpenRecipes}
                    className="text-xs px-3.5 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600 flex items-center gap-1.5"
                  >
                    <BookOpen className="h-3.5 w-3.5" /> Dataset recipes
                  </button>
                </div>
              )}
            </div>
          ) : nothingShown ? (
            <div className="h-full flex items-center justify-center text-sm text-zinc-600">
              No clips match the current filter.
            </div>
          ) : layout === 'list' ? (
            <ClipListTable
              groups={shownPairs}
              loose={shownLoose}
              grouped={isIcLora && shownPairs.length > 0}
              pairBadges={pairBadges}
              selectedIds={selectedIds}
              editable={editable}
              sort={listSort}
              onSortChange={onListSort}
              onSelect={handleSelect}
              onSelectGroup={handleSelectGroup}
              onOpen={setDetailClipId}
              onContextMenu={openContextMenu}
              onCaptionChange={onCaptionChange}
              onSelectAll={selectAllShown}
              onClearSelection={clearSelection}
            />
          ) : (
            <div className="space-y-4">
              {shownPairs.length > 0 && (
                <section>
                  {sectioned && (
                    <SectionHeader
                      label="Examples"
                      count={shownPairs.length}
                      hint="Input(s) → output the model learns to produce"
                    />
                  )}
                  {pairView === 'sideBySide' ? (
                    <div className="flex flex-col gap-2">
                      {shownPairs.map((group) => (
                        <PairCard
                          key={`set:${group.id}`}
                          group={group}
                          selectedIds={selectedIds}
                          onSelect={handleSelectGroup}
                          onOpen={setDetailClipId}
                          onContextMenu={openContextMenu}
                          layout="sideBySide"
                        />
                      ))}
                    </div>
                  ) : pairView === 'flat' ? (
                    <div className={`grid ${DENSITY_GRID[density]} gap-2`}>
                      {shownPairs.flatMap((group) =>
                        [...group.controls, ...group.targets].map((clip) => {
                          const badge = pairBadges.get(clip.id) ?? null
                          return (
                            <ClipCard
                              key={clip.id}
                              clip={clip}
                              selected={selectedIds.has(clip.id)}
                              onSelect={handleSelect}
                              onOpen={setDetailClipId}
                              onContextMenu={openContextMenu}
                              pairBadge={badge}
                              highlighted={badge != null && hoveredSet === badge.index}
                              onHoverSet={setHoveredSet}
                              onTriage={editable ? (id, t) => onSetTriage([id], t) : undefined}
                            />
                          )
                        }),
                      )}
                    </div>
                  ) : (
                    <div className={`grid ${DENSITY_GRID[density]} gap-2`}>
                      {shownPairs.map((group) => (
                        <PairCard
                          key={`set:${group.id}`}
                          group={group}
                          selectedIds={selectedIds}
                          onSelect={handleSelectGroup}
                          onOpen={setDetailClipId}
                          onContextMenu={openContextMenu}
                          layout="combined"
                        />
                      ))}
                    </div>
                  )}
                </section>
              )}

              {shownLoose.length > 0 && (
                <section>
                  {sectioned && <SectionHeader label="Ungrouped clips" count={shownLoose.length} />}
                  <div className={`grid ${DENSITY_GRID[density]} gap-2`}>
                    {shownLoose.map((clip) => (
                      <ClipCard key={clip.id} clip={clip} selected={selectedIds.has(clip.id)} onSelect={handleSelect} onOpen={setDetailClipId} onContextMenu={openContextMenu} onTriage={editable ? (id, t) => onSetTriage([id], t) : undefined} />
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}
        </main>

        {selectedCount > 0 && (
          <BulkActionsPill
            selectedCount={selectedCount}
            editable={editable}
            isIcLora={isIcLora}
            busyAction={busyAction}
            onCaption={() => onCaptionSelected(selectedList)}
            onCaptionTools={() => setBulkCaptionOpen(true)}
            onFrameEdit={() => setBulkFrameOpen(true)}
            onGenerate={() => setWizardBatchOpen(true)}
            onCrop={(ratio) => onCropSelected(selectedList, ratio)}
            onTrim={() => setBulkTrimOpen(true)}
            onNormalize={(fps) => onNormalizeSelected(selectedList, fps)}
            onGroup={handleGroup}
            onUngroup={handleUngroup}
            onReverseRoles={handleReverseRoles}
            hasPaired={hasPairedSelection}
            canReverse={flippablePairCount > 0}
            canSplit={canSplit}
            onSceneSplit={() => onSceneSplitSelected(selectedList)}
            onSegment={(s) => onSegmentSelected(selectedList, s)}
            onKeep={() => onSetTriage(selectedList, 'keep')}
            onReject={() => onSetTriage(selectedList, 'reject')}
            onHoldout={() => onSetTriage(selectedList, 'holdout')}
            onNewCollection={handleCreateCollection}
            onRemove={() => handleRemove(selectedList)}
            onClear={clearSelection}
          />
        )}

        {reviewOpen && (
          <EditReviewModal
            jobs={reviewSessionJobs}
            getSourcePoster={sourcePosterFor}
            onApprove={approveDerivation}
            onRegenerate={(id) => void regenerateDerivationEdit(id)}
            onDiscard={cancelDerivation}
            onApproveAll={() =>
              reviewSessionJobs.filter((j) => j.status === 'review').forEach((j) => void approveDerivation(j.id))
            }
            onCancelAll={() => {
              void cancelAllDerivations(dataset.id)
              setReviewOpen(false)
            }}
            onClose={() => setReviewOpen(false)}
          />
        )}
      </div>

      {inspectorOpen && (
        <button
          type="button"
          aria-label="Close collection inspector"
          onClick={() => {
            setInspectorOpen(false)
            inspectorToggleRef.current?.focus()
          }}
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
        />
      )}
      <aside
        id="lora-collection-inspector"
        role={inspectorOpen ? 'dialog' : 'complementary'}
        aria-modal={inspectorOpen ? 'true' : undefined}
        aria-label="Collection inspector"
        className={`fixed inset-y-0 right-0 z-50 flex w-[min(24rem,92vw)] flex-col border-l border-zinc-800 bg-background shadow-2xl transition-transform lg:static lg:z-auto lg:block lg:w-72 lg:flex-shrink-0 lg:translate-x-0 lg:shadow-none ${
          inspectorOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {inspectorOpen && (
          <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-3 py-2 lg:hidden">
            <span className="text-sm font-semibold text-white">Collection details</span>
            <button
              type="button"
              autoFocus
              onClick={() => {
                setInspectorOpen(false)
                inspectorToggleRef.current?.focus()
              }}
              aria-label="Close collection inspector"
              className="flex h-8 w-8 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}
        <DatasetInspector
          datasetId={dataset.id}
          datasetName={dataset.name}
          datasetType={dataset.type}
          triggerWord={triggerWord}
          onSetType={onSetType}
          life={life}
          clips={clips}
          selectedIds={selectedIds}
          editable={editable}
          errorText={userFacingDatasetError(dataset.error)}
          onRename={onRename}
          onSetTrigger={onSetTrigger}
          onDelete={onDelete}
          onArchive={onArchive}
          onExport={() => setExportOpen(true)}
          trashCount={trashedClips.length}
          onOpenTrash={() => setTrashOpen(true)}
          singleClip={singleClip}
          canMakePair={driverChoices.length > 0}
          onCaptionChange={onCaptionChange}
          onAutoCaption={(id) => onCaptionSelected([id])}
          onOpenClip={setDetailClipId}
          onEditClip={startEdit}
          onFrameEditClip={startFrameEdit}
          onMakePairClip={startTarget}
          onVariantClip={startVariant}
          onCancelPreprocess={onCancelPreprocess}
          onCancelUpload={onCancelUpload}
          onViewSettings={() => setSettingsOpen(true)}
        />
      </aside>

      {editorClip && (
        <ClipEditor
          clip={editorClip}
          busy={busyAction}
          onClose={() => setEditorClipId(null)}
          onApply={(edits) => {
            onApplyEdit(editorClip.id, edits)
            setEditorClipId(null)
          }}
          onRevert={() => {
            onRevertEdit(editorClip.id)
            setEditorClipId(null)
          }}
        />
      )}

      {wizardClip && (
        <CreateTargetWizard
          clip={wizardClip}
          mode={wizardMode}
          drivers={driverChoices}
          datasetId={dataset.id}
          datasetType={dataset.type}
          initialPreview={wizardClip.editPreview}
          onPreviewChange={(preview) => setEditPreview(wizardClip.id, preview)}
          onAttachStillInput={async (sourceId, { framePath, caption }) => {
            await onCreateReference(sourceId, {
              localPath: framePath,
              sourcePath: framePath,
              caption,
              origin: 'ai_derived',
              referencePath: null,
              probe: null,
              durationSeconds: null,
            })
          }}
          onClose={() => setWizardClipId(null)}
          onSubmitted={(_count, requiresReview) => {
            setWizardClipId(null)
            if (requiresReview) setReviewOpen(true)
          }}
        />
      )}

      {wizardBatchOpen && selectedClips.length > 1 && (
        <CreateTargetWizard
          clip={selectedClips[0]}
          batchClips={selectedClips}
          mode="target"
          drivers={driverChoices}
          datasetId={dataset.id}
          datasetType={dataset.type}
          onAttachStillInput={async (sourceId, { framePath, caption }) => {
            await onCreateReference(sourceId, {
              localPath: framePath,
              sourcePath: framePath,
              caption,
              origin: 'ai_derived',
              referencePath: null,
              probe: null,
              durationSeconds: null,
            })
          }}
          onClose={() => setWizardBatchOpen(false)}
          onSubmitted={(_count, requiresReview) => {
            setWizardBatchOpen(false)
            if (requiresReview) setReviewOpen(true)
          }}
        />
      )}

      {frameEditClip && (
        <FrameEditModal
          clip={frameEditClip}
          datasetId={dataset.id}
          onClose={() => setFrameEditClipId(null)}
        />
      )}

      {bulkFrameOpen && selectedClips.length > 0 && (
        <FrameEditModal
          clip={selectedClips[0]}
          batchClips={selectedClips}
          datasetId={dataset.id}
          onClose={() => setBulkFrameOpen(false)}
        />
      )}

      {bulkCaptionOpen && selectedClips.length > 0 && (
        <BulkCaptionModal
          clips={selectedClips}
          isIcLora={isIcLora}
          onClose={() => setBulkCaptionOpen(false)}
          onApply={onApplyCaptions}
        />
      )}

      {bulkTrimOpen && selectedClips.length > 0 && (
        <BulkTrimModal
          clips={selectedClips}
          busy={busyAction}
          onClose={() => setBulkTrimOpen(false)}
          onApply={(plan) => {
            onTrimSelected(selectedList, plan)
            setBulkTrimOpen(false)
          }}
        />
      )}

      {groupModalOpen && selectedClips.length > 1 && (
        <GroupPairModal
          clips={selectedClips}
          onClose={() => setGroupModalOpen(false)}
          onConfirm={onGroupPair}
        />
      )}

      {detailClip && (
        <ClipDetailModal
          clip={detailClip}
          clips={visibleClips}
          editable={editable}
          canMakePair={driverChoices.length > 0}
          onClose={() => setDetailClipId(null)}
          onNavigate={setDetailClipId}
          onCaptionChange={onCaptionChange}
          onAutoCaption={onAutoCaption}
          onReassignRoles={editable ? onReassignRoles : undefined}
          onUngroup={editable ? onUngroup : undefined}
          onEdit={startEdit}
          onFrameEdit={startFrameEdit}
          onMakePair={startTarget}
          onVariant={startVariant}
          onRemove={removeOne}
        />
      )}

      {exportOpen && (
        <DatasetExportModal
          datasetName={dataset.name}
          totalClips={clips.length}
          rejectedCount={rejectedCount}
          isIcLora={isIcLora}
          busy={exportBusy}
          onClose={() => setExportOpen(false)}
          onExport={handleExport}
        />
      )}

      {trashOpen && (
        <TrashModal
          clips={trashedClips}
          onRestore={(ids) => onRestoreClips(ids)}
          onPurge={(ids) => {
            onPurgeClips(ids)
            if (trashedClips.length <= ids.length) setTrashOpen(false)
          }}
          onClose={() => setTrashOpen(false)}
        />
      )}

      {settingsOpen && (
        <TrainingSettingsModal
          preprocessed={life.preprocessed}
          training={life.training}
          onClose={() => setSettingsOpen(false)}
        />
      )}

      {menu && menuItems.length > 0 && (
        <ClipContextMenu x={menu.x} y={menu.y} items={menuItems} onClose={() => setMenu(null)} />
      )}
      {bgMenu && bgMenuItems.length > 0 && (
        <ClipContextMenu x={bgMenu.x} y={bgMenu.y} items={bgMenuItems} onClose={() => setBgMenu(null)} />
      )}
    </div>
  )
}
