import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { ArrowLeft, BookOpen, Check, ChevronDown, Cloud, Cpu, MemoryStick, PanelLeft, Plus, Sparkles, Wand2, X } from 'lucide-react'
import { loraLog } from '../../lib/lora-log'
import { useView } from '../../contexts/ViewContext'
import { useAppSettings, type LoraProvider } from '../../contexts/AppSettingsContext'
import { useProjects } from '../../contexts/ProjectContext'
import {
  useLoraTraining,
  type ClipEdits,
  type ClipInput,
  type LocalTrainerEligibility,
  type LoraDataset,
  type LoraDatasetType,
  type LoraPreprocessed,
  type LoraTrainingJob,
} from '../../contexts/LoraTrainingContext'
import {
  CreateDatasetModal,
  PreprocessModal,
  TrainPipelineModal,
  StartTrainingModal,
  RunpodGpuRecoveryModal,
  TrainingLogsModal,
} from '../../components/lora/LoraModals'
import { TrainingProfileEditor } from '../../components/lora/TrainingProfileEditor'
import { LocalTrainingSetupWizard } from '../../components/lora/LocalTrainingSetupWizard'
import { confirmAction } from '../../components/ui/confirm-dialog'
import { WslMemoryOptimizer } from '../../components/lora/WslMemoryOptimizer'
import {
  StudioStoreProvider,
  clipFromDataset,
  clipFromInput,
  createStudioStore,
  toClipInput,
  type ClipTriage,
  type StudioStoreApi,
} from '../studio/studio-store'
import { centeredCrop, snapDown32, type ClipProbeLike } from '../../lib/lora-quality'
import { useSpritePipeline } from './use-sprite-pipeline'
import { deriveLifecycle } from './lifecycle'
import { MenuBar, type MenuDefinition } from '../../components/MenuBar'
import { CollectionsSidebar } from './CollectionsSidebar'
import { GuidedTour, shouldAutoStartTour } from './help/GuidedTour'
import { DatasetRecipes } from './help/DatasetRecipes'
import { CollectionView } from './CollectionView'
import { ErrorBoundary } from '../../components/ErrorBoundary'
import { PrepRecoveryModal } from './PrepRecoveryModal'
import type { TrimPlan } from './BulkTrimModal'
import { RunView } from './RunView'
import type { PodLifecycleInfo } from './ComputePanel'
import { derivePodWork } from './compute-work'
import { AddClipsModal } from './AddClipsModal'
import { PexelsBrowser } from './PexelsBrowser'
import { UploadConfirmModal } from './UploadConfirmModal'
import { PublishWizard } from './PublishWizard'
import { LibraryView } from './LibraryView'
import { useLoraInferenceRegistry } from '../../hooks/use-lora-inference-registry'
import type { Selection } from './selection'
import {
  loadLoraUiPreferences,
  saveLoraUiPreferences,
} from '../../lib/lora-ui-persistence'

// How many clips to auto-caption at once. Each is an independent Gemini call
// the backend runs on its thread pool, so a small pool turns a serial batch
// into a few parallel rounds. Capped to stay under Gemini's per-key rate
// limits (overflow requests just wait for a free worker).
const CAPTION_CONCURRENCY = 4

// Snapshot a dataset's clips as fresh ClipInputs (path references only — no
// files are copied), so a copy/merge can seed a brand-new draft collection.
function datasetToInputs(dataset: LoraDataset): ClipInput[] {
  return dataset.clips.map((c) => toClipInput(clipFromDataset(c)))
}

function EmptyWorkspace({
  onNew,
  onStartTour,
  onOpenRecipes,
}: {
  onNew: () => void
  onStartTour: () => void
  onOpenRecipes: () => void
}) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center px-6">
      <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-blue-500/20 to-blue-500/20 border border-blue-500/30 flex items-center justify-center">
        <Wand2 className="h-7 w-7 text-blue-300" />
      </div>
      <div>
        <p className="text-base font-semibold text-white">Train your first LoRA</p>
        <p className="text-xs text-zinc-500 mt-1.5 max-w-sm leading-relaxed">
          A collection is one dataset → one LoRA. Gather clips that show your concept (or send assets from Gen Space with
          &ldquo;To LoRA&rdquo;), caption them, then train on RunPod or your local GPU.
        </p>
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={onNew}
          className="text-xs px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white flex items-center gap-1.5"
        >
          <Plus className="h-3.5 w-3.5" /> New collection
        </button>
        <button
          onClick={onStartTour}
          className="text-xs px-3.5 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600 flex items-center gap-1.5"
        >
          <Sparkles className="h-3.5 w-3.5" /> Take the tour
        </button>
        <button
          onClick={onOpenRecipes}
          className="text-xs px-3.5 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600 flex items-center gap-1.5"
        >
          <BookOpen className="h-3.5 w-3.5" /> Dataset recipes
        </button>
      </div>
    </div>
  )
}

// Top-right provider pill, now a selector. Shows the active backend ("RunPod"
// or "Local GPU") and opens a menu with both options. RunPod is always
// selectable; Local GPU is gated on `eligibility.eligible` (disabled with the
// reason as help text otherwise, and showing the detected GPU/VRAM when ready).
// Selecting an option persists it via `onSelect` (settings patch upstream).
export function ProviderSelector({
  provider,
  eligibility,
  onSelect,
  onSetup,
  onOptimizeMemory,
}: {
  provider: LoraProvider
  eligibility: LocalTrainerEligibility | null
  onSelect: (p: LoraProvider) => void
  /** Open the in-app WSL2 setup wizard (only offered on Windows when not yet eligible). */
  onSetup: () => void
  /** Open the WSL2 memory optimizer (offered on Windows once local is eligible). */
  onOptimizeMemory: () => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const [menuPosition, setMenuPosition] = useState({ left: 8, top: 8 })
  const isWindows = window.electronAPI.platform === 'win32'

  useLayoutEffect(() => {
    if (!open || !ref.current || !menuRef.current) return
    const trigger = ref.current.getBoundingClientRect()
    const menu = menuRef.current.getBoundingClientRect()
    const left = Math.max(8, Math.min(trigger.right - menu.width, window.innerWidth - menu.width - 8))
    const below = trigger.bottom + 8
    const top = below + menu.height <= window.innerHeight - 8
      ? below
      : Math.max(8, trigger.top - menu.height - 8)
    setMenuPosition({ left, top })
  }, [open])

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      const target = e.target as Node
      if (
        ref.current
        && !ref.current.contains(target)
        && !menuRef.current?.contains(target)
      ) setOpen(false)
    }
    const close = () => setOpen(false)
    document.addEventListener('mousedown', onDoc)
    window.addEventListener('resize', close)
    window.addEventListener('scroll', close, true)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      window.removeEventListener('resize', close)
      window.removeEventListener('scroll', close, true)
    }
  }, [open])

  const localEligible = eligibility?.eligible ?? false
  // When not yet eligible but we're on Windows, the row becomes a "set it up"
  // affordance that launches the wizard instead of a dead, disabled option.
  const canSetup = !localEligible && isWindows
  const gpu = eligibility?.gpuName ?? 'Local GPU'
  const vram = eligibility?.vramGb ? ` · ${eligibility.vramGb} GB` : ''
  const localHint = localEligible
    ? `Detected ${gpu}${vram}.`
    : eligibility?.reason || 'Checking this machine for local training support…'

  const select = (p: LoraProvider) => {
    onSelect(p)
    setOpen(false)
  }

  return (
    <div ref={ref} data-tour="provider" className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600 flex items-center gap-1.5"
      >
        {provider === 'local' ? <Cpu className="h-3.5 w-3.5" /> : <Cloud className="h-3.5 w-3.5" />}
        {provider === 'local' ? 'Local GPU' : 'RunPod'}
        <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
      </button>

      {open && createPortal(
        <div
          ref={menuRef}
          role="menu"
          className="fixed z-[9999] w-64 rounded-lg border border-zinc-700 bg-zinc-800 p-1.5 shadow-xl"
          style={menuPosition}
        >
          <button
            role="menuitemradio"
            aria-checked={provider === 'runpod'}
            onClick={() => select('runpod')}
            className={`w-full flex items-start gap-2.5 px-2.5 py-2 rounded-md text-left transition-colors ${
              provider === 'runpod' ? 'bg-white/10' : 'hover:bg-zinc-700/60'
            }`}
          >
            <Cloud className="h-4 w-4 mt-0.5 shrink-0 text-zinc-300" />
            <span className="flex-1">
              <span className="flex items-center gap-1.5 text-sm text-white">
                RunPod
                {provider === 'runpod' && <Check className="h-3.5 w-3.5 text-blue-400" />}
              </span>
              <span className="block text-[11px] text-zinc-500 mt-0.5">
                Cloud GPU — billed only while running.
              </span>
            </span>
          </button>

          <button
            role="menuitemradio"
            aria-checked={provider === 'local'}
            onClick={() => {
              if (localEligible) select('local')
              else if (canSetup) {
                onSetup()
                setOpen(false)
              }
            }}
            disabled={!localEligible && !canSetup}
            title={!localEligible && !canSetup ? localHint : undefined}
            className={`w-full flex items-start gap-2.5 px-2.5 py-2 rounded-md text-left transition-colors ${
              localEligible
                ? provider === 'local'
                  ? 'bg-white/10'
                  : 'hover:bg-zinc-700/60'
                : canSetup
                  ? 'hover:bg-zinc-700/60'
                  : 'cursor-not-allowed opacity-60'
            }`}
          >
            <Cpu className={`h-4 w-4 mt-0.5 shrink-0 ${localEligible || canSetup ? 'text-zinc-300' : 'text-zinc-600'}`} />
            <span className="flex-1">
              <span
                className={`flex items-center gap-1.5 text-sm ${
                  localEligible ? 'text-white' : canSetup ? 'text-zinc-200' : 'text-zinc-500'
                }`}
              >
                Local GPU
                {provider === 'local' && <Check className="h-3.5 w-3.5 text-blue-400" />}
              </span>
              {canSetup ? (
                <span className="block text-[11px] mt-0.5 text-blue-400">Set up local training →</span>
              ) : (
                <span className={`block text-[11px] mt-0.5 ${localEligible ? 'text-emerald-400' : 'text-amber-400/90'}`}>
                  {localHint}
                </span>
              )}
            </span>
          </button>

          {localEligible && isWindows && (
            <button
              role="menuitem"
              onClick={() => {
                onOptimizeMemory()
                setOpen(false)
              }}
              className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-left text-zinc-400 hover:text-white hover:bg-zinc-700/60"
            >
              <MemoryStick className="h-4 w-4 shrink-0 text-zinc-400" />
              <span className="flex-1">
                <span className="block text-sm">Optimize WSL2 memory</span>
                <span className="block text-[11px] text-zinc-500 mt-0.5">
                  Raise WSL2's RAM limit so preprocess doesn't run out of memory.
                </span>
              </span>
            </button>
          )}
        </div>,
        document.body,
      )}
    </div>
  )
}

export function LoraWorkspace({
  projectId = null,
  embedded = false,
}: {
  projectId?: string | null
  embedded?: boolean
}) {
  const { goHome } = useView()
  const { settings, updateSettings } = useAppSettings()
  const { setCurrentTab, setGenSpaceLoraSource } = useProjects()
  const {
    datasets,
    folders,
    preprocessed,
    trainingJobs,
    loading,
    refresh,
    setActive,
    localEligibility,
    loadLocalEligibility,
    createDataset,
    updateDataset,
    deleteDataset,
    archiveDataset,
    unarchiveDataset,
    uploadDataset,
    importDataset,
    cancelTraining,
    deleteTraining,
    archiveTraining,
    unarchiveTraining,
    retryTrainingDownload,
    resumeTraining,
    resetTraining,
    resumePreprocessing,
    deletePreprocessed,
    cancelPreprocessing,
    cancelUpload,
    renameDataset,
    createFolder,
    renameFolder,
    moveFolder,
    deleteFolder,
    moveDataset,
    captionClip,
    applyClipEdits,
    probeClip,
    splitScenes,
  } = useLoraTraining()

  useEffect(() => {
    setActive(true)
    return () => setActive(false)
  }, [setActive])

  const scoped = projectId != null
  const activeDatasets = useMemo(
    () => datasets.filter((dataset) => !dataset.archivedAt),
    [datasets],
  )
  const archivedDatasets = useMemo(
    () => datasets.filter((dataset) => Boolean(dataset.archivedAt)),
    [datasets],
  )
  const visibleDatasets = useMemo(
    () => (scoped ? activeDatasets.filter((d) => d.originatingProjectId === projectId) : activeDatasets),
    [activeDatasets, scoped, projectId],
  )
  const visibleArchivedDatasets = useMemo(
    () => (
      scoped
        ? archivedDatasets.filter((dataset) => dataset.originatingProjectId === projectId)
        : archivedDatasets
    ),
    [archivedDatasets, projectId, scoped],
  )
  // Only show runs/trained that belong to the visible datasets when scoped.
  const visibleTraining = useMemo(() => {
    const activeJobs = trainingJobs.filter((job) => !job.archivedAt)
    if (!scoped) return activeJobs
    const datasetIds = new Set(visibleDatasets.map((d) => d.id))
    const preIds = new Set(preprocessed.filter((p) => datasetIds.has(p.datasetId)).map((p) => p.id))
    return activeJobs.filter((j) => preIds.has(j.preprocessedId))
  }, [scoped, trainingJobs, visibleDatasets, preprocessed])
  const visibleArchivedTraining = useMemo(() => {
    const archivedJobs = trainingJobs.filter((job) => Boolean(job.archivedAt))
    if (!scoped) return archivedJobs
    const projectDatasetIds = new Set(
      datasets
        .filter((dataset) => dataset.originatingProjectId === projectId)
        .map((dataset) => dataset.id),
    )
    const projectPreIds = new Set(
      preprocessed
        .filter((item) => projectDatasetIds.has(item.datasetId))
        .map((item) => item.id),
    )
    return archivedJobs.filter((job) => projectPreIds.has(job.preprocessedId))
  }, [datasets, preprocessed, projectId, scoped, trainingJobs])

  const [selection, setSelection] = useState<Selection>(null)
  // Explicit Compute-panel navigation may target work owned by another
  // project. Keep that one selection viewable without weakening normal
  // project scoping for sidebar navigation.
  const [globalSelectionKey, setGlobalSelectionKey] = useState<string | null>(null)
  const loraRegistry = useLoraInferenceRegistry()
  const [showCreate, setShowCreate] = useState(false)
  const [addClipsTarget, setAddClipsTarget] = useState<LoraDataset | null>(null)
  const [uploadTarget, setUploadTarget] = useState<LoraDataset | null>(null)
  const [showPexels, setShowPexels] = useState(false)
  const [preprocessTarget, setPreprocessTarget] = useState<LoraDataset | null>(null)
  // Failed/cancelled preprocess recovery (resume reusing workspace vs. reset
  // from scratch). Holds the preprocessed item to recover + its dataset.
  const [prepRecovery, setPrepRecovery] = useState<{ preprocessed: LoraPreprocessed; dataset: LoraDataset } | null>(null)
  const [prepRecoveryBusy, setPrepRecoveryBusy] = useState(false)
  const [trainTarget, setTrainTarget] = useState<LoraPreprocessed | null>(null)
  // One-click pipeline confirm sheet (upload → preprocess → train).
  const [pipelineTarget, setPipelineTarget] = useState<LoraDataset | null>(null)
  const [logsTarget, setLogsTarget] = useState<LoraTrainingJob | null>(null)
  const [gpuRecoveryTarget, setGpuRecoveryTarget] = useState<LoraTrainingJob | null>(null)
  const [publishTarget, setPublishTarget] = useState<LoraTrainingJob | null>(null)
  const [droppedPaths, setDroppedPaths] = useState<string[] | null>(null)
  const [busyAction, setBusyAction] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [tourOpen, setTourOpen] = useState(false)
  const [recipesOpen, setRecipesOpen] = useState(false)
  const [showProfiles, setShowProfiles] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(
    () => loadLoraUiPreferences().sidebarWidth,
  )
  const sidebarToggleRef = useRef<HTMLButtonElement>(null)
  const autoTourConsideredRef = useRef(false)

  useEffect(() => {
    if (!sidebarOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      setSidebarOpen(false)
      sidebarToggleRef.current?.focus()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [sidebarOpen])

  const updateSidebarWidth = (next: number) => {
    const clamped = Math.min(520, Math.max(220, Math.round(next)))
    setSidebarWidth(clamped)
    saveLoraUiPreferences({ sidebarWidth: clamped })
  }

  const startSidebarResize = (event: React.PointerEvent<HTMLDivElement>) => {
    if (window.matchMedia('(max-width: 1023px)').matches) return
    event.preventDefault()
    const onMove = (moveEvent: PointerEvent) => updateSidebarWidth(moveEvent.clientX)
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  // Auto-start corrected tour versions only after the initial dataset load.
  // Empty workspaces keep the create/import path unobstructed; Help can still
  // launch the centered tour manually.
  useEffect(() => {
    if (loading || autoTourConsideredRef.current) return
    autoTourConsideredRef.current = true
    if (shouldAutoStartTour(window.localStorage, visibleDatasets.length > 0)) {
      setTourOpen(true)
    }
  }, [loading, visibleDatasets.length])

  const clipLabel = useCallback(
    (path: string) => path.split(/[\\/]/).pop() || path,
    [],
  )

  // Single place to surface a batch op's outcome: log it for the console
  // trace and, on any failure, raise a dismissible banner so prep errors are
  // never silent.
  const reportOp = useCallback((op: string, failures: string[]) => {
    if (failures.length) {
      loraLog.error(op, `${failures.length} failed`, failures)
      setNotice(`${op}: ${failures.length} clip(s) failed — ${failures[0]}`)
    } else {
      loraLog.info(`${op} done`)
    }
  }, [])

  // Keep selection valid; default to the first dataset.
  useEffect(() => {
    if (selection) {
      const selectionKey = `${selection.kind}:${selection.kind === 'library' ? '' : selection.id}`
      const allowGlobal = selectionKey === globalSelectionKey
      if (
        selection.kind === 'dataset'
        && !(allowGlobal ? datasets : visibleDatasets).some((d) => d.id === selection.id)
      ) {
        setSelection(null)
        setGlobalSelectionKey(null)
      } else if (
        selection.kind === 'run'
        && !(allowGlobal ? trainingJobs : visibleTraining).some((j) => j.id === selection.id)
      ) {
        setSelection(null)
        setGlobalSelectionKey(null)
      }
      return
    }
    if (visibleDatasets.length > 0) setSelection({ kind: 'dataset', id: visibleDatasets[0].id })
  }, [
    selection,
    globalSelectionKey,
    datasets,
    trainingJobs,
    visibleDatasets,
    visibleTraining,
  ])

  const selectedDataset =
    selection?.kind === 'dataset'
      ? (globalSelectionKey === `dataset:${selection.id}` ? datasets : visibleDatasets)
          .find((d) => d.id === selection.id) ?? null
      : null
  const selectedRun =
    selection?.kind === 'run'
      ? (globalSelectionKey === `run:${selection.id}` ? trainingJobs : visibleTraining)
          .find((j) => j.id === selection.id) ?? null
      : null

  // Account-wide pod ownership. Project scoping controls sidebar content only;
  // it must never make another project's paid compute look idle.
  const podWorkById = useMemo(
    () => derivePodWork(datasets, preprocessed, trainingJobs),
    [datasets, preprocessed, trainingJobs],
  )
  const activePodIds = useMemo(() => new Set(podWorkById.keys()), [podWorkById])
  const podLifecycleById = useMemo(() => {
    const lifecycle = new Map<string, PodLifecycleInfo>()
    for (const dataset of datasets) {
      const podId = dataset.target?.podId
      if (!podId) continue
      const keepAliveMs = dataset.keepAliveUntil ? Date.parse(dataset.keepAliveUntil) : NaN
      const finalActivityMs = dataset.finalActivityAt ? Date.parse(dataset.finalActivityAt) : NaN
      const autoStopAt = Number.isFinite(keepAliveMs) && keepAliveMs > Date.now()
        ? new Date(keepAliveMs).toISOString()
        : Number.isFinite(finalActivityMs) && settings.runpodIdleStopMinutes > 0
          ? new Date(finalActivityMs + settings.runpodIdleStopMinutes * 60_000).toISOString()
          : null
      lifecycle.set(podId, {
        autoStopAt,
        autoStopDisabled: settings.runpodIdleStopMinutes === 0,
        releaseStatus: dataset.releaseStatus ?? null,
        releaseError: dataset.releaseError ?? null,
        workspacePolicy: dataset.workspacePolicy,
      })
    }
    return lifecycle
  }, [datasets, settings.runpodIdleStopMinutes])

  // One studio store per open collection; recreated when the dataset changes.
  // Sprite merges recover from the durable clip-jobs ledger on recreation.
  const storeRef = useRef<{ id: string; store: StudioStoreApi } | null>(null)
  if (selectedDataset && storeRef.current?.id !== selectedDataset.id) {
    storeRef.current = { id: selectedDataset.id, store: createStudioStore(selectedDataset) }
  }
  const store = selectedDataset && storeRef.current ? storeRef.current.store : null

  useSpritePipeline(store)

  const life = useMemo(
    () => (selectedDataset ? deriveLifecycle(selectedDataset, preprocessed, trainingJobs) : null),
    [selectedDataset, preprocessed, trainingJobs],
  )

  const provider = settings.loraProvider
  // Training readiness is provider-specific: RunPod needs a saved API key;
  // Local GPU needs no key at all — it's "ready" once this machine is eligible
  // (WSL2 + a CUDA GPU). Gating local training on a RunPod key was a bug that
  // left local-only users with a permanently greyed-out "Train LoRA" button.
  const credentialsReady =
    provider === 'local' ? (localEligibility?.eligible ?? false) : settings.hasRunpodApiKey
  const providerLabel = provider === 'local' ? 'Local GPU' : 'RunPod'
  const openCredentials = () =>
    window.dispatchEvent(new CustomEvent('open-settings', { detail: { tab: 'loraTrainer' } }))

  // Probe local-training capability when the trainer view mounts so the
  // provider pill's menu can gate the "Local GPU" option (and show the
  // detected GPU) without the user first opening a training modal. Lazy +
  // cached in the context, so this is a no-op after the first call.
  useEffect(() => {
    void loadLocalEligibility()
  }, [loadLocalEligibility])

  // Guided WSL2 setup wizard, opened from the provider pill when the user picks
  // "Local GPU" before this machine is eligible.
  const [showLocalSetup, setShowLocalSetup] = useState(false)
  const [showWslMemory, setShowWslMemory] = useState(false)

  // Persist the chosen provider so every run inherits it (read back from
  // settings). RunPod is always selectable; Local only when eligible — picking
  // it while ineligible launches the setup wizard instead.
  const onSelectProvider = useCallback(
    (next: LoraProvider) => {
      if (next === 'local' && !localEligibility?.eligible) {
        setShowLocalSetup(true)
        return
      }
      updateSettings({ loraProvider: next })
      // Switching to RunPod without a saved key: jump straight to entering one.
      if (next === 'runpod' && !settings.hasRunpodApiKey) openCredentials()
    },
    [settings.hasRunpodApiKey, localEligibility?.eligible, updateSettings],
  )

  // After the wizard reports the GPU is ready: re-probe eligibility, switch the
  // active provider to local, and close the wizard.
  const onLocalSetupReady = useCallback(() => {
    void loadLocalEligibility()
    updateSettings({ loraProvider: 'local' })
    setShowLocalSetup(false)
  }, [loadLocalEligibility, updateSettings])

  const persist = useCallback(async () => {
    if (!store || !selectedDataset) return false
    const clips = store.getState().clips.map(toClipInput)
    const res = await updateDataset(selectedDataset.id, { clips })
    // Surface failures instead of swallowing them. The backend only accepts
    // edits while a dataset is draft/upload_failed, so captions (or any edit)
    // made after it's been uploaded were silently dropped and vanished on
    // reload — show why rather than feign success.
    if (!res.ok) {
      loraLog.error('persist', res.error)
      setNotice(
        selectedDataset.status === 'uploaded' || selectedDataset.status === 'uploading'
          ? "Changes weren't saved: this collection is already uploaded. Duplicate it to edit captions, then re-upload."
          : `Changes weren't saved: ${res.error}`,
      )
      return false
    }
    // Reconcile against the authoritative response so client-created clips
    // and all persisted edits use the same stable ids as the backend ledger.
    store.getState().setClips(res.data.clips.map(clipFromDataset))
    return true
  }, [store, selectedDataset, updateDataset])

  const handlePrimary = useCallback(() => {
    if (!life || !selectedDataset) return
    switch (life.primary?.kind) {
      case 'add-clips':
        setAddClipsTarget(selectedDataset)
        break
      case 'upload':
        setUploadTarget(selectedDataset)
        break
      case 'preprocess':
        setPreprocessTarget(selectedDataset)
        break
      case 'recover-prep':
        if (life.preprocessed) setPrepRecovery({ preprocessed: life.preprocessed, dataset: selectedDataset })
        break
      case 'recover-gpu':
        setPipelineTarget(selectedDataset)
        break
      case 'train-pipeline':
        setPipelineTarget(selectedDataset)
        break
      case 'train':
        if (life.preprocessed) setTrainTarget(life.preprocessed)
        break
      case 'view-run':
        if (life.training) setSelection({ kind: 'run', id: life.training.id })
        break
      case 'use-lora':
        if (life.training?.localLoraPath) {
          window.electronAPI.showItemInFolder({ filePath: life.training.localLoraPath })
        }
        break
    }
  }, [life, selectedDataset])

  // Confirmed from the pre-upload modal. Any clips the user chose to drop are
  // marked rejected (the only subset the upload API honors) and persisted
  // before we kick off the upload, so the backend ships exactly what was shown.
  const confirmUpload = useCallback(
    async (rejectIds: string[]) => {
      if (!uploadTarget) return
      const id = uploadTarget.id
      if (store && rejectIds.length) {
        store.getState().setClipTriage(rejectIds, 'reject')
        await persist()
      }
      setUploadTarget(null)
      await uploadDataset(id, provider)
    },
    [uploadTarget, store, persist, uploadDataset, provider],
  )

  const onRename = useCallback(
    (name: string) => {
      if (selectedDataset && name) void updateDataset(selectedDataset.id, { name })
    },
    [selectedDataset, updateDataset],
  )

  const onSetTrigger = useCallback(
    (word: string) => {
      if (!selectedDataset) return
      store?.getState().setTriggerWord(word || null)
      void updateDataset(selectedDataset.id, { triggerWord: word || null })
    },
    [selectedDataset, store, updateDataset],
  )

  const onSetType = useCallback(
    (type: LoraDatasetType) => {
      if (!selectedDataset || selectedDataset.type === type) return
      void updateDataset(selectedDataset.id, { type })
    },
    [selectedDataset, updateDataset],
  )

  const onDelete = useCallback(async () => {
    if (!selectedDataset) return
    if (await confirmAction({
      title: `Delete “${selectedDataset.name}”?`,
      message: 'The collection will be deleted and its remote compute released.',
      confirmLabel: 'Delete collection',
      variant: 'destructive',
    })) {
      void deleteDataset(selectedDataset.id)
      setSelection(null)
    }
  }, [selectedDataset, deleteDataset])

  // Open the Add-clips modal for the currently-open collection (toolbar action).
  const onAddClips = useCallback(() => {
    if (selectedDataset) setAddClipsTarget(selectedDataset)
  }, [selectedDataset])

  // Open the Pexels stock-media browser for the currently-open collection.
  const onBrowsePexels = useCallback(() => {
    if (selectedDataset) setShowPexels(true)
  }, [selectedDataset])

  // Copy a collection into a brand-new draft (clip path references only). Works
  // on any source status since we only read its clips and create a new draft.
  const onDuplicateDataset = useCallback(
    async (id: string) => {
      const src = datasets.find((d) => d.id === id)
      if (!src) return
      loraLog.info('duplicate collection', { id, clips: src.clips.length })
      const result = await createDataset(
        `${src.name} copy`,
        src.triggerWord ?? null,
        datasetToInputs(src),
        src.originatingProjectId ?? projectId,
        src.type,
      )
      if (result.ok) setSelection({ kind: 'dataset', id: result.data.id })
      else reportOp('Duplicate collection', [result.error])
    },
    [datasets, createDataset, reportOp, projectId],
  )

  // Combine the selected collections into a new draft (non-destructive — the
  // originals are left intact so the user can delete them if they want).
  const onMergeDatasets = useCallback(
    async (ids: string[]) => {
      const chosen = visibleDatasets.filter((d) => ids.includes(d.id))
      if (chosen.length < 2) return
      loraLog.info('merge collections', { ids })
      const clips = chosen.flatMap(datasetToInputs)
      const name = chosen.map((d) => d.name).join(' + ').slice(0, 80)
      // If any source is an IC-LoRA collection, the merge keeps reference→target
      // pairing, so the result must stay IC-LoRA too.
      const mergedType = chosen.some((d) => d.type === 'ic_lora') ? 'ic_lora' : 'standard'
      const result = await createDataset(name, chosen[0].triggerWord ?? null, clips, projectId, mergedType)
      if (result.ok) setSelection({ kind: 'dataset', id: result.data.id })
      else reportOp('Merge collections', [result.error])
    },
    [visibleDatasets, createDataset, reportOp, projectId],
  )

  // Confirm + delete one or more collections from the sidebar.
  const onDeleteDatasets = useCallback(
    async (ids: string[]) => {
      if (ids.length === 0) return
      const label = ids.length === 1 ? 'this collection' : `${ids.length} collections`
      if (!await confirmAction({
        title: `Delete ${label}?`,
        message: 'Remote compute associated with the selected collections will be released.',
        confirmLabel: ids.length === 1 ? 'Delete collection' : 'Delete collections',
        variant: 'destructive',
      })) return
      for (const id of ids) {
        await deleteDataset(id)
        if (selection?.kind === 'dataset' && selection.id === id) setSelection(null)
      }
    },
    [deleteDataset, selection],
  )

  const onArchiveDatasets = useCallback(
    async (ids: string[]) => {
      const errors: string[] = []
      for (const id of ids) {
        const result = await archiveDataset(id)
        if (!result.ok) errors.push(result.error)
        if (selection?.kind === 'dataset' && selection.id === id) setSelection(null)
      }
      reportOp('Archive datasets', errors)
    },
    [archiveDataset, reportOp, selection],
  )

  const onArchiveTraining = useCallback(
    async (id: string) => {
      const result = await archiveTraining(id)
      reportOp('Archive run', result.ok ? [] : [result.error])
      if (result.ok && selection?.kind === 'run' && selection.id === id) setSelection(null)
    },
    [archiveTraining, reportOp, selection],
  )

  // Spin up a new draft collection from a set of clips (e.g. the current
  // gallery selection), then jump to it.
  const onCreateCollectionFromClips = useCallback(
    async (inputs: ClipInput[]) => {
      if (inputs.length === 0) return
      loraLog.info('new collection from clips', { count: inputs.length })
      // Carry over the source collection's type so a "new collection from
      // selection" out of an IC-LoRA dataset stays IC-LoRA.
      const result = await createDataset(
        'New collection',
        selectedDataset?.triggerWord ?? null,
        inputs,
        projectId,
        selectedDataset?.type,
      )
      if (result.ok) setSelection({ kind: 'dataset', id: result.data.id })
      else reportOp('New collection', [result.error])
    },
    [createDataset, reportOp, projectId, selectedDataset],
  )

  // Import a dataset bundle (folder or .zip) exported from another LTX
  // Desktop, copying its clips into local storage and opening it.
  const onImportDataset = useCallback(
    async (source: 'folder' | 'zip') => {
      const electron = window.electronAPI
      let sourcePath: string | null = null
      if (source === 'zip') {
        const picked = await electron?.showOpenFileDialog?.({
          title: 'Import dataset bundle',
          filters: [{ name: 'Dataset bundle', extensions: ['zip'] }],
        })
        sourcePath = picked?.[0] ?? null
      } else {
        sourcePath = (await electron?.showOpenDirectoryDialog?.({ title: 'Select a dataset bundle folder' })) ?? null
      }
      if (!sourcePath) return
      loraLog.info('import dataset', { source })
      const result = await importDataset(sourcePath)
      if (result.ok) setSelection({ kind: 'dataset', id: result.data.id })
      else reportOp('Import dataset', [result.error])
    },
    [importDataset, reportOp],
  )

  const onCaptionSelected = useCallback(
    async (ids: string[]) => {
      if (!store) return
      setBusyAction(true)
      setNotice(null)
      loraLog.info('caption start', { count: ids.length })
      const clips = store.getState().clips
      const targets = ids
        .map((id) => clips.find((c) => c.id === id))
        .filter((c): c is NonNullable<typeof c> => Boolean(c))
        .map((c) => ({ id: c.id, localPath: c.localPath }))
      const failures: string[] = []
      // Bounded parallelism: a shared cursor hands each worker the next clip,
      // so up to CAPTION_CONCURRENCY captions run at once instead of strictly
      // one-by-one. `cursor++` is atomic here (no await between read and
      // increment on JS's single thread), so workers never double-claim.
      let cursor = 0
      const worker = async () => {
        while (cursor < targets.length) {
          const t = targets[cursor++]
          const res = await captionClip(t.localPath, false)
          if (res.ok) store.getState().setClipCaption(t.id, res.data)
          else failures.push(`${clipLabel(t.localPath)}: ${res.error}`)
        }
      }
      const poolSize = Math.min(CAPTION_CONCURRENCY, targets.length)
      await Promise.all(Array.from({ length: poolSize }, () => worker()))
      await persist()
      setBusyAction(false)
      reportOp('Auto-caption', failures)
    },
    [store, captionClip, persist, reportOp, clipLabel],
  )

  const onRemoveSelected = useCallback(
    async (ids: string[]) => {
      if (!store) return
      store.getState().trashClips(ids)
      await persist()
    },
    [store, persist],
  )

  const onRestoreClips = useCallback(
    async (ids: string[]) => {
      if (!store) return
      store.getState().restoreClips(ids)
      await persist()
    },
    [store, persist],
  )

  const onPurgeClips = useCallback(
    async (ids: string[]) => {
      if (!store) return
      store.getState().removeClips(ids)
      await persist()
    },
    [store, persist],
  )

  const onCropSelected = useCallback(
    async (ids: string[], ratio: [number, number]) => {
      if (!store) return
      setBusyAction(true)
      setNotice(null)
      loraLog.info('crop start', { count: ids.length, ratio })
      const clips = store.getState().clips
      const failures: string[] = []
      for (const id of ids) {
        const clip = clips.find((c) => c.id === id)
        // Skip clips without dimensions or already cropped — re-cropping from
        // source needs the source dims, which only match the probe when there's
        // no prior crop (trim never changes dimensions).
        if (!clip?.probe || clip.edits?.crop) continue
        const crop = centeredCrop(clip.probe.width, clip.probe.height, ratio[0], ratio[1])
        const edits: ClipEdits = {
          crop,
          trim: clip.edits?.trim ?? null,
          scale: clip.edits?.scale ?? null,
          fps: clip.edits?.fps ?? null,
          speed: clip.edits?.speed ?? null,
          mute: clip.edits?.mute ?? false,
          reverse: clip.edits?.reverse ?? false,
        }
        const res = await applyClipEdits(clip.sourcePath, edits)
        if (res.ok) {
          store.getState().applyEditResult(id, {
            localPath: res.data.derivedPath,
            probe: res.data.probe,
            edits,
          })
        } else failures.push(`${clipLabel(clip.sourcePath)}: ${res.error}`)
      }
      await persist()
      setBusyAction(false)
      reportOp('Crop', failures)
    },
    [store, applyClipEdits, persist, reportOp, clipLabel],
  )

  const onApplyEdit = useCallback(
    async (clipId: string, edits: ClipEdits) => {
      if (!store) return
      const clip = store.getState().clips.find((c) => c.id === clipId)
      if (!clip) return
      setBusyAction(true)
      setNotice(null)
      loraLog.info('apply-edit start', { clipId, edits })
      const res = await applyClipEdits(clip.sourcePath, edits)
      if (res.ok) {
        store.getState().applyEditResult(clipId, {
          localPath: res.data.derivedPath,
          probe: res.data.probe,
          edits,
        })
        await persist()
        loraLog.info('apply-edit done', { clipId })
      } else reportOp('Apply edit', [`${clipLabel(clip.sourcePath)}: ${res.error}`])
      setBusyAction(false)
    },
    [store, applyClipEdits, persist, reportOp, clipLabel],
  )

  const onRevertEdit = useCallback(
    async (clipId: string) => {
      if (!store) return
      const clip = store.getState().clips.find((c) => c.id === clipId)
      if (!clip) return
      setBusyAction(true)
      setNotice(null)
      loraLog.info('revert-edit start', { clipId })
      store.getState().resetClipEdits(clipId)
      // Re-measure the source so badges/health reflect the original file.
      const probe = await probeClip(clip.sourcePath)
      if (probe.ok) store.getState().setClipProbe(clipId, probe.data)
      else reportOp('Revert edit (re-probe)', [`${clipLabel(clip.sourcePath)}: ${probe.error}`])
      await persist()
      setBusyAction(false)
    },
    [store, probeClip, persist, reportOp, clipLabel],
  )

  const onNormalizeSelected = useCallback(
    async (ids: string[], targetFps: number | null) => {
      if (!store) return
      setBusyAction(true)
      setNotice(null)
      loraLog.info('normalize start', { count: ids.length, targetFps })
      const clips = store.getState().clips
      const failures: string[] = []
      for (const id of ids) {
        const clip = clips.find((c) => c.id === id)
        if (!clip?.probe) continue
        // Snap to /32 dims relative to the post-crop size; preserve any
        // existing trim/crop and stack scale + fps on top.
        const baseW = clip.edits?.crop?.width ?? clip.probe.width
        const baseH = clip.edits?.crop?.height ?? clip.probe.height
        const w = snapDown32(baseW)
        const h = snapDown32(baseH)
        const needsScale = w !== baseW || h !== baseH
        const needsFps = targetFps != null && Math.abs((clip.probe.fps || 0) - targetFps) > 0.01
        if (!needsScale && !needsFps) continue
        const edits: ClipEdits = {
          trim: clip.edits?.trim ?? null,
          crop: clip.edits?.crop ?? null,
          scale: needsScale ? { width: w, height: h } : (clip.edits?.scale ?? null),
          fps: targetFps ?? clip.edits?.fps ?? null,
          speed: clip.edits?.speed ?? null,
          mute: clip.edits?.mute ?? false,
          reverse: clip.edits?.reverse ?? false,
        }
        const res = await applyClipEdits(clip.sourcePath, edits)
        if (res.ok) {
          store.getState().applyEditResult(id, {
            localPath: res.data.derivedPath,
            probe: res.data.probe,
            edits,
          })
        } else failures.push(`${clipLabel(clip.sourcePath)}: ${res.error}`)
      }
      await persist()
      setBusyAction(false)
      reportOp('Normalize', failures)
    },
    [store, applyClipEdits, persist, reportOp, clipLabel],
  )

  // Bulk trim: apply one trim recipe to every selected clip as a
  // non-destructive edit, mapping the shared recipe onto each clip's own
  // duration (and composing on top of any existing trim window).
  const onTrimSelected = useCallback(
    async (ids: string[], plan: TrimPlan) => {
      if (!store) return
      setBusyAction(true)
      setNotice(null)
      loraLog.info('trim start', { count: ids.length, plan })
      const clips = store.getState().clips
      const failures: string[] = []
      for (const id of ids) {
        const clip = clips.find((c) => c.id === id)
        const len = clip?.probe?.durationSeconds ?? clip?.durationSeconds ?? 0
        if (!clip || len <= 0) continue
        // Anchor on any existing trim start so trims compose in source time.
        const start0 = clip.edits?.trim?.startSeconds ?? 0
        let startSeconds: number
        let endSeconds: number
        if (plan.mode === 'cap') {
          if (plan.seconds <= 0 || len <= plan.seconds + 0.05) continue // already short enough
          startSeconds = start0
          endSeconds = start0 + plan.seconds
        } else {
          if (plan.head <= 0 && plan.tail <= 0) continue
          if (len - plan.head - plan.tail <= 0.05) {
            failures.push(`${clipLabel(clip.sourcePath)}: trim would remove the whole clip`)
            continue
          }
          startSeconds = start0 + plan.head
          endSeconds = start0 + len - plan.tail
        }
        const edits: ClipEdits = {
          trim: { startSeconds, endSeconds },
          crop: clip.edits?.crop ?? null,
          scale: clip.edits?.scale ?? null,
          fps: clip.edits?.fps ?? null,
          speed: clip.edits?.speed ?? null,
          mute: clip.edits?.mute ?? false,
          reverse: clip.edits?.reverse ?? false,
        }
        const res = await applyClipEdits(clip.sourcePath, edits)
        if (res.ok) {
          store.getState().applyEditResult(id, {
            localPath: res.data.derivedPath,
            probe: res.data.probe,
            edits,
          })
        } else failures.push(`${clipLabel(clip.sourcePath)}: ${res.error}`)
      }
      await persist()
      setBusyAction(false)
      reportOp('Trim', failures)
    },
    [store, applyClipEdits, persist, reportOp, clipLabel],
  )

  const onCreatePair = useCallback(
    async (input: ClipInput) => {
      if (!store) return false
      store.getState().addClips([clipFromInput(input)])
      return persist()
    },
    [store, persist],
  )

  // Fold a "generate a reference" result: the AI clip becomes the source
  // clip's single conditioning *input*. The released trainer conditions on one
  // reference per example, so we refuse to append a second input to a source
  // that already has one rather than silently dropping it. This is the mirror
  // of onCreatePair, where the AI clip is itself the target.
  const onCreateReference = useCallback(
    async (sourceClipId: string, input: ClipInput) => {
      if (!store) return false
      const source = store.getState().clips.find((c) => c.id === sourceClipId)
      const newClip = clipFromInput({ ...input, referencePath: null, referencePaths: [] })
      if (source && source.referencePaths.length > 0) {
        // Preserve the completed artifact even if the source was regrouped
        // while generation was running. It remains available as an ungrouped
        // clip instead of being silently discarded.
        store.getState().addClips([newClip])
        reportOp('Add input', [
          'This example already has an input, so the generated result was added as an ungrouped clip.',
        ])
        return persist()
      }
      store.getState().addClips([newClip])
      if (source) {
        store.getState().groupAsPair(sourceClipId, [newClip.id])
      }
      return persist()
    },
    [store, persist, reportOp],
  )

  // Bulk variant of onCreatePair: append many derived clips with a single
  // persist (used by the batch restyle/variant flow).
  const onCreateMany = useCallback(
    async (inputs: ClipInput[]) => {
      if (!store || inputs.length === 0) return
      store.getState().addClips(inputs.map(clipFromInput))
      await persist()
    },
    [store, persist],
  )

  const onAddStill = useCallback(
    async ({ id, framePath, caption, driverPath, probe }: { id?: string; framePath: string; caption: string; driverPath: string; probe?: ClipProbeLike | null }) => {
      if (!store) return false
      store.getState().addImageClip({ id, localPath: framePath, caption, driverPath, probe })
      return persist()
    },
    [store, persist],
  )

  const onCaptionChange = useCallback(
    async (clipId: string, caption: string) => {
      if (!store) return
      store.getState().setClipCaption(clipId, caption)
      await persist()
    },
    [store, persist],
  )

  // Bulk caption text edits (prefix/suffix/find-replace/set). Captions are
  // computed by the caller; we just apply them and persist once.
  const onApplyCaptions = useCallback(
    async (updates: Array<{ id: string; caption: string }>) => {
      if (!store || updates.length === 0) return
      const st = store.getState()
      for (const u of updates) st.setClipCaption(u.id, u.caption)
      await persist()
    },
    [store, persist],
  )

  const onSetTriage = useCallback(
    async (ids: string[], triage: ClipTriage | null) => {
      if (!store || ids.length === 0) return
      store.getState().setClipTriage(ids, triage)
      await persist()
    },
    [store, persist],
  )

  const onGroupPair = useCallback(
    async (targetId: string, referenceIds: string[]) => {
      if (!store) return
      loraLog.info('group-pair', { targetId, references: referenceIds.length })
      store.getState().groupAsPair(targetId, referenceIds)
      await persist()
    },
    [store, persist],
  )

  const onUngroup = useCallback(
    async (ids: string[]) => {
      if (!store) return
      loraLog.info('ungroup', { count: ids.length })
      store.getState().ungroupClips(ids)
      await persist()
    },
    [store, persist],
  )

  const onSceneSplitSelected = useCallback(
    async (ids: string[]) => {
      if (!store) return
      setBusyAction(true)
      setNotice(null)
      loraLog.info('scene-split start', { count: ids.length })
      const clips = store.getState().clips
      const failures: string[] = []
      for (const id of ids) {
        const clip = clips.find((c) => c.id === id)
        if (!clip) continue
        const res = await splitScenes(clip.sourcePath)
        if (!res.ok) {
          failures.push(`${clipLabel(clip.sourcePath)}: ${res.error}`)
          continue
        }
        if (res.data.length <= 1) continue
        const segments = res.data.map((scene) =>
          clipFromInput({
            localPath: scene.localPath,
            caption: clip.caption,
            origin: clip.origin,
            probe: scene.probe,
            durationSeconds: scene.probe.durationSeconds,
          }),
        )
        store.getState().addClips(segments)
        store.getState().removeClips([id])
      }
      await persist()
      setBusyAction(false)
      reportOp('Scene split', failures)
    },
    [store, splitScenes, persist, reportOp, clipLabel],
  )

  const onSegmentSelected = useCallback(
    async (ids: string[], seconds: number) => {
      if (!store) return
      setBusyAction(true)
      setNotice(null)
      loraLog.info('segment start', { count: ids.length, seconds })
      const clips = store.getState().clips
      const failures: string[] = []
      for (const id of ids) {
        const clip = clips.find((c) => c.id === id)
        const duration = clip?.probe?.durationSeconds
        if (!clip || !duration || duration <= seconds + 0.1) continue
        const segments: ClipInput[] = []
        for (let start = 0; start < duration - 0.1; start += seconds) {
          const end = Math.min(start + seconds, duration)
          const edits: ClipEdits = {
            trim: { startSeconds: start, endSeconds: end },
            crop: null,
            scale: null,
            fps: null,
            speed: null,
            mute: false,
            reverse: false,
          }
          const res = await applyClipEdits(clip.sourcePath, edits)
          if (res.ok) {
            segments.push({
              localPath: res.data.derivedPath,
              sourcePath: res.data.derivedPath,
              caption: clip.caption,
              origin: clip.origin,
              probe: res.data.probe,
              durationSeconds: res.data.probe.durationSeconds,
            })
          } else failures.push(`${clipLabel(clip.sourcePath)} @ ${start.toFixed(1)}s: ${res.error}`)
        }
        if (segments.length) {
          store.getState().addClips(segments.map(clipFromInput))
          store.getState().removeClips([id])
        }
      }
      await persist()
      setBusyAction(false)
      reportOp('Segment', failures)
    },
    [store, applyClipEdits, persist, reportOp, clipLabel],
  )

  const onDropFiles = useCallback((paths: string[]) => {
    if (paths.length) setDroppedPaths(paths)
  }, [])

  const onAddClipsDone = useCallback(
    async (inputs: ClipInput[]) => {
      if (store) {
        store.getState().addClips(inputs.map(clipFromInput))
        await persist()
      }
      setAddClipsTarget(null)
      setDroppedPaths(null)
    },
    [store, persist],
  )

  const closeAddClips = useCallback(() => {
    setAddClipsTarget(null)
    setDroppedPaths(null)
  }, [])

  const showAddClips = (addClipsTarget != null || droppedPaths != null) && selectedDataset != null

  // Desktop-style top menus (mirrors the video editor's MenuBar). Actions read
  // selection from the live store at click time so they never go stale.
  const loraMenus: MenuDefinition[] = useMemo(() => {
    const ds = selectedDataset
    const editable = ds ? ds.status === 'draft' || ds.status === 'upload_failed' : false
    const hasClips = (ds?.clips.length ?? 0) > 0
    const selectedIds = (): string[] => (store ? [...store.getState().selectedIds] : [])
    const selectedInputs = (): ClipInput[] => {
      if (!store) return []
      const st = store.getState()
      return st.clips.filter((c) => st.selectedIds.has(c.id)).map(toClipInput)
    }
    return [
      {
        id: 'file',
        label: 'File',
        items: [
          { id: 'new', label: 'New collection', shortcut: 'Ctrl+N', action: () => setShowCreate(true) },
          { id: 'import', label: 'Import media…', shortcut: 'Ctrl+I', action: onAddClips, disabled: !ds || !editable },
          { id: 'pexels', label: 'Browse Pexels…', action: onBrowsePexels, disabled: !ds || !editable },
          { id: 'sep-file-1', label: '', separator: true },
          { id: 'duplicate', label: 'Duplicate collection', action: () => { if (ds) void onDuplicateDataset(ds.id) }, disabled: !ds },
          { id: 'train', label: 'Train LoRA…', action: () => { if (ds) setPipelineTarget(ds) }, disabled: !ds || !editable || !hasClips },
          { id: 'sep-file-2', label: '', separator: true },
          { id: 'profiles', label: 'Training profiles…', action: () => setShowProfiles(true) },
          { id: 'sep-file-3', label: '', separator: true },
          { id: 'delete', label: 'Delete collection', action: onDelete, disabled: !ds },
        ],
      },
      {
        id: 'edit',
        label: 'Edit',
        items: [
          { id: 'select-all', label: 'Select all', shortcut: 'Ctrl+A', action: () => store?.getState().selectAll(), disabled: !ds || !hasClips },
          { id: 'clear', label: 'Clear selection', action: () => store?.getState().clearSelection(), disabled: !ds },
          { id: 'sep-edit-1', label: '', separator: true },
          { id: 'caption', label: 'Auto-caption selected', action: () => { const ids = selectedIds(); if (ids.length) void onCaptionSelected(ids) }, disabled: !ds },
          { id: 'new-from-sel', label: 'New collection from selection', action: () => { const inputs = selectedInputs(); if (inputs.length) void onCreateCollectionFromClips(inputs) }, disabled: !ds },
          { id: 'sep-edit-2', label: '', separator: true },
          { id: 'remove', label: 'Remove selected', action: () => { const ids = selectedIds(); if (ids.length) void onRemoveSelected(ids) }, disabled: !ds || !editable },
        ],
      },
      {
        // id != 'help' so the MenuBar doesn't render its built-in search box.
        id: 'guide',
        label: 'Help',
        items: [
          { id: 'tour', label: 'Guided tour', action: () => setTourOpen(true) },
          { id: 'recipes', label: 'Dataset recipes & best practices', action: () => setRecipesOpen(true) },
        ],
      },
    ]
  }, [selectedDataset, store, onAddClips, onBrowsePexels, onDuplicateDataset, onDelete, onCaptionSelected, onCreateCollectionFromClips, onRemoveSelected])

  return (
    <div className={`${embedded ? 'h-full' : 'h-screen'} bg-background flex flex-col relative`}>
      {notice && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-50 max-w-lg w-[90%]">
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-red-500/40 bg-red-950/90 shadow-lg text-xs text-red-200">
            <span className="flex-1 break-words">{notice}</span>
            <button
              onClick={() => setNotice(null)}
              className="shrink-0 text-red-300 hover:text-white"
              aria-label="Dismiss"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}
      {embedded ? (
        <div className="flex items-center justify-between gap-2 border-b border-zinc-800 px-3 py-2 sm:px-6 sm:pr-32">
          <button
            ref={sidebarToggleRef}
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open collections"
            aria-expanded={sidebarOpen}
            aria-controls="lora-collections-drawer"
            className="flex h-8 w-8 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 lg:hidden"
          >
            <PanelLeft className="h-4 w-4" />
          </button>
          <ProviderSelector
            provider={provider}
            eligibility={localEligibility}
            onSelect={onSelectProvider}
            onSetup={() => setShowLocalSetup(true)}
            onOptimizeMemory={() => setShowWslMemory(true)}
          />
        </div>
      ) : (
        <header className="flex items-center gap-2 border-b border-zinc-800 px-3 py-3 sm:gap-3 sm:px-6 sm:py-4 sm:pr-32">
          <button
            ref={sidebarToggleRef}
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open collections"
            aria-expanded={sidebarOpen}
            aria-controls="lora-collections-drawer"
            className="flex h-8 w-8 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 lg:hidden"
          >
            <PanelLeft className="h-4 w-4" />
          </button>
          <button
            onClick={goHome}
            aria-label="Back to home"
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <ArrowLeft className="h-4 w-4" />
          </button>
          <Wand2 className="h-5 w-5 text-blue-400" />
          <div className="flex-1">
            <h1 className="text-lg font-semibold text-white">LoRA Studio</h1>
            <p className="hidden text-xs text-zinc-500 sm:block">
              Curate clips and train LTX LoRAs on RunPod or your local GPU.
            </p>
          </div>
          <ProviderSelector
            provider={provider}
            eligibility={localEligibility}
            onSelect={onSelectProvider}
            onSetup={() => setShowLocalSetup(true)}
            onOptimizeMemory={() => setShowWslMemory(true)}
          />
        </header>
      )}

      {showLocalSetup && (
        <LocalTrainingSetupWizard onClose={() => setShowLocalSetup(false)} onReady={onLocalSetupReady} />
      )}
      {showWslMemory && <WslMemoryOptimizer onClose={() => setShowWslMemory(false)} />}

      <div data-tour="menubar">
        <MenuBar menus={loraMenus} />
      </div>

      <div className="relative flex min-h-0 flex-1">
        {sidebarOpen && (
          <button
            type="button"
            aria-label="Close collections"
            onClick={() => {
              setSidebarOpen(false)
              sidebarToggleRef.current?.focus()
            }}
            className="fixed inset-0 z-[54] bg-black/60 lg:hidden"
          />
        )}
        <div
          id="lora-collections-drawer"
          style={{ '--lora-sidebar-width': `${sidebarWidth}px` } as CSSProperties}
          className={`fixed inset-y-0 left-0 z-[55] flex min-h-0 w-72 max-w-[88vw] flex-col overflow-hidden bg-background shadow-2xl transition-transform lg:static lg:z-auto lg:h-full lg:max-h-full lg:w-[var(--lora-sidebar-width)] lg:max-w-none lg:flex-none lg:translate-x-0 lg:shadow-none ${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          {sidebarOpen && (
            <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-3 py-2 lg:hidden">
              <span className="text-sm font-semibold text-white">LoRA Studio</span>
              <button
                type="button"
                autoFocus
                onClick={() => {
                  setSidebarOpen(false)
                  sidebarToggleRef.current?.focus()
                }}
                aria-label="Close collections"
                className="flex h-8 w-8 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}
          <CollectionsSidebar
          datasets={visibleDatasets}
          archivedDatasets={visibleArchivedDatasets}
          folders={folders}
          preprocessed={preprocessed}
          trainingJobs={visibleTraining}
          archivedTrainingJobs={visibleArchivedTraining}
          selection={selection}
          libraryCount={loraRegistry.entries.length}
          onSelectDataset={(id) => {
            setGlobalSelectionKey(null)
            setSelection({ kind: 'dataset', id })
            setSidebarOpen(false)
          }}
          onSelectRun={(id) => {
            setGlobalSelectionKey(null)
            setSelection({ kind: 'run', id })
            setSidebarOpen(false)
          }}
          onSelectLibrary={() => {
            setGlobalSelectionKey(null)
            setSelection({ kind: 'library' })
            setSidebarOpen(false)
          }}
          onNewDataset={() => setShowCreate(true)}
          onImportDataset={(source) => void onImportDataset(source)}
          onDuplicate={(id) => void onDuplicateDataset(id)}
          onMerge={(ids) => void onMergeDatasets(ids)}
          onDelete={(ids) => void onDeleteDatasets(ids)}
          onArchive={(ids) => void onArchiveDatasets(ids)}
          onArchiveRun={(id) => void onArchiveTraining(id)}
          onRestoreDataset={async (id) => {
            const result = await unarchiveDataset(id)
            reportOp('Restore dataset', result.ok ? [] : [result.error])
          }}
          onRestoreRun={async (id) => {
            const result = await unarchiveTraining(id)
            reportOp('Restore run', result.ok ? [] : [result.error])
          }}
          onDeleteArchivedDataset={async (id) => {
            await deleteDataset(id)
          }}
          onDeleteArchivedRun={async (id) => {
            await deleteTraining(id)
          }}
          onRename={(id, name) => void renameDataset(id, name)}
          onCreateFolder={(name, parentId) => void createFolder(name, parentId)}
          onRenameFolder={(id, name) => void renameFolder(id, name)}
          onMoveFolder={(id, parentId) => void moveFolder(id, parentId)}
          onDeleteFolder={(id, recursive) => void deleteFolder(id, recursive)}
          onMoveDataset={(id, folderId) => void moveDataset(id, folderId)}
          activePodIds={activePodIds}
          workByPodId={podWorkById}
          lifecycleByPodId={podLifecycleById}
          onOpenPodWork={(target) => {
            setGlobalSelectionKey(`${target.kind}:${target.id}`)
            setSelection({ kind: target.kind, id: target.id })
            setSidebarOpen(false)
          }}
        />
        </div>

        <div
          role="separator"
          aria-label="Resize LoRA sidebar and asset gallery"
          aria-orientation="vertical"
          aria-valuemin={220}
          aria-valuemax={520}
          aria-valuenow={sidebarWidth}
          tabIndex={0}
          onPointerDown={startSidebarResize}
          onDoubleClick={() => updateSidebarWidth(256)}
          onKeyDown={(event) => {
            if (event.key === 'ArrowLeft') {
              event.preventDefault()
              updateSidebarWidth(sidebarWidth - 12)
            } else if (event.key === 'ArrowRight') {
              event.preventDefault()
              updateSidebarWidth(sidebarWidth + 12)
            } else if (event.key === 'Home') {
              event.preventDefault()
              updateSidebarWidth(256)
            }
          }}
          className="hidden w-1.5 shrink-0 cursor-col-resize border-x border-zinc-800 bg-zinc-950 hover:bg-blue-500/20 focus:outline-none focus:ring-1 focus:ring-blue-500 lg:block"
        />

        <ErrorBoundary
          resetKey={selection ? `${selection.kind}:${selection.kind === 'library' ? '' : selection.id}` : 'empty'}
          title="This LoRA view could not be displayed"
        >
        {selection?.kind === 'library' ? (
          <LibraryView
            registry={loraRegistry}
            onTryInGenSpace={(loraId) => {
              setGenSpaceLoraSource({ loraId })
              setCurrentTab('gen-space')
            }}
            onOpenTrainingRun={(jobId) => setSelection({ kind: 'run', id: jobId })}
          />
        ) : selectedDataset && store && life ? (
          <StudioStoreProvider store={store}>
            <CollectionView
              dataset={selectedDataset}
              life={life}
              provider={provider}
              credentialsReady={credentialsReady}
              busyAction={busyAction}
              onPrimary={handlePrimary}
              onRename={onRename}
              onSetTrigger={onSetTrigger}
              onSetType={onSetType}
              onDelete={onDelete}
              onArchive={() => selectedDataset && void onArchiveDatasets([selectedDataset.id])}
              onCaptionSelected={onCaptionSelected}
              onRemoveSelected={onRemoveSelected}
              onRestoreClips={onRestoreClips}
              onPurgeClips={onPurgeClips}
              onCropSelected={onCropSelected}
              onTrimSelected={onTrimSelected}
              onApplyEdit={onApplyEdit}
              onRevertEdit={onRevertEdit}
              onCreatePair={onCreatePair}
              onCreateReference={onCreateReference}
              onCreateMany={onCreateMany}
              onNormalizeSelected={onNormalizeSelected}
              onSceneSplitSelected={onSceneSplitSelected}
              onSegmentSelected={onSegmentSelected}
              onAddStill={onAddStill}
              onCaptionChange={onCaptionChange}
              onApplyCaptions={onApplyCaptions}
              onSetTriage={onSetTriage}
              onGroupPair={onGroupPair}
              onUngroup={onUngroup}
              onDropFiles={onDropFiles}
              onAddClips={onAddClips}
              onBrowsePexels={onBrowsePexels}
              onCreateCollection={(inputs) => void onCreateCollectionFromClips(inputs)}
              onOpenRecipes={() => setRecipesOpen(true)}
              onCancelPreprocess={(id) => void cancelPreprocessing(id)}
              onCancelUpload={() =>
                selectedDataset && void cancelUpload(selectedDataset.id)
              }
            />
          </StudioStoreProvider>
        ) : selectedRun ? (
          <RunView
            job={selectedRun}
            preprocessed={preprocessed.find((p) => p.id === selectedRun.preprocessedId) ?? null}
            dataset={(() => {
              const pre = preprocessed.find((p) => p.id === selectedRun.preprocessedId)
              return pre ? datasets.find((d) => d.id === pre.datasetId) ?? null : null
            })()}
            onCancel={(id) => void cancelTraining(id)}
            onDelete={(id) => {
              void deleteTraining(id)
              setSelection(null)
            }}
            onArchive={(id) => void onArchiveTraining(id)}
            podWork={(() => {
              const podId = selectedRun.target?.podId
                ?? datasets.find((dataset) => {
                  const prep = preprocessed.find((item) => item.id === selectedRun.preprocessedId)
                  return prep?.datasetId === dataset.id
                })?.target?.podId
              return podId ? podWorkById.get(podId) : undefined
            })()}
            onOpenPodWork={(target) => {
              setGlobalSelectionKey(`${target.kind}:${target.id}`)
              setSelection({ kind: target.kind, id: target.id })
            }}
            onOpenLogs={(job) => setLogsTarget(job)}
            onPublish={(job) => setPublishTarget(job)}
            onRetryDownload={(id) => void retryTrainingDownload(id)}
            onResume={(id) => void resumeTraining(id)}
            onReset={(id) => void resetTraining(id)}
            onTryInGenSpace={(job) => {
              // The backend registry exposes user-trained adapters as
              // `user-<trainingJobId>`; mirror that id so Gen Space's picker
              // preselects the matching entry once the registry loads.
              setGenSpaceLoraSource({ loraId: `user-${job.id}` })
              setCurrentTab('gen-space')
            }}
            onChooseAnotherGpu={(job) => setGpuRecoveryTarget(job)}
          />
        ) : (
          <EmptyWorkspace
            onNew={() => setShowCreate(true)}
            onStartTour={() => setTourOpen(true)}
            onOpenRecipes={() => setRecipesOpen(true)}
          />
        )}
        </ErrorBoundary>
      </div>

      {showCreate && (
        <CreateDatasetModal originatingProjectId={projectId} onClose={() => setShowCreate(false)} />
      )}
      {showAddClips && (
        <AddClipsModal
          initialPaths={droppedPaths ?? undefined}
          onClose={closeAddClips}
          onAdd={(inputs) => void onAddClipsDone(inputs)}
        />
      )}
      {showPexels && selectedDataset != null && store != null && (
        <PexelsBrowser
          onClose={() => setShowPexels(false)}
          onAdd={async (inputs) => {
            store.getState().addClips(inputs.map(clipFromInput))
            await persist()
          }}
        />
      )}
      {uploadTarget && store && (
        <UploadConfirmModal
          datasetName={uploadTarget.name}
          datasetType={uploadTarget.type}
          triggerWord={uploadTarget.triggerWord ?? null}
          providerLabel={providerLabel}
          store={store}
          onCancel={() => setUploadTarget(null)}
          onConfirm={(rejectIds) => void confirmUpload(rejectIds)}
        />
      )}
      {preprocessTarget && (
        <PreprocessModal dataset={preprocessTarget} onClose={() => setPreprocessTarget(null)} />
      )}
      {prepRecovery && (
        <PrepRecoveryModal
          preprocessed={prepRecovery.preprocessed}
          datasetName={prepRecovery.dataset.name}
          busy={prepRecoveryBusy}
          onResume={() => {
            setPrepRecoveryBusy(true)
            void resumePreprocessing(prepRecovery.preprocessed.id).finally(() => {
              setPrepRecoveryBusy(false)
              setPrepRecovery(null)
            })
          }}
          onReset={() => {
            setPrepRecoveryBusy(true)
            // Full reset: delete the preprocessed entry entirely so the dataset
            // returns to its "Uploaded" state and the "Train LoRA" button is
            // available again — the user re-configures all settings (captioner,
            // resolution, etc.) from scratch. Unlike resetPreprocessing (which
            // re-runs with the same settings), this lets them pick a different
            // captioner to work around e.g. a VRAM limit.
            void deletePreprocessed(prepRecovery.preprocessed.id).finally(() => {
              setPrepRecoveryBusy(false)
              setPrepRecovery(null)
            })
          }}
          onOptimizeMemory={() => {
            setPrepRecovery(null)
            setShowWslMemory(true)
          }}
          onClose={() => setPrepRecovery(null)}
        />
      )}
      {pipelineTarget && (
        <TrainPipelineModal dataset={pipelineTarget} onClose={() => setPipelineTarget(null)} />
      )}
      {trainTarget && (
        <StartTrainingModal
          preprocessed={trainTarget}
          onClose={() => setTrainTarget(null)}
          onManageProfiles={() => setShowProfiles(true)}
        />
      )}
      {logsTarget && (
        <TrainingLogsModal trainingId={logsTarget.id} name={logsTarget.name} onClose={() => setLogsTarget(null)} />
      )}
      {gpuRecoveryTarget && (() => {
        const recoveryPreprocessed = preprocessed.find(
          (item) => item.id === gpuRecoveryTarget.preprocessedId,
        ) ?? null
        const recoveryDataset = recoveryPreprocessed
          ? datasets.find((item) => item.id === recoveryPreprocessed.datasetId) ?? null
          : null
        return (
          <RunpodGpuRecoveryModal
            job={gpuRecoveryTarget}
            dataset={recoveryDataset}
            preprocessed={recoveryPreprocessed}
            onClose={() => setGpuRecoveryTarget(null)}
            onRecovered={refresh}
          />
        )
      })()}
      {publishTarget && (
        <PublishWizard job={publishTarget} onClose={() => setPublishTarget(null)} />
      )}
      {showProfiles && <TrainingProfileEditor onClose={() => setShowProfiles(false)} />}

      <GuidedTour
        open={tourOpen}
        onClose={() => setTourOpen(false)}
        onOpenRecipes={() => setRecipesOpen(true)}
        datasetType={selectedDataset?.type ?? null}
      />
      <DatasetRecipes
        open={recipesOpen}
        onClose={() => setRecipesOpen(false)}
        onNewCollection={() => { setRecipesOpen(false); setShowCreate(true) }}
        onStartTour={() => { setRecipesOpen(false); setTourOpen(true) }}
      />
    </div>
  )
}
