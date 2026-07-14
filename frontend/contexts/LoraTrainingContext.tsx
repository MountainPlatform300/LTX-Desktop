import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import {
  ApiClient,
  type ApiSuccessOf,
  type ApiRequestBodyOf,
  type UpdateLoraProfileBody,
  type PublishLoraPreviewBody,
  type PublishLoraExportBody,
} from '../lib/api-client'
import { logger } from '../lib/logger'
import { useToast } from './ToastContext'

// ---------------------------------------------------------------------------
// Types — re-exported from the generated OpenAPI client so backend renames or
// new fields surface as TS errors in the panel at compile time.
// ---------------------------------------------------------------------------

export type LoraDatasetsState = ApiSuccessOf<'listLoraDatasets'>
export type LoraDataset = LoraDatasetsState['datasets'][number]
export type LoraDatasetStatus = LoraDataset['status']
export type LoraDatasetClip = LoraDataset['clips'][number]
export type LoraFolder = LoraDatasetsState['folders'][number]

export type LoraPreprocessedState = ApiSuccessOf<'listLoraPreprocessed'>
export type LoraPreprocessed = LoraPreprocessedState['items'][number]
export type LoraPreprocessStatus = LoraPreprocessed['status']

export type LoraTrainingJobsState = ApiSuccessOf<'listLoraTraining'>
export type LoraTrainingJob = LoraTrainingJobsState['items'][number]
export type LoraTrainingStatus = LoraTrainingJob['status']
export type LoraProvider = LoraTrainingJob['provider']

/** Capability probe for local (WSL2) training (eligible / reason / GPU info). */
export type LocalTrainerEligibility = ApiSuccessOf<'getLoraLocalEligibility'>

export type LoraProfilesState = ApiSuccessOf<'listLoraProfiles'>
export type LoraProfile = LoraProfilesState['profiles'][number]
/** The full tunable trainer config carried by a profile / run. */
export type LoraTrainingConfig = LoraProfile['config']
export type CreateProfileBody = ApiRequestBodyOf<'createLoraProfile'>

export type CreateDatasetBody = ApiRequestBodyOf<'createLoraDataset'>
export type StartPreprocessingBody = ApiRequestBodyOf<'startLoraPreprocessing'>
export type StartTrainingBody = ApiRequestBodyOf<'startLoraTraining'>
export type StartTrainingPipelineBody = ApiRequestBodyOf<'startLoraTrainingPipeline'>
export type ClipInput = NonNullable<CreateDatasetBody['clips']>[number]
/** Collection type: `standard` LoRA vs `ic_lora` (reference → target). */
export type LoraDatasetType = NonNullable<CreateDatasetBody['type']>
export type ClipProbe = ApiSuccessOf<'probeLoraClip'>['probe']
export type ClipEdits = NonNullable<ClipInput['edits']>
export type ApplyEditsResult = ApiSuccessOf<'applyLoraClipEdits'>
export type SceneSplitResult = ApiSuccessOf<'splitLoraScenes'>
export type LoraScene = SceneSplitResult['scenes'][number]
export type DerivedClipResult = ApiSuccessOf<'restyleLoraClip'>
export type NanoBananaModel = NonNullable<ApiRequestBodyOf<'editLoraFrame'>['model']>
/** Which editor runs a LoRA frame edit: "fal" (Nano Banana, remote) or "klein" (FLUX.2 [klein] 9B, local). */
export type FrameEditEngine = NonNullable<ApiRequestBodyOf<'editLoraFrame'>['engine']>
export type MotionEditEngine = NonNullable<ApiRequestBodyOf<'motionEditLoraClip'>['engine']>
export type MotionEditOptions = {
  prompt?: string
  engine?: MotionEditEngine
  videoStrength?: number
  characterOrientation?: 'video' | 'image'
  /** Kling O3 ("kling_o3") only: keep the source clip's original audio. */
  keepAudio?: boolean
}
export type ClipJob = ApiSuccessOf<'listLoraClipJobs'>['jobs'][number]

export type PexelsSearchBody = ApiRequestBodyOf<'searchPexels'>
export type PexelsSearchResult = ApiSuccessOf<'searchPexels'>
export type PexelsMediaItem = PexelsSearchResult['items'][number]
export type PexelsDownloadResult = ApiSuccessOf<'downloadPexels'>

export type DerivationJob = ApiSuccessOf<'listLoraDerivations'>['jobs'][number]

// LoRA publication (model card) types.
export type PublishPreviewBody = PublishLoraPreviewBody
export type PublishExportBody = PublishLoraExportBody
export type PublishPreviewResult = ApiSuccessOf<'publishLoraPreview'>
export type PublishExportResult = ApiSuccessOf<'publishLoraExport'>
export type PublishPlatform = PublishExportBody['platforms'][number]
export type PublicationMeta = PublishExportBody['meta']
export type PublicationExample = NonNullable<PublishExportBody['examples']>[number]
export type DerivationStatus = DerivationJob['status']
export type CreateDerivationBody = ApiRequestBodyOf<'createLoraDerivation'>
/** Stage-3 backend: local IC-LoRA depth/canny drive vs remote Kling. */
export type DerivationEngine = NonNullable<CreateDerivationBody['engine']>
/** Stage-2 frame-edit engine: "fal" (Nano Banana) or "klein" (local FLUX.2 [klein] 9B). */
export type DerivationEditEngine = NonNullable<CreateDerivationBody['editEngine']>
export type DerivationConditioning = NonNullable<CreateDerivationBody['conditioningType']>
/** Which side of the example the AI result becomes: the target (start from a
 *  reference) or a reference (start from a target). IC-LoRA only. */
export type DerivationDirection = NonNullable<CreateDerivationBody['direction']>

type Result<T = void> = { ok: true; data: T } | { ok: false; error: string }

interface LoraTrainingContextValue {
  datasets: LoraDataset[]
  folders: LoraFolder[]
  preprocessed: LoraPreprocessed[]
  trainingJobs: LoraTrainingJob[]
  derivationJobs: DerivationJob[]
  profiles: LoraProfile[]
  loading: boolean
  // Bump the polling cadence while the panel is mounted/visible so status
  // transitions feel responsive; drop it otherwise to keep cost ~zero.
  setActive: (active: boolean) => void
  refresh: () => Promise<void>

  // Local (WSL2) training capability. Fetched lazily (call `loadLocalEligibility`
  // when a training modal opens); `null` until the first probe resolves.
  localEligibility: LocalTrainerEligibility | null
  localEligibilityLoading: boolean
  loadLocalEligibility: () => Promise<void>

  createDataset: (
    name: string,
    triggerWord: string | null,
    clips: ClipInput[],
    originatingProjectId?: string | null,
    type?: LoraDatasetType,
  ) => Promise<Result<LoraDataset>>
  updateDataset: (
    id: string,
    patch: { name?: string | null; triggerWord?: string | null; clips?: ClipInput[] | null; type?: LoraDatasetType | null },
  ) => Promise<Result<LoraDataset>>
  deleteDataset: (id: string) => Promise<void>
  archiveDataset: (id: string) => Promise<Result<LoraDataset>>
  unarchiveDataset: (id: string) => Promise<Result<LoraDataset>>
  uploadDataset: (id: string, provider: LoraProvider) => Promise<Result<LoraDataset>>
  exportDataset: (
    id: string,
    opts: {
      destPath: string
      format: 'folder' | 'zip'
      includeRejected: boolean
      // Build the bundle's train_config.yaml from this saved profile; omit
      // (or null) to use the trainer defaults.
      profileId?: string | null
      // IC-LoRA training-ready normalization (ignored for standard LoRA).
      icLoraFps?: number
      icLoraShortSide?: number
      icLoraBucketFrames?: number
      icLoraMaxDurationSeconds?: number | null
      forbiddenCaptionWords?: string[]
      // Which supplementary files to include (dataset clips + dataset.json are
      // always written). Default true to match the prior export behavior.
      includeConfig?: boolean
      includeReadme?: boolean
      includeManifest?: boolean
      includeModelCard?: boolean
    },
  ) => Promise<Result<{ exportPath: string; clipCount: number; droppedPairs: string[] }>>
  importDataset: (sourcePath: string) => Promise<Result<LoraDataset>>
  // LoRA publication: render the model card(s) for preview, and write the
  // publication bundle (card + examples + weights) to a chosen folder.
  publishPreview: (trainingId: string, body: PublishPreviewBody) => Promise<Result<PublishPreviewResult>>
  publishExport: (trainingId: string, body: PublishExportBody) => Promise<Result<PublishExportResult>>

  startPreprocessing: (body: StartPreprocessingBody) => Promise<Result<LoraPreprocessed>>
  cancelPreprocessing: (id: string) => Promise<void>
  cancelUpload: (id: string) => Promise<void>
  renameDataset: (id: string, name: string) => Promise<void>
  createFolder: (name: string, parentId: string | null) => Promise<void>
  renameFolder: (id: string, name: string) => Promise<void>
  moveFolder: (id: string, parentId: string | null) => Promise<void>
  deleteFolder: (id: string, recursive: boolean) => Promise<void>
  moveDataset: (id: string, folderId: string | null) => Promise<void>
  resumePreprocessing: (id: string) => Promise<void>
  resetPreprocessing: (id: string) => Promise<void>
  deletePreprocessed: (id: string) => Promise<void>

  startTraining: (body: StartTrainingBody) => Promise<Result<LoraTrainingJob>>
  /** One-click: upload → preprocess → train in one action. Returns the dataset. */
  startTrainingPipeline: (body: StartTrainingPipelineBody) => Promise<Result<LoraDataset>>
  cancelTraining: (id: string) => Promise<void>
  deleteTraining: (id: string) => Promise<void>
  archiveTraining: (id: string) => Promise<Result<LoraTrainingJob>>
  unarchiveTraining: (id: string) => Promise<Result<LoraTrainingJob>>
  retryTrainingDownload: (id: string) => Promise<void>
  resumeTraining: (id: string) => Promise<void>
  resetTraining: (id: string) => Promise<void>

  // Reusable training profiles (named config bundles). Starting a run with a
  // profileId snapshots its config onto the job, so later edits don't rewrite
  // finished runs.
  createProfile: (body: CreateProfileBody) => Promise<Result<LoraProfile>>
  updateProfile: (id: string, patch: UpdateLoraProfileBody) => Promise<Result<LoraProfile>>
  deleteProfile: (id: string) => Promise<void>

  testConnection: (provider: LoraProvider) => Promise<{ ok: boolean; message: string }>
  fetchLogs: (trainingId: string) => Promise<string[]>
  // Desktop-side auto-caption of a single clip via Gemini. Decoupled from the
  // dataset so the caller can show per-clip progress and let the user edit
  // before persisting via updateDataset.
  captionClip: (videoPath: string, withAudio: boolean) => Promise<Result<string>>
  // Desktop-side ffmpeg probe of a single clip (duration / resolution / fps /
  // audio). Used to badge clips and drive quality warnings before upload.
  probeClip: (videoPath: string) => Promise<Result<ClipProbe>>
  // Render a trimmed/cropped derivative of a source clip (non-destructive —
  // the original path is untouched). Returns the derived path + fresh probe.
  applyClipEdits: (sourcePath: string, edits: ClipEdits) => Promise<Result<ApplyEditsResult>>
  // Detect scene cuts in a long clip and render each segment to its own file.
  splitScenes: (sourcePath: string, threshold?: number) => Promise<Result<LoraScene[]>>
  // AI dataset prep (Fal, BYOK). Edit a frame with Nano Banana; animate a
  // (usually edited) still into a clip; restyle an existing clip (vid2vid).
  editFrame: (
    sourcePath: string,
    prompt: string,
    opts?: { timeSeconds?: number; model?: NanoBananaModel; engine?: FrameEditEngine },
  ) => Promise<Result<string>>
  // Extract a single frame from a video at `timeSeconds` to a PNG on disk and
  // return its path. Used to show the *actual* frame being edited (vs the
  // clip's poster) so before/after comparisons line up.
  extractFrame: (sourcePath: string, timeSeconds: number) => Promise<Result<string>>
  animateFrame: (imagePath: string, prompt: string) => Promise<Result<DerivedClipResult>>
  restyleClip: (sourcePath: string, prompt: string) => Promise<Result<DerivedClipResult>>
  // Motion-locked paired generation: drive motion from `sourcePath` while
  // anchoring content to `referenceImagePath` (e.g. a Nano-Banana-edited
  // first frame). Produces the "after" half of an aligned edit pair.
  motionEditClip: (
    sourcePath: string,
    referenceImagePath: string,
    opts?: MotionEditOptions,
  ) => Promise<Result<DerivedClipResult>>
  // Pexels stock-media browser (BYOK): search photos/videos and download a
  // chosen asset into app storage so it can be added to the collection.
  searchPexels: (body: PexelsSearchBody) => Promise<Result<PexelsSearchResult>>
  downloadPexels: (
    item: Pick<PexelsMediaItem, 'downloadUrl' | 'kind' | 'downloadExt'>,
  ) => Promise<Result<PexelsDownloadResult>>
  // Local clip-prep jobs: enqueue background sprite/filmstrip generation for a
  // batch of source clips (powers the curation gallery's hover-scrub) and poll
  // the durable ledger for results. Stateless w.r.t. datasets — keyed by path.
  enqueueClipJobs: (sourcePaths: string[]) => Promise<Result<ClipJob[]>>
  listClipJobs: () => Promise<Result<ClipJob[]>>
  // Background target/variant generation. `createDerivation` enqueues the
  // staged pipeline (frame edit -> local IC-LoRA drive or remote Kling) and
  // returns immediately; poll `derivationJobs` for progress/results.
  createDerivation: (body: CreateDerivationBody) => Promise<Result<DerivationJob>>
  cancelDerivation: (id: string) => Promise<void>
  // Abort a whole bulk Fal run at once. Scope to a dataset to leave other
  // collections' jobs running; omit to cancel everything.
  cancelAllDerivations: (datasetId?: string) => Promise<void>
  retryDerivation: (id: string) => Promise<void>
  dismissDerivation: (id: string) => Promise<void>
  // Review-gate controls: approve a paused edit so the motion drive runs, or
  // regenerate the edit (optionally with a new prompt) before approving.
  approveDerivation: (id: string) => Promise<void>
  regenerateDerivationEdit: (id: string, editPrompt?: string) => Promise<void>
}

const LoraTrainingContext = createContext<LoraTrainingContextValue | null>(null)

// Faster while the panel is open (status changes drive the wizard); slower
// when it's closed so the backend isn't polled for a feature nobody's using.
const ACTIVE_INTERVAL_MS = 2000
const IDLE_INTERVAL_MS = 8000
const LOG_REPEAT_INTERVAL_MS = 60_000

function errorMessage(error: unknown, fallback: string): string {
  return (error as { message?: string })?.message ?? fallback
}

export function LoraTrainingProvider({ children }: { children: ReactNode }) {
  const [datasets, setDatasets] = useState<LoraDataset[]>([])
  const [folders, setFolders] = useState<LoraFolder[]>([])
  const [preprocessed, setPreprocessed] = useState<LoraPreprocessed[]>([])
  const [trainingJobs, setTrainingJobs] = useState<LoraTrainingJob[]>([])
  const [derivationJobs, setDerivationJobs] = useState<DerivationJob[]>([])
  const [profiles, setProfiles] = useState<LoraProfile[]>([])
  const [loading, setLoading] = useState(true)
  const [localEligibility, setLocalEligibility] = useState<LocalTrainerEligibility | null>(null)
  const [localEligibilityLoading, setLocalEligibilityLoading] = useState(false)
  // De-dupe concurrent probes (e.g. two modals mounting) so we hit the
  // endpoint once even if `loadLocalEligibility` is called repeatedly.
  const eligibilityInFlightRef = useRef<Promise<void> | null>(null)

  const intervalMsRef = useRef(IDLE_INTERVAL_MS)
  const inFlightRef = useRef<Promise<void> | null>(null)
  const lastFailureRef = useRef<{ loggedAt: number } | null>(null)
  const wasFailingRef = useRef(false)

  const fetchState = useCallback(async (): Promise<void> => {
    if (inFlightRef.current) return inFlightRef.current
    const promise = (async () => {
      const [d, p, t, g, pr] = await Promise.all([
        ApiClient.listLoraDatasets({ includeArchived: true }),
        ApiClient.listLoraPreprocessed(),
        ApiClient.listLoraTraining({ includeArchived: true }),
        ApiClient.listLoraDerivations(),
        ApiClient.listLoraProfiles(),
      ])
      // Treat the set atomically: only update state when all reads succeed,
      // otherwise keep the last good snapshot and log (throttled).
      if (d.ok && p.ok && t.ok && g.ok && pr.ok) {
        setDatasets(d.data.datasets)
        setFolders(d.data.folders)
        setPreprocessed(p.data.items)
        setTrainingJobs(t.data.items)
        setDerivationJobs(g.data.jobs)
        setProfiles(pr.data.profiles)
        if (wasFailingRef.current) {
          logger.info('LoRA trainer: state fetch recovered')
          wasFailingRef.current = false
          lastFailureRef.current = null
        }
      } else {
        const now = Date.now()
        const last = lastFailureRef.current
        if (!last || now - last.loggedAt >= LOG_REPEAT_INTERVAL_MS) {
          logger.warn('LoRA trainer: state fetch failed')
          lastFailureRef.current = { loggedAt: now }
        }
        wasFailingRef.current = true
      }
    })().finally(() => {
      inFlightRef.current = null
      setLoading(false)
    })
    inFlightRef.current = promise
    return promise
  }, [])

  useEffect(() => {
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout> | null = null
    const tick = async () => {
      if (cancelled) return
      await fetchState()
      if (cancelled) return
      timeoutId = setTimeout(tick, intervalMsRef.current)
    }
    void tick()
    return () => {
      cancelled = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [fetchState])

  // Active notifications for the background derivation pipeline. We diff each
  // polled snapshot against the previous one so we fire once per transition
  // (not once per poll): when an edit becomes ready to review, and when a video
  // finishes. The first snapshot only seeds the baseline (no toast on load).
  const { addToast } = useToast()
  const prevDerivStatusRef = useRef<Map<string, DerivationJob['status']> | null>(null)
  useEffect(() => {
    const prev = prevDerivStatusRef.current
    const cur = new Map(derivationJobs.map((j) => [j.id, j.status]))
    if (prev !== null) {
      let videosDone = 0
      for (const job of derivationJobs) {
        const before = prev.get(job.id)
        if (job.status === 'completed' && before !== undefined && before !== 'completed') {
          videosDone += 1
        }
      }
      if (videosDone > 0) {
        addToast({
          variant: 'success',
          title: `${videosDone} video${videosDone > 1 ? 's' : ''} ready`,
          description: 'Generation completed. Open the collection to review the result.',
        })
      }
      // Fire once when a review wave starts (0 -> >0 jobs awaiting review).
      const prevReview = [...prev.values()].filter((s) => s === 'review').length
      const curReview = [...cur.values()].filter((s) => s === 'review').length
      if (prevReview === 0 && curReview > 0) {
        addToast({
          variant: 'warning',
          title: 'Edits ready to review',
          description: 'Approve them before their videos generate.',
        })
      }
    }
    prevDerivStatusRef.current = cur
  }, [derivationJobs, addToast])

  const setActive = useCallback((active: boolean) => {
    intervalMsRef.current = active ? ACTIVE_INTERVAL_MS : IDLE_INTERVAL_MS
    if (active) void fetchState()
  }, [fetchState])

  const refresh = useCallback(async () => {
    await fetchState()
  }, [fetchState])

  // Lazy capability probe for local (WSL2) training. Cached after the first
  // success so reopening a modal doesn't re-probe; concurrent calls share one
  // in-flight request. The endpoint never errors, but if the request itself
  // fails (offline backend), we leave the result null so the UI stays gated.
  const loadLocalEligibility = useCallback(async (): Promise<void> => {
    if (eligibilityInFlightRef.current) return eligibilityInFlightRef.current
    setLocalEligibilityLoading(true)
    const promise = (async () => {
      const result = await ApiClient.getLoraLocalEligibility()
      if (result.ok) {
        setLocalEligibility(result.data)
      } else {
        logger.warn('LoRA trainer: local-eligibility probe failed')
      }
    })().finally(() => {
      eligibilityInFlightRef.current = null
      setLocalEligibilityLoading(false)
    })
    eligibilityInFlightRef.current = promise
    return promise
  }, [])

  // ----- Dataset mutations -----

  const createDataset = useCallback<LoraTrainingContextValue['createDataset']>(async (name, triggerWord, clips, originatingProjectId, type) => {
    const result = await ApiClient.createLoraDataset({
      name,
      type: type ?? 'standard',
      triggerWord,
      clips,
      originatingProjectId: originatingProjectId ?? null,
    })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to create dataset') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const updateDataset = useCallback<LoraTrainingContextValue['updateDataset']>(async (id, patch) => {
    const result = await ApiClient.updateLoraDataset(id, {
      name: patch.name ?? null,
      type: patch.type ?? null,
      triggerWord: patch.triggerWord ?? null,
      clips: patch.clips ?? null,
    })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to update dataset') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const deleteDataset = useCallback(async (id: string) => {
    const result = await ApiClient.deleteLoraDataset(id)
    // 204 surfaces as a synthetic empty-success "error"; only a real 4xx/5xx
    // with a body is worth logging. Either way we refetch to reflect the ledger.
    if (!result.ok && result.status !== 'default') {
      logger.warn(`LoRA trainer: delete dataset failed for ${id}`)
    }
    await fetchState()
  }, [fetchState])

  const archiveDataset = useCallback<LoraTrainingContextValue['archiveDataset']>(async (id) => {
    const result = await ApiClient.archiveLoraDataset(id)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to archive dataset') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const unarchiveDataset = useCallback<LoraTrainingContextValue['unarchiveDataset']>(async (id) => {
    const result = await ApiClient.unarchiveLoraDataset(id)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to restore dataset') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const uploadDataset = useCallback<LoraTrainingContextValue['uploadDataset']>(async (id, provider) => {
    const result = await ApiClient.uploadLoraDataset(id, { provider })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to start upload') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const exportDataset = useCallback<LoraTrainingContextValue['exportDataset']>(async (id, opts) => {
    const result = await ApiClient.exportLoraDataset(id, {
      destPath: opts.destPath,
      format: opts.format,
      includeRejected: opts.includeRejected,
      profileId: opts.profileId ?? null,
      // These carry server-side defaults; the generated body type still lists
      // them, so always send (the backend ignores them for standard LoRA).
      icLoraFps: opts.icLoraFps ?? 25,
      icLoraShortSide: opts.icLoraShortSide ?? 576,
      icLoraBucketFrames: opts.icLoraBucketFrames ?? 49,
      ...(opts.icLoraMaxDurationSeconds != null
        ? { icLoraMaxDurationSeconds: opts.icLoraMaxDurationSeconds }
        : {}),
      ...(opts.forbiddenCaptionWords ? { forbiddenCaptionWords: opts.forbiddenCaptionWords } : {}),
      // Defaults carry server-side; the generated body type still lists them,
      // so always send (true preserves the prior all-in export).
      includeConfig: opts.includeConfig ?? true,
      includeReadme: opts.includeReadme ?? true,
      includeManifest: opts.includeManifest ?? true,
      includeModelCard: opts.includeModelCard ?? true,
    })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to export dataset') }
    return {
      ok: true,
      data: {
        exportPath: result.data.exportPath,
        clipCount: result.data.clipCount,
        droppedPairs: result.data.droppedPairs ?? [],
      },
    }
  }, [])

  const importDataset = useCallback<LoraTrainingContextValue['importDataset']>(async (sourcePath) => {
    const result = await ApiClient.importLoraDataset({ sourcePath })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to import dataset') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const publishPreview = useCallback<LoraTrainingContextValue['publishPreview']>(async (trainingId, body) => {
    const result = await ApiClient.publishLoraPreview(trainingId, body)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to build preview') }
    return { ok: true, data: result.data }
  }, [])

  const publishExport = useCallback<LoraTrainingContextValue['publishExport']>(async (trainingId, body) => {
    const result = await ApiClient.publishLoraExport(trainingId, body)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to export publication') }
    return { ok: true, data: result.data }
  }, [])

  // ----- Preprocessing mutations -----

  const startPreprocessing = useCallback<LoraTrainingContextValue['startPreprocessing']>(async (body) => {
    const result = await ApiClient.startLoraPreprocessing(body)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to start preprocessing') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const cancelPreprocessing = useCallback(async (id: string) => {
    const result = await ApiClient.cancelLoraPreprocessing(id)
    if (!result.ok) logger.warn(`LoRA trainer: cancel preprocessing failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const cancelUpload = useCallback(async (id: string) => {
    const result = await ApiClient.cancelLoraUpload(id)
    if (!result.ok) logger.warn(`LoRA trainer: cancel upload failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const renameDataset = useCallback(async (id: string, name: string) => {
    const result = await ApiClient.renameLoraDataset(id, { name })
    if (!result.ok) logger.warn(`LoRA trainer: rename dataset failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const createFolder = useCallback(async (name: string, parentId: string | null) => {
    const result = await ApiClient.createLoraFolder({ name, parentId })
    if (!result.ok) logger.warn(`LoRA trainer: create folder failed`)
    await fetchState()
  }, [fetchState])

  const renameFolder = useCallback(async (id: string, name: string) => {
    const result = await ApiClient.renameLoraFolder(id, { name })
    if (!result.ok) logger.warn(`LoRA trainer: rename folder failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const moveFolder = useCallback(async (id: string, parentId: string | null) => {
    const result = await ApiClient.moveLoraFolder(id, { parentId })
    if (!result.ok) logger.warn(`LoRA trainer: move folder failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const deleteFolder = useCallback(async (id: string, recursive: boolean) => {
    const result = await ApiClient.deleteLoraFolder(id, recursive)
    if (!result.ok) logger.warn(`LoRA trainer: delete folder failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const moveDataset = useCallback(async (id: string, folderId: string | null) => {
    const result = await ApiClient.moveLoraDataset(id, { folderId })
    if (!result.ok) logger.warn(`LoRA trainer: move dataset failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const resumePreprocessing = useCallback(async (id: string) => {
    const result = await ApiClient.resumeLoraPreprocessing(id)
    if (!result.ok) logger.warn(`LoRA trainer: resume preprocessing failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const resetPreprocessing = useCallback(async (id: string) => {
    const result = await ApiClient.resetLoraPreprocessing(id)
    if (!result.ok) logger.warn(`LoRA trainer: reset preprocessing failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const deletePreprocessed = useCallback(async (id: string) => {
    const result = await ApiClient.deleteLoraPreprocessed(id)
    if (!result.ok && result.status !== 'default') {
      logger.warn(`LoRA trainer: delete preprocessed failed for ${id}`)
    }
    await fetchState()
  }, [fetchState])

  // ----- Training mutations -----

  const startTraining = useCallback<LoraTrainingContextValue['startTraining']>(async (body) => {
    const result = await ApiClient.startLoraTraining(body)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to start training') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const startTrainingPipeline = useCallback<LoraTrainingContextValue['startTrainingPipeline']>(async (body) => {
    const result = await ApiClient.startLoraTrainingPipeline(body)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to start training') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const cancelTraining = useCallback(async (id: string) => {
    const result = await ApiClient.cancelLoraTraining(id)
    if (!result.ok) logger.warn(`LoRA trainer: cancel training failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const deleteTraining = useCallback(async (id: string) => {
    const result = await ApiClient.deleteLoraTraining(id)
    if (!result.ok && result.status !== 'default') {
      logger.warn(`LoRA trainer: delete training failed for ${id}`)
    }
    await fetchState()
  }, [fetchState])

  const archiveTraining = useCallback<LoraTrainingContextValue['archiveTraining']>(async (id) => {
    const result = await ApiClient.archiveLoraTraining(id)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to archive run') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const unarchiveTraining = useCallback<LoraTrainingContextValue['unarchiveTraining']>(async (id) => {
    const result = await ApiClient.unarchiveLoraTraining(id)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to restore run') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const retryTrainingDownload = useCallback(async (id: string) => {
    const result = await ApiClient.retryLoraTrainingDownload(id)
    if (!result.ok) logger.warn(`LoRA trainer: retry download failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const resumeTraining = useCallback(async (id: string) => {
    const result = await ApiClient.resumeLoraTraining(id)
    if (!result.ok) logger.warn(`LoRA trainer: resume training failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const resetTraining = useCallback(async (id: string) => {
    const result = await ApiClient.resetLoraTraining(id)
    if (!result.ok) logger.warn(`LoRA trainer: reset training failed for ${id}`)
    await fetchState()
  }, [fetchState])

  // ----- Training profiles -----

  const createProfile = useCallback<LoraTrainingContextValue['createProfile']>(async (body) => {
    const result = await ApiClient.createLoraProfile(body)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to create profile') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const updateProfile = useCallback<LoraTrainingContextValue['updateProfile']>(async (id, patch) => {
    const result = await ApiClient.updateLoraProfile(id, patch)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to update profile') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const deleteProfile = useCallback(async (id: string) => {
    const result = await ApiClient.deleteLoraProfile(id)
    if (!result.ok && result.status !== 'default') {
      logger.warn(`LoRA trainer: delete profile failed for ${id}`)
    }
    await fetchState()
  }, [fetchState])

  const testConnection = useCallback(async (provider: LoraProvider) => {
    const result = await ApiClient.testLoraConnection({ provider })
    if (!result.ok) {
      return { ok: false, message: errorMessage(result.error, 'Connection test failed') }
    }
    return { ok: result.data.ok, message: result.data.message }
  }, [])

  const fetchLogs = useCallback(async (trainingId: string): Promise<string[]> => {
    const result = await ApiClient.getLoraTrainingLogs(trainingId)
    if (!result.ok) return []
    return result.data.lines
  }, [])

  const captionClip = useCallback<LoraTrainingContextValue['captionClip']>(async (videoPath, withAudio) => {
    const result = await ApiClient.captionLoraClip({ videoPath, withAudio })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to caption clip') }
    return { ok: true, data: result.data.caption }
  }, [])

  const probeClip = useCallback<LoraTrainingContextValue['probeClip']>(async (videoPath) => {
    const result = await ApiClient.probeLoraClip({ videoPath })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to probe clip') }
    return { ok: true, data: result.data.probe }
  }, [])

  const applyClipEdits = useCallback<LoraTrainingContextValue['applyClipEdits']>(async (sourcePath, edits) => {
    const result = await ApiClient.applyLoraClipEdits({ sourcePath, edits })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to apply edits') }
    return { ok: true, data: result.data }
  }, [])

  const splitScenes = useCallback<LoraTrainingContextValue['splitScenes']>(async (sourcePath, threshold) => {
    const result = await ApiClient.splitLoraScenes({ sourcePath, threshold: threshold ?? 0.4 })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to split scenes') }
    return { ok: true, data: result.data.scenes }
  }, [])

  const editFrame = useCallback<LoraTrainingContextValue['editFrame']>(async (sourcePath, prompt, opts) => {
    const result = await ApiClient.editLoraFrame({
      sourcePath,
      prompt,
      timeSeconds: opts?.timeSeconds ?? 0,
      model: opts?.model ?? null,
      engine: opts?.engine ?? 'fal',
    })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to edit frame') }
    return { ok: true, data: result.data.framePath }
  }, [])

  const extractFrame = useCallback<LoraTrainingContextValue['extractFrame']>(async (sourcePath, timeSeconds) => {
    const result = await ApiClient.extractMediaFrame({ sourcePath, timeSeconds })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to extract frame') }
    return { ok: true, data: result.data.path }
  }, [])

  const animateFrame = useCallback<LoraTrainingContextValue['animateFrame']>(async (imagePath, prompt) => {
    const result = await ApiClient.animateLoraFrame({ imagePath, prompt })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to animate frame') }
    return { ok: true, data: result.data }
  }, [])

  const restyleClip = useCallback<LoraTrainingContextValue['restyleClip']>(async (sourcePath, prompt) => {
    const result = await ApiClient.restyleLoraClip({ sourcePath, prompt })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to restyle clip') }
    return { ok: true, data: result.data }
  }, [])

  const motionEditClip = useCallback<LoraTrainingContextValue['motionEditClip']>(
    async (sourcePath, referenceImagePath, opts) => {
      const result = await ApiClient.motionEditLoraClip({
        sourcePath,
        referenceImagePath,
        prompt: opts?.prompt ?? '',
        engine: opts?.engine ?? 'ltx_v2v',
        videoStrength: opts?.videoStrength ?? 0.5,
        characterOrientation: opts?.characterOrientation ?? 'video',
        keepAudio: opts?.keepAudio ?? true,
      })
      if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to generate paired clip') }
      return { ok: true, data: result.data }
  }, [])

  const searchPexels = useCallback<LoraTrainingContextValue['searchPexels']>(async (body) => {
    const result = await ApiClient.searchPexels(body)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Pexels search failed') }
    return { ok: true, data: result.data }
  }, [])

  const downloadPexels = useCallback<LoraTrainingContextValue['downloadPexels']>(async (item) => {
    const result = await ApiClient.downloadPexels({
      url: item.downloadUrl,
      kind: item.kind,
      ext: item.downloadExt,
    })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to download from Pexels') }
    return { ok: true, data: result.data }
  }, [])

  const enqueueClipJobs = useCallback<LoraTrainingContextValue['enqueueClipJobs']>(async (sourcePaths) => {
    const result = await ApiClient.enqueueLoraClipJobs({ sourcePaths, kind: 'sprite' })
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to enqueue clip jobs') }
    return { ok: true, data: result.data.jobs }
  }, [])

  const listClipJobs = useCallback<LoraTrainingContextValue['listClipJobs']>(async () => {
    const result = await ApiClient.listLoraClipJobs()
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to list clip jobs') }
    return { ok: true, data: result.data.jobs }
  }, [])

  const createDerivation = useCallback<LoraTrainingContextValue['createDerivation']>(async (body) => {
    const result = await ApiClient.createLoraDerivation(body)
    if (!result.ok) return { ok: false, error: errorMessage(result.error, 'Failed to start generation') }
    await fetchState()
    return { ok: true, data: result.data }
  }, [fetchState])

  const cancelDerivation = useCallback<LoraTrainingContextValue['cancelDerivation']>(async (id) => {
    const result = await ApiClient.cancelLoraDerivation(id)
    if (!result.ok) logger.warn(`LoRA trainer: cancel derivation failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const cancelAllDerivations = useCallback<LoraTrainingContextValue['cancelAllDerivations']>(async (datasetId) => {
    const result = await ApiClient.cancelAllLoraDerivations({ datasetId: datasetId ?? null })
    if (!result.ok) logger.warn('LoRA trainer: cancel-all derivations failed')
    await fetchState()
  }, [fetchState])

  const retryDerivation = useCallback<LoraTrainingContextValue['retryDerivation']>(async (id) => {
    const result = await ApiClient.retryLoraDerivation(id)
    if (!result.ok) logger.warn(`LoRA trainer: retry derivation failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const dismissDerivation = useCallback<LoraTrainingContextValue['dismissDerivation']>(async (id) => {
    const result = await ApiClient.dismissLoraDerivation(id)
    if (!result.ok) logger.warn(`LoRA trainer: dismiss derivation failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const approveDerivation = useCallback<LoraTrainingContextValue['approveDerivation']>(async (id) => {
    const result = await ApiClient.approveLoraDerivation(id)
    if (!result.ok) logger.warn(`LoRA trainer: approve derivation failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const regenerateDerivationEdit = useCallback<LoraTrainingContextValue['regenerateDerivationEdit']>(async (id, editPrompt) => {
    const result = await ApiClient.regenerateLoraDerivationEdit(id, editPrompt)
    if (!result.ok) logger.warn(`LoRA trainer: regenerate derivation edit failed for ${id}`)
    await fetchState()
  }, [fetchState])

  const value = useMemo<LoraTrainingContextValue>(() => ({
    datasets,
    folders,
    preprocessed,
    trainingJobs,
    derivationJobs,
    profiles,
    loading,
    setActive,
    refresh,
    localEligibility,
    localEligibilityLoading,
    loadLocalEligibility,
    createDataset,
    updateDataset,
    deleteDataset,
    archiveDataset,
    unarchiveDataset,
    uploadDataset,
    exportDataset,
    importDataset,
    publishPreview,
    publishExport,
    startPreprocessing,
    cancelPreprocessing,
    cancelUpload,
    renameDataset,
    createFolder,
    renameFolder,
    moveFolder,
    deleteFolder,
    moveDataset,
    resumePreprocessing,
    resetPreprocessing,
    deletePreprocessed,
    startTraining,
    startTrainingPipeline,
    cancelTraining,
    deleteTraining,
    archiveTraining,
    unarchiveTraining,
    retryTrainingDownload,
    resumeTraining,
    resetTraining,
    createProfile,
    updateProfile,
    deleteProfile,
    testConnection,
    fetchLogs,
    captionClip,
    probeClip,
    applyClipEdits,
    splitScenes,
    editFrame,
    extractFrame,
    animateFrame,
    restyleClip,
    motionEditClip,
    searchPexels,
    downloadPexels,
    enqueueClipJobs,
    listClipJobs,
    createDerivation,
    cancelDerivation,
    cancelAllDerivations,
    retryDerivation,
    dismissDerivation,
    approveDerivation,
    regenerateDerivationEdit,
  }), [
    datasets,
    folders,
    preprocessed,
    trainingJobs,
    derivationJobs,
    profiles,
    loading,
    setActive,
    refresh,
    localEligibility,
    localEligibilityLoading,
    loadLocalEligibility,
    createDataset,
    updateDataset,
    deleteDataset,
    archiveDataset,
    unarchiveDataset,
    uploadDataset,
    exportDataset,
    importDataset,
    publishPreview,
    publishExport,
    startPreprocessing,
    cancelPreprocessing,
    cancelUpload,
    renameDataset,
    createFolder,
    renameFolder,
    moveFolder,
    deleteFolder,
    moveDataset,
    resumePreprocessing,
    resetPreprocessing,
    deletePreprocessed,
    startTraining,
    startTrainingPipeline,
    cancelTraining,
    deleteTraining,
    archiveTraining,
    unarchiveTraining,
    retryTrainingDownload,
    resumeTraining,
    resetTraining,
    createProfile,
    updateProfile,
    deleteProfile,
    testConnection,
    fetchLogs,
    captionClip,
    probeClip,
    applyClipEdits,
    splitScenes,
    editFrame,
    extractFrame,
    animateFrame,
    restyleClip,
    motionEditClip,
    searchPexels,
    downloadPexels,
    enqueueClipJobs,
    listClipJobs,
    createDerivation,
    cancelDerivation,
    cancelAllDerivations,
    retryDerivation,
    dismissDerivation,
    approveDerivation,
    regenerateDerivationEdit,
  ])

  return <LoraTrainingContext.Provider value={value}>{children}</LoraTrainingContext.Provider>
}

export function useLoraTraining(): LoraTrainingContextValue {
  const ctx = useContext(LoraTrainingContext)
  if (!ctx) {
    throw new Error('useLoraTraining must be used within LoraTrainingProvider')
  }
  return ctx
}
