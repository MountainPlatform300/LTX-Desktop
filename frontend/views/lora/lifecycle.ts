import type {
  LoraDataset,
  LoraPreprocessed,
  LoraTrainingJob,
} from '../../contexts/LoraTrainingContext'

// The LoRA pipeline (draft -> uploaded -> preprocessed -> trained) used to be
// three separate lists the user had to reason about. Here we collapse it into
// one per-dataset lifecycle: a single stage + the one primary action that
// advances it. "Preprocessed" stops being a user-facing object — it's just an
// internal step surfaced as the dataset's status.

export type LoraStage =
  | 'empty'
  | 'draft'
  | 'uploading'
  | 'uploaded'
  | 'preparing'
  | 'ready'
  | 'training'
  | 'trained'
  | 'error'
  | 'cancelled'

export type StageTone = 'neutral' | 'progress' | 'success' | 'error'

export type ActionKind =
  | 'add-clips'
  | 'upload'
  | 'preprocess'
  | 'recover-prep'
  | 'recover-gpu'
  | 'train'
  // One-click: upload → preprocess → train in a single confirm-and-go action.
  | 'train-pipeline'
  | 'view-run'
  | 'use-lora'

export interface LifecycleAction {
  kind: ActionKind
  label: string
  /** Cloud actions are disabled until provider credentials are configured. */
  needsCredentials: boolean
}

export interface Lifecycle {
  stage: LoraStage
  /** Short status label for the collection chip / inspector. */
  label: string
  /** Longer sub-stage detail (e.g. the current upload phase) shown under the
   *  label so a long-running stage tells the user what's actually happening. */
  detail?: string | null
  /** Structured progress for the side panel: 0-100 percent + ETA in seconds.
   *  Present during the model download; null for phases without a measure. */
  percent?: number | null
  etaSeconds?: number | null
  tone: StageTone
  busy: boolean
  /** The single action that advances this dataset, or null while busy. */
  primary: LifecycleAction | null
  /** Derived rows for the inspector / run monitor. */
  preprocessed: LoraPreprocessed | null
  training: LoraTrainingJob | null
  /** True while an upload cancel has been requested but not yet finalized by
   *  the reconciler (pod release in flight) — drives the "Cancelling…" state. */
  cancelRequested: boolean
}

function pickTraining(jobs: LoraTrainingJob[]): LoraTrainingJob | null {
  if (jobs.length === 0) return null
  const newestFirst = [...jobs].sort((a, b) => b.createdAt.localeCompare(a.createdAt))
  return (
    newestFirst.find((j) => j.status === 'running' || j.status === 'pending') ??
    newestFirst[0]
  )
}

function pickPreprocessed(items: LoraPreprocessed[]): LoraPreprocessed | null {
  if (items.length === 0) return null
  const newestFirst = [...items].sort((a, b) => b.createdAt.localeCompare(a.createdAt))
  return (
    newestFirst.find(
      (p) => p.status === 'pending' || p.status === 'captioning' || p.status === 'preprocessing',
    ) ??
    newestFirst[0]
  )
}

function providerLabel(provider: string | null | undefined): 'RunPod' | 'this computer' {
  return provider === 'runpod' ? 'RunPod' : 'this computer'
}

export function deriveLifecycle(
  dataset: LoraDataset,
  preprocessed: LoraPreprocessed[],
  trainingJobs: LoraTrainingJob[],
): Lifecycle {
  const myPre = preprocessed.filter((p) => p.datasetId === dataset.id)
  const myPreIds = new Set(myPre.map((p) => p.id))
  const myTraining = trainingJobs.filter((j) => myPreIds.has(j.preprocessedId))

  const pre = pickPreprocessed(myPre)
  const training = pickTraining(myTraining)

  const base = { preprocessed: pre, training, cancelRequested: dataset.cancelRequested }

  // 1) An active training job is real work and remains authoritative.
  if (training?.status === 'running' || training?.status === 'pending') {
    const cancelling = training.cancelRequested
    const destination = providerLabel(training.provider)
    const percent =
      training.currentStep != null && training.totalSteps != null && training.totalSteps > 0
        ? Math.min(100, Math.round((training.currentStep / training.totalSteps) * 100))
        : null
    return {
      ...base,
      stage: 'training',
      label: cancelling ? 'Cancelling' : training.status === 'pending' ? 'Queued to train' : 'Training',
      detail:
        cancelling
          ? 'Cancelling…'
          : training.statusDetail ??
            (training.status === 'pending'
              ? `Waiting to start on ${destination}…`
              : `Preparing training on ${destination}…`),
      percent,
      etaSeconds: training.etaSeconds ?? null,
      tone: 'progress',
      busy: true,
      primary: { kind: 'view-run', label: 'View run', needsCredentials: false },
    }
  }

  // 2) A new upload cycle must outrank terminal rows from an older cycle.
  if (dataset.status === 'uploading') {
    return {
      ...base,
      stage: 'uploading',
      label: dataset.cancelRequested ? 'Cancelling' : 'Preparing workspace',
      detail: dataset.cancelRequested ? 'Cancelling preparation…' : dataset.statusDetail ?? null,
      percent: dataset.statusPercent ?? null,
      etaSeconds: dataset.statusEtaSeconds ?? null,
      tone: 'progress',
      busy: true,
      primary: null,
    }
  }

  // 3) Newest terminal training result.
  if (training) {
    if (training.status === 'gpu_selection_required') {
      return {
        ...base,
        stage: 'error',
        label: 'GPU unavailable',
        detail: training.statusDetail ?? 'Choose another RunPod GPU to continue.',
        tone: 'error',
        busy: false,
        primary: { kind: 'view-run', label: 'Choose another GPU', needsCredentials: true },
      }
    }
    if (training.status === 'completed') {
      return {
        ...base,
        stage: 'trained',
        label: 'Trained',
        tone: 'success',
        busy: false,
        primary: { kind: 'use-lora', label: 'Reveal LoRA', needsCredentials: false },
      }
    }
    if (training.status === 'failed') {
      return {
        ...base,
        stage: 'error',
        label: 'Training failed',
        tone: 'error',
        busy: false,
        primary: { kind: 'view-run', label: 'View logs', needsCredentials: false },
      }
    }
    // cancelled falls through to the preprocessed/dataset stage below.
  }

  // 4) Preprocessing status.
  if (pre) {
    if (pre.status === 'ready') {
      return {
        ...base,
        stage: 'ready',
        label: 'Ready to train',
        tone: 'success',
        busy: false,
        primary: { kind: 'train', label: 'Train', needsCredentials: true },
      }
    }
    if (pre.status === 'pending' || pre.status === 'captioning' || pre.status === 'preprocessing') {
      const label = pre.status === 'captioning' ? 'Captioning' : pre.status === 'preprocessing' ? 'Preparing' : 'Queued'
      // Surface the backend's live preprocess progress (phase + % + ETA, written
      // to the dataset by the runner's log-tail poll) so a multi-minute
      // captioning/latent-caching run visibly ticks instead of sitting on a
      // static "Preparing". Fall back to a stage-specific hint before the first
      // log parse lands (statusDetail is null briefly at start).
      const fallbackDetail =
        pre.status === 'captioning'
          ? 'Captioning clips…'
          : pre.status === 'preprocessing'
            ? dataset.type === 'ic_lora'
              ? 'Encoding input/output pairs…'
              : 'Caching training latents…'
            : `Waiting to prepare on ${providerLabel(dataset.target?.provider)}…`
      return {
        ...base,
        stage: 'preparing',
        label,
        tone: 'progress',
        busy: true,
        primary: null,
        detail: dataset.statusDetail ?? fallbackDetail,
        percent: dataset.statusPercent ?? null,
        etaSeconds: dataset.statusEtaSeconds ?? null,
      }
    }
    if (pre.status === 'failed') {
      return {
        ...base,
        stage: 'error',
        label: 'Preparation failed',
        detail: pre.statusDetail ?? pre.error ?? null,
        tone: 'error',
        busy: false,
        // Opens the resume/reset modal: Resume reuses the uploaded workspace +
        // cached captions (re-runs the latent-caching step that typically
        // OOMs); Reset wipes the cached latents/captions and starts over.
        primary: { kind: 'recover-prep', label: 'Resume prep', needsCredentials: true },
      }
    }
    // cancelled -> fall through to dataset status.
  }

  // 5) Dataset upload status.
  switch (dataset.status) {
    case 'gpu_selection_required':
      return {
        ...base,
        stage: 'error',
        label: 'GPU unavailable',
        detail: dataset.statusDetail ?? 'Choose another RunPod GPU to continue.',
        tone: 'error',
        busy: false,
        primary: { kind: 'recover-gpu', label: 'Choose another GPU', needsCredentials: true },
      }
    case 'uploaded':
      return {
        ...base,
        stage: 'uploaded',
        label: 'Clips ready',
        tone: 'success',
        busy: false,
        primary: { kind: 'train-pipeline', label: 'Train LoRA', needsCredentials: true },
      }
    case 'upload_failed':
      return {
        ...base,
        stage: 'error',
        label: 'Preparation failed',
        detail: dataset.error ?? null,
        tone: 'error',
        busy: false,
        primary: { kind: 'upload', label: 'Retry preparation', needsCredentials: true },
      }
    case 'cancelled':
      return {
        ...base,
        stage: 'cancelled',
        label: 'Cancelled',
        tone: 'neutral',
        busy: false,
        primary: { kind: 'upload', label: 'Retry preparation', needsCredentials: true },
      }
    case 'draft':
    default:
      if (dataset.clips.filter((c) => !c.deletedAt).length === 0) {
        return {
          ...base,
          stage: 'empty',
          label: 'Empty',
          tone: 'neutral',
          busy: false,
          primary: { kind: 'add-clips', label: 'Add clips', needsCredentials: false },
        }
      }
      return {
        ...base,
        stage: 'draft',
        label: 'Draft',
        tone: 'neutral',
        busy: false,
        primary: { kind: 'train-pipeline', label: 'Train LoRA', needsCredentials: true },
      }
  }
}

export const STAGE_DOT: Record<StageTone, string> = {
  neutral: 'bg-zinc-500',
  progress: 'bg-blue-400',
  success: 'bg-emerald-400',
  error: 'bg-red-400',
}

export const STAGE_TEXT: Record<StageTone, string> = {
  neutral: 'text-zinc-400',
  progress: 'text-blue-400',
  success: 'text-emerald-400',
  error: 'text-red-400',
}
