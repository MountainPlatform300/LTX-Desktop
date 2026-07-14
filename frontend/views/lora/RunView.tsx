import { useCallback, useEffect, useState } from 'react'
import {
  Archive,
  CheckCircle2,
  Cloud,
  Download,
  FolderOpen,
  Loader2,
  RotateCcw,
  ScrollText,
  Sparkles,
  Square,
  Trash2,
  XCircle,
} from 'lucide-react'
import type { LoraDataset, LoraPreprocessed, LoraTrainingJob } from '../../contexts/LoraTrainingContext'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { ApiClient, type ApiSuccessOf } from '../../lib/api-client'
import { GpuStatusPanel } from '../../components/lora/GpuStatusPanel'
import { ValidationFeedPanel } from '../../components/lora/ValidationFeedPanel'
import { confirmAction } from '../../components/ui/confirm-dialog'
import type { PodWorkTarget } from './ComputePanel'

function formatEta(seconds: number): string {
  if (seconds < 60) return `~${Math.max(1, Math.round(seconds))}s left`
  if (seconds < 3600) return `~${Math.round(seconds / 60)}m left`
  const h = Math.floor(seconds / 3600)
  const m = Math.round((seconds % 3600) / 60)
  return m > 0 ? `~${h}h ${m}m left` : `~${h}h left`
}

function formatDuration(startedAt: string | null, completedAt: string | null): string | null {
  if (!startedAt || !completedAt) return null
  const secs = Math.round((new Date(completedAt).getTime() - new Date(startedAt).getTime()) / 1000)
  if (!Number.isFinite(secs) || secs < 0) return null
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  const s = secs % 60
  if (h) return `${h}h ${m}m`
  if (m) return `${m}m ${s}s`
  return `${s}s`
}

function formatSeconds(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null
  const rounded = Math.round(seconds)
  const h = Math.floor(rounded / 3600)
  const m = Math.floor((rounded % 3600) / 60)
  const s = rounded % 60
  if (h) return `${h}h ${m}m`
  if (m) return `${m}m ${s}s`
  return `${s}s`
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1 border-b border-zinc-800/60 last:border-0">
      <span className="text-[11px] text-zinc-500 shrink-0">{label}</span>
      <span className="text-[11px] text-zinc-300 text-right break-words font-mono">{value}</span>
    </div>
  )
}

function RunSummary({
  job,
  dataset,
  preprocessed,
}: {
  job: LoraTrainingJob
  dataset: LoraDataset | null
  preprocessed: LoraPreprocessed | null
}) {
  // Total = click (createdAt) -> done. Setup = click -> first step (pod boot +
  // model load + step-0 validation, the silent ~10 min). Training = first step
  // -> done (actual stepping).
  const total = formatDuration(job.createdAt ?? null, job.completedAt ?? null)
  const setup = formatDuration(job.createdAt ?? null, job.firstStepAt ?? null)
  const training = formatDuration(job.firstStepAt ?? null, job.completedAt ?? null)
  const steps = job.totalSteps ?? job.config.steps
  const lowVram = job.config.preset === 'low_vram'
  const optimizer = job.config.optimizerType || (lowVram ? 'adamw8bit' : 'adamw')
  const quant = job.config.quantization || (lowVram ? 'int8-quanto' : 'none')
  const gpu = job.gpuType ? (job.gpuVramGb ? `${job.gpuType} (${job.gpuVramGb} GB)` : job.gpuType) : '—'
  const provider = job.provider === 'runpod' ? 'RunPod' : 'Local GPU'
  const datasetType =
    dataset?.type === 'ic_lora' ? 'IC-LoRA' : dataset?.type === 'standard' ? 'Standard LoRA' : undefined
  const attributedTime = formatSeconds(job.attributedSeconds)
  const hourlyRate = job.capturedHourlyRate ?? job.computeRatePerHr
  const estimatedComputeCost = job.provider === 'runpod'
    ? job.attributedCost != null
      ? `$${job.attributedCost.toFixed(2)}${hourlyRate != null ? ` at $${hourlyRate.toFixed(2)}/hr` : ''} (RunPod invoice is authoritative)`
      : 'Unavailable for this run (RunPod invoice is authoritative)'
    : null
  const podPreparation = formatDuration(
    job.podPreparationStartedAt ?? null,
    job.podPreparationEndedAt ?? null,
  )
  const attributedSetup = formatDuration(
    job.trainingSetupStartedAt ?? null,
    job.trainingSetupEndedAt ?? null,
  )
  const attributedSteps = formatDuration(
    job.trainingStepsStartedAt ?? job.firstStepAt ?? null,
    job.trainingStepsEndedAt ?? job.completedAt ?? null,
  )
  const rows: Array<[string, string | null | undefined]> = [
    ['Training job time', total],
    ['Attributed RunPod time', job.provider === 'runpod' ? attributedTime : null],
    ['Estimated RunPod cost', estimatedComputeCost],
    ['Pod preparation', job.provider === 'runpod' ? podPreparation : null],
    ['Training setup', attributedSetup ?? setup],
    ['Training steps', attributedSteps ?? training],
    ['GPU', gpu],
    ['Provider', provider],
    ['Steps', String(steps)],
    ['Rank / alpha', `${job.config.rank} / ${job.config.alpha}`],
    ['Preset', job.config.preset],
    ['Learning rate', String(job.config.learningRate)],
    ['Optimizer', optimizer],
    ['Quantization', quant],
    ['Dataset', dataset?.name],
    ['Type', datasetType],
    ['Training clips', dataset ? String(dataset.clips.length) : undefined],
    ['Resolution', preprocessed?.resolutionBuckets],
    ['Audio', preprocessed ? (preprocessed.withAudio ? 'yes' : 'no') : undefined],
    ['Trigger word', job.config.triggerWord || '—'],
    ['Started', job.startedAt ? new Date(job.startedAt).toLocaleString() : undefined],
    ['Completed', job.completedAt ? new Date(job.completedAt).toLocaleString() : undefined],
  ]
  return (
    <div className="bg-zinc-900/40 border border-zinc-800 rounded-lg p-4">
      <p className="text-xs font-medium text-zinc-300 mb-2">Run summary</p>
      <div className="space-y-0">
        {rows
          .filter((r): r is [string, string] => typeof r[1] === 'string' && r[1].length > 0)
          .map(([label, value]) => (
            <SummaryRow key={label} label={label} value={value} />
          ))}
      </div>
    </div>
  )
}

function StatusBadge({ job }: { job: LoraTrainingJob }) {
  const hasStartedSteps = (job.currentStep ?? 0) > 0
  const map = {
    pending: { tone: 'text-zinc-300 bg-zinc-700/40', label: 'Queued', busy: true },
    running: {
      tone: 'text-blue-400 bg-blue-500/10',
      label: hasStartedSteps ? 'Training' : 'Training setup',
      busy: true,
    },
    completed: { tone: 'text-emerald-400 bg-emerald-500/10', label: 'Completed', busy: false },
    failed: { tone: 'text-red-400 bg-red-500/10', label: 'Failed', busy: false },
    cancelled: { tone: 'text-zinc-400 bg-zinc-700/40', label: 'Cancelled', busy: false },
    gpu_selection_required: { tone: 'text-amber-300 bg-amber-500/10', label: 'GPU needed', busy: false },
  } as const
  const s = map[job.status]
  return (
    <span className={`text-[11px] px-2 py-0.5 rounded-full inline-flex items-center gap-1 ${s.tone}`}>
      {s.busy && <Loader2 className="h-3 w-3 animate-spin" />}
      {s.label}
    </span>
  )
}

type RunpodPod = ApiSuccessOf<'listRunpodPods'>[number]

function PostRunPodBanner({
  job,
  dataset,
  work,
  onOpenWork,
}: {
  job: LoraTrainingJob
  dataset: LoraDataset | null
  work?: PodWorkTarget
  onOpenWork?: (target: PodWorkTarget) => void
}) {
  const { settings } = useAppSettings()
  const podId = dataset?.target?.podId ?? job.target?.podId
  const [pod, setPod] = useState<RunpodPod | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!podId) return
    const result = await ApiClient.listRunpodPods()
    if (result.ok) setPod(result.data.find((item) => item.id === podId) ?? null)
  }, [podId])

  useEffect(() => {
    if (!podId) return
    void refresh()
    const refreshId = window.setInterval(() => void refresh(), 15_000)
    const tickId = window.setInterval(() => setNow(Date.now()), 1_000)
    return () => {
      window.clearInterval(refreshId)
      window.clearInterval(tickId)
    }
  }, [podId, refresh])

  if (job.provider !== 'runpod' || !podId || !pod?.running) return null

  const keepAliveMs = dataset?.keepAliveUntil ? Date.parse(dataset.keepAliveUntil) : NaN
  const finalActivityMs = dataset?.finalActivityAt ? Date.parse(dataset.finalActivityAt) : NaN
  const autoStopMs = Number.isFinite(keepAliveMs) && keepAliveMs > now
    ? keepAliveMs
    : Number.isFinite(finalActivityMs) && settings.runpodIdleStopMinutes > 0
      ? finalActivityMs + settings.runpodIdleStopMinutes * 60_000
      : NaN
  const remainingSeconds = Number.isFinite(autoStopMs)
    ? Math.max(0, Math.ceil((autoStopMs - now) / 1000))
    : null
  const spend = pod.uptimeSeconds != null && pod.costPerHr != null
    ? (pod.uptimeSeconds / 3600) * pod.costPerHr
    : null
  const workspacePolicy = job.runpodSelection?.workspacePolicy ?? dataset?.workspacePolicy
  const destructive = workspacePolicy !== 'primary_cache'
  const claimedByOtherWork = work && !(work.kind === 'run' && work.id === job.id)

  const runAction = async (action: 'release' | 'keep') => {
    if (action === 'release' && destructive && !await confirmAction({
      title: 'Terminate ephemeral pod?',
      message: 'The pod and its temporary disk will be permanently deleted.',
      confirmLabel: 'Terminate pod',
      variant: 'destructive',
    })) return
    setBusy(true)
    setMessage(null)
    const result = action === 'keep'
      ? await ApiClient.keepRunpodPodAlive(podId, 30)
      : destructive
        ? await ApiClient.terminateRunpodPod(podId)
        : await ApiClient.stopRunpodPod(podId)
    if (!result.ok) setMessage(result.error.message)
    else await refresh()
    setBusy(false)
  }

  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/[0.06] p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold text-amber-200">RunPod pod is still running</p>
          <p className="mt-0.5 text-[11px] text-zinc-400">
            {pod.costPerHr != null ? `$${pod.costPerHr.toFixed(2)}/hr` : 'Rate unavailable'}
            {spend != null ? ` · ~$${spend.toFixed(2)} since pod start` : ''}
          </p>
          <p className="mt-1 text-[11px] font-medium text-amber-300">
            {claimedByOtherWork
              ? `In use by ${work.label} (${work.stage})`
              : settings.runpodIdleStopMinutes === 0
                ? 'Auto-stop is off'
                : remainingSeconds != null
                  ? `Auto-${destructive ? 'terminate' : 'stop'} in ${formatEta(remainingSeconds).replace(/^~/, '').replace(/ left$/, '')}`
                  : `Auto-${destructive ? 'terminate' : 'stop'} scheduled`}
          </p>
          {dataset?.releaseStatus === 'failed' && (
            <p className="mt-1 text-[10px] text-red-300">
              Auto-stop failed: {dataset.releaseError || 'Unknown RunPod error'}
            </p>
          )}
          {message && <p className="mt-1 text-[10px] text-red-300">{message}</p>}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {claimedByOtherWork && onOpenWork ? (
            <button
              type="button"
              onClick={() => onOpenWork(work)}
              className="rounded-md bg-blue-600 px-2.5 py-1.5 text-[10px] font-medium text-white hover:bg-blue-500"
            >
              View {work.kind}
            </button>
          ) : (
            <>
              <button
                type="button"
                disabled={busy}
                onClick={() => void runAction('keep')}
                className="rounded-md border border-zinc-700 px-2.5 py-1.5 text-[10px] font-medium text-zinc-200 hover:bg-zinc-800 disabled:opacity-40"
              >
                Keep running 30 minutes
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => void runAction('release')}
                className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1.5 text-[10px] font-medium text-white disabled:opacity-40 ${
                  destructive ? 'bg-red-600 hover:bg-red-500' : 'bg-amber-600 hover:bg-amber-500'
                }`}
              >
                <Square className="h-3 w-3" />
                {destructive ? 'Terminate now' : 'Stop now'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export function RunView({
  job,
  dataset = null,
  preprocessed = null,
  onCancel,
  onDelete,
  onArchive,
  onOpenLogs,
  onPublish,
  onRetryDownload,
  onResume,
  onReset,
  onTryInGenSpace,
  onChooseAnotherGpu,
  podWork,
  onOpenPodWork,
}: {
  job: LoraTrainingJob
  dataset?: LoraDataset | null
  preprocessed?: LoraPreprocessed | null
  onCancel: (id: string) => void
  onDelete: (id: string) => void
  onArchive: (id: string) => void
  onOpenLogs: (job: LoraTrainingJob) => void
  onPublish: (job: LoraTrainingJob) => void
  onRetryDownload: (id: string) => void
  onResume: (id: string) => void
  onReset: (id: string) => void
  onTryInGenSpace: (job: LoraTrainingJob) => void
  onChooseAnotherGpu?: (job: LoraTrainingJob) => void
  podWork?: PodWorkTarget
  onOpenPodWork?: (target: PodWorkTarget) => void
}) {
  const active = job.status === 'pending' || job.status === 'running'
  const providerLabel = job.provider === 'runpod' ? 'RunPod' : 'Local GPU'
  const [resetConfirmOpen, setResetConfirmOpen] = useState(false)
  const progress =
    job.totalSteps && job.totalSteps > 0
      ? Math.round(((job.currentStep ?? 0) / job.totalSteps) * 100)
      : null
  const canResume = job.status === 'failed' || job.status === 'cancelled'
  const canReset = job.status === 'failed' || job.status === 'cancelled' || job.status === 'completed'

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="max-w-2xl mx-auto px-6 py-8 space-y-6">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold text-white truncate">{job.name}</h2>
              <StatusBadge job={job} />
            </div>
            <p className="text-xs text-zinc-500 mt-1">
              {providerLabel} · rank {job.config.rank} · {job.config.steps} steps · {job.config.preset}
            </p>
          </div>
          <div className="flex items-center gap-1 flex-shrink-0">
            {job.target?.remoteJobId && (
              <button
                onClick={() => onOpenLogs(job)}
                title="View logs"
                className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800"
              >
                <ScrollText className="h-4 w-4" />
              </button>
            )}
            {job.status === 'completed' && (
              <button
                onClick={() => onPublish(job)}
                className="text-xs px-3 py-1.5 rounded-md bg-blue-600 text-white hover:bg-blue-500 flex items-center gap-1.5"
              >
                <Cloud className="h-3.5 w-3.5" /> Publish…
              </button>
            )}
            {active ? (
              <button
                onClick={() => onCancel(job.id)}
                disabled={job.cancelRequested}
                className="text-xs px-3 py-1.5 rounded-md bg-zinc-800 text-zinc-200 hover:bg-zinc-700 disabled:opacity-40 flex items-center gap-1.5"
              >
                <XCircle className="h-3.5 w-3.5" /> {job.cancelRequested ? 'Cancelling…' : 'Cancel'}
              </button>
            ) : (
              <>
                {canResume && (
                  <button
                    onClick={() => onResume(job.id)}
                    title="Re-run from the last saved checkpoint (reuses the preprocessed dataset)"
                    className="text-xs px-3 py-1.5 rounded-md bg-blue-600 text-white hover:bg-blue-500 flex items-center gap-1.5"
                  >
                    <RotateCcw className="h-3.5 w-3.5" /> Resume
                  </button>
                )}
                {canReset && (
                  <button
                    onClick={() => setResetConfirmOpen(true)}
                    title="Clear all progress and retrain from scratch"
                    className="text-xs px-3 py-1.5 rounded-md bg-zinc-800 text-zinc-200 hover:bg-zinc-700 flex items-center gap-1.5"
                  >
                    <RotateCcw className="h-3.5 w-3.5" /> Reset
                  </button>
                )}
                <button
                  onClick={() => onArchive(job.id)}
                  title="Archive run"
                  className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800"
                >
                  <Archive className="h-4 w-4" />
                </button>
                <button
                  onClick={() => onDelete(job.id)}
                  title="Delete run"
                  className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-500 hover:text-red-400 hover:bg-zinc-800"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </>
            )}
          </div>
        </div>

        {/* Success call-to-action first, then the full run details — both at the
            top so a finished run greets the user with the result + next steps
            above the validation feed. */}
        {job.status === 'completed' && job.localLoraPath && (
          <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-4 flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-emerald-400 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-emerald-300 font-medium">LoRA ready</p>
              <p className="text-[11px] text-zinc-500 font-mono truncate">{job.localLoraPath}</p>
            </div>
            <button
              onClick={() => window.electronAPI.showItemInFolder({ filePath: job.localLoraPath as string })}
              className="text-xs px-3 py-1.5 rounded-lg bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 flex items-center gap-1.5 whitespace-nowrap"
            >
              <FolderOpen className="h-3.5 w-3.5" /> Reveal file
            </button>
            <button
              onClick={() => onTryInGenSpace(job)}
              className="text-xs px-3 py-1.5 rounded-lg bg-violet-600 text-white hover:bg-violet-500 flex items-center gap-1.5 whitespace-nowrap"
            >
              <Sparkles className="h-3.5 w-3.5" /> Try in Gen Space
            </button>
          </div>
        )}

        {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') && (
          <PostRunPodBanner
            job={job}
            dataset={dataset}
            work={podWork}
            onOpenWork={onOpenPodWork}
          />
        )}

        {/* Full run details once it's no longer pending/running — time taken,
            GPU, steps, rank, dataset, etc. (also persisted to <id>.run-summary.md
            next to the adapter on disk). Shown at the top so a finished run's
            summary is the first thing visible, above the validation feed. */}
        {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') &&
          job.startedAt && (
            <RunSummary job={job} dataset={dataset} preprocessed={preprocessed} />
          )}

        {active && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>{job.statusDetail ?? (job.status === 'pending' ? 'Queued to train…' : 'Preparing training…')}</span>
              {progress != null && (
                <span>
                  step {job.currentStep ?? 0}/{job.totalSteps} · {progress}%
                  {job.status === 'running' && job.etaSeconds != null ? ` · ${formatEta(job.etaSeconds)}` : ''}
                </span>
              )}
            </div>
            <div className="h-2 w-full rounded-full bg-zinc-800 overflow-hidden">
              {progress == null ? (
                <div className="h-full w-1/3 rounded-full bg-blue-500/80 animate-pulse" />
              ) : (
                <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${progress}%` }} />
              )}
            </div>
          </div>
        )}

        {active && <GpuStatusPanel job={job} />}

        <ValidationFeedPanel job={job} />

        {job.error && (
          <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-4">
            <p className="text-sm text-red-300 font-medium mb-1">Run failed</p>
            <p className="text-[11px] text-red-400 whitespace-pre-wrap break-words font-mono">{job.error}</p>
            {/* A download-step failure means training finished and the adapter
                still lives on the network volume — offer a no-retrain recovery. */}
            {job.remoteOutputDir && /download failed|artifact not found/i.test(job.error) && (
              <button
                onClick={() => onRetryDownload(job.id)}
                className="mt-3 text-xs px-3 py-1.5 rounded-lg bg-blue-600 text-white hover:bg-blue-500 flex items-center gap-1.5"
              >
                <Download className="h-3.5 w-3.5" /> Retry download
              </button>
            )}
            {onChooseAnotherGpu && /gpu_selection_required|gpu (?:selection|capacity)|capacity changed|out of stock/i.test(job.error) && (
              <button
                onClick={() => onChooseAnotherGpu(job)}
                className="mt-3 rounded-lg bg-amber-500 px-3 py-1.5 text-xs font-medium text-zinc-950 hover:bg-amber-400"
              >
                Choose another GPU and continue
              </button>
            )}
          </div>
        )}
      </div>

      {resetConfirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-md rounded-lg bg-zinc-900 border border-zinc-700 p-5 shadow-xl">
            <div className="flex items-center gap-2 mb-2">
              <RotateCcw className="h-4 w-4 text-amber-400" />
              <h3 className="text-sm font-semibold text-white">Reset training run?</h3>
            </div>
            <p className="text-xs text-zinc-400 mb-3">
              This clears all progress for <span className="text-zinc-200 font-medium">{job.name}</span> and
              retrains from step 0. The preprocessed dataset is kept, but the following will be deleted:
            </p>
            <ul className="text-xs text-zinc-400 space-y-1.5 mb-4">
              <li className="flex gap-2">
                <Trash2 className="h-3.5 w-3.5 mt-0.5 text-red-400 shrink-0" />
                <span>
                  The remote training output on the GPU workspace — all saved checkpoints and
                  validation samples.
                </span>
              </li>
              <li className="flex gap-2">
                <Trash2 className="h-3.5 w-3.5 mt-0.5 text-red-400 shrink-0" />
                <span>
                  The local run folder — the downloaded LoRA weights, run summary, and validation
                  media.
                </span>
              </li>
              <li className="flex gap-2">
                <Trash2 className="h-3.5 w-3.5 mt-0.5 text-red-400 shrink-0" />
                <span>Progress counters and the validation feed (reset to zero).</span>
              </li>
            </ul>
            <p className="text-[11px] text-zinc-500 mb-4">
              Training will restart from step 0 using the same config. This cannot be undone.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setResetConfirmOpen(false)}
                className="text-xs px-3 py-1.5 rounded-md bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setResetConfirmOpen(false)
                  onReset(job.id)
                }}
                className="text-xs px-3 py-1.5 rounded-md bg-red-600 text-white hover:bg-red-500 flex items-center gap-1.5"
              >
                <Trash2 className="h-3.5 w-3.5" /> Reset & retrain
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}
