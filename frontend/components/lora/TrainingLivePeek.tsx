import { useMemo } from 'react'
import { useLoraTraining, type LoraTrainingJob } from '../../contexts/LoraTrainingContext'
import { useBackendMediaUrl } from '../../lib/backend-media'

/**
 * Find the active (pending/running) training job whose preprocessed dataset
 * belongs to the given dataset. Used by the collection-side live indicator so
 * the user sees training progress without leaving the dataset view.
 */
function useActiveJobForDataset(datasetId: string | null): LoraTrainingJob | null {
  const { trainingJobs, preprocessed } = useLoraTraining()
  return useMemo(() => {
    if (!datasetId) return null
    const preIds = new Set(
      preprocessed.filter((p) => p.datasetId === datasetId).map((p) => p.id),
    )
    return (
      trainingJobs.find(
        (j) => (j.status === 'running' || j.status === 'pending') && preIds.has(j.preprocessedId),
      ) ?? null
    )
  }, [trainingJobs, preprocessed, datasetId])
}

function MiniBar({ pct }: { pct: number }) {
  const clamped = Math.max(0, Math.min(100, pct))
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-zinc-800">
      <div
        className="h-full rounded-full bg-blue-500 transition-all duration-300"
        style={{ width: `${clamped}%` }}
      />
    </div>
  )
}

/**
 * Compact live training indicator for the collection inspector: shows the
 * latest validation sample (so the user can watch the model improve) plus a
 * GPU VRAM mini-bar, while training runs in the background. Renders nothing
 * when the dataset has no active run.
 */
export function TrainingLivePeek({ datasetId }: { datasetId: string | null }) {
  const job = useActiveJobForDataset(datasetId)
  if (!job) return null

  const feed = job.validationFeed ?? []
  const latest = feed.length > 0 ? feed[feed.length - 1] : null
  const vramPct =
    job.gpuStatus && job.gpuStatus.vramTotalMb > 0
      ? (job.gpuStatus.vramUsedMb / job.gpuStatus.vramTotalMb) * 100
      : null
  const hasStarted = job.status === 'running' && (job.currentStep ?? 0) > 0

  return (
    <div className="space-y-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-zinc-200">
          {hasStarted ? 'Training live' : 'Training setup'}
        </span>
        <span className="text-[10px] tabular-nums text-zinc-500">
          step {job.currentStep ?? 0}/{job.totalSteps || '—'}
        </span>
      </div>
      {hasStarted && latest ? (
        <LatestSample mediaUrl={latest.mediaUrl} step={latest.step} />
      ) : hasStarted ? (
        <p className="text-[10px] text-zinc-500">Waiting for the first validation sample…</p>
      ) : (
        <p className="text-[10px] text-zinc-500">
          {job.statusDetail ?? (job.status === 'pending' ? 'Waiting to start…' : 'Preparing training…')}
        </p>
      )}
      {vramPct != null && job.gpuStatus ? (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px] text-zinc-500">
            <span>VRAM</span>
            <span className="tabular-nums">{Math.round(vramPct)}%</span>
          </div>
          <MiniBar pct={vramPct} />
        </div>
      ) : null}
    </div>
  )
}

function LatestSample({ mediaUrl, step }: { mediaUrl: string; step: number }) {
  const { url, error } = useBackendMediaUrl(mediaUrl)
  return (
    <div className="space-y-1">
      <div className="aspect-video w-full overflow-hidden rounded bg-black">
        {error ? (
          <div className="flex h-full items-center justify-center text-[10px] text-zinc-500">
            failed to load
          </div>
        ) : url ? (
          <video src={url} muted loop playsInline className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full items-center justify-center text-[10px] text-zinc-500">
            loading…
          </div>
        )}
      </div>
      <span className="text-[10px] tabular-nums text-zinc-500">latest · step {step}</span>
    </div>
  )
}
