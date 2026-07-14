import { useMemo } from 'react'
import { FolderOpen } from 'lucide-react'
import type { LoraTrainingJob } from '../../contexts/LoraTrainingContext'
import { useBackendMediaUrl } from '../../lib/backend-media'

type FeedItem = NonNullable<LoraTrainingJob['validationFeed']>[number]
type Checkpoint = NonNullable<LoraTrainingJob['checkpoints']>[number]

function SourceBadge({ source }: { source: FeedItem['source'] }) {
  if (source === 'holdout') {
    return (
      <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
        holdout
      </span>
    )
  }
  return (
    <span className="rounded bg-violet-500/15 px-1.5 py-0.5 text-[10px] font-medium text-violet-300">
      prompt
    </span>
  )
}

function FeedCard({ item }: { item: FeedItem }) {
  const { url, error } = useBackendMediaUrl(item.mediaUrl)
  // The backend sets `referenceMediaUrl` to the literal sentinel "staged" when
  // the sample was reference-conditioned (the reference video lives on the
  // remote pod, not locally — there's no backend route to serve it). So we show
  // a static "reference input" indicator instead of trying to fetch a URL.
  const hasReference = item.referenceMediaUrl != null

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/40">
      <div className={`relative w-full bg-black ${item.mediaType === 'audio' ? 'flex h-24 items-center px-3' : 'aspect-video'}`}>
        {error ? (
          <div className="flex h-full items-center justify-center text-[11px] text-zinc-500">
            Failed to load
          </div>
        ) : url && item.mediaType === 'audio' ? (
          <audio src={url} controls className="w-full" />
        ) : url ? (
          <video
            src={url}
            controls
            muted
            loop
            playsInline
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-[11px] text-zinc-500">
            loading…
          </div>
        )}
      </div>
      <div className="space-y-1 p-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] tabular-nums text-zinc-400">sample {item.sampleIndex}</span>
          <SourceBadge source={item.source} />
        </div>
        {item.prompt && (
          <p className="line-clamp-2 text-xs leading-snug text-zinc-400">{item.prompt}</p>
        )}
        {hasReference && (
          <p className="pt-1 text-[10px] text-zinc-500">conditioned on a reference input</p>
        )}
      </div>
    </div>
  )
}

function CheckpointCard({ checkpoint }: { checkpoint: Checkpoint }) {
  const filename = checkpoint.localPath.split(/[\\/]/).pop() ?? checkpoint.localPath
  return (
    <div className="flex flex-col justify-between gap-3 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300">
            checkpoint
          </span>
          <span className="text-[11px] tabular-nums text-zinc-400">step {checkpoint.step}</span>
        </div>
        <p className="break-all text-[11px] leading-snug text-zinc-500" title={checkpoint.localPath}>
          {filename}
        </p>
      </div>
      <button
        type="button"
        onClick={() => window.electronAPI.showItemInFolder({ filePath: checkpoint.localPath })}
        className="inline-flex items-center justify-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-800/60 px-2.5 py-1.5 text-[11px] font-medium text-zinc-200 transition hover:bg-zinc-700/60"
      >
        <FolderOpen className="h-3.5 w-3.5" /> Reveal file
      </button>
    </div>
  )
}

/**
 * In-training results: adapter checkpoints + validation samples, grouped by
 * step (newest first). The trainer saves a `lora_weights_step_N.safetensors`
 * checkpoint at the checkpoint interval and emits validation samples at the
 * validation interval; both are downloaded live and paired here by step so the
 * user can reveal a checkpoint next to the outputs it produced. Holds a
 * compact empty state until the first artifacts land.
 */
export function ValidationFeedPanel({ job }: { job: LoraTrainingJob }) {
  const items = job.validationFeed ?? []
  const checkpoints = job.checkpoints ?? []

  // Union of steps that have either a checkpoint or a validation sample,
  // descending. Within a step, validation samples sort by sampleIndex and the
  // checkpoint (if any) leads the row.
  const groups = useMemo(() => {
    const byStep = new Map<number, FeedItem[]>()
    for (const it of items) {
      const arr = byStep.get(it.step) ?? []
      arr.push(it)
      byStep.set(it.step, arr)
    }
    for (const arr of byStep.values()) {
      arr.sort((a, b) => a.sampleIndex - b.sampleIndex)
    }
    const ckptByStep = new Map<number, Checkpoint>()
    for (const c of checkpoints) {
      ckptByStep.set(c.step, c)
    }
    const steps = new Set<number>([...byStep.keys(), ...ckptByStep.keys()])
    return [...steps]
      .sort((a, b) => b - a)
      .map((step) => [step, byStep.get(step) ?? [], ckptByStep.get(step)] as const)
  }, [items, checkpoints])

  const totalArtifacts = items.length + checkpoints.length

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-zinc-200">Checkpoints &amp; validation</h3>
        <span className="text-[11px] text-zinc-500">
          {totalArtifacts} artifact{totalArtifacts === 1 ? '' : 's'}
        </span>
      </div>
      {groups.length === 0 ? (
        <div className="rounded-lg border border-dashed border-zinc-800 bg-zinc-900/30 p-6 text-center">
          <p className="text-xs text-zinc-500">
            Checkpoints and validation samples appear here as training progresses —
            a checkpoint at step {job.config?.checkpointInterval ?? 250} and samples
            at step {job.config?.validationInterval ?? 50}, {'…'}
          </p>
        </div>
      ) : (
        <div className="space-y-5">
          {groups.map(([step, arr, checkpoint]) => (
            <div key={step} className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-medium tabular-nums text-zinc-300">
                  step {step}
                </span>
                <span className="h-px flex-1 bg-zinc-800" />
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {checkpoint && <CheckpointCard checkpoint={checkpoint} />}
                {arr.map((it) => (
                  <FeedCard key={`${it.step}-${it.sampleIndex}`} item={it} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
