import { useMemo, useRef, useState } from 'react'
import {
  AlertCircle,
  Ban,
  CheckCircle2,
  Clock,
  Film,
  GripVertical,
  Image as ImageIcon,
  ListTodo,
  Loader2,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react'
import { Button } from './ui/button'
import { useQueue, type QueueItem, type QueueItemStatus, type VideoQueuePayload } from '../contexts/QueueContext'
import { useLoraTraining, type DerivationJob } from '../contexts/LoraTrainingContext'
import { useProjects } from '../contexts/ProjectContext'
import { cn } from '@/lib/utils'

/**
 * Side panel for the durable generation queue, toggled from the header
 * `QueueBadgeButton`. Owns no state of its own — it renders straight off
 * the `QueueContext` polling snapshot and calls back into the context's
 * mutation helpers, which refetch on success so the panel updates within
 * a poll tick.
 *
 * Layout: a right-anchored dropdown with a header (title + counts +
 * pause/resume + close), a status filter strip, a compact "add to queue"
 * composer, and a scrollable item list. Pending rows are drag-reorderable
 * (HTML5 DnD) and dispatch a single `reorder` call with the full new
 * pending permutation; the backend validates the permutation atomically.
 */

type FilterId = 'all' | 'pending' | 'running' | 'completed' | 'failed'

const FILTERS: { id: FilterId; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'pending', label: 'Pending' },
  { id: 'running', label: 'Running' },
  { id: 'completed', label: 'Done' },
  { id: 'failed', label: 'Failed' },
]

// Review-paused jobs are active work that needs user attention, so keep them
// visible/countable in the app-wide queue alongside executing jobs.
const LORA_ACTIVE: ReadonlyArray<DerivationJob['status']> = [
  'pending',
  'editing',
  'review',
  'approved',
  'generating',
]
const LORA_STATUS_LABEL: Record<DerivationJob['status'], string> = {
  pending: 'Queued',
  editing: 'Editing frame…',
  review: 'Awaiting review',
  approved: 'Queued for video…',
  generating: 'Generating…',
  completed: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export function QueueSidePanel() {
  const {
    state,
    isPanelOpen,
    setIsPanelOpen,
    pendingCount,
    runningCount,
    runningProgress,
    enqueue,
    cancelPending,
    cancelRunning,
    removeItem,
    requeue,
    reorder,
    pause,
    resume,
    clearCompleted,
    clearFailed,
  } = useQueue()
  const { activeProject } = useProjects()
  const activeProjectId = activeProject?.id ?? null
  const {
    derivationJobs,
    datasets: loraDatasets,
    cancelDerivation,
    cancelAllDerivations,
    retryDerivation,
    dismissDerivation,
  } = useLoraTraining()

  const [tab, setTab] = useState<'general' | 'lora'>('general')
  const [filter, setFilter] = useState<FilterId>('all')
  const [draftPrompt, setDraftPrompt] = useState('')
  const [adding, setAdding] = useState(false)

  // Drag-reorder: tracks the source index within the pending subset. -1
  // means no drag in progress. Kept in a ref (not state) so DnD events
  // don't trigger re-renders on every dragover.
  const dragFromRef = useRef<number>(-1)

  const pendingItems = useMemo(
    () => state.items.filter((i) => i.status === 'pending'),
    [state.items],
  )

  const visibleItems = useMemo(() => {
    if (filter === 'all') return state.items
    if (filter === 'pending') return pendingItems
    return state.items.filter((i) => i.status === filter)
  }, [state.items, pendingItems, filter])

  const counts = useMemo(() => {
    const c: Record<QueueItemStatus, number> = {
      pending: 0,
      running: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
    }
    for (const i of state.items) c[i.status] += 1
    return c
  }, [state.items])

  const loraVisibleJobs = derivationJobs
  const loraActiveCount = useMemo(
    () => loraVisibleJobs.filter((j) => LORA_ACTIVE.includes(j.status)).length,
    [loraVisibleJobs],
  )
  const loraDatasetName = useMemo(() => {
    const byId = new Map(loraDatasets.map((d) => [d.id, d.name]))
    return (datasetId: string | undefined | null) =>
      (datasetId && byId.get(datasetId)) || 'Dataset'
  }, [loraDatasets])

  if (!isPanelOpen) return null

  const handleAdd = async () => {
    const prompt = draftPrompt.trim()
    if (!prompt) return
    setAdding(true)
    const payload: VideoQueuePayload = {
      kind: 'video',
      request: {
        prompt,
        model: 'fast',
        duration: 5,
        resolution: '1080p',
        fps: 24,
        audio: false,
        cameraMotion: 'none',
        negativePrompt: '',
        aspectRatio: '16:9',
      },
    }
    const result = await enqueue(payload, {
      originatingProjectId: activeProjectId ?? undefined,
      source: 'queue_manual',
    })
    setAdding(false)
    if (result.ok) setDraftPrompt('')
  }

  const handleDrop = (targetPendingIndex: number) => {
    const from = dragFromRef.current
    dragFromRef.current = -1
    if (from === -1 || from === targetPendingIndex) return
    const ids = pendingItems.map((i) => i.id)
    const [moved] = ids.splice(from, 1)
    ids.splice(targetPendingIndex, 0, moved)
    void reorder(ids)
  }

  return (
    <>
      {/* Click-away backdrop. */}
      <div
        className="fixed inset-0 z-[90]"
        onClick={() => setIsPanelOpen(false)}
      />
      <aside
        className="fixed right-3 top-14 z-[100] w-[380px] max-h-[calc(100vh-5rem)] flex flex-col rounded-lg border border-zinc-800 bg-zinc-900/95 backdrop-blur-sm shadow-2xl"
        role="dialog"
        aria-label="Generation queue"
      >
        {/* Header */}
        <div className="flex items-center gap-2 px-3 py-2.5 border-b border-zinc-800">
          <span className="text-sm font-semibold text-zinc-100">Queue</span>
          <span className="text-[11px] font-mono text-zinc-500">
            {pendingCount + runningCount} active
          </span>
          <div className="ml-auto flex items-center gap-1">
            {state.paused ? (
              <IconAction label="Resume queue" onClick={resume}>
                <Play className="h-3.5 w-3.5" />
              </IconAction>
            ) : (
              <IconAction
                label={pendingCount === 0 ? 'Pause queue' : `Pause queue (${pendingCount} pending)`}
                onClick={pause}
                disabled={pendingCount === 0 && runningCount === 0}
              >
                <Pause className="h-3.5 w-3.5" />
              </IconAction>
            )}
            <IconAction label="Close" onClick={() => setIsPanelOpen(false)}>
              <X className="h-4 w-4" />
            </IconAction>
          </div>
        </div>

        {/* Tab switch: general durable queue vs LoRA Trainer derivation jobs. */}
        <div className="flex items-center gap-1 px-2 py-1.5 border-b border-zinc-800">
          <TabButton active={tab === 'general'} onClick={() => setTab('general')}>
            <ListTodo className="h-3.5 w-3.5" /> Queue
            {pendingCount + runningCount > 0 && <span className="ml-1 text-zinc-500">{pendingCount + runningCount}</span>}
          </TabButton>
          <TabButton active={tab === 'lora'} onClick={() => setTab('lora')}>
            <Sparkles className="h-3.5 w-3.5" /> LoRA Trainer
            {loraActiveCount > 0 && <span className="ml-1 text-zinc-500">{loraActiveCount}</span>}
          </TabButton>
        </div>

        {tab === 'general' && (
        <>
        {/* Status filter strip */}
        <div className="flex items-center gap-1 px-3 py-2 border-b border-zinc-800 overflow-x-auto">
          {FILTERS.map((f) => {
            const count =
              f.id === 'all'
                ? state.items.length
                : f.id === 'pending'
                  ? counts.pending
                  : f.id === 'running'
                    ? counts.running
                    : f.id === 'completed'
                      ? counts.completed
                      : counts.failed
            const active = filter === f.id
            return (
              <button
                key={f.id}
                type="button"
                onClick={() => setFilter(f.id)}
                className={cn(
                  'px-2 py-1 rounded-md text-[11px] font-medium transition-colors whitespace-nowrap',
                  active
                    ? 'bg-zinc-700 text-zinc-100'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800',
                )}
              >
                {f.label}
                {count > 0 && <span className="ml-1 text-zinc-500">{count}</span>}
              </button>
            )
          })}
        </div>

        {/* Compact add-to-queue composer */}
        <form
          className="flex items-center gap-1.5 px-3 py-2 border-b border-zinc-800"
          onSubmit={(e) => {
            e.preventDefault()
            void handleAdd()
          }}
        >
          <input
            type="text"
            value={draftPrompt}
            onChange={(e) => setDraftPrompt(e.target.value)}
            placeholder="Add a prompt to the queue…"
            className="flex-1 h-8 px-2 rounded-md bg-zinc-800 text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
          />
          <Button
            type="submit"
            size="icon"
            variant="secondary"
            className="h-8 w-8"
            disabled={adding || !draftPrompt.trim()}
            title="Add to queue"
          >
            {adding ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-4 w-4" />}
          </Button>
        </form>

        {/* Item list */}
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
          {visibleItems.length === 0 && (
            <EmptyState filter={filter} totalCount={state.items.length} />
          )}
          {visibleItems.map((item) => {
            const pendingIndex = pendingItems.findIndex((i) => i.id === item.id)
            const isPending = item.status === 'pending'
            return (
              <QueueRow
                key={item.id}
                item={item}
                progress={runningProgress}
                draggable={isPending}
                onDragStart={() => {
                  dragFromRef.current = pendingIndex
                }}
                onDrop={() => handleDrop(pendingIndex)}
                onCancelPending={() => void cancelPending(item.id)}
                onCancelRunning={() => void cancelRunning()}
                onRemove={() => void removeItem(item.id)}
                onRequeue={() => void requeue(item.id)}
              />
            )
          })}
        </div>

        {/* Footer: history cleanup */}
        {(counts.completed > 0 || counts.failed > 0) && (
          <div className="flex items-center gap-2 px-3 py-2 border-t border-zinc-800">
            {counts.completed > 0 && (
              <Button size="sm" variant="ghost" className="h-7 text-[11px]" onClick={() => void clearCompleted()}>
                Clear done ({counts.completed})
              </Button>
            )}
            {counts.failed > 0 && (
              <Button size="sm" variant="ghost" className="h-7 text-[11px] text-red-400 hover:text-red-300" onClick={() => void clearFailed()}>
                Clear failed ({counts.failed})
              </Button>
            )}
          </div>
        )}
        </>
        )}

        {tab === 'lora' && (
          <LoraQueueTab
            jobs={loraVisibleJobs}
            activeCount={loraActiveCount}
            datasetName={loraDatasetName}
            onCancel={(id) => void cancelDerivation(id)}
            onCancelAll={() => void cancelAllDerivations()}
            onRetry={(id) => void retryDerivation(id)}
            onDismiss={(id) => void dismissDerivation(id)}
          />
        )}
      </aside>
    </>
  )
}

// ---------------------------------------------------------------------------
// Row
// ---------------------------------------------------------------------------

interface QueueRowProps {
  item: QueueItem
  progress: { progress: number; phase: string; currentStep: number | null; totalSteps: number | null } | null
  draggable: boolean
  onDragStart: () => void
  onDrop: () => void
  onCancelPending: () => void
  onCancelRunning: () => void
  onRemove: () => void
  onRequeue: () => void
}

function QueueRow({
  item,
  progress,
  draggable,
  onDragStart,
  onDrop,
  onCancelPending,
  onCancelRunning,
  onRemove,
  onRequeue,
}: QueueRowProps) {
  const isVideo = item.payload.kind === 'video' || item.payload.kind === 'lora'
  const isRunning = item.status === 'running'
  const isPending = item.status === 'pending'
  const pct = isRunning ? Math.max(0, Math.min(100, progress?.progress ?? 0)) : 0
  const prompt = payloadDisplayPrompt(item.payload)

  return (
    <div
      draggable={draggable}
      onDragStart={onDragStart}
      onDragOver={(e) => {
        if (draggable) e.preventDefault()
      }}
      onDrop={onDrop}
      className={cn(
        'group rounded-md border px-2 py-2 transition-colors',
        'border-zinc-800 bg-zinc-800/40 hover:bg-zinc-800/70',
        isRunning && 'border-emerald-500/40 bg-emerald-500/5',
        item.status === 'failed' && 'border-red-500/30 bg-red-500/5',
        draggable && 'cursor-grab active:cursor-grabbing',
      )}
    >
      <div className="flex items-start gap-2">
        {draggable ? (
          <GripVertical className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-600 group-hover:text-zinc-400" />
        ) : (
          <StatusIcon status={item.status} />
        )}

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <KindBadge isVideo={isVideo} />
            <span className="truncate text-xs text-zinc-200" title={prompt}>
              {prompt}
            </span>
          </div>

          <MetaLine item={item} />

          {isRunning && (
            <div className="mt-1.5">
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-700">
                <div
                  className="h-full bg-emerald-500 transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="mt-1 flex items-center justify-between text-[10px] text-zinc-500">
                <span>{progress?.phase || 'working'}</span>
                <span className="font-mono">{pct.toFixed(0)}%</span>
              </div>
            </div>
          )}

          {item.status === 'failed' && item.error && (
            <p className="mt-1 text-[11px] text-red-400/90 line-clamp-2" title={item.error}>
              {item.error}
            </p>
          )}
        </div>

        {/* Row actions */}
        <div className="flex shrink-0 items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          {isPending && (
            <IconAction label="Cancel" onClick={onCancelPending} small>
              <Ban className="h-3.5 w-3.5" />
            </IconAction>
          )}
          {isRunning && (
            <IconAction label="Cancel generation" onClick={onCancelRunning} small>
              <Ban className="h-3.5 w-3.5" />
            </IconAction>
          )}
          {(item.status === 'completed' || item.status === 'failed' || item.status === 'cancelled') && (
            <IconAction label="Re-queue" onClick={onRequeue} small>
              <RotateCcw className="h-3.5 w-3.5" />
            </IconAction>
          )}
          {!isRunning && (
            <IconAction label="Remove" onClick={onRemove} small>
              <Trash2 className="h-3.5 w-3.5" />
            </IconAction>
          )}
        </div>
      </div>
    </div>
  )
}

function MetaLine({ item }: { item: QueueItem }) {
  if (item.payload.kind === 'video') {
    const r = item.payload.request
    return (
      <span className="mt-0.5 block text-[10px] text-zinc-500">
        {r.resolution} · {r.duration}s · {r.fps}fps{r.imagePath ? ' · i2v' : r.audioPath ? ' · a2v' : ''}
      </span>
    )
  }
  if (item.payload.kind === 'lora') {
    const r = item.payload.request
    return (
      <span className="mt-0.5 block text-[10px] text-zinc-500">
        LoRA · {r.variant === 'standard' ? 'style' : r.variant === 'union_control' ? 'control' : 'ref video'}
      </span>
    )
  }
  const r = item.payload.request
  return (
    <span className="mt-0.5 block text-[10px] text-zinc-500">
      {r.width}×{r.height} · {r.numSteps} steps{r.numImages > 1 ? ` · ×${r.numImages}` : ''}
    </span>
  )
}

// Pull a human-readable prompt out of any queue payload kind. The LoRA
// generate request is a discriminated union; the standard variant nests the
// prompt inside `request`, while the IC-LoRA variants carry it at the top.
function payloadDisplayPrompt(payload: QueueItem['payload']): string {
  if (payload.kind === 'lora') {
    const r = payload.request
    return r.variant === 'video_input_ic_lora' ? r.prompt : r.request.prompt
  }
  return payload.request.prompt
}

function KindBadge({ isVideo }: { isVideo: boolean }) {
  return isVideo ? (
    <Film className="h-3 w-3 shrink-0 text-sky-400/80" />
  ) : (
    <ImageIcon className="h-3 w-3 shrink-0 text-violet-400/80" />
  )
}

function StatusIcon({ status }: { status: QueueItemStatus }) {
  switch (status) {
    case 'running':
      return <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin text-emerald-400" />
    case 'completed':
      return <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" />
    case 'failed':
      return <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-400" />
    case 'cancelled':
      return <Ban className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />
    case 'pending':
    default:
      return <Clock className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />
  }
}

function IconAction({
  label,
  onClick,
  disabled,
  small,
  children,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
  small?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'flex items-center justify-center rounded-md text-zinc-400 hover:text-zinc-100 hover:bg-zinc-700 transition-colors disabled:opacity-40 disabled:pointer-events-none',
        small ? 'h-6 w-6' : 'h-7 w-7',
      )}
    >
      {children}
    </button>
  )
}

function EmptyState({ filter, totalCount }: { filter: FilterId; totalCount: number }) {
  const msg =
    filter === 'all'
      ? totalCount === 0
        ? 'Queue is empty. Generate or add a prompt to start lining up work.'
        : 'No items.'
      : `No ${filter} items.`
  return (
    <div className="px-3 py-10 text-center text-xs text-zinc-500">
      <Clock className="mx-auto mb-2 h-6 w-6 text-zinc-700" />
      {msg}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab bar
// ---------------------------------------------------------------------------

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-colors',
        active
          ? 'bg-zinc-700 text-zinc-100'
          : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800',
      )}
    >
      {children}
    </button>
  )
}

// ---------------------------------------------------------------------------
// LoRA Trainer tab — derivation jobs (target/variant generation)
// ---------------------------------------------------------------------------

function LoraQueueTab({
  jobs,
  activeCount,
  datasetName,
  onCancel,
  onCancelAll,
  onRetry,
  onDismiss,
}: {
  jobs: DerivationJob[]
  activeCount: number
  datasetName: (datasetId: string | undefined | null) => string
  onCancel: (id: string) => void
  onCancelAll: () => void
  onRetry: (id: string) => void
  onDismiss: (id: string) => void
}) {
  return (
    <>
      {activeCount > 0 && (
        <div className="flex items-center justify-end px-3 py-2 border-b border-zinc-800">
          <button
            type="button"
            onClick={onCancelAll}
            className="text-[11px] px-2 py-1 rounded-md border border-zinc-700 text-zinc-300 hover:text-white hover:border-red-500/50 hover:bg-red-500/10 transition-colors flex items-center gap-1"
          >
            <X className="h-3 w-3" /> Cancel all LoRA jobs ({activeCount})
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {jobs.length === 0 && (
          <div className="px-3 py-10 text-center text-xs text-zinc-500">
            <Sparkles className="mx-auto mb-2 h-6 w-6 text-zinc-700" />
            No LoRA generation jobs. Generate examples from a collection to populate this queue.
          </div>
        )}
        {jobs.map((job) => {
          const isActive = LORA_ACTIVE.includes(job.status)
          const isReview = job.status === 'review'
          const isFailed = job.status === 'failed'
          const engineLabel = job.engine === 'kling' ? 'Kling' : job.engine === 'kling_o3' ? 'Kling O3' : 'LTX'
          return (
            <div
              key={job.id}
              className={cn(
                'group rounded-md border px-2 py-2 transition-colors',
                'border-zinc-800 bg-zinc-800/40 hover:bg-zinc-800/70',
                isActive && 'border-blue-500/30 bg-blue-500/5',
                isFailed && 'border-red-500/30 bg-red-500/5',
              )}
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 shrink-0">
                  {isReview ? (
                    <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
                  ) : isActive ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />
                  ) : isFailed ? (
                    <AlertCircle className="h-3.5 w-3.5 text-red-400" />
                  ) : (
                    <Sparkles className="h-3.5 w-3.5 text-zinc-500" />
                  )}
                </span>

                <div className="min-w-0 flex-1">
                  <span className="block truncate text-xs text-zinc-200">
                    {job.label || 'Generated clip'}
                  </span>
                  <span className={cn('mt-0.5 block truncate text-[10px]', isFailed ? 'text-red-400' : 'text-zinc-500')}>
                    {LORA_STATUS_LABEL[job.status]}
                    {isFailed && job.error ? ` · ${job.error}` : ''} · {engineLabel} · {datasetName(job.datasetId)}
                  </span>
                </div>

                <div className="flex shrink-0 items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                  {isActive ? (
                    <IconAction label="Cancel" onClick={() => onCancel(job.id)} small>
                      <Ban className="h-3.5 w-3.5" />
                    </IconAction>
                  ) : (
                    <>
                      {isFailed && (
                        <IconAction label="Retry" onClick={() => onRetry(job.id)} small>
                          <RotateCcw className="h-3.5 w-3.5" />
                        </IconAction>
                      )}
                      <IconAction label="Dismiss" onClick={() => onDismiss(job.id)} small>
                        <Trash2 className="h-3.5 w-3.5" />
                      </IconAction>
                    </>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}
