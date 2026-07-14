import { ListTodo, Loader2, Sparkles } from 'lucide-react'
import { useQueue } from '../contexts/QueueContext'
import { useLoraTraining } from '../contexts/LoraTrainingContext'

/**
 * Header icon + active-count badge for the durable batch queue.
 *
 * Sits next to the Settings/Logs icons in the App-level top-right
 * controls strip. Click toggles the QueueSidePanel via the queue context.
 *
 * The count combines the general durable queue (pending + running) with the
 * LoRA Trainer's active derivation jobs, so background dataset-prep work is
 * visible from anywhere in the app without opening the panel.
 *
 * Three visual states:
 *   1. General queue running — wider emerald pill with a spinner, the active
 *      count, and the live progress percentage (e.g. "3 · 53%").
 *   2. Only LoRA derivation jobs running — blue pill with a spinner and the
 *      active count (no per-job progress to show here).
 *   3. Idle — list icon + emerald count badge for pending+running across both
 *      queues.
 *
 * Why a custom button rather than the shadcn `Button` primitive: the
 * existing Logs/Settings icons in App.tsx use raw `<button>` with the
 * same Tailwind styling, so matching that pattern keeps the header
 * strip visually aligned even when the badge expands into a pill.
 */

// Mirrors `LORA_ACTIVE` in QueueSidePanel. Review-paused work is included
// because it requires user action and must not disappear from the badge.
const LORA_ACTIVE_STATUSES = new Set(['pending', 'editing', 'review', 'approved', 'generating'])

export function QueueBadgeButton() {
  const { activeCount, runningCount, isPanelOpen, setIsPanelOpen, runningProgress } = useQueue()
  const { derivationJobs } = useLoraTraining()

  const loraActiveCount = derivationJobs.filter((j) =>
    LORA_ACTIVE_STATUSES.has(j.status),
  ).length
  const totalActive = activeCount + loraActiveCount

  const handleClick = () => setIsPanelOpen(!isPanelOpen)

  if (runningCount > 0) {
    const pct = Math.max(0, Math.min(100, runningProgress?.progress ?? 0))
    return (
      <button
        data-tour="lora-queue"
        type="button"
        onClick={handleClick}
        className="relative h-8 px-2.5 flex items-center gap-1.5 rounded-md border border-emerald-500/40 bg-emerald-500/10 text-emerald-200 hover:bg-emerald-500/20 transition-colors"
        title={`Generation queue (${totalActive} active, ${pct.toFixed(0)}%)`}
      >
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        <span className="text-[11px] font-mono leading-none">
          {totalActive} · {pct.toFixed(0)}%
        </span>
      </button>
    )
  }

  if (loraActiveCount > 0) {
    return (
      <button
        data-tour="lora-queue"
        type="button"
        onClick={handleClick}
        className="relative h-8 px-2.5 flex items-center gap-1.5 rounded-md border border-blue-500/40 bg-blue-500/10 text-blue-200 hover:bg-blue-500/20 transition-colors"
        title={`LoRA generation jobs (${loraActiveCount} active)`}
      >
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        <Sparkles className="h-3 w-3" />
        <span className="text-[11px] font-mono leading-none">{loraActiveCount}</span>
      </button>
    )
  }

  return (
    <button
      data-tour="lora-queue"
      type="button"
      onClick={handleClick}
      className="relative h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
      title={
        totalActive === 0
          ? 'Generation queue (empty)'
          : `Generation queue (${totalActive} pending)`
      }
    >
      <ListTodo className="h-4 w-4" />
      {totalActive > 0 && (
        <span
          className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 rounded-full bg-emerald-500 text-[10px] font-semibold text-zinc-900 flex items-center justify-center"
          aria-label={`${totalActive} items in queue`}
        >
          {totalActive > 99 ? '99+' : totalActive}
        </span>
      )}
    </button>
  )
}
