import { useEffect, useRef, useState } from 'react'
import { ArrowRight, Check, CheckCheck, ChevronLeft, ChevronRight, Loader2, RotateCw, Trash2, X } from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import type { DerivationJob } from '../../contexts/LoraTrainingContext'

/**
 * Focused lightbox for the edit-review phase. The Generate wizard hands off
 * here right after queuing: edits stream in (Queued -> Editing -> Ready), and
 * the user approves / redoes / discards each as it lands. Shows a large
 * before -> after so quality is actually judgeable, with prev/next, a
 * filmstrip, and Approve-all-ready for bulk.
 *
 * `jobs` (controlled by the parent's polled state) is the *edit phase* of the
 * current collection — pending / editing / review. It's non-blocking: closing
 * leaves the jobs running; the gallery banner reopens this. The modal
 * self-closes once the session has fully drained (after having shown work).
 */
export function EditReviewModal({
  jobs,
  getSourcePoster,
  onApprove,
  onRegenerate,
  onDiscard,
  onApproveAll,
  onCancelAll,
  onClose,
}: {
  jobs: DerivationJob[]
  getSourcePoster: (job: DerivationJob) => string | null
  onApprove: (id: string) => void
  onRegenerate: (id: string) => void
  onDiscard: (id: string) => void
  onApproveAll: () => void
  onCancelAll: () => void
  onClose: () => void
}) {
  const [index, setIndex] = useState(0)
  const seenRef = useRef(false)
  if (jobs.length > 0) seenRef.current = true

  // Close once the whole session has drained — but only after we've actually
  // shown some work (so the brief empty window right after opening doesn't
  // close us before the jobs land).
  useEffect(() => {
    if (seenRef.current && jobs.length === 0) onClose()
  }, [jobs.length, onClose])

  const readyCount = jobs.filter((j) => j.status === 'review').length

  const shell = (children: React.ReactNode, footer?: React.ReactNode) => (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center"
      onKeyDown={(e) => {
        e.stopPropagation()
        if (e.key === 'ArrowLeft') setIndex((i) => (i - 1 + jobs.length) % Math.max(jobs.length, 1))
        if (e.key === 'ArrowRight') setIndex((i) => (i + 1) % Math.max(jobs.length, 1))
      }}
      tabIndex={-1}
    >
      <div className="absolute inset-0 bg-black/75 backdrop-blur-sm" onClick={onClose} />
      <div className="relative flex flex-col w-full max-w-3xl mx-4 rounded-2xl border border-zinc-700/80 bg-zinc-900 shadow-2xl overflow-hidden">
        <div className="flex items-center gap-3 px-5 py-3.5 border-b border-zinc-800">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30">
            <Check className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-white leading-tight">Review edits</h2>
            <p className="text-[11px] text-zinc-500">Approve the frames before their videos generate</p>
          </div>
          <span className="ml-auto text-[11px] text-zinc-500">
            {jobs.length > 0 ? `${readyCount}/${jobs.length} ready` : ''}
          </span>
          {readyCount > 1 && (
            <button
              onClick={onApproveAll}
              className="text-[11px] px-2.5 py-1.5 rounded-lg border border-blue-500/40 bg-blue-500/10 text-blue-200 hover:bg-blue-500/20 flex items-center gap-1.5 transition-colors"
            >
              <CheckCheck className="h-3.5 w-3.5" />
              Approve all ready ({readyCount})
            </button>
          )}
          {jobs.length > 0 && (
            <button
              onClick={onCancelAll}
              title="Cancel the entire bulk generation"
              className="text-[11px] px-2.5 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-red-500/50 hover:bg-red-500/10 flex items-center gap-1.5 transition-colors"
            >
              <X className="h-3.5 w-3.5" />
              Cancel all
            </button>
          )}
          <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>
        {children}
        {footer}
      </div>
    </div>
  )

  // Brief "preparing" window: opened (e.g. straight from the wizard) before the
  // jobs have populated.
  if (jobs.length === 0) {
    return shell(
      <div className="px-5 py-12 flex flex-col items-center gap-3 text-zinc-400">
        <Loader2 className="h-6 w-6 animate-spin text-blue-400" />
        <p className="text-xs">Preparing edits…</p>
      </div>,
    )
  }

  const safeIndex = Math.min(index, jobs.length - 1)
  const job = jobs[safeIndex]
  const before = getSourcePoster(job)
  const after = job.editedFramePath
  const isReady = job.status === 'review'
  const go = (delta: number) => setIndex((i) => (i + delta + jobs.length) % jobs.length)

  return shell(
    <div className="px-5 py-4">
      <div className="flex items-stretch gap-3">
        <figure className="flex-1 min-w-0 space-y-1">
          <figcaption className="text-[10px] uppercase tracking-wide text-zinc-500">Source</figcaption>
          <div className="relative aspect-video rounded-lg overflow-hidden bg-zinc-950 ring-1 ring-zinc-800 flex items-center justify-center">
            {before ? (
              <img src={pathToFileUrl(before)} alt="source frame" className="w-full h-full object-contain" />
            ) : (
              <span className="text-[11px] text-zinc-600">No source preview</span>
            )}
          </div>
        </figure>
        <div className="flex items-center text-zinc-600">
          <ArrowRight className="h-4 w-4" />
        </div>
        <figure className="flex-1 min-w-0 space-y-1">
          <figcaption className="text-[10px] uppercase tracking-wide text-amber-300/80">Edited</figcaption>
          <div className="relative aspect-video rounded-lg overflow-hidden bg-zinc-950 ring-1 ring-amber-500/30 flex items-center justify-center">
            {isReady && after ? (
              <img src={pathToFileUrl(after)} alt="edited frame" className="w-full h-full object-contain" />
            ) : (
              <div className="flex flex-col items-center gap-1.5 text-zinc-500">
                <Loader2 className="h-5 w-5 animate-spin text-blue-400" />
                <span className="text-[10px]">{job.status === 'editing' ? 'Editing frame…' : 'Queued…'}</span>
              </div>
            )}
          </div>
        </figure>
      </div>

      {job.label && <p className="mt-3 text-xs text-zinc-400 truncate">{job.label}</p>}

      {jobs.length > 1 && (
        <div className="mt-3 flex gap-1.5 overflow-x-auto pb-1">
          {jobs.map((j, i) => (
            <button
              key={j.id}
              onClick={() => setIndex(i)}
              className={`relative w-16 shrink-0 aspect-video rounded-md overflow-hidden bg-zinc-950 ring-1 transition-colors flex items-center justify-center ${
                i === safeIndex ? 'ring-amber-400' : 'ring-zinc-800 hover:ring-zinc-600'
              }`}
            >
              {j.status === 'review' && j.editedFramePath ? (
                <img src={pathToFileUrl(j.editedFramePath)} alt="" className="w-full h-full object-contain" />
              ) : (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400/70" />
              )}
            </button>
          ))}
        </div>
      )}
    </div>,
    <div className="flex items-center justify-between gap-3 px-5 py-3.5 border-t border-zinc-800">
      <div className="flex items-center gap-1">
        {jobs.length > 1 && (
          <>
            <button onClick={() => go(-1)} title="Previous" className="h-8 w-8 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 flex items-center justify-center">
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button onClick={() => go(1)} title="Next" className="h-8 w-8 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 flex items-center justify-center">
              <ChevronRight className="h-4 w-4" />
            </button>
          </>
        )}
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={() => onDiscard(job.id)}
          className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-red-500/50 hover:bg-red-500/10 flex items-center gap-1.5 transition-colors"
        >
          <Trash2 className="h-3.5 w-3.5" /> Discard
        </button>
        <button
          onClick={() => onRegenerate(job.id)}
          disabled={!isReady}
          className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 transition-colors"
        >
          <RotateCw className="h-3.5 w-3.5" /> Redo
        </button>
        <button
          onClick={() => onApprove(job.id)}
          disabled={!isReady}
          className="text-xs px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 transition-colors"
        >
          <Check className="h-3.5 w-3.5" /> Approve
        </button>
      </div>
    </div>,
  )
}
