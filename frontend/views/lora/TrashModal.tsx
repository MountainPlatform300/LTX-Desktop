import { useState } from 'react'
import { RotateCcw, Trash2, X } from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import type { StudioClip } from '../studio/studio-store'

function clipName(clip: StudioClip): string {
  return clip.caption.trim() || clip.localPath.split('/').pop() || clip.localPath
}

function deletedAgo(iso: string | null): string {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000))
  if (secs < 60) return 'just now'
  const mins = Math.round(secs / 60)
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.round(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.round(hrs / 24)
  return `${days}d ago`
}

/**
 * The dataset's recycle bin. Clips here are soft-deleted: hidden from the
 * gallery and excluded from pairing, readiness, training and export. They can
 * be restored (back to the gallery) or permanently deleted. "Empty trash" is
 * guarded by an inline confirm since permanent deletion can't be undone.
 */
export function TrashModal({
  clips,
  onRestore,
  onPurge,
  onClose,
}: {
  clips: StudioClip[]
  onRestore: (ids: string[]) => void
  onPurge: (ids: string[]) => void
  onClose: () => void
}) {
  const [confirmEmpty, setConfirmEmpty] = useState(false)
  const allIds = clips.map((c) => c.id)

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-lg mx-4 flex flex-col max-h-[80vh]">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold text-white">Recycle bin</h2>
            <span className="text-xs text-zinc-500">
              {clips.length} clip{clips.length === 1 ? '' : 's'}
            </span>
          </div>
          <button
            onClick={onClose}
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 pt-3 pb-2 flex items-center justify-between gap-2 border-b border-zinc-800/60">
          <p className="text-[11px] text-zinc-500 leading-snug">
            Removed clips are kept here and excluded from training &amp; export. Restore to bring
            them back, or delete permanently to free space.
          </p>
          {clips.length > 0 && (
            <div className="flex items-center gap-1.5 shrink-0">
              <button
                onClick={() => onRestore(allIds)}
                className="text-[11px] px-2.5 py-1 rounded-md border border-zinc-700 text-zinc-200 hover:bg-zinc-800 flex items-center gap-1"
              >
                <RotateCcw className="h-3 w-3" />
                Restore all
              </button>
              {confirmEmpty ? (
                <button
                  onClick={() => {
                    onPurge(allIds)
                    setConfirmEmpty(false)
                  }}
                  className="text-[11px] px-2.5 py-1 rounded-md bg-red-600 hover:bg-red-500 text-white flex items-center gap-1"
                >
                  <Trash2 className="h-3 w-3" />
                  Delete forever?
                </button>
              ) : (
                <button
                  onClick={() => setConfirmEmpty(true)}
                  className="text-[11px] px-2.5 py-1 rounded-md border border-red-500/40 text-red-300 hover:bg-red-500/10 flex items-center gap-1"
                >
                  <Trash2 className="h-3 w-3" />
                  Empty trash
                </button>
              )}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-3">
          {clips.length === 0 ? (
            <div className="py-12 text-center text-zinc-500 text-sm">The recycle bin is empty.</div>
          ) : (
            <ul className="space-y-1.5">
              {clips.map((clip) => {
                const url = clip.posterPath ? pathToFileUrl(clip.posterPath) : null
                return (
                  <li
                    key={clip.id}
                    className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-800/30 px-2.5 py-2"
                  >
                    <div
                      className="h-11 w-16 shrink-0 rounded-md bg-zinc-950 bg-cover bg-center border border-zinc-700"
                      style={url ? { backgroundImage: `url("${url}")` } : {}}
                    />
                    <div className="min-w-0 flex-1">
                      <p className="text-xs text-zinc-200 truncate">{clipName(clip)}</p>
                      <p className="text-[11px] text-zinc-500">Removed {deletedAgo(clip.deletedAt)}</p>
                    </div>
                    <button
                      onClick={() => onRestore([clip.id])}
                      title="Restore to gallery"
                      className="h-7 px-2 flex items-center gap-1 rounded-md text-zinc-300 hover:text-emerald-300 hover:bg-zinc-800 text-[11px]"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      Restore
                    </button>
                    <button
                      onClick={() => onPurge([clip.id])}
                      title="Delete permanently"
                      className="h-7 w-7 flex items-center justify-center rounded-md text-zinc-500 hover:text-red-400 hover:bg-zinc-800"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
