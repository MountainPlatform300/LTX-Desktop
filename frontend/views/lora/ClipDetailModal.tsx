import { useEffect, useMemo, useState } from 'react'
import {
  ArrowLeft,
  ArrowLeftRight,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Crop,
  Image as ImageIcon,
  Loader2,
  Sparkles,
  Trash2,
  Unlink,
  Wand2,
  X,
} from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import { formatDuration } from '../../lib/lora-quality'
import { derivePairs, groupMemberIds, pairReadiness } from '../../lib/lora-pairs'
import type { StudioClip } from '../studio/studio-store'

// Large single-clip workspace: opened by double-click / the card's expand
// button. Shows a full-size preview, inline caption editing, prev/next
// navigation through the current (filtered) gallery order, and the clip's
// authoring actions. Action buttons defer to the dedicated modals (trim,
// frame-edit, make-pair, variant) so behavior stays consistent everywhere.
export function ClipDetailModal({
  clip,
  clips,
  editable,
  canMakePair,
  onClose,
  onNavigate,
  onCaptionChange,
  onAutoCaption,
  onEdit,
  onFrameEdit,
  onMakePair,
  onVariant,
  onRemove,
  onReassignRoles,
  onUngroup,
}: {
  clip: StudioClip
  clips: StudioClip[]
  editable: boolean
  canMakePair: boolean
  onClose: () => void
  onNavigate: (id: string) => void
  onCaptionChange: (id: string, caption: string) => void
  onAutoCaption?: (id: string) => Promise<{ ok: boolean; error?: string }>
  onEdit: (id: string) => void
  onFrameEdit: (id: string) => void
  onMakePair: (id: string) => void
  onVariant: (id: string) => void
  onRemove: (id: string) => void
  /** Rebuild this clip's example with a new reference/target partition. */
  onReassignRoles?: (referenceIds: string[], targetIds: string[]) => void
  /** Dissolve the example back into loose clips. */
  onUngroup?: (ids: string[]) => void
}) {
  const isImage = clip.kind === 'image'
  const index = clips.findIndex((c) => c.id === clip.id)
  const prev = index > 0 ? clips[index - 1] : null
  const next = index >= 0 && index < clips.length - 1 ? clips[index + 1] : null

  const [caption, setCaption] = useState(clip.caption)
  const [captioning, setCaptioning] = useState(false)
  const [captionError, setCaptionError] = useState<string | null>(null)
  useEffect(() => {
    setCaption(clip.caption)
    setCaptionError(null)
  }, [clip.id, clip.caption])

  const commitCaption = () => {
    if (caption.trim() !== clip.caption) onCaptionChange(clip.id, caption.trim())
  }

  const runAutoCaption = async () => {
    if (!onAutoCaption || captioning) return
    setCaptioning(true)
    setCaptionError(null)
    const res = await onAutoCaption(clip.id)
    if (!res.ok) setCaptionError(res.error ?? 'Captioning failed')
    setCaptioning(false)
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return
      if (e.key === 'ArrowLeft' && prev) onNavigate(prev.id)
      else if (e.key === 'ArrowRight' && next) onNavigate(next.id)
      else if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [prev, next, onNavigate, onClose])

  const previewUrl = pathToFileUrl(clip.localPath)
  const posterUrl = clip.posterPath ? pathToFileUrl(clip.posterPath) : null
  const duration = clip.probe?.durationSeconds ?? clip.durationSeconds ?? null
  const resolution = clip.probe ? `${clip.probe.width}×${clip.probe.height}` : null

  // The full IC-LoRA example this clip belongs to: its reference(s) → target(s).
  const group = useMemo(() => {
    const { pairs } = derivePairs(clips)
    return pairs.find((g) => groupMemberIds(g).includes(clip.id)) ?? null
  }, [clips, clip.id])
  const isPaired = !!group
  const refs = group?.controls ?? []
  const targets = group?.targets ?? []
  const groupTone = group ? pairReadiness(group).tone : 'ready'

  // Role swaps: rebuild the example so the moved clip flips sides. Always keep
  // at least one reference and one target.
  const moveToTargets = (id: string) => {
    if (!group || !onReassignRoles) return
    const refIds = refs.map((c) => c.id).filter((x) => x !== id)
    if (refIds.length === 0) return
    onReassignRoles(refIds, [...targets.map((t) => t.id), id])
  }
  const moveToRefs = (id: string) => {
    if (!group || !onReassignRoles) return
    const tgtIds = targets.map((t) => t.id).filter((x) => x !== id)
    if (tgtIds.length === 0) return
    onReassignRoles([...refs.map((c) => c.id), id], tgtIds)
  }

  const renderMember = (member: StudioClip, side: 'ref' | 'target') => {
    const poster = member.posterPath ? pathToFileUrl(member.posterPath) : null
    const isOpen = member.id === clip.id
    const canMove = side === 'ref' ? refs.length > 1 : targets.length > 1
    const duration = member.probe ? formatDuration(member.probe.durationSeconds) : null
    return (
      <div
        key={member.id}
        onClick={() => onNavigate(member.id)}
        title="Open this clip"
        className={`group/m relative shrink-0 w-[7.5rem] rounded-lg overflow-hidden border cursor-pointer transition-colors ${
          isOpen ? 'border-blue-500 ring-1 ring-blue-500/50' : 'border-zinc-800 hover:border-zinc-600'
        }`}
      >
        <div
          className="h-16 w-full bg-zinc-950 bg-cover bg-center"
          style={poster ? { backgroundImage: `url("${poster}")` } : undefined}
        />
        <span className="absolute top-1 left-1 text-[9px] px-1 py-0.5 rounded bg-black/70 text-white uppercase tracking-wide">
          {side === 'ref' ? 'Input' : 'Output'}
        </span>
        {duration && (
          <span className="absolute bottom-1 right-1 text-[9px] px-1 py-0.5 rounded bg-black/70 text-zinc-200">
            {duration}
          </span>
        )}
        {editable && onReassignRoles && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              side === 'ref' ? moveToTargets(member.id) : moveToRefs(member.id)
            }}
            disabled={!canMove}
            title={
              side === 'ref'
                ? canMove
                  ? 'Use as output'
                  : 'Keep at least one input'
                : canMove
                  ? 'Use as input'
                  : 'Keep at least one output'
            }
            className="absolute top-1 right-1 h-5 w-5 rounded-md bg-black/70 text-white/90 hover:bg-black/90 disabled:opacity-30 disabled:hover:bg-black/70 flex items-center justify-center opacity-0 group-hover/m:opacity-100 disabled:opacity-30"
          >
            {side === 'ref' ? <ArrowRight className="h-3 w-3" /> : <ArrowLeft className="h-3 w-3" />}
          </button>
        )}
      </div>
    )
  }

  const btnNeutral =
    'w-full text-xs px-3 py-2 rounded-lg border border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 transition-colors'
  const btnPrimary =
    'w-full text-xs px-3 py-2 rounded-lg bg-blue-600/90 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 transition-colors'

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-5xl mx-4 max-h-[88vh] flex flex-col">
        <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-800">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-sm font-medium text-white truncate">
              {clip.localPath.split('/').pop()}
            </span>
            {index >= 0 && (
              <span className="text-[11px] text-zinc-500 shrink-0">
                {index + 1} / {clips.length}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => prev && onNavigate(prev.id)}
              disabled={!prev}
              title="Previous (←)"
              className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-30"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              onClick={() => next && onNavigate(next.id)}
              disabled={!next}
              title="Next (→)"
              className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-30"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
            <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 min-h-0 flex">
          <div className="flex-1 min-w-0 flex flex-col bg-black">
            <div className="flex-1 min-h-0 flex items-center justify-center p-3">
              {isImage ? (
                <img src={previewUrl} alt="clip" className="max-h-[72vh] max-w-full object-contain" />
              ) : (
                <video
                  key={clip.localPath}
                  src={previewUrl}
                  poster={posterUrl ?? undefined}
                  controls
                  autoPlay
                  loop
                  className="max-h-[72vh] max-w-full object-contain"
                />
              )}
            </div>

            {/* The full example as a horizontal filmstrip: reference(s) → target. */}
            {isPaired && (
              <div className="shrink-0 border-t border-zinc-800 bg-zinc-950/70 px-3 py-2">
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] uppercase tracking-wide text-blue-300 font-medium">Example</span>
                    <span className="flex items-center gap-1.5 text-[10px] text-zinc-500">
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${
                          groupTone === 'ready' ? 'bg-emerald-400' : groupTone === 'warn' ? 'bg-amber-400' : 'bg-red-400'
                        }`}
                      />
                      {groupTone === 'ready' ? 'Ready' : groupTone === 'warn' ? 'Review' : 'Needs fix'}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    {editable && onReassignRoles && (
                      <span className="text-[10px] text-zinc-500 hidden sm:inline">
                        Swap roles with the ↔ arrows
                      </span>
                    )}
                    {editable && onReassignRoles && refs.length > 0 && targets.length > 0 && (
                      <button
                        onClick={() => onReassignRoles(targets.map((t) => t.id), refs.map((c) => c.id))}
                        title="Swap every input ↔ output in this example"
                        className="text-zinc-500 hover:text-zinc-200 flex items-center gap-1 text-[10px]"
                      >
                        <ArrowLeftRight className="h-3.5 w-3.5" /> Reverse
                      </button>
                    )}
                    {editable && onUngroup && group && (
                      <button
                        onClick={() => onUngroup(groupMemberIds(group))}
                        title="Dissolve this example back into loose clips"
                        className="text-zinc-500 hover:text-zinc-200 flex items-center gap-1 text-[10px]"
                      >
                        <Unlink className="h-3.5 w-3.5" /> Ungroup
                      </button>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-3 overflow-x-auto pb-1">
                  <div className="flex items-center gap-2">
                    {refs.length > 0 ? (
                      refs.map((m) => renderMember(m, 'ref'))
                    ) : (
                      <div className="h-16 w-[7.5rem] shrink-0 rounded-lg border border-dashed border-red-500/40 flex items-center justify-center text-[10px] text-red-400 text-center px-1">
                        Input missing
                      </div>
                    )}
                  </div>
                  <ArrowRight className="h-5 w-5 shrink-0 text-blue-400" />
                  <div className="flex items-center gap-2">
                    {targets.map((m) => renderMember(m, 'target'))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="w-80 flex-shrink-0 border-l border-zinc-800 flex flex-col overflow-y-auto">
            <div className="p-4 space-y-3 border-b border-zinc-800">
              <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                <span className={`px-1.5 py-0.5 rounded font-medium ${isImage ? 'bg-sky-500/20 text-sky-300' : 'bg-zinc-800 text-zinc-300'}`}>
                  {isImage ? 'Still image' : 'Video'}
                </span>
                {resolution && <span className="px-1.5 py-0.5 rounded bg-zinc-800/70 text-zinc-400">{resolution}</span>}
                {!isImage && duration != null && (
                  <span className="px-1.5 py-0.5 rounded bg-zinc-800/70 text-zinc-400">{formatDuration(duration)}</span>
                )}
                {!isImage && clip.probe && clip.probe.fps > 0 && (
                  <span className="px-1.5 py-0.5 rounded bg-zinc-800/70 text-zinc-400">{Math.round(clip.probe.fps)}fps</span>
                )}
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] uppercase tracking-wide text-zinc-500">Caption</span>
                  {onAutoCaption && (
                    <button
                      onClick={runAutoCaption}
                      disabled={captioning}
                      title="Generate a caption for this clip with AI"
                      className="text-[10px] px-1.5 py-0.5 rounded-md border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-500 disabled:opacity-60 flex items-center gap-1 transition-colors"
                    >
                      {captioning ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}
                      {captioning ? 'Captioning…' : 'Auto-caption'}
                    </button>
                  )}
                </div>
                <textarea
                  value={caption}
                  onChange={(e) => setCaption(e.target.value)}
                  onBlur={commitCaption}
                  rows={4}
                  placeholder="Describe this clip…"
                  className={`w-full text-xs bg-zinc-950 border rounded-md px-2 py-1.5 text-zinc-200 placeholder:text-zinc-600 focus:outline-none resize-none transition-colors ${
                    captioning ? 'border-blue-500/50' : 'border-zinc-800 focus:border-zinc-600'
                  }`}
                />
                {captionError && <p className="mt-1 text-[10px] text-red-400">{captionError}</p>}
              </div>
            </div>

            {editable && (
              <div className="p-4 space-y-3">
                {!isImage ? (
                  <>
                    <div className="space-y-2">
                      <span className="text-[10px] uppercase tracking-wide text-zinc-500">Edit</span>
                      <button className={btnNeutral} onClick={() => onEdit(clip.id)}>
                        <Crop className="h-3.5 w-3.5 text-zinc-400" /> Trim &amp; crop
                      </button>
                      <button className={btnNeutral} onClick={() => onFrameEdit(clip.id)}>
                        <ImageIcon className="h-3.5 w-3.5 text-zinc-400" /> Frame edit (AI)
                      </button>
                    </div>
                    <div className="space-y-2 pt-1">
                      <span className="text-[10px] uppercase tracking-wide text-zinc-500">Generate</span>
                      <button className={btnPrimary} onClick={() => onMakePair(clip.id)} title="AI-generate the edited output and link it to this clip as a training example">
                        <Sparkles className="h-3.5 w-3.5" /> Generate example
                      </button>
                      <button className={btnNeutral} onClick={() => onVariant(clip.id)}>
                        <Wand2 className="h-3.5 w-3.5 text-zinc-400" /> Variant
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="space-y-2">
                    <span className="text-[10px] uppercase tracking-wide text-zinc-500">Generate</span>
                    <button className={btnPrimary} onClick={() => onVariant(clip.id)}>
                      <Wand2 className="h-3.5 w-3.5" /> Animate (i2v)
                    </button>
                    <button className={btnNeutral} onClick={() => onMakePair(clip.id)} disabled={!canMakePair} title={canMakePair ? 'AI-generate a motion-locked output and link it as a training example' : 'Add a video clip to use as the motion driver'}>
                      <Sparkles className="h-3.5 w-3.5 text-zinc-400" /> Generate example (motion-lock)
                    </button>
                  </div>
                )}
                <div className="pt-2 mt-1 border-t border-zinc-800">
                  <button
                    className="w-full text-xs px-3 py-2 rounded-lg text-zinc-400 hover:text-red-300 hover:bg-red-500/10 flex items-center gap-2 transition-colors"
                    onClick={() => onRemove(clip.id)}
                  >
                    <Trash2 className="h-3.5 w-3.5" /> Move to Trash
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
