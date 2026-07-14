import { useEffect, useRef, useState } from 'react'
import { Loader2, Plus, Sparkles, Split, Trash2, X } from 'lucide-react'
import { useLoraTraining, type ClipInput } from '../../contexts/LoraTrainingContext'
import { clipWarnings, probeBadges } from '../../lib/lora-quality'
import { isImagePath } from '../../lib/file-url'
import { ImportNormalizeOptions, useImportNormalizeSpec } from './ImportNormalizeOptions'
import { importSpecActive, normalizeImportInputs } from '../../lib/lora-import-normalize'

// Imports media into an already-open collection. Accepts videos and image
// stills (images become reference frames). Mirrors the create-dataset flow
// (file pick + scene-split + local probe + optional auto-caption) but hands the
// finished ClipInput[] back to the workspace instead of creating a dataset.
export function AddClipsModal({
  onClose,
  onAdd,
  initialPaths,
}: {
  onClose: () => void
  onAdd: (clips: ClipInput[]) => void
  initialPaths?: string[]
}) {
  const { probeClip, captionClip, splitScenes, applyClipEdits } = useLoraTraining()
  const [clips, setClips] = useState<ClipInput[]>(() =>
    (initialPaths ?? []).map((localPath) => ({ localPath, caption: '', origin: 'imported' as const })),
  )
  const [error, setError] = useState<string | null>(null)
  const [splitting, setSplitting] = useState(false)
  const [captioning, setCaptioning] = useState<{ done: number; total: number } | null>(null)
  const [normSpec, setNormSpec] = useImportNormalizeSpec()
  const [normalizing, setNormalizing] = useState<{ done: number; total: number } | null>(null)

  const handleImport = async () => {
    if (!importSpecActive(normSpec)) {
      onAdd(clips)
      return
    }
    setError(null)
    setNormalizing({ done: 0, total: clips.length })
    const { inputs, failures } = await normalizeImportInputs(clips, normSpec, applyClipEdits, {
      onProgress: setNormalizing,
    })
    setNormalizing(null)
    if (failures.length > 0) setError(`Some clips couldn't be normalized: ${failures.join('; ')}`)
    onAdd(inputs)
  }
  const fileInputRef = useRef<HTMLInputElement>(null)
  const splitInputRef = useRef<HTMLInputElement>(null)
  const probingRef = useRef<Set<string>>(new Set())

  const addFiles = (files: FileList | null) => {
    if (!files) return
    const next: ClipInput[] = []
    for (const file of Array.from(files)) {
      const localPath = window.electronAPI?.getPathForFile(file)
      if (localPath) next.push({ localPath, caption: '', origin: 'imported' })
    }
    setClips((prev) => [...prev, ...next])
  }

  const importScenes = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    const localPath = window.electronAPI?.getPathForFile(files[0])
    if (!localPath) return
    setError(null)
    setSplitting(true)
    const res = await splitScenes(localPath)
    setSplitting(false)
    if (!res.ok) {
      setError(res.error)
      return
    }
    const scenes: ClipInput[] = res.data.map((scene) => ({
      localPath: scene.localPath,
      caption: '',
      origin: 'imported',
      probe: scene.probe,
      durationSeconds: scene.probe.durationSeconds,
    }))
    setClips((prev) => [...prev, ...scenes])
  }

  // Auto-probe any video lacking a cached measurement (local ffmpeg, fast).
  // Images carry no ffmpeg probe, so skip them.
  useEffect(() => {
    for (const clip of clips) {
      if (clip.probe || isImagePath(clip.localPath) || probingRef.current.has(clip.localPath)) continue
      probingRef.current.add(clip.localPath)
      void (async () => {
        const res = await probeClip(clip.localPath)
        if (res.ok) {
          setClips((prev) =>
            prev.map((c) =>
              c.localPath === clip.localPath && !c.probe
                ? { ...c, probe: res.data, durationSeconds: res.data.durationSeconds }
                : c,
            ),
          )
        }
      })()
    }
  }, [clips, probeClip])

  const captionMissing = async () => {
    // The video captioner can't read stills, so only target uncaptioned videos.
    const targets = clips.flatMap((c, i) =>
      c.caption.trim() || isImagePath(c.localPath) ? [] : [{ path: c.localPath, index: i }],
    )
    if (targets.length === 0) return
    setError(null)
    let done = 0
    setCaptioning({ done, total: targets.length })
    for (const target of targets) {
      const result = await captionClip(target.path, false)
      done += 1
      if (result.ok) {
        const caption = result.data
        setClips((prev) => prev.map((c, i) => (i === target.index ? { ...c, caption } : c)))
      } else {
        setError(result.error)
      }
      setCaptioning({ done, total: targets.length })
    }
    setCaptioning(null)
  }

  const removeAt = (index: number) => setClips((prev) => prev.filter((_, i) => i !== index))

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-lg mx-4">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="text-base font-semibold text-white">Import media</h2>
          <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-3 max-h-[60vh] overflow-y-auto">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-300">{clips.length} item{clips.length === 1 ? '' : 's'}</span>
            <div className="flex items-center gap-3">
              <button
                onClick={() => splitInputRef.current?.click()}
                disabled={splitting}
                title="Pick one long video and auto-split it into per-scene clips"
                className="text-xs text-zinc-400 hover:text-zinc-200 disabled:opacity-50 flex items-center gap-1"
              >
                {splitting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Split className="h-3 w-3" />} Split a video
              </button>
              <button onClick={() => fileInputRef.current?.click()} className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
                <Plus className="h-3 w-3" /> Add files
              </button>
            </div>
            <input ref={fileInputRef} type="file" accept="video/*,image/*" multiple className="hidden" onChange={(e) => { addFiles(e.target.files); e.target.value = '' }} />
            <input ref={splitInputRef} type="file" accept="video/*" className="hidden" onChange={(e) => { void importScenes(e.target.files); e.target.value = '' }} />
          </div>

          {clips.length > 0 && (
            <button
              onClick={() => void captionMissing()}
              disabled={captioning !== null}
              className="text-xs text-blue-300 hover:text-blue-200 disabled:opacity-50 flex items-center gap-1.5"
            >
              {captioning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              {captioning ? `Captioning ${captioning.done}/${captioning.total}...` : 'Auto-caption empty clips'}
            </button>
          )}

          {error && <p className="text-xs text-red-400">{error}</p>}

          {clips.length > 0 && (
            <ImportNormalizeOptions value={normSpec} onChange={setNormSpec} disabled={normalizing !== null} />
          )}

          {clips.length === 0 ? (
            <div className="text-xs text-zinc-600 border border-dashed border-zinc-700 rounded-lg py-6 text-center">
              Nothing yet. Add the videos or images you want to bring into this collection.
            </div>
          ) : (
            <div className="space-y-2">
              {clips.map((clip, index) => {
                const warnings = clipWarnings(clip)
                return (
                  <div key={`${clip.localPath}:${index}`} className="flex items-center gap-2 bg-zinc-800/40 rounded-lg px-2.5 py-2">
                    <div className="min-w-0 flex-1">
                      <p className="text-[11px] text-zinc-300 truncate font-mono">{clip.localPath}</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {clip.probe && probeBadges(clip.probe).map((b) => (
                          <span key={b} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-400">{b}</span>
                        ))}
                        {warnings.map((w, i) => (
                          <span key={i} className={`text-[10px] px-1.5 py-0.5 rounded ${w.level === 'error' ? 'bg-red-500/10 text-red-400' : 'bg-amber-500/10 text-amber-400'}`}>{w.text}</span>
                        ))}
                      </div>
                    </div>
                    <button onClick={() => removeAt(index)} className="h-7 w-7 flex items-center justify-center rounded-md text-zinc-500 hover:text-red-400 hover:bg-zinc-800 flex-shrink-0">
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex justify-end gap-2">
          <button onClick={onClose} className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white">Cancel</button>
          <button
            onClick={() => void handleImport()}
            disabled={clips.length === 0 || captioning !== null || normalizing !== null}
            className="text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-1.5"
          >
            {normalizing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
            {normalizing
              ? `Normalizing ${normalizing.done}/${normalizing.total}…`
              : `Import ${clips.length > 0 ? clips.length : ''} item${clips.length === 1 ? '' : 's'}`}
          </button>
        </div>
      </div>
    </div>
  )
}
