import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Crop, Film, Images, Loader2, Plus, Scissors, Sparkles, Split, Trash2, Wand2 } from 'lucide-react'
import { Button } from '../ui/button'
import { Dialog } from '../ui/dialog'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { PexelsBrowser } from '../../views/lora/PexelsBrowser'
import { ImportNormalizeOptions, useImportNormalizeSpec } from '../../views/lora/ImportNormalizeOptions'
import { importSpecActive, normalizeImportInputs } from '../../lib/lora-import-normalize'
import {
  useLoraTraining,
  type ClipInput,
  type ClipProbe,
  type LoraDataset,
  type LoraDatasetType,
  type LoraPreprocessed,
  type LoraProvider,
  type LoraTrainingConfig,
  type NanoBananaModel,
} from '../../contexts/LoraTrainingContext'
import { centeredCrop, clipWarnings, datasetHealth, formatDuration, probeBadges, type CropRect } from '../../lib/lora-quality'
import { DatasetHealthMeter } from './DatasetHealth'
import { ConfigField, CollapsibleSection } from './TrainingConfigControls'
import { PRIMARY_FIELDS, SECTIONS, SECTION_FIELDS } from './trainingConfigFields'
import { useLocalWslMemoryGate } from './useLocalWslMemoryGate'
import { GpuSelector, InfoHint, ProfilePicker, ResolutionInput } from './trainingFormParts'
import {
  activeVramGb,
  defaultProfileIdForVram,
  effectiveProfileConfig,
  ExpertOverrideWarning,
  isRiskyTrainingOverride,
} from './trainingSafety'
import { RunpodTrainingSetup, type RunpodEstimateWorkload } from './RunpodTrainingSetup'
import type { RunpodSelection } from '../../lib/runpod-contracts'
import { isGpuSelectionRequired } from '../../lib/runpod-contracts'

// Shared shell so each modal stays focused on its form. Stops keydown
// propagation so the app's global keyboard shortcuts don't fire while typing.
//
// Layout is a flex column capped at the viewport: only `children` scrolls, so an
// optional `pinned` region (and the footer) stay visible no matter how long the
// scrollable body gets.
function ModalShell({
  title,
  onClose,
  children,
  footer,
  pinned,
  closeDisabled,
  className,
}: {
  title: string
  onClose: () => void
  children: React.ReactNode
  footer: React.ReactNode
  /** Always-visible region between the scrollable body and the footer. */
  pinned?: React.ReactNode
  closeDisabled?: boolean
  className?: string
}) {
  return (
    <Dialog
      title={title}
      onClose={onClose}
      footer={footer}
      pinned={pinned}
      closeDisabled={closeDisabled}
      className={className}
    >
      {children}
    </Dialog>
  )
}

function FieldError({ message }: { message: string | null }) {
  if (!message) return null
  return <p className="text-xs text-red-400">{message}</p>
}

// ---------------------------------------------------------------------------
// Create dataset
// ---------------------------------------------------------------------------

export function CreateDatasetModal({ onClose, originatingProjectId = null }: { onClose: () => void; originatingProjectId?: string | null }) {
  const { createDataset, captionClip, probeClip, splitScenes, applyClipEdits } = useLoraTraining()
  const [name, setName] = useState('')
  const [datasetType, setDatasetType] = useState<LoraDatasetType>('standard')
  const [triggerWord, setTriggerWord] = useState('')
  const [clips, setClips] = useState<ClipInput[]>([])
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [normSpec, setNormSpec] = useImportNormalizeSpec()
  const [normalizing, setNormalizing] = useState<{ done: number; total: number } | null>(null)
  const [captionAudio, setCaptionAudio] = useState(false)
  const [busyClips, setBusyClips] = useState<Set<number>>(new Set())
  const [captionAll, setCaptionAll] = useState<{ done: number; total: number } | null>(null)
  const [prepareIndex, setPrepareIndex] = useState<number | null>(null)
  const [splitting, setSplitting] = useState(false)
  const [showPexels, setShowPexels] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const splitInputRef = useRef<HTMLInputElement>(null)
  // Paths already submitted for probing, so the auto-probe effect doesn't
  // loop on failures or fire twice for the same clip.
  const probingRef = useRef<Set<string>>(new Set())

  const captioningBusy = busyClips.size > 0 || captionAll !== null
  const health = datasetHealth(clips)

  const addFiles = (files: FileList | null) => {
    if (!files) return
    const next: ClipInput[] = []
    for (const file of Array.from(files)) {
      const localPath = window.electronAPI?.getPathForFile(file)
      if (localPath) next.push({ localPath, caption: '', origin: 'imported' })
    }
    setClips((prev) => [...prev, ...next])
  }

  // Split a long video into per-scene clips (already rendered, probed).
  const importScenes = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    const localPath = window.electronAPI?.getPathForFile(files[0])
    if (!localPath) return
    setError(null)
    setSplitting(true)
    const res = await splitScenes(localPath)
    setSplitting(false)
    if (!res.ok) { setError(res.error); return }
    const scenes: ClipInput[] = res.data.map((scene) => ({
      localPath: scene.localPath,
      caption: '',
      origin: 'imported',
      probe: scene.probe,
      durationSeconds: scene.probe.durationSeconds,
    }))
    setClips((prev) => [...prev, ...scenes])
  }

  // Auto-probe any clip without a cached probe. Runs locally (ffmpeg) so it's
  // fast and free; results badge the clip and feed the dataset health meter.
  useEffect(() => {
    for (const clip of clips) {
      if (clip.probe || probingRef.current.has(clip.localPath)) continue
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

  const setBusy = (index: number, busy: boolean) => {
    setBusyClips((prev) => {
      const next = new Set(prev)
      if (busy) next.add(index)
      else next.delete(index)
      return next
    })
  }

  const captionOne = async (index: number) => {
    const clip = clips[index]
    if (!clip) return
    setError(null)
    setBusy(index, true)
    const result = await captionClip(clip.localPath, captionAudio)
    setBusy(index, false)
    if (result.ok) {
      setClips((prev) => prev.map((c, i) => (i === index ? { ...c, caption: result.data } : c)))
    } else {
      setError(result.error)
    }
  }

  // Caption every clip that doesn't already have one, sequentially so the
  // user sees steady progress and we don't hammer the API in parallel.
  const captionMissing = async () => {
    const targets = clips.flatMap((c, i) => (c.caption.trim() ? [] : [{ path: c.localPath, index: i }]))
    if (targets.length === 0) return
    setError(null)
    let done = 0
    setCaptionAll({ done, total: targets.length })
    for (const target of targets) {
      const result = await captionClip(target.path, captionAudio)
      done += 1
      if (result.ok) {
        const caption = result.data
        setClips((prev) => prev.map((c, i) => (i === target.index ? { ...c, caption } : c)))
      } else {
        setError(result.error)
      }
      setCaptionAll({ done, total: targets.length })
    }
    setCaptionAll(null)
  }

  const submit = async () => {
    if (!name.trim()) {
      setError('Please name the dataset.')
      return
    }
    if (clips.length === 0) {
      setError('Add at least one video clip.')
      return
    }
    let finalClips = clips
    if (importSpecActive(normSpec)) {
      setNormalizing({ done: 0, total: clips.length })
      const res = await normalizeImportInputs(clips, normSpec, applyClipEdits, {
        onProgress: setNormalizing,
      })
      setNormalizing(null)
      finalClips = res.inputs
      if (res.failures.length > 0) {
        setError(`Some clips couldn't be normalized: ${res.failures.join('; ')}`)
        return
      }
    }
    setSaving(true)
    const result = await createDataset(name.trim(), triggerWord.trim() || null, finalClips, originatingProjectId, datasetType)
    setSaving(false)
    if (!result.ok) {
      setError(result.error)
      return
    }
    onClose()
  }

  return (
    <ModalShell
      title="New Dataset"
      onClose={onClose}
      closeDisabled={saving || captioningBusy || normalizing !== null || splitting}
      pinned={
        clips.length > 0 ? (
          <>
            <DatasetHealthMeter health={health} />
            <ImportNormalizeOptions value={normSpec} onChange={setNormSpec} disabled={normalizing !== null} />
            <FieldError message={error} />
          </>
        ) : error ? (
          <FieldError message={error} />
        ) : undefined
      }
      footer={
        <>
          <Button variant="outline" className="border-zinc-700" disabled={saving || captioningBusy || normalizing !== null || splitting} onClick={onClose}>Cancel</Button>
          <Button className="bg-blue-600 hover:bg-blue-500" disabled={saving || captioningBusy || normalizing !== null} onClick={() => void submit()}>
            {normalizing ? `Normalizing ${normalizing.done}/${normalizing.total}…` : saving ? 'Creating...' : 'Create'}
          </Button>
        </>
      }
    >
      <div className="space-y-1.5">
        <label htmlFor="create-dataset-name" className="text-xs font-medium text-zinc-300">Name</label>
        <input
          id="create-dataset-name"
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="My concept"
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <fieldset className="space-y-1.5">
        <legend className="text-xs font-medium text-zinc-300">Type</legend>
        <div className="grid grid-cols-2 gap-2">
          {([
            { id: 'standard', title: 'Standard LoRA', desc: 'Learn a look, style, or subject from individual clips.' },
            { id: 'ic_lora', title: 'IC-LoRA', desc: 'Learn a transformation from input → output examples.' },
          ] as const).map((opt) => (
            <button
              key={opt.id}
              type="button"
              aria-pressed={datasetType === opt.id}
              onClick={() => setDatasetType(opt.id)}
              className={`text-left px-3 py-2.5 rounded-lg border transition-colors ${
                datasetType === opt.id
                  ? 'bg-blue-500/15 border-blue-500/50 text-white'
                  : 'bg-zinc-800/40 border-zinc-700 text-zinc-300 hover:border-zinc-600'
              }`}
            >
              <div className="text-xs font-medium">{opt.title}</div>
              <div className="text-[11px] text-zinc-500 mt-0.5">{opt.desc}</div>
            </button>
          ))}
        </div>
      </fieldset>
      <div className="space-y-1.5">
        <label htmlFor="create-dataset-trigger" className="text-xs font-medium text-zinc-300">Trigger word (optional)</label>
        <input
          id="create-dataset-trigger"
          value={triggerWord}
          onChange={(e) => setTriggerWord(e.target.value)}
          placeholder="e.g. MYSTYLE"
          spellCheck={false}
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <p className="text-[11px] text-zinc-500">
          Prepended during preprocessing and automatically added at inference.
          {datasetType === 'ic_lora' ? ' Do not type it into IC-LoRA target captions.' : ''}
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-zinc-300">Clips ({clips.length})</span>
          <div className="flex items-center gap-3">
            <button
              onClick={() => splitInputRef.current?.click()}
              disabled={splitting}
              title="Pick one long video and auto-split it into per-scene clips"
              className="text-xs text-zinc-400 hover:text-zinc-200 disabled:opacity-50 flex items-center gap-1"
            >
              {splitting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Split className="h-3 w-3" />} Split a video
            </button>
            <button
              onClick={() => setShowPexels(true)}
              title="Browse free stock photos and videos from Pexels"
              className="text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1"
            >
              <Images className="h-3 w-3" /> Browse Pexels
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
            >
              <Plus className="h-3 w-3" /> Add videos
            </button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            multiple
            className="hidden"
            onChange={(e) => { addFiles(e.target.files); e.target.value = '' }}
          />
          <input
            ref={splitInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => { void importScenes(e.target.files); e.target.value = '' }}
          />
        </div>

        {clips.length > 0 && (
          <div className="flex items-center justify-between gap-2 bg-zinc-800/40 rounded-lg px-2.5 py-2">
            <button
              onClick={() => void captionMissing()}
              disabled={captioningBusy}
              className="text-xs text-blue-300 hover:text-blue-200 disabled:opacity-50 flex items-center gap-1.5"
            >
              {captionAll ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              {captionAll ? `Captioning ${captionAll.done}/${captionAll.total}...` : 'Auto-caption empty clips'}
            </button>
            <label className="flex items-center gap-1.5 text-[11px] text-zinc-400 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={captionAudio}
                onChange={(e) => setCaptionAudio(e.target.checked)}
                className="accent-blue-500"
              />
              Describe audio
            </label>
          </div>
        )}

        {clips.length === 0 ? (
          <div className="text-xs text-zinc-600 border border-dashed border-zinc-700 rounded-lg py-6 text-center">
            No clips yet. Add the videos that show your concept.
          </div>
        ) : (
          <div className="space-y-2">
            {clips.map((clip, index) => {
              const busy = busyClips.has(index)
              const trimmedTrigger = triggerWord.trim()
              const warnings = clipWarnings(clip)
              return (
                <div key={`${clip.localPath}-${index}`} className="bg-zinc-800/60 rounded-lg p-2 space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs text-zinc-300 truncate font-mono">{clip.localPath.split('/').pop()}</span>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <button
                        onClick={() => setPrepareIndex(index)}
                        title="Trim or crop this clip"
                        aria-label={`Prepare clip ${index + 1}`}
                        className="text-zinc-400 hover:text-blue-300"
                      >
                        <Scissors className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => void captionOne(index)}
                        disabled={captioningBusy}
                        title="Auto-caption this clip with Gemini"
                        aria-label={`Auto-caption clip ${index + 1}`}
                        className="text-blue-400 hover:text-blue-300 disabled:opacity-40"
                      >
                        {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                      </button>
                      <button
                        onClick={() => setClips((prev) => prev.filter((_, i) => i !== index))}
                        aria-label={`Remove clip ${index + 1}`}
                        className="text-zinc-500 hover:text-red-400"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                  {clip.probe ? (
                    <div className="flex flex-wrap items-center gap-1">
                      {probeBadges(clip.probe).map((b) => (
                        <span key={b} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-400 font-mono">{b}</span>
                      ))}
                      {clip.edits && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 inline-flex items-center gap-1">
                          <Scissors className="h-2.5 w-2.5" /> edited
                        </span>
                      )}
                      {warnings.map((w) => (
                        <span
                          key={w.text}
                          className={`text-[10px] px-1.5 py-0.5 rounded inline-flex items-center gap-1 ${w.level === 'error' ? 'bg-red-500/10 text-red-400' : 'bg-amber-500/10 text-amber-400'}`}
                        >
                          <AlertTriangle className="h-2.5 w-2.5" /> {w.text}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div className="text-[10px] text-zinc-600 inline-flex items-center gap-1">
                      <Loader2 className="h-2.5 w-2.5 animate-spin" /> Reading clip…
                    </div>
                  )}
                  <textarea
                    aria-label={`Caption for clip ${index + 1}`}
                    value={clip.caption}
                    onChange={(e) => setClips((prev) => prev.map((c, i) => (i === index ? { ...c, caption: e.target.value } : c)))}
                    placeholder={datasetType === 'ic_lora'
                      ? 'Required: describe the desired output only; do not include the trigger'
                      : 'Caption (optional — click the sparkle to auto-caption)'}
                    rows={2}
                    className="w-full px-2 py-1 bg-zinc-900 border border-zinc-700 rounded text-xs text-white placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
                  />
                  {trimmedTrigger && clip.caption.trim() && (
                    <p className="text-[10px] text-zinc-500 truncate">
                      At training: <span className="text-blue-400 font-medium">{trimmedTrigger}</span> {clip.caption.trim()}
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
      {prepareIndex != null && clips[prepareIndex] && (
        <PrepareClipModal
          clip={clips[prepareIndex]}
          onApply={(updated) => setClips((prev) => prev.map((c, i) => (i === prepareIndex ? updated : c)))}
          onClose={() => setPrepareIndex(null)}
        />
      )}
      {showPexels && (
        <PexelsBrowser
          onClose={() => setShowPexels(false)}
          onAdd={(incoming) => setClips((prev) => [...prev, ...incoming])}
          normalizeOnAdd={false}
        />
      )}
    </ModalShell>
  )
}

// ---------------------------------------------------------------------------
// Prepare a clip — non-destructive trim + crop
// ---------------------------------------------------------------------------

const ASPECT_PRESETS: Array<{ id: string; label: string; ratio: [number, number] | null }> = [
  { id: 'original', label: 'Original', ratio: null },
  { id: '16:9', label: '16:9', ratio: [16, 9] },
  { id: '9:16', label: '9:16', ratio: [9, 16] },
  { id: '1:1', label: '1:1', ratio: [1, 1] },
]

export function PrepareClipModal({
  clip,
  onApply,
  onClose,
}: {
  clip: ClipInput
  onApply: (updated: ClipInput) => void
  onClose: () => void
}) {
  const { applyClipEdits, probeClip, editFrame, animateFrame, restyleClip } = useLoraTraining()
  // Edits always derive from the untouched source so they stay re-editable.
  const source = clip.sourcePath ?? clip.localPath
  const [tab, setTab] = useState<'trim' | 'crop' | 'edit' | 'restyle'>('trim')
  const [sourceProbe, setSourceProbe] = useState<ClipProbe | null>(null)
  const [start, setStart] = useState(clip.edits?.trim?.startSeconds ?? 0)
  const [end, setEnd] = useState<number | null>(clip.edits?.trim?.endSeconds ?? null)
  const [crop, setCrop] = useState<CropRect | null>(
    clip.edits?.crop
      ? { x: clip.edits.crop.x, y: clip.edits.crop.y, width: clip.edits.crop.width, height: clip.edits.crop.height }
      : null,
  )
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // AI tabs (Fal, BYOK). Edit a frame then optionally animate it; or
  // restyle the whole clip. Both produce a new ai-derived clip.
  const [editPrompt, setEditPrompt] = useState('')
  const [motionPrompt, setMotionPrompt] = useState('')
  const [restylePrompt, setRestylePrompt] = useState('')
  const [editedFrame, setEditedFrame] = useState<string | null>(null)
  const [nanoModel, setNanoModel] = useState<NanoBananaModel | ''>('')
  const isTrimCrop = tab === 'trim' || tab === 'crop'

  const replaceWithDerived = (derivedPath: string, probe: ClipProbe) => {
    onApply({
      ...clip,
      localPath: derivedPath,
      sourcePath: source,
      origin: 'ai_derived',
      edits: undefined,
      probe,
      durationSeconds: probe.durationSeconds,
    })
    onClose()
  }

  const runEditFrame = async () => {
    setError(null)
    setBusy(true)
    const res = await editFrame(source, editPrompt, {
      timeSeconds: start,
      model: nanoModel || undefined,
    })
    setBusy(false)
    if (!res.ok) { setError(res.error); return }
    setEditedFrame(res.data)
  }

  const runAnimate = async () => {
    if (!editedFrame) return
    setError(null)
    setBusy(true)
    const res = await animateFrame(editedFrame, motionPrompt)
    setBusy(false)
    if (!res.ok) { setError(res.error); return }
    replaceWithDerived(res.data.derivedPath, res.data.probe)
  }

  const runRestyle = async () => {
    setError(null)
    setBusy(true)
    const res = await restyleClip(source, restylePrompt)
    setBusy(false)
    if (!res.ok) { setError(res.error); return }
    replaceWithDerived(res.data.derivedPath, res.data.probe)
  }

  useEffect(() => {
    let cancelled = false
    void (async () => {
      const res = await probeClip(source)
      if (cancelled) return
      if (res.ok) {
        setSourceProbe(res.data)
        setEnd((prev) => prev ?? res.data.durationSeconds)
      } else {
        setError(res.error)
      }
    })()
    return () => { cancelled = true }
  }, [source, probeClip])

  const duration = sourceProbe?.durationSeconds ?? 0
  const effectiveEnd = end ?? duration
  const trimmedDuration = Math.max(0, effectiveEnd - start)
  const hasTrim = sourceProbe != null && (start > 0.01 || effectiveEnd < duration - 0.01)

  const selectAspect = (ratio: [number, number] | null) => {
    if (!sourceProbe) return
    if (ratio == null) { setCrop(null); return }
    setCrop(centeredCrop(sourceProbe.width, sourceProbe.height, ratio[0], ratio[1]))
  }

  const apply = async () => {
    setError(null)
    const trim = hasTrim ? { startSeconds: Number(start.toFixed(2)), endSeconds: Number(effectiveEnd.toFixed(2)) } : null
    if (!trim && !crop) {
      // Cleared all edits — reset to the pristine source, no render needed.
      onApply({ ...clip, localPath: source, sourcePath: undefined, edits: undefined, probe: sourceProbe ?? clip.probe, durationSeconds: sourceProbe?.durationSeconds ?? clip.durationSeconds })
      onClose()
      return
    }
    setBusy(true)
    const edits = { trim, crop, scale: null, fps: null, speed: null, mute: false, reverse: false }
    const res = await applyClipEdits(source, edits)
    setBusy(false)
    if (!res.ok) { setError(res.error); return }
    onApply({
      ...clip,
      localPath: res.data.derivedPath,
      sourcePath: source,
      edits,
      probe: res.data.probe,
      durationSeconds: res.data.probe.durationSeconds,
    })
    onClose()
  }

  return (
    <ModalShell
      title="Prepare clip"
      onClose={onClose}
      closeDisabled={busy}
      footer={
        <>
          <Button variant="outline" className="border-zinc-700" disabled={busy} onClick={onClose}>{isTrimCrop ? 'Cancel' : 'Close'}</Button>
          {isTrimCrop && (
            <Button className="bg-blue-600 hover:bg-blue-500" disabled={busy || !sourceProbe} onClick={() => void apply()}>
              {busy ? 'Rendering…' : 'Apply'}
            </Button>
          )}
        </>
      }
    >
      {!sourceProbe ? (
        <div className="text-xs text-zinc-500 flex items-center gap-2 py-6 justify-center">
          <Loader2 className="h-4 w-4 animate-spin" /> Reading clip…
        </div>
      ) : (
        <>
          <div role="tablist" aria-label="Clip preparation mode" className="grid grid-cols-2 gap-1 bg-zinc-800/60 rounded-lg p-1 sm:grid-cols-4">
            <TabButton active={tab === 'trim'} onClick={() => setTab('trim')} icon={Scissors} label="Trim" />
            <TabButton active={tab === 'crop'} onClick={() => setTab('crop')} icon={Crop} label="Crop" />
            <TabButton active={tab === 'edit'} onClick={() => setTab('edit')} icon={Wand2} label="AI edit" />
            <TabButton active={tab === 'restyle'} onClick={() => setTab('restyle')} icon={Film} label="Restyle" />
          </div>

          {tab === 'trim' ? (
            <div className="space-y-3">
              <p className="text-[11px] text-zinc-500">Source is {formatDuration(duration)}. Keep the segment that best shows your concept.</p>
              <div className="grid grid-cols-2 gap-3">
                <NumberField label="Start (s)" value={Number(start.toFixed(2))} min={0} max={Math.max(0, duration - 0.1)} step={0.1} onChange={(v) => setStart(Math.min(v, effectiveEnd - 0.1))} />
                <NumberField label="End (s)" value={Number(effectiveEnd.toFixed(2))} min={0.1} max={duration} step={0.1} onChange={(v) => setEnd(Math.max(v, start + 0.1))} />
              </div>
              <p className="text-[11px] text-zinc-400">Result: <span className="text-white font-medium">{formatDuration(trimmedDuration)}</span></p>
            </div>
          ) : tab === 'crop' ? (
            <div className="space-y-3">
              <p className="text-[11px] text-zinc-500">Source is {sourceProbe.width}×{sourceProbe.height}. Crop to a consistent shape (snapped to multiples of 32).</p>
              <div className="grid grid-cols-4 gap-2">
                {ASPECT_PRESETS.map((preset) => {
                  const active = preset.ratio == null ? crop == null : crop != null && Math.abs(crop.width / crop.height - preset.ratio[0] / preset.ratio[1]) < 0.02
                  return (
                    <button
                      key={preset.id}
                      onClick={() => selectAspect(preset.ratio)}
                      className={`px-2 py-2 rounded-lg text-xs border-2 transition-colors ${active ? 'border-blue-500 bg-blue-500/10 text-white' : 'border-zinc-700 text-zinc-400 hover:text-white'}`}
                    >
                      {preset.label}
                    </button>
                  )
                })}
              </div>
              <p className="text-[11px] text-zinc-400">
                Result: <span className="text-white font-medium">{crop ? `${crop.width}×${crop.height}` : `${sourceProbe.width}×${sourceProbe.height} (uncropped)`}</span>
              </p>
            </div>
          ) : tab === 'edit' ? (
            <div className="space-y-3">
              <p className="text-[11px] text-zinc-500">
                Edit a frame with Nano Banana (e.g. remove an object), then turn the edited still into a new clip. Uses your Fal API key.
              </p>
              <div className="flex items-center gap-2">
                <label htmlFor="prepare-ai-model" className="text-[11px] text-zinc-400">Model</label>
                <select
                  id="prepare-ai-model"
                  value={nanoModel}
                  onChange={(e) => setNanoModel(e.target.value as NanoBananaModel | '')}
                  className="flex-1 bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-white"
                >
                  <option value="">Default (Nano Banana 2)</option>
                  <option value="nano-banana-2">Nano Banana 2</option>
                  <option value="nano-banana-pro">Nano Banana Pro</option>
                </select>
              </div>
              <textarea
                aria-label="Frame edit instructions"
                value={editPrompt}
                onChange={(e) => setEditPrompt(e.target.value)}
                placeholder="Describe the frame edit (e.g. remove the logo on the shirt)"
                rows={2}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-xs text-white resize-none"
              />
              <Button
                className="w-full bg-blue-600 hover:bg-blue-500"
                disabled={busy || !editPrompt.trim()}
                onClick={() => void runEditFrame()}
              >
                {busy ? 'Editing frame…' : editedFrame ? 'Re-edit frame' : 'Edit frame'}
              </Button>
              {editedFrame && (
                <div className="space-y-2 border-t border-zinc-800 pt-3">
                  <p className="text-[11px] text-zinc-500">Edited frame ready. Animate it into a clip (image-to-video):</p>
                  <textarea
                    aria-label="Animation motion"
                    value={motionPrompt}
                    onChange={(e) => setMotionPrompt(e.target.value)}
                    placeholder="Describe the motion (e.g. slow pan, person turns to camera)"
                    rows={2}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-xs text-white resize-none"
                  />
                  <Button
                    className="w-full bg-blue-600 hover:bg-blue-500"
                    disabled={busy || !motionPrompt.trim()}
                    onClick={() => void runAnimate()}
                  >
                    {busy ? 'Generating video…' : 'Animate to video'}
                  </Button>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-[11px] text-zinc-500">
                Re-render this clip in a new style (video-to-video) to grow your dataset. Uses your Fal API key. Creates a new derived clip; the original is kept.
              </p>
              <textarea
                aria-label="Restyle instructions"
                value={restylePrompt}
                onChange={(e) => setRestylePrompt(e.target.value)}
                placeholder="Describe the new style (e.g. claymation, watercolor, 1980s film grain)"
                rows={3}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-xs text-white resize-none"
              />
              <Button
                className="w-full bg-blue-600 hover:bg-blue-500"
                disabled={busy || !restylePrompt.trim()}
                onClick={() => void runRestyle()}
              >
                {busy ? 'Restyling (this can take a minute)…' : 'Restyle clip'}
              </Button>
            </div>
          )}
        </>
      )}
      <FieldError message={error} />
    </ModalShell>
  )
}

function TabButton({ active, onClick, icon: Icon, label }: { active: boolean; onClick: () => void; icon: typeof Scissors; label: string }) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs transition-colors ${active ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-white'}`}
    >
      <Icon className="h-3.5 w-3.5" /> {label}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Send a Gen Space clip to a LoRA dataset
// ---------------------------------------------------------------------------

export function SendToLoraModal({
  videoPath,
  suggestedCaption,
  originatingProjectId,
  onClose,
  onDone,
}: {
  videoPath: string
  suggestedCaption?: string
  originatingProjectId: string | null
  onClose: () => void
  onDone?: (datasetName: string) => void
}) {
  const { datasets, createDataset, updateDataset } = useLoraTraining()
  // Only draft / retryable datasets can still take new clips; once a
  // dataset is uploading/uploaded its remote copy is in flight.
  const editable = datasets.filter((d) => d.status === 'draft' || d.status === 'upload_failed')
  const [target, setTarget] = useState<string>(() => (editable[0]?.id ?? '__new__'))
  const [newName, setNewName] = useState('')
  const [caption, setCaption] = useState(suggestedCaption?.trim() ?? '')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const submit = async () => {
    setError(null)
    const clip: ClipInput = { localPath: videoPath, caption: caption.trim(), origin: 'gen_space' }
    setSaving(true)
    if (target === '__new__') {
      if (!newName.trim()) {
        setSaving(false)
        setError('Name the new dataset.')
        return
      }
      const result = await createDataset(newName.trim(), null, [clip], originatingProjectId)
      setSaving(false)
      if (!result.ok) { setError(result.error); return }
      onDone?.(newName.trim())
      onClose()
      return
    }

    const dataset = editable.find((d) => d.id === target)
    if (!dataset) {
      setSaving(false)
      setError('That dataset is no longer available.')
      return
    }
    const existing: ClipInput[] = dataset.clips.map((c) => ({
      localPath: c.localPath,
      caption: c.caption,
      durationSeconds: c.durationSeconds,
      referencePath: c.referencePath,
      referencePaths: c.referencePaths,
      origin: c.origin,
      probe: c.probe,
    }))
    const result = await updateDataset(dataset.id, { clips: [...existing, clip] })
    setSaving(false)
    if (!result.ok) { setError(result.error); return }
    onDone?.(dataset.name)
    onClose()
  }

  return (
    <ModalShell
      title="Add clip to LoRA dataset"
      onClose={onClose}
      closeDisabled={saving}
      footer={
        <>
          <Button variant="outline" className="border-zinc-700" disabled={saving} onClick={onClose}>Cancel</Button>
          <Button className="bg-blue-600 hover:bg-blue-500" disabled={saving} onClick={() => void submit()}>
            {saving ? 'Adding...' : 'Add clip'}
          </Button>
        </>
      }
    >
      <p className="text-xs text-zinc-400">
        Add this generated clip to a training dataset. LoRAs you train are available across all projects.
      </p>

      <div className="space-y-1.5">
        <label htmlFor="send-to-lora-dataset" className="text-xs font-medium text-zinc-300">Dataset</label>
        <select
          id="send-to-lora-dataset"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {editable.map((d) => (
            <option key={d.id} value={d.id}>{d.name} ({d.clips.length} clips)</option>
          ))}
          <option value="__new__">+ New dataset…</option>
        </select>
      </div>

      {target === '__new__' && (
        <div className="space-y-1.5">
          <label htmlFor="send-to-lora-new-name" className="text-xs font-medium text-zinc-300">New dataset name</label>
          <input
            id="send-to-lora-new-name"
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="My concept"
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      )}

      <div className="space-y-1.5">
        <label htmlFor="send-to-lora-caption" className="text-xs font-medium text-zinc-300">Caption</label>
        <textarea
          id="send-to-lora-caption"
          value={caption}
          onChange={(e) => setCaption(e.target.value)}
          rows={2}
          placeholder="Describe the clip (you can auto-caption later in the trainer)"
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-xs text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        />
        {suggestedCaption?.trim() && (
          <p className="text-[10px] text-zinc-500">Pre-filled from the generation prompt — edit to describe what's actually on screen.</p>
        )}
      </div>
      <FieldError message={error} />
    </ModalShell>
  )
}

// ---------------------------------------------------------------------------
// Start preprocessing
// ---------------------------------------------------------------------------

export function PreprocessModal({ dataset, onClose }: { dataset: LoraDataset; onClose: () => void }) {
  const { startPreprocessing } = useLoraTraining()
  const wslGate = useLocalWslMemoryGate()
  // 49 frames matches the dataset-prep bucket (clips are normalized to a 49-frame
  // `8k+1` bucket). A larger frame count here would make process_dataset.py skip
  // every clip (it drops clips shorter than the bucket), producing an empty set.
  const [resolutionBuckets, setResolutionBuckets] = useState('768x448x49')
  const [withAudio, setWithAudio] = useState(false)
  const [autoCaption, setAutoCaption] = useState(dataset.type !== 'ic_lora')
  const [captionerType, setCaptionerType] = useState<'qwen_omni' | 'gemini_flash'>('gemini_flash')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const provider: LoraProvider = dataset.target?.provider ?? 'runpod'

  const submit = async () => {
    setSaving(true)
    // Provider is inferred from the dataset's upload target (the route does the
    // same server-side). Gate local runs on WSL2 having enough memory first.
    await wslGate.ensureForRun(provider, async () => {
      const result = await startPreprocessing({
        datasetId: dataset.id,
        resolutionBuckets: resolutionBuckets.trim(),
        withAudio: dataset.type === 'ic_lora' ? false : withAudio,
        autoCaption: dataset.type === 'ic_lora' ? false : autoCaption,
        captionerType,
      })
      setSaving(false)
      if (!result.ok) {
        setError(result.error)
        return
      }
      onClose()
    })
    setSaving(false)
  }

  return (
    <>
    <ModalShell
      title={`Preprocess "${dataset.name}"`}
      onClose={onClose}
      closeDisabled={saving}
      footer={
        <>
          <Button variant="outline" className="border-zinc-700" disabled={saving} onClick={onClose}>Cancel</Button>
          <Button className="bg-blue-600 hover:bg-blue-500" disabled={saving} onClick={() => void submit()}>
            {saving ? 'Starting...' : 'Start Preprocessing'}
          </Button>
        </>
      }
    >
      <ResolutionInput value={resolutionBuckets} onChange={setResolutionBuckets} />

      {dataset.type === 'ic_lora' ? (
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-[11px] leading-relaxed text-amber-200">
          IC-LoRA target captions must be reviewed before training. Remote auto-captioning is unavailable because it would discard input/output pairing metadata.
        </div>
      ) : (
        <ToggleRow label="Auto-caption missing clips" hint={`Generate captions on the ${provider === 'runpod' ? 'RunPod training GPU' : 'local training GPU'} only for clips that do not already have one.`} checked={autoCaption} onChange={setAutoCaption} />
      )}
      {dataset.type !== 'ic_lora' && autoCaption && (
        <div className="space-y-1.5 pl-1">
          <label htmlFor="preprocess-captioner" className="text-xs font-medium text-zinc-300">Captioner</label>
          <select
            id="preprocess-captioner"
            value={captionerType}
            onChange={(e) => setCaptionerType(e.target.value as 'qwen_omni' | 'gemini_flash')}
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="gemini_flash">Gemini Flash (API — uses your Gemini key)</option>
            <option value="qwen_omni">Qwen Omni (on the training GPU)</option>
          </select>
        </div>
      )}
      {dataset.type !== 'ic_lora' && (
        <ToggleRow label="Train with audio" hint="Joint audio-video training. Every clip must already contain an audio track." checked={withAudio} onChange={setWithAudio} />
      )}
      <FieldError message={error} />
    </ModalShell>
    {wslGate.dialog}
    </>
  )
}

// ---------------------------------------------------------------------------
// One-click pipeline (upload → preprocess → train)
// ---------------------------------------------------------------------------

export function TrainPipelineModal({ dataset, onClose }: { dataset: LoraDataset; onClose: () => void }) {
  const { startTrainingPipeline, localEligibility, profiles } = useLoraTraining()
  const { settings, updateSettings, getRunpodInventory } = useAppSettings()
  const wslGate = useLocalWslMemoryGate()
  const pendingPipeline = dataset.pendingPipeline
  const [name, setName] = useState(pendingPipeline?.name ?? dataset.name)
  const [description, setDescription] = useState(pendingPipeline?.description ?? '')
  // Preset-aware default: the "Low VRAM" profile defaults to 512x512x49 (the
  // official trainer's smaller, motion-focused / memory-tight bucket — ~24%
  // fewer latent tokens than 768x448, helpful on 32 GB), while Standard and the
  // "Auto" pick keep the high-quality 16:9 bucket 768x448x49. Switching the
  // profile re-targets this default only until the user manually edits the
  // field (bucketTouched), so an explicit choice survives a profile change.
  const [resolutionBuckets, setResolutionBuckets] = useState(
    pendingPipeline?.resolutionBuckets ?? '768x448x49',
  )
  const [bucketTouched, setBucketTouched] = useState(pendingPipeline !== undefined && pendingPipeline !== null)
  const [autoCaption, setAutoCaption] = useState(
    pendingPipeline?.autoCaption ?? dataset.type !== 'ic_lora',
  )
  const [captionerType, setCaptionerType] = useState<'qwen_omni' | 'gemini_flash'>(
    pendingPipeline?.captionerType ?? 'gemini_flash',
  )
  // Joint audio-video training is opt-in: the backend bakes this into the
  // preprocessed latents, so it must be on before preprocessing starts. Every
  // clip must already contain an audio track.
  const [withAudio, setWithAudio] = useState(pendingPipeline?.withAudio ?? false)
  // null = "Auto (recommended)" — the backend auto-matches the preset to the
  // GPU's VRAM. A concrete id snapshots that profile's config onto the run.
  const [profileId, setProfileId] = useState<string | null>(null)
  // Per-run config fork, seeded from the picked profile when customizing. Lets
  // the user keep a profile's defaults but tweak a few knobs for this run only.
  const [customize, setCustomize] = useState(pendingPipeline !== undefined && pendingPipeline !== null)
  const [customConfig, setCustomConfig] = useState<LoraTrainingConfig | null>(
    pendingPipeline ? { ...pendingPipeline.config } : null,
  )
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [allowUnsafeOverride, setAllowUnsafeOverride] = useState(false)
  const isGpuRecovery = dataset.status === 'gpu_selection_required'
  const [step, setStep] = useState<'details' | 'runpod'>(
    isGpuRecovery ? 'runpod' : 'details',
  )
  const [runpodSelection, setRunpodSelection] = useState<RunpodSelection | null>(null)
  const [capacityMessage, setCapacityMessage] = useState<string | null>(null)
  // Pre-fill the trigger word from the dataset so the user sees what the run
  // will train on and can edit it. The backend stamps this onto the dataset
  // before preprocessing, so the token reaches the captions.
  const [triggerWord, setTriggerWord] = useState(
    pendingPipeline?.config.triggerWord ?? dataset.triggerWord ?? '',
  )

  const isIcLora = dataset.type === 'ic_lora'
  // Pre-fill validation prompts from the dataset's captions (standard/t2v) so
  // the user sees and can approve what the run validates against. IC-LoRA
  // ignores bare prompts (validation uses a reference video), so there's no
  // prompt field for it — just a note about the auto-picked clip.
  const [validationPromptsText, setValidationPromptsText] = useState(() =>
    isIcLora
      ? ''
      : (pendingPipeline?.config.validationPrompts ?? suggestedValidationPrompts(dataset)).join('\n'),
  )
  const hasCaptions = dataset.clips.some((c) => (c.caption ?? '').trim() && !c.deletedAt)

  const provider = isGpuRecovery ? 'runpod' : settings.loraProvider
  const vram = activeVramGb(
    provider,
    runpodSelection?.gpuVramGb ?? settings.runpodGpuVramGb,
    localEligibility,
  )
  const selectedProfile = profiles.find((p) => p.id === profileId) ?? null
  const recommendedProfileId = useMemo(
    () => defaultProfileIdForVram(profiles, vram, dataset.type),
    [profiles, vram, dataset.type],
  )
  // Auto still resolves to a concrete recommended profile for editing. The
  // selection remains Auto unless the user enables a per-run customization.
  const customizationBaseProfile =
    selectedProfile ?? profiles.find((profile) => profile.id === recommendedProfileId) ?? null

  const defaultBucketForProfile = (id: string | null): string => {
    const p = profiles.find((pp) => pp.id === id)
    return p?.config.preset === 'low_vram' ? '512x512x49' : '768x448x49'
  }

  const enableCustomize = () => {
    setCustomize(true)
    const effective = effectiveProfileConfig(customizationBaseProfile, vram, profileId === null)
    if (effective) setCustomConfig(effective)
  }

  const onPickProfile = (id: string | null) => {
    setProfileId(id)
    // Re-target the bucket default to the new profile's preset unless the user
    // already edited it. Re-seed an in-progress customization from the newly
    // picked profile, so switching the base profile restarts the diff from its
    // defaults.
    if (!bucketTouched) setResolutionBuckets(defaultBucketForProfile(id))
    if (customize) {
      const next = profiles.find((p) => p.id === (id ?? recommendedProfileId))
      const effective = effectiveProfileConfig(next ?? null, vram, id === null)
      if (effective) setCustomConfig(effective)
    }
  }

  const setConfigValue = (key: string, value: unknown) => {
    setCustomConfig((c) => (c ? { ...c, [key]: value } : c))
  }

  const requestedConfig =
    customize && customConfig
      ? customConfig
      : effectiveProfileConfig(customizationBaseProfile, vram, profileId === null)
  const riskyOverride =
    (profileId !== null || (customize && customConfig !== null)) &&
    isRiskyTrainingOverride(requestedConfig, vram, resolutionBuckets)
  useEffect(() => {
    if (!bucketTouched && profileId === null) {
      setResolutionBuckets(vram >= 80 ? '768x448x49' : '512x512x49')
    }
  }, [bucketTouched, profileId, vram])
  // RunPod-only guard: local eligibility already vetted the machine's GPU.
  const tooSmall = provider === 'runpod' && vram > 0 && vram < 32
  const estimateInputs: RunpodEstimateWorkload = {
    config: requestedConfig ?? undefined,
    clipCount: dataset.clips.filter((clip) => !clip.deletedAt && clip.triage !== 'reject').length,
    totalClipSeconds: dataset.clips.reduce(
      (sum, clip) => sum + (clip.durationSeconds ?? clip.probe?.durationSeconds ?? 0),
      0,
    ),
    preprocessed: false,
    resolutionBuckets,
    mode: dataset.type,
    withAudio: dataset.type === 'ic_lora' ? false : withAudio,
  }

  const submit = async () => {
    if (provider === 'runpod') {
      if (!runpodSelection) {
        setCapacityMessage('Choose an available GPU to continue.')
        setStep('runpod')
        return
      }
      setSaving(true)
      const stock = await getRunpodInventory()
      const freshGpu = stock.ok
        ? stock.data.gpus.find((gpu) => gpu.id === runpodSelection.gpuType)
        : undefined
      const stillAvailable = Boolean(freshGpu?.available) && (
        runpodSelection.workspacePolicy === 'primary_cache'
          ? stock.ok && stock.data.volumes.some(
              (volume) =>
                volume.id === runpodSelection.volumeId
                && volume.savedModelReadiness === 'ready'
                && (volume.availableGpuIds ?? []).includes(runpodSelection.gpuType),
            )
          : !runpodSelection.datacenter
            || freshGpu?.bestAvailableRegion === runpodSelection.datacenter
      )
      if (!stillAvailable) {
        setSaving(false)
        setCapacityMessage('That GPU is no longer available in the selected region. Choose again from the refreshed list.')
        setStep('runpod')
        return
      }
    }
    setSaving(true)
    await wslGate.ensureForRun(provider, async () => {
      // A per-run customization sends an inline config (no profileId); otherwise
      // the backend snapshots the selected profile's config (or auto-matches).
      const result = customize && customConfig
        ? await startTrainingPipeline({
            datasetId: dataset.id,
            name: name.trim() || dataset.name,
            description: description.trim() || null,
            resolutionBuckets: resolutionBuckets.trim(),
            withAudio: dataset.type === 'ic_lora' ? false : withAudio,
            autoCaption: dataset.type === 'ic_lora' ? false : autoCaption,
            captionerType,
            provider,
            workspacePolicy: provider === 'runpod' ? runpodSelection?.workspacePolicy : undefined,
            runpodSelection: provider === 'runpod' ? runpodSelection : undefined,
            config: customConfig,
            allowUnsafeOverride,
            triggerWordOverride: triggerWord.trim() || null,
            validationPrompts: parsePromptList(validationPromptsText),
          })
        : await startTrainingPipeline({
            datasetId: dataset.id,
            name: name.trim() || dataset.name,
            description: description.trim() || null,
            resolutionBuckets: resolutionBuckets.trim(),
            withAudio: dataset.type === 'ic_lora' ? false : withAudio,
            autoCaption: dataset.type === 'ic_lora' ? false : autoCaption,
            captionerType,
            provider,
            workspacePolicy: provider === 'runpod' ? runpodSelection?.workspacePolicy : undefined,
            runpodSelection: provider === 'runpod' ? runpodSelection : undefined,
            profileId,
            allowUnsafeOverride,
            triggerWordOverride: triggerWord.trim() || null,
            validationPrompts: parsePromptList(validationPromptsText),
          })
      setSaving(false)
      if (!result.ok) {
        if (provider === 'runpod' && isGpuSelectionRequired(result.error)) {
          setCapacityMessage(result.error)
          setStep('runpod')
        } else {
          setError(result.error)
        }
        return
      }
      onClose()
    })
    setSaving(false)
  }

  return (
    <>
    <ModalShell
      title={isGpuRecovery ? 'Choose another GPU' : 'Train LoRA'}
      onClose={onClose}
      closeDisabled={saving}
      className={provider === 'runpod' && step === 'runpod' ? 'max-w-4xl' : 'max-w-lg'}
      footer={
        <>
          <Button
            variant="outline"
            className="border-zinc-700"
            disabled={saving}
            onClick={() => provider === 'runpod' && step === 'runpod' ? setStep('details') : onClose()}
          >
            {provider === 'runpod' && step === 'runpod' ? 'Back' : 'Cancel'}
          </Button>
          <Button
            className="bg-blue-600 hover:bg-blue-500"
            disabled={saving || tooSmall || (riskyOverride && !allowUnsafeOverride) || (step === 'runpod' && !runpodSelection)}
            onClick={() => {
              if (provider === 'runpod' && step === 'details') setStep('runpod')
              else void submit()
            }}
          >
            {saving
              ? 'Rechecking stock…'
              : provider === 'runpod' && step === 'details'
                ? 'Choose GPU'
                : isGpuRecovery
                  ? 'Continue same pipeline'
                  : 'Start training'}
          </Button>
        </>
      }
    >
      {provider === 'runpod' && step === 'runpod' ? (
        <RunpodTrainingSetup
          value={runpodSelection}
          onChange={(selection) => {
            setRunpodSelection(selection)
            setCapacityMessage(null)
          }}
          estimateInputs={estimateInputs}
          disabled={saving}
          capacityMessage={capacityMessage}
          allowUnsafeOverride={allowUnsafeOverride}
          onAllowUnsafeOverrideChange={setAllowUnsafeOverride}
        />
      ) : (
      <>
      <div className="space-y-1.5">
        <label htmlFor="pipeline-run-name" className="text-xs font-medium text-zinc-300">Run name</label>
        <input
          id="pipeline-run-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <label htmlFor="pipeline-description" className="text-xs font-medium text-zinc-300">
            What does this LoRA do? <span className="font-normal text-zinc-500">(optional)</span>
          </label>
          <InfoHint content="Used as library metadata and to generate accurate auto-prompt instructions. It is intentionally separate from the run name." />
        </div>
        <textarea
          id="pipeline-description"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          maxLength={500}
          rows={2}
          placeholder={isIcLora
            ? 'e.g. Removes foreground people and reconstructs the hidden background'
            : 'e.g. Applies a soft cinematic watercolor style'}
          className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {isIcLora && (
          <p className="text-[10px] leading-relaxed text-blue-300/80">
            Recommended for IC-LoRA: this tells the prompt assistant what transformation to describe.
          </p>
        )}
      </div>

      <GpuSelector
        provider={provider}
        onChange={(p) => updateSettings({ loraProvider: p })}
        localEligibility={localEligibility}
        runpodGpuType={settings.runpodGpuType}
        runpodVramGb={settings.runpodGpuVramGb}
      />

      <ProfilePicker
        profiles={profiles}
        value={profileId}
        onChange={onPickProfile}
        datasetType={dataset.type}
        vramGb={vram}
        autoProfileId={recommendedProfileId}
      />

      <ResolutionInput
        value={resolutionBuckets}
        onChange={(v) => {
          setBucketTouched(true)
          setResolutionBuckets(v)
        }}
      />

      {riskyOverride && (
        <ExpertOverrideWarning
          checked={allowUnsafeOverride}
          onChange={setAllowUnsafeOverride}
          provider={provider}
        />
      )}

      {dataset.type === 'ic_lora' ? (
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-[11px] leading-relaxed text-amber-200">
          IC-LoRA target captions must be reviewed before training. Remote auto-captioning is unavailable because it would discard input/output pairing metadata.
        </div>
      ) : (
        <ToggleRow label="Auto-caption missing clips" hint={`Generate captions on the ${provider === 'runpod' ? 'RunPod training GPU' : 'local training GPU'} only for clips that do not already have one.`} checked={autoCaption} onChange={setAutoCaption} />
      )}
      {dataset.type !== 'ic_lora' && autoCaption && (
        <div className="space-y-1.5 pl-1">
          <label htmlFor="pipeline-captioner" className="text-xs font-medium text-zinc-300">Captioner</label>
          <select
            id="pipeline-captioner"
            value={captionerType}
            onChange={(e) => setCaptionerType(e.target.value as 'qwen_omni' | 'gemini_flash')}
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="gemini_flash">Gemini Flash (API — uses your Gemini key)</option>
            <option value="qwen_omni">Qwen Omni (on the training GPU)</option>
          </select>
        </div>
      )}

      {dataset.type !== 'ic_lora' && (
        <ToggleRow label="Train with audio" hint="Joint audio-video training. Every clip must already contain an audio track." checked={withAudio} onChange={setWithAudio} />
      )}

      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <label htmlFor="pipeline-trigger" className="text-xs font-medium text-zinc-300">Trigger word</label>
          <InfoHint content="Pre-filled from the dataset. This token is injected into the training captions so the LoRA learns to activate on it. Edit it here to override the dataset's trigger word for this run." />
        </div>
        <input
          id="pipeline-trigger"
          value={triggerWord}
          onChange={(e) => setTriggerWord(e.target.value)}
          placeholder="e.g. TOK"
          spellCheck={false}
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {isIcLora ? (
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-medium text-zinc-300">Validation</span>
            <InfoHint content="IC-LoRA validation conditions on a reference video, so a bare text prompt isn't enough. The run auto-uses one of your dataset clips as the validation reference (its caption is the prompt). Mark a clip as Holdout in the dataset studio to choose which one." />
          </div>
          <p className="text-[11px] text-zinc-500">
            Validation will auto-use a clip from this dataset as the reference, so you get a progress feed during training. Mark a clip as <span className="text-amber-400">Holdout</span> in the dataset studio to choose which one.
          </p>
        </div>
      ) : (
        <ValidationPromptsField
          value={validationPromptsText}
          onChange={setValidationPromptsText}
          note={
            autoCaption && !hasCaptions
              ? 'Captions will be generated during preprocessing; leave empty to auto-use them.'
              : 'Defaults to captions from this dataset. Edit to taste, or empty to auto-use generated captions.'
          }
        />
      )}

      {!customize ? (
        <button
          onClick={enableCustomize}
          disabled={!customizationBaseProfile}
          className="text-[11px] text-blue-400 hover:text-blue-300 disabled:text-zinc-600"
        >
          Customize for this run…
        </button>
      ) : (
        customConfig && (
          <div className="space-y-3 border-t border-zinc-800 pt-3">
            <div className="flex items-center justify-between">
              <p className="text-[11px] text-zinc-500">
                These overrides apply to this run only and won&rsquo;t change the profile.
              </p>
              <button
                onClick={() => { setCustomize(false); setCustomConfig(null) }}
                className="text-[11px] text-zinc-400 hover:text-white shrink-0"
              >
                Reset
              </button>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {PRIMARY_FIELDS.map((field) => (
                <ConfigField
                  key={field.key}
                  field={field}
                  value={customConfig[field.key]}
                  onChange={(v) => setConfigValue(field.key, v)}
                />
              ))}
            </div>
            {SECTIONS.map((section) => (
              <CollapsibleSection key={section.id} title={section.title} defaultOpen={section.defaultOpen}>
                {SECTION_FIELDS[section.id].map((field) => (
                  <ConfigField
                    key={field.key}
                    field={field}
                    value={customConfig[field.key]}
                    onChange={(v) => setConfigValue(field.key, v)}
                  />
                ))}
              </CollapsibleSection>
            ))}
          </div>
        )
      )}

      {tooSmall && (
        <p className="text-[11px] text-red-400">
          The selected GPU has only {vram} GB of VRAM; LoRA training needs at least 32 GB. Switch to Local GPU above or pick a larger GPU in Settings.
        </p>
      )}
      <FieldError message={error} />
      </>
      )}
    </ModalShell>
    {wslGate.dialog}
    </>
  )
}

// ---------------------------------------------------------------------------
// Start training
// ---------------------------------------------------------------------------

export function StartTrainingModal({
  preprocessed,
  onClose,
  onManageProfiles,
}: {
  preprocessed: LoraPreprocessed
  onClose: () => void
  onManageProfiles?: () => void
}) {
  const { profiles, datasets, startTraining, localEligibility } = useLoraTraining()
  const { settings, updateSettings, getRunpodInventory } = useAppSettings()
  const wslGate = useLocalWslMemoryGate()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [triggerWord, setTriggerWord] = useState('')
  const [customize, setCustomize] = useState(false)
  // Per-run config fork, seeded from the picked profile when customizing.
  const [customConfig, setCustomConfig] = useState<LoraTrainingConfig | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [allowUnsafeOverride, setAllowUnsafeOverride] = useState(false)
  const [step, setStep] = useState<'details' | 'runpod'>('details')
  const [runpodSelection, setRunpodSelection] = useState<RunpodSelection | null>(null)
  const [capacityMessage, setCapacityMessage] = useState<string | null>(null)

  const provider = settings.loraProvider
  const vram = activeVramGb(
    provider,
    runpodSelection?.gpuVramGb ?? settings.runpodGpuVramGb,
    localEligibility,
  )

  const dataset = datasets.find((d) => d.id === preprocessed.datasetId) ?? null
  const isIcLora = dataset?.type === 'ic_lora'
  // Pre-fill the trigger word from the dataset so the user sees what the run
  // will train on and can edit it per-run. The dataset's trigger word is what
  // preprocessing injects into captions; an explicit value here overrides it
  // for this run only.
  useEffect(() => {
    if (dataset?.triggerWord) setTriggerWord(dataset.triggerWord)
  }, [dataset?.triggerWord, dataset?.id])
  // Pre-fill validation prompts from the dataset's captions (standard/t2v) so
  // the user sees and can approve what the run validates against. IC-LoRA
  // ignores bare prompts (validation uses a reference video), so there's no
  // prompt field for it — just a note about the auto-picked clip.
  const [validationPromptsText, setValidationPromptsText] = useState(() =>
    dataset && !isIcLora ? suggestedValidationPrompts(dataset).join('\n') : '',
  )

  // Auto chooses the recommended training goal for the dataset type. Hardware
  // adaptation is applied independently from the selected goal.
  const recommendedProfileId = useMemo(
    () => defaultProfileIdForVram(profiles, vram, dataset?.type ?? 'standard'),
    [profiles, vram, dataset?.type],
  )
  const [pickedProfileId, setPickedProfileId] = useState<string | null>(null)
  const profileId = pickedProfileId ?? recommendedProfileId

  const selectedProfile = profiles.find((p) => p.id === profileId) ?? null
  const requestedConfig =
    customize && customConfig
      ? customConfig
      : effectiveProfileConfig(selectedProfile, vram, pickedProfileId === null)
  const riskyOverride = isRiskyTrainingOverride(
    requestedConfig,
    vram,
    preprocessed.effectiveResolutionBuckets ?? preprocessed.resolutionBuckets,
  )
  const estimateInputs: RunpodEstimateWorkload = {
    config: requestedConfig ?? undefined,
    clipCount: dataset?.clips.filter((clip) => !clip.deletedAt && clip.triage !== 'reject').length ?? 1,
    totalClipSeconds: dataset?.clips.reduce(
      (sum, clip) => sum + (clip.durationSeconds ?? clip.probe?.durationSeconds ?? 0),
      0,
    ) ?? 0,
    preprocessed: true,
    resolutionBuckets: preprocessed.effectiveResolutionBuckets ?? preprocessed.resolutionBuckets,
    mode: dataset?.type ?? 'standard',
    withAudio: preprocessed.withAudio,
  }

  const enableCustomize = () => {
    setCustomize(true)
    const effective = effectiveProfileConfig(selectedProfile, vram, pickedProfileId === null)
    if (effective) setCustomConfig(effective)
  }

  const onPickProfile = (id: string | null) => {
    setPickedProfileId(id)
    // Re-seed an in-progress customization from the newly picked profile (or
    // the recommendation when reverting to Auto).
    if (customize) {
      const next = profiles.find((p) => p.id === (id ?? recommendedProfileId))
      const effective = effectiveProfileConfig(next ?? null, vram, id === null)
      if (effective) setCustomConfig(effective)
    }
  }

  const setConfigValue = (key: string, value: unknown) => {
    setCustomConfig((c) => (c ? { ...c, [key]: value } : c))
  }

  const submit = async () => {
    if (!name.trim()) {
      setError('Please name this training run.')
      return
    }
    if (!profileId && !customConfig) {
      setError('Pick a training profile.')
      return
    }
    if (provider === 'runpod') {
      if (!runpodSelection) {
        setCapacityMessage('Choose an available GPU to continue.')
        setStep('runpod')
        return
      }
      setSaving(true)
      const stock = await getRunpodInventory()
      const stillAvailable = stock.ok && stock.data.gpus.some(
        (gpu) =>
          gpu.id === runpodSelection.gpuType
          && gpu.available,
      )
      if (!stillAvailable) {
        setSaving(false)
        setCapacityMessage('That GPU is no longer available. Choose another GPU from the refreshed list.')
        setStep('runpod')
        return
      }
    }
    const trigger = triggerWord.trim() || null
    setSaving(true)
    // A per-run customization sends an inline config. A null picked id keeps
    // Auto authoritative on the backend instead of disguising it as an
    // explicit profile selection.
    const validationPrompts = parsePromptList(validationPromptsText)
    const runStart = async () => {
      const result =
        customize && customConfig
          ? await startTraining({
              preprocessedId: preprocessed.id,
              name: name.trim(),
              description: description.trim() || null,
              config: customConfig,
              allowUnsafeOverride,
              triggerWordOverride: trigger,
              provider,
              runpodSelection: provider === 'runpod' ? runpodSelection : undefined,
              validationPrompts,
            })
          : await startTraining({
              preprocessedId: preprocessed.id,
              name: name.trim(),
              description: description.trim() || null,
              profileId: pickedProfileId,
              allowUnsafeOverride,
              triggerWordOverride: trigger,
              provider,
              runpodSelection: provider === 'runpod' ? runpodSelection : undefined,
              validationPrompts,
            })
      setSaving(false)
      if (!result.ok) {
        if (provider === 'runpod' && isGpuSelectionRequired(result.error)) {
          setCapacityMessage(result.error)
          setStep('runpod')
        } else {
          setError(result.error)
        }
        return
      }
      onClose()
    }
    await wslGate.ensureForRun(provider, runStart)
    setSaving(false)
  }

  return (
    <>
    <ModalShell
      title="Start Training Run"
      onClose={onClose}
      closeDisabled={saving}
      className={provider === 'runpod' && step === 'runpod' ? 'max-w-4xl' : 'max-w-lg'}
      footer={
        <>
          <Button
            variant="outline"
            className="border-zinc-700"
            disabled={saving}
            onClick={() => provider === 'runpod' && step === 'runpod' ? setStep('details') : onClose()}
          >
            {provider === 'runpod' && step === 'runpod' ? 'Back' : 'Cancel'}
          </Button>
          <Button
            className="bg-blue-600 hover:bg-blue-500"
            disabled={saving || (riskyOverride && !allowUnsafeOverride) || (step === 'runpod' && !runpodSelection)}
            onClick={() => {
              if (provider === 'runpod' && step === 'details') setStep('runpod')
              else void submit()
            }}
          >
            {saving ? 'Rechecking stock…' : provider === 'runpod' && step === 'details' ? 'Choose GPU' : 'Start Training'}
          </Button>
        </>
      }
    >
      {provider === 'runpod' && step === 'runpod' ? (
        <>
          <RunpodTrainingSetup
            value={runpodSelection}
            onChange={(selection) => {
              setRunpodSelection(selection)
              setCapacityMessage(null)
            }}
            estimateInputs={estimateInputs}
            disabled={saving}
            capacityMessage={capacityMessage}
            allowUnsafeOverride={allowUnsafeOverride}
            onAllowUnsafeOverrideChange={setAllowUnsafeOverride}
          />
        </>
      ) : (
      <>
      <div className="space-y-1.5">
        <label htmlFor="training-run-name" className="text-xs font-medium text-zinc-300">Run name</label>
        <input
          id="training-run-name"
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="my-lora-v1"
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <label htmlFor="training-description" className="text-xs font-medium text-zinc-300">
            What does this LoRA do? <span className="font-normal text-zinc-500">(optional)</span>
          </label>
          <InfoHint content="Used as library metadata and to generate accurate auto-prompt instructions. It is not part of the reusable training profile." />
        </div>
        <textarea
          id="training-description"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          maxLength={500}
          rows={2}
          placeholder={dataset?.type === 'ic_lora'
            ? 'e.g. Removes foreground people and reconstructs the hidden background'
            : 'e.g. Applies a soft cinematic watercolor style'}
          className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {dataset?.type === 'ic_lora' && (
          <p className="text-[10px] leading-relaxed text-blue-300/80">
            Recommended for IC-LoRA: this tells the prompt assistant what transformation to describe.
          </p>
        )}
      </div>

      <GpuSelector
        provider={provider}
        onChange={(p) => updateSettings({ loraProvider: p })}
        localEligibility={localEligibility}
        runpodGpuType={settings.runpodGpuType}
        runpodVramGb={settings.runpodGpuVramGb}
      />

      <ProfilePicker
        profiles={profiles}
        value={pickedProfileId}
        onChange={onPickProfile}
        onManageProfiles={onManageProfiles}
        datasetType={dataset?.type ?? 'standard'}
        vramGb={vram}
        autoProfileId={recommendedProfileId}
      />

      {riskyOverride && (
        <ExpertOverrideWarning
          checked={allowUnsafeOverride}
          onChange={setAllowUnsafeOverride}
          provider={provider}
        />
      )}

      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <label htmlFor="training-trigger" className="text-xs font-medium text-zinc-300">Trigger word</label>
          <InfoHint content="Pre-filled from the dataset. This token is injected into the training captions so the LoRA learns to activate on it. Edit it here to override the dataset's trigger word for this run only." />
        </div>
        <input
          id="training-trigger"
          value={triggerWord}
          onChange={(e) => setTriggerWord(e.target.value)}
          placeholder="e.g. TOK"
          spellCheck={false}
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {isIcLora ? (
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-medium text-zinc-300">Validation</span>
            <InfoHint content="IC-LoRA validation conditions on a reference video, so a bare text prompt isn't enough. The run auto-uses one of your dataset clips as the validation reference (its caption is the prompt). Mark a clip as Holdout in the dataset studio to choose which one." />
          </div>
          <p className="text-[11px] text-zinc-500">
            Validation will auto-use a clip from this dataset as the reference, so you get a progress feed during training. Mark a clip as <span className="text-amber-400">Holdout</span> in the dataset studio to choose which one.
          </p>
        </div>
      ) : (
        <ValidationPromptsField
          value={validationPromptsText}
          onChange={setValidationPromptsText}
          note="Defaults to captions from this dataset. Edit to taste, or empty to auto-use generated captions."
        />
      )}

      {!customize ? (
        <button
          onClick={enableCustomize}
          disabled={!selectedProfile}
          className="text-[11px] text-blue-400 hover:text-blue-300 disabled:text-zinc-600"
        >
          Customize for this run…
        </button>
      ) : (
        customConfig && (
          <div className="space-y-3 border-t border-zinc-800 pt-3">
            <div className="flex items-center justify-between">
              <p className="text-[11px] text-zinc-500">
                These overrides apply to this run only and won't change the profile.
              </p>
              <button
                onClick={() => { setCustomize(false); setCustomConfig(null) }}
                className="text-[11px] text-zinc-400 hover:text-white shrink-0"
              >
                Reset
              </button>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {PRIMARY_FIELDS.map((field) => (
                <ConfigField
                  key={field.key}
                  field={field}
                  value={customConfig[field.key]}
                  onChange={(v) => setConfigValue(field.key, v)}
                />
              ))}
            </div>
            {SECTIONS.map((section) => (
              <CollapsibleSection key={section.id} title={section.title} defaultOpen={section.defaultOpen}>
                {SECTION_FIELDS[section.id].map((field) => (
                  <ConfigField
                    key={field.key}
                    field={field}
                    value={customConfig[field.key]}
                    onChange={(v) => setConfigValue(field.key, v)}
                  />
                ))}
              </CollapsibleSection>
            ))}
          </div>
        )
      )}
      <FieldError message={error} />
      </>
      )}
    </ModalShell>
    {wslGate.dialog}
    </>
  )
}

export function RunpodGpuRecoveryModal({
  job,
  dataset,
  preprocessed,
  onClose,
  onRecovered,
}: {
  job: { id: string; config: LoraTrainingConfig; error?: string | null }
  dataset: LoraDataset | null
  preprocessed: LoraPreprocessed | null
  onClose: () => void
  onRecovered?: () => Promise<void>
}) {
  const { getRunpodInventory, reselectRunpod } = useAppSettings()
  const [selection, setSelection] = useState<RunpodSelection | null>(null)
  const [saving, setSaving] = useState(false)
  const [allowUnsafeOverride, setAllowUnsafeOverride] = useState(false)
  const [message, setMessage] = useState(
    job.error || 'GPU capacity changed. Choose another GPU to continue this same job.',
  )
  const recoveryResolutionBuckets = preprocessed?.effectiveResolutionBuckets
    ?? preprocessed?.resolutionBuckets
    ?? '768x448x49'
  const riskySelection = selection
    ? isRiskyTrainingOverride(job.config, selection.gpuVramGb, recoveryResolutionBuckets)
    : false
  const submit = async () => {
    if (!selection) return
    setSaving(true)
    const stock = await getRunpodInventory()
    const freshGpu = stock.ok
      ? stock.data.gpus.find((gpu) => gpu.id === selection.gpuType)
      : undefined
    const available = Boolean(freshGpu?.available) && (
      selection.workspacePolicy === 'primary_cache'
        ? stock.ok && stock.data.volumes.some(
            (volume) =>
              volume.id === selection.volumeId
              && volume.savedModelReadiness === 'ready'
              && (volume.availableGpuIds ?? []).includes(selection.gpuType),
          )
        : !selection.datacenter || freshGpu?.bestAvailableRegion === selection.datacenter
    )
    if (!available) {
      setSaving(false)
      setMessage('That GPU is no longer available in the selected region. Choose another GPU from the refreshed list.')
      return
    }
    const result = await reselectRunpod({ kind: 'training', id: job.id }, selection)
    if (!result.ok) {
      setSaving(false)
      setMessage(result.error.message)
      return
    }
    await onRecovered?.()
    setSaving(false)
    onClose()
  }
  return (
    <ModalShell
      title="Choose another GPU"
      onClose={onClose}
      closeDisabled={saving}
      className="max-w-4xl"
      footer={
        <>
          <Button variant="outline" className="border-zinc-700" disabled={saving} onClick={onClose}>Cancel</Button>
          <Button className="bg-amber-500 text-zinc-950 hover:bg-amber-400" disabled={saving || !selection || (riskySelection && !allowUnsafeOverride)} onClick={() => void submit()}>
            {saving ? 'Rechecking stock…' : 'Continue same job'}
          </Button>
        </>
      }
    >
      <RunpodTrainingSetup
        value={selection}
        onChange={(next) => { setSelection(next); setMessage('GPU capacity changed. Choose another GPU to continue this same job.') }}
        estimateInputs={{
          config: job.config,
          clipCount: dataset?.clips.length ?? 1,
          totalClipSeconds: dataset?.clips.reduce(
            (sum, clip) => sum + (clip.durationSeconds ?? clip.probe?.durationSeconds ?? 0),
            0,
          ) ?? 0,
          preprocessed: Boolean(preprocessed),
          resolutionBuckets: recoveryResolutionBuckets,
          mode: dataset?.type ?? 'standard',
          withAudio: preprocessed?.withAudio ?? false,
        }}
        disabled={saving}
        capacityMessage={message}
        allowUnsafeOverride={allowUnsafeOverride}
        onAllowUnsafeOverrideChange={setAllowUnsafeOverride}
      />
    </ModalShell>
  )
}

// ---------------------------------------------------------------------------
// Logs
// ---------------------------------------------------------------------------

export function TrainingLogsModal({ trainingId, name, onClose }: { trainingId: string; name: string; onClose: () => void }) {
  const { fetchLogs } = useLoraTraining()
  const [lines, setLines] = useState<string[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const next = await fetchLogs(trainingId)
      if (cancelled) return
      setLines(next)
      setLoading(false)
    }
    void load()
    const interval = setInterval(() => { void load() }, 4000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [fetchLogs, trainingId])

  return (
    <ModalShell title={`Logs — ${name}`} onClose={onClose} footer={<Button className="bg-zinc-700 hover:bg-zinc-600" onClick={onClose}>Close</Button>}>
      {loading ? (
        <p className="text-xs text-zinc-500">Loading logs...</p>
      ) : lines.length === 0 ? (
        <p className="text-xs text-zinc-500">No logs yet.</p>
      ) : (
        <pre className="text-[11px] text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-950 rounded-lg p-3 max-h-[50vh] overflow-y-auto border border-zinc-800">
          {lines.join('\n')}
        </pre>
      )}
    </ModalShell>
  )
}

// ---------------------------------------------------------------------------
// Small shared inputs
// ---------------------------------------------------------------------------

function ToggleRow({ label, hint, checked, onChange }: { label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-lg bg-zinc-800/50 px-3 py-2.5 focus-within:ring-2 focus-within:ring-blue-500">
      <div className="flex items-center gap-1.5">
        <span className="text-sm text-zinc-200">{label}</span>
        {hint && <InfoHint content={hint} />}
      </div>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="sr-only"
      />
      <span aria-hidden="true" className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${checked ? 'bg-blue-500' : 'bg-zinc-700'}`}>
        <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${checked ? 'translate-x-5' : 'translate-x-0'}`} />
      </span>
    </label>
  )
}

function NumberField({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="block space-y-1.5">
      <span className="block text-xs font-medium text-zinc-300">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
    </label>
  )
}

/** Up to two distinct training-clip captions, preferring ones that mention the
 * trigger word. Mirrors the backend auto-seed so the field pre-fills with what
 * the run will actually use. */
function suggestedValidationPrompts(dataset: LoraDataset): string[] {
  const trigger = (dataset.triggerWord ?? '').trim().toLowerCase()
  const trainClips = dataset.clips.filter(
    (c) => !c.deletedAt && c.triage !== 'reject' && c.triage !== 'holdout',
  )
  const seeded: string[] = []
  const seen = new Set<string>()
  const consider = (cap: string): boolean => {
    const key = cap.trim().toLowerCase()
    if (!cap.trim() || seen.has(key)) return false
    seen.add(key)
    seeded.push(cap.trim())
    return true
  }
  if (trigger) {
    for (const c of trainClips) {
      if (seeded.length >= 3) break
      const cap = c.caption ?? ''
      if (cap && cap.toLowerCase().includes(trigger)) consider(cap)
    }
  }
  for (const c of trainClips) {
    if (seeded.length >= 3) break
    consider(c.caption ?? '')
  }
  return seeded.slice(0, 3)
}

function parsePromptList(text: string): string[] {
  return text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
}

/** Editable validation-prompts field, pre-filled from the dataset's captions
 * (or the user's existing config) so the user can read and approve what the
 * run will validate against, instead of the generic placeholder. */
function ValidationPromptsField({
  value,
  onChange,
  note,
}: {
  value: string
  onChange: (v: string) => void
  note?: string
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <label htmlFor="training-validation-prompts" className="text-xs font-medium text-zinc-300">Validation prompts</label>
        <InfoHint content="One prompt per line. Sampled during training to monitor progress. Defaults to captions from your dataset; edit to taste." />
      </div>
      <textarea
        id="training-validation-prompts"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        spellCheck={false}
        placeholder="One prompt per line (leave empty to auto-use your captions)"
        className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
      />
      {note && <p className="text-[11px] text-zinc-500">{note}</p>}
    </div>
  )
}

