import { useEffect, useState } from 'react'
import { ChevronUp, Cloud, Loader2, Wand2, X } from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import { useLoraTraining, type CreateDerivationBody, type FrameEditEngine } from '../../contexts/LoraTrainingContext'
import { SettingsDropdown, LightricksIcon } from '../../components/ui/settings-dropdown'
import type { StudioClip } from '../studio/studio-store'

const EDIT_SUGGESTIONS = [
  'Remove the background',
  'Change the outfit',
  'Add falling snow',
  'Make it nighttime',
  'Replace the person',
]

// Step 1 of the paired-dataset pipeline, surfaced on its own so the edited
// still lands in the gallery for review BEFORE any video is generated:
//   pick a frame → edit it → the still is added to the gallery.
//
// The edit itself runs on the LoRA Trainer queue (a `frame_edit` derivation
// job), so it doesn't block the UI and a local GPU (FLUX.2 Klein) edit backs
// off while another generation is in flight instead of failing. When the job
// completes, the still is folded into the gallery automatically (it remembers
// its source clip / driver, so a later "Make pair" can motion-lock it back onto
// the original video).
//
// Batch mode (`batchClips`): applies the SAME edit instruction to a frame from
// every selected clip — one queued job per clip.
export function FrameEditModal({
  clip,
  batchClips,
  datasetId,
  onClose,
}: {
  clip: StudioClip
  batchClips?: StudioClip[]
  datasetId: string
  onClose: () => void
}) {
  const { createDerivation, extractFrame } = useLoraTraining()
  const isBatch = (batchClips?.length ?? 0) > 1
  const duration = clip.probe?.durationSeconds ?? clip.durationSeconds ?? 0
  const sourcePoster = clip.posterPath ? pathToFileUrl(clip.posterPath) : null

  const [frameTime, setFrameTime] = useState(0)
  const [framePosition, setFramePosition] = useState(0)
  const [prompt, setPrompt] = useState('')
  const [engine, setEngine] = useState<FrameEditEngine>('klein')
  const [sourceFramePath, setSourceFramePath] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Show the actual frame at the chosen time as the "before" — not the clip's
  // poster, which is a fixed mid-clip thumbnail and made the preview misleading.
  // Extraction is deterministic at the same timestamp the edit uses, so the
  // preview always lines up with what the queued job will edit. Debounced so
  // dragging the slider doesn't fire an ffmpeg call per tick; falls back to the
  // poster.
  useEffect(() => {
    if (isBatch) return
    let cancelled = false
    const handle = window.setTimeout(() => {
      void (async () => {
        const res = await extractFrame(clip.sourcePath, frameTime)
        if (!cancelled && res.ok) setSourceFramePath(res.data)
      })()
    }, 200)
    return () => {
      cancelled = true
      window.clearTimeout(handle)
    }
  }, [clip.sourcePath, frameTime, isBatch, extractFrame])

  const sourceUrl = sourceFramePath ? pathToFileUrl(sourceFramePath) : sourcePoster

  const appendSuggestion = (text: string) =>
    setPrompt((p) => (p.trim() ? `${p.trim()}, ${text.toLowerCase()}` : text))

  // Build a `frame_edit` derivation job body for one clip. The edit runs on the
  // LoRA Trainer queue; on completion the still folds into the gallery via
  // CollectionView (driverPath is remembered so it can be motion-locked later).
  const buildBody = (target: StudioClip): CreateDerivationBody => {
    const dur = target.probe?.durationSeconds ?? target.durationSeconds ?? 0
    const at = isBatch ? framePosition * dur : frameTime
    const caption = prompt.trim()
    return {
      driverPath: target.localPath,
      // Still entries use the image directly; videos get a frame extracted at
      // `frameTimeSeconds` by the runner.
      framePath: target.kind === 'image' ? target.localPath : null,
      referencePath: null,
      direction: 'frame_edit',
      datasetId,
      sourceClipId: target.id,
      frameTimeSeconds: target.kind === 'image' ? 0 : at,
      editPrompt: prompt.trim(),
      editEngine: engine,
      // No review for frame edits — the still lands in the gallery for triage.
      requireReview: false,
      scenePrompt: '',
      // Motion-drive fields are unused for frame_edit (no animate step); pass
      // valid literals so the request type is satisfied.
      engine: 'ltx_local',
      conditioningType: 'depth',
      conditioningStrength: 1,
      characterOrientation: 'video',
      keepAudio: true,
      frameEdited: true,
      caption,
      label: caption || target.caption || 'Edited still',
    }
  }

  const enqueueOne = async (target: StudioClip) => createDerivation(buildBody(target))

  const runEdit = async () => {
    if (!prompt.trim()) return
    setBusy(true)
    setError(null)
    const res = await enqueueOne(clip)
    setBusy(false)
    if (!res.ok) {
      setError(res.error)
      return
    }
    onClose()
  }

  const runBatch = async () => {
    const targets = batchClips ?? []
    if (!prompt.trim() || targets.length === 0) return
    setBusy(true)
    setError(null)
    let failures = 0
    for (const target of targets) {
      const res = await enqueueOne(target)
      if (!res.ok) failures++
    }
    setBusy(false)
    if (failures > 0) {
      setError(`${failures} of ${targets.length} clips could not be queued.`)
      return
    }
    onClose()
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-2xl mx-4">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="text-base font-semibold text-white">
            {isBatch
              ? `Edit first frame · ${batchClips?.length ?? 0} clips (${engine === 'klein' ? 'FLUX.2 Klein' : 'Nano Banana'})`
              : `Edit a frame (${engine === 'klein' ? 'FLUX.2 Klein' : 'Nano Banana'})`}
          </h2>
          <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4">
          {isBatch ? (
            <>
              <div className="text-[11px] text-zinc-400 bg-zinc-800/40 border border-zinc-800 rounded-md px-3 py-2">
                The same edit is applied to a frame from{' '}
                <span className="text-white font-medium">{batchClips?.length ?? 0} clips</span>. Each edit is queued on
                the LoRA Trainer; as the jobs finish, the new stills land in the gallery and remember their source clip,
                so you can later motion-lock them back into training examples.
              </div>
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[11px] text-zinc-400">
                  <span>Frame position</span>
                  <span className="font-mono">{Math.round(framePosition * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={framePosition}
                  onChange={(e) => setFramePosition(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>
            </>
          ) : (
            <>
              <div className="space-y-1">
                <div className="text-[11px] text-zinc-500">Frame to edit</div>
                <div className="aspect-video rounded-lg overflow-hidden bg-zinc-950">
                  {sourceUrl && <img src={sourceUrl} alt="source frame" className="w-full h-full object-contain" />}
                </div>
              </div>

              {duration > 0 && (
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-[11px] text-zinc-400">
                    <span>Frame to edit</span>
                    <span className="font-mono">{frameTime.toFixed(1)}s</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={duration}
                    step={0.1}
                    value={frameTime}
                    onChange={(e) => setFrameTime(Number(e.target.value))}
                    className="w-full accent-blue-500"
                  />
                </div>
              )}
            </>
          )}

          <div className="space-y-2">
            {/* GenSpace-style generation box mirroring the "Generate example"
             *  wizard's Frame & edit step: a borderless textarea over a slim
             *  settings toolbar (image-model dropdown on the left, where-it-runs
             *  hint on the right), then a row of quick edits with the edit
             *  action. overflow-visible so the upward-opening model menu isn't
             *  clipped by the card's top edge. */}
            <div className="bg-zinc-950 border border-zinc-800 rounded-2xl overflow-visible focus-within:border-zinc-700 transition-colors">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                autoFocus
                placeholder="e.g. remove the person on the left, add falling snow, change the shirt to red…"
                className="h-[72px] w-full resize-none overflow-y-auto bg-transparent px-3.5 py-3 text-sm leading-5 text-white placeholder:text-zinc-500 focus:outline-none"
              />
              <div className="flex items-center gap-0.5 border-t border-zinc-800/60 px-1.5 py-1.5 text-xs text-zinc-400">
                <SettingsDropdown
                  title="IMAGE MODEL"
                  value={engine}
                  onChange={(v) => setEngine(v as FrameEditEngine)}
                  options={[
                    {
                      value: 'klein',
                      label: 'FLUX.2 Klein',
                      description: 'Local · GPU',
                      icon: <LightricksIcon className="h-4 w-4" />,
                      tooltip: 'FLUX.2 [klein] 9B — local, needs the gated HuggingFace checkpoint downloaded',
                    },
                    {
                      value: 'fal',
                      label: 'Nano Banana',
                      description: 'Remote · Fal',
                      icon: <Cloud className="h-4 w-4" />,
                      tooltip: 'Nano Banana — remote, needs a Fal API key',
                    },
                  ]}
                  tooltip={
                    engine === 'klein'
                      ? 'FLUX.2 [klein] 9B (local, needs the gated HuggingFace checkpoint downloaded)'
                      : 'Nano Banana (remote, needs a Fal API key)'
                  }
                  trigger={
                    <>
                      {engine === 'klein' ? (
                        <LightricksIcon className="h-4 w-4 text-zinc-200" />
                      ) : (
                        <Cloud className="h-4 w-4" />
                      )}
                      <span className="font-medium text-zinc-200">
                        {engine === 'klein' ? 'FLUX.2 Klein' : 'Nano Banana'}
                      </span>
                      <ChevronUp className="h-3 w-3 text-zinc-500" />
                    </>
                  }
                />
                <span className="ml-auto whitespace-nowrap pr-1.5 text-[11px] text-zinc-500">
                  {engine === 'klein' ? 'Runs on your GPU' : 'Runs on Fal · cloud'}
                </span>
              </div>
              <div className="flex items-center gap-1 px-2 py-1.5 border-t border-zinc-800/60">
                <div className="flex-1 flex flex-wrap gap-1 min-w-0">
                  {EDIT_SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => appendSuggestion(s)}
                      className="text-[11px] px-2 py-0.5 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
                    >
                      + {s}
                    </button>
                  ))}
                </div>
                {!isBatch && (
                  <button
                    onClick={() => void runEdit()}
                    disabled={!prompt.trim() || busy}
                    className="shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border border-blue-500/40 bg-blue-500/10 text-blue-200 hover:bg-blue-500/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
                    Edit frame
                  </button>
                )}
              </div>
            </div>
            <span className="text-[11px] text-zinc-500">
              {engine === 'klein'
                ? 'Local GPU edit — needs the Klein checkpoint (Model Status). Queued on the LoRA Trainer, so it waits its turn if the GPU is busy.'
                : 'Remote edit — queued on the LoRA Trainer and added to the gallery when it finishes.'}
            </span>
          </div>

          {error && <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-md px-3 py-2">{error}</div>}
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex items-center justify-end gap-2">
          <button onClick={onClose} className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white">Cancel</button>
          {isBatch && (
            <button
              onClick={runBatch}
              disabled={busy || !prompt.trim()}
              className="text-xs px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
              Edit {batchClips?.length ?? 0} frames
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
