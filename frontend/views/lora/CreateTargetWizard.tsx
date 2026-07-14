import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronUp,
  Cloud,
  Film,
  Gauge,
  Image as ImageIcon,
  Layers,
  Loader2,
  RefreshCw,
  Sparkles,
  Volume2,
  VolumeX,
  Wand2,
  X,
} from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import { Tooltip } from '../../components/ui/tooltip'
import { InfoTip } from '../../components/ui/info-tip'
import { SettingsDropdown, LightricksIcon } from '../../components/ui/settings-dropdown'
import {
  useLoraTraining,
  type CreateDerivationBody,
  type DerivationConditioning,
  type DerivationDirection,
  type DerivationEditEngine,
  type DerivationEngine,
  type LoraDatasetType,
} from '../../contexts/LoraTrainingContext'

/**
 * A compact segmented control: minimal by default, with a per-option tooltip
 * that carries the explanation on hover so the surface stays clean.
 */
function SegToggle<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: ReadonlyArray<{ value: T; label: string; tip: string }>
  onChange: (v: T) => void
}) {
  return (
    <div className="flex rounded-lg border border-zinc-800 bg-zinc-950 p-0.5 text-[11px]">
      {options.map((o) => (
        <Tooltip key={o.value} content={o.tip} side="bottom" wide>
          <button
            type="button"
            onClick={() => onChange(o.value)}
            className={`rounded-md px-2.5 py-1 font-medium transition-colors ${
              value === o.value
                ? 'bg-blue-500/20 text-blue-200 ring-1 ring-blue-500/30'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {o.label}
          </button>
        </Tooltip>
      ))}
    </div>
  )
}
import type { EditFramePreview, StudioClip } from '../studio/studio-store'

const EDIT_SUGGESTIONS = [
  'Remove the background',
  'Change the outfit',
  'Add falling snow',
  'Make it nighttime',
  'Replace the person',
]

// LTX conditioning strength presets, surfaced as a composer-bar dropdown so the
// control matches GenSpace's settings pills (the raw value is still any 0–1).
const STRENGTH_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: '1', label: 'Strict · 100%' },
  { value: '0.85', label: 'Strong · 85%' },
  { value: '0.7', label: 'Balanced · 70%' },
  { value: '0.5', label: 'Loose · 50%' },
]

type Stage = 1 | 2
const STAGES: ReadonlyArray<{ n: Stage; label: string; hint: string; detail: string }> = [
  {
    n: 1,
    label: 'Frame & edit',
    hint: 'Pick the frame to start from, then optionally change what is in it.',
    detail:
      'Choose which frame of the clip to start from, then optionally describe a change — the selected image model (FLUX.2 Klein locally, or Nano Banana via Fal) edits that single frame (add, remove, or replace things) before it’s animated. Leave the prompt blank to keep the frame as-is.',
  },
  {
    n: 2,
    label: 'Animate',
    hint: 'Turn the frame into a clip using the source motion.',
    detail:
      'Drives the chosen frame into a video using the source clip’s motion. Pick the model and its settings in the bar below; the finished clip is paired with this one as a training example.',
  },
]

const FRAME_PRESETS: ReadonlyArray<readonly [string, number]> = [
  ['Start', 0],
  ['Middle', 0.5],
  ['End', 1],
]

/**
 * A video thumbnail that seeks to a fraction (0–1) of its duration, so the
 * frame slider shows the *actual* frame the user is choosing instead of a
 * static poster. Falls back to the poster until metadata loads.
 */
function ScrubFrame({
  videoUrl,
  posterUrl,
  fraction,
  className,
  showTimeBadge = false,
}: {
  videoUrl: string
  posterUrl: string | null
  fraction: number
  className?: string
  /** Overlay the resolved timestamp — makes a relative position concrete
   *  per clip (the same fraction lands on a different second in each). */
  showTimeBadge?: boolean
}) {
  const ref = useRef<HTMLVideoElement>(null)
  const [duration, setDuration] = useState(0)
  const time = Math.min(duration, Math.max(0, fraction * duration))

  useEffect(() => {
    const v = ref.current
    if (v && duration > 0) v.currentTime = time
  }, [time, duration])

  return (
    <>
      <video
        ref={ref}
        src={videoUrl}
        poster={posterUrl ?? undefined}
        muted
        playsInline
        preload="metadata"
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
        className={className}
      />
      {showTimeBadge && duration > 0 && (
        <span className="absolute bottom-1 right-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-mono text-zinc-200">
          {time.toFixed(1)}s
        </span>
      )}
    </>
  )
}

/**
 * The unified "generate an example/variant" pipeline, surfaced as an explicit
 * 2-step flow that mirrors how the clip is actually produced:
 *
 *   1. Frame & edit — pick the starting frame (videos; stills skip this) and
 *      optionally change its content with the selected image model (FLUX.2 Klein
 *      locally, or Nano Banana via Fal — add/remove/replace).
 *   2. Animate — drive the (edited) still into a clip using the original as the
 *      motion driver:
 *        • LTX (control): local IC-LoRA depth/canny conditioning.
 *        • Kling: remote motion-control.
 *
 * Submitting enqueues a *background* job (see `createDerivation`) so the app
 * stays usable; progress shows up in the jobs tray and the finished clip lands
 * in the gallery. In an IC-LoRA collection the result is a paired *target* (its
 * reference is the driver); in a standard collection it's a standalone
 * *variant*.
 */
export function CreateTargetWizard({
  clip,
  batchClips,
  mode = 'target',
  drivers,
  datasetId,
  datasetType,
  onClose,
  onSubmitted,
  initialPreview = null,
  onPreviewChange,
  onAttachStillInput,
}: {
  clip: StudioClip
  /** When >1, applies the same recipe to every selected clip. */
  batchClips?: StudioClip[]
  /** `target` builds a paired IC-LoRA example; `variant` makes a standalone,
   *  ungrouped clip. In a standard collection everything is a variant. */
  mode?: 'target' | 'variant'
  /** Candidate driving videos (used when the source is a still). */
  drivers?: Array<{ path: string; label: string }>
  datasetId: string
  datasetType: LoraDatasetType
  onClose: () => void
  /** `requiresReview` is true when any queued job paused for edit review, so
   *  the caller can hand off to the review modal. */
  onSubmitted: (count: number, requiresReview: boolean) => void
  /** A previously committed frame edit for this clip (single-clip only), so the
   *  wizard restores it on reopen instead of forcing a regenerate. */
  initialPreview?: EditFramePreview | null
  /** Persist (or clear) the committed frame edit so it survives closing the
   *  wizard. Called whenever a preview is generated. Single-clip only. */
  onPreviewChange?: (preview: EditFramePreview | null) => void
  /** Attach a still as this example's input (a looped reference on export).
   *  Used by the "Generate input -> Still image -> Reference image" path, which
   *  needs no background job. Single-clip IC-LoRA only. */
  onAttachStillInput?: (
    sourceClipId: string,
    opts: { framePath: string; caption: string },
  ) => Promise<void> | void
}) {
  const { createDerivation, editFrame, extractFrame } = useLoraTraining()
  const isBatch = (batchClips?.length ?? 0) > 1
  const batchCount = batchClips?.length ?? 0
  const isStill = clip.kind === 'image'
  // Whether this run produces a *paired* example. Only IC-LoRA target mode
  // pairs; "variant" (and any standard-LoRA run) yields a standalone clip.
  const paired = datasetType === 'ic_lora' && mode === 'target'
  const duration = clip.probe?.durationSeconds ?? clip.durationSeconds ?? 0
  const sourcePoster = clip.posterPath ? pathToFileUrl(clip.posterPath) : null

  // Two steps for everyone: "Frame & edit" (stills hide the frame picker since
  // a still is already the anchor frame) then "Animate".
  const [stage, setStage] = useState<Stage>(1)
  // IC-LoRA examples can be built from either end: start from a reference and
  // generate the target (default), or start from a target and generate a
  // reference. Standard LoRA always produces a standalone variant.
  const [direction, setDirection] = useState<DerivationDirection>('target')
  const genReference = paired && direction === 'reference'
  // When generating the example's INPUT, the input can be a video (animated,
  // new) or a still image. A still input is a persistent reference, looped into
  // `reference_video` on export. (First-frame conditioning is a training-config
  // setting in the profile, not a per-example role, so it lives there instead.)
  const [inputMedium, setInputMedium] = useState<'video' | 'image'>('video')
  // Still-input applies when generating the input of a video-sourced example
  // (single or batch). Attaching a still reference needs no Animate step, so
  // the flow collapses to a single stage.
  const genStillInput = genReference && inputMedium === 'image' && !isStill
  const singleStage = genStillInput
  // Restore a previously committed frame edit (single-clip only) so reopening
  // the wizard shows the edited still instead of forcing a regenerate.
  const restored = !isBatch ? initialPreview : null
  const [frameTime, setFrameTime] = useState(restored?.frameTimeSeconds ?? 0)
  const [framePosition, setFramePosition] = useState(0)
  const [editPrompt, setEditPrompt] = useState(restored?.prompt ?? '')
  const [editEngine, setEditEngine] = useState<DerivationEditEngine>('klein')
  const [scenePrompt, setScenePrompt] = useState('')
  const [engine, setEngine] = useState<DerivationEngine>('ltx_local')
  const [conditioning, setConditioning] = useState<DerivationConditioning>('depth')
  const [conditioningStrength, setConditioningStrength] = useState(1.0)
  const [orientation, setOrientation] = useState<'video' | 'image'>('video')
  // Kling O3 v2v edit only: keep the source clip's original audio.
  const [keepAudio, setKeepAudio] = useState(true)
  const [driverPath, setDriverPath] = useState<string | null>(
    clip.driverPath ?? drivers?.[0]?.path ?? null,
  )
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Foreground Nano Banana preview (single-clip only). The committed PNG is fed
  // straight to the motion job so it never re-edits — no tokens wasted on a
  // frame the user hasn't seen. `previewPrompt` tracks which prompt produced it,
  // so editing the text afterwards marks the preview as stale.
  const [previewPath, setPreviewPath] = useState<string | null>(restored?.path ?? null)
  const [previewPrompt, setPreviewPrompt] = useState(restored?.prompt ?? '')
  const [previewing, setPreviewing] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const previewStale = previewPath !== null && editPrompt.trim() !== previewPrompt
  const previewCommitted = !isBatch && previewPath !== null && !previewStale

  // Generating the output (target) has no input medium; reset it so the toggle
  // doesn't carry a stale "still" choice back when the user flips direction.
  useEffect(() => {
    if (!genReference) setInputMedium('video')
  }, [genReference])

  // The reference-still flow is a single step; never strand the user on a
  // hidden Animate stage when they switch into it.
  useEffect(() => {
    if (singleStage) setStage(1)
  }, [singleStage])

  const runPreview = async () => {
    const prompt = editPrompt.trim()
    if (!prompt || previewing) return
    setPreviewing(true)
    setPreviewError(null)
    const res = await editFrame(clip.localPath, prompt, {
      timeSeconds: isStill ? 0 : frameTime,
      engine: editEngine,
    })
    setPreviewing(false)
    if (!res.ok) {
      setPreviewError(res.error)
      return
    }
    setPreviewPath(res.data)
    setPreviewPrompt(prompt)
    // Remember it so closing/reopening the wizard restores this exact edit.
    onPreviewChange?.({ path: res.data, prompt, frameTimeSeconds: isStill ? 0 : frameTime })
  }

  const driverOptions = useMemo(() => {
    const list = drivers ? [...drivers] : []
    if (driverPath && !list.some((d) => d.path === driverPath)) {
      list.unshift({ path: driverPath, label: 'Source clip' })
    }
    return list
  }, [drivers, driverPath])

  // A still needs a separate motion source; a video drives itself.
  const needsDriver = isStill
  const driverMissing = needsDriver && !driverPath

  const buildBody = (target: StudioClip): CreateDerivationBody => {
    const driver = isStill ? driverPath! : target.localPath
    const dur = target.probe?.durationSeconds ?? target.durationSeconds ?? 0
    const editText = editPrompt.trim()
    // The caption should describe the edit regardless of whether it ran in the
    // foreground (committed preview) or backgrounds during the job.
    const caption = editText || target.caption
    // When the user previewed and approved an edit (single clip, not stale),
    // hand the job that exact PNG and blank the prompt so it isn't re-edited.
    const useCommitted = previewCommitted && previewPath !== null && target.id === clip.id
    // Gate on review whenever an edit will actually run in the background and
    // wasn't already approved via the foreground preview — i.e. bulk edits and
    // un-previewed single edits pause for review before the motion drive.
    const requireReview = !useCommitted && editText.length > 0
    // Whether a Nano Banana edit is involved (committed preview or a prompt
    // that will run in the background). Kling O3 only uses the still as an
    // appearance reference when this is true.
    const frameEdited = useCommitted || editText.length > 0
    return {
      driverPath: driver,
      framePath: useCommitted ? previewPath : target.kind === 'image' ? target.localPath : null,
      // When generating the target, the driver is its reference. When
      // generating a reference, the new clip is a leaf — the source becomes the
      // target and is linked on fold (see onCreateReference).
      referencePath: paired && !genReference ? driver : null,
      direction: mode === 'variant' ? 'variant' : genReference ? 'reference' : 'target',
      datasetId,
      sourceClipId: target.id,
      frameTimeSeconds: isBatch ? framePosition * dur : frameTime,
      editPrompt: useCommitted ? '' : editText,
      editEngine,
      requireReview,
      scenePrompt: scenePrompt.trim(),
      engine,
      conditioningType: conditioning,
      conditioningStrength,
      characterOrientation: orientation,
      keepAudio,
      frameEdited,
      caption,
      label: caption || target.caption || 'Generated clip',
    }
  }

  const submit = async () => {
    const targets = isBatch ? batchClips ?? [] : [clip]
    if (targets.length === 0) return
    setBusy(true)
    setError(null)
    let failures = 0
    let requiresReview = false
    for (const target of targets) {
      const body = buildBody(target)
      if (body.requireReview) requiresReview = true
      const res = await createDerivation(body)
      if (!res.ok) failures++
    }
    setBusy(false)
    if (failures > 0) {
      setError(`${failures} of ${targets.length} could not be queued.`)
      return
    }
    onSubmitted(targets.length, requiresReview)
    onClose()
  }

  // Resolve the chosen still to a PNG on disk: a committed Nano-Banana edit when
  // present, else the raw frame at the picked time.
  const resolveStillPath = async (): Promise<string | null> => {
    if (previewCommitted && previewPath) return previewPath
    const ex = await extractFrame(clip.sourcePath, frameTime)
    if (!ex.ok) {
      setError(ex.error)
      return null
    }
    return ex.data
  }

  // Still-image input flow: attach the still as the example's input directly —
  // no background job; export loops it into `reference_video`. In batch, each
  // selected clip contributes its own starting frame (optionally edited with
  // the same instruction), grabbed at the chosen relative position.
  const submitStillInput = async () => {
    setBusy(true)
    setError(null)
    const editText = editPrompt.trim()

    if (isBatch) {
      const targets = batchClips ?? []
      let failures = 0
      for (const target of targets) {
        const dur = target.probe?.durationSeconds ?? target.durationSeconds ?? 0
        const at = framePosition * dur
        const made = editText
          ? await editFrame(target.sourcePath, editText, { timeSeconds: at })
          : await extractFrame(target.sourcePath, at)
        if (!made.ok) {
          failures++
          continue
        }
        const caption = editText || target.caption
        await onAttachStillInput?.(target.id, { framePath: made.data, caption })
      }
      setBusy(false)
      if (failures > 0) {
        setError(`${failures} of ${targets.length} could not be added.`)
        return
      }
      onSubmitted(targets.length, false)
      onClose()
      return
    }

    const stillPath = await resolveStillPath()
    if (!stillPath) {
      setBusy(false)
      return
    }
    const caption = editText || clip.caption
    await onAttachStillInput?.(clip.id, { framePath: stillPath, caption })
    setBusy(false)
    onSubmitted(1, false)
    onClose()
  }

  const handleSubmit = () => void (genStillInput ? submitStillInput() : submit())

  const goNext = () => setStage((s) => (s < 2 ? ((s + 1) as Stage) : s))
  const goBack = () => setStage((s) => (s > 1 ? ((s - 1) as Stage) : s))
  const appendSuggestion = (text: string) =>
    setEditPrompt((p) => (p.trim() ? `${p.trim()}, ${text.toLowerCase()}` : text))

  // The flow's goal is a training example; the input/output direction is chosen
  // on the toggle below, so the headline names the outcome, not the half.
  const noun = paired ? 'example' : 'variant'
  const title = isBatch ? `Generate ${batchCount} ${noun}s` : `Generate ${noun}`
  const subtitle = paired
    ? 'Teach your LoRA a change — by example'
    : 'Add an AI variation of this clip'
  // Keep the precise reference→target wording out of the headline; surface it
  // on hover for users who want the exact training semantics.
  const subtitleTip = !paired
    ? 'Generates a standalone variation of this clip — added to the dataset on its own, not grouped into an example.'
    : genReference
      ? 'Pairs the new clip (an input) with this clip (the output) as one training example.'
      : 'Pairs this clip (the input) with the new clip (the output) as one training example.'

  // Role of the existing source clip ("Before") and of the AI result ("After"),
  // which flip with the generation direction.
  const sourceRole = !paired ? 'Source clip' : genReference ? 'Output' : 'Input'
  const resultRole = !paired ? 'Variant' : genReference ? 'Input' : 'Output'
  const sourceRoleTip = !paired
    ? 'Source clip'
    : genReference
      ? 'Output — the clip your LoRA learns to produce'
      : 'Input — the clip your LoRA learns from'
  const resultRoleTip = !paired
    ? "Variant — a standalone new clip (not part of an example)"
    : genStillInput
      ? "Input — a still image kept as the conditioning reference for this clip (the output)"
      : genReference
        ? "Input — the new conditioning clip you're creating (paired with this clip as the output)"
        : "Output — the new clip you're creating (paired with this clip as the input)"

  // The frame the "Before" preview scrubs to: a fraction of the clip's length.
  const beforeFraction = isBatch ? framePosition : duration > 0 ? frameTime / duration : 0
  // Whether the inline frame picker is meaningful (videos with a known length).
  const showFramePicker = !isStill && (isBatch || duration > 0)

  // Attaching a still reference skips the Animate step.
  const railStages = singleStage ? [STAGES[0]] : STAGES
  const isLastStage = singleStage || stage === 2

  // Keep the banner reading Input -> Output left-to-right: the existing clip
  // sits on whichever side its role is. When generating the input, the existing
  // clip is the Output (right) and the new clip is the Input (left); generating
  // the output keeps the existing clip as the Input (left). Variants (standard
  // LoRA) have no input/output split, so the source stays on the left.
  const existingOnLeft = !paired || sourceRole === 'Input'

  // LTX local control is the primary motion engine; the Fal cloud engines
  // (Kling, Kling O3) sit alongside it in one MODEL dropdown.
  const useLtxEngine = engine === 'ltx_local'
  // Short label for the MODEL pill in the composer bar.
  const engineShortLabel = engine === 'kling' ? 'Kling' : engine === 'kling_o3' ? 'Kling O3' : 'LTX · Local'
  // Right-aligned hint that balances the toolbar and says where this model runs.
  const engineHint = useLtxEngine ? 'Runs on your GPU' : 'Runs on Fal · cloud'

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="relative flex flex-col w-full max-w-3xl mx-4 rounded-2xl border border-zinc-700/80 bg-zinc-900 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-start justify-between gap-3 px-5 pt-4 pb-3">
          <div className="flex items-start gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500/30 to-blue-500/20 text-blue-200 ring-1 ring-blue-500/30">
              <Sparkles className="h-4 w-4" />
            </span>
            <div>
              <h2 className="text-[15px] font-semibold text-white leading-tight">{title}</h2>
              <div className="flex items-center gap-1.5">
                <p className="text-[11px] text-zinc-500">{subtitle}</p>
                <InfoTip content={subtitleTip} side="bottom" label="What this does" />
              </div>
            </div>
          </div>
          <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Stepper — reads as a flow that ends in the new clip, so the arc is
         *  visible at a glance without extra prose. */}
        <div className="flex items-center gap-2 px-5 pb-2">
          {railStages.map((s) => {
            const isCurrent = stage === s.n
            const isDone = stage > s.n
            return (
              <div key={s.n} className="flex flex-1 items-center gap-2">
                <button
                  onClick={() => setStage(s.n)}
                  title={s.hint}
                  className="flex items-center gap-2 min-w-0"
                >
                  <span
                    className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold transition-colors ${
                      isCurrent
                        ? 'bg-blue-500 text-white'
                        : isDone
                          ? 'bg-blue-500/20 text-blue-300'
                          : 'bg-zinc-800 text-zinc-500'
                    }`}
                  >
                    {isDone ? <Check className="h-3.5 w-3.5" /> : s.n}
                  </span>
                  <span className="min-w-0 hidden sm:block">
                    <span className={`block text-[11px] font-medium leading-none ${isCurrent ? 'text-white' : isDone ? 'text-zinc-300' : 'text-zinc-500'}`}>
                      {s.label}
                    </span>
                  </span>
                </button>
                <InfoTip content={s.detail} side="bottom" label={`What “${s.label}” does`} />
                <span className={`h-px flex-1 ${stage > s.n ? 'bg-blue-500/40' : 'bg-zinc-800'}`} />
              </div>
            )
          })}
          {/* Result endpoint — the new clip these steps produce. */}
          <div
            className="flex items-center gap-1.5 shrink-0 text-blue-300"
            title={resultRoleTip}
          >
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-500/15 ring-1 ring-blue-500/30">
              <Sparkles className="h-3.5 w-3.5" />
            </span>
            <span className="text-[11px] font-medium hidden sm:block">{paired ? resultRole : 'New clip'}</span>
          </div>
        </div>

        {/* Body — a horizontal Input → Output banner (what you're making) with
         *  the current step's controls full-width beneath it. */}
        <div className="px-5 pb-1 min-h-[240px] max-h-[64vh] overflow-y-auto space-y-3.5">
          <div className="space-y-2.5">
            {/* Choices read left-to-right and reveal progressively: pick what to
             *  generate, then (for an input) its medium, then (for a still) its
             *  training role. Each option's explanation lives in a hover tooltip
             *  so the default surface stays clean. These are framing decisions, so
             *  they live on step 1 only — the Animate step stays focused on motion. */}
            {stage === 1 && paired && !isStill && (
              <div className="flex flex-wrap items-center gap-2.5">
                <SegToggle
                  value={direction}
                  onChange={setDirection}
                  options={[
                    {
                      value: 'target',
                      label: 'Generate output',
                      tip: 'Generate the output — the clip your LoRA learns to produce from this clip (the input).',
                    },
                    {
                      value: 'reference',
                      label: 'Generate input',
                      tip: 'Generate the input — the conditioning clip paired with this clip as the output.',
                    },
                  ]}
                />
                {genReference && (
                  <SegToggle
                    value={inputMedium}
                    onChange={setInputMedium}
                    options={[
                      { value: 'video', label: 'Video', tip: 'Generate a video clip as the input.' },
                      {
                        value: 'image',
                        label: 'Still image',
                        tip: 'Use a still image as the input — kept as the conditioning reference for the whole clip (looped into a video on export).',
                      },
                    ]}
                  />
                )}
              </div>
            )}

            {isBatch ? (
              <div className="flex items-center gap-3">
                <div className={`flex-1 min-w-0 space-y-1 ${existingOnLeft ? 'order-1' : 'order-3'}`}>
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-zinc-400 cursor-help" title={sourceRoleTip}>
                      {sourceRole} <span className="text-zinc-600">· these clips</span>
                    </span>
                    <span className="text-[10px] text-zinc-600">{batchCount} clips</span>
                  </div>
                  <div className="grid grid-cols-3 gap-1.5">
                    {(batchClips ?? []).map((c) => (
                      <div key={c.id} className="relative aspect-video rounded-md overflow-hidden bg-zinc-950 ring-1 ring-zinc-800">
                        <ScrubFrame
                          videoUrl={pathToFileUrl(c.localPath)}
                          posterUrl={c.posterPath ? pathToFileUrl(c.posterPath) : null}
                          fraction={framePosition}
                          showTimeBadge
                          className="w-full h-full object-cover"
                        />
                      </div>
                    ))}
                  </div>
                </div>
                <ArrowRight className="h-5 w-5 shrink-0 text-blue-400/80 order-2" />
                <div className={`flex-1 min-w-0 space-y-1 ${existingOnLeft ? 'order-3' : 'order-1'}`}>
                  <span className="text-[11px] text-zinc-400">{resultRole} <span className="text-zinc-600">· new</span></span>
                  <div className="relative flex h-32 w-full flex-col items-center justify-center gap-1.5 rounded-lg bg-zinc-950 ring-1 ring-zinc-800 text-zinc-600">
                    <Sparkles className="h-5 w-5 text-blue-400/70" />
                    <span className="text-[10px]">{batchCount} new clips</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-start gap-3">
                <figure className={`min-w-0 space-y-1.5 ${existingOnLeft ? 'order-1' : 'order-3'}`}>
                  <span className="flex h-4 items-center gap-1 text-[11px] text-zinc-400 cursor-help" title={sourceRoleTip}>
                    {sourceRole}
                    <span className="text-zinc-600">· this clip</span>
                  </span>
                  <div className="relative h-44 w-full rounded-lg overflow-hidden bg-zinc-950 ring-1 ring-zinc-800">
                    <ScrubFrame
                      videoUrl={pathToFileUrl(clip.localPath)}
                      posterUrl={sourcePoster}
                      fraction={beforeFraction}
                      className="w-full h-full object-cover"
                    />
                  </div>
                </figure>

                <div className="space-y-1.5 order-2">
                  <span className="block h-4" />
                  <div className="flex h-44 items-center justify-center">
                    <ArrowRight className="h-5 w-5 text-blue-400/80" />
                  </div>
                </div>

                <figure className={`min-w-0 space-y-1.5 ${existingOnLeft ? 'order-3' : 'order-1'}`}>
                  <span className="flex h-4 items-center gap-1 text-[11px] text-zinc-400 cursor-help" title={resultRoleTip}>
                    {resultRole}
                    <span className="text-zinc-600">· new</span>
                    {previewCommitted && <Check className="h-3 w-3 text-emerald-400" />}
                  </span>
                  <div className="relative h-44 w-full rounded-lg overflow-hidden bg-zinc-950 ring-1 ring-zinc-800 flex items-center justify-center">
                    {previewing ? (
                      <div className="flex flex-col items-center gap-1.5 text-zinc-500">
                        <Loader2 className="h-5 w-5 animate-spin text-blue-400" />
                        <span className="text-[10px]">Editing frame…</span>
                      </div>
                    ) : previewPath ? (
                      <>
                        <img src={pathToFileUrl(previewPath)} alt="result preview" className={`w-full h-full object-cover ${previewStale ? 'opacity-40' : ''}`} />
                        {previewStale && (
                          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                            <span className="text-[10px] text-amber-300 flex items-center gap-1 bg-black/60 px-2 py-1 rounded">
                              <AlertTriangle className="h-3 w-3" /> Outdated — regenerate
                            </span>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="flex flex-col items-center gap-1.5 text-zinc-600 px-4 text-center">
                        <Sparkles className="h-5 w-5 text-blue-400/70" />
                        <span className="text-[10px]">Your new clip</span>
                      </div>
                    )}
                  </div>
                </figure>
              </div>
            )}

            {/* Starting frame — scrubs the Input preview live. Presets sit under
             *  the track at the position they jump to (Start/Middle/End). */}
            {stage === 1 && showFramePicker && (
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <span
                    className="text-[11px] text-zinc-400 cursor-help"
                    title={
                      isBatch
                        ? 'Where in each clip to grab the starting frame. Relative, since clips differ in length — Middle is each clip’s midpoint.'
                        : 'The frame the model animates from, using the source motion. Start (the first frame) is the usual choice.'
                    }
                  >
                    Starting frame
                  </span>
                  {!isBatch && <span className="font-mono text-[10px] text-zinc-500">{frameTime.toFixed(1)}s</span>}
                </div>
                <input
                  type="range"
                  min={0}
                  max={isBatch ? 1 : duration}
                  step={isBatch ? 0.01 : 0.1}
                  value={isBatch ? framePosition : frameTime}
                  onChange={(e) => (isBatch ? setFramePosition(Number(e.target.value)) : setFrameTime(Number(e.target.value)))}
                  className="w-full accent-blue-500"
                />
                <div className="flex items-center justify-between">
                  {FRAME_PRESETS.map(([label, v]) => {
                    const active = isBatch
                      ? Math.abs(framePosition - v) < 0.005
                      : Math.abs(frameTime - v * duration) < 0.05
                    return (
                      <button
                        key={label}
                        onClick={() => (isBatch ? setFramePosition(v) : setFrameTime(v * duration))}
                        className={`rounded px-1.5 py-0.5 text-[10px] transition-colors ${
                          active ? 'font-medium text-blue-300' : 'text-zinc-500 hover:text-zinc-300'
                        }`}
                      >
                        {label}
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-zinc-800/70" />

          <div className="space-y-3">
          {/* Stage 1 — edit the (chosen) frame */}
          {stage === 1 && (
            <div className="space-y-3">
              <div className="space-y-2">
                {/* No "Edit instruction" label — the prompt box speaks for itself.
                 *  GenSpace-style card mirroring the Animate step: a borderless
                 *  textarea over a slim settings toolbar (image-model dropdown on
                 *  the left, where-it-runs hint on the right), then a row of quick
                 *  edits with the Preview action. The model defaults to local
                 *  FLUX.2 Klein; Nano Banana (Fal) is the cloud alternative. */}
                <div className="bg-zinc-950 border border-zinc-800 rounded-2xl overflow-visible focus-within:border-zinc-700 transition-colors">
                  <textarea
                    value={editPrompt}
                    onChange={(e) => setEditPrompt(e.target.value)}
                    autoFocus
                    placeholder="e.g. remove the person on the left, add falling snow…"
                    className="h-[72px] w-full resize-none overflow-y-auto bg-transparent px-3.5 py-3 text-sm leading-5 text-white placeholder:text-zinc-500 focus:outline-none"
                  />
                  <div className="flex items-center gap-0.5 border-t border-zinc-800/60 px-1.5 py-1.5 text-xs text-zinc-400">
                    <SettingsDropdown
                      title="IMAGE MODEL"
                      value={editEngine}
                      onChange={(v) => setEditEngine(v as DerivationEditEngine)}
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
                        editEngine === 'klein'
                          ? 'FLUX.2 [klein] 9B (local, needs the gated HuggingFace checkpoint downloaded)'
                          : 'Nano Banana (remote, needs a Fal API key)'
                      }
                      trigger={
                        <>
                          {editEngine === 'klein' ? (
                            <LightricksIcon className="h-4 w-4 text-zinc-200" />
                          ) : (
                            <Cloud className="h-4 w-4" />
                          )}
                          <span className="font-medium text-zinc-200">
                            {editEngine === 'klein' ? 'FLUX.2 Klein' : 'Nano Banana'}
                          </span>
                          <ChevronUp className="h-3 w-3 text-zinc-500" />
                        </>
                      }
                    />
                    <span className="ml-auto whitespace-nowrap pr-1.5 text-[11px] text-zinc-500">
                      {editEngine === 'klein' ? 'Runs on your GPU' : 'Runs on Fal · cloud'}
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
                        onClick={() => void runPreview()}
                        disabled={!editPrompt.trim() || previewing}
                        className="shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border border-blue-500/40 bg-blue-500/10 text-blue-200 hover:bg-blue-500/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      >
                        {previewing ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : previewPath ? (
                          <RefreshCw className="h-3.5 w-3.5" />
                        ) : (
                          <Wand2 className="h-3.5 w-3.5" />
                        )}
                        {previewPath ? (previewStale ? 'Update' : 'Regenerate') : 'Preview edit'}
                      </button>
                    )}
                  </div>
                </div>
                {editEngine === 'klein' && (
                  <span className="text-[11px] text-zinc-500">
                    Local GPU edit — needs the Klein checkpoint (Model Status). Runs on the GPU, so it can’t run
                    while another generation is in flight.
                  </span>
                )}
              </div>

              {previewError && (
                <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-md px-3 py-2">{previewError}</div>
              )}

              {isBatch && (
                <div className="flex items-start gap-2 rounded-lg bg-blue-500/5 border border-blue-500/20 px-3 py-2 text-[11px] text-zinc-400">
                  <Wand2 className="h-3.5 w-3.5 text-blue-400 shrink-0 mt-0.5" />
                  <span>
                    {genStillInput
                      ? "Each clip's starting frame becomes that clip's input."
                      : editPrompt.trim()
                        ? <>Edited frames <span className="text-zinc-200">pause for your review</span> before each video generates.</>
                        : 'Leave blank to re-drive motion only.'}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Stage 2 — animate */}
          {stage === 2 && (
            <div className="space-y-3">
              {/* GenSpace-style composer: the scene prompt on top, then one
               *  settings bar holding the model and only the controls that apply
               *  (output length/size are inherited from the source clip). Menus
               *  open upward to sit above the bar. */}
              <div className="overflow-visible rounded-2xl border border-zinc-800 bg-zinc-950 transition-colors focus-within:border-zinc-700">
                <textarea
                  value={scenePrompt}
                  onChange={(e) => setScenePrompt(e.target.value)}
                  placeholder="Describe the output clip… (optional)"
                  className="h-[72px] w-full resize-none overflow-y-auto bg-transparent px-3.5 py-3 text-sm leading-5 text-white placeholder:text-zinc-500 focus:outline-none"
                />
                <div className="flex items-center gap-0.5 border-t border-zinc-800/60 px-1.5 py-1.5 text-xs text-zinc-400">
                  {needsDriver && (
                    <>
                      <SettingsDropdown
                        title="DRIVER"
                        value={driverPath ?? ''}
                        onChange={(v) => setDriverPath(v || null)}
                        options={driverOptions.map((d) => ({ value: d.path, label: d.label }))}
                        trigger={
                          <>
                            <Film className="h-3.5 w-3.5" />
                            <span className="max-w-[120px] truncate font-medium text-zinc-300">
                              {driverOptions.find((d) => d.path === driverPath)?.label ?? 'Select driver…'}
                            </span>
                            <ChevronUp className="h-3 w-3 text-zinc-500" />
                          </>
                        }
                      />
                      <span className="mx-0.5 h-4 w-px bg-zinc-700" />
                    </>
                  )}

                  <SettingsDropdown
                    title="MODEL"
                    value={engine}
                    onChange={(v) => setEngine(v as DerivationEngine)}
                    options={[
                      { value: 'ltx_local', label: 'LTX', description: 'Local', icon: <LightricksIcon className="h-4 w-4" /> },
                      { value: 'kling', label: 'Kling', description: 'Character motion', icon: <Cloud className="h-4 w-4" /> },
                      { value: 'kling_o3', label: 'Kling O3', description: 'Prompt-driven edit', icon: <Cloud className="h-4 w-4" /> },
                    ]}
                    trigger={
                      <>
                        {useLtxEngine ? <LightricksIcon className="h-4 w-4 text-zinc-200" /> : <Cloud className="h-4 w-4" />}
                        <span className="font-medium text-zinc-200">{engineShortLabel}</span>
                        <ChevronUp className="h-3 w-3 text-zinc-500" />
                      </>
                    }
                  />

                  {useLtxEngine && (
                    <>
                      <span className="mx-0.5 h-4 w-px bg-zinc-700" />
                      <SettingsDropdown
                        title="MOTION CONTROL"
                        value={conditioning}
                        onChange={(v) => setConditioning(v as DerivationConditioning)}
                        options={[
                          { value: 'depth', label: 'Depth — volumetric 3D form' },
                          { value: 'canny', label: 'Canny — edge outlines' },
                          { value: 'pose', label: 'Pose — body keypoints' },
                        ]}
                        trigger={
                          <>
                            <Layers className="h-3.5 w-3.5" />
                            <span className="text-zinc-300">{conditioning === 'canny' ? 'Canny' : conditioning === 'pose' ? 'Pose' : 'Depth'}</span>
                            <ChevronUp className="h-3 w-3 text-zinc-500" />
                          </>
                        }
                      />
                      <SettingsDropdown
                        title="STRENGTH"
                        value={String(conditioningStrength)}
                        onChange={(v) => setConditioningStrength(Number(v))}
                        options={STRENGTH_OPTIONS}
                        trigger={
                          <>
                            <Gauge className="h-3.5 w-3.5" />
                            <span className="text-zinc-300">{Math.round(conditioningStrength * 100)}%</span>
                            <ChevronUp className="h-3 w-3 text-zinc-500" />
                          </>
                        }
                      />
                    </>
                  )}

                  {engine === 'kling' && (
                    <>
                      <span className="mx-0.5 h-4 w-px bg-zinc-700" />
                      <SettingsDropdown
                        title="ORIENTATION"
                        value={orientation}
                        onChange={(v) => setOrientation(v as 'video' | 'image')}
                        options={[
                          { value: 'video', label: 'Video — full body + camera motion' },
                          { value: 'image', label: 'Image — preserve source framing' },
                        ]}
                        trigger={
                          <>
                            {orientation === 'image' ? <ImageIcon className="h-3.5 w-3.5" /> : <Film className="h-3.5 w-3.5" />}
                            <span className="text-zinc-300">{orientation === 'image' ? 'Image' : 'Video'}</span>
                            <ChevronUp className="h-3 w-3 text-zinc-500" />
                          </>
                        }
                      />
                    </>
                  )}

                  {engine === 'kling_o3' && (
                    <>
                      <span className="mx-0.5 h-4 w-px bg-zinc-700" />
                      <SettingsDropdown
                        title="AUDIO"
                        value={keepAudio ? 'keep' : 'drop'}
                        onChange={(v) => setKeepAudio(v === 'keep')}
                        options={[
                          { value: 'keep', label: 'Keep original audio' },
                          { value: 'drop', label: 'Drop audio' },
                        ]}
                        trigger={
                          <>
                            {keepAudio ? <Volume2 className="h-3.5 w-3.5" /> : <VolumeX className="h-3.5 w-3.5" />}
                            <span className="text-zinc-300">{keepAudio ? 'Audio on' : 'Audio off'}</span>
                            <ChevronUp className="h-3 w-3 text-zinc-500" />
                          </>
                        }
                      />
                    </>
                  )}

                  <span className="ml-auto whitespace-nowrap pr-1.5 text-[11px] text-zinc-500">{engineHint}</span>
                </div>
              </div>
            </div>
          )}

          {error && <div className="mt-3 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-md px-3 py-2">{error}</div>}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-zinc-800 px-5 py-3">
          <button
            onClick={stage > 1 ? goBack : onClose}
            className="text-sm px-3.5 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600 flex items-center gap-1.5 transition-colors"
          >
            {stage > 1 ? <><ArrowLeft className="h-4 w-4" /> Back</> : 'Cancel'}
          </button>
          <div className="flex items-center gap-3">
            <span className="text-[11px] text-zinc-600">Step {railStages.findIndex((s) => s.n === stage) + 1} of {railStages.length}</span>
            {!isLastStage ? (
              <button
                onClick={goNext}
                className="text-sm font-medium px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white flex items-center gap-2 transition-colors"
              >
                Next <ArrowRight className="h-4 w-4" />
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={busy || driverMissing}
                title={
                  genStillInput
                    ? "Adds the still as this example's input."
                    : 'Runs in the background — you can keep working.'
                }
                className="text-sm font-medium px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                {genStillInput
                  ? isBatch
                    ? `Add ${batchCount} inputs`
                    : 'Add input'
                  : isBatch
                    ? `Queue ${batchCount}`
                    : 'Queue generation'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
