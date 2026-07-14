// Pure, presentation-side dataset-quality heuristics for the LoRA trainer.
//
// These never gate training — they advise. Keeping the thresholds and
// scoring here (rather than the backend) lets us tune the guidance without
// an API round-trip, and keeps the rules unit-testable in isolation.

export interface ClipProbeLike {
  durationSeconds: number
  width: number
  height: number
  fps: number
  frameCount: number
  hasAudio: boolean
  videoCodec?: string | null
}

export interface ClipLike {
  caption?: string
  probe?: ClipProbeLike | null
}

export type WarningLevel = 'warn' | 'error'

export interface QualityWarning {
  level: WarningLevel
  text: string
}

// --- Thresholds (tuned for LTX-2 LoRA training) -------------------------

export const RECOMMENDED_MIN_CLIPS = 10
export const HARD_MIN_CLIPS = 3
// Very short clips carry little motion; very long ones waste preprocessing
// time and usually contain multiple scenes that should be split.
export const CLIP_MIN_SECONDS = 1
export const CLIP_MAX_SECONDS = 30
// Below this on the short edge, detail is too low for a faithful LoRA.
export const RECOMMENDED_MIN_DIM = 512
export const HARD_MIN_DIM = 256
// Far above the training resolution: such source is downscaled (on the desktop
// before upload, and again by the trainer to the bucket), so the extra detail
// isn't used. Purely informational — not a problem to fix.
export const HIGH_RES_SHORT_EDGE = 1440
// Short-side cap the desktop applies to oversized standard-dataset clips before
// upload (mirrors STANDARD_MAX_SHORT_SIDE in the backend runner).
export const STANDARD_TRAIN_SHORT_EDGE = 768

export function aspectRatioKey(width: number, height: number): string {
  if (width <= 0 || height <= 0) return 'unknown'
  const ratio = width / height
  // Bucket to the nearest common ratio so tiny encode differences don't
  // read as "inconsistent".
  const candidates: Array<[string, number]> = [
    ['16:9', 16 / 9],
    ['9:16', 9 / 16],
    ['1:1', 1],
    ['4:3', 4 / 3],
    ['3:4', 3 / 4],
    ['21:9', 21 / 9],
  ]
  let best = candidates[0]
  for (const c of candidates) {
    if (Math.abs(c[1] - ratio) < Math.abs(best[1] - ratio)) best = c
  }
  return Math.abs(best[1] - ratio) / best[1] < 0.08 ? best[0] : `${ratio.toFixed(2)}:1`
}

export interface CropRect {
  x: number
  y: number
  width: number
  height: number
}

// Snap down to the nearest multiple of 32 (LTX VAE constraint), min 32.
export function snapDown32(value: number): number {
  return Math.max(32, Math.floor(value / 32) * 32)
}

// Largest centered crop of `sourceW×sourceH` matching aspect `aw:ah`, with
// both dimensions snapped to multiples of 32 (so the result is bucket-legal).
export function centeredCrop(sourceW: number, sourceH: number, aw: number, ah: number): CropRect {
  const targetRatio = aw / ah
  const sourceRatio = sourceW / sourceH
  let cw: number
  let ch: number
  if (sourceRatio > targetRatio) {
    ch = sourceH
    cw = Math.round(sourceH * targetRatio)
  } else {
    cw = sourceW
    ch = Math.round(sourceW / targetRatio)
  }
  cw = Math.min(snapDown32(cw), snapDown32(sourceW))
  ch = Math.min(snapDown32(ch), snapDown32(sourceH))
  const x = Math.max(0, Math.floor((sourceW - cw) / 2))
  const y = Math.max(0, Math.floor((sourceH - ch) / 2))
  return { x, y, width: cw, height: ch }
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}m ${s}s`
}

// Compact badges shown on a clip card.
export function probeBadges(probe: ClipProbeLike): string[] {
  const badges = [
    formatDuration(probe.durationSeconds),
    `${probe.width}×${probe.height}`,
  ]
  if (probe.fps > 0) badges.push(`${Math.round(probe.fps)}fps`)
  badges.push(probe.hasAudio ? 'audio' : 'no audio')
  return badges
}

// Per-clip quality warnings. `requireAudio` flips "no audio" from neutral
// into an error (audio-video training needs an audio track on every clip).
export function clipWarnings(
  clip: ClipLike,
  opts: { requireAudio?: boolean } = {},
): QualityWarning[] {
  const warnings: QualityWarning[] = []
  const probe = clip.probe
  if (!probe) {
    return warnings
  }
  if (probe.durationSeconds > 0 && probe.durationSeconds < CLIP_MIN_SECONDS) {
    warnings.push({ level: 'warn', text: `Very short (${formatDuration(probe.durationSeconds)})` })
  }
  if (probe.durationSeconds > CLIP_MAX_SECONDS) {
    warnings.push({ level: 'warn', text: `Long clip (${formatDuration(probe.durationSeconds)}) — consider splitting` })
  }
  const shortEdge = Math.min(probe.width, probe.height)
  if (shortEdge > 0 && shortEdge < HARD_MIN_DIM) {
    warnings.push({ level: 'error', text: `Low resolution (${probe.width}×${probe.height})` })
  } else if (shortEdge > 0 && shortEdge < RECOMMENDED_MIN_DIM) {
    warnings.push({ level: 'warn', text: `Below ${RECOMMENDED_MIN_DIM}px short edge` })
  }
  if (opts.requireAudio && !probe.hasAudio) {
    warnings.push({ level: 'error', text: 'No audio track (required for audio training)' })
  }
  return warnings
}

export interface DatasetHealth {
  clipCount: number
  captionedCount: number
  probedCount: number
  totalDurationSeconds: number
  aspectRatios: string[]
  aspectConsistent: boolean
  minShortEdge: number | null
  maxShortEdge: number | null
  warningCount: number
  errorCount: number
  // 0–100 readiness score for the meter.
  score: number
}

export function datasetHealth(
  clips: ClipLike[],
  opts: { requireAudio?: boolean } = {},
): DatasetHealth {
  const clipCount = clips.length
  let captionedCount = 0
  let probedCount = 0
  let totalDurationSeconds = 0
  let minShortEdge: number | null = null
  let maxShortEdge: number | null = null
  let warningCount = 0
  let errorCount = 0
  const aspectSet = new Set<string>()

  for (const clip of clips) {
    if ((clip.caption ?? '').trim()) captionedCount += 1
    const probe = clip.probe
    if (probe) {
      probedCount += 1
      totalDurationSeconds += Math.max(0, probe.durationSeconds)
      const shortEdge = Math.min(probe.width, probe.height)
      if (shortEdge > 0) {
        minShortEdge = minShortEdge == null ? shortEdge : Math.min(minShortEdge, shortEdge)
        maxShortEdge = maxShortEdge == null ? shortEdge : Math.max(maxShortEdge, shortEdge)
      }
      aspectSet.add(aspectRatioKey(probe.width, probe.height))
    }
    for (const w of clipWarnings(clip, opts)) {
      if (w.level === 'error') errorCount += 1
      else warningCount += 1
    }
  }

  const aspectConsistent = aspectSet.size <= 1

  // Score: weighted blend of the things that most affect LoRA quality.
  let score = 0
  if (clipCount > 0) {
    const countScore = Math.min(1, clipCount / RECOMMENDED_MIN_CLIPS) * 40
    const captionScore = (captionedCount / clipCount) * 30
    const resScore = minShortEdge == null ? 0 : (minShortEdge >= RECOMMENDED_MIN_DIM ? 15 : minShortEdge >= HARD_MIN_DIM ? 8 : 0)
    const aspectScore = aspectConsistent ? 15 : 5
    const penalty = errorCount * 6
    score = Math.max(0, Math.min(100, Math.round(countScore + captionScore + resScore + aspectScore - penalty)))
  }

  return {
    clipCount,
    captionedCount,
    probedCount,
    totalDurationSeconds,
    aspectRatios: Array.from(aspectSet),
    aspectConsistent,
    minShortEdge,
    maxShortEdge,
    warningCount,
    errorCount,
    score,
  }
}

export interface PreflightCheck {
  ok: boolean
  // A "blocker" failing should stop upload; a soft check only advises.
  blocker: boolean
  label: string
  detail?: string
}

// Pre-upload checklist. Only items the LTX-2 trainer actually enforces (or
// that make a run pointless) are blockers; the rest advise but don't block.
// Verified against the LTX-2 trainer (`Lightricks/LTX-2`):
//   - `PrecomputedDataset._validate_setup` requires ≥1 valid sample (matching
//     latents + conditions [+ audio_latents]) — no minimum clip count, and no
//     aspect-ratio consistency check (mixed ratios are cropped/bucketed to the
//     resolution bucket).
//   - `process_dataset.py` requires a caption *column* to exist, but
//     `process_captions.py` happily encodes empty per-row captions — and the
//     one-click pipeline auto-captions before preprocess anyway, so per-clip
//     captioning is not a pre-flight requirement.
// So "Every clip captioned" and "Consistent aspect ratio" are intentionally
// NOT here: they're not required to train and only caused confusion (a 0/90
// caption count reads as a blocker when auto-captioning will fill it in).
export function preflightChecks(
  clips: ClipLike[],
  opts: { triggerWord?: string | null; requireAudio?: boolean } = {},
): PreflightCheck[] {
  const health = datasetHealth(clips, opts)
  const checks: PreflightCheck[] = []

  checks.push({
    ok: health.clipCount >= HARD_MIN_CLIPS,
    blocker: true,
    label: `At least ${HARD_MIN_CLIPS} clips`,
    detail: `${health.clipCount} added`,
  })
  checks.push({
    ok: health.clipCount >= RECOMMENDED_MIN_CLIPS,
    blocker: false,
    label: `${RECOMMENDED_MIN_CLIPS}+ clips recommended`,
    detail: `${health.clipCount} added`,
  })
  checks.push({
    ok: health.errorCount === 0,
    blocker: false,
    label: 'No quality errors',
    detail: health.errorCount > 0 ? `${health.errorCount} clip(s) flagged` : 'all good',
  })
  // Informational only: very high-res source (4K/8K) is downscaled to the
  // training resolution before upload, so the extra detail isn't used. Shown as
  // satisfied (ok) — it's expected/handled, not something to fix. Omitted
  // entirely for normal-res datasets to avoid noise.
  if (health.maxShortEdge != null && health.maxShortEdge >= HIGH_RES_SHORT_EDGE) {
    checks.push({
      ok: true,
      blocker: false,
      label: 'High-res source — downscaled for training',
      detail: `${health.maxShortEdge}px → ≤${STANDARD_TRAIN_SHORT_EDGE}px`,
    })
  }
  return checks
}
