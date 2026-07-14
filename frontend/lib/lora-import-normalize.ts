import type { ApplyEditsResult, ClipEdits, ClipInput, ClipProbe } from '../contexts/LoraTrainingContext'
import { isImagePath } from './file-url'

/**
 * Optional "normalize on import" recipe. Lets the user clean up clips the
 * moment they enter a collection (from local files or Pexels) so the whole
 * dataset shares one fps/resolution and a sane max length — which means far
 * fewer pairs get dropped at IC-LoRA export. All three knobs are off by
 * default (import behaves exactly as before unless the user opts in).
 */
export interface ImportNormalizeSpec {
  /** Cap clip length: trim to the first `maxSeconds` (only if longer). */
  trim: { enabled: boolean; maxSeconds: number }
  /** Resize so the shorter side is `shortSide` px — aspect preserved, never upscaled. */
  resolution: { enabled: boolean; shortSide: number }
  /** Force a single frame rate across the dataset. */
  fps: { enabled: boolean; value: number }
}

export const DEFAULT_IMPORT_NORMALIZE: ImportNormalizeSpec = {
  trim: { enabled: false, maxSeconds: 10 },
  resolution: { enabled: false, shortSide: 720 },
  fps: { enabled: false, value: 25 },
}

export const SHORT_SIDE_CHOICES = [1080, 720, 576, 512] as const
export const FPS_CHOICES = [24, 25, 30] as const

export function importSpecActive(spec: ImportNormalizeSpec): boolean {
  return spec.trim.enabled || spec.resolution.enabled || spec.fps.enabled
}

const evenDim = (n: number): number => Math.max(2, Math.round(n / 2) * 2)

/**
 * Build the non-destructive edit stack for one clip from `spec` + its probe,
 * or `null` when nothing is needed (no probe, or the clip already complies —
 * this is what makes the helper idempotent and safe to run more than once).
 */
export function computeImportEdits(
  probe: ClipProbe | null | undefined,
  spec: ImportNormalizeSpec,
): ClipEdits | null {
  if (!probe) return null
  let changed = false
  const edits: ClipEdits = {
    trim: null,
    crop: null,
    scale: null,
    fps: null,
    speed: null,
    mute: false,
    reverse: false,
  }

  if (
    spec.trim.enabled &&
    spec.trim.maxSeconds > 0 &&
    probe.durationSeconds > spec.trim.maxSeconds + 0.05
  ) {
    edits.trim = { startSeconds: 0, endSeconds: spec.trim.maxSeconds }
    changed = true
  }

  if (spec.resolution.enabled && spec.resolution.shortSide > 0) {
    const currentShort = Math.min(probe.width, probe.height)
    // Only ever downscale — upscaling invents detail and bloats files.
    if (currentShort > spec.resolution.shortSide) {
      const ratio = spec.resolution.shortSide / currentShort
      edits.scale = { width: evenDim(probe.width * ratio), height: evenDim(probe.height * ratio) }
      changed = true
    }
  }

  if (spec.fps.enabled && spec.fps.value > 0 && Math.abs((probe.fps || 0) - spec.fps.value) > 0.01) {
    edits.fps = spec.fps.value
    changed = true
  }

  return changed ? edits : null
}

export interface NormalizeProgress {
  done: number
  total: number
}

type ApplyEdits = (
  sourcePath: string,
  edits: ClipEdits,
) => Promise<{ ok: true; data: ApplyEditsResult } | { ok: false; error: string }>

const labelOf = (path: string): string => path.split(/[\\/]/).pop() || path

/**
 * Render every clip that needs normalization through the existing per-clip
 * edit pipeline, with bounded concurrency, and return the updated inputs
 * (derived path + fresh probe + the applied edits; `sourcePath` is preserved
 * so the original is recoverable). Clips that don't need work — images, clips
 * without a probe, or already-compliant clips — pass through untouched.
 */
export async function normalizeImportInputs(
  inputs: ClipInput[],
  spec: ImportNormalizeSpec,
  applyClipEdits: ApplyEdits,
  opts: { concurrency?: number; onProgress?: (p: NormalizeProgress) => void } = {},
): Promise<{ inputs: ClipInput[]; failures: string[] }> {
  const result = inputs.slice()
  const failures: string[] = []
  if (!importSpecActive(spec)) return { inputs: result, failures }

  const work = result.flatMap((clip, index) =>
    !isImagePath(clip.localPath) && computeImportEdits(clip.probe, spec) ? [index] : [],
  )
  const total = work.length
  if (total === 0) return { inputs: result, failures }

  let done = 0
  opts.onProgress?.({ done, total })
  let cursor = 0
  const worker = async () => {
    while (cursor < work.length) {
      const index = work[cursor++]
      const clip = result[index]
      const edits = computeImportEdits(clip.probe, spec)
      if (!edits) continue
      const source = clip.sourcePath ?? clip.localPath
      const res = await applyClipEdits(source, edits)
      if (res.ok) {
        result[index] = {
          ...clip,
          localPath: res.data.derivedPath,
          probe: res.data.probe,
          durationSeconds: res.data.probe.durationSeconds,
          edits,
          sourcePath: source,
        }
      } else {
        failures.push(`${labelOf(clip.localPath)}: ${res.error}`)
      }
      done += 1
      opts.onProgress?.({ done, total })
    }
  }
  const poolSize = Math.min(opts.concurrency ?? 3, total)
  await Promise.all(Array.from({ length: poolSize }, () => worker()))
  return { inputs: result, failures }
}
