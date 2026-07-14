import { describe, expect, it, vi } from 'vitest'
import type {
  ApplyEditsResult,
  ClipEdits,
  ClipInput,
  ClipProbe,
} from '../contexts/LoraTrainingContext'
import {
  computeImportEdits,
  DEFAULT_IMPORT_NORMALIZE,
  importSpecActive,
  normalizeImportInputs,
  type ImportNormalizeSpec,
} from './lora-import-normalize'

function probe(overrides: Partial<ClipProbe> = {}): ClipProbe {
  return {
    durationSeconds: 20,
    width: 1920,
    height: 1080,
    fps: 30,
    frameCount: 600,
    hasAudio: true,
    ...overrides,
  }
}

function clip(localPath: string, overrides: Partial<ClipInput> = {}): ClipInput {
  return {
    localPath,
    caption: '',
    origin: 'imported',
    probe: probe(),
    ...overrides,
  }
}

function spec(overrides: Partial<ImportNormalizeSpec> = {}): ImportNormalizeSpec {
  return {
    trim: { enabled: true, maxSeconds: 10 },
    resolution: { enabled: true, shortSide: 720 },
    fps: { enabled: true, value: 24 },
    ...overrides,
  }
}

describe('LoRA import normalization edits', () => {
  it('is inactive by default and does nothing without a usable probe', () => {
    expect(importSpecActive(DEFAULT_IMPORT_NORMALIZE)).toBe(false)
    expect(computeImportEdits(null, spec())).toBeNull()
    expect(computeImportEdits(probe(), DEFAULT_IMPORT_NORMALIZE)).toBeNull()
  })

  it('honors trim and fps tolerances and never upscales', () => {
    const nearlyCompliant = probe({
      durationSeconds: 10.05,
      width: 640,
      height: 360,
      fps: 24.009,
    })

    expect(computeImportEdits(nearlyCompliant, spec())).toBeNull()
    expect(
      computeImportEdits(
        { ...nearlyCompliant, durationSeconds: 10.051, fps: 24.02 },
        spec(),
      ),
    ).toMatchObject({
      trim: { startSeconds: 0, endSeconds: 10 },
      scale: null,
      fps: 24,
    })
  })

  it('downscales proportionally to even dimensions and builds a complete edit stack', () => {
    expect(computeImportEdits(probe({ width: 1921, height: 1081 }), spec())).toEqual({
      trim: { startSeconds: 0, endSeconds: 10 },
      crop: null,
      scale: { width: 1280, height: 720 },
      fps: 24,
      speed: null,
      mute: false,
      reverse: false,
    })
  })
})

describe('normalizing LoRA import inputs', () => {
  it('passes through inactive recipes without mutating the input array', async () => {
    const inputs = [clip('C:/clips/already.mp4')]
    const applyClipEdits = vi.fn()

    const result = await normalizeImportInputs(
      inputs,
      DEFAULT_IMPORT_NORMALIZE,
      applyClipEdits,
    )

    expect(result).toEqual({ inputs, failures: [] })
    expect(result.inputs).not.toBe(inputs)
    expect(applyClipEdits).not.toHaveBeenCalled()
  })

  it('skips images, missing probes, and already-compliant clips', async () => {
    const inputs = [
      clip('C:/clips/still.PNG'),
      clip('C:/clips/unprobed.mp4', { probe: null }),
      clip('C:/clips/ready.mp4', {
        probe: probe({
          durationSeconds: 10,
          width: 1280,
          height: 720,
          fps: 24,
        }),
      }),
    ]
    const applyClipEdits = vi.fn()

    const result = await normalizeImportInputs(inputs, spec(), applyClipEdits)

    expect(result).toEqual({ inputs, failures: [] })
    expect(applyClipEdits).not.toHaveBeenCalled()
  })

  it('preserves the original source and is idempotent after a successful render', async () => {
    const source = 'C:/clips/original.mp4'
    const normalizedProbe = probe({
      durationSeconds: 10,
      width: 1280,
      height: 720,
      fps: 24,
      frameCount: 240,
    })
    const applyClipEdits = vi.fn(
      async (
        _sourcePath: string,
        _edits: ClipEdits,
      ): Promise<{ ok: true; data: ApplyEditsResult }> => ({
        ok: true,
        data: {
          derivedPath: 'C:/clips/derived.mp4',
          probe: normalizedProbe,
        },
      }),
    )
    const progress: Array<{ done: number; total: number }> = []

    const first = await normalizeImportInputs(
      [clip('C:/clips/previous-derived.mp4', { sourcePath: source })],
      spec(),
      applyClipEdits,
      { onProgress: (value) => progress.push(value) },
    )

    expect(applyClipEdits).toHaveBeenCalledOnce()
    expect(applyClipEdits.mock.calls[0][0]).toBe(source)
    expect(first.inputs[0]).toMatchObject({
      localPath: 'C:/clips/derived.mp4',
      sourcePath: source,
      durationSeconds: 10,
      probe: normalizedProbe,
    })
    expect(progress).toEqual([
      { done: 0, total: 1 },
      { done: 1, total: 1 },
    ])

    const secondApply = vi.fn()
    const second = await normalizeImportInputs(first.inputs, spec(), secondApply)

    expect(second).toEqual(first)
    expect(secondApply).not.toHaveBeenCalled()
  })

  it('reports failed filenames, leaves failed clips unchanged, and completes progress', async () => {
    const input = clip('C:\\clips\\broken.mp4')
    const progress: Array<{ done: number; total: number }> = []

    const result = await normalizeImportInputs(
      [input],
      spec(),
      async () => ({ ok: false, error: 'ffmpeg failed' }),
      { onProgress: (value) => progress.push(value) },
    )

    expect(result.inputs[0]).toBe(input)
    expect(result.failures).toEqual(['broken.mp4: ffmpeg failed'])
    expect(progress).toEqual([
      { done: 0, total: 1 },
      { done: 1, total: 1 },
    ])
  })

  it('respects the requested concurrency limit', async () => {
    let active = 0
    let maxActive = 0
    const applyClipEdits = async (
      sourcePath: string,
    ): Promise<{ ok: true; data: ApplyEditsResult }> => {
      active += 1
      maxActive = Math.max(maxActive, active)
      await new Promise((resolve) => setTimeout(resolve, 0))
      active -= 1
      return {
        ok: true,
        data: {
          derivedPath: `${sourcePath}.normalized.mp4`,
          probe: probe({
            durationSeconds: 10,
            width: 1280,
            height: 720,
            fps: 24,
          }),
        },
      }
    }

    await normalizeImportInputs(
      Array.from({ length: 5 }, (_, index) => clip(`C:/clips/${index}.mp4`)),
      spec(),
      applyClipEdits,
      { concurrency: 2 },
    )

    expect(maxActive).toBe(2)
  })
})
