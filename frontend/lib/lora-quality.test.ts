import { describe, expect, it } from 'vitest'
import {
  aspectRatioKey,
  centeredCrop,
  clipWarnings,
  datasetHealth,
  preflightChecks,
  probeBadges,
  snapDown32,
  type ClipProbeLike,
} from './lora-quality'

function probe(overrides: Partial<ClipProbeLike> = {}): ClipProbeLike {
  return {
    durationSeconds: 5,
    width: 1280,
    height: 720,
    fps: 24,
    frameCount: 120,
    hasAudio: true,
    ...overrides,
  }
}

describe('LoRA quality geometry helpers', () => {
  it('buckets near-common ratios and preserves unusual or invalid ratios', () => {
    expect(aspectRatioKey(1920, 1080)).toBe('16:9')
    expect(aspectRatioKey(1000, 1050)).toBe('1:1')
    expect(aspectRatioKey(1600, 1000)).toBe('1.60:1')
    expect(aspectRatioKey(0, 1080)).toBe('unknown')
  })

  it('creates a centered, VAE-aligned crop without exceeding the source', () => {
    expect(centeredCrop(1920, 1080, 1, 1)).toEqual({
      x: 432,
      y: 12,
      width: 1056,
      height: 1056,
    })
    expect(snapDown32(63)).toBe(32)
    expect(snapDown32(64)).toBe(64)
  })
})

describe('LoRA clip quality warnings', () => {
  it('treats exact duration and resolution thresholds as acceptable', () => {
    expect(
      clipWarnings({
        probe: probe({
          durationSeconds: 1,
          width: 512,
          height: 512,
        }),
      }),
    ).toEqual([])
    expect(clipWarnings({ probe: probe({ durationSeconds: 30 }) })).toEqual([])
  })

  it('distinguishes hard errors from advisory warnings and audio requirements', () => {
    expect(
      clipWarnings(
        {
          probe: probe({
            durationSeconds: 0.5,
            width: 255,
            height: 720,
            hasAudio: false,
          }),
        },
        { requireAudio: true },
      ),
    ).toEqual([
      { level: 'warn', text: 'Very short (0.5s)' },
      { level: 'error', text: 'Low resolution (255×720)' },
      { level: 'error', text: 'No audio track (required for audio training)' },
    ])
    expect(clipWarnings({ probe: probe({ width: 256, height: 720 }) })).toEqual([
      { level: 'warn', text: 'Below 512px short edge' },
    ])
    expect(clipWarnings({ probe: null }, { requireAudio: true })).toEqual([])
  })

  it('emits compact badges while omitting an unknown frame rate', () => {
    expect(probeBadges(probe({ durationSeconds: 61.2, fps: 0, hasAudio: false }))).toEqual([
      '1m 1s',
      '1280×720',
      'no audio',
    ])
  })
})

describe('LoRA dataset quality summaries', () => {
  it('ignores negative duration, counts only meaningful captions, and detects mixed ratios', () => {
    const health = datasetHealth([
      { caption: ' described ', probe: probe({ durationSeconds: 2, width: 640, height: 360 }) },
      { caption: '   ', probe: probe({ durationSeconds: -4, width: 360, height: 640 }) },
      { caption: 'caption only' },
    ])

    expect(health).toEqual({
      clipCount: 3,
      captionedCount: 2,
      probedCount: 2,
      totalDurationSeconds: 2,
      aspectRatios: ['16:9', '9:16'],
      aspectConsistent: false,
      minShortEdge: 360,
      maxShortEdge: 360,
      warningCount: 2,
      errorCount: 0,
      score: 45,
    })
  })

  it('keeps trainer requirements separate from recommendations and reports downscaling', () => {
    const clips = Array.from({ length: 3 }, () => ({
      caption: '',
      probe: probe({ width: 2560, height: 1440 }),
    }))
    const checks = preflightChecks(clips)

    expect(checks).toEqual([
      {
        ok: true,
        blocker: true,
        label: 'At least 3 clips',
        detail: '3 added',
      },
      {
        ok: false,
        blocker: false,
        label: '10+ clips recommended',
        detail: '3 added',
      },
      {
        ok: true,
        blocker: false,
        label: 'No quality errors',
        detail: 'all good',
      },
      {
        ok: true,
        blocker: false,
        label: 'High-res source — downscaled for training',
        detail: '1440px → ≤768px',
      },
    ])
  })
})
