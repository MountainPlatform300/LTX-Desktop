import { describe, expect, it } from 'vitest'
import type { StudioClip } from '../views/studio/studio-store'
import {
  countReadyPairs,
  derivePairs,
  pairReadiness,
} from './lora-pairs'

function clip(
  id: string,
  overrides: Partial<StudioClip> = {},
): StudioClip {
  const path = `C:/clips/${id}.mp4`
  return {
    id,
    sourcePath: path,
    localPath: path,
    caption: 'A useful training caption',
    kind: 'video',
    origin: 'imported',
    referencePath: null,
    referencePaths: [],
    driverPath: null,
    edits: null,
    posterPath: null,
    spritePath: null,
    spriteTiles: null,
    probe: null,
    durationSeconds: 2,
    triage: null,
    deletedAt: null,
    editPreview: null,
    ...overrides,
  }
}

describe('LoRA pair derivation', () => {
  it('groups targets sharing a reference and excludes members from loose clips', () => {
    const control = clip('control')
    const targetA = clip('target-a', { referencePath: control.localPath })
    const targetB = clip('target-b', { referencePath: control.localPath })
    const loose = clip('loose')

    const result = derivePairs([control, targetA, targetB, loose])

    expect(result.pairs).toHaveLength(1)
    expect(result.pairs[0].controls.map((item) => item.id)).toEqual(['control'])
    expect(result.pairs[0].targets.map((item) => item.id)).toEqual([
      'target-a',
      'target-b',
    ])
    expect([...result.looseClipIds]).toEqual(['loose'])
    expect(countReadyPairs(result.pairs)).toBe(1)
  })

  it('reports missing references as an error', () => {
    const target = clip('target', {
      referencePath: 'C:/clips/missing.mp4',
    })
    const group = derivePairs([target]).pairs[0]

    expect(pairReadiness(group)).toEqual({
      tone: 'error',
      reasons: ['Input clip is missing from this dataset'],
    })
  })

  it('warns when an output has no caption', () => {
    const control = clip('control')
    const target = clip('target', {
      caption: '',
      referencePath: control.localPath,
    })

    const readiness = pairReadiness(derivePairs([control, target]).pairs[0])

    expect(readiness.tone).toBe('warn')
    expect(readiness.reasons).toContain('Edited clip needs a caption')
  })
})
