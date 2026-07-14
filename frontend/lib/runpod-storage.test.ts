import { describe, expect, it } from 'vitest'
import {
  estimateRunpodStorageMonthlyUsd,
  formatRunpodStorageMonthlyUsd,
} from './runpod-storage'

describe('RunPod saved model storage pricing', () => {
  it('uses the published tier boundary', () => {
    expect(estimateRunpodStorageMonthlyUsd(250)).toBeCloseTo(17.5)
    expect(estimateRunpodStorageMonthlyUsd(1500)).toBeCloseTo(95)
    expect(formatRunpodStorageMonthlyUsd(250)).toBe('$17.50')
  })

  it('clamps negative capacity', () => {
    expect(estimateRunpodStorageMonthlyUsd(-1)).toBe(0)
  })
})
