import { beforeEach, describe, expect, it } from 'vitest'
import {
  loadLoraUiPreferences,
  saveLoraUiPreferences,
} from './lora-ui-persistence'

beforeEach(() => {
  localStorage.clear()
})

describe('LoRA UI preference migrations', () => {
  it('migrates legacy gallery and import settings into the versioned record', () => {
    localStorage.setItem('lora.pairView', 'sideBySide')
    localStorage.setItem('lora.pairFilter', 'pairsOnly')
    localStorage.setItem('lora.galleryLayout', 'list')
    localStorage.setItem(
      'ltx.lora.importNormalize',
      JSON.stringify({
        trim: { enabled: true, maxSeconds: 8 },
        resolution: { enabled: true, shortSide: 576 },
        fps: { enabled: true, value: 24 },
      }),
    )

    const preferences = loadLoraUiPreferences()

    expect(preferences.pairView).toBe('sideBySide')
    expect(preferences.pairFilter).toBe('pairsOnly')
    expect(preferences.galleryLayout).toBe('list')
    expect(preferences.importNormalize.trim.maxSeconds).toBe(8)
    expect(preferences.importNormalize.resolution.shortSide).toBe(576)
    expect(preferences.importNormalize.fps.value).toBe(24)
    expect(localStorage.getItem('lora.uiPrefs.v1')).not.toBeNull()
    expect(localStorage.getItem('lora.pairView')).toBeNull()
    expect(localStorage.getItem('ltx.lora.importNormalize')).toBeNull()
  })

  it('repairs malformed and unsupported values with safe defaults', () => {
    localStorage.setItem(
      'lora.uiPrefs.v1',
      JSON.stringify({
        pairView: 'unknown',
        pairFilter: 42,
        galleryLayout: 'tiles',
        importNormalize: {
          trim: { enabled: 'yes', maxSeconds: -1 },
          resolution: { enabled: true, shortSide: 999 },
          fps: { enabled: true, value: 120 },
        },
      }),
    )

    const preferences = loadLoraUiPreferences()

    expect(preferences.pairView).toBe('combined')
    expect(preferences.pairFilter).toBe('all')
    expect(preferences.galleryLayout).toBe('grid')
    expect(preferences.importNormalize.trim).toEqual({
      enabled: false,
      maxSeconds: 10,
    })
    expect(preferences.importNormalize.resolution.shortSide).toBe(720)
    expect(preferences.importNormalize.fps.value).toBe(25)
  })

  it('patches one preference without dropping the others', () => {
    saveLoraUiPreferences({ pairView: 'flat' })
    const updated = saveLoraUiPreferences({ galleryLayout: 'list' })

    expect(updated.pairView).toBe('flat')
    expect(updated.galleryLayout).toBe('list')
    expect(updated.pairFilter).toBe('all')
  })

  it('persists and clamps the Compute pane split', () => {
    expect(saveLoraUiPreferences({ computePanePercent: 48 }).computePanePercent).toBe(48)
    expect(saveLoraUiPreferences({ computePanePercent: 90 }).computePanePercent).toBe(60)
    expect(saveLoraUiPreferences({ computePanePercent: 5 }).computePanePercent).toBe(24)
  })

  it('persists sidebar sections, collapse state, and width safely', () => {
    const updated = saveLoraUiPreferences({
      sidebarSectionSizes: { datasets: 50, runs: 20, compute: 30 },
      collapsedSidebarSections: ['runs'],
      sidebarWidth: 420,
    })

    expect(updated.sidebarSectionSizes).toEqual({
      datasets: 50,
      runs: 20,
      compute: 30,
    })
    expect(updated.collapsedSidebarSections).toEqual(['runs'])
    expect(updated.sidebarWidth).toBe(420)
    expect(saveLoraUiPreferences({ sidebarWidth: 900 }).sidebarWidth).toBe(520)
  })
})
