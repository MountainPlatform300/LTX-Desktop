import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  GuidedTour,
  LEGACY_TOUR_DONE_KEYS,
  TOUR_DONE_KEY,
  TOUR_VERSION,
  getTourSteps,
  markTourComplete,
  shouldAutoStartTour,
} from './GuidedTour'

beforeEach(() => window.localStorage.clear())
afterEach(cleanup)

describe('LoRA guided tour contracts', () => {
  it('intentionally reopens v2 for users who completed v1, then cleans up v1', () => {
    expect(TOUR_VERSION).toBe(2)
    window.localStorage.setItem(LEGACY_TOUR_DONE_KEYS[0], '1')

    expect(shouldAutoStartTour(window.localStorage, true)).toBe(true)
    expect(shouldAutoStartTour(window.localStorage, false)).toBe(false)

    markTourComplete(window.localStorage)
    expect(window.localStorage.getItem(TOUR_DONE_KEY)).toBe('1')
    expect(window.localStorage.getItem(LEGACY_TOUR_DONE_KEYS[0])).toBeNull()
    expect(shouldAutoStartTour(window.localStorage, true)).toBe(false)
  })

  it('branches Standard and IC-LoRA captions without manual trigger instructions', () => {
    const standard = getTourSteps('standard').map((step) => `${step.title} ${step.body}`).join(' ')
    const icLora = getTourSteps('ic_lora').map((step) => `${step.title} ${step.body}`).join(' ')
    const allCopy = `${standard} ${icLora}`

    expect(standard).toContain('Caption Standard clips')
    expect(standard).toContain('Prepare injects the collection trigger exactly once')
    expect(icLora).toContain('Caption IC-LoRA outputs')
    expect(icLora).toContain('Reference inputs do not need captions')
    expect(icLora).toContain('remote auto-caption is unavailable')
    expect(allCopy).not.toMatch(/prepend (the|your) trigger/i)
  })

  it('covers provider, queue, library, and verified-trigger behavior', () => {
    const copy = getTourSteps(null).map((step) => step.body).join(' ')

    expect(copy).toContain('Local GPU')
    expect(copy).toContain('RunPod')
    expect(copy).toContain('Auto (recommended)')
    expect(copy).toContain('Cancel all LoRA jobs')
    expect(copy).toContain('global across collections')
    expect(copy).toContain('LoRA Library')
    expect(copy).toContain('never guesses one from a name or filename')
    expect(copy).toContain('Apply LoRA')
  })

  it('renders an accessible dialog and marks completion when skipped', () => {
    const onClose = vi.fn()
    render(
      <GuidedTour
        open
        datasetType="standard"
        onClose={onClose}
        onOpenRecipes={vi.fn()}
      />,
    )

    const dialog = screen.getByRole('dialog', { name: 'Welcome to LoRA Studio' })
    expect(dialog.getAttribute('aria-modal')).toBe('true')
    fireEvent.keyDown(dialog, { key: 'Escape' })

    expect(onClose).toHaveBeenCalledOnce()
    expect(window.localStorage.getItem(TOUR_DONE_KEY)).toBe('1')
  })
})
