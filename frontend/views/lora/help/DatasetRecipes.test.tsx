import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  BEST_PRACTICES,
  DATASET_RECIPES,
  DATASET_RECIPES_VERSION,
  DatasetRecipes,
} from './DatasetRecipes'

afterEach(cleanup)

describe('dataset recipe contracts', () => {
  it('publishes v2 target-only caption and safe-hardware guidance', () => {
    const copy = [
      ...DATASET_RECIPES.flatMap((recipe) => recipe.steps),
      ...BEST_PRACTICES,
    ].join(' ')

    expect(DATASET_RECIPES_VERSION).toBe(2)
    expect(copy).toContain('Prepare injects the normalized collection trigger exactly once')
    expect(copy).toContain('IC-LoRA targets must be captioned before upload')
    expect(copy).toContain('references do not need captions')
    expect(copy).toContain('32 GB local is experimental')
    expect(copy).toContain('48 GB+ is the safer cloud baseline')
    expect(copy).toContain('80 GB+ is the standard tier')
    expect(copy).toContain('Local GPU requires CUDA through WSL2')
    expect(copy).toContain('RunPod is billed while active')
    expect(copy).not.toMatch(/prepend (the|a|your) (rare )?trigger/i)
  })

  it('renders accessibly and keeps both next actions available', () => {
    const onNewCollection = vi.fn()
    const onStartTour = vi.fn()
    render(
      <DatasetRecipes
        open
        onClose={vi.fn()}
        onNewCollection={onNewCollection}
        onStartTour={onStartTour}
      />,
    )

    expect(screen.getByRole('dialog', { name: 'Dataset recipes' })).toBeTruthy()
    expect(screen.getByText('Guidance v2')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'New collection' }))
    fireEvent.click(screen.getByRole('button', { name: 'Take the tour' }))
    expect(onNewCollection).toHaveBeenCalledOnce()
    expect(onStartTour).toHaveBeenCalledOnce()
  })
})
