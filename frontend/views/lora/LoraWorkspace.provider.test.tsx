import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ProviderSelector } from './LoraWorkspace'

beforeEach(() => {
  Object.defineProperty(window, 'electronAPI', {
    configurable: true,
    value: { platform: 'win32' },
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('ProviderSelector', () => {
  it('portals its menu outside clipping workspace containers', () => {
    const { container } = render(
      <div style={{ overflow: 'hidden', width: 100 }}>
        <ProviderSelector
          provider="runpod"
          eligibility={null}
          onSelect={() => {}}
          onSetup={() => {}}
          onOptimizeMemory={() => {}}
        />
      </div>,
    )

    fireEvent.click(screen.getByRole('button', { name: /RunPod/ }))
    const menu = screen.getByRole('menu')
    expect(container.contains(menu)).toBe(false)
    expect(menu.style.position).toBe('')
    expect(menu.className).toContain('fixed')
  })
})
