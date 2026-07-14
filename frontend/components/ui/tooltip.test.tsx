import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Tooltip } from './tooltip'

const rect = (left: number, top: number, width: number, height: number): DOMRect => ({
  left,
  top,
  width,
  height,
  right: left + width,
  bottom: top + height,
  x: left,
  y: top,
  toJSON: () => ({}),
})

describe('Tooltip viewport positioning', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 200 })
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 120 })
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      return this.getAttribute('role') === 'tooltip'
        ? rect(0, 0, 80, 30)
        : rect(190, 2, 20, 20)
    })
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('flips below a top-edge trigger and clamps inside the right viewport edge', () => {
    render(
      <Tooltip content="Stop this pod">
        <button type="button">Stop</button>
      </Tooltip>,
    )

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'Stop' }).parentElement!)
    act(() => vi.advanceTimersByTime(500))

    const tooltip = screen.getByRole('tooltip')
    expect(tooltip.style.visibility).toBe('visible')
    expect(tooltip.style.top).toBe('28px')
    expect(tooltip.style.left).toBe('112px')
  })

  it('recomputes while visible when the viewport changes', () => {
    render(
      <Tooltip content="Details" side="right">
        <button type="button">Info</button>
      </Tooltip>,
    )
    fireEvent.mouseEnter(screen.getByRole('button', { name: 'Info' }).parentElement!)
    act(() => vi.advanceTimersByTime(500))

    fireEvent.resize(window)
    act(() => vi.runOnlyPendingTimers())

    expect(screen.getByRole('tooltip').style.visibility).toBe('visible')
  })
})
