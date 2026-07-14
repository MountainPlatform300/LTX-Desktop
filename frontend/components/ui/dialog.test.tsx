import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Dialog } from './dialog'

afterEach(cleanup)

describe('Dialog', () => {
  it('labels the dialog, traps focus, closes with Escape, and restores focus', () => {
    const opener = document.createElement('button')
    document.body.append(opener)
    opener.focus()
    const onClose = vi.fn()
    const rendered = render(
      <Dialog title="Train LoRA" onClose={onClose} footer={<button type="button">Start</button>}>
        <input autoFocus aria-label="Run name" />
      </Dialog>,
    )

    const dialog = screen.getByRole('dialog', { name: 'Train LoRA' })
    expect(screen.getByLabelText('Run name')).toBe(document.activeElement)

    screen.getByRole('button', { name: 'Start' }).focus()
    fireEvent.keyDown(dialog, { key: 'Tab' })
    expect(screen.getByRole('button', { name: 'Close Train LoRA' })).toBe(document.activeElement)

    fireEvent.keyDown(dialog, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledOnce()

    rendered.unmount()
    expect(opener).toBe(document.activeElement)
    opener.remove()
  })

  it('guards every close path while busy', () => {
    const onClose = vi.fn()
    const { container } = render(
      <Dialog title="Preprocess" onClose={onClose} closeDisabled>
        Busy
      </Dialog>,
    )

    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    fireEvent.mouseDown(container.querySelector('[aria-hidden="true"]') as Element)
    fireEvent.click(screen.getByRole('button', { name: 'Close Preprocess' }))

    expect(onClose).not.toHaveBeenCalled()
  })
})
