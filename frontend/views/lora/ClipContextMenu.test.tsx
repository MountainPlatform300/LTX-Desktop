import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { ClipContextMenu } from './ClipContextMenu'

afterEach(() => cleanup())

describe('ClipContextMenu', () => {
  it('portals viewport coordinates outside transformed sidebar ancestors', () => {
    const { container } = render(
      <div style={{ transform: 'translateX(0)', overflow: 'hidden' }}>
        <ClipContextMenu
          x={140}
          y={90}
          items={[{ label: 'Open' }]}
          onClose={() => {}}
        />
      </div>,
    )

    const item = screen.getByRole('button', { name: 'Open' })
    expect(container.contains(item)).toBe(false)
    expect(item.parentElement?.style.left).toBe('140px')
    expect(item.parentElement?.style.top).toBe('90px')
  })
})
