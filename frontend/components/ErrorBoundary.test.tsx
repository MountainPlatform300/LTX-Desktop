import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ErrorBoundary } from './ErrorBoundary'

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

function BrokenView(): never {
  throw new Error('render failed')
}

describe('ErrorBoundary', () => {
  it('isolates a render failure and recovers when the view changes', async () => {
    const rendered = render(
      <ErrorBoundary resetKey="broken">
        <BrokenView />
      </ErrorBoundary>,
    )

    expect(screen.getByRole('alert')).toBeTruthy()
    expect(screen.getByText('This view could not be displayed')).toBeTruthy()

    rendered.rerender(
      <ErrorBoundary resetKey="recovered">
        <div>Recovered content</div>
      </ErrorBoundary>,
    )

    expect(await screen.findByText('Recovered content')).toBeTruthy()
  })
})
