import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { LoraInferenceEntry } from '../../hooks/use-lora-inference-registry'

vi.mock('../../lib/backend-media', () => ({
  useBackendMediaUrl: () => ({ url: 'blob:example', error: null }),
}))

import { ExampleThumb } from './LibraryView'

const ENTRY: LoraInferenceEntry = {
  id: 'example',
  name: 'Before and after',
  kind: 'imported',
  variant: 'video_input_ic_lora',
  available: true,
  conditioningTypes: [],
  promptTemplateCustomized: false,
  exampleMediaType: 'image',
}

afterEach(cleanup)

describe('LoRA example framing', () => {
  it('fits fixed gallery images without cropping the source frame', () => {
    const rendered = render(<ExampleThumb entry={ENTRY} className="h-full w-full" />)
    const image = rendered.getByRole('img')
    expect(image.className).toContain('object-contain')
    expect(image.className).not.toContain('object-cover')
  })

  it('uses intrinsic sizing for detailed ultrawide videos', () => {
    const rendered = render(
      <ExampleThumb entry={{ ...ENTRY, exampleMediaType: 'video' }} intrinsic controls />,
    )
    const video = rendered.container.querySelector('video')
    expect(video?.className).toContain('object-contain')
    expect(video?.className).toContain('h-auto')
    expect(video?.hasAttribute('controls')).toBe(true)
  })
})
