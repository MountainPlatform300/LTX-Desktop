import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ImageModelSpec } from '@/lib/image-generation-model-specs'
import { startGalleryImageEdit, toggleElementFullscreen } from './genspace-actions'

const specs = [
  { id: 'z-image', is_edit_model: false, downloaded: true, inference_status: 'available' },
  { id: 'flux-klein', is_edit_model: true, downloaded: false, inference_status: 'available' },
] as ImageModelSpec[]

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Gen Space gallery image editing', () => {
  it('selects the edit model, retains the image, clears the prompt, and focuses the composer', () => {
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      callback(0)
      return 1
    })
    const composer = document.createElement('textarea')
    composer.scrollIntoView = vi.fn()
    composer.focus = vi.fn()
    const setMode = vi.fn()
    const setImageModelId = vi.fn()
    const setInputImage = vi.fn()
    const setPrompt = vi.fn()

    startGalleryImageEdit({
      assetPath: 'C:\\images\\source.png',
      imageModelSpecs: specs,
      setMode,
      setImageModelId,
      setInputImage,
      setPrompt,
      promptComposer: composer,
    })

    expect(setMode).toHaveBeenCalledWith('image')
    expect(setImageModelId).toHaveBeenCalledWith('flux-klein')
    expect(setInputImage).toHaveBeenCalledWith('C:\\images\\source.png')
    expect(setPrompt).toHaveBeenCalledWith('')
    expect(composer.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth', block: 'center' })
    expect(composer.focus).toHaveBeenCalledWith({ preventScroll: true })
  })

  it('keeps the edit input without falling back when no edit model is listed', () => {
    vi.stubGlobal('requestAnimationFrame', () => 1)
    const setImageModelId = vi.fn()
    const setInputImage = vi.fn()

    startGalleryImageEdit({
      assetPath: 'source.png',
      imageModelSpecs: specs.filter((spec) => !spec.is_edit_model),
      setMode: vi.fn(),
      setImageModelId,
      setInputImage,
      setPrompt: vi.fn(),
      promptComposer: null,
    })

    expect(setImageModelId).not.toHaveBeenCalled()
    expect(setInputImage).toHaveBeenCalledWith('source.png')
  })
})

describe('Gen Space comparison fullscreen', () => {
  it('enters and exits fullscreen on the complete comparison element', async () => {
    const viewer = document.createElement('div')
    viewer.requestFullscreen = vi.fn().mockResolvedValue(undefined)
    const exitFullscreen = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(document, 'exitFullscreen', { configurable: true, value: exitFullscreen })
    Object.defineProperty(document, 'fullscreenElement', { configurable: true, value: null })

    await expect(toggleElementFullscreen(viewer)).resolves.toBe('entered')
    expect(viewer.requestFullscreen).toHaveBeenCalledOnce()

    Object.defineProperty(document, 'fullscreenElement', { configurable: true, value: viewer })
    await expect(toggleElementFullscreen(viewer)).resolves.toBe('exited')
    expect(exitFullscreen).toHaveBeenCalledOnce()
  })

  it('reports unavailable and rejected fullscreen requests without throwing', async () => {
    await expect(toggleElementFullscreen(null)).resolves.toBe('unavailable')
    const viewer = document.createElement('div')
    viewer.requestFullscreen = vi.fn().mockRejectedValue(new Error('denied'))
    await expect(toggleElementFullscreen(viewer)).resolves.toBe('failed')
  })
})
