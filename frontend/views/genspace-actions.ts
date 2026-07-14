import type { ImageModelSpec } from '@/lib/image-generation-model-specs'

export interface GalleryImageEditOptions {
  assetPath: string
  imageModelSpecs: ImageModelSpec[]
  setMode: (mode: 'image') => void
  setImageModelId: (modelId: string) => void
  setInputImage: (path: string) => void
  setPrompt: (prompt: string) => void
  promptComposer: HTMLTextAreaElement | null
}

export function startGalleryImageEdit({
  assetPath,
  imageModelSpecs,
  setMode,
  setImageModelId,
  setInputImage,
  setPrompt,
  promptComposer,
}: GalleryImageEditOptions): void {
  const editModel = imageModelSpecs.find((spec) => spec.is_edit_model)
  setMode('image')
  if (editModel) setImageModelId(editModel.id)
  setInputImage(assetPath)
  setPrompt('')
  requestAnimationFrame(() => {
    promptComposer?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    promptComposer?.focus({ preventScroll: true })
  })
}

export type FullscreenToggleResult = 'entered' | 'exited' | 'unavailable' | 'failed'

export async function toggleElementFullscreen(element: HTMLElement | null): Promise<FullscreenToggleResult> {
  try {
    if (element && document.fullscreenElement === element) {
      await document.exitFullscreen()
      return 'exited'
    }
    if (!element?.requestFullscreen) return 'unavailable'
    await element.requestFullscreen()
    return 'entered'
  } catch {
    return 'failed'
  }
}
