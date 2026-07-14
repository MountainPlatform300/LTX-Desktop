import { useCallback, useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import type { ImageModelSpecsResponse } from '../lib/image-generation-model-specs'

interface ImageGenerationModelSpecsState {
  modelSpecs: ImageModelSpecsResponse | null
  isLoading: boolean
  errorMessage: string | null
  refresh: () => void
}

export function useImageGenerationModelSpecs(): ImageGenerationModelSpecsState {
  const [state, setState] = useState<Omit<ImageGenerationModelSpecsState, 'refresh'>>({
    modelSpecs: null,
    isLoading: true,
    errorMessage: null,
  })

  const fetchSpecs = useCallback((signal: AbortSignal) => {
    void (async () => {
      const result = await ApiClient.getImageModelSpecs(undefined, { signal })
      if (signal.aborted) return
      if (result.ok) {
        setState({ modelSpecs: result.data, isLoading: false, errorMessage: null })
        return
      }
      setState({ modelSpecs: null, isLoading: false, errorMessage: result.error.message })
    })()
  }, [])

  useEffect(() => {
    const abortController = new AbortController()
    fetchSpecs(abortController.signal)
    return () => {
      abortController.abort()
    }
  }, [fetchSpecs])

  const refresh = useCallback(() => {
    const abortController = new AbortController()
    fetchSpecs(abortController.signal)
  }, [fetchSpecs])

  return { ...state, refresh }
}
