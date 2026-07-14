import { useCallback, useEffect, useState } from 'react'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'

// A single LoRA the user can apply from Gen Space — the official LTX-2 union
// IC-LoRA (canny/depth/pose) plus user-trained adapters from completed jobs.
export type LoraInferenceEntry = ApiSuccessOf<'getLoraInferenceRegistry'>['entries'][number]

export type LoraInferenceVariant = LoraInferenceEntry['variant']
export type LoraInferenceConditioningType = 'canny' | 'depth' | 'pose'

export interface LoraInferenceRegistryState {
  entries: LoraInferenceEntry[]
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

/**
 * Fetches the in-app LoRA inference registry (the Gen Space "Apply LoRA"
 * picker source). Refetches on mount and on each `refresh()` — the registry is
 * derived from the training ledger, so a freshly completed job appears after a
 * refresh without an explicit bridge call.
 */
export function useLoraInferenceRegistry(): LoraInferenceRegistryState {
  const [entries, setEntries] = useState<LoraInferenceEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    const result = await ApiClient.getLoraInferenceRegistry()
    if (result.ok) {
      setEntries(result.data.entries)
      setError(null)
    } else {
      const message = (result.error as { message?: string })?.message ?? 'Failed to load LoRAs'
      setError(message)
      logger.warn(`LoraInferenceRegistry: fetch failed (${message})`)
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return { entries, loading, error, refresh }
}
