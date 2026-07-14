import { useState, useEffect, useCallback } from 'react'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import { useHfAuth } from './use-hf-auth'
import { useHfModelAccess } from './use-hf-model-access'

// Mirrors the param/response types used by the model-download endpoints so the
// gate stays in sync with the generated OpenAPI client without hand-rolling.
type StartModelDownloadBody = NonNullable<ApiRequestBodyOf<'startModelDownload'>>
type ModelCheckpointID = NonNullable<StartModelDownloadBody['cp_ids']>[number]
type DownloadProgress = ApiSuccessOf<'getModelDownloadProgress'>
type HfAuthStatus = ApiSuccessOf<'getHuggingFaceAuthStatus'>['status']
type ModelAccessMap = ApiSuccessOf<'checkModelAccess'>['access']

export interface ModelDownloadGateItem {
  id: ModelCheckpointID
  label: string
  downloaded: boolean
  progress: number
  status: string
}

export interface UseIcLoraModelGateResult {
  /** True when every required checkpoint is present on disk. */
  ready: boolean
  checking: boolean
  downloading: boolean
  error: string | null
  gateItems: ModelDownloadGateItem[]
  hfAuthStatus: HfAuthStatus
  hfAuthPolling: boolean
  startHuggingFaceLogin: () => Promise<void>
  accessMap: ModelAccessMap
  allAuthorized: boolean
  startDownload: () => void
  refresh: () => void
}

/**
 * Owns the IC-LoRA model-download gate for the Gen Space LoRA flow.
 *
 * When `active`, fetches `GET /api/models/ltx-ic-lora-recommendation` to learn
 * which checkpoints the union IC-LoRA still needs (the union adapter itself,
 * the MiDaS depth processor, and — for pose — the DW pose processor + YOLOX
 * person detector), then drives HuggingFace auth → access checks →
 * `startModelDownload` → progress polling until everything is local. This is
 * the same machinery the old `ICLoraPanel` used inline; extracted so the new
 * PromptBar LoRA pill can gate pose/depth without a separate surface.
 */
export function useIcLoraModelGate(active: boolean): UseIcLoraModelGateResult {
  const [requiredCpIds, setRequiredCpIds] = useState<ModelCheckpointID[]>([])
  const [checking, setChecking] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)

  const needsModels = requiredCpIds.length > 0
  const ready = !needsModels

  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin } = useHfAuth(active && needsModels)
  const { accessMap, allAuthorized } = useHfModelAccess(requiredCpIds, hfAuthStatus)

  const checkAvailability = useCallback(async () => {
    setChecking(true)
    const result = await ApiClient.getLtxIcLoraRecommendation()
    if (!result.ok) {
      logger.warn(`Failed to fetch IC-LoRA model status: ${result.error.message}`)
      setError(result.error.message)
      setChecking(false)
      return
    }
    const cps = result.data.cps_to_download
    setRequiredCpIds(cps)
    if (cps.length === 0) {
      setDownloading(false)
      setProgress(null)
      setError(null)
      setSessionId(null)
    }
    setChecking(false)
  }, [])

  useEffect(() => {
    if (!active) {
      // Deactivated (LoRA cleared / switched away) — drop stale state so a
      // re-selection re-checks from a clean slate.
      setRequiredCpIds([])
      setChecking(false)
      setDownloading(false)
      setProgress(null)
      setError(null)
      setSessionId(null)
      return
    }
    void checkAvailability()
  }, [active, checkAvailability])

  useEffect(() => {
    if (!needsModels || !downloading || !sessionId) return

    const pollProgress = async () => {
      const result = await ApiClient.getModelDownloadProgress({ sessionId })
      if (!result.ok) {
        logger.warn(`Failed polling IC-LoRA download progress: ${result.error.message}`)
        return
      }
      const payload = result.data
      setProgress(payload)
      if (payload.status === 'error') {
        setDownloading(false)
        setError(payload.error || 'Model download failed')
        return
      }
      if (payload.status === 'complete') {
        setDownloading(false)
        await checkAvailability()
      }
    }

    void pollProgress()
    const interval = setInterval(() => { void pollProgress() }, 1000)
    return () => clearInterval(interval)
  }, [needsModels, downloading, sessionId, checkAvailability])

  const startDownload = useCallback(async () => {
    if (downloading) return
    setError(null)
    const result = await ApiClient.startModelDownload({
      type: 'download',
      cp_ids: [...requiredCpIds],
    })
    if (!result.ok) {
      logger.warn(`Failed to start IC-LoRA download: ${result.error.message}`)
      setError(result.error.message)
      return
    }
    const started = result.data
    if (started.status === 'started') {
      setSessionId(started.sessionId)
      setDownloading(true)
      return
    }
    setError('Unexpected response while starting IC-LoRA download')
  }, [downloading, requiredCpIds])

  const refresh = useCallback(() => { void checkAvailability() }, [checkAvailability])

  const runningProgress = progress?.status === 'downloading' ? progress : null
  const gateItemIds = [...new Set([...requiredCpIds, ...(runningProgress?.all_files ?? [])])]
  const gateItems: ModelDownloadGateItem[] = gateItemIds.map((cpId) => {
    const downloaded = !requiredCpIds.includes(cpId)
    const isCompleted = runningProgress?.completed_files?.includes(cpId) ?? false
    const isCurrent = downloading && runningProgress?.current_downloading_file === cpId
    const pct = downloaded
      ? 100
      : isCompleted
        ? 100
        : isCurrent
          ? (runningProgress?.current_file_progress ?? 0)
          : 0
    const status = downloaded
      ? 'Ready'
      : isCompleted
        ? 'Complete'
        : isCurrent
          ? 'Downloading'
          : 'Missing'
    return { id: cpId, label: cpId, downloaded, progress: pct, status }
  })

  return {
    ready,
    checking,
    downloading,
    error,
    gateItems,
    hfAuthStatus,
    hfAuthPolling,
    startHuggingFaceLogin,
    accessMap,
    allAuthorized,
    startDownload,
    refresh,
  }
}
