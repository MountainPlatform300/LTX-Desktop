import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { resetBackendCredentials } from '../lib/backend'
import {
  ApiClient,
  type ApiRequestBodyOf,
  type ApiSuccessOf,
} from '../lib/api-client'
import {
  normalizeRunpodInventory,
  type RunpodApiResult,
  type RunpodEstimate,
  type RunpodEstimateRequest,
  type RunpodInventory,
  type RunpodSelection,
} from '../lib/runpod-contracts'

/** Trainer backend the LoRA trainer's runs use (persisted setting). Sourced
 *  from the generated settings response so it stays in lockstep with the API. */
export type LoraProvider = ApiSuccessOf<'getSettings'>['loraProvider']

export interface AppSettings {
  useTorchCompile: boolean
  hasLtxApiKey: boolean
  userPrefersLtxApiVideoGenerations: boolean
  hasFalApiKey: boolean
  hasGeminiApiKey: boolean
  hasPexelsApiKey: boolean
  useLocalTextEncoder: boolean
  // Opt-in higher-quality IC-LoRA base: when on AND the dev checkpoint +
  // distilled v1.1 LoRA are downloaded, IC-LoRA generations use the dev base
  // with the distilled LoRA stacked @0.5 (ComfyUI dev + distilled-LoRA flow).
  // Adds ~54 GB of optional downloads; off by default.
  useDevQualityBase: boolean
  promptCacheSize: number
  promptEnhancerEnabledT2V: boolean
  promptEnhancerEnabledI2V: boolean
  seedLocked: boolean
  lockedSeed: number
  modelsDir: string
  // --- LoRA trainer (cloud GPU control plane) ---
  // Secrets are masked: the backend returns only has* flags, never the raw
  // key. The remaining fields are non-secret config persisted via the normal
  // settings sync.
  hasRunpodApiKey: boolean
  hasHfToken: boolean
  loraRemoteModelPath: string
  loraRemoteTextEncoderPath: string
  loraRemoteWorkspaceDir: string
  runpodGpuType: string
  // VRAM (GB) of the selected GPU, persisted alongside runpodGpuType so the
  // backend can auto-match the training preset to the hardware. 0 = unknown.
  runpodGpuVramGb: number
  // Max simultaneous in-flight Fal requests for the trainer's dataset-prep
  // generation (Nano Banana edits, Kling). Higher = faster bulk runs, more
  // likely to hit Fal rate limits (which auto-retry). Clamped 1..20 backend.
  loraFalConcurrency: number
  // RunPod auto-provisioning: when on, a fresh pod is bootstrapped (trainer
  // install + optional model download) before the first upload.
  loraAutoProvision: boolean
  loraTrainerRepoUrl: string
  loraTrainerRepoRef: string
  loraModelHfRepo: string
  // Single checkpoint file pulled from the (huge) model repo during
  // provisioning; also drives the auto-derived remote model path.
  loraModelCheckpointFile: string
  loraTextEncoderHfRepo: string
  runpodImage: string
  runpodNetworkVolumeId: string
  // When on, the connect flow auto-creates/reuses a network volume so the
  // multi-GB weights download once and survive pod teardown.
  runpodKeepModelCached: boolean
  // Size (GB) for a newly created cache/workspace. 250 is the recommended
  // minimum; users can choose more for many datasets or checkpoints.
  runpodVolumeSizeGb: number
  // Auto-stop an idle training pod after this many minutes (0 = never).
  runpodIdleStopMinutes: number
  // Trainer backend new runs use. Selected via the trainer's top-right provider
  // pill and persisted here; defaults to 'runpod' (cloud GPU, unchanged flow).
  loraProvider: LoraProvider
}

export const DEFAULT_APP_SETTINGS: AppSettings = {
  useTorchCompile: false,
  hasLtxApiKey: false,
  userPrefersLtxApiVideoGenerations: false,
  hasFalApiKey: false,
  hasGeminiApiKey: false,
  hasPexelsApiKey: false,
  useLocalTextEncoder: false,
  useDevQualityBase: false,
  promptCacheSize: 1,
  promptEnhancerEnabledT2V: false,
  promptEnhancerEnabledI2V: false,
  seedLocked: false,
  lockedSeed: 42,
  modelsDir: '',
  hasRunpodApiKey: false,
  hasHfToken: false,
  loraRemoteModelPath: '',
  loraRemoteTextEncoderPath: '',
  loraRemoteWorkspaceDir: '/workspace',
  runpodGpuType: '',
  runpodGpuVramGb: 0,
  loraFalConcurrency: 4,
  loraAutoProvision: true,
  loraTrainerRepoUrl: 'https://github.com/Lightricks/LTX-2.git',
  loraTrainerRepoRef: 'main',
  loraModelHfRepo: 'Lightricks/LTX-2.3',
  loraModelCheckpointFile: 'ltx-2.3-22b-dev.safetensors',
  loraTextEncoderHfRepo: 'google/gemma-3-12b-it-qat-q4_0-unquantized',
  runpodImage: '',
  runpodNetworkVolumeId: '',
  runpodKeepModelCached: false,
  runpodVolumeSizeGb: 250,
  runpodIdleStopMinutes: 10,
  loraProvider: 'runpod',
}

type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface AppSettingsContextValue {
  settings: AppSettings
  isLoaded: boolean
  runtimePolicyLoaded: boolean
  updateSettings: (patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => void
  refreshSettings: () => Promise<void>
  saveLtxApiKey: (value: string) => Promise<void>
  saveFalApiKey: (value: string) => Promise<void>
  saveGeminiApiKey: (value: string) => Promise<void>
  savePexelsApiKey: (value: string) => Promise<void>
  saveRunpodApiKey: (value: string) => Promise<void>
  saveHfToken: (value: string) => Promise<void>
  connectRunpod: () => Promise<ConnectRunpodResult>
  getRunpodInventory: () => Promise<RunpodApiResult<RunpodInventory>>
  estimateRunpodTraining: (request: RunpodEstimateRequest) => Promise<RunpodApiResult<RunpodEstimate>>
  reselectRunpod: (
    target: { kind: 'dataset' | 'training'; id: string },
    selection: RunpodSelection,
  ) => Promise<RunpodApiResult<unknown>>
  createRunpodVolume: (
    body: ApiRequestBodyOf<'createRunpodVolume'>,
  ) => Promise<RunpodVolumeActionResult>
  selectRunpodVolume: (volumeId: string) => Promise<RunpodVolumeActionResult>
  disableRunpodCache: () => Promise<RunpodVolumeActionResult>
  relocateRunpodVolume: (
    body: ApiRequestBodyOf<'relocateRunpodVolume'>,
  ) => Promise<RunpodVolumeActionResult>
  deleteRunpodVolume: (volumeId: string) => Promise<RunpodVolumeActionResult>

  forceApiGenerations: boolean
  shouldVideoGenerateWithLtxApi: boolean
}

export type ConnectRunpodResult = ApiSuccessOf<'connectRunpod'>
export type RunpodVolumeActionResult = ApiSuccessOf<'createRunpodVolume'>

const AppSettingsContext = createContext<AppSettingsContextValue | null>(null)

function toBackendProcessStatus(value: unknown): BackendProcessStatus | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as { status?: unknown }
  if (record.status === 'alive' || record.status === 'restarting' || record.status === 'dead') {
    return record.status
  }
  return null
}

type AppSettingsInput = Partial<AppSettings>

function normalizeAppSettings(data: AppSettingsInput): AppSettings {
  return {
    useTorchCompile: data.useTorchCompile ?? DEFAULT_APP_SETTINGS.useTorchCompile,
    hasLtxApiKey: data.hasLtxApiKey ?? DEFAULT_APP_SETTINGS.hasLtxApiKey,
    userPrefersLtxApiVideoGenerations: data.userPrefersLtxApiVideoGenerations ?? DEFAULT_APP_SETTINGS.userPrefersLtxApiVideoGenerations,
    hasFalApiKey: data.hasFalApiKey ?? DEFAULT_APP_SETTINGS.hasFalApiKey,
    hasGeminiApiKey: data.hasGeminiApiKey ?? DEFAULT_APP_SETTINGS.hasGeminiApiKey,
    hasPexelsApiKey: data.hasPexelsApiKey ?? DEFAULT_APP_SETTINGS.hasPexelsApiKey,
    useLocalTextEncoder: data.useLocalTextEncoder ?? DEFAULT_APP_SETTINGS.useLocalTextEncoder,
    useDevQualityBase: data.useDevQualityBase ?? DEFAULT_APP_SETTINGS.useDevQualityBase,
    promptCacheSize: data.promptCacheSize ?? DEFAULT_APP_SETTINGS.promptCacheSize,
    promptEnhancerEnabledT2V: data.promptEnhancerEnabledT2V ?? DEFAULT_APP_SETTINGS.promptEnhancerEnabledT2V,
    promptEnhancerEnabledI2V: data.promptEnhancerEnabledI2V ?? DEFAULT_APP_SETTINGS.promptEnhancerEnabledI2V,
    seedLocked: data.seedLocked ?? DEFAULT_APP_SETTINGS.seedLocked,
    lockedSeed: data.lockedSeed ?? DEFAULT_APP_SETTINGS.lockedSeed,
    modelsDir: data.modelsDir ?? DEFAULT_APP_SETTINGS.modelsDir,
    hasRunpodApiKey: data.hasRunpodApiKey ?? DEFAULT_APP_SETTINGS.hasRunpodApiKey,
    hasHfToken: data.hasHfToken ?? DEFAULT_APP_SETTINGS.hasHfToken,
    loraRemoteModelPath: data.loraRemoteModelPath ?? DEFAULT_APP_SETTINGS.loraRemoteModelPath,
    loraRemoteTextEncoderPath: data.loraRemoteTextEncoderPath ?? DEFAULT_APP_SETTINGS.loraRemoteTextEncoderPath,
    loraRemoteWorkspaceDir: data.loraRemoteWorkspaceDir ?? DEFAULT_APP_SETTINGS.loraRemoteWorkspaceDir,
    runpodGpuType: data.runpodGpuType ?? DEFAULT_APP_SETTINGS.runpodGpuType,
    runpodGpuVramGb: data.runpodGpuVramGb ?? DEFAULT_APP_SETTINGS.runpodGpuVramGb,
    loraFalConcurrency: data.loraFalConcurrency ?? DEFAULT_APP_SETTINGS.loraFalConcurrency,
    loraAutoProvision: data.loraAutoProvision ?? DEFAULT_APP_SETTINGS.loraAutoProvision,
    loraTrainerRepoUrl: data.loraTrainerRepoUrl ?? DEFAULT_APP_SETTINGS.loraTrainerRepoUrl,
    loraTrainerRepoRef: data.loraTrainerRepoRef ?? DEFAULT_APP_SETTINGS.loraTrainerRepoRef,
    loraModelHfRepo: data.loraModelHfRepo ?? DEFAULT_APP_SETTINGS.loraModelHfRepo,
    loraModelCheckpointFile: data.loraModelCheckpointFile ?? DEFAULT_APP_SETTINGS.loraModelCheckpointFile,
    loraTextEncoderHfRepo: data.loraTextEncoderHfRepo ?? DEFAULT_APP_SETTINGS.loraTextEncoderHfRepo,
    runpodImage: data.runpodImage ?? DEFAULT_APP_SETTINGS.runpodImage,
    runpodNetworkVolumeId: data.runpodNetworkVolumeId ?? DEFAULT_APP_SETTINGS.runpodNetworkVolumeId,
    runpodKeepModelCached: data.runpodKeepModelCached ?? DEFAULT_APP_SETTINGS.runpodKeepModelCached,
    runpodVolumeSizeGb: data.runpodVolumeSizeGb ?? DEFAULT_APP_SETTINGS.runpodVolumeSizeGb,
    runpodIdleStopMinutes: data.runpodIdleStopMinutes ?? DEFAULT_APP_SETTINGS.runpodIdleStopMinutes,
    loraProvider: data.loraProvider ?? DEFAULT_APP_SETTINGS.loraProvider,
  }
}

type RuntimePolicyPayload = ApiSuccessOf<'getRuntimePolicy'>

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_APP_SETTINGS)
  const [isLoaded, setIsLoaded] = useState(false)
  const [runtimePolicyLoaded, setRuntimePolicyLoaded] = useState(false)
  const [forceApiGenerations, setForceApiGenerations] = useState(true)
  const [backendProcessStatus, setBackendProcessStatus] = useState<BackendProcessStatus | null>(null)

  useEffect(() => {
    if (backendProcessStatus !== 'alive') return

    let cancelled = false
    setRuntimePolicyLoaded(false)

    const fetchRuntimePolicy = async () => {
      const result = await ApiClient.getRuntimePolicy()
      if (!result.ok) {
        if (!cancelled) {
          // Fail closed until policy can be read.
          setForceApiGenerations(true)
          setRuntimePolicyLoaded(true)
        }
        return
      }

      const payload = result.data as RuntimePolicyPayload
      if (typeof payload.force_api_generations !== 'boolean') {
        if (!cancelled) {
          setForceApiGenerations(true)
        }
      } else if (!cancelled) {
        setForceApiGenerations(payload.force_api_generations)
      }

      if (!cancelled) {
        setRuntimePolicyLoaded(true)
      }
    }

    void fetchRuntimePolicy()

    return () => {
      cancelled = true
    }
  }, [backendProcessStatus])

  useEffect(() => {
    let cancelled = false

    const applyStatus = (value: unknown) => {
      const nextStatus = toBackendProcessStatus(value)
      if (!nextStatus || cancelled) {
        return
      }
      if (nextStatus === 'alive') {
        resetBackendCredentials()
      }
      setBackendProcessStatus(nextStatus)
    }

    const unsubscribe = window.electronAPI.onBackendHealthStatus((data) => {
      applyStatus(data)
    })

    void window.electronAPI.getBackendHealthStatus()
      .then((snapshot) => {
        applyStatus(snapshot)
      })
      .catch(() => {
        // Snapshot is optional at startup; subscription continues to listen for pushes.
      })

    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [])

  const refreshSettings = useCallback(async () => {
    const result = await ApiClient.getSettings()
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    setSettings(normalizeAppSettings(result.data))
    setIsLoaded(true)
  }, [])

  useEffect(() => {
    if (isLoaded || backendProcessStatus !== 'alive') return

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null

    const fetchSettings = async () => {
      try {
        await refreshSettings()
        if (cancelled) return
      } catch {
        if (!cancelled) {
          retryTimer = setTimeout(fetchSettings, 1000)
        }
      }
    }

    fetchSettings()

    return () => {
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
    }
  }, [backendProcessStatus, isLoaded, refreshSettings])

  useEffect(() => {
    if (!isLoaded || backendProcessStatus !== 'alive') return
    const syncTimer = setTimeout(async () => {
      // Strip read-only masked flags (has*) and host-managed fields the
      // backend owns; everything else round-trips through the settings sync.
      const {
        hasLtxApiKey: _a, hasFalApiKey: _b, hasGeminiApiKey: _c, modelsDir: _d,
        hasRunpodApiKey: _e, hasPexelsApiKey: _g, hasHfToken: _h,
        ...syncPayload
      } = settings
      const result = await ApiClient.updateSettings(syncPayload)
      if (!result.ok) {
        // Best-effort settings sync.
      }
    }, 150)
    return () => clearTimeout(syncTimer)
  }, [backendProcessStatus, isLoaded, settings])

  const updateSettings = useCallback((patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => {
    if (typeof patch === 'function') {
      setSettings((prev) => patch(prev))
      return
    }
    setSettings((prev) => ({ ...prev, ...patch }))
  }, [])

  const saveLtxApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ ltxApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveGeminiApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ geminiApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveFalApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ falApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const savePexelsApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ pexelsApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveRunpodApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ runpodApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveHfToken = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ hfToken: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const connectRunpod = useCallback(async (): Promise<ConnectRunpodResult> => {
    const result = await ApiClient.connectRunpod()
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    // Pull any server-side stale-volume cleanup back into the durable UI state.
    // Paid volume creation is handled only by the explicit volume action.
    await refreshSettings()
    return result.data
  }, [refreshSettings])

  const getRunpodInventory = useCallback(async (): Promise<RunpodApiResult<RunpodInventory>> => {
    const result = await ApiClient.connectRunpod()
    if (!result.ok) {
      return { ok: false, error: { code: 'RUNPOD_INVENTORY_FAILED', message: result.error.message } }
    }
    if (!result.data.ok) {
      return { ok: false, error: { code: 'RUNPOD_INVENTORY_FAILED', message: result.data.message } }
    }
    return { ok: true, data: normalizeRunpodInventory(result.data) }
  }, [])

  const estimateRunpodTraining = useCallback(async (
    request: RunpodEstimateRequest,
  ): Promise<RunpodApiResult<RunpodEstimate>> => {
    const result = await ApiClient.estimateRunpodTraining(request)
    if (!result.ok) {
      return { ok: false, error: { code: 'RUNPOD_ESTIMATE_FAILED', message: result.error.message } }
    }
    return { ok: true, data: result.data }
  }, [])

  const reselectRunpod = useCallback(async (
    target: { kind: 'dataset' | 'training'; id: string },
    selection: RunpodSelection,
  ): Promise<RunpodApiResult<unknown>> => {
    const result = target.kind === 'dataset'
      ? await ApiClient.reselectLoraDatasetRunpod(target.id, { selection })
      : await ApiClient.reselectLoraTrainingRunpod(target.id, { selection })
    if (!result.ok) {
      return { ok: false, error: { code: 'RUNPOD_RESELECT_FAILED', message: result.error.message } }
    }
    return { ok: true, data: result.data }
  }, [])

  const createRunpodVolume = useCallback(async (
    body: ApiRequestBodyOf<'createRunpodVolume'>,
  ): Promise<RunpodVolumeActionResult> => {
    const result = await ApiClient.createRunpodVolume(body)
    if (!result.ok) throw new Error(result.error.message)
    await refreshSettings()
    return result.data
  }, [refreshSettings])

  const selectRunpodVolume = useCallback(async (
    volumeId: string,
  ): Promise<RunpodVolumeActionResult> => {
    const result = await ApiClient.selectRunpodVolume({ volumeId })
    if (!result.ok) throw new Error(result.error.message)
    await refreshSettings()
    return result.data
  }, [refreshSettings])

  const disableRunpodCache = useCallback(async (): Promise<RunpodVolumeActionResult> => {
    const result = await ApiClient.disableRunpodCache()
    if (!result.ok) throw new Error(result.error.message)
    await refreshSettings()
    return result.data
  }, [refreshSettings])

  const relocateRunpodVolume = useCallback(async (
    body: ApiRequestBodyOf<'relocateRunpodVolume'>,
  ): Promise<RunpodVolumeActionResult> => {
    const result = await ApiClient.relocateRunpodVolume(body)
    if (!result.ok) throw new Error(result.error.message)
    await refreshSettings()
    return result.data
  }, [refreshSettings])

  const deleteRunpodVolume = useCallback(async (
    volumeId: string,
  ): Promise<RunpodVolumeActionResult> => {
    const result = await ApiClient.deleteRunpodVolume(volumeId)
    if (!result.ok) throw new Error(result.error.message)
    await refreshSettings()
    return result.data
  }, [refreshSettings])

  const shouldVideoGenerateWithLtxApi =
    forceApiGenerations || (settings.userPrefersLtxApiVideoGenerations && settings.hasLtxApiKey)

  const contextValue = useMemo<AppSettingsContextValue>(
    () => ({
      settings,
      isLoaded,
      runtimePolicyLoaded,
      updateSettings,
      refreshSettings,
      saveLtxApiKey,
      saveFalApiKey,
      saveGeminiApiKey,
      savePexelsApiKey,
      saveRunpodApiKey,
      saveHfToken,
      connectRunpod,
      getRunpodInventory,
      estimateRunpodTraining,
      reselectRunpod,
      createRunpodVolume,
      selectRunpodVolume,
      disableRunpodCache,
      relocateRunpodVolume,
      deleteRunpodVolume,
      forceApiGenerations,
      shouldVideoGenerateWithLtxApi,
    }),
    [connectRunpod, createRunpodVolume, deleteRunpodVolume, disableRunpodCache, estimateRunpodTraining, forceApiGenerations, getRunpodInventory, isLoaded, refreshSettings, relocateRunpodVolume, reselectRunpod, runtimePolicyLoaded, saveFalApiKey, saveGeminiApiKey, savePexelsApiKey, saveLtxApiKey, saveRunpodApiKey, saveHfToken, selectRunpodVolume, settings, shouldVideoGenerateWithLtxApi, updateSettings],
  )

  return <AppSettingsContext.Provider value={contextValue}>{children}</AppSettingsContext.Provider>
}

export function useAppSettings() {
  const context = useContext(AppSettingsContext)
  if (!context) {
    throw new Error('useAppSettings must be used within AppSettingsProvider')
  }
  return context
}
