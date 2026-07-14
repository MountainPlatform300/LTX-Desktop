import type { ApiRequestBodyOf, ApiSuccessOf } from './api-client'

export type RunpodConnectResult = ApiSuccessOf<'connectRunpod'>
export type RunpodVolume = NonNullable<RunpodConnectResult['volumes']>[number]
export type RunpodPod = NonNullable<RunpodConnectResult['pods']>[number]
export type RunpodSelection = NonNullable<ApiRequestBodyOf<'startLoraTraining'>['runpodSelection']>
export type RunpodWorkspacePolicy = RunpodSelection['workspacePolicy']
export type RunpodEstimateRequest = ApiRequestBodyOf<'estimateRunpodTraining'>
export type RunpodEstimate = ApiSuccessOf<'estimateRunpodTraining'>
export type RunpodEstimateWorkload = Omit<
  RunpodEstimateRequest,
  | 'gpuType'
  | 'gpuVramGb'
  | 'gpuPricePerHr'
  | 'storageReadiness'
  | 'estimatedModelDownloadBytes'
  | 'idleTimeoutMinutes'
  | 'storageSizeGb'
>

export interface RunpodGpuOffer {
  id: string
  label: string
  memoryGb: number
  pricePerHr: number | null
  available: boolean
  activeRegionAvailable: boolean
  availableElsewhere: boolean
  bestAvailableRegion: string | null
  recommended: boolean
}

export interface RunpodInventory {
  message: string
  gpus: RunpodGpuOffer[]
  volumes: RunpodVolume[]
  pods: RunpodPod[]
  activeVolumeId: string | null
  datacenter: string
  cacheEnabled: boolean
  savedModelReadiness: RunpodConnectResult['savedModelReadiness']
  estimatedModelDownloadBytes: number | null
}

export type RunpodApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: { code: string; message: string } }

export function normalizeRunpodInventory(data: RunpodConnectResult): RunpodInventory {
  const regions = data.regionHealth ?? []
  const activeRegion = data.datacenter ?? ''
  return {
    message: data.message,
    activeVolumeId: data.activeVolumeId ?? null,
    datacenter: activeRegion,
    cacheEnabled: data.cacheEnabled,
    volumes: data.volumes ?? [],
    pods: data.pods ?? [],
    gpus: (data.gpus ?? []).map((gpu) => {
      const elsewhere = regions.find(
        (region) =>
          region.datacenterId !== activeRegion
          && (region.availableGpuIds ?? []).includes(gpu.id),
      )
      return {
        id: gpu.id,
        label: gpu.label,
        memoryGb: gpu.memoryGb,
        pricePerHr: gpu.pricePerHr ?? null,
        available: gpu.available,
        activeRegionAvailable: gpu.activeRegionAvailable ?? gpu.available,
        availableElsewhere: gpu.availableElsewhere ?? Boolean(elsewhere),
        bestAvailableRegion: gpu.bestAvailableRegion
          ?? (gpu.available ? activeRegion : elsewhere?.datacenterId ?? null),
        recommended: gpu.recommended ?? gpu.memoryGb >= 80,
      }
    }),
    savedModelReadiness: data.savedModelReadiness,
    estimatedModelDownloadBytes: data.estimatedModelDownloadBytes ?? null,
  }
}

export function isGpuSelectionRequired(message: string): boolean {
  return /gpu_selection_required|gpu (?:selection|capacity)|capacity changed|out of stock/i.test(message)
}
