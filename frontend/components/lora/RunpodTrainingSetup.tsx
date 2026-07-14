import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Check, Clock3, Download, HardDrive, Info, Loader2, RefreshCw, Search, WalletCards } from 'lucide-react'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { estimateRunpodTrainingFallback } from '../../lib/runpod-estimate'
import { ExpertOverrideWarning, isRiskyTrainingOverride } from './trainingSafety'
import type {
  RunpodEstimate,
  RunpodEstimateWorkload,
  RunpodGpuOffer,
  RunpodInventory,
  RunpodSelection,
} from '../../lib/runpod-contracts'

export type { RunpodEstimateWorkload } from '../../lib/runpod-contracts'

interface Props {
  value: RunpodSelection | null
  onChange: (selection: RunpodSelection) => void
  estimateInputs: RunpodEstimateWorkload
  disabled?: boolean
  capacityMessage?: string | null
  allowUnsafeOverride?: boolean
  onAllowUnsafeOverrideChange?: (checked: boolean) => void
}

const money = (value: number) => `$${value.toFixed(2)}`
const duration = (seconds: number) => {
  const minutes = Math.max(1, Math.round(seconds / 60))
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}
const selectable = (gpu: RunpodGpuOffer) => gpu.available

function readyCacheForGpu(gpu: RunpodGpuOffer, inventory: RunpodInventory) {
  return inventory.volumes.find((volume) =>
    volume.createdByApp
    && volume.savedModelReadiness === 'ready'
    && (volume.availableGpuIds ?? []).includes(gpu.id))
}

function availability(gpu: RunpodGpuOffer, inventory: RunpodInventory): string {
  if (!selectable(gpu)) return 'Currently unavailable'
  const cache = readyCacheForGpu(gpu, inventory)
  if (cache) return `No model download required · Cache in ${cache.datacenterId}`
  return gpu.bestAvailableRegion
    ? `Available in ${gpu.bestAvailableRegion} · Model download required`
    : 'Available globally · Model download required'
}

export function RunpodTrainingSetup({
  value,
  onChange,
  estimateInputs,
  disabled = false,
  capacityMessage,
  allowUnsafeOverride = false,
  onAllowUnsafeOverrideChange,
}: Props) {
  const { settings, getRunpodInventory, estimateRunpodTraining } = useAppSettings()
  const [inventory, setInventory] = useState<RunpodInventory | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [minVram, setMinVram] = useState(0)
  const [availableOnly, setAvailableOnly] = useState(false)
  const [estimate, setEstimate] = useState<RunpodEstimate | null>(null)
  const [estimating, setEstimating] = useState(false)
  const [fallbackEstimate, setFallbackEstimate] = useState(false)
  const searchRef = useRef<HTMLInputElement>(null)

  const refresh = async () => {
    setLoading(true)
    setError(null)
    const result = await getRunpodInventory()
    setLoading(false)
    if (result.ok) setInventory(result.data)
    else setError(result.error.message)
  }

  useEffect(() => {
    void refresh()
    const frame = requestAnimationFrame(() => searchRef.current?.focus())
    return () => cancelAnimationFrame(frame)
    // Opening/remounting this step intentionally performs a fresh preflight.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const isRiskyGpu = (gpu: RunpodGpuOffer) => isRiskyTrainingOverride(
    estimateInputs.config ?? null,
    gpu.memoryGb,
    estimateInputs.resolutionBuckets,
  )
  const selectedGpu = inventory?.gpus.find((gpu) => gpu.id === value?.gpuType) ?? null
  const selectedGpuIsRisky = selectedGpu ? isRiskyGpu(selectedGpu) : false
  const selectedCache = inventory && selectedGpu
    ? inventory.volumes.find((volume) => volume.id === value?.volumeId)
      ?? readyCacheForGpu(selectedGpu, inventory)
    : null
  const estimateKey = JSON.stringify({
    selectedGpu,
    value,
    estimateInputs,
    readiness: selectedCache?.savedModelReadiness,
    downloadBytes: inventory?.estimatedModelDownloadBytes,
    idle: settings.runpodIdleStopMinutes,
    storageSize: selectedCache?.sizeGb,
  })
  useEffect(() => {
    if (!inventory || !selectedGpu || selectedGpu.pricePerHr == null || !value) {
      setEstimate(null)
      setFallbackEstimate(false)
      return
    }
    let cancelled = false
    const request = {
      ...estimateInputs,
      gpuType: selectedGpu.id,
      gpuVramGb: selectedGpu.memoryGb,
      gpuPricePerHr: selectedGpu.pricePerHr,
      storageReadiness: selectedCache?.savedModelReadiness ?? 'missing' as const,
      estimatedModelDownloadBytes: inventory.estimatedModelDownloadBytes,
      idleTimeoutMinutes: settings.runpodIdleStopMinutes,
      storageSizeGb: selectedCache?.sizeGb ?? 0,
    }
    const timer = setTimeout(async () => {
      setEstimating(true)
      const result = await estimateRunpodTraining(request)
      if (cancelled) return
      setEstimating(false)
      if (result.ok) {
        setEstimate(result.data)
        setFallbackEstimate(false)
      } else {
        setEstimate(estimateRunpodTrainingFallback(request))
        setFallbackEstimate(true)
      }
    }, 200)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
    // The serialized key intentionally covers nested config/workload fields.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [estimateKey, estimateRunpodTraining])

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return (inventory?.gpus ?? [])
      .filter((gpu) => !needle || `${gpu.label} ${gpu.id} ${gpu.memoryGb}`.toLowerCase().includes(needle))
      .filter((gpu) => gpu.memoryGb >= minVram)
      .filter((gpu) => !availableOnly || selectable(gpu))
      .sort((a, b) =>
        Number(!isRiskyGpu(b)) - Number(!isRiskyGpu(a))
        || Number(b.recommended && selectable(b)) - Number(a.recommended && selectable(a))
        || Number(selectable(b)) - Number(selectable(a))
        || (a.pricePerHr ?? Infinity) - (b.pricePerHr ?? Infinity))
  }, [availableOnly, estimateInputs.config, estimateInputs.resolutionBuckets, inventory, minVram, query])

  const chooseGpu = (gpu: RunpodGpuOffer) => {
    if (!inventory || !selectable(gpu) || disabled) return
    const cache = readyCacheForGpu(gpu, inventory)
    onChange({
      gpuType: gpu.id,
      gpuVramGb: gpu.memoryGb,
      datacenter: cache?.datacenterId ?? gpu.bestAvailableRegion ?? '',
      workspacePolicy: cache ? 'primary_cache' : 'ephemeral_any_region',
      volumeId: cache?.id ?? null,
    })
  }

  const groups = [
    ['recommended', 'Recommended for this training'],
    ['other', 'Other compatible GPUs'],
    ['incompatible', 'Not compatible with selected profile'],
    ['unavailable', 'Currently unavailable'],
  ] as const
  const group = (gpu: RunpodGpuOffer) =>
    !selectable(gpu)
      ? 'unavailable'
      : isRiskyGpu(gpu)
        ? 'incompatible'
        : gpu.recommended
          ? 'recommended'
          : 'other'

  return (
    <div className="space-y-3" aria-busy={loading || estimating}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-white">Choose a RunPod GPU</h3>
          <p className="text-[11px] text-zinc-500">Live stock is refreshed whenever this dialog opens and again before start.</p>
        </div>
        <button type="button" aria-label="Refresh RunPod inventory" onClick={() => void refresh()} disabled={loading || disabled} className="rounded p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-white disabled:opacity-50">
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {capacityMessage && (
        <div role="alert" className="flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2.5 text-xs text-amber-200">
          <AlertTriangle className="h-4 w-4 shrink-0" /> {capacityMessage}
        </div>
      )}

      <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto]">
        <label className="relative">
          <span className="sr-only">Search compatible GPUs</span>
          <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-zinc-500" />
          <input ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search GPU or VRAM" className="w-full rounded-lg border border-zinc-700 bg-zinc-800 py-2 pl-8 pr-3 text-xs text-white focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </label>
        <label className="flex items-center gap-2 text-[11px] text-zinc-400">
          <span>Minimum VRAM</span>
          <select value={minVram} onChange={(event) => setMinVram(Number(event.target.value))} className="rounded border border-zinc-700 bg-zinc-800 px-2 py-2 text-xs text-white">
            <option value={0}>Any</option><option value={32}>32 GB</option><option value={48}>48 GB</option><option value={80}>80 GB</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-[11px] text-zinc-400">
          <input type="checkbox" checked={availableOnly} onChange={(event) => setAvailableOnly(event.target.checked)} /> Available only
        </label>
      </div>

      <div className="h-64 overflow-y-auto rounded-lg border border-zinc-700 bg-zinc-950/30">
        {loading ? (
          <div className="flex h-full items-center justify-center gap-2 text-xs text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Checking RunPod…</div>
        ) : error ? (
          <div role="alert" className="p-4 text-xs text-red-400">{error}</div>
        ) : visible.length === 0 ? (
          <div className="p-4 text-center text-xs text-zinc-500">No compatible GPUs match these filters.</div>
        ) : groups.map(([id, label]) => {
          const rows = visible.filter((gpu) => group(gpu) === id)
          if (!rows.length) return null
          return (
            <section key={id} aria-labelledby={`gpu-group-${id}`}>
              <h4 id={`gpu-group-${id}`} className="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-900/95 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-zinc-400">{label}</h4>
              {rows.map((gpu) => {
                const risky = selectable(gpu) && isRiskyGpu(gpu)
                const status = risky
                  ? 'Likely to run out of memory with selected profile'
                  : availability(gpu, inventory!)
                const selected = value?.gpuType === gpu.id
                const cache = readyCacheForGpu(gpu, inventory!)
                const explanation = risky
                  ? `This profile exceeds the conservative settings for a ${gpu.memoryGb} GB GPU. Choose the Low VRAM profile or explicitly accept the expert override.`
                  : cache
                  ? `LTX Desktop will automatically attach your ready saved-model storage in ${cache.datacenterId}. GPU billing still applies during setup and training.`
                  : status.includes('Model download required')
                    ? 'No ready saved-model storage matches this GPU location, so model files download before training. Existing saved storage continues billing separately.'
                    : null
                return (
                  <button key={gpu.id} type="button" disabled={!selectable(gpu) || disabled} aria-pressed={selected} onClick={() => chooseGpu(gpu)} className={`w-full border-b border-zinc-800/80 px-3 py-2 text-left focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60 ${selected ? 'bg-blue-500/15' : 'hover:bg-zinc-800/60'}`}>
                    <div className="grid gap-1 sm:grid-cols-[minmax(0,1.2fr)_60px_75px_minmax(0,1.7fr)] sm:items-center">
                      <span className="flex min-w-0 items-center gap-2 text-xs font-medium text-white">{selected && <Check className="h-3.5 w-3.5 text-blue-400" />}<span className="truncate">{gpu.label}</span></span>
                      <span className="text-[11px] text-zinc-400">{gpu.memoryGb} GB</span>
                      <span className="text-[11px] text-zinc-300">{gpu.pricePerHr == null ? 'Rate unknown' : `${money(gpu.pricePerHr)}/hr`}</span>
                      <span className={`flex items-center gap-1.5 text-[11px] ${risky ? 'text-amber-300' : 'text-zinc-400'}`}>
                        <span className={`h-2 w-2 shrink-0 rounded-full ${!selectable(gpu) ? 'bg-zinc-500' : risky ? 'bg-amber-400' : cache ? 'bg-emerald-400' : 'bg-blue-400'}`} />
                        {status}
                        {explanation && <Info className="h-3 w-3 shrink-0" aria-label={`About ${status}`}><title>{explanation}</title></Info>}
                      </span>
                    </div>
                  </button>
                )
              })}
            </section>
          )
        })}
      </div>

      {selectedGpuIsRisky && onAllowUnsafeOverrideChange && (
        <div className="space-y-1.5">
          <p className="text-[11px] text-amber-300">
            The selected {selectedGpu?.memoryGb} GB GPU is below this profile&apos;s conservative VRAM tier. Choose the Low VRAM profile, or acknowledge the risk below.
          </p>
          <ExpertOverrideWarning
            checked={allowUnsafeOverride}
            onChange={onAllowUnsafeOverrideChange}
            provider="runpod"
          />
        </div>
      )}

      {selectedGpu && value && (
        <section aria-labelledby="runpod-estimate-heading" className="overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900/70">
          <div className="flex flex-wrap items-start justify-between gap-2 border-b border-zinc-800 px-4 py-3">
            <div>
              <h4 id="runpod-estimate-heading" className="text-xs font-semibold text-white">Estimated RunPod charge</h4>
              <p className="mt-0.5 text-[10px] text-zinc-500">For this training run; actual billing follows GPU runtime.</p>
            </div>
            <span className="rounded-full bg-zinc-800 px-2 py-1 text-[10px] text-zinc-400">
              {estimate
                ? `${estimate.confidence[0].toUpperCase()}${estimate.confidence.slice(1)} confidence · ${estimate.matchedHistoryCount} similar ${estimate.matchedHistoryCount === 1 ? 'run' : 'runs'}`
                : 'Calculating estimate…'}
            </span>
          </div>

          <div className="grid gap-px bg-zinc-800 sm:grid-cols-2">
            <div className="bg-zinc-900 px-4 py-3">
              <div className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                <WalletCards className="h-3.5 w-3.5" /> GPU charge
              </div>
              <p className="mt-1 text-lg font-semibold tabular-nums text-white">
                {estimate ? `${money(estimate.lowGpuCost)}–${money(estimate.highGpuCost)}` : 'Unavailable'}
              </p>
              <p className="text-[10px] text-zinc-500">One-time estimate for this run</p>
            </div>
            <div className="bg-zinc-900 px-4 py-3">
              <div className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                <Clock3 className="h-3.5 w-3.5" /> GPU rental time
              </div>
              <p className="mt-1 text-lg font-semibold tabular-nums text-white">
                {estimate ? `${duration(estimate.lowSeconds)}–${duration(estimate.highSeconds)}` : 'Unavailable'}
              </p>
              <p className="text-[10px] text-zinc-500">Includes setup, training, and idle buffer</p>
            </div>
          </div>

          {estimate && (
            <div className="border-t border-zinc-800 px-4 py-3">
              <p className="mb-2 text-[10px] font-medium uppercase tracking-wide text-zinc-500">Estimated time by phase</p>
              <div className="flex flex-wrap gap-1.5">
                {estimate.phases.map((phase) => (
                  <span key={phase.phase} className="rounded-md bg-zinc-800/80 px-2 py-1 text-[10px] text-zinc-300">
                    <span className="capitalize text-zinc-500">{phase.phase}</span>{' '}
                    <span className="tabular-nums">{duration(phase.lowSeconds)}–{duration(phase.highSeconds)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="border-t border-zinc-800 px-4 py-3">
            {selectedCache ? (
              <div className="flex gap-2.5 rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-2.5">
                <HardDrive className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                <div>
                  <p className="text-[11px] font-medium text-emerald-200">Using your saved models in {selectedCache.datacenterId}</p>
                  <p className="mt-0.5 text-[10px] leading-relaxed text-zinc-400">
                    No model download is required. This existing storage costs {estimate ? money(estimate.storageMonthlyCost) : '—'}/month while retained, regardless of whether you run training.
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex gap-2.5 rounded-lg border border-blue-500/20 bg-blue-500/5 p-2.5">
                <Download className="mt-0.5 h-4 w-4 shrink-0 text-blue-400" />
                <div>
                  <p className="text-[11px] font-medium text-blue-200">Model download included in this estimate</p>
                  <p className="mt-0.5 text-[10px] leading-relaxed text-zinc-400">
                    This GPU cannot use ready saved-model storage. Any other retained volumes continue billing separately and are not included in the GPU charge above.
                  </p>
                </div>
              </div>
            )}
            <p className="mt-2 text-[10px] text-zinc-500">
              {fallbackEstimate
                ? 'Server estimate unavailable; showing a fallback low-confidence planning range.'
                : 'RunPod bills actual GPU time, so the final amount may fall outside this range.'}
            </p>
          </div>
        </section>
      )}
    </div>
  )
}
