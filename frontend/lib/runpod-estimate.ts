import type { RunpodEstimate, RunpodEstimateRequest } from './runpod-contracts'
import { estimateRunpodStorageMonthlyUsd } from './runpod-storage'

export function estimateRunpodTrainingFallback(inputs: RunpodEstimateRequest): RunpodEstimate {
  const phases: RunpodEstimate['phases'] = [
    { phase: 'provision', lowSeconds: 90, highSeconds: 300 },
  ]
  if (inputs.storageReadiness !== 'ready') {
    phases.push({ phase: 'upload', lowSeconds: 600, highSeconds: 2400 })
  }
  if (!inputs.preprocessed) {
    const prepBase = Math.max(180, inputs.totalClipSeconds * 8)
    phases.push({ phase: 'preprocess', lowSeconds: prepBase, highSeconds: prepBase * 2.5 })
  }
  const steps = Number(inputs.config?.steps ?? 1000)
  const work = Math.max(1, Number.isFinite(steps) ? steps : 1000) * Math.max(1, inputs.clipCount)
  const trainBase = Math.max(900, work * 0.35)
  phases.push(
    { phase: 'train', lowSeconds: trainBase, highSeconds: trainBase * 2.2 },
    { phase: 'idle', lowSeconds: 60, highSeconds: 300 },
  )

  const lowSeconds = phases.reduce((sum, phase) => sum + phase.lowSeconds, 0)
  const highSeconds = phases.reduce((sum, phase) => sum + phase.highSeconds, 0)
  return {
    lowSeconds,
    highSeconds,
    lowGpuCost: lowSeconds / 3600 * inputs.gpuPricePerHr,
    highGpuCost: highSeconds / 3600 * inputs.gpuPricePerHr,
    storageMonthlyCost: estimateRunpodStorageMonthlyUsd(inputs.storageSizeGb),
    confidence: 'low',
    matchedHistoryCount: 0,
    downloadBytes: inputs.estimatedModelDownloadBytes ?? null,
    phases,
  }
}
