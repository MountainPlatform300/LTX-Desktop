import type { LoraTrainingJob } from '../../contexts/LoraTrainingContext'

function formatMb(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${Math.round(mb)} MB`
}

function Bar({ label, value, suffix }: { label: string; value: number; suffix: string }) {
  const pct = Math.max(0, Math.min(100, value))
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px] text-zinc-400">
        <span>{label}</span>
        <span className="tabular-nums text-zinc-300">
          {Math.round(value)}
          {suffix}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full bg-blue-500 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

/**
 * Live GPU telemetry for a training run (VRAM, utilization, temp). Renders only
 * when the backend has reported a status for this job; hidden otherwise (e.g.
 * before the first poll, or for a provider that doesn't expose telemetry).
 */
export function GpuStatusPanel({ job }: { job: LoraTrainingJob }) {
  const gpu = job.gpuStatus
  if (!gpu) return null

  const vramPct = gpu.vramTotalMb > 0 ? (gpu.vramUsedMb / gpu.vramTotalMb) * 100 : 0

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-zinc-200">GPU</span>
          <span className="text-xs text-zinc-400">{gpu.name}</span>
        </div>
        {gpu.tempC != null && (
          <span className="text-[11px] tabular-nums text-zinc-400">{Math.round(gpu.tempC)}°C</span>
        )}
      </div>
      <div className="space-y-3">
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[11px] text-zinc-400">
            <span>VRAM</span>
            <span className="tabular-nums text-zinc-300">
              {formatMb(gpu.vramUsedMb)} / {formatMb(gpu.vramTotalMb)}
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-300"
              style={{ width: `${vramPct}%` }}
            />
          </div>
        </div>
        <Bar label="GPU utilization" value={gpu.gpuUtilPct} suffix="%" />
        <Bar label="Memory utilization" value={gpu.memUtilPct} suffix="%" />
      </div>
    </div>
  )
}
