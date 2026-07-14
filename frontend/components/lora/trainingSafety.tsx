import { AlertTriangle } from 'lucide-react'
import type {
  LocalTrainerEligibility,
  LoraProfile,
  LoraProvider,
  LoraTrainingConfig,
} from '../../contexts/LoraTrainingContext'
import type { LoraDataset } from '../../contexts/LoraTrainingContext'

// VRAM the active run will use, provider-aware: local training probes the
// machine's own GPU; RunPod uses the selected cloud GPU. 0 = unknown.
export function activeVramGb(
  provider: LoraProvider,
  runpodVramGb: number,
  localEligibility: LocalTrainerEligibility | null,
): number {
  return provider === 'local' ? (localEligibility?.vramGb ?? 0) : runpodVramGb
}

type LoraDatasetType = LoraDataset['type']

export function compatibleProfiles(
  profiles: LoraProfile[],
  datasetType: LoraDatasetType,
): LoraProfile[] {
  return profiles.filter((profile) =>
    (profile.datasetTypes ?? ['standard', 'ic_lora']).includes(datasetType),
  )
}

// Auto chooses a training goal from the dataset type. Memory adaptation is
// applied separately, so hardware labels no longer clutter the profile list.
export function defaultProfileIdForVram(
  profiles: LoraProfile[],
  vramGb: number,
  datasetType: LoraDatasetType = 'standard',
): string | null {
  return profiles.find(
    (profile) =>
      profile.builtin &&
      profile.autoRecommended &&
      (profile.datasetTypes ?? ['standard', 'ic_lora']).includes(datasetType) &&
      (profile.minVramGb == null || vramGb >= profile.minVramGb),
  )?.id ?? compatibleProfiles(profiles, datasetType)[0]?.id ?? null
}

export function materializeConfigForVram(
  config: LoraTrainingConfig,
  vramGb: number,
): LoraTrainingConfig {
  if (vramGb >= 80) return { ...config }
  const rank = Math.min(config.rank, 16)
  return {
    ...config,
    preset: 'low_vram',
    rank,
    alpha: Math.min(config.alpha, rank),
    batchSize: 1,
    enableGradientCheckpointing: true,
    optimizerType: 'adamw8bit',
    quantization: 'int8-quanto',
    loadTextEncoderIn8bit: true,
    offloadOptimizerDuringValidation: true,
    skipInitialValidation: true,
  }
}

export function effectiveProfileConfig(
  profile: LoraProfile | null,
  vramGb: number,
  hardwareAdaptive = false,
): LoraTrainingConfig | null {
  if (!profile) return null
  return hardwareAdaptive ? materializeConfigForVram(profile.config, vramGb) : { ...profile.config }
}

export function isRiskyTrainingOverride(
  config: LoraTrainingConfig | null,
  vramGb: number,
  resolutionBuckets?: string,
): boolean {
  if (!config || vramGb >= 80) return false
  const unsafeConfig =
    config.preset !== 'low_vram' ||
    config.rank > 16 ||
    config.batchSize !== 1 ||
    config.enableGradientCheckpointing === false ||
    (config.optimizerType != null && config.optimizerType !== 'adamw8bit') ||
    (config.quantization != null && !['int8-quanto', 'int4-quanto'].includes(config.quantization)) ||
    config.loadTextEncoderIn8bit === false ||
    config.offloadOptimizerDuringValidation === false ||
    config.skipInitialValidation === false
  const unsafeBucket = resolutionBuckets
    ? resolutionBuckets.split(';').some((raw) => {
        const [width, height, frames] = raw.trim().split('x').map(Number)
        return !width || !height || !frames || width * height > 512 * 512 || frames > 49
      })
    : false
  return unsafeConfig || unsafeBucket
}

export function ExpertOverrideWarning({
  checked,
  onChange,
  provider,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  provider: LoraProvider
}) {
  return (
    <label className="flex items-start gap-2.5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2.5 text-xs text-amber-100">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5"
      />
      <span>
        <span className="flex items-center gap-1.5 font-medium">
          <AlertTriangle className="h-3.5 w-3.5" /> Expert override
        </span>
        <span className="mt-1 block text-[11px] leading-relaxed text-amber-200/80">
          These settings exceed the conservative profile for this GPU and may run out of memory
          {provider === 'runpod' ? ' or waste RunPod charges' : ''}. Continue only if you understand the risk.
        </span>
      </span>
    </label>
  )
}
