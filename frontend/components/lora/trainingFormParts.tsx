// Shared, minimal building blocks for the LoRA training modals. Each piece is
// intentionally small and self-contained so the modals stay quiet and
// one-line-friendly: explanatory sentences live behind the InfoHint tooltip
// instead of being printed under every field.
//
// Reused by PreprocessModal, TrainPipelineModal, and StartTrainingModal so the
// three entry points look and behave consistently (same GPU switch, same
// profile picker, same resolution field).

import { useId, type ReactNode } from 'react'
import { Info } from 'lucide-react'
import { Tooltip } from '../ui/tooltip'
import type {
  LocalTrainerEligibility,
  LoraDataset,
  LoraProfile,
  LoraProvider,
} from '../../contexts/LoraTrainingContext'

// A compact "(i)" icon that hides a helper sentence in a hover tooltip. This is
// the main lever for keeping the forms uncluttered: the label stays, the long
// explanation moves into the tooltip.
export function InfoHint({
  content,
  side = 'top',
}: {
  content: ReactNode
  side?: 'top' | 'bottom' | 'left' | 'right'
}) {
  return (
    <Tooltip wide side={side} content={content}>
      <Info className="inline-block h-3.5 w-3.5 text-zinc-600 hover:text-zinc-400 align-middle cursor-help" />
    </Tooltip>
  )
}

// Opens the app's Settings modal on the LoRA Trainer tab. The app listens for
// this event at the top level, so any component can trigger it without prop
// drilling a setter.
function openLoraSettings() {
  window.dispatchEvent(new CustomEvent('open-settings', { detail: { tab: 'loraTrainer' } }))
}

// Inline Local / RunPod switch. It persists straight to settings.loraProvider —
// the same source of truth as the workspace's top-right provider pill — so the
// choice is reflected everywhere and never drifts. Local is only selectable
// when the machine is eligible (CUDA + WSL2); otherwise it's disabled.
export function GpuSelector({
  provider,
  onChange,
  localEligibility,
  runpodGpuType,
  runpodVramGb,
}: {
  provider: LoraProvider
  onChange: (p: LoraProvider) => void
  localEligibility: LocalTrainerEligibility | null
  runpodGpuType: string
  runpodVramGb: number
}) {
  const localEligible = localEligibility?.eligible ?? false
  const vram = provider === 'local' ? (localEligibility?.vramGb ?? 0) : runpodVramGb
  const gpuLabel =
    provider === 'local'
      ? localEligibility?.gpuName
        ? `${localEligibility.gpuName}${vram ? ` · ${vram} GB` : ''}`
        : 'Local GPU'
      : runpodGpuType
        ? `${runpodGpuType}${vram ? ` · ${vram} GB` : ''}`
        : 'No GPU selected'

  const hint =
    provider === 'local'
      ? 'Trains on your PC\u2019s GPU through WSL2. No cloud cost; needs a CUDA-capable GPU and WSL2 set up.'
      : 'Trains on a rented RunPod GPU. You\u2019re billed only while the run is active; the run summary shows the total time. Pick the GPU model in Settings.'

  return (
    <fieldset className="space-y-1.5">
      <legend className="text-xs font-medium text-zinc-300">
        GPU{' '}
        <InfoHint content={hint} />
      </legend>
      <div className="grid grid-cols-2 gap-1.5 rounded-lg bg-zinc-800/60 border border-zinc-700 p-1">
        <button
          type="button"
          aria-pressed={provider === 'local'}
          disabled={!localEligible}
          onClick={() => onChange('local')}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            provider === 'local' ? 'bg-blue-600 text-white' : 'text-zinc-300 hover:bg-zinc-700/60'
          } ${!localEligible ? 'opacity-40 cursor-not-allowed' : ''}`}
        >
          Local GPU
        </button>
        <button
          type="button"
          aria-pressed={provider === 'runpod'}
          onClick={() => onChange('runpod')}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            provider === 'runpod' ? 'bg-blue-600 text-white' : 'text-zinc-300 hover:bg-zinc-700/60'
          }`}
        >
          RunPod
        </button>
      </div>
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span className="text-zinc-500 truncate" title={gpuLabel}>{gpuLabel}</span>
        <button type="button" onClick={openLoraSettings} className="text-blue-400 hover:text-blue-300 shrink-0">
          Settings
        </button>
      </div>
    </fieldset>
  )
}

// Profile dropdown with an "Auto (recommended)" first option. A null value
// means "let the backend pick" — the caller decides whether to resolve that to
// a concrete profile id or send null through (the pipeline auto-matches).
export function ProfilePicker({
  profiles,
  value,
  onChange,
  onManageProfiles,
  datasetType = 'standard',
  vramGb = 0,
  autoProfileId,
}: {
  profiles: LoraProfile[]
  value: string | null
  onChange: (id: string | null) => void
  onManageProfiles?: () => void
  datasetType?: LoraDataset['type']
  vramGb?: number
  autoProfileId?: string | null
}) {
  const selectId = useId()
  const compatible = profiles.filter((profile) =>
    (profile.datasetTypes ?? ['standard', 'ic_lora']).includes(datasetType),
  )
  const resolved = profiles.find((profile) => profile.id === (value ?? autoProfileId)) ?? null
  const hardwareNote =
    value === null && vramGb < 80
      ? 'Adjusted with the official Low VRAM settings: rank 16 and int8 memory mode.'
      : null
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <label htmlFor={selectId} className="text-xs font-medium text-zinc-300">Training profile</label>
          <InfoHint content={'Profiles describe what you are training. Auto chooses by dataset type, then safely adapts memory settings to the selected GPU.'} />
        </div>
        {onManageProfiles && (
          <button type="button" onClick={onManageProfiles} className="text-[11px] text-blue-400 hover:text-blue-300">
            Edit
          </button>
        )}
      </div>
      {compatible.length === 0 ? (
        <p className="text-[11px] text-amber-400">{'No profiles found. Open \u201CEdit\u201D to create one.'}</p>
      ) : (
        <select
          id={selectId}
          value={value ?? ''}
          onChange={(e) => onChange(e.target.value || null)}
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">
            Auto{resolved ? ` — ${resolved.name}` : ' (recommended)'}
          </option>
          {compatible.map((p) => (
            <option
              key={p.id}
              value={p.id}
              disabled={p.minVramGb != null && vramGb < p.minVramGb}
            >
              {p.name}{p.minVramGb != null ? ` (${p.minVramGb} GB+)` : ''}
            </option>
          ))}
        </select>
      )}
      {resolved?.description && (
        <p className="text-[10px] leading-relaxed text-zinc-500">{resolved.description}</p>
      )}
      {hardwareNote && <p className="text-[10px] text-blue-300/80">{hardwareNote}</p>}
    </div>
  )
}

// Resolution buckets input. The format rules (multiples of 32, frames % 8 == 1)
// are the kind of detail that clutters a form if printed — they go in the
// InfoHint next to the label.
export function ResolutionInput({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  const inputId = useId()
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <label htmlFor={inputId} className="text-xs font-medium text-zinc-300">Resolution</label>
        <InfoHint content={'Format: WIDTHxHEIGHTxFRAMES. Width/height must be multiples of 32; frames must satisfy frames % 8 == 1 (e.g. 89). Separate multiple buckets with ";".'} />
      </div>
      <input
        id={inputId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
    </div>
  )
}
