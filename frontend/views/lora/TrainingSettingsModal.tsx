import { X } from 'lucide-react'
import type {
  LoraPreprocessed,
  LoraTrainingJob,
} from '../../contexts/LoraTrainingContext'

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1 border-b border-zinc-800/60 last:border-0">
      <span className="text-[11px] text-zinc-500 shrink-0">{label}</span>
      <span className="text-[11px] text-zinc-300 text-right break-words font-mono">{value}</span>
    </div>
  )
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="bg-zinc-900/40 border border-zinc-800 rounded-lg p-4">
      <p className="text-xs font-medium text-zinc-300 mb-2">{title}</p>
      <div className="space-y-0">{children}</div>
    </section>
  )
}

/**
 * Read-only view of the preprocessing + training settings for a collection, at
 * any stage. Surfaced from the dataset inspector so the user can confirm what a
 * run is/will be using without waiting for the post-run summary — useful while
 * prep or training is in progress. Before a training job exists only the prep
 * settings are available (the training config is chosen at train start).
 */
export function TrainingSettingsModal({
  preprocessed,
  training,
  onClose,
}: {
  preprocessed: LoraPreprocessed | null
  training: LoraTrainingJob | null
  onClose: () => void
}) {
  const hasPre = preprocessed != null
  const hasJob = training != null
  const cfg = training?.config

  // Mirror RunView's preset-derived defaults so the shown optimizer/quant
  // matches what the trainer actually uses when the field is unset.
  const lowVram = cfg?.preset === 'low_vram'
  const optimizer = cfg?.optimizerType || (lowVram ? 'adamw8bit' : 'adamw')
  const quant = cfg?.quantization || (lowVram ? 'int8-quanto' : 'none')

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative flex max-h-[85vh] w-full max-w-lg mx-4 flex-col bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl">
        <div className="flex shrink-0 items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="text-base font-semibold text-white">Training settings</h2>
          <button
            onClick={onClose}
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-4">
          {!hasPre && !hasJob && (
            <p className="text-xs text-zinc-500 leading-relaxed">
              No preprocessing or training settings yet. Start training to record a
              configuration here.
            </p>
          )}

          {hasPre && preprocessed && (
            <Group title="Preprocessing">
              <Row label="Resolution buckets" value={preprocessed.resolutionBuckets} />
              {preprocessed.effectiveResolutionBuckets && (
                <Row label="Effective resolution" value={preprocessed.effectiveResolutionBuckets} />
              )}
              <Row label="Train with audio" value={preprocessed.withAudio ? 'yes' : 'no'} />
              <Row label="Auto-caption" value={preprocessed.autoCaption ? 'yes' : 'no'} />
              <Row label="Captioner" value={preprocessed.captionerType} />
              <Row label="Status" value={preprocessed.status} />
              {preprocessed.startedAt && (
                <Row label="Started" value={new Date(preprocessed.startedAt).toLocaleString()} />
              )}
            </Group>
          )}

          {hasJob && training && cfg && (
            <Group title="Training config">
              <Row label="Provider" value={training.provider} />
              <Row label="Preset" value={cfg.preset} />
              <Row label="Rank / alpha" value={`${cfg.rank} / ${cfg.alpha}`} />
              <Row label="Steps" value={String(cfg.steps)} />
              <Row label="Learning rate" value={String(cfg.learningRate)} />
              <Row label="Batch size" value={String(cfg.batchSize)} />
              <Row label="Grad accumulation" value={String(cfg.gradientAccumulationSteps)} />
              <Row label="Optimizer" value={optimizer} />
              <Row label="Quantization" value={quant} />
              <Row label="Scheduler" value={cfg.schedulerType} />
              <Row label="Mixed precision" value={cfg.mixedPrecisionMode} />
              <Row label="Gradient checkpointing" value={cfg.enableGradientCheckpointing ? 'on' : 'off'} />
              <Row label="Target modules" value={(cfg.targetModules ?? []).join(', ')} />
              <Row label="Trigger word" value={cfg.triggerWord || '—'} />
              <Row label="Train with audio" value={cfg.withAudio ? 'yes' : 'no'} />
              <Row label="Seed" value={String(cfg.seed)} />
            </Group>
          )}

          {hasJob && training && cfg && (
            <Group title="Validation">
              <Row label="Resolution" value={`${cfg.validationVideoWidth}×${cfg.validationVideoHeight}`} />
              <Row label="Frames / fps" value={`${cfg.validationVideoFrames} / ${cfg.validationFrameRate}`} />
              <Row label="Inference steps" value={String(cfg.validationInferenceSteps)} />
              <Row label="Interval" value={String(cfg.validationInterval)} />
              <Row label="Guidance scale" value={String(cfg.validationGuidanceScale)} />
              <Row label="Seed" value={String(cfg.validationSeed)} />
              <Row label="STG scale / mode" value={`${cfg.stgScale} / ${cfg.stgMode}`} />
              <Row label="STG blocks" value={(cfg.stgBlocks ?? []).join(', ')} />
              <Row label="Negative prompt" value={cfg.validationNegativePrompt} />
              <Row label="Validation prompts" value={String(cfg.validationPrompts?.length ?? 0)} />
            </Group>
          )}

          {hasJob && training && cfg && (
            <Group title="Checkpoints">
              <Row label="Interval" value={String(cfg.checkpointInterval)} />
              <Row label="Keep last N" value={String(cfg.checkpointKeepLastN)} />
              <Row label="Precision" value={cfg.checkpointPrecision} />
            </Group>
          )}
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex justify-end gap-2 shrink-0">
          <button
            onClick={onClose}
            className="text-xs px-3 py-1.5 rounded-lg bg-zinc-700 hover:bg-zinc-600 text-white"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
