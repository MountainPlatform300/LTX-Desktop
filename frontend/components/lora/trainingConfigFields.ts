import type { LoraTrainingConfig } from '../../contexts/LoraTrainingContext'

// Declarative descriptors for the trainer config form. Keeping field
// metadata (label, help, control type, range, options, section) as data —
// rather than hand-writing each input — keeps the editor and its tooltips
// consistent and trivial to extend when the backend grows a new knob.

export type ConfigSection =
  | 'optimization'
  | 'acceleration'
  | 'validation'
  | 'checkpoints'
  | 'advanced'

export type ConfigControl =
  | 'number'
  | 'nullableNumber'
  | 'text'
  | 'nullableText'
  | 'toggle'
  | 'triBool'
  | 'enum'
  | 'nullableEnum'
  | 'stringList'
  | 'intList'

// Only the tunable keys (system-owned `withAudio` / `triggerWord` are set per
// run, never in a profile).
export type ConfigFieldKey = Exclude<keyof LoraTrainingConfig, 'preset' | 'withAudio' | 'triggerWord'>

export interface ConfigFieldDescriptor {
  key: ConfigFieldKey
  label: string
  help: string
  control: ConfigControl
  min?: number
  max?: number
  step?: number
  options?: { value: string; label: string }[]
  placeholder?: string
}

// Always-visible primary knobs (rendered above the collapsible sections).
export const PRIMARY_FIELDS: ConfigFieldDescriptor[] = [
  {
    key: 'rank',
    label: 'LoRA rank',
    help: 'Dimensionality of the LoRA adapters. Higher rank can capture more detail but increases size and the risk of overfitting. 16–64 is typical.',
    control: 'number',
    min: 1,
    max: 256,
    step: 1,
  },
  {
    key: 'alpha',
    label: 'Alpha',
    help: 'LoRA scaling factor. Commonly set equal to the rank; the effective update is scaled by alpha / rank.',
    control: 'number',
    min: 1,
    max: 256,
    step: 1,
  },
  {
    key: 'learningRate',
    label: 'Learning rate',
    help: 'Optimizer step size. Too high diverges; too low trains slowly. 1e-4 is a good starting point for LoRA.',
    control: 'number',
    min: 0,
    max: 1,
    step: 0.00001,
  },
  {
    key: 'steps',
    label: 'Training steps',
    help: 'Total optimizer steps. More steps fit the data harder; watch validation samples for overfitting.',
    control: 'number',
    min: 1,
    max: 100000,
    step: 50,
  },
]

// Grouped collapsible sections (rendered in this order).
export const SECTIONS: { id: ConfigSection; title: string; defaultOpen: boolean }[] = [
  { id: 'optimization', title: 'Optimization', defaultOpen: false },
  { id: 'acceleration', title: 'Acceleration & memory', defaultOpen: false },
  { id: 'validation', title: 'Validation', defaultOpen: false },
  { id: 'checkpoints', title: 'Checkpoints', defaultOpen: false },
  { id: 'advanced', title: 'Advanced', defaultOpen: false },
]

export const SECTION_FIELDS: Record<ConfigSection, ConfigFieldDescriptor[]> = {
  optimization: [
    {
      key: 'batchSize',
      label: 'Batch size',
      help: 'Samples processed per step. Raises VRAM use; combine with gradient accumulation when memory is tight.',
      control: 'number',
      min: 1,
      max: 64,
      step: 1,
    },
    {
      key: 'gradientAccumulationSteps',
      label: 'Gradient accumulation',
      help: 'Accumulate gradients over this many micro-batches before an optimizer step — an effective batch-size multiplier without extra VRAM.',
      control: 'number',
      min: 1,
      max: 256,
      step: 1,
    },
    {
      key: 'maxGradNorm',
      label: 'Max gradient norm',
      help: 'Gradient-clipping threshold for training stability. 1.0 is a safe default; 0 disables clipping.',
      control: 'number',
      min: 0,
      max: 100,
      step: 0.1,
    },
    {
      key: 'optimizerType',
      label: 'Optimizer',
      help: 'Optimization algorithm. adamw8bit halves optimizer-state memory. "Auto" follows the preset.',
      control: 'nullableEnum',
      options: [
        { value: 'adamw', label: 'AdamW' },
        { value: 'adamw8bit', label: 'AdamW 8-bit' },
      ],
    },
    {
      key: 'schedulerType',
      label: 'LR scheduler',
      help: 'How the learning rate changes over training. Linear and cosine decay are common; constant holds it fixed.',
      control: 'enum',
      options: [
        { value: 'linear', label: 'Linear' },
        { value: 'constant', label: 'Constant' },
        { value: 'cosine', label: 'Cosine' },
      ],
    },
    {
      key: 'enableGradientCheckpointing',
      label: 'Gradient checkpointing',
      help: 'Trades compute for memory by recomputing activations in the backward pass. Recommended for large models.',
      control: 'toggle',
    },
  ],
  acceleration: [
    {
      key: 'mixedPrecisionMode',
      label: 'Mixed precision',
      help: 'Numeric format for training. bf16 is the most stable on modern GPUs; fp16 saves memory on older ones.',
      control: 'enum',
      options: [
        { value: 'bf16', label: 'bf16' },
        { value: 'fp16', label: 'fp16' },
      ],
    },
    {
      key: 'quantization',
      label: 'Quantization',
      help: 'Quantize the base model to save VRAM at a small quality cost. "Auto" follows the preset. int8 halves the base-model memory; int4 halves it again (needed to train the 22B model on a 32 GB GPU) at a further quality cost.',
      control: 'nullableEnum',
      options: [
        { value: 'null', label: 'None' },
        { value: 'int8-quanto', label: 'int8-quanto' },
        { value: 'int4-quanto', label: 'int4-quanto' },
      ],
    },
    {
      key: 'loadTextEncoderIn8bit',
      label: 'Text encoder in 8-bit',
      help: 'Load the text encoder in 8-bit to save memory. "Auto" follows the preset.',
      control: 'triBool',
    },
    {
      key: 'offloadOptimizerDuringValidation',
      label: 'Offload optimizer in validation',
      help: 'Move optimizer state off the GPU while sampling validation videos, freeing VRAM for inference. "Auto" follows the preset.',
      control: 'triBool',
    },
  ],
  validation: [
    {
      key: 'validationNegativePrompt',
      label: 'Negative prompt',
      help: 'Shared negative prompt applied to every validation sample.',
      control: 'text',
    },
    {
      key: 'validationVideoWidth',
      label: 'Sample width',
      help: 'Width (px) of validation videos. Must be a multiple of 32.',
      control: 'number',
      min: 32,
      max: 2048,
      step: 32,
    },
    {
      key: 'validationVideoHeight',
      label: 'Sample height',
      help: 'Height (px) of validation videos. Must be a multiple of 32.',
      control: 'number',
      min: 32,
      max: 2048,
      step: 32,
    },
    {
      key: 'validationVideoFrames',
      label: 'Sample frames',
      help: 'Number of frames per validation video. Must satisfy frames % 8 == 1 (e.g. 49, 89).',
      control: 'number',
      min: 1,
      max: 513,
      step: 8,
    },
    {
      key: 'validationFrameRate',
      label: 'Sample frame rate',
      help: 'Frames per second for validation playback.',
      control: 'number',
      min: 1,
      max: 60,
      step: 1,
    },
    {
      key: 'validationInferenceSteps',
      label: 'Inference steps',
      help: 'Denoising steps when sampling validation videos. More steps = higher quality but slower.',
      control: 'number',
      min: 1,
      max: 500,
      step: 1,
    },
    {
      key: 'validationInterval',
      label: 'Validation interval',
      help: 'Run validation every N training steps. (IC-LoRA runs skip validation regardless.)',
      control: 'number',
      min: 1,
      max: 100000,
      step: 50,
    },
    {
      key: 'validationGuidanceScale',
      label: 'Guidance scale',
      help: 'Classifier-free guidance strength for validation samples. Higher follows the prompt more strictly.',
      control: 'number',
      min: 0,
      max: 30,
      step: 0.5,
    },
    {
      key: 'validationSeed',
      label: 'Validation seed',
      help: 'Fixed seed so validation samples are comparable across checkpoints.',
      control: 'number',
      min: 0,
      max: 2147483647,
      step: 1,
    },
  ],
  checkpoints: [
    {
      key: 'checkpointInterval',
      label: 'Checkpoint interval',
      help: 'Save a checkpoint every N training steps.',
      control: 'number',
      min: 1,
      max: 100000,
      step: 50,
    },
    {
      key: 'checkpointKeepLastN',
      label: 'Keep last N',
      help: 'Retain only the most recent N checkpoints; older ones are pruned to save disk.',
      control: 'number',
      min: 1,
      max: 100,
      step: 1,
    },
    {
      key: 'checkpointPrecision',
      label: 'Checkpoint precision',
      help: 'Numeric precision the saved weights are stored in.',
      control: 'enum',
      options: [
        { value: 'bfloat16', label: 'bfloat16' },
        { value: 'float16', label: 'float16' },
        { value: 'float32', label: 'float32' },
      ],
    },
  ],
  advanced: [
    {
      key: 'dropout',
      label: 'LoRA dropout',
      help: 'Dropout applied to the LoRA layers. Small values (0–0.1) can help regularize small datasets.',
      control: 'number',
      min: 0,
      max: 1,
      step: 0.05,
    },
    {
      key: 'targetModules',
      label: 'Target modules',
      help: 'Attention sub-modules the LoRA adapts. One per line. Defaults target the attention projections.',
      control: 'stringList',
    },
    {
      key: 'firstFrameConditioningP',
      label: 'First-frame conditioning prob.',
      help: 'How often training conditions on the output clip\'s first frame (a probability, not a strength). 1.0 = the LoRA always relies on the provided start frame; lower values mix in samples trained without it. Useful for "start frame + motion reference -> animated output" LoRAs, where you supply the start frame at inference. Leave on Auto for the per-mode default (0.5 standard, 0.1 IC-LoRA).',
      control: 'nullableNumber',
      min: 0,
      max: 1,
      step: 0.05,
    },
    {
      key: 'numDataloaderWorkers',
      label: 'Dataloader workers',
      help: 'Background processes feeding data to the trainer. Higher can speed up IO-bound runs.',
      control: 'number',
      min: 0,
      max: 16,
      step: 1,
    },
    {
      key: 'stgScale',
      label: 'STG scale',
      help: 'Spatiotemporal-guidance strength used in validation sampling.',
      control: 'number',
      min: 0,
      max: 30,
      step: 0.5,
    },
    {
      key: 'stgBlocks',
      label: 'STG blocks',
      help: 'Transformer block indices STG is applied to. Comma-separated (e.g. 29).',
      control: 'intList',
    },
    {
      key: 'stgMode',
      label: 'STG mode',
      help: 'Spatiotemporal-guidance mode passed to the trainer (e.g. stg_av).',
      control: 'text',
    },
    {
      key: 'skipInitialValidation',
      label: 'Skip initial validation',
      help: 'Skip the validation pass before training starts. "Auto" follows the run mode (IC-LoRA always skips).',
      control: 'triBool',
    },
    {
      key: 'timestepSamplingMode',
      label: 'Timestep sampling',
      help: 'Flow-matching timestep sampling distribution (e.g. shifted_logit_normal).',
      control: 'text',
    },
    {
      key: 'pushToHub',
      label: 'Push to Hugging Face Hub',
      help: 'Upload the trained adapter to the Hub when finished.',
      control: 'toggle',
    },
    {
      key: 'hubModelId',
      label: 'Hub model id',
      help: 'Destination repo (e.g. your-name/my-lora). Required only when pushing to the Hub.',
      control: 'nullableText',
      placeholder: 'username/model-name',
    },
    {
      key: 'wandbEnabled',
      label: 'Weights & Biases logging',
      help: 'Stream training metrics to Weights & Biases (requires a configured W&B environment).',
      control: 'toggle',
    },
    {
      key: 'seed',
      label: 'Training seed',
      help: 'Global RNG seed for reproducible runs.',
      control: 'number',
      min: 0,
      max: 2147483647,
      step: 1,
    },
    {
      key: 'loadCheckpoint',
      label: 'Resume from checkpoint',
      help: 'Remote path to a checkpoint to resume from. Leave blank to train from scratch.',
      control: 'nullableText',
      placeholder: '/workspace/.../checkpoint',
    },
  ],
}
