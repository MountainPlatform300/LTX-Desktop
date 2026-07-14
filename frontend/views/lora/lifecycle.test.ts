import { describe, expect, it } from 'vitest'
import type {
  LoraDataset,
  LoraPreprocessed,
  LoraTrainingJob,
} from '../../contexts/LoraTrainingContext'
import { deriveLifecycle } from './lifecycle'

function dataset(overrides: Partial<LoraDataset> = {}): LoraDataset {
  return {
    id: 'dataset-1',
    status: 'draft',
    cancelRequested: false,
    ...overrides,
  } as unknown as LoraDataset
}

function preprocessed(
  overrides: Partial<LoraPreprocessed> = {},
): LoraPreprocessed {
  return {
    id: 'prep-1',
    datasetId: 'dataset-1',
    createdAt: '2026-01-01T00:00:00Z',
    status: 'ready',
    ...overrides,
  } as unknown as LoraPreprocessed
}

function training(
  overrides: Partial<LoraTrainingJob> = {},
): LoraTrainingJob {
  return {
    id: 'training-1',
    preprocessedId: 'prep-1',
    createdAt: '2026-01-01T00:00:00Z',
    status: 'running',
    ...overrides,
  } as unknown as LoraTrainingJob
}

describe('LoRA lifecycle', () => {
  it('surfaces a running training phase, progress, and ETA', () => {
    const lifecycle = deriveLifecycle(
      dataset(),
      [preprocessed()],
      [
        training({
          statusDetail: 'Training',
          currentStep: 25,
          totalSteps: 100,
          etaSeconds: 90,
        }),
      ],
    )

    expect(lifecycle.stage).toBe('training')
    expect(lifecycle.detail).toBe('Training')
    expect(lifecycle.percent).toBe(25)
    expect(lifecycle.etaSeconds).toBe(90)
    expect(lifecycle.primary?.kind).toBe('view-run')
  })

  it('surfaces upload sub-phase progress', () => {
    const lifecycle = deriveLifecycle(
      dataset({
        status: 'uploading',
        statusDetail: 'Downloading model',
        statusPercent: 42,
        statusEtaSeconds: 30,
      }),
      [],
      [],
    )

    expect(lifecycle.stage).toBe('uploading')
    expect(lifecycle.detail).toBe('Downloading model')
    expect(lifecycle.percent).toBe(42)
    expect(lifecycle.etaSeconds).toBe(30)
    expect(lifecycle.busy).toBe(true)
  })

  it('uses provider-aware queued wording', () => {
    const runpod = deriveLifecycle(
      dataset(),
      [preprocessed()],
      [training({ status: 'pending', provider: 'runpod', statusDetail: null })],
    )
    const local = deriveLifecycle(
      dataset(),
      [preprocessed()],
      [training({ status: 'pending', provider: 'local', statusDetail: null })],
    )

    expect(runpod.detail).toBe('Waiting to start on RunPod…')
    expect(local.detail).toBe('Waiting to start on this computer…')
  })

  it('does not describe IC-LoRA preprocessing as captioning', () => {
    const lifecycle = deriveLifecycle(
      dataset({ type: 'ic_lora' }),
      [preprocessed({ status: 'preprocessing' })],
      [],
    )

    expect(lifecycle.detail).toBe('Encoding input/output pairs…')
  })

  it('surfaces GPU reselection instead of falling back to ready', () => {
    const lifecycle = deriveLifecycle(
      dataset({ status: 'uploaded' }),
      [preprocessed()],
      [training({ status: 'gpu_selection_required', provider: 'runpod' })],
    )

    expect(lifecycle.label).toBe('GPU unavailable')
    expect(lifecycle.detail).toContain('Choose another RunPod GPU')
  })

  it('routes dataset-level GPU recovery to the selector instead of upload confirmation', () => {
    const lifecycle = deriveLifecycle(
      dataset({
        status: 'gpu_selection_required',
        statusDetail: 'Selected GPU is out of stock.',
      }),
      [],
      [],
    )

    expect(lifecycle.primary).toEqual({
      kind: 'recover-gpu',
      label: 'Choose another GPU',
      needsCredentials: true,
    })
  })

  it('uses the newest completed run as the terminal result', () => {
    const lifecycle = deriveLifecycle(
      dataset({ status: 'uploaded' }),
      [preprocessed()],
      [
        training({
          id: 'old-failure',
          createdAt: '2026-01-01T00:00:00Z',
          status: 'failed',
        }),
        training({
          id: 'new-success',
          createdAt: '2026-01-02T00:00:00Z',
          status: 'completed',
        }),
      ],
    )

    expect(lifecycle.stage).toBe('trained')
    expect(lifecycle.training?.id).toBe('new-success')
    expect(lifecycle.primary?.kind).toBe('use-lora')
  })
})
