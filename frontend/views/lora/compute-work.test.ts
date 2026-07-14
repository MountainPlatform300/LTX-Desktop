import { describe, expect, it } from 'vitest'
import type {
  LoraDataset,
  LoraPreprocessed,
  LoraTrainingJob,
} from '../../contexts/LoraTrainingContext'
import { derivePodWork } from './compute-work'

describe('derivePodWork', () => {
  it('uses global training jobs regardless of the open project scope', () => {
    const jobs = [
      {
        id: 'run-other-project',
        name: 'Winter',
        status: 'running',
        target: { podId: 'pod-1' },
      },
    ] as unknown as LoraTrainingJob[]

    const work = derivePodWork([], [], jobs)

    expect(work.get('pod-1')).toEqual({
      kind: 'run',
      id: 'run-other-project',
      label: 'Winter',
      stage: 'training',
    })
  })

  it('prefers a run over preprocessing or upload on a shared pod', () => {
    const datasets = [
      { id: 'dataset-1', name: 'Dataset', status: 'uploading', target: { podId: 'pod-1' } },
    ] as unknown as LoraDataset[]
    const preprocessed = [
      { id: 'prep-1', datasetId: 'dataset-1', status: 'preprocessing', target: { podId: 'pod-1' } },
    ] as unknown as LoraPreprocessed[]
    const jobs = [
      { id: 'run-1', name: 'Run', status: 'pending', target: { podId: 'pod-1' } },
    ] as unknown as LoraTrainingJob[]

    expect(derivePodWork(datasets, preprocessed, jobs).get('pod-1')?.kind).toBe('run')
  })
})
