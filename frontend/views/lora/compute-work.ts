import type {
  LoraDataset,
  LoraPreprocessed,
  LoraTrainingJob,
} from '../../contexts/LoraTrainingContext'
import type { PodWorkTarget } from './ComputePanel'

/** Account-wide mapping from paid pods to the work currently consuming them. */
export function derivePodWork(
  datasets: LoraDataset[],
  preprocessed: LoraPreprocessed[],
  trainingJobs: LoraTrainingJob[],
): Map<string, PodWorkTarget> {
  const work = new Map<string, PodWorkTarget>()
  // The most specific user-facing target wins when several ledger entities
  // share one workspace.
  for (const job of trainingJobs) {
    if (job.status !== 'running' && job.status !== 'pending') continue
    const podId = job.target?.podId
    if (!podId) continue
    work.set(podId, {
      kind: 'run',
      id: job.id,
      label: job.name,
      stage: job.status === 'running' ? 'training' : 'starting',
    })
  }
  for (const item of preprocessed) {
    if (!['pending', 'captioning', 'preprocessing'].includes(item.status)) continue
    const podId = item.target?.podId
    if (!podId || work.has(podId)) continue
    const dataset = datasets.find((candidate) => candidate.id === item.datasetId)
    work.set(podId, {
      kind: 'dataset',
      id: item.datasetId,
      label: dataset?.name ?? 'Dataset',
      stage: item.status,
    })
  }
  for (const dataset of datasets) {
    if (dataset.status !== 'uploading') continue
    const podId = dataset.target?.podId
    if (!podId || work.has(podId)) continue
    work.set(podId, {
      kind: 'dataset',
      id: dataset.id,
      label: dataset.name,
      stage: 'uploading',
    })
  }
  return work
}
