import { describe, expect, it } from 'vitest'
import { userFacingDatasetError } from './CollectionView'

describe('userFacingDatasetError', () => {
  it('migrates stale Settings-based GPU guidance at render time', () => {
    const message = userFacingDatasetError(
      "GPU 'NVIDIA A100 80GB PCIe' is unavailable right now. "
      + 'In-stock training GPUs: none right now. '
      + 'Pick one in Settings → LoRA Trainer and retry.',
    )

    expect(message).toContain("GPU 'NVIDIA A100 80GB PCIe' is unavailable")
    expect(message).toContain('Return to GPU selection')
    expect(message).toContain('dataset and progress are preserved')
    expect(message).not.toContain('Settings')
  })
})
