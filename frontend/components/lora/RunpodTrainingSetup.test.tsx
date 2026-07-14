import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RunpodTrainingSetup, type RunpodEstimateWorkload } from './RunpodTrainingSetup'

const getRunpodInventory = vi.fn()
const estimateRunpodTraining = vi.fn()

vi.mock('../../contexts/AppSettingsContext', () => ({
  useAppSettings: () => ({
    settings: { runpodIdleStopMinutes: 10 },
    getRunpodInventory,
    estimateRunpodTraining,
  }),
}))

const inventory = {
  message: 'Connected',
  activeVolumeId: 'volume-1',
  datacenter: 'EU-RO-1',
  cacheEnabled: true,
  savedModelReadiness: 'ready',
  estimatedModelDownloadBytes: 12_000_000_000,
  volumes: [{
    id: 'volume-1',
    name: 'models',
    sizeGb: 250,
    datacenterId: 'EU-RO-1',
    createdByApp: true,
    active: true,
    regionHealth: 'healthy',
    qualifyingGpuAvailable: true,
    availableGpuIds: ['a100', 'l40s'],
    savedModelReadiness: 'ready',
  }],
  pods: [],
  gpus: [
    { id: 'a100', label: 'A100', memoryGb: 80, pricePerHr: 1.5, available: true, activeRegionAvailable: true, availableElsewhere: true, bestAvailableRegion: 'EU-RO-1', recommended: true },
    { id: 'l40s', label: 'L40S', memoryGb: 48, pricePerHr: 0.9, available: true, activeRegionAvailable: true, availableElsewhere: false, bestAvailableRegion: 'EU-RO-1', recommended: false },
    { id: 'h200', label: 'H200', memoryGb: 141, pricePerHr: 3.5, available: true, activeRegionAvailable: false, availableElsewhere: true, bestAvailableRegion: 'US-TX-3', recommended: true },
    { id: 'h100', label: 'H100', memoryGb: 80, pricePerHr: 2.5, available: false, activeRegionAvailable: false, availableElsewhere: false, bestAvailableRegion: null, recommended: true },
  ],
}

const inputs = {
  clipCount: 4,
  totalClipSeconds: 20,
  preprocessed: false,
  mode: 'standard' as const,
  resolutionBuckets: '768x448x49',
  withAudio: false,
}

describe('RunpodTrainingSetup', () => {
  beforeEach(() => {
    getRunpodInventory.mockResolvedValue({ ok: true, data: inventory })
    estimateRunpodTraining.mockResolvedValue({
      ok: true,
      data: {
        lowSeconds: 600,
        highSeconds: 900,
        lowGpuCost: 0.25,
        highGpuCost: 0.38,
        storageMonthlyCost: 17.5,
        confidence: 'high',
        matchedHistoryCount: 3,
        downloadBytes: 0,
        phases: [{ phase: 'train', lowSeconds: 600, highSeconds: 900 }],
      },
    })
  })
  afterEach(() => cleanup())

  it('shows a full searchable grouped GPU list', async () => {
    render(<RunpodTrainingSetup value={null} onChange={() => {}} estimateInputs={inputs} />)
    expect(await screen.findByText('Recommended for this training')).toBeTruthy()
    expect(screen.getByText('Other compatible GPUs')).toBeTruthy()
    expect(screen.getByText('H100')).toBeTruthy()
    fireEvent.change(screen.getByLabelText('Search compatible GPUs'), { target: { value: 'L40' } })
    expect(screen.getByText('L40S')).toBeTruthy()
    expect(screen.queryByText('A100')).toBeNull()
  })

  it('labels cache-compatible and other global stock honestly', async () => {
    render(<RunpodTrainingSetup value={null} onChange={() => {}} estimateInputs={inputs} />)
    expect((await screen.findAllByText('No model download required · Cache in EU-RO-1')).length).toBeGreaterThan(0)
    expect(screen.getByText('Available in US-TX-3 · Model download required')).toBeTruthy()
    expect(screen.getAllByLabelText('About No model download required · Cache in EU-RO-1').length).toBeGreaterThan(0)
  })

  it('keeps an available-elsewhere GPU selectable', async () => {
    const onChange = vi.fn()
    render(<RunpodTrainingSetup value={null} onChange={onChange} estimateInputs={inputs} />)
    await screen.findByText('Available in US-TX-3 · Model download required')
    fireEvent.click(screen.getByRole('button', { name: /H200/ }))
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      gpuType: 'h200',
      datacenter: 'US-TX-3',
      workspacePolicy: 'ephemeral_any_region',
      volumeId: null,
    }))
  })

  it('automatically associates compatible ready storage', async () => {
    const onChange = vi.fn()
    render(<RunpodTrainingSetup value={null} onChange={onChange} estimateInputs={inputs} />)
    await screen.findByText('A100')
    fireEvent.click(screen.getByRole('button', { name: /A100/ }))
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      gpuType: 'a100',
      datacenter: 'EU-RO-1',
      workspacePolicy: 'primary_cache',
      volumeId: 'volume-1',
    }))
    expect(screen.queryByRole('radio')).toBeNull()
  })

  it('separates GPUs that are likely to OOM and requires expert acknowledgement', async () => {
    const onChange = vi.fn()
    const onAllowUnsafeOverrideChange = vi.fn()
    const riskyInputs = {
      ...inputs,
      config: {
        preset: 'standard' as const,
        rank: 32,
        batchSize: 1,
        loadTextEncoderIn8bit: false,
        offloadOptimizerDuringValidation: false,
        skipInitialValidation: false,
      } as NonNullable<RunpodEstimateWorkload['config']>,
    }
    const { rerender } = render(
      <RunpodTrainingSetup
        value={null}
        onChange={onChange}
        estimateInputs={riskyInputs}
        onAllowUnsafeOverrideChange={onAllowUnsafeOverrideChange}
      />,
    )
    expect(await screen.findByText('Not compatible with selected profile')).toBeTruthy()
    expect(screen.getAllByText('Likely to run out of memory with selected profile')).not.toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: /L40S/ }))
    const selection = onChange.mock.calls[0][0]
    rerender(
      <RunpodTrainingSetup
        value={selection}
        onChange={onChange}
        estimateInputs={riskyInputs}
        onAllowUnsafeOverrideChange={onAllowUnsafeOverrideChange}
      />,
    )
    fireEvent.click(screen.getByRole('checkbox', { name: /Expert override/ }))
    expect(onAllowUnsafeOverrideChange).toHaveBeenCalledWith(true)
  })

  it('refreshes stock on mount and on demand', async () => {
    render(<RunpodTrainingSetup value={null} onChange={() => {}} estimateInputs={inputs} />)
    await waitFor(() => expect(getRunpodInventory).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByLabelText('Refresh RunPod inventory'))
    await waitFor(() => expect(getRunpodInventory).toHaveBeenCalledTimes(2))
  })

  it('uses the backend estimate with GPU and readiness inputs', async () => {
    const onChange = vi.fn()
    const { rerender } = render(<RunpodTrainingSetup value={null} onChange={onChange} estimateInputs={inputs} />)
    await screen.findByText('A100')
    fireEvent.click(screen.getByRole('button', { name: /A100/ }))
    const selection = onChange.mock.calls[0][0]
    rerender(<RunpodTrainingSetup value={selection} onChange={onChange} estimateInputs={inputs} />)
    await waitFor(() => expect(estimateRunpodTraining).toHaveBeenCalledWith(expect.objectContaining({
      gpuType: 'a100',
      gpuVramGb: 80,
      gpuPricePerHr: 1.5,
      storageReadiness: 'ready',
      estimatedModelDownloadBytes: 12_000_000_000,
      idleTimeoutMinutes: 10,
      storageSizeGb: 250,
    })))
    expect(await screen.findByText('High confidence · 3 similar runs')).toBeTruthy()
  })

  it('labels the client estimate when the endpoint fails', async () => {
    estimateRunpodTraining.mockResolvedValueOnce({
      ok: false,
      error: { code: 'RUNPOD_ESTIMATE_FAILED', message: 'offline' },
    })
    const selection = {
      gpuType: 'a100',
      gpuVramGb: 80,
      datacenter: 'EU-RO-1',
      workspacePolicy: 'primary_cache' as const,
      volumeId: 'volume-1',
    }
    render(<RunpodTrainingSetup value={selection} onChange={() => {}} estimateInputs={inputs} />)
    expect(await screen.findByText(/fallback low-confidence planning range/i)).toBeTruthy()
  })
})
