import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  DEFAULT_APP_SETTINGS,
  type AppSettings,
} from '../contexts/AppSettingsContext'
import { SettingsModal } from './SettingsModal'

const contextMock = vi.hoisted(() => ({
  current: {} as Record<string, unknown>,
}))

vi.mock('../contexts/AppSettingsContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../contexts/AppSettingsContext')>()
  return {
    ...actual,
    useAppSettings: () => contextMock.current,
  }
})

vi.mock('../hooks/use-hf-auth', () => ({
  useHfAuth: () => ({
    hfAuthStatus: 'unauthenticated',
    hfAuthPolling: false,
    startHuggingFaceLogin: vi.fn(),
    handleHuggingFaceLogout: vi.fn(),
  }),
}))

vi.mock('../hooks/use-hf-model-access', () => ({
  useHfModelAccess: () => ({
    accessMap: {},
    allAuthorized: false,
  }),
}))

function makeContext(settings: AppSettings) {
  return {
    settings,
    updateSettings: vi.fn(),
    saveLtxApiKey: vi.fn(),
    saveFalApiKey: vi.fn(),
    saveGeminiApiKey: vi.fn(),
    savePexelsApiKey: vi.fn(),
    saveRunpodApiKey: vi.fn(),
    saveHfToken: vi.fn(),
    createRunpodVolume: vi.fn().mockResolvedValue({ ok: true, message: 'Created' }),
    selectRunpodVolume: vi.fn().mockResolvedValue({ ok: true, message: 'Selected' }),
    disableRunpodCache: vi.fn().mockResolvedValue({ ok: true, message: 'Disabled' }),
    relocateRunpodVolume: vi.fn().mockResolvedValue({ ok: true, message: 'Relocated' }),
    deleteRunpodVolume: vi.fn().mockResolvedValue({ ok: true, message: 'Deleted' }),
    connectRunpod: vi.fn().mockResolvedValue({
      ok: true,
      message: 'Connected',
      datacenter: 'US-TX-3',
      activeVolumeId: 'vol-primary',
      cacheEnabled: true,
      requiresVolumeSelection: false,
      regionHealth: [
        {
          datacenterId: 'US-TX-3',
          status: 'healthy',
          qualifyingGpuAvailable: true,
          availableGpuIds: ['H100'],
        },
        {
          datacenterId: 'EU-RO-1',
          status: 'healthy',
          qualifyingGpuAvailable: true,
          availableGpuIds: ['H100'],
        },
      ],
      volumes: [
        {
          id: 'vol-primary',
          name: 'ltx-desktop-lora',
          sizeGb: 500,
          datacenterId: 'US-TX-3',
          createdByApp: true,
          active: true,
          regionHealth: 'healthy',
          qualifyingGpuAvailable: true,
          availableGpuIds: ['H100'],
          savedModelReadiness: 'ready',
        },
        {
          id: 'vol-old',
          name: 'ltx-desktop-lora-old',
          sizeGb: 250,
          datacenterId: 'EU-RO-1',
          createdByApp: true,
          active: false,
          regionHealth: 'healthy',
          qualifyingGpuAvailable: true,
          availableGpuIds: ['H100'],
          savedModelReadiness: 'ready',
        },
      ],
      pods: [],
      gpus: [
        {
          id: 'A100',
          label: 'A100 80GB',
          memoryGb: 80,
          pricePerHr: 1.89,
          available: false,
        },
        {
          id: 'H100',
          label: 'H100 80GB',
          memoryGb: 80,
          pricePerHr: 2.29,
          available: true,
        },
      ],
    }),
    forceApiGenerations: true,
  }
}

beforeEach(() => {
  contextMock.current = makeContext({
    ...DEFAULT_APP_SETTINGS,
    hasRunpodApiKey: true,
    runpodGpuType: 'A100',
    runpodGpuVramGb: 80,
    runpodKeepModelCached: true,
    runpodNetworkVolumeId: 'vol-primary',
  })
  Object.defineProperty(window, 'electronAPI', {
    configurable: true,
    value: {
      hfGatingEnabled: false,
      getAnalyticsState: vi.fn().mockResolvedValue({ analyticsEnabled: false }),
      getProjectAssetsPath: vi.fn().mockResolvedValue('C:\\Projects'),
    },
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('SettingsModal RunPod regional cache', () => {
  it('keeps account storage and pods separate from per-run GPU selection', async () => {
    render(<SettingsModal isOpen onClose={vi.fn()} initialTab="loraTrainer" />)

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    expect(await screen.findByText('Saved models · US-TX-3')).toBeTruthy()
    expect(screen.getAllByText(/Models ready/)).toHaveLength(2)
    expect(screen.getByText(/about \$35(?:\.00)?\/month/i)).toBeTruthy()
    expect(screen.getByText(/GPU selection happens in the training dialog/)).toBeTruthy()
    expect(screen.queryByRole('combobox', { name: /GPU/i })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Use cache' })).toBeNull()
    expect(screen.queryByText(/Primary cache/)).toBeNull()
    expect(screen.getAllByRole('button', { name: 'Delete storage' })).toHaveLength(2)
  })

  it('uses 250 GB as the default for new caches', () => {
    expect(DEFAULT_APP_SETTINGS.runpodVolumeSizeGb).toBe(250)
  })
})
