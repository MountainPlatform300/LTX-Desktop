import { AlertCircle, Check, ChevronDown, Cloud, Download, Film, Folder, HardDrive, Info, KeyRound, Layers, Loader2, Settings, Sparkles, X, Zap } from 'lucide-react'
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from './ui/button'
import { useAppSettings, type AppSettings, type ConnectRunpodResult } from '../contexts/AppSettingsContext'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import {
  estimateRunpodStorageMonthlyUsd,
  formatRunpodStorageMonthlyUsd,
} from '../lib/runpod-storage'
import { ApiKeyHelperRow, LtxApiKeyInput, LtxApiKeyHelperRow } from './LtxApiKeyInput'
import { useHfAuth } from '../hooks/use-hf-auth'
import { useHfModelAccess } from '../hooks/use-hf-model-access'
import { confirmAction } from './ui/confirm-dialog'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  initialTab?: TabId
}

type TabId = 'general' | 'apiKeys' | 'promptEnhancer' | 'loraTrainer' | 'about'

export function SettingsModal({ isOpen, onClose, initialTab }: SettingsModalProps) {
  const {
    settings,
    updateSettings,
    saveLtxApiKey,
    saveFalApiKey,
    saveGeminiApiKey,
    savePexelsApiKey,
    saveRunpodApiKey,
    saveHfToken,
    connectRunpod,
    createRunpodVolume,
    deleteRunpodVolume,
    forceApiGenerations,
  } = useAppSettings()
  const onSettingsChange = (next: AppSettings) => updateSettings(next)
  const [activeTab, setActiveTab] = useState<TabId>('general')
  const [ltxApiKeyInput, setLtxApiKeyInput] = useState('')
  const ltxApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [focusLtxApiKeyInputOnTabChange, setFocusLtxApiKeyInputOnTabChange] = useState(false)
  const [falApiKeyInput, setFalApiKeyInput] = useState('')
  const falApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [geminiApiKeyInput, setGeminiApiKeyInput] = useState('')
  const geminiApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [pexelsApiKeyInput, setPexelsApiKeyInput] = useState('')
  const [runpodApiKeyInput, setRunpodApiKeyInput] = useState('')
  const [hfTokenInput, setHfTokenInput] = useState('')
  const [testConnectionState, setTestConnectionState] = useState<
    { status: 'idle' } | { status: 'testing' } | { status: 'done'; ok: boolean; message: string }
  >({ status: 'idle' })
  const [runpodConnectState, setRunpodConnectState] = useState<
    | { status: 'idle' }
    | { status: 'connecting' }
    | {
        status: 'connected'
        gpus: NonNullable<ConnectRunpodResult['gpus']>
        volumes: NonNullable<ConnectRunpodResult['volumes']>
        pods: NonNullable<ConnectRunpodResult['pods']>
        activeVolumeId: string | null
        datacenter: string
        cacheEnabled: boolean
        requiresVolumeSelection: boolean
        regionHealth: NonNullable<ConnectRunpodResult['regionHealth']>
      }
    | { status: 'error'; message: string }
  >({ status: 'idle' })
  const [terminatingPodId, setTerminatingPodId] = useState<string | null>(null)
  const [volumeActionState, setVolumeActionState] = useState<
    | { status: 'idle' }
    | { status: 'working'; action: string }
    | { status: 'done'; ok: boolean; message: string }
  >({ status: 'idle' })
  const [showRunpodAdvanced, setShowRunpodAdvanced] = useState(false)
  const [textEncoderRecommendation, setTextEncoderRecommendation] = useState<ApiSuccessOf<'getTextEncoderRecommendation'> | null>(null)
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<ApiSuccessOf<'getModelDownloadProgress'> | null>(null)
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin, handleHuggingFaceLogout } = useHfAuth(isOpen)
  const textEncoderModelTypes = useMemo(
    () => (forceApiGenerations || !textEncoderRecommendation?.cp_to_download
      ? []
      : [textEncoderRecommendation.cp_to_download]),
    [forceApiGenerations, textEncoderRecommendation?.cp_to_download],
  )
  const { accessMap: teAccessMap, allAuthorized: teAllAuthorized } = useHfModelAccess(textEncoderModelTypes, hfAuthStatus)
  const [appVersion, setAppVersion] = useState('')
  const [noticesText, setNoticesText] = useState<string | null>(null)
  const [noticesLoading, setNoticesLoading] = useState(false)
  const [showNotices, setShowNotices] = useState(false)
  const [modelLicenseText, setModelLicenseText] = useState<string | null>(null)
  const [modelLicenseLoading, setModelLicenseLoading] = useState(false)
  const [showModelLicense, setShowModelLicense] = useState(false)
  const [analyticsEnabled, setAnalyticsEnabled] = useState(false)
  const [projectAssetsPath, setProjectAssetsPath] = useState('')
  const managedCacheVolumes = runpodConnectState.status === 'connected'
    ? runpodConnectState.volumes.filter((volume) => volume.createdByApp)
    : []
  const cacheVolumeSizeGb = settings.runpodVolumeSizeGb
  const newCacheMonthlyEstimate = estimateRunpodStorageMonthlyUsd(settings.runpodVolumeSizeGb)

  // Sync active tab with initialTab prop when modal opens
  useEffect(() => {
    if (isOpen && initialTab) {
      setActiveTab(initialTab)
    }
  }, [isOpen, initialTab])

  useEffect(() => {
    if (!isOpen || activeTab !== 'apiKeys' || !focusLtxApiKeyInputOnTabChange) return

    const frameId = window.requestAnimationFrame(() => {
      ltxApiKeyInputRef.current?.focus()
    })
    setFocusLtxApiKeyInputOnTabChange(false)

    return () => {
      window.cancelAnimationFrame(frameId)
    }
  }, [activeTab, focusLtxApiKeyInputOnTabChange, isOpen])

  // Fetch app version when About tab is shown
  useEffect(() => {
    if (activeTab !== 'about' || appVersion) return
    window.electronAPI.getAppInfo().then(info => setAppVersion(info.version)).catch(() => {})
  }, [activeTab, appVersion])

  // Fetch analytics state when modal opens
  useEffect(() => {
    if (!isOpen) return
    window.electronAPI.getAnalyticsState()
      .then((state: { analyticsEnabled: boolean }) => setAnalyticsEnabled(state.analyticsEnabled))
      .catch(() => {})
    window.electronAPI.getProjectAssetsPath()
      .then((p: string) => setProjectAssetsPath(p))
      .catch(() => {})
  }, [isOpen])

  // Fetch text encoder recommendation when modal opens
  useEffect(() => {
    if (!isOpen || forceApiGenerations) return

    const fetchRecommendation = async () => {
      const result = await ApiClient.getTextEncoderRecommendation()
      if (!result.ok) {
        logger.error(`Failed to fetch text encoder recommendation: ${result.error.message}`)
        return
      }

      const data = result.data
      setTextEncoderRecommendation(data)
      if (data.cp_to_download === null) {
        setIsDownloading(false)
      }
    }

    void fetchRecommendation()
  }, [forceApiGenerations, isOpen])

  // Poll download progress via session ID
  useEffect(() => {
    if (!isDownloading || !downloadSessionId) return

    const poll = async () => {
      const result = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (!result.ok) return
      setDownloadProgress(result.data)
      if (result.data.status === 'complete') {
        setIsDownloading(false)
        setDownloadSessionId(null)
        const rec = await ApiClient.getTextEncoderRecommendation()
        if (rec.ok) setTextEncoderRecommendation(rec.data)
      } else if (result.data.status === 'error') {
        setDownloadError(result.data.error ?? 'Download failed')
        setIsDownloading(false)
        setDownloadSessionId(null)
      }
    }

    void poll()
    const interval = setInterval(() => { void poll() }, 1000)
    return () => clearInterval(interval)
  }, [isDownloading, downloadSessionId])

  // Handle text encoder download
  const handleDownloadTextEncoder = async () => {
    if (!textEncoderRecommendation?.cp_to_download) return
    setIsDownloading(true)
    setDownloadError(null)
    setDownloadProgress(null)
    const result = await ApiClient.startModelDownload({
      type: 'download',
      cp_ids: [textEncoderRecommendation.cp_to_download],
    })
    if (!result.ok) {
      setDownloadError(result.error.message)
      setIsDownloading(false)
      return
    }
    if (result.data.status === 'started') {
      setDownloadSessionId(result.data.sessionId)
    }
  }

  if (!isOpen) return null

  const handleToggleTorchCompile = () => {
    onSettingsChange({
      ...settings,
      useTorchCompile: !settings.useTorchCompile,
    })
  }

  const handleToggleLocalEncoder = () => {
    onSettingsChange({
      ...settings,
      useLocalTextEncoder: !settings.useLocalTextEncoder,
    })
  }

  const handleToggleDevQualityBase = () => {
    onSettingsChange({
      ...settings,
      useDevQualityBase: !settings.useDevQualityBase,
    })
  }

  const openApiKeysAndFocusLtxInput = () => {
    setActiveTab('apiKeys')
    setFocusLtxApiKeyInputOnTabChange(true)
  }

  const handlePromptCacheSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const size = Math.max(0, Math.min(1000, parseInt(e.target.value) || 100))
    onSettingsChange({
      ...settings,
      promptCacheSize: size,
    })
  }

  // Prompt Enhancer handlers
  const handleTogglePromptEnhancer = (mode: 't2v' | 'i2v') => {
    if (mode === 't2v') {
      onSettingsChange({ ...settings, promptEnhancerEnabledT2V: !settings.promptEnhancerEnabledT2V })
    } else {
      onSettingsChange({ ...settings, promptEnhancerEnabledI2V: !settings.promptEnhancerEnabledI2V })
    }
  }
  // Analytics handler
  const handleToggleAnalytics = () => {
    const next = !analyticsEnabled
    setAnalyticsEnabled(next)
    window.electronAPI.setAnalyticsEnabled({ enabled: next }).catch(() => {})
  }

  // Seed handlers
  const handleToggleSeedLock = () => {
    onSettingsChange({
      ...settings,
      seedLocked: !settings.seedLocked,
    })
  }

  const handleLockedSeedChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 0
    onSettingsChange({
      ...settings,
      lockedSeed: Math.max(0, Math.min(2147483647, value)),
    })
  }

  const handleRandomizeSeed = () => {
    onSettingsChange({
      ...settings,
      lockedSeed: Math.floor(Math.random() * 2147483647),
    })
  }

  const handleLoadModelLicense = async () => {
    setModelLicenseLoading(true)
    try {
      const text = await window.electronAPI.fetchLicenseText()
      setModelLicenseText(text)
      setShowModelLicense(true)
    } catch (e) {
      logger.error(`Failed to load model license: ${e}`)
    } finally {
      setModelLicenseLoading(false)
    }
  }

  const handleLoadNotices = async () => {
    setNoticesLoading(true)
    try {
      const text = await window.electronAPI.getNoticesText()
      setNoticesText(text)
      setShowNotices(true)
    } catch (e) {
      logger.error(`Failed to load notices: ${e}`)
    } finally {
      setNoticesLoading(false)
    }
  }

  const handleTestLoraConnection = async () => {
    setTestConnectionState({ status: 'testing' })
    const result = await ApiClient.testLoraConnection({ provider: 'runpod' })
    if (!result.ok) {
      setTestConnectionState({ status: 'done', ok: false, message: result.error.message })
      return
    }
    setTestConnectionState({ status: 'done', ok: result.data.ok, message: result.data.message })
  }

  // One-click RunPod connect: save the typed key (if any), validate it, and
  // pull back the account's GPUs + volumes to drive the picker. Auto-selects
  // a recommended GPU when none is chosen yet.
  const handleConnectRunpod = async () => {
    setRunpodConnectState({ status: 'connecting' })
    try {
      const trimmed = runpodApiKeyInput.trim()
      if (trimmed) {
        await saveRunpodApiKey(trimmed)
        setRunpodApiKeyInput('')
      }
      // Legacy settings could have caching enabled without a selected volume.
      // Clear that state before discovery so Connect can never create a paid
      // resource implicitly; the explicit Create storage button owns creation.
      if (settings.runpodKeepModelCached && !settings.runpodNetworkVolumeId) {
        const disabled = await ApiClient.updateSettings({ runpodKeepModelCached: false })
        if (!disabled.ok) throw new Error(disabled.error.message)
        updateSettings({ runpodKeepModelCached: false })
      }
      const data = await connectRunpod()
      if (!data.ok) {
        setRunpodConnectState({ status: 'error', message: data.message })
        return
      }
      const gpus = data.gpus ?? []
      if (!settings.runpodGpuType && gpus.length > 0) {
        // Auto-select the cheapest in-stock GPU that can train (the list is
        // already filtered to >=32GB). Visible + overridable, not a silent
        // switch. GPUs without a price sort last.
        const available = gpus.filter((g) => g.available)
        const pool = available.length > 0 ? available : gpus
        const cheapest = [...pool].sort(
          (a, b) =>
            (a.pricePerHr ?? Number.POSITIVE_INFINITY) - (b.pricePerHr ?? Number.POSITIVE_INFINITY) ||
            a.memoryGb - b.memoryGb,
        )[0]
        if (cheapest)
          updateSettings({ runpodGpuType: cheapest.id, runpodGpuVramGb: cheapest.memoryGb ?? 0 })
      }
      setRunpodConnectState({
        status: 'connected',
        gpus,
        volumes: data.volumes ?? [],
        pods: data.pods ?? [],
        activeVolumeId: data.activeVolumeId ?? null,
        datacenter: data.datacenter ?? '',
        cacheEnabled: data.cacheEnabled ?? false,
        requiresVolumeSelection: data.requiresVolumeSelection ?? false,
        regionHealth: data.regionHealth ?? [],
      })
    } catch (err) {
      setRunpodConnectState({
        status: 'error',
        message: err instanceof Error ? err.message : 'Connection failed',
      })
    }
  }

  const runVolumeAction = async (
    action: string,
    operation: () => Promise<{ message: string }>,
  ) => {
    setVolumeActionState({ status: 'working', action })
    try {
      const result = await operation()
      await handleConnectRunpod()
      setVolumeActionState({ status: 'done', ok: true, message: result.message })
    } catch (error) {
      const rawMessage = error instanceof Error ? error.message : `${action} failed`
      const message = /^not found$/i.test(rawMessage.trim())
        ? 'Restart LTX Desktop once to load the new RunPod volume controls.'
        : rawMessage
      setVolumeActionState({
        status: 'done',
        ok: false,
        message,
      })
    }
  }

  const handleTerminatePod = async (podId: string) => {
    setTerminatingPodId(podId)
    try {
      await ApiClient.terminateRunpodPod(podId)
      // Refresh the connect snapshot so the terminated pod drops off the list.
      await handleConnectRunpod()
    } finally {
      setTerminatingPodId(null)
    }
  }

  const tabs = [
    { id: 'general' as TabId, label: 'General', icon: Settings },
    { id: 'apiKeys' as TabId, label: 'API Keys', icon: KeyRound },
    { id: 'promptEnhancer' as TabId, label: 'Prompt Enhancer', icon: Sparkles },
    { id: 'loraTrainer' as TabId, label: 'LoRA Trainer', icon: Cloud },
    { id: 'about' as TabId, label: 'About', icon: Info },
  ]

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-xl mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-zinc-400" />
            <h2 className="text-lg font-semibold text-white">Settings</h2>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-8 w-8 text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-zinc-800">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-white border-b-2 border-blue-500 -mb-px'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* Content */}
        <div className="px-6 py-5 space-y-6 h-[60vh] overflow-y-auto">
          {activeTab === 'general' && (
            <>
              {/* Project Assets Path */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Project Assets Path</h3>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  Where generated video and image assets are saved. Each project gets a subfolder.
                </p>
                <div className="flex gap-2">
                  <div className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 text-sm truncate select-text">
                    {projectAssetsPath || <span className="text-zinc-600">Not set</span>}
                  </div>
                  <Button
                    variant="outline"
                    className="border-zinc-700 flex-shrink-0"
                    onClick={async () => {
                      const result = await window.electronAPI.openProjectAssetsPathChangeDialog()
                      if (result.success) {
                        setProjectAssetsPath(result.path)
                      }
                    }}
                  >
                    <Folder className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {!forceApiGenerations && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Film className="h-4 w-4 text-blue-400" />
                    <h3 className="text-sm font-semibold text-white">Videos Generation</h3>
                  </div>

                  <div
                    className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                      settings.userPrefersLtxApiVideoGenerations ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                    }`}
                    onClick={() => {
                      if (!settings.hasLtxApiKey) {
                        openApiKeysAndFocusLtxInput()
                        return
                      }
                      onSettingsChange({
                        ...settings,
                        userPrefersLtxApiVideoGenerations: !settings.userPrefersLtxApiVideoGenerations,
                      })
                    }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4 text-blue-400" />
                          <span className="text-sm font-medium text-white">Generate With API</span>
                        </div>
                        <p className="text-xs text-zinc-400 mt-1">
                          Use LTX API for video generation when an LTX API key is configured.
                        </p>
                      </div>
                      <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                        settings.userPrefersLtxApiVideoGenerations ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                      }`}>
                        {settings.userPrefersLtxApiVideoGenerations && <Check className="h-3 w-3 text-white" />}
                      </div>
                    </div>

                    {!settings.hasLtxApiKey && (
                      <div className="mt-2 text-xs text-amber-400 flex items-center gap-1.5">
                        <AlertCircle className="h-3 w-3" />
                        API key required — configure it in the API Keys tab.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Text Encoding Section */}
              {!forceApiGenerations && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M15 7h3a5 5 0 0 1 5 5 5 5 0 0 1-5 5h-3m-6 0H6a5 5 0 0 1-5-5 5 5 0 0 1 5-5h3" />
                    <line x1="8" y1="12" x2="16" y2="12" />
                  </svg>
                  <h3 className="text-sm font-semibold text-white">Text Encoding</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Text encoding converts your prompt into data the AI understands. Choose how to do this.
                </p>

                {/* LTX API Option (Default) */}
                <div
                  className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                    !settings.useLocalTextEncoder ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                  }`}
                  onClick={() => {
                    if (!settings.useLocalTextEncoder) return
                    if (!settings.hasLtxApiKey) {
                      openApiKeysAndFocusLtxInput()
                      return
                    }
                    handleToggleLocalEncoder()
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Zap className="h-4 w-4 text-blue-400" />
                        <span className="text-sm font-medium text-white">LTX API</span>
                        <span className="text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">Recommended</span>
                      </div>
                      <p className="text-xs text-zinc-400 mt-1">
                        Fast cloud-based text encoding (~1 second). Requires an LTX API key configured in the API Keys tab.
                      </p>
                    </div>
                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                      !settings.useLocalTextEncoder ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                    }`}>
                      {!settings.useLocalTextEncoder && <Check className="h-3 w-3 text-white" />}
                    </div>
                  </div>

                  {/* Warning when selected but no key */}
                  {!settings.useLocalTextEncoder && !settings.hasLtxApiKey && (
                    <div className="mt-2 text-xs text-amber-400 flex items-center gap-1.5">
                      <AlertCircle className="h-3 w-3" />
                      API key required — configure it in the API Keys tab.
                    </div>
                  )}

                  {/* Prompt Cache Size — only relevant for API text encoding */}
                  {!settings.useLocalTextEncoder && settings.hasLtxApiKey && (
                    <div className="flex items-center justify-between mt-3 pt-3 border-t border-zinc-700/50">
                      <div>
                        <label className="text-xs text-white">Prompt Cache</label>
                        <p className="text-xs text-zinc-500">Skip repeat encoding calls</p>
                      </div>
                      <input
                        type="number"
                        min="0"
                        max="1000"
                        value={settings.promptCacheSize ?? 100}
                        onChange={handlePromptCacheSizeChange}
                        onClick={(e) => e.stopPropagation()}
                        className="w-16 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded text-xs text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  )}
                </div>

                {/* Local Encoder Option */}
                <div
                  className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                    settings.useLocalTextEncoder ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                  }`}
                  onClick={() => !settings.useLocalTextEncoder && handleToggleLocalEncoder()}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <svg className="h-4 w-4 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <rect x="4" y="4" width="16" height="16" rx="2" />
                          <path d="M9 9h6m-6 3h6m-6 3h4" />
                        </svg>
                        <span className="text-sm font-medium text-white">Local Encoder</span>
                      </div>
                      <p className="text-xs text-zinc-400 mt-1">
                        Run on your computer (~23 seconds). Requires 25 GB download.
                      </p>
                    </div>
                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                      settings.useLocalTextEncoder ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                    }`}>
                      {settings.useLocalTextEncoder && <Check className="h-3 w-3 text-white" />}
                    </div>
                  </div>

                  {/* Download Status - show when this option is selected */}
                  {settings.useLocalTextEncoder && (
                    <div className="mt-3 pt-3 border-t border-zinc-700/50">
                      {textEncoderRecommendation?.cp_to_download === null ? (
                        <div className="flex items-center gap-2 text-xs text-green-400">
                          <Check className="h-4 w-4" />
                          <span>Downloaded ({textEncoderRecommendation?.expected_size_gb ?? 0} GB)</span>
                        </div>
                      ) : isDownloading ? (
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-zinc-300">Downloading text encoder...</span>
                            <span className="text-zinc-500">{downloadProgress?.status === 'downloading' ? Math.round(downloadProgress.current_file_progress) : 0}%</span>
                          </div>
                          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div className="h-full transition-all duration-300 bg-blue-500" style={{ width: `${downloadProgress?.status === 'downloading' ? downloadProgress.current_file_progress : 0}%` }} />
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-xs text-amber-400">
                            <AlertCircle className="h-4 w-4" />
                            <span>Not downloaded ({textEncoderRecommendation?.expected_size_gb || 0} GB required)</span>
                          </div>
                          {hfAuthStatus === 'authenticated' && !teAllAuthorized && Object.keys(teAccessMap).length > 0 && (
                            <div className="space-y-1.5 mb-2">
                              {Object.entries(teAccessMap)
                                .filter(([, status]) => status === 'not_authorized')
                                .map(([repoId]) => (
                                  <div key={repoId} className="flex items-center justify-between bg-zinc-900 rounded px-2 py-1.5">
                                    <span className="text-[10px] text-zinc-400 font-mono">{repoId}</span>
                                    <button
                                      onClick={(e) => { e.stopPropagation(); window.electronAPI.openHuggingFaceRepo({ repoId }) }}
                                      className="text-[10px] text-indigo-400 hover:text-indigo-300 font-medium"
                                    >
                                      Request access
                                    </button>
                                  </div>
                                ))}
                            </div>
                          )}
                          <Button
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              void handleDownloadTextEncoder()
                            }}
                            disabled={!textEncoderRecommendation?.cp_to_download || !teAllAuthorized || hfAuthStatus !== 'authenticated'}
                            className="w-full bg-blue-600 hover:bg-blue-500 text-white text-xs"
                          >
                            <Download className="h-3 w-3 mr-2" />
                            Download Text Encoder
                          </Button>
                          {downloadError && (
                            <p className="text-xs text-red-400">{downloadError}</p>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              )}

              {/* IC-LoRA Quality Base (opt-in dev + distilled-LoRA overlay) */}
              {!forceApiGenerations && (
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Layers className="h-4 w-4 text-blue-400" />
                      <h3 className="text-sm font-semibold text-white">IC-LoRA Quality Base</h3>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Use the full dev checkpoint with the distilled v1.1 LoRA stacked at 0.5 for IC-LoRA
                      generations — the higher-quality base from the ComfyUI flow. Off by default: adds ~54 GB
                      of optional downloads (dev checkpoint + distilled LoRA). IC-LoRA falls back to the
                      distilled checkpoint until both are downloaded.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleToggleDevQualityBase}
                    className={`relative shrink-0 w-10 h-6 rounded-full transition-colors ${
                      settings.useDevQualityBase ? 'bg-blue-600' : 'bg-zinc-700'
                    }`}
                    aria-pressed={settings.useDevQualityBase}
                    title="Toggle the dev + distilled-LoRA IC-LoRA quality base"
                  >
                    <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                      settings.useDevQualityBase ? 'translate-x-4' : ''
                    }`} />
                  </button>
                </div>
                {settings.useDevQualityBase && (
                  <div className="flex items-center gap-1.5 text-xs text-amber-400">
                    <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                    <span>Enable the toggle, then open the model manager to download the dev checkpoint + distilled LoRA (~54 GB).</span>
                  </div>
                )}
              </div>
              )}

              {/* Torch Compile Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-orange-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Torch Compile
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Compiles the model for optimized inference. <span className="text-orange-400">Experimental:</span> First
                      generation can take 5-10+ minutes for compilation. Subsequent generations may be
                      20-40% faster. Requires app restart to take effect.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleTorchCompile}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.useTorchCompile ? 'bg-orange-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.useTorchCompile ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.useTorchCompile
                    ? 'bg-orange-500/10 text-orange-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.useTorchCompile ? 'bg-orange-400' : 'bg-zinc-600'
                  }`} />
                  {settings.useTorchCompile ? 'Optimized inference (recommended)' : 'Standard inference'}
                </div>
              </div>

              {/* Seed Lock Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Lock Seed
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Use the same seed for reproducible generations. When unlocked, a random seed is used each time.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleSeedLock}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.seedLocked ? 'bg-emerald-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.seedLocked ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Seed input - only show when locked */}
                {settings.seedLocked && (
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      value={settings.lockedSeed ?? 42}
                      onChange={handleLockedSeedChange}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      placeholder="Enter seed..."
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleRandomizeSeed}
                      className="h-9 px-3 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                      title="Generate random seed"
                    >
                      <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
                      </svg>
                    </Button>
                  </div>
                )}

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.seedLocked
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.seedLocked ? 'bg-emerald-400' : 'bg-zinc-600'
                  }`} />
                  {settings.seedLocked ? `Seed locked: ${settings.lockedSeed ?? 42}` : 'Random seed each generation'}
                </div>
              </div>

              {/* Usage Analytics Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10" />
                        <line x1="12" y1="20" x2="12" y2="4" />
                        <line x1="6" y1="20" x2="6" y2="14" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Usage Analytics
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Share pseudonymous app-launch data with Lightricks.
                      The payload includes a random installation ID and basic technical and fork information — never prompts, credentials, paths, or generated content.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleAnalytics}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      analyticsEnabled ? 'bg-violet-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        analyticsEnabled ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

              </div>
            </>
          )}

          {activeTab === 'apiKeys' && (
            <>
              {/* LTX API Key Section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">LTX API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your LTX API key is used for cloud text encoding, prompt enhancement, and API video generation.
                  Add your key below to unlock these features.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={ltxApiKeyInputRef}
                      value={ltxApiKeyInput}
                      onChange={(e) => setLtxApiKeyInput(e.target.value)}
                      placeholder={settings.hasLtxApiKey ? 'Enter new key to replace...' : 'Enter your LTX API key...'}
                      stopPropagation
                      className="flex-1"
                    />
                    <button
                      onClick={() => {
                        const trimmed = ltxApiKeyInput.trim()
                        if (!trimmed) return
                        void saveLtxApiKey(trimmed)
                        setLtxApiKeyInput('')
                      }}
                      disabled={!ltxApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <LtxApiKeyHelperRow stopPropagation />
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasLtxApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-amber-500/10 text-amber-400'
                    }`}>
                      {settings.hasLtxApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          API key required
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* FAL API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <KeyRound className="h-4 w-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">FAL AI</h3>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">Optional</span>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your FAL AI key is used for generating images with Z Image Turbo when API generations are enabled.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={falApiKeyInputRef}
                      value={falApiKeyInput}
                      onChange={(e) => setFalApiKeyInput(e.target.value)}
                      placeholder={settings.hasFalApiKey ? 'Enter new key to replace...' : 'Enter your FAL AI API key...'}
                      stopPropagation
                      className="flex-1"
                    />
                    <button
                      onClick={() => {
                        const trimmed = falApiKeyInput.trim()
                        if (!trimmed) return
                        void saveFalApiKey(trimmed)
                        setFalApiKeyInput('')
                      }}
                      disabled={!falApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <ApiKeyHelperRow
                    stopPropagation
                    label="Get FAL API key"
                    onOpenKey={() => window.electronAPI.openFalApiKeyPage()}
                  />
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasFalApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-zinc-800 text-zinc-500'
                    }`}>
                      {settings.hasFalApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          Optional
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Gemini API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-white">Gemini API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your Gemini API key is used for AI-powered prompt suggestions when filling timeline gaps.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <input
                      ref={geminiApiKeyInputRef}
                      type="password"
                      value={geminiApiKeyInput}
                      onChange={(e) => setGeminiApiKeyInput(e.target.value)}
                      placeholder={settings.hasGeminiApiKey ? 'Enter new key to replace...' : 'Enter your Gemini API key...'}
                      onKeyDown={(e) => e.stopPropagation()}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      onClick={() => {
                        const trimmed = geminiApiKeyInput.trim()
                        if (!trimmed) return
                        void saveGeminiApiKey(trimmed)
                        setGeminiApiKeyInput('')
                      }}
                      disabled={!geminiApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasGeminiApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-amber-500/10 text-amber-400'
                    }`}>
                      {settings.hasGeminiApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          API key required
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <a
                      href="https://aistudio.google.com/app/apikey"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 transition-colors underline underline-offset-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Get Gemini API key →
                    </a>
                  </div>
                </div>
              </div>

              {/* Pexels API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Film className="h-4 w-4 text-teal-400" />
                  <h3 className="text-sm font-semibold text-white">Pexels API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your Pexels API key powers the stock-media browser in the LoRA trainer, letting you search and add free photos and videos to a training collection.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={pexelsApiKeyInput}
                      onChange={(e) => setPexelsApiKeyInput(e.target.value)}
                      placeholder={settings.hasPexelsApiKey ? 'Enter new key to replace...' : 'Enter your Pexels API key...'}
                      onKeyDown={(e) => e.stopPropagation()}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      onClick={() => {
                        const trimmed = pexelsApiKeyInput.trim()
                        if (!trimmed) return
                        void savePexelsApiKey(trimmed)
                        setPexelsApiKeyInput('')
                      }}
                      disabled={!pexelsApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasPexelsApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-amber-500/10 text-amber-400'
                    }`}>
                      {settings.hasPexelsApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          API key required
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <a
                      href="https://www.pexels.com/api/new/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 transition-colors underline underline-offset-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Get Pexels API key →
                    </a>
                  </div>
                </div>
              </div>

              {/* HuggingFace Account */}
              {window.electronAPI.hfGatingEnabled && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-orange-400" />
                  <h3 className="text-sm font-semibold text-white">HuggingFace</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Sign in to HuggingFace to download model files.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                    hfAuthStatus === 'authenticated'
                      ? 'bg-green-500/10 text-green-400'
                      : 'bg-amber-500/10 text-amber-400'
                  }`}>
                    {hfAuthStatus === 'authenticated' ? (
                      <>
                        <Check className="h-3 w-3" />
                        Signed in
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-3 w-3" />
                        Not signed in
                      </>
                    )}
                  </div>

                  {hfAuthStatus === 'authenticated' ? (
                    <button
                      onClick={handleHuggingFaceLogout}
                      className="px-3 py-2 bg-zinc-700 text-white text-sm rounded-lg hover:bg-zinc-600 transition-colors"
                    >
                      Sign out
                    </button>
                  ) : (
                    <button
                      onClick={startHuggingFaceLogin}
                      disabled={hfAuthPolling}
                      className="px-3 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors"
                    >
                      {hfAuthPolling ? 'Waiting for sign in...' : 'Sign in with HuggingFace'}
                    </button>
                  )}
                </div>
              </div>
              )}

              {/* Training credentials — used by the LoRA Trainer. The HuggingFace
                  token unlocks the gated Gemma text encoder; the RunPod key
                  connects the cloud GPU. Connect/validate RunPod in the LoRA
                  Trainer tab, where the GPU picker lives. */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Cloud className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Training</h3>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  Credentials for LoRA training. Save them here, then switch to the LoRA Trainer tab to connect your GPU and tune training settings.
                </p>

                {/* HuggingFace token — needed for the gated Gemma text encoder
                    (the LTX-2.3 checkpoint repo is public). */}
                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-700 text-[11px] font-semibold text-white">1</span>
                    <h3 className="text-sm font-semibold text-white">HuggingFace token</h3>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    The Gemma text encoder is gated. Paste a HuggingFace token (read access) and accept the Gemma license — one-time. The LTX-2.3 checkpoint itself is public.
                  </p>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={hfTokenInput}
                      onChange={(e) => setHfTokenInput(e.target.value)}
                      placeholder={settings.hasHfToken ? 'Saved — enter a new token to replace…' : 'hf_…'}
                      onKeyDown={(e) => e.stopPropagation()}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      onClick={() => {
                        const trimmed = hfTokenInput.trim()
                        if (!trimmed) return
                        void saveHfToken(trimmed)
                        setHfTokenInput('')
                      }}
                      disabled={!hfTokenInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save
                    </button>
                  </div>
                  <KeyStatusBadge configured={settings.hasHfToken} />
                  <div className="flex flex-col gap-1">
                    <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()} className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2">Create a HuggingFace token →</a>
                    <a href="https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized" target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()} className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2">Accept the Gemma license →</a>
                  </div>
                </div>

                {/* RunPod API key — saved here; connected/validated in the LoRA
                    Trainer tab, where GPUs are picked. */}
                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-700 text-[11px] font-semibold text-white">2</span>
                    <h3 className="text-sm font-semibold text-white">RunPod API key</h3>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    Bring your own RunPod account for cloud GPU training. Save the key here, then connect it in the LoRA Trainer tab to choose a GPU.
                  </p>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={runpodApiKeyInput}
                      onChange={(e) => setRunpodApiKeyInput(e.target.value)}
                      placeholder={settings.hasRunpodApiKey ? 'Saved — enter a new key to replace…' : 'Paste your RunPod API key…'}
                      onKeyDown={(e) => e.stopPropagation()}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      onClick={() => {
                        const trimmed = runpodApiKeyInput.trim()
                        if (!trimmed) return
                        void saveRunpodApiKey(trimmed)
                        setRunpodApiKeyInput('')
                      }}
                      disabled={!runpodApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save
                    </button>
                  </div>
                  <KeyStatusBadge configured={settings.hasRunpodApiKey} />
                  <a href="https://www.runpod.io/console/user/settings" target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()} className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2">Get RunPod API key →</a>
                </div>
              </div>
            </>
          )}

          {activeTab === 'promptEnhancer' && (
            <>
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Prompt Enhancer</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Automatically enhances your prompts via the LTX API with rich visual details, sound descriptions,
                  and motion cues to help generate higher quality videos. Control independently for each generation type.
                </p>

                {!settings.hasLtxApiKey ? (
                  <div className="space-y-4 mt-2">
                    <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-4 space-y-3">
                      <div className="flex items-start gap-2.5">
                        <AlertCircle className="h-4 w-4 text-amber-400 mt-0.5 flex-shrink-0" />
                        <div className="space-y-2">
                          <p className="text-sm text-amber-300 font-medium">LTX API key required</p>
                          <p className="text-xs text-zinc-400 leading-relaxed">
                            Prompt enhancement runs server-side on the LTX API. To use this feature, you need to configure
                            an API key in the API Keys tab.
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => setActiveTab('apiKeys')}
                        className="w-full mt-1 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-lg transition-colors"
                      >
                        Set API Key
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* T2V Toggle */}
                    <div
                      className="flex items-center justify-between bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 cursor-pointer"
                      onClick={() => handleTogglePromptEnhancer('t2v')}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-blue-400 bg-blue-400/10 px-1.5 py-0.5 rounded">T2V</span>
                        <div>
                          <span className="text-sm text-zinc-200">Text-to-Video</span>
                          <p className="text-[10px] text-zinc-500 mt-0.5">
                            {settings.promptEnhancerEnabledT2V ? 'Prompts will be enhanced before T2V generation' : 'T2V prompts used as-is'}
                          </p>
                        </div>
                      </div>
                      <div className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                        settings.promptEnhancerEnabledT2V ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}>
                        <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                          settings.promptEnhancerEnabledT2V ? 'translate-x-5' : 'translate-x-0'
                        }`} />
                      </div>
                    </div>

                    {/* I2V Toggle */}
                    <div
                      className="flex items-center justify-between bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 cursor-pointer"
                      onClick={() => handleTogglePromptEnhancer('i2v')}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded">I2V</span>
                        <div>
                          <span className="text-sm text-zinc-200">Image-to-Video</span>
                          <p className="text-[10px] text-zinc-500 mt-0.5">
                            {settings.promptEnhancerEnabledI2V ? 'Prompts will be enhanced before I2V generation' : 'I2V prompts used as-is'}
                          </p>
                        </div>
                      </div>
                      <div className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                        settings.promptEnhancerEnabledI2V ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}>
                        <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                          settings.promptEnhancerEnabledI2V ? 'translate-x-5' : 'translate-x-0'
                        }`} />
                      </div>
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {activeTab === 'loraTrainer' && (
            <>
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Cloud className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Cloud GPU (RunPod)</h3>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  LoRA training runs on a remote GPU (LTX-2 needs Linux + CUDA). Add your RunPod
                  API key and HuggingFace token in the API Keys tab, then connect here to manage
                  account-level storage and pods. Pick a GPU in each training dialog.
                </p>
              </div>

              {/* Credentials live in the API Keys tab — show status here with a
                  deep-link so users don't hunt for where to enter them. */}
              <div className="space-y-2 pt-4 border-t border-zinc-800">
                <div className="flex items-center justify-between gap-3 bg-zinc-800/50 rounded-lg px-3 py-2">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-zinc-400">HuggingFace token</span>
                    <KeyStatusBadge configured={settings.hasHfToken} />
                  </div>
                  <button onClick={() => setActiveTab('apiKeys')} className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2">
                    Manage in API Keys →
                  </button>
                </div>
                <div className="flex items-center justify-between gap-3 bg-zinc-800/50 rounded-lg px-3 py-2">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-zinc-400">RunPod API key</span>
                    <KeyStatusBadge configured={settings.hasRunpodApiKey} />
                  </div>
                  <button onClick={() => setActiveTab('apiKeys')} className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2">
                    Manage in API Keys →
                  </button>
                </div>
              </div>

              {/* Connect validates the saved key and loads account resources.
                  Per-run GPU selection intentionally lives in training dialogs. */}
              <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold text-white">Connect RunPod</h3>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => void handleConnectRunpod()}
                    disabled={runpodConnectState.status === 'connecting' || !settings.hasRunpodApiKey}
                    className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap flex items-center gap-1.5"
                  >
                    {runpodConnectState.status === 'connecting' ? (<><Loader2 className="h-4 w-4 animate-spin" /> Connecting…</>) : 'Connect'}
                  </button>
                </div>
                {!settings.hasRunpodApiKey && (
                  <p className="text-[11px] text-amber-400 flex items-center gap-1">
                    <AlertCircle className="h-3 w-3" /> Add your RunPod API key in the API Keys tab first.
                  </p>
                )}
                {runpodConnectState.status === 'connected' && (
                  <div className="text-xs px-3 py-2 rounded-lg flex items-center gap-1.5 bg-green-500/10 text-green-400">
                    <Check className="h-3.5 w-3.5" />
                    Connected — {runpodConnectState.gpus.length} global GPU types
                  </div>
                )}
                {runpodConnectState.status === 'error' && (
                  <div className="text-xs px-3 py-2 rounded-lg flex items-center gap-1.5 bg-red-500/10 text-red-400">
                    <AlertCircle className="h-3.5 w-3.5" /> {runpodConnectState.message}
                  </div>
                )}
              </div>

              {(
                <div className="space-y-4 pt-4 border-t border-zinc-800">
                  <p className="rounded-lg border border-blue-500/20 bg-blue-500/5 px-3 py-2 text-[11px] text-blue-200">
                    GPU selection happens in the training dialog. It shows global stock and automatically uses compatible saved-model storage when available.
                  </p>

                  {/* Account-level storage, idle-stop, pods, and defaults. */}
                  <div className="space-y-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-1.5 text-sm font-medium text-white">
                        <HardDrive className="h-3.5 w-3.5 text-emerald-400" /> Saved model storage
                      </div>
                      <p className="text-xs text-zinc-500 leading-relaxed">
                        Optional paid regional storage avoids repeated model downloads. Creating storage is always an explicit action, and it bills monthly until deleted.
                      </p>
                      {managedCacheVolumes.length === 0 && (
                        <p className="text-[11px] text-amber-400">
                          The suggested {cacheVolumeSizeGb.toLocaleString()} GB size costs about {formatRunpodStorageMonthlyUsd(cacheVolumeSizeGb)}/month.
                        </p>
                      )}
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-xs font-medium text-zinc-300">Auto-stop idle pod after (minutes)</label>
                      <input
                        type="number"
                        min={0}
                        max={240}
                        step={1}
                        value={settings.runpodIdleStopMinutes}
                        onChange={(e) => {
                          const n = Math.round(Number(e.target.value))
                          if (!Number.isFinite(n)) return
                          updateSettings({ runpodIdleStopMinutes: Math.max(0, Math.min(240, n)) })
                        }}
                        onKeyDown={(e) => e.stopPropagation()}
                        className="w-28 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
                      />
                      <p className="text-[11px] text-zinc-600">0 disables auto-stop. Pods bill per minute while running.</p>
                    </div>
                  </div>

                  {runpodConnectState.status === 'connected' && (
                    <div className="space-y-3 rounded-lg border border-zinc-700 bg-zinc-900/40 p-3">
                      {runpodConnectState.regionHealth.length === 0 && (
                        <p className="rounded-md bg-amber-500/10 px-2.5 py-2 text-xs text-amber-300">
                          Restart LTX Desktop once to load regional GPU health and volume actions.
                        </p>
                      )}
                      {managedCacheVolumes.length > 0 ? (
                        <div className="space-y-2">
                          {managedCacheVolumes.map((volume) => (
                            <div
                              key={volume.id}
                              className="flex flex-wrap items-start justify-between gap-2 rounded-md bg-zinc-800/60 px-2.5 py-2"
                            >
                              <div className="min-w-0">
                                <div className="text-xs font-medium text-white">
                                  Saved models · {volume.datacenterId || 'Unknown region'}
                                </div>
                                <div className="text-[11px] text-zinc-500">
                                  {volume.sizeGb.toLocaleString()} GB · about {formatRunpodStorageMonthlyUsd(volume.sizeGb)}/month
                                </div>
                                <div className={`mt-1 text-[10px] ${
                                  volume.savedModelReadiness === 'ready'
                                    ? 'text-emerald-300'
                                    : volume.savedModelReadiness === 'missing'
                                      ? 'text-amber-300'
                                      : 'text-zinc-400'
                                }`}>
                                  {volume.savedModelReadiness === 'ready'
                                    ? 'Models ready — compatible GPUs avoid the model download'
                                    : volume.savedModelReadiness === 'missing'
                                      ? 'Models need downloading before this storage can accelerate a run'
                                      : 'Model readiness will be confirmed before training'}
                                </div>
                              </div>
                              <button
                                type="button"
                                disabled={volumeActionState.status === 'working'}
                                onClick={async () => {
                                  if (!await confirmAction({
                                    title: `Delete ${volume.sizeGb} GB cloud storage?`,
                                    message: `Downloaded models and remote data in ${volume.datacenterId || 'this region'} will be permanently removed.`,
                                    confirmLabel: 'Delete storage',
                                    variant: 'destructive',
                                  })) return
                                  void runVolumeAction(
                                    'Deleting storage',
                                    () => deleteRunpodVolume(volume.id),
                                  )
                                }}
                                className="rounded-md bg-red-500/10 px-2 py-1 text-[10px] text-red-300 hover:bg-red-500/20 disabled:opacity-50"
                              >
                                Delete storage
                              </button>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-xs text-zinc-400">
                          No saved-model storage. All GPUs remain available; the selected run downloads models when needed.
                        </p>
                      )}

                      {managedCacheVolumes.length === 0 && (
                        <div className="space-y-2 rounded-md bg-zinc-800/50 p-2.5">
                          <div className="flex flex-wrap items-baseline justify-between gap-1">
                            <span className="text-xs font-medium text-zinc-200">Size for a new cache</span>
                            <span className="text-[11px] text-zinc-500">
                              About ${newCacheMonthlyEstimate.toFixed(0)}/month
                            </span>
                          </div>
                          <div className="grid grid-cols-3 gap-1.5">
                            {[250, 500, 1000].map((sizeGb) => (
                              <button
                                key={sizeGb}
                                type="button"
                                onClick={() => updateSettings({ runpodVolumeSizeGb: sizeGb })}
                                className={`rounded-md border px-2 py-2 text-xs transition-colors ${
                                  settings.runpodVolumeSizeGb === sizeGb
                                    ? 'border-blue-500 bg-blue-500/15 text-white'
                                    : 'border-zinc-700 bg-zinc-900/50 text-zinc-400 hover:border-zinc-600'
                                }`}
                              >
                                <span className="block font-medium">{sizeGb} GB</span>
                                {sizeGb === 250 && (
                                  <span className="block text-[9px] text-emerald-400">Recommended</span>
                                )}
                              </button>
                            ))}
                          </div>
                          <p className="text-[10px] leading-relaxed text-zinc-500">
                            250 GB fits the roughly 83 GB trainer payload and typical datasets.
                            Choose more for many datasets or retained checkpoints. Custom sizes remain in Advanced.
                          </p>
                        </div>
                      )}
                      {managedCacheVolumes.length === 0 && (
                        <button
                          type="button"
                          disabled={volumeActionState.status === 'working'}
                          onClick={() => void runVolumeAction('Creating cache', () => createRunpodVolume({
                            sizeGb: settings.runpodVolumeSizeGb,
                          }))}
                          className="rounded-lg bg-blue-600 px-3 py-2 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-50"
                        >
                          Create cache
                        </button>
                      )}

                      {volumeActionState.status === 'working' && (
                        <p className="flex items-center gap-1.5 text-xs text-blue-300">
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          {volumeActionState.action}…
                        </p>
                      )}
                      {volumeActionState.status === 'done' && (
                        <p className={`text-xs ${volumeActionState.ok ? 'text-emerald-300' : 'text-red-300'}`}>
                          {volumeActionState.message}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Active pods on the account — visibility + one-click teardown
                      so stray pods (e.g. left by a failed run) don't keep billing. */}
                  {runpodConnectState.status === 'connected' && runpodConnectState.pods.length > 0 && (
                    <div className="space-y-2">
                      <h3 className="text-sm font-semibold text-white">Active pods</h3>
                      <p className="text-xs text-zinc-500 leading-relaxed">
                        Pods currently on your RunPod account. They bill per minute while running — terminate any you don't need.
                      </p>
                      <div className="space-y-1.5">
                        {runpodConnectState.pods.map((pod) => (
                          <div key={pod.id} className="flex items-center justify-between gap-3 bg-zinc-800/50 rounded-lg px-3 py-2">
                            <div className="min-w-0">
                              <div className="flex items-center gap-2 text-sm text-white truncate">
                                <span className={`h-2 w-2 rounded-full flex-shrink-0 ${pod.status === 'RUNNING' ? 'bg-emerald-500' : 'bg-zinc-500'}`} />
                                <span className="truncate">{pod.gpu || pod.id}</span>
                                {pod.createdByApp && (
                                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-300 flex-shrink-0">this app</span>
                                )}
                              </div>
                              <div className="text-[11px] text-zinc-500 truncate">
                                {pod.status}{pod.costPerHr != null ? ` · $${pod.costPerHr.toFixed(2)}/hr` : ''} · {pod.id}
                              </div>
                            </div>
                            <button
                              onClick={() => void handleTerminatePod(pod.id)}
                              disabled={terminatingPodId === pod.id || !pod.createdByApp}
                              title={pod.createdByApp ? 'Terminate this app-owned pod' : 'Only pods created by LTX Desktop can be terminated here'}
                              className="px-2.5 py-1.5 text-xs rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 disabled:opacity-50 transition-colors whitespace-nowrap"
                            >
                              {!pod.createdByApp ? 'Managed in RunPod' : terminatingPodId === pod.id ? 'Terminating…' : 'Terminate'}
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Advanced — defaults are sensible; most users never open this. */}
                  <div className="pt-2">
                    <button
                      onClick={() => setShowRunpodAdvanced((v) => !v)}
                      className="flex items-center gap-1.5 text-xs font-medium text-zinc-400 hover:text-white transition-colors"
                    >
                      <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showRunpodAdvanced ? 'rotate-180' : ''}`} />
                      Advanced
                    </button>
                    {showRunpodAdvanced && (
                      <div className="mt-3 space-y-4">
                        <LabeledInput label="Workspace dir" placeholder="/workspace" value={settings.loraRemoteWorkspaceDir} onChange={(v) => updateSettings({ loraRemoteWorkspaceDir: v })} />
                        <LabeledInput label="Network volume ID" placeholder="Auto-created when caching is on" value={settings.runpodNetworkVolumeId} onChange={(v) => updateSettings({ runpodNetworkVolumeId: v })} />
                        <div className="space-y-1.5">
                          <label className="text-xs font-medium text-zinc-300">New volume size (GB)</label>
                          <input
                            type="number"
                            min={250}
                            max={4000}
                            step={10}
                            value={settings.runpodVolumeSizeGb}
                            onChange={(e) => {
                              const n = Math.round(Number(e.target.value))
                              if (!Number.isFinite(n)) return
                              updateSettings({ runpodVolumeSizeGb: Math.max(250, Math.min(4000, n)) })
                            }}
                            onKeyDown={(e) => e.stopPropagation()}
                            className="w-28 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
                          />
                          <p className="text-[11px] text-zinc-600">Models and the trainer need about 83GB. 250GB is recommended for typical use; choose more for many datasets or checkpoints. Existing volumes cannot be shrunk here.</p>
                        </div>
                        <LabeledInput label="Model checkpoint HF repo" placeholder="Lightricks/LTX-2.3" value={settings.loraModelHfRepo} onChange={(v) => updateSettings({ loraModelHfRepo: v })} />
                        <LabeledInput label="Model checkpoint file" placeholder="ltx-2.3-22b-dev.safetensors" value={settings.loraModelCheckpointFile} onChange={(v) => updateSettings({ loraModelCheckpointFile: v })} />
                        <LabeledInput label="Text encoder HF repo" placeholder="google/gemma-3-12b-it-qat-q4_0-unquantized" value={settings.loraTextEncoderHfRepo} onChange={(v) => updateSettings({ loraTextEncoderHfRepo: v })} />
                        <LabeledInput label="Model checkpoint path (override)" placeholder="Derived from workspace dir" value={settings.loraRemoteModelPath} onChange={(v) => updateSettings({ loraRemoteModelPath: v })} />
                        <LabeledInput label="Text encoder path (override)" placeholder="Derived from workspace dir" value={settings.loraRemoteTextEncoderPath} onChange={(v) => updateSettings({ loraRemoteTextEncoderPath: v })} />
                        <LabeledInput label="Custom pod image" placeholder="Defaults to a stock CUDA image" value={settings.runpodImage} onChange={(v) => updateSettings({ runpodImage: v })} />
                        <div className="space-y-1.5">
                          <label className="text-xs font-medium text-zinc-300">Trainer source</label>
                          <div className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 font-mono text-xs text-zinc-300 break-all">
                            {settings.loraTrainerRepoUrl}@{settings.loraTrainerRepoRef}
                          </div>
                          <p className="text-[11px] text-zinc-600">
                            Security-pinned to the officially documented LTX-2 trainer revision.
                          </p>
                        </div>
                        <div className="flex items-center justify-between gap-4">
                          <label className="text-sm font-medium text-white">Auto-provision pod</label>
                          <button
                            onClick={() => updateSettings({ loraAutoProvision: !settings.loraAutoProvision })}
                            className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${settings.loraAutoProvision ? 'bg-emerald-500' : 'bg-zinc-700'}`}
                          >
                            <span className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${settings.loraAutoProvision ? 'translate-x-5' : 'translate-x-0'}`} />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-yellow-400" />
                  <h3 className="text-sm font-semibold text-white">Dataset-prep generation</h3>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  How many AI generation requests (Nano Banana edits, Kling) the trainer sends to
                  Fal at once during bulk generation. Higher is faster but more likely to hit Fal
                  rate limits — those are retried automatically. Local GPU generations always run
                  one at a time.
                </p>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-300">
                    Simultaneous Fal requests
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    step={1}
                    value={settings.loraFalConcurrency}
                    onChange={(e) => {
                      const n = Math.round(Number(e.target.value))
                      if (!Number.isFinite(n)) return
                      updateSettings({ loraFalConcurrency: Math.max(1, Math.min(20, n)) })
                    }}
                    onKeyDown={(e) => e.stopPropagation()}
                    className="w-28 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono"
                  />
                  <p className="text-[11px] text-zinc-600">Between 1 and 20.</p>
                </div>
              </div>

              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <Button
                  onClick={() => void handleTestLoraConnection()}
                  disabled={testConnectionState.status === 'testing'}
                  className="w-full bg-blue-600 hover:bg-blue-500 text-white"
                >
                  {testConnectionState.status === 'testing' ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Testing connection...
                    </>
                  ) : (
                    'Test Connection'
                  )}
                </Button>
                {testConnectionState.status === 'done' && (
                  <div className={`text-xs px-3 py-2 rounded-lg flex items-center gap-1.5 ${
                    testConnectionState.ok ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
                  }`}>
                    {testConnectionState.ok ? <Check className="h-3.5 w-3.5" /> : <AlertCircle className="h-3.5 w-3.5" />}
                    {testConnectionState.message}
                  </div>
                )}
              </div>
            </>
          )}

          {activeTab === 'about' && (
            <>
              {showModelLicense ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">LTX-2 Model License</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowModelLicense(false)}
                      className="h-7 px-2 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                    >
                      Back
                    </Button>
                  </div>
                  <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-800/50 rounded-lg p-4 max-h-[50vh] overflow-y-auto border border-zinc-700/50">
                    {modelLicenseText}
                  </pre>
                </div>
              ) : showNotices ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">Third-Party Notices</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowNotices(false)}
                      className="h-7 px-2 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                    >
                      Back
                    </Button>
                  </div>
                  <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-800/50 rounded-lg p-4 max-h-[50vh] overflow-y-auto border border-zinc-700/50">
                    {noticesText}
                  </pre>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* App Identity */}
                  <div className="text-center space-y-2">
                    <h3 className="text-lg font-bold text-white">LTX Desktop</h3>
                    <p className="text-sm text-zinc-400">Version {appVersion || '...'}</p>
                    <p className="text-xs text-zinc-500">Unofficial fork with LoRA training tools</p>
                    <p className="text-[11px] text-zinc-600">
                      Based on Lightricks/LTX-Desktop; not an official Lightricks release.
                    </p>
                  </div>

                  {/* License */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <Info className="h-4 w-4 text-blue-400" />
                      <span className="text-sm font-medium text-white">License</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      Licensed under the Apache License, Version 2.0
                    </p>
                  </div>

                  {/* LTX-2 Model License */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                      </svg>
                      <span className="text-sm font-medium text-white">LTX-2 Model License</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      The LTX-2 model is subject to the LTX-2 Community License Agreement, accepted during first-run setup.
                    </p>
                    <Button
                      size="sm"
                      onClick={handleLoadModelLicense}
                      disabled={modelLicenseLoading}
                      className="w-full bg-zinc-700 hover:bg-zinc-600 text-white text-xs"
                    >
                      {modelLicenseLoading ? 'Loading...' : 'View Model License'}
                    </Button>
                  </div>

                  {/* Third-Party Notices */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                      </svg>
                      <span className="text-sm font-medium text-white">Third-Party Notices</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      This application uses open-source software and AI models subject to their own license terms.
                    </p>
                    <Button
                      size="sm"
                      onClick={handleLoadNotices}
                      disabled={noticesLoading}
                      className="w-full bg-zinc-700 hover:bg-zinc-600 text-white text-xs"
                    >
                      {noticesLoading ? 'Loading...' : 'View Third-Party Notices'}
                    </Button>
                  </div>

                  {/* Copyright */}
                  <p className="text-center text-xs text-zinc-600">
                    Copyright © 2026 Lightricks and LTX Desktop contributors
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-zinc-800 flex justify-end">
          <Button
            onClick={onClose}
            className="bg-zinc-700 hover:bg-zinc-600 text-white"
          >
            Done
          </Button>
        </div>
      </div>
    </div>
  )
}

function KeyStatusBadge({ configured }: { configured: boolean }) {
  return (
    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
      configured ? 'bg-green-500/10 text-green-400' : 'bg-amber-500/10 text-amber-400'
    }`}>
      {configured ? (
        <>
          <Check className="h-3 w-3" />
          Configured
        </>
      ) : (
        <>
          <AlertCircle className="h-3 w-3" />
          Not configured
        </>
      )}
    </div>
  )
}

function LabeledInput({
  label,
  placeholder,
  value,
  onChange,
}: {
  label: string
  placeholder: string
  value: string
  onChange: (value: string) => void
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-zinc-300">{label}</label>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.stopPropagation()}
        spellCheck={false}
        className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono"
      />
    </div>
  )
}

export type { AppSettings, TabId as SettingsTabId }
