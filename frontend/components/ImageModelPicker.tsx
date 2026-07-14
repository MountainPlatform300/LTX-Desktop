import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronUp, Download, FolderOpen, FolderInput, Loader2, Lock, MoreVertical } from 'lucide-react'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import { InfoHint } from './lora/trainingFormParts'
import type { ImageModelSpec } from '../lib/image-generation-model-specs'

type StartModelDownloadBody = NonNullable<ApiRequestBodyOf<'startModelDownload'>>
type ModelCheckpointID = NonNullable<StartModelDownloadBody['cp_ids']>[number]
type DownloadProgress = ApiSuccessOf<'getModelDownloadProgress'>
type HfAuthStatus = ApiSuccessOf<'getHuggingFaceAuthStatus'>['status']

interface ImageModelPickerProps {
  specs: ImageModelSpec[]
  selectedId: string
  onSelect: (id: string) => void
  // Refetch the catalog after a download completes so the `downloaded` flag
  // flips and the picker re-renders without a manual refresh.
  onSpecsChanged: () => void
}

function formatBytes(bytes: number): string {
  const gb = bytes / 1_000_000_000
  if (gb >= 1) return `${gb >= 10 ? Math.round(gb) : gb.toFixed(1)} GB`
  const mb = bytes / 1_000_000
  return `${Math.round(mb)} MB`
}

function formatSpeed(bytesPerSec: number): string {
  if (bytesPerSec <= 0) return ''
  const mb = bytesPerSec / 1_000_000
  if (mb >= 1) return `${mb >= 100 ? Math.round(mb) : mb.toFixed(1)} MB/s`
  const kb = bytesPerSec / 1_000
  return `${Math.round(kb)} KB/s`
}

function formatEta(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return ''
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`
}

function statusLabel(spec: ImageModelSpec): string {
  if (spec.downloaded) {
    return spec.inference_status === 'coming_soon' ? 'Downloaded · coming soon' : 'Downloaded'
  }
  const size = formatBytes(spec.size_bytes)
  return spec.gated ? `Gated · ${size}` : size
}

function tooltipContent(spec: ImageModelSpec): string {
  const inference = spec.inference_status === 'available'
    ? 'Inference: available'
    : 'Inference: coming soon (downloadable now, generation wired in a follow-up)'
  const gated = spec.gated
    ? 'Gated on HuggingFace — accept the license on the repo page and sign in via Settings \u2192 HuggingFace before downloading.'
    : 'Not gated.'
  return `${spec.description}\n\nLicense: ${spec.license}\nSize: ${formatBytes(spec.size_bytes)}\n${gated}\n${inference}`
}

/**
 * Gen Space image-model picker. Replaces the old static "Z-Image Turbo" badge
 * with a popover listing every catalogued image model (Z-Image plus the
 * open-weight additions). Each row carries an "i" tooltip with license, size,
 * gating, and inference status, and a per-row Download button for missing
 * models (with live progress).
 *
 * Gated models route through the HuggingFace auth flow before the download
 * starts. Auth is handled here directly against the backend endpoints rather
 * than via the `useHfAuth`/`useHfModelAccess` hooks, because those hooks
 * short-circuit to "authenticated" when the app-level `hfGatingEnabled` flag
 * is off — but a per-spec gated repo (Ideogram 4, FLUX.1 Krea) still requires a
 * token regardless of that global flag. Querying the real status here keeps the
 * picker honest in both configurations.
 */
export function ImageModelPicker({ specs, selectedId, onSelect, onSpecsChanged }: ImageModelPickerProps) {
  const [isOpen, setIsOpen] = useState(false)
  const popoverRef = useRef<HTMLDivElement>(null)

  const selectedSpec = specs.find((s) => s.id === selectedId) ?? specs[0] ?? null

  // Single active download at a time (the backend only allows one session).
  const [downloadingCpId, setDownloadingCpId] = useState<ModelCheckpointID | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)

  // Real HF auth status (independent of the global hfGatingEnabled flag).
  const [hfAuthStatus, setHfAuthStatus] = useState<HfAuthStatus>('not_authenticated')
  // A gated download the user has already clicked — auto-proceeds once HF auth
  // + repo access are confirmed, so the user doesn't have to click twice.
  const [pendingGatedDownload, setPendingGatedDownload] = useState<ModelCheckpointID | null>(null)

  useEffect(() => {
    let active = true
    void (async () => {
      const result = await ApiClient.getHuggingFaceAuthStatus()
      if (active && result.ok) setHfAuthStatus(result.data.status)
    })()
    return () => {
      active = false
    }
  }, [])

  const refreshHfAuthStatus = useCallback(async (): Promise<HfAuthStatus> => {
    const result = await ApiClient.getHuggingFaceAuthStatus()
    if (result.ok) {
      setHfAuthStatus(result.data.status)
      return result.data.status
    }
    return 'not_authenticated'
  }, [])

  const startHfLogin = useCallback(async () => {
    const result = await ApiClient.startHuggingFaceLogin()
    if (!result.ok) {
      setDownloadError(`HuggingFace sign-in failed: ${result.error.message}`)
      return
    }
    const params = result.data
    setHfAuthStatus('pending')
    await window.electronAPI.openHuggingFaceAuth({
      clientId: params.client_id,
      redirectUri: params.redirect_uri,
      scope: params.scope,
      state: params.state,
      codeChallenge: params.code_challenge,
      codeChallengeMethod: params.code_challenge_method,
    })
  }, [])

  // Poll HF auth status while a login is pending (user completing OAuth in the
  // browser). When it flips to authenticated, recheck access for the gated
  // model the user was trying to download and auto-start it if authorized.
  useEffect(() => {
    if (hfAuthStatus !== 'pending') return
    const interval = setInterval(async () => {
      const status = await refreshHfAuthStatus()
      if (status === 'authenticated') {
        if (pendingGatedDownload) {
          void tryDownloadGated(pendingGatedDownload)
        }
      }
    }, 2000)
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hfAuthStatus, pendingGatedDownload, refreshHfAuthStatus])

  const startDownload = useCallback(async (cpId: ModelCheckpointID) => {
    setDownloadError(null)
    setDownloadProgress(null)
    const result = await ApiClient.startModelDownload({ type: 'download', cp_ids: [cpId] })
    if (!result.ok) {
      setDownloadError(result.error.message)
      return
    }
    const data = result.data
    if (data.status === 'started') {
      setDownloadingCpId(cpId)
      setDownloadSessionId(data.sessionId)
    } else {
      setDownloadError('Unexpected response while starting model download.')
    }
  }, [])

  const tryDownloadGated = useCallback(
    async (cpId: ModelCheckpointID) => {
      const status = await refreshHfAuthStatus()
      if (status !== 'authenticated') {
        setPendingGatedDownload(cpId)
        setDownloadError('Sign in to HuggingFace to download this gated model.')
        await startHfLogin()
        return
      }
      // Authenticated — verify the user accepted the repo's license gate.
      const accessResult = await ApiClient.checkModelAccess({ cp_ids: [cpId] })
      if (!accessResult.ok) {
        setDownloadError(accessResult.error.message)
        return
      }
      const spec = specs.find((s) => s.checkpoint_id === cpId) ?? null
      const access = spec ? accessResult.data.access[spec.repo_id] : undefined
      if (access && access !== 'authorized') {
        setPendingGatedDownload(cpId)
        setDownloadError(
          `You haven\u2019t accepted the license for ${spec?.repo_id ?? 'this repo'} yet. Open the repo page on HuggingFace, accept the gate, then click Download again.`,
        )
        if (spec) {
          void window.electronAPI.openHuggingFaceRepo({ repoId: spec.repo_id })
        }
        return
      }
      setPendingGatedDownload(null)
      void startDownload(cpId)
    },
    [refreshHfAuthStatus, specs, startDownload, startHfLogin],
  )

  const handleDownloadClick = useCallback(
    (spec: ImageModelSpec) => {
      if (downloadingCpId) return
      if (!spec.gated) {
        void startDownload(spec.checkpoint_id)
        return
      }
      void tryDownloadGated(spec.checkpoint_id)
    },
    [downloadingCpId, startDownload, tryDownloadGated],
  )

  // Per-row kebab menu (Reveal in Explorer / Load from location).
  const [openMenuCpId, setOpenMenuCpId] = useState<ModelCheckpointID | null>(null)

  const handleRevealInExplorer = useCallback(async (spec: ImageModelSpec) => {
    setOpenMenuCpId(null)
    const result = await ApiClient.getCheckpointPath({ cp_id: spec.checkpoint_id })
    if (!result.ok) {
      setDownloadError(result.error.message)
      return
    }
    await window.electronAPI.showItemInFolder({ filePath: result.data.path })
  }, [])

  const handleLoadFromLocation = useCallback(
    async (spec: ImageModelSpec) => {
      setOpenMenuCpId(null)
      const dir = await window.electronAPI.showOpenDirectoryDialog({
        title: `Select the folder containing ${spec.display_name}`,
      })
      if (!dir) return
      setDownloadError(null)
      const result = await ApiClient.loadModelFromPath({
        cp_id: spec.checkpoint_id,
        sourcePath: dir,
      })
      if (!result.ok) {
        setDownloadError(result.error.message)
        return
      }
      onSpecsChanged()
    },
    [onSpecsChanged],
  )

  // Poll progress while a download session is active.
  useEffect(() => {
    if (!downloadSessionId) return
    const poll = async () => {
      const result = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (!result.ok) {
        logger.error(`Image model download progress error: ${result.error.message}`)
        return
      }
      const progress = result.data
      setDownloadProgress(progress)
      if (progress.status === 'complete') {
        setDownloadingCpId(null)
        setDownloadSessionId(null)
        setDownloadProgress(null)
        onSpecsChanged()
      } else if (progress.status === 'error') {
        setDownloadError(progress.error || 'Download failed.')
        setDownloadingCpId(null)
        setDownloadSessionId(null)
        setDownloadProgress(null)
      }
    }
    void poll()
    const interval = setInterval(() => void poll(), 500)
    return () => clearInterval(interval)
  }, [downloadSessionId, onSpecsChanged])

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    if (isOpen) document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  const activeDownloadSpec = downloadingCpId
    ? specs.find((s) => s.checkpoint_id === downloadingCpId) ?? null
    : null
  // The backend `total_progress` is already a 0-99 percentage (it computes
  // `downloaded / expected * 100` server-side). Don't multiply again — that
  // was the bug showing 458% / 1682%.
  const progress = downloadProgress?.status === 'downloading' ? downloadProgress : null
  const activePercent = progress ? Math.round(progress.total_progress ?? 0) : null
  const downloadedBytes = progress?.total_downloaded_bytes ?? 0
  const expectedBytes = progress?.expected_total_bytes ?? activeDownloadSpec?.size_bytes ?? 0
  const speed = progress?.speed_bytes_per_sec ?? 0
  const remainingBytes = Math.max(0, expectedBytes - downloadedBytes)
  const etaSeconds = speed > 0 ? remainingBytes / speed : Infinity

  return (
    <div ref={popoverRef} className="relative flex items-center gap-1.5">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-1.5 px-2 py-1 rounded-md transition-colors ${
          isOpen ? 'bg-zinc-700 hover:bg-zinc-700' : 'bg-zinc-800/50 hover:bg-zinc-800'
        }`}
      >
        <span className="text-zinc-300 font-medium max-w-[120px] truncate">
          {selectedSpec?.display_name ?? 'Select model'}
        </span>
        <ChevronUp className="h-3 w-3 text-zinc-500" />
      </button>

      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 bg-zinc-800 border border-zinc-700 rounded-md p-2 min-w-[260px] shadow-xl z-[9999]">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Image Model</div>
          <div className="space-y-1">
            {specs.map((spec) => {
              const isSelected = spec.id === selectedId
              const isDownloading = downloadingCpId === spec.checkpoint_id
              const rowPercent = isDownloading ? activePercent : null
              return (
                <div
                  key={spec.id}
                  className={`rounded-md transition-colors ${
                    isSelected ? 'bg-white/20' : 'hover:bg-zinc-700'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 px-2 py-2">
                    <button
                      type="button"
                      onClick={() => {
                        onSelect(spec.id)
                        setIsOpen(false)
                      }}
                      className="flex items-center gap-2 text-left min-w-0 flex-1"
                    >
                      <span className="flex flex-col leading-tight min-w-0">
                        <span className={`text-sm truncate ${isSelected ? 'text-white' : 'text-zinc-300'}`}>
                          {spec.display_name}
                        </span>
                        <span className="text-[11px] text-zinc-500 truncate">{statusLabel(spec)}</span>
                      </span>
                    </button>
                    <span className="flex items-center gap-1 shrink-0">
                      <InfoHint content={tooltipContent(spec)} side="left" />
                      {spec.downloaded ? (
                        <>
                          {isSelected && (
                            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                          <div className="relative">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation()
                                setOpenMenuCpId(openMenuCpId === spec.checkpoint_id ? null : spec.checkpoint_id)
                              }}
                              className="p-1 rounded text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                              title="More actions"
                            >
                              <MoreVertical className="h-3.5 w-3.5" />
                            </button>
                            {openMenuCpId === spec.checkpoint_id && (
                              <div className="absolute right-0 top-full mt-1 z-[10000] bg-zinc-900 border border-zinc-700 rounded-md py-1 min-w-[160px] shadow-xl">
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    void handleRevealInExplorer(spec)
                                  }}
                                  className="flex items-center gap-2 w-full px-2 py-1.5 text-[11px] text-zinc-300 hover:bg-zinc-700 text-left"
                                >
                                  <FolderOpen className="h-3.5 w-3.5" />
                                  Reveal in Explorer
                                </button>
                              </div>
                            )}
                          </div>
                        </>
                      ) : (
                        <>
                          <button
                            type="button"
                            disabled={!!downloadingCpId}
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDownloadClick(spec)
                            }}
                            className="flex items-center gap-1 px-2 py-1 rounded text-[11px] bg-blue-600/80 hover:bg-blue-600 text-white disabled:opacity-40 disabled:cursor-not-allowed"
                            title={spec.gated ? 'Download (gated — requires HuggingFace sign-in)' : 'Download'}
                          >
                            {spec.gated ? <Lock className="h-3 w-3" /> : <Download className="h-3 w-3" />}
                            {isDownloading && rowPercent !== null ? `${rowPercent}%` : formatBytes(spec.size_bytes)}
                          </button>
                          <div className="relative">
                            <button
                              type="button"
                              disabled={!!downloadingCpId}
                              onClick={(e) => {
                                e.stopPropagation()
                                setOpenMenuCpId(openMenuCpId === spec.checkpoint_id ? null : spec.checkpoint_id)
                              }}
                              className="p-1 rounded text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed"
                              title="More actions"
                            >
                              <MoreVertical className="h-3.5 w-3.5" />
                            </button>
                            {openMenuCpId === spec.checkpoint_id && (
                              <div className="absolute right-0 top-full mt-1 z-[10000] bg-zinc-900 border border-zinc-700 rounded-md py-1 min-w-[160px] shadow-xl">
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    void handleLoadFromLocation(spec)
                                  }}
                                  className="flex items-center gap-2 w-full px-2 py-1.5 text-[11px] text-zinc-300 hover:bg-zinc-700 text-left"
                                >
                                  <FolderInput className="h-3.5 w-3.5" />
                                  Load from location…
                                </button>
                              </div>
                            )}
                          </div>
                        </>
                      )}
                    </span>
                  </div>
                  {isDownloading && rowPercent !== null && (
                    <div className="px-2 pb-2">
                      <div className="h-1 w-full bg-zinc-700 rounded overflow-hidden">
                        <div className="h-full bg-blue-500 transition-all" style={{ width: `${rowPercent}%` }} />
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
          {activeDownloadSpec && (
            <div className="mt-2 pt-2 border-t border-zinc-700 text-[11px] text-zinc-400 space-y-1">
              <div className="flex items-center gap-1.5">
                <Loader2 className="h-3 w-3 animate-spin shrink-0" />
                <span>
                  Downloading {activeDownloadSpec.display_name}
                  {activePercent !== null ? ` · ${activePercent}%` : ''}
                </span>
              </div>
              {progress && expectedBytes > 0 && (
                <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 pl-4.5 text-zinc-500">
                  <span>
                    {formatBytes(downloadedBytes)} / {formatBytes(expectedBytes)}
                  </span>
                  {speed > 0 && <span>{formatSpeed(speed)}</span>}
                  {isFinite(etaSeconds) && etaSeconds > 0 && <span>{formatEta(etaSeconds)} left</span>}
                </div>
              )}
            </div>
          )}
          {downloadError && (
            <div className="mt-2 pt-2 border-t border-zinc-700 text-[11px] text-red-400 whitespace-pre-wrap">
              {downloadError}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
