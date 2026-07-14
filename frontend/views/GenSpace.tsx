import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Trash2, Download, Image, Video, X,
  Heart, Film, Volume2, VolumeX, Sparkles,
  Clock, Monitor, ChevronUp, Scissors, Music,
  ChevronLeft, ChevronRight, Copy, Check, Wand2, Layers,
  Play, Pause, Loader2, SlidersHorizontal, Settings, Pencil, Maximize2, Minimize2,
} from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import type { GenSpaceRetakeSource } from '../contexts/ProjectContext'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { useGeneration, getImageDimensions } from '../hooks/use-generation'
import { useQueue, type QueuePayload } from '../contexts/QueueContext'
import { ApiClient, type ApiRequestBodyOf } from '../lib/api-client'
import { useVideoGenerationModelSpecs } from '../hooks/use-video-generation-model-specs'
import { useImageGenerationModelSpecs } from '../hooks/use-image-generation-model-specs'
import type { ImageModelSpec } from '../lib/image-generation-model-specs'
import { createLocalGenerationError, type GenerationError } from '../lib/generation-errors'
import { useRetake } from '../hooks/use-retake'
import { useLoraInferenceRegistry, type LoraInferenceEntry, type LoraInferenceConditioningType } from '../hooks/use-lora-inference-registry'
import { LoraPickerPopover, type LoraPickerValue } from '../components/lora/LoraPickerPopover'
import { ImageModelPicker } from '../components/ImageModelPicker'
import { useIcLoraModelGate } from '../hooks/use-ic-lora-model-gate'
import { ModelDownloadGate } from '../components/models/ModelDownloadGate'
import type { Asset } from '../types/project-model'
import { GenerationErrorDialog } from '../components/GenerationErrorDialog'
import { addVisualAssetToProject } from '../lib/asset-copy'
import { pathToFileUrl } from '../lib/file-url'
import {
  areVideoGenerationSettingsEquivalent,
  getVideoGenerationModelSpecs,
  resolveVideoGenerationOptions,
  sanitizeVideoGenerationSettings,
  type VideoGenerationModelSpecItem,
} from '../lib/video-generation-model-specs'
import { logger } from '../lib/logger'
import { RetakePanel } from '../components/RetakePanel'
import { SendToLoraModal } from '../components/lora/LoraModals'
import { FreeApiKeyBubble } from '../components/FreeApiKeyBubble'
import { SettingsDropdown, LightricksIcon } from '../components/ui/settings-dropdown'
import { PopoverMenu } from '../components/ui/popover-menu'
import { Switch } from '../components/ui/switch'
import { Tooltip } from '../components/ui/tooltip'
import { startGalleryImageEdit, toggleElementFullscreen } from './genspace-actions'

const LORA_CONDITIONING_TYPES: { value: LoraInferenceConditioningType; label: string }[] = [
  { value: 'canny', label: 'Canny' },
  { value: 'depth', label: 'Depth' },
  { value: 'pose', label: 'Pose' },
]

const LORA_STRENGTH_OPTIONS = [
  { value: '0.5', label: '0.50' },
  { value: '0.75', label: '0.75' },
  { value: '1', label: '1.00' },
  { value: '1.25', label: '1.25' },
  { value: '1.5', label: '1.50' },
  { value: '2', label: '2.00' },
]

// Supported IC-LoRA output durations (seconds @ 24fps). Mirrors the backend
// `_IC_LORA_DURATIONS`. The reference is resampled / freeze-padded to match,
// so this is the user's length control over an IC-LoRA generation — AR +
// resolution stay reference-derived to keep the adapter on-distribution.
const LORA_DURATION_OPTIONS = [
  { value: '5', label: '5 Sec' },
  { value: '6', label: '6 Sec' },
  { value: '8', label: '8 Sec' },
  { value: '10', label: '10 Sec' },
]

// IC-LoRA OUTPUT resolution. Stage-1 diffusion always runs at the adapter's
// native 540p bucket (on-distribution, VRAM-safe); 720p/1080p are produced by
// the spatial upsampler (Stage 2, x2) rather than diffusing natively at high
// res — exactly like the t2v fast pipeline. A native 1080p first-pass overflows
// a 32GB GPU and crawls, so we never take that path. Defaults to 540p.
const LORA_RESOLUTION_OPTIONS = [
  { value: '540p', label: '540p' },
  { value: '720p', label: '720p' },
  { value: '1080p', label: '1080p' },
]

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

// Asset card with hover overlays
export function AssetCard({
  asset,
  onDelete,
  onPlay,
  onDragStart,
  onCreateVideo,
  onEditImage,
  onRetake,
  onApplyLora,
  onSendToLora,
  onToggleFavorite
}: {
  asset: Asset
  onDelete: () => void
  onPlay: () => void
  onDragStart: (e: React.DragEvent, asset: Asset) => void
  onCreateVideo?: (asset: Asset) => void
  onEditImage?: (asset: Asset) => void
  onRetake?: (asset: Asset) => void
  onApplyLora?: (asset: Asset) => void
  onSendToLora?: (asset: Asset) => void
  onToggleFavorite?: () => void
}) {
  const hoverVideoRef = useRef<HTMLVideoElement>(null)
  const [isHovered, setIsHovered] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isMuted, setIsMuted] = useState(true)
  const [volume, setVolume] = useState(0.5)
  const isFavorite = asset.favorite || false

  useEffect(() => {
    if (asset.type !== 'video') return
    if (!isHovered) {
      setCurrentTime(0)
      return
    }
    if (hoverVideoRef.current) {
      hoverVideoRef.current.muted = isMuted
      hoverVideoRef.current.volume = volume
      hoverVideoRef.current.play().catch(() => {})
    }
  }, [asset.type, isHovered, isMuted, volume])

  const handleTimeUpdate = () => {
    if (hoverVideoRef.current) {
      setCurrentTime(hoverVideoRef.current.currentTime)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation()
    const a = document.createElement('a')
    a.href = pathToFileUrl(asset.path)
    a.download = asset.path.split('/').pop() || `${asset.type}-${asset.id}`
    a.click()
  }

  return (
    <div
      className="relative group cursor-pointer rounded-xl overflow-hidden bg-zinc-900 asset-card-cq"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false)
        setCurrentTime(0)
      }}
      onClick={onPlay}
      draggable={asset.type === 'image'}
      onDragStart={(e) => asset.type === 'image' && onDragStart(e, asset)}
    >
      {asset.type === 'video' ? (
        <div className="relative w-full aspect-video bg-zinc-900">
          {asset.bigThumbnailPath && (
            <img
              src={pathToFileUrl(asset.bigThumbnailPath)}
              alt=""
              className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-150 ${
                isHovered ? 'opacity-0' : 'opacity-100'
              }`}
            />
          )}
          {isHovered && (
            <video
              ref={hoverVideoRef}
              src={pathToFileUrl(asset.path)}
              className="absolute inset-0 w-full h-full object-cover"
              muted={isMuted}
              loop
              autoPlay
              playsInline
              preload="metadata"
              onTimeUpdate={handleTimeUpdate}
            />
          )}
        </div>
      ) : (
        <img src={pathToFileUrl(asset.path)} alt="" className="w-full aspect-video object-contain" />
      )}
      
      {/* Favorite heart - always visible when favorited */}
      {isFavorite && !isHovered && (
        <button
          onClick={(e) => { e.stopPropagation(); onToggleFavorite?.() }}
          className="absolute top-2 left-2 p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white transition-colors z-10"
        >
          <Heart className="h-3.5 w-3.5 fill-current" />
        </button>
      )}
      
      {/* Hover overlay */}
      <div className={`absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/30 transition-opacity duration-200 group-focus-within:opacity-100 ${
        isHovered ? 'opacity-100' : 'opacity-0'
      }`}>
        {/* Top buttons */}
        <div className="absolute top-2 left-2 right-2 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 asset-card-top-row flex-1 min-w-0 overflow-x-auto no-scrollbar">
            <button
              onClick={(e) => { e.stopPropagation(); onToggleFavorite?.() }}
              className={`p-1.5 rounded-lg backdrop-blur-md transition-colors shrink-0 ${
                isFavorite ? 'bg-white/20 text-white' : 'bg-black/40 text-white hover:bg-black/60'
              }`}
            >
              <Heart className={`h-3.5 w-3.5 ${isFavorite ? 'fill-current' : ''}`} />
            </button>

            {asset.type === 'image' && (
              <>
                <button
                  onClick={(e) => { e.stopPropagation(); onEditImage?.(asset) }}
                  title="Edit image"
                  aria-label="Edit image"
                  className="asset-card-pill px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
                >
                  <Pencil className="h-3 w-3" />
                  <span className="asset-card-label">Edit Image</span>
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); onCreateVideo?.(asset) }}
                  title="Create video"
                  aria-label="Create video"
                  className="asset-card-pill px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap shrink-0"
                >
                  <Film className="h-3 w-3" />
                  <span className="asset-card-label">Create video</span>
                </button>
              </>
            )}
            {asset.type === 'video' && (
              <>
                <button
                  onClick={(e) => { e.stopPropagation(); onRetake?.(asset) }}
                  title="Retake"
                  className="asset-card-pill px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap shrink-0"
                >
                  <Scissors className="h-3 w-3" />
                  <span className="asset-card-label">Retake</span>
                </button>
                {onApplyLora && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onApplyLora(asset) }}
                    title="Apply LoRA"
                    className="asset-card-pill px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap shrink-0"
                  >
                    <Layers className="h-3 w-3" />
                    <span className="asset-card-label">Apply LoRA</span>
                  </button>
                )}
                {onSendToLora && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onSendToLora(asset) }}
                    title="Add this clip to a LoRA training dataset"
                    className="asset-card-pill px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap shrink-0"
                  >
                    <Wand2 className="h-3 w-3" />
                    <span className="asset-card-label">To LoRA</span>
                  </button>
                )}
              </>
            )}
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            <button
              onClick={handleDownload}
              className="p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
            </button>
            {/* Tools button hidden for now */}
          </div>
        </div>
        
        {/* Bottom controls for video */}
        {asset.type === 'video' && (
          <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5 asset-card-bottom-row">
              <div className="px-2 py-1 rounded-lg bg-black/50 backdrop-blur-md text-white text-xs font-mono">
                {formatTime(currentTime)}
              </div>
              <div className="flex items-center gap-1.5 rounded-lg bg-black/40 backdrop-blur-md pl-1.5 pr-2 py-1">
                <button
                  onClick={(e) => { e.stopPropagation(); setIsMuted(!isMuted) }}
                  className="text-white hover:text-white/80 transition-colors"
                  aria-label={isMuted ? 'Unmute' : 'Mute'}
                >
                  {isMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
                </button>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={isMuted ? 0 : volume}
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                  onChange={(e) => {
                    e.stopPropagation()
                    const next = parseFloat(e.target.value)
                    setVolume(next)
                    if (next === 0) {
                      setIsMuted(true)
                    } else if (isMuted) {
                      setIsMuted(false)
                    }
                  }}
                  className="asset-card-vol-slider w-16 h-1 accent-white cursor-pointer"
                  aria-label="Volume"
                />
              </div>
            </div>
          </div>
        )}

        {/* Delete button (subtle, bottom right) */}
        {(
          <button
            onClick={(e) => { e.stopPropagation(); onDelete() }}
            className="absolute bottom-2 right-2 p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white/70 hover:bg-red-500/80 hover:text-white transition-colors opacity-0 group-hover:opacity-100"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      
    </div>
  )
}

function AspectIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="5" width="18" height="14" rx="2" />
    </svg>
  )
}

// Prompt bar component matching the design
// One-shot "write my prompt" button for the IC-LoRA toolbars. Calls Gemini
// Flash with the reference video + the LoRA's per-LoRA system prompt and fills
// the prompt box. Disabled until a reference video is attached and a Gemini
// API key is configured.
function AutoPromptButton({
  onClick,
  loading,
  disabled,
  hasKey,
  hasRefVideo,
  iconOnly = false,
}: {
  onClick: () => void
  loading: boolean
  disabled: boolean
  hasKey: boolean
  hasRefVideo: boolean
  /** Compact icon-only render for the prompt-box overlay (no "Auto" label). */
  iconOnly?: boolean
}) {
  const title = !hasKey
    ? 'Add a Gemini API key in Settings to auto-write a prompt from the reference video.'
    : !hasRefVideo
      ? 'Attach a reference video to auto-write a prompt.'
      : 'Auto-write a prompt from the reference video using Gemini Flash.'
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      title={title}
      className={`flex shrink-0 items-center gap-1 rounded-md text-blue-300 hover:bg-blue-600/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent ${
        iconOnly ? 'p-1.5 bg-zinc-800/80 backdrop-blur-sm ring-1 ring-zinc-700' : 'px-2 py-1.5'
      }`}
    >
      {loading ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
      ) : (
        <Wand2 className="h-3.5 w-3.5" />
      )}
      {!iconOnly && <span className="text-[10px] uppercase tracking-wide">Auto</span>}
    </button>
  )
}

// One strength slider row: label + live numeric readout + range input.
// Replaces the old discrete LORA_STRENGTH_OPTIONS dropdowns with a continuous
// 0.00–2.00 control, which is friendlier in a combined popover.
function SliderRow({
  label,
  value,
  onChange,
  help,
}: {
  label: string
  value: number
  onChange: (next: number) => void
  help?: string
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-zinc-300" title={help}>{label}</span>
        <span className="text-xs text-zinc-400 font-medium tabular-nums">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={2}
        step={0.05}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 accent-blue-500 cursor-pointer"
      />
    </div>
  )
}

// Combined Strengths popover. For video_input_ic_lora it shows SCALE + COND
// (the "combine scale and condition strengths" control); for union_control it
// shows only the control strength (STR). Both reuse the existing handlers.
function StrengthsPopover({
  scale,
  onScaleChange,
  condStrength,
  onCondStrengthChange,
  showScale,
}: {
  scale: number
  onScaleChange: (next: number) => void
  condStrength: number
  onCondStrengthChange: (next: number) => void
  showScale: boolean
}) {
  const triggerTitle = showScale
    ? 'Strengths — LoRA scale and conditioning strength. Lower = subtler, higher = stronger.'
    : 'Control strength — how tightly the output follows the control signal. Lower = looser, higher = more rigid.'
  return (
    <PopoverMenu
      title="STRENGTHS"
      triggerTitle={triggerTitle}
      trigger={
        <>
          <SlidersHorizontal className="h-3.5 w-3.5" />
          {showScale ? (
            // Two sliders inside → a single number on the closed trigger would
            // be ambiguous (which one?), so show a neutral label instead.
            <span className="text-zinc-300 font-medium">Strengths</span>
          ) : (
            <>
              <span className="text-zinc-500 text-[10px]">STR</span>
              <span className="text-zinc-300 font-medium">{condStrength.toFixed(2)}</span>
            </>
          )}
          <ChevronUp className="h-3 w-3 text-zinc-500" />
        </>
      }
    >
      <div className="w-[200px] space-y-3">
        {showScale && (
          <SliderRow
            label="LoRA scale"
            value={scale}
            onChange={onScaleChange}
            help="How strongly the LoRA adapter weights influence the base model."
          />
        )}
        <SliderRow
          label={showScale ? 'Conditioning' : 'Control strength'}
          value={condStrength}
          onChange={onCondStrengthChange}
          help={
            showScale
              ? "How tightly the output follows the reference video's motion and structure."
              : 'How tightly the output follows the control signal.'
          }
        />
      </div>
    </PopoverMenu>
  )
}

// Settings popover: the two boolean output options as switches, replacing the
// old on-bar REFINE + AUDIO buttons. "Stage 2" is the renamed refine toggle.
function SettingsPopover({
  refine,
  onRefineChange,
  preserveAudio,
  onPreserveAudioChange,
}: {
  refine: boolean
  onRefineChange: (next: boolean) => void
  preserveAudio: boolean
  onPreserveAudioChange: (next: boolean) => void
}) {
  return (
    <PopoverMenu
      title="SETTINGS"
      triggerTitle="Output settings — Stage 2 refine and audio."
      trigger={
        <>
          <Settings className="h-3.5 w-3.5" />
          <ChevronUp className="h-3 w-3 text-zinc-500" />
        </>
      }
    >
      <div className="w-[230px] space-y-0.5">
        <Switch
          checked={refine}
          onChange={onRefineChange}
          label="Stage 2"
          description="2× upsample + detail pass after generation."
        />
        <Switch
          checked={preserveAudio}
          onChange={onPreserveAudioChange}
          label="Preserve audio"
          description="Use the reference video's audio, trimmed to output length."
        />
      </div>
    </PopoverMenu>
  )
}

// Two-row layout: prompt row on top, settings row below
function PromptBar({
  mode,
  onModeChange,
  prompt,
  onPromptChange,
  onGenerate,
  isGenerating,
  inputImage,
  onInputImageChange,
  inputAudio,
  onInputAudioChange,
  settings,
  onSettingsChange,
  videoModelSpecs,
  videoSettingsMessage,
  imageModelSpecs,
  selectedImageModelId,
  onSelectImageModel,
  onRefreshImageModelSpecs,
  imageSettingsMessage,
  imageEditModelSelected,
  canGenerate,
  buttonLabel,
  buttonIcon,
  selectedLora,
  forceApiGenerations,
  loraScale,
  onLoraScaleChange,
  loraCondType,
  onLoraCondTypeChange,
  loraCondStrength,
  onLoraCondStrengthChange,
  loraRefVideo,
  onLoraRefVideoChange,
  loraDuration,
  onLoraDurationChange,
  loraResolution,
  onLoraResolutionChange,
  loraRefine,
  onLoraRefineChange,
  preserveAudio,
  onPreserveAudioChange,
  onAutoPrompt,
  autoPromptLoading,
  autoPromptAvailable,
  autoPromptHasKey,
  loraPickerOpen,
  onToggleLoraPicker,
  onClearLora,
  onSelectLora,
  onDeletedLora,
  loraEntries,
  loraLoading,
  loraError,
  onRefreshLora,
  promptTextareaRef,
}: {
  mode: 'image' | 'video' | 'retake'
  onModeChange: (mode: 'image' | 'video' | 'retake') => void
  prompt: string
  onPromptChange: (prompt: string) => void
  onGenerate: () => void
  isGenerating: boolean
  canGenerate: boolean
  buttonLabel: string
  buttonIcon: React.ReactNode
  inputImage: string | null
  onInputImageChange: (path: string | null) => void
  inputAudio: string | null
  onInputAudioChange: (path: string | null) => void
  settings: {
    model: string
    duration: number
    videoResolution: string
    fps: number
    aspectRatio: string
    imageResolution: string
    variations: number
    audio?: boolean
  }
  onSettingsChange: (settings: any) => void
  videoModelSpecs: VideoGenerationModelSpecItem[]
  videoSettingsMessage?: string | null
  imageModelSpecs: ImageModelSpec[]
  selectedImageModelId: string
  onSelectImageModel: (id: string) => void
  onRefreshImageModelSpecs: () => void
  imageSettingsMessage: string | null
  // True when the selected image model is an instruction-based editing model
  // (FLUX.2 [klein] 9B). Shows the input-image slot in image mode so the user
  // can supply a reference image to edit.
  imageEditModelSelected: boolean
  selectedLora: LoraInferenceEntry | null
  forceApiGenerations: boolean
  loraScale: number
  onLoraScaleChange: (scale: number) => void
  loraCondType: LoraInferenceConditioningType
  onLoraCondTypeChange: (type: LoraInferenceConditioningType) => void
  loraCondStrength: number
  onLoraCondStrengthChange: (strength: number) => void
  loraRefVideo: string | null
  onLoraRefVideoChange: (path: string | null) => void
  loraDuration: number
  onLoraDurationChange: (duration: number) => void
  loraResolution: '540p' | '720p' | '1080p'
  onLoraResolutionChange: (resolution: '540p' | '720p' | '1080p') => void
  loraRefine: boolean
  onLoraRefineChange: (refine: boolean) => void
  preserveAudio: boolean
  onPreserveAudioChange: (preserve: boolean) => void
  onAutoPrompt: () => void
  autoPromptLoading: boolean
  autoPromptAvailable: boolean
  autoPromptHasKey: boolean
  loraPickerOpen: boolean
  onToggleLoraPicker: () => void
  onClearLora: () => void
  onSelectLora: (entry: LoraInferenceEntry) => void
  onDeletedLora?: (deletedId: string) => void
  loraEntries: LoraInferenceEntry[]
  loraLoading: boolean
  loraError: string | null
  onRefreshLora: () => void
  promptTextareaRef?: React.RefObject<HTMLTextAreaElement>
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isAudioDragOver, setIsAudioDragOver] = useState(false)
  const [isVideoDragOver, setIsVideoDragOver] = useState(false)
  const isRetake = mode === 'retake'
  // LoRAs that drive generation off a reference video replace the image/audio
  // I2V/A2V slots; a standard (style) LoRA stacks on top of a normal video gen
  // so the image/audio slots stay visible.
  const loraNeedsRefVideo = selectedLora !== null
    && (selectedLora.variant === 'union_control' || selectedLora.variant === 'video_input_ic_lora')
  const showImageAudioSlots = mode === 'video' && !isRetake && !loraNeedsRefVideo
  // Image editing (FLUX.2 Klein): show the input-image slot in image mode when
  // an edit model is selected, so the user can drop a reference image to edit.
  // Audio isn't relevant for image editing, so the audio slot stays video-only.
  const showEditImageSlot = mode === 'image' && imageEditModelSelected
  const showImageSlot = showImageAudioSlots || showEditImageSlot
  const showRefVideoSlot = mode === 'video' && !isRetake && loraNeedsRefVideo
  // Auto-prompt overlay lives on the prompt box (not the crowded bottom row)
  // and is only relevant for the reference-video LoRA variants.
  const showAutoPromptOverlay = showRefVideoSlot
  const resolvedVideoOptions = mode === 'video'
    ? resolveVideoGenerationOptions({
        settings,
        modelSpecs: videoModelSpecs,
        hasAudio: Boolean(inputAudio),
      })
    : null
  const showVideoFpsControl = Boolean(
    resolvedVideoOptions
    && resolvedVideoOptions.hasCompatibleOptions
    && resolvedVideoOptions.fpsOptions.length > 1,
  )

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      const asset = JSON.parse(assetData) as Asset
      if (asset.type === 'image') {
        onInputImageChange(asset.path)
      }
    }
  }

  const handleAudioDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsAudioDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      const asset = JSON.parse(assetData) as Asset
      if (asset.type === 'audio') {
        onInputAudioChange(asset.path)
      }
    }

    // Handle file drops
    const file = e.dataTransfer.files?.[0]
    if (file) {
      const ext = file.name.split('.').pop()?.toLowerCase()
      if (['mp3', 'wav', 'ogg', 'aac', 'flac', 'm4a'].includes(ext || '')) {
        const filePath = window.electronAPI?.getPathForFile(file)
        if (filePath) {
          onInputAudioChange(filePath)
        }
      }
    }
  }

  const handleAudioFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const filePath = window.electronAPI?.getPathForFile(file)
      if (filePath) {
        onInputAudioChange(filePath)
      }
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const filePath = window.electronAPI?.getPathForFile(file)
      if (filePath) {
        onInputImageChange(filePath)
      } else {
        const url = URL.createObjectURL(file)
        onInputImageChange(url)
      }
    }
  }
  
  const handleVideoFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('video/')) {
      const filePath = window.electronAPI?.getPathForFile(file)
      if (filePath) {
        onLoraRefVideoChange(filePath)
      }
    }
  }

  const handleVideoDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsVideoDragOver(false)
    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      const asset = JSON.parse(assetData) as Asset
      if (asset.type === 'video') {
        onLoraRefVideoChange(asset.path)
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isGenerating && canGenerate) {
      e.preventDefault()
      onGenerate()
    }
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-visible">
      {/* Top row: Image ref | Prompt | Generate */}
      <div className="flex items-start">
        {/* Input image drop zone — video I2V, or image editing (Klein) when an
            edit model is selected. Hidden when a LoRA that drives generation
            off a reference video is active. */}
        {showImageSlot && (
          <div
            role={showEditImageSlot ? 'group' : undefined}
            aria-label={showEditImageSlot ? (inputImage ? 'Selected image edit input' : 'Image edit input') : undefined}
            className={`relative w-10 h-10 mx-2 mt-2 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
              isDragOver ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700 hover:border-zinc-500'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            title={showEditImageSlot ? (inputImage ? 'Reference image — click to change' : 'Add a reference image to edit') : undefined}
          >
            {inputImage ? (
              <>
                <img src={pathToFileUrl(inputImage)} alt={showEditImageSlot ? 'Selected image edit input' : ''} className="w-full h-full object-cover rounded-md" />
                <button
                  onClick={(e) => { e.stopPropagation(); onInputImageChange(null) }}
                  aria-label={showEditImageSlot ? 'Remove image edit input' : 'Remove input image'}
                  className="absolute -top-1 -right-1 p-0.5 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3 w-3" />
                </button>
              </>
            ) : (
              <Image className="h-4 w-4 text-zinc-500" />
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        )}

        {/* Audio drop zone — only in video mode without a ref-video LoRA */}
        {showImageAudioSlots && (
          <div
            className={`relative w-10 h-10 mt-2 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
              isAudioDragOver ? 'border-emerald-500 bg-emerald-500/10' : inputAudio ? 'border-emerald-600' : 'border-zinc-700 hover:border-zinc-500'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsAudioDragOver(true) }}
            onDragLeave={() => setIsAudioDragOver(false)}
            onDrop={handleAudioDrop}
            onClick={() => audioInputRef.current?.click()}
            title={inputAudio ? 'Audio attached — click to change' : 'Attach audio for A2V'}
          >
            {inputAudio ? (
              <>
                <Music className="h-4 w-4 text-emerald-400" />
                <button
                  onClick={(e) => { e.stopPropagation(); onInputAudioChange(null) }}
                  className="absolute -top-1 -right-1 p-0.5 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3 w-3" />
                </button>
              </>
            ) : (
              <Music className="h-4 w-4 text-zinc-500" />
            )}
            <input
              ref={audioInputRef}
              type="file"
              accept=".mp3,.wav,.ogg,.aac,.flac,.m4a"
              onChange={handleAudioFileSelect}
              className="hidden"
            />
          </div>
        )}

        {/* Reference video slot — union_control / video_input_ic_lora LoRAs
            drive generation off a source clip (canny/depth/pose or raw video
            input). Drag a video asset here or click to pick a file. */}
        {showRefVideoSlot && (
          <div
            className={`relative w-10 h-10 mx-2 mt-2 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
              isVideoDragOver ? 'border-blue-500 bg-blue-500/10' : loraRefVideo ? 'border-blue-600' : 'border-zinc-700 hover:border-zinc-500'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsVideoDragOver(true) }}
            onDragLeave={() => setIsVideoDragOver(false)}
            onDrop={handleVideoDrop}
            onClick={() => videoInputRef.current?.click()}
            title={loraRefVideo ? 'Reference video attached — click to change' : 'Attach a reference video'}
          >
            {loraRefVideo ? (
              <>
                <video src={pathToFileUrl(loraRefVideo)} className="w-full h-full object-cover rounded-md" muted />
                <button
                  onClick={(e) => { e.stopPropagation(); onLoraRefVideoChange(null) }}
                  className="absolute -top-1 -right-1 p-0.5 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3 w-3" />
                </button>
              </>
            ) : (
              <Film className="h-4 w-4 text-zinc-500" />
            )}
            <input
              ref={videoInputRef}
              type="file"
              accept="video/*"
              onChange={handleVideoFileSelect}
              className="hidden"
            />
          </div>
        )}

        {/* Prompt input - fills remaining width */}
        <div className="relative flex-1 min-w-0 py-1">
          <textarea
            ref={promptTextareaRef}
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={mode === 'retake'
              ? "Describe what should happen in the selected section..."
              : loraNeedsRefVideo
                ? "Describe the style or transformation to apply to the reference video..."
              : mode === 'image'
                ? "A close-up of a woman talking on the phone..."
                : "The woman sips from a cup of coffee..."
            }
            className="w-full bg-transparent text-white text-sm placeholder:text-zinc-500 focus:outline-none px-2 py-2 pr-9 resize-none overflow-y-auto h-[70px] leading-5"
          />
          {showAutoPromptOverlay && (
            <div className="absolute top-1 right-1">
              <AutoPromptButton
                onClick={onAutoPrompt}
                loading={autoPromptLoading}
                disabled={!autoPromptAvailable}
                hasKey={autoPromptHasKey}
                hasRefVideo={!!loraRefVideo}
                iconOnly
              />
            </div>
          )}
        </div>

      </div>
      
      {/* Bottom row: Mode selector + Settings */}
      <div className="flex items-center gap-0.5 px-1.5 py-1.5 border-t border-zinc-800/60 text-xs text-zinc-400">
        {/* Mode dropdown */}
        <SettingsDropdown
          title="MODE"
          value={mode}
          onChange={(v) => onModeChange(v as 'image' | 'video' | 'retake')}
          options={[
            { value: 'image', label: 'Generate Images', icon: <Image className="h-4 w-4" /> },
            { value: 'video', label: 'Generate Videos', icon: <Video className="h-4 w-4" /> },
            { value: 'retake', label: 'Retake', icon: <Scissors className="h-4 w-4" /> },
          ]}
          trigger={
            <>
              {mode === 'image' ? <Image className="h-3.5 w-3.5" /> : mode === 'retake' ? <Scissors className="h-3.5 w-3.5" /> : <Video className="h-3.5 w-3.5" />}
              <span className="text-zinc-300 font-medium">{mode === 'image' ? 'Image' : mode === 'retake' ? 'Retake' : 'Video'}</span>
              <ChevronUp className="h-3 w-3 text-zinc-500" />
            </>
          }
        />

        {/* LoRA pill — opens the picker popover. Only relevant in video mode;
            hidden in retake/image where no adapter applies. Standard LoRAs
            stack on a normal video gen; union_control / video_input_ic_lora
            switch the prompt bar into reference-video mode. Hidden entirely
            when local generation is unavailable (forceApiGenerations): LoRAs
            run on the local GPU and can't be applied via the cloud API. */}
        {mode === 'video' && !isRetake && !forceApiGenerations && (
          <div className="relative flex items-center gap-0.5">
            <button
              type="button"
              onClick={onToggleLoraPicker}
              aria-haspopup="dialog"
              aria-expanded={loraPickerOpen}
              aria-label={selectedLora ? `Change applied LoRA, currently ${selectedLora.name}` : 'Apply a LoRA'}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md transition-colors max-w-[150px] ${
                selectedLora
                  ? 'bg-blue-600/20 text-blue-200 ring-1 ring-blue-500/40'
                  : 'bg-zinc-800/50 text-zinc-300 hover:bg-zinc-800'
              }`}
            >
              <Layers className="h-3.5 w-3.5 shrink-0" />
              {selectedLora ? (
                <Tooltip content={selectedLora.name} side="top">
                  <span className="font-medium max-w-[72px] min-w-0 truncate">{selectedLora.name}</span>
                </Tooltip>
              ) : (
                <span className="font-medium max-w-[72px] min-w-0 truncate">Apply LoRA</span>
              )}
            </button>
            {selectedLora && (
              <button
                type="button"
                onClick={onClearLora}
                aria-label={`Remove ${selectedLora.name}`}
                className="rounded-md p-1 text-blue-200 hover:bg-blue-500/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                title="Remove LoRA"
              >
                <X className="h-3 w-3" />
              </button>
            )}
            <LoraPickerPopover
              open={loraPickerOpen}
              selectedId={selectedLora?.id ?? null}
              conditioningType={loraCondType}
              entries={loraEntries}
              loading={loraLoading}
              error={loraError}
              onRefresh={onRefreshLora}
              onSelect={(value: LoraPickerValue) => {
                onSelectLora(value.entry)
                onLoraCondTypeChange(value.conditioningType)
                onToggleLoraPicker()
              }}
              onClose={onToggleLoraPicker}
              onDeleted={onDeletedLora}
            />
          </div>
        )}

        <div className="flex-1" />

        {isRetake ? (
          <div className="text-[10px] text-zinc-500 pr-2">Trim in the panel above, then retake</div>
        ) : mode === 'video' && selectedLora?.variant === 'union_control' ? (
          <>
            <SettingsDropdown
              title="DURATION"
              value={String(loraDuration)}
              onChange={(v) => onLoraDurationChange(parseInt(v))}
              options={LORA_DURATION_OPTIONS}
              tooltip="Output length — the generated video duration. The reference is resampled or freeze-padded to match."
              trigger={
                <>
                  <Clock className="h-3.5 w-3.5" />
                  <span>{loraDuration}s</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <SettingsDropdown
              title="RESOLUTION"
              value={loraResolution}
              onChange={(v) => onLoraResolutionChange(v as '540p' | '720p' | '1080p')}
              options={LORA_RESOLUTION_OPTIONS}
              tooltip="Output resolution. Every setting diffuses at the adapter's native 540p (on-distribution, fast); 720p/1080p are produced by the upsampler, so they stay VRAM-safe instead of diffusing natively at high res."
              trigger={
                <>
                  <Monitor className="h-3.5 w-3.5" />
                  <span>{loraResolution.replace('p', '')}</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <SettingsDropdown
              title="CONDITIONING TYPE"
              value={loraCondType}
              onChange={(v) => onLoraCondTypeChange(v as LoraInferenceConditioningType)}
              options={LORA_CONDITIONING_TYPES.map(ct => ({ value: ct.value, label: ct.label }))}
              tooltip="Control signal — the structural cue extracted from the reference (canny edges, depth, or pose) that the output follows."
              trigger={
                <>
                  <span className="text-zinc-300 font-medium">{LORA_CONDITIONING_TYPES.find(ct => ct.value === loraCondType)?.label || 'Canny'}</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <StrengthsPopover
              scale={loraScale}
              onScaleChange={onLoraScaleChange}
              condStrength={loraCondStrength}
              onCondStrengthChange={onLoraCondStrengthChange}
              showScale={false}
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <SettingsPopover
              refine={loraRefine}
              onRefineChange={onLoraRefineChange}
              preserveAudio={preserveAudio}
              onPreserveAudioChange={onPreserveAudioChange}
            />
          </>
        ) : mode === 'video' && selectedLora?.variant === 'video_input_ic_lora' ? (
          <>
            <SettingsDropdown
              title="DURATION"
              value={String(loraDuration)}
              onChange={(v) => onLoraDurationChange(parseInt(v))}
              options={LORA_DURATION_OPTIONS}
              tooltip="Output length — the generated video duration. The reference is resampled or freeze-padded to match."
              trigger={
                <>
                  <Clock className="h-3.5 w-3.5" />
                  <span>{loraDuration}s</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <SettingsDropdown
              title="RESOLUTION"
              value={loraResolution}
              onChange={(v) => onLoraResolutionChange(v as '540p' | '720p' | '1080p')}
              options={LORA_RESOLUTION_OPTIONS}
              tooltip="Output resolution. Every setting diffuses at the adapter's native 540p (on-distribution, fast); 720p/1080p are produced by the upsampler, so they stay VRAM-safe instead of diffusing natively at high res."
              trigger={
                <>
                  <Monitor className="h-3.5 w-3.5" />
                  <span>{loraResolution.replace('p', '')}</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <StrengthsPopover
              scale={loraScale}
              onScaleChange={onLoraScaleChange}
              condStrength={loraCondStrength}
              onCondStrengthChange={onLoraCondStrengthChange}
              showScale
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <SettingsPopover
              refine={loraRefine}
              onRefineChange={onLoraRefineChange}
              preserveAudio={preserveAudio}
              onPreserveAudioChange={onPreserveAudioChange}
            />
          </>
        ) : mode === 'image' ? (
          <>
            {/* Model picker — replaces the static Z-Image badge. Lists every
                catalogued image model with per-row download + tooltips. */}
            <ImageModelPicker
              specs={imageModelSpecs}
              selectedId={selectedImageModelId}
              onSelect={onSelectImageModel}
              onSpecsChanged={onRefreshImageModelSpecs}
            />

            {/* Coming-soon pill for a downloaded model without inference yet. */}
            {(() => {
              const spec = imageModelSpecs.find((s) => s.id === selectedImageModelId)
              if (spec?.downloaded && spec.inference_status === 'coming_soon') {
                return (
                  <span className="flex items-center gap-1 px-2 py-1 rounded-md bg-amber-500/15 text-amber-300 text-[11px] font-medium ring-1 ring-amber-500/30">
                    Coming soon
                  </span>
                )
              }
              return null
            })()}

            {imageSettingsMessage && (
              <span className="text-[10px] text-zinc-500 max-w-[180px] truncate" title={imageSettingsMessage}>
                {imageSettingsMessage}
              </span>
            )}

            {/* Resolution dropdown */}
            <SettingsDropdown
              title="IMAGE RESOLUTION"
              value={settings.imageResolution}
              onChange={(v) => onSettingsChange({ ...settings, imageResolution: v })}
              options={[
                { value: '1080p', label: '1080p' },
                { value: '1440p', label: '1440p' },
                { value: '2048p', label: '2048p' },
              ]}
              trigger={
                <>
                  <Monitor className="h-3.5 w-3.5" />
                  <span>{settings.imageResolution.replace('p', '')}</span>
                </>
              }
            />
            
            {/* Aspect ratio dropdown */}
            <SettingsDropdown
              title="RATIO"
              value={settings.aspectRatio}
              onChange={(v) => onSettingsChange({ ...settings, aspectRatio: v })}
              options={[
                { value: '16:9', label: '16:9' },
                { value: '1:1', label: '1:1' },
                { value: '9:16', label: '9:16' },
              ]}
              trigger={
                <>
                  <AspectIcon className="h-3.5 w-3.5" />
                  <span>{settings.aspectRatio}</span>
                </>
              }
            />
            
          </>
        ) : (
          <>
            {/* Standard (style) LoRA: show its scale alongside the normal
                video model/duration/resolution controls — the adapter stacks
                on top of a regular fast-video generation. */}
            {selectedLora?.variant === 'standard' && (
              <>
                <SettingsDropdown
                  title="LORA SCALE"
                  value={String(loraScale)}
                  onChange={(v) => onLoraScaleChange(parseFloat(v))}
                  options={LORA_STRENGTH_OPTIONS}
                  tooltip="LoRA scale — how strongly the LoRA adapter weights influence the base model. Lower = subtler, higher = stronger stylization."
                  trigger={
                    <>
                      <span className="text-zinc-500 text-[10px]">SCALE</span>
                      <span className="text-zinc-300 font-medium">{loraScale.toFixed(2)}</span>
                      <ChevronUp className="h-3 w-3 text-zinc-500" />
                    </>
                  }
                />
                <div className="w-px h-4 bg-zinc-700 mx-0.5" />
              </>
            )}
            {resolvedVideoOptions && resolvedVideoOptions.hasCompatibleOptions ? (
              <>
                <SettingsDropdown
                  title="MODEL"
                  value={resolvedVideoOptions.selectedModel ?? settings.model}
                  onChange={(v) => onSettingsChange({ ...settings, model: v })}
                  options={resolvedVideoOptions.modelOptions.map((item) => ({
                    value: item.pipeline,
                    label: item.spec.display_name,
                  }))}
                  trigger={
                    <>
                      <LightricksIcon className="h-3.5 w-3.5" />
                      <span className="text-zinc-300 font-medium">
                        {resolvedVideoOptions.modelOptions.find((item) => item.pipeline === resolvedVideoOptions.selectedModel)?.spec.display_name
                          ?? settings.model}
                      </span>
                    </>
                  }
                />

                <div className="w-px h-4 bg-zinc-700 mx-0.5" />

                <SettingsDropdown
                  title="DURATION"
                  value={String(resolvedVideoOptions.selectedDuration ?? settings.duration)}
                  onChange={(v) => onSettingsChange({ ...settings, duration: parseInt(v) })}
                  options={resolvedVideoOptions.durationOptions.map((value) => ({ value: String(value), label: `${value} Sec` }))}
                  trigger={
                    <>
                      <Clock className="h-3.5 w-3.5" />
                      <span>{resolvedVideoOptions.selectedDuration ?? settings.duration}s</span>
                    </>
                  }
                />

                <SettingsDropdown
                  title="RESOLUTION"
                  value={resolvedVideoOptions.selectedResolution ?? settings.videoResolution}
                  onChange={(v) => onSettingsChange({ ...settings, videoResolution: v })}
                  options={resolvedVideoOptions.resolutionOptions.map((value) => ({ value, label: value }))}
                  trigger={
                    <>
                      <Monitor className="h-3.5 w-3.5" />
                      <span>{(resolvedVideoOptions.selectedResolution ?? settings.videoResolution).replace('p', '')}</span>
                    </>
                  }
                />

                {showVideoFpsControl && (
                  <SettingsDropdown
                    title="FPS"
                    value={String(resolvedVideoOptions.selectedFps ?? settings.fps)}
                    onChange={(v) => onSettingsChange({ ...settings, fps: parseInt(v) })}
                    options={resolvedVideoOptions.fpsOptions.map((value) => ({ value: String(value), label: `${value}` }))}
                    trigger={
                      <>
                        <Film className="h-3.5 w-3.5" />
                        <span>{resolvedVideoOptions.selectedFps ?? settings.fps} FPS</span>
                      </>
                    }
                  />
                )}

                <SettingsDropdown
                  title="ASPECT RATIO"
                  value={settings.aspectRatio}
                  onChange={(v) => onSettingsChange({ ...settings, aspectRatio: v })}
                  options={[
                    { value: '16:9', label: '16:9' },
                    { value: '9:16', label: '9:16' },
                  ]}
                  trigger={
                    <>
                      <AspectIcon className="h-3.5 w-3.5" />
                      <span>{settings.aspectRatio}</span>
                    </>
                  }
                />
              </>
            ) : (
              <div className="px-2 py-1.5 rounded-md bg-zinc-800/60 text-zinc-500 text-xs">
                {videoSettingsMessage || 'Loading generation settings...'}
              </div>
            )}
          </>
        )}
        
        {/* Generate button */}
        <button
          onClick={onGenerate}
          disabled={isGenerating || !canGenerate}
          className={`flex items-center gap-1.5 ml-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all flex-shrink-0 ${
            isGenerating || !canGenerate
              ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
              : 'bg-white text-black hover:bg-zinc-200'
          }`}
        >
          <span className={isGenerating ? 'animate-pulse' : ''}>{buttonIcon}</span>
          {buttonLabel}
        </button>
      </div>
    </div>
  )
}

// Gallery size icon components
function GridSmallIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="2" width="4" height="4" rx="0.5" />
      <rect x="8" y="2" width="4" height="4" rx="0.5" />
      <rect x="14" y="2" width="4" height="4" rx="0.5" />
      <rect x="20" y="2" width="2" height="4" rx="0.5" />
      <rect x="2" y="8" width="4" height="4" rx="0.5" />
      <rect x="8" y="8" width="4" height="4" rx="0.5" />
      <rect x="14" y="8" width="4" height="4" rx="0.5" />
      <rect x="20" y="8" width="2" height="4" rx="0.5" />
      <rect x="2" y="14" width="4" height="4" rx="0.5" />
      <rect x="8" y="14" width="4" height="4" rx="0.5" />
      <rect x="14" y="14" width="4" height="4" rx="0.5" />
      <rect x="20" y="14" width="2" height="4" rx="0.5" />
    </svg>
  )
}

function GridMediumIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="2" width="6" height="6" rx="1" />
      <rect x="10" y="2" width="6" height="6" rx="1" />
      <rect x="18" y="2" width="4" height="6" rx="1" />
      <rect x="2" y="10" width="6" height="6" rx="1" />
      <rect x="10" y="10" width="6" height="6" rx="1" />
      <rect x="18" y="10" width="4" height="6" rx="1" />
      <rect x="2" y="18" width="6" height="4" rx="1" />
      <rect x="10" y="18" width="6" height="4" rx="1" />
      <rect x="18" y="18" width="4" height="4" rx="1" />
    </svg>
  )
}

function GridLargeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="2" width="9" height="9" rx="1.5" />
      <rect x="13" y="2" width="9" height="9" rx="1.5" />
      <rect x="2" y="13" width="9" height="9" rx="1.5" />
      <rect x="13" y="13" width="9" height="9" rx="1.5" />
    </svg>
  )
}

type GallerySize = 'small' | 'medium' | 'large'

const gallerySizeClasses: Record<GallerySize, string> = {
  small: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-7',
  medium: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5',
  large: 'grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3',
}

const DEFAULT_VIDEO_SETTINGS = {
  model: 'fast',
  duration: 5,
  videoResolution: '540p',
  fps: 24,
  aspectRatio: '16:9',
  imageResolution: '1080p',
  variations: 1,
  audio: true,
}

export function GenSpace() {
  const {
    activeProject,
    addAsset,
    addTakeToAsset,
    deleteAsset,
    toggleFavorite,
    genSpaceEditImagePath,
    setGenSpaceEditImagePath,
    setGenSpaceEditMode,
    genSpaceAudioPath,
    setGenSpaceAudioPath,
    genSpaceRetakeSource,
    setGenSpaceRetakeSource,
    setPendingRetakeUpdate,
    genSpaceIcLoraSource,
    setGenSpaceIcLoraSource,
    genSpaceLoraSource,
    setGenSpaceLoraSource,
    currentTab,
  } = useProjects()
  const currentProjectId = activeProject?.id ?? null
  const { shouldVideoGenerateWithLtxApi, forceApiGenerations, settings: appSettings } = useAppSettings()
  const {
    modelSpecs: videoGenerationModelSpecsResponse,
    isLoading: isLoadingVideoGenerationModelSpecs,
    errorMessage: videoGenerationModelSpecsErrorMessage,
  } = useVideoGenerationModelSpecs()
  const {
    modelSpecs: imageGenerationModelSpecsResponse,
    refresh: refreshImageModelSpecs,
  } = useImageGenerationModelSpecs()
  const imageModelSpecs: ImageModelSpec[] = imageGenerationModelSpecsResponse?.models ?? []
  const [imageModelId, setImageModelId] = useState<string>('z-image-turbo')
  const selectedImageSpec = imageModelSpecs.find((s) => s.id === imageModelId) ?? null
  // Generate is gated when the selected image model isn't downloaded yet, or
  // its inference isn't wired (coming-soon). Surfaced as a message under the
  // picker and via canSubmit so the button is disabled with an honest reason.
  const imageSettingsMessage: string | null = (() => {
    if (!selectedImageSpec) return null
    if (!selectedImageSpec.downloaded) {
      return `Download ${selectedImageSpec.display_name} from the model picker to generate.`
    }
    if (selectedImageSpec.inference_status === 'coming_soon') {
      return `${selectedImageSpec.display_name} is downloaded, but inference is coming soon.`
    }
    return null
  })()
  const canGenerateImage = Boolean(selectedImageSpec?.downloaded && selectedImageSpec.inference_status === 'available')
  const [mode, setMode] = useState<'image' | 'video' | 'retake'>('video')
  const [prompt, setPrompt] = useState('')
  const promptComposerRef = useRef<HTMLTextAreaElement>(null)
  const [inputImage, setInputImage] = useState<string | null>(null)
  const [inputAudio, setInputAudio] = useState<string | null>(null)
  const [localError, setLocalError] = useState<GenerationError | null>(null)
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null)
  const [sendToLoraAsset, setSendToLoraAsset] = useState<{ path: string; prompt?: string } | null>(null)
  const [copiedPrompt, setCopiedPrompt] = useState(false)
  // Result viewer mode for IC-LoRA outputs: "result" (default), "before-after"
  // (reference | output side by side), or "control" (union control video).
  const [resultViewMode, setResultViewMode] = useState<'result' | 'before-after' | 'control'>('result')

  // IC-LoRA result extras (before/after + control views). Populated by the
  // queue asset router for union_control / video_input_ic_lora outputs.
  const referenceVideoPath = selectedAsset?.generationParams?.referenceVideoPath
  const controlVideoPath = selectedAsset?.generationParams?.controlVideoPath
  const hasIcLoraViews = !!(referenceVideoPath || controlVideoPath)
  // Union-control LoRAs extract a control signal (canny/depth/pose) from the
  // driving video. When one is present, the before/after view becomes a
  // three-column driving | control | output comparison instead of the usual
  // two-column reference | output.
  const isUnionControl = selectedAsset?.generationParams?.loraVariant === 'union_control'
  const showControlInBa = isUnionControl && !!controlVideoPath

  useEffect(() => {
    setResultViewMode('result')
  }, [selectedAsset?.id])

  // Unified before/after playback: a single play/pause + scrubber drives both
  // videos in sync. The output video is the timeline source of truth; the
  // reference is seeked to match (clamped to its own duration).
  const baLeftRef = useRef<HTMLVideoElement>(null)
  const baRightRef = useRef<HTMLVideoElement>(null)
  // Union-control before/after adds a third, middle column: the control
  // signal (canny/depth/pose) extracted from the driving video. It plays in
  // sync with the reference + output, so it needs its own ref driven by the
  // same play/seek handlers.
  const baControlRef = useRef<HTMLVideoElement>(null)
  const [baPlaying, setBaPlaying] = useState(false)
  const [baTime, setBaTime] = useState(0)
  const [baDuration, setBaDuration] = useState(0)
  const [baExporting, setBaExporting] = useState(false)
  const comparisonViewerRef = useRef<HTMLDivElement>(null)
  const [baFullscreen, setBaFullscreen] = useState(false)
  const [baFullscreenError, setBaFullscreenError] = useState<string | null>(null)

  useEffect(() => {
    const handleFullscreenChange = () => {
      setBaFullscreen(document.fullscreenElement === comparisonViewerRef.current)
      setBaFullscreenError(null)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  const toggleBaFullscreen = useCallback(async () => {
    setBaFullscreenError(null)
    const result = await toggleElementFullscreen(comparisonViewerRef.current)
    if (result === 'unavailable') {
      setBaFullscreenError('Fullscreen is not available in this window.')
    } else if (result === 'failed') {
      logger.warn('Before/after fullscreen request failed')
      setBaFullscreenError('Could not enter fullscreen. Playback is still available.')
    }
  }, [])

  const toggleBaPlay = useCallback(() => {
    const right = baRightRef.current
    const left = baLeftRef.current
    const control = baControlRef.current
    if (!right) return
    if (right.paused) {
      void right.play()
      void left?.play()
      void control?.play()
    } else {
      right.pause()
      left?.pause()
      control?.pause()
    }
  }, [])

  const seekBa = useCallback((t: number) => {
    const right = baRightRef.current
    const left = baLeftRef.current
    const control = baControlRef.current
    if (!right) return
    right.currentTime = t
    if (left) left.currentTime = Math.min(t, left.duration || t)
    if (control) control.currentTime = Math.min(t, control.duration || t)
    setBaTime(t)
  }, [])

  const handleDownloadSideBySide = useCallback(async () => {
    if (!referenceVideoPath || !selectedAsset || baExporting) return
    const dest = await window.electronAPI.showSaveDialog({
      title: 'Save side-by-side export',
      defaultPath: `before-after-${selectedAsset.id}.mp4`,
      filters: [{ name: 'MP4 Video', extensions: ['mp4'] }],
    })
    if (!dest) return
    setBaExporting(true)
    try {
      const result = await window.electronAPI.exportSideBySide({
        leftPath: referenceVideoPath,
        rightPath: selectedAsset.path,
        outputPath: dest,
      })
      if (!result.success) {
        logger.error(`Side-by-side export failed: ${result.error}`)
      }
    } catch (err) {
      logger.error(`Side-by-side export failed: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setBaExporting(false)
    }
  }, [referenceVideoPath, selectedAsset, baExporting])
  const [showFavorites, setShowFavorites] = useState(false)
  const [gallerySize, setGallerySize] = useState<GallerySize>('medium')
  const [showSizeMenu, setShowSizeMenu] = useState(false)
  const sizeMenuRef = useRef<HTMLDivElement>(null)
  const persistedVideoKeyRef = useRef<string | null>(null)
  const retakeSubmissionRef = useRef<{
    prompt: string
    input: {
      videoPath: string | null
      startTime: number
      duration: number
      videoDuration: number
    }
  } | null>(null)
  const [settings, setSettings] = useState(() => ({ ...DEFAULT_VIDEO_SETTINGS }))
  const videoModelSpecs = getVideoGenerationModelSpecs(videoGenerationModelSpecsResponse, {
    useApiSpecs: shouldVideoGenerateWithLtxApi,
  })
  const videoSettingsMessage = isLoadingVideoGenerationModelSpecs
    ? 'Loading generation settings...'
    : videoGenerationModelSpecsErrorMessage
      ? `Could not load generation settings: ${videoGenerationModelSpecsErrorMessage}`
      : null
  const sanitizeVideoSettings = useCallback(
    (next: typeof settings) => {
      if (mode !== 'video' || videoModelSpecs.length === 0) return next
      return sanitizeVideoGenerationSettings(next, videoModelSpecs, {
        hasAudio: Boolean(inputAudio),
      }) ?? next
    },
    [inputAudio, mode, videoModelSpecs],
  )
  
  const {
    isGenerating,
    progress,
    statusMessage,
    videoPath,
    imagePaths,
    error,
    reset,
  } = useGeneration()

  // Durable generation queue: the main Generate button enqueues video/image
  // work instead of blocking on the synchronous single-flight path, so the
  // user can line up generations and keep working. Retake / IC-LoRA keep
  // their own synchronous flows; the queue runner cooperates with them via
  // the shared single-flight slot. Completed queue items are routed into
  // the project by the global QueueAssetRouter.
  const { enqueue } = useQueue()

  const {
    submitRetake,
    resetRetake,
    isRetaking,
    retakeStatus,
    retakeError,
    retakeResult,
  } = useRetake()

  const [retakeInput, setRetakeInput] = useState({
    videoPath: null as string | null,
    startTime: 0,
    duration: 0,
    videoDuration: 0,
    ready: false,
  })
  const [retakePanelKey, setRetakePanelKey] = useState(0)
  const [retakeInitial, setRetakeInitial] = useState<{
    videoPath: string | null
    duration?: number
  }>({ videoPath: null, duration: undefined })
  const [activeRetakeSource, setActiveRetakeSource] = useState<GenSpaceRetakeSource | null>(null)

  // ----- In-app LoRA inference (Gen Space "Apply LoRA") -----
  // The registry is the picker source (official union IC-LoRA + user-trained
  // adapters). One fetch shared with the PromptBar pill and the asset-card
  // action. Selection drives a `kind: 'lora'` queue payload instead of the
  // old synchronous IC-LoRA flow.
  const loraRegistry = useLoraInferenceRegistry()
  const [selectedLora, setSelectedLora] = useState<LoraInferenceEntry | null>(null)
  const [loraScale, setLoraScale] = useState(1.0)
  const [loraCondType, setLoraCondType] = useState<LoraInferenceConditioningType>('canny')
  const [loraCondStrength, setLoraCondStrength] = useState(1.0)
  const [loraRefVideo, setLoraRefVideo] = useState<string | null>(null)
  const [loraDuration, setLoraDuration] = useState(5)
  const [loraResolution, setLoraResolution] = useState<'540p' | '720p' | '1080p'>('540p')
  const [loraPreserveAudio, setLoraPreserveAudio] = useState(false)
  const [loraRefine, setLoraRefine] = useState(false)
  const [loraPickerOpen, setLoraPickerOpen] = useState(false)
  const [autoPromptLoading, setAutoPromptLoading] = useState(false)
  // Union-control IC-LoRAs need preprocessing models on disk (the union
  // adapter + MiDaS depth, and for pose the DW processor + YOLOX person
  // detector). Gate generation until they're downloaded — mirrors the old
  // ICLoraPanel download flow, now driven from the PromptBar LoRA pill.
  const unionGateActive = !forceApiGenerations
    && mode === 'video'
    && selectedLora?.variant === 'union_control'
  const icLoraModelGate = useIcLoraModelGate(unionGateActive)
  // When the editor bridges a clip in ("Apply LoRA" from a timeline clip), we
  // remember the source so the result can land as a take on the source asset
  // once the queue item completes. (Take-replacement routing for `kind: 'lora'
  // ` queue items is a follow-up; the value is retained here for that wiring.)
  const [, setActiveLoraSource] = useState<{
    assetId?: string
    linkedClipIds?: string[]
  } | null>(null)

  const clearLora = useCallback(() => {
    setSelectedLora(null)
    setLoraRefVideo(null)
    setLoraPickerOpen(false)
    setActiveLoraSource(null)
  }, [])

  const selectLora = useCallback((entry: LoraInferenceEntry) => {
    setSelectedLora(entry)
    setLoraPickerOpen(false)
    // Coerce the conditioning type to one the entry actually supports (the
    // official union IC-LoRA advertises canny/depth/pose; user adapters vary).
    if (entry.variant === 'union_control' && !entry.conditioningTypes.includes(loraCondType)) {
      setLoraCondType(entry.conditioningTypes[0] ?? 'canny')
    }
  }, [loraCondType])
  
  // Handle incoming frame from the Video Editor for editing
  useEffect(() => {
    if (genSpaceEditImagePath) {
      setMode('video')
      setInputImage(genSpaceEditImagePath)
      setPrompt('')
      setGenSpaceEditImagePath(null)
      setGenSpaceEditMode(null)
    }
  }, [genSpaceEditImagePath, setGenSpaceEditImagePath, setGenSpaceEditMode])

  // Handle incoming audio from the Video Editor for A2V
  useEffect(() => {
    if (genSpaceAudioPath) {
      setMode('video')
      setInputAudio(genSpaceAudioPath)
      setPrompt('')
      setGenSpaceAudioPath(null)
    }
  }, [genSpaceAudioPath, setGenSpaceAudioPath])

  useEffect(() => {
    if (!genSpaceRetakeSource) return
    setMode('retake')
    setPrompt('')
    setActiveRetakeSource(genSpaceRetakeSource)
    setRetakeInitial({
      videoPath: genSpaceRetakeSource.videoPath,
      duration: genSpaceRetakeSource.duration,
    })
    setRetakePanelKey((prev) => prev + 1)
    setGenSpaceRetakeSource(null)
  }, [genSpaceRetakeSource, setGenSpaceRetakeSource])

  // Bridge from the Video Editor ("Apply LoRA" on a timeline clip): drop the
  // clip into the reference-video slot, switch to video mode, and preselect
  // the official union IC-LoRA if it's already in the registry. The result
  // lands as a fresh asset via the queue; automatic take-replacement of the
  // source clip is a follow-up (the queue router doesn't carry linkedClipIds).
  useEffect(() => {
    if (!genSpaceIcLoraSource) return
    if (forceApiGenerations) {
      setGenSpaceIcLoraSource(null)
      return
    }
    setMode('video')
    setPrompt('')
    setLoraRefVideo(genSpaceIcLoraSource.videoPath)
    setActiveLoraSource({
      assetId: genSpaceIcLoraSource.assetId,
      linkedClipIds: genSpaceIcLoraSource.linkedClipIds,
    })
    const unionEntry = loraRegistry.entries.find(
      (e) => e.kind === 'official_union' && e.variant === 'union_control' && e.available,
    )
    if (unionEntry) {
      selectLora(unionEntry)
    } else {
      // Registry not loaded yet (or no official LoRA) — surface the picker so
      // the user can choose an adapter for the dropped clip.
      setLoraPickerOpen(true)
    }
    setGenSpaceIcLoraSource(null)
  }, [genSpaceIcLoraSource, forceApiGenerations, setGenSpaceIcLoraSource, loraRegistry.entries, selectLora])

  // LoRAs are local-only (standard) or run on the local IC-LoRA pipeline; the
  // API generation path can't apply them, so drop the selection when forced.
  useEffect(() => {
    if (forceApiGenerations && selectedLora) {
      clearLora()
    }
  }, [forceApiGenerations, selectedLora, clearLora])

  // GenSpace stays mounted while hidden (see Project.tsx), so the registry's
  // mount-time fetch only runs once — before a training job that completes on
  // the LoRA Trainer tab has written its adapter. Refetch whenever the user
  // lands on Gen Space so a freshly trained LoRA appears in the picker (and the
  // "Try in Gen Space" bridge below can resolve it) without an app restart.
  useEffect(() => {
    if (currentTab !== 'gen-space') return
    void loraRegistry.refresh()
  }, [currentTab, loraRegistry.refresh])

  // Bridge from the LoRA trainer RunView ("Try in Gen Space"): preselect the
  // adapter by its registry id. The registry may still be loading when the
  // bridge fires, so we depend on `loraRegistry.entries` and resolve once it
  // arrives. Clearing the source prevents re-selection on later reloads.
  useEffect(() => {
    if (!genSpaceLoraSource) return
    if (forceApiGenerations) {
      setGenSpaceLoraSource(null)
      return
    }
    const entry = loraRegistry.entries.find((e) => e.id === genSpaceLoraSource.loraId)
    if (entry) {
      setMode('video')
      setPrompt('')
      selectLora(entry)
      setGenSpaceLoraSource(null)
    }
  }, [genSpaceLoraSource, forceApiGenerations, setGenSpaceLoraSource, loraRegistry.entries, selectLora])

  useEffect(() => {
    if (mode !== 'video' || videoModelSpecs.length === 0) return
    setSettings((prev) => {
      const next = sanitizeVideoSettings(prev)
      return areVideoGenerationSettingsEquivalent(prev, next) ? prev : next
    })
  }, [mode, sanitizeVideoSettings, videoModelSpecs.length])

  useEffect(() => {
    if (retakeError) {
      setLocalError(createLocalGenerationError(retakeError))
    }
  }, [retakeError])

  // Only show assets that were generated (have generationParams), not imported files
  const assets = (activeProject?.assets || []).filter(a => a.generationParams)
  const [lastPrompt, setLastPrompt] = useState('')
  
  // When video generation completes, add to project assets
  useEffect(() => {
    if (!videoPath || !currentProjectId || isGenerating) return

    const generationKey = videoPath
    if (persistedVideoKeyRef.current === generationKey) return
    persistedVideoKeyRef.current = generationKey

    const genMode = inputAudio
      ? 'audio-to-video'
      : inputImage ? 'image-to-video' : 'text-to-video'
    const savedVideoSettings = sanitizeVideoSettings(settings)

    ;(async () => {
      try {
        const copied = await addVisualAssetToProject(videoPath, currentProjectId, 'video')
        if (!copied) throw new Error('Could not persist generated video to project storage')
        addAsset(currentProjectId, {
          type: 'video',
          path: copied.path,
          bigThumbnailPath: copied.bigThumbnailPath,
          smallThumbnailPath: copied.smallThumbnailPath,
          width: copied.width,
          height: copied.height,
          prompt: lastPrompt,
          resolution: savedVideoSettings.videoResolution,
          duration: savedVideoSettings.duration,
          generationParams: {
            mode: genMode as 'text-to-video' | 'image-to-video' | 'audio-to-video',
            prompt: lastPrompt,
            model: savedVideoSettings.model,
            duration: savedVideoSettings.duration,
            resolution: savedVideoSettings.videoResolution,
            fps: savedVideoSettings.fps,
            audio: savedVideoSettings.audio || false,
            cameraMotion: 'none',
            imageAspectRatio: savedVideoSettings.aspectRatio,
            imageSteps: 4,
            inputImageUrl: inputImage || undefined,
            inputAudioUrl: inputAudio || undefined,
          },
          takes: [{
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          }],
          activeTakeIndex: 0,
        })
        reset()
      } catch (err) {
        persistedVideoKeyRef.current = null
        logger.error(`Failed to persist generated video asset: ${err}`)
      }
    })()
  }, [videoPath, currentProjectId, isGenerating, sanitizeVideoSettings, settings, inputImage, inputAudio, lastPrompt, addAsset, reset])

  // When retake completes, add as take or new asset
  useEffect(() => {
    if (!retakeResult || !currentProjectId || isRetaking) return
    const submission = retakeSubmissionRef.current
    if (!submission) return
    retakeSubmissionRef.current = null

    ;(async () => {
      const usedPrompt = submission.prompt
      const usedInput = submission.input
      const copied = await addVisualAssetToProject(retakeResult.videoPath, currentProjectId, 'video')
      if (!copied) {
        logger.error('Could not persist retake result to project storage')
        setLocalError(createLocalGenerationError('Failed to save retake output to project storage.'))
        setActiveRetakeSource(null)
        resetRetake()
        return
      }

      if (activeRetakeSource?.assetId) {
        const sourceAsset = activeProject?.assets?.find(a => a.id === activeRetakeSource.assetId)
        if (sourceAsset) {
          const newTakeIndex = sourceAsset.takes ? sourceAsset.takes.length : 1
          addTakeToAsset(currentProjectId, sourceAsset.id, {
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          })
          if (activeRetakeSource.linkedClipIds?.length) {
            setPendingRetakeUpdate({
              assetId: sourceAsset.id,
              clipIds: activeRetakeSource.linkedClipIds,
              newTakeIndex,
            })
          }
        }
      } else {
        addAsset(currentProjectId, {
          type: 'video',
          path: copied.path,
          bigThumbnailPath: copied.bigThumbnailPath,
          smallThumbnailPath: copied.smallThumbnailPath,
          width: copied.width,
          height: copied.height,
          prompt: usedPrompt,
          resolution: '',
          duration: usedInput.duration,
          generationParams: {
            mode: 'retake',
            prompt: usedPrompt,
            model: 'pro',
            duration: usedInput.duration,
            resolution: '',
            fps: 24,
            audio: true,
            cameraMotion: 'none',
            retakeVideoPath: copied.path,
            retakeStartTime: usedInput.startTime,
            retakeDuration: usedInput.duration,
            retakeMode: 'replace_audio_and_video',
          },
          takes: [{
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          }],
          activeTakeIndex: 0,
        })
        setMode('video')
      }

      setActiveRetakeSource(null)
      resetRetake()
    })()
  }, [retakeResult, isRetaking, currentProjectId, activeProject?.assets, activeRetakeSource, addAsset, addTakeToAsset, setPendingRetakeUpdate, resetRetake])

  // When image generation/editing completes, add all images to project assets
  useEffect(() => {
    if (imagePaths.length > 0 && currentProjectId && !isGenerating) {
      const genMode = 'text-to-image'
      ;(async () => {
        for (let i = 0; i < imagePaths.length; i++) {
          const imgPath = imagePaths[i]
          const exists = assets.some(a => a.path === imgPath)
          if (!exists) {
            const copied = await addVisualAssetToProject(imgPath, currentProjectId, 'image')
            if (!copied) {
              logger.error(`Could not persist generated image to project storage: ${imgPath}`)
              continue
            }
            addAsset(currentProjectId, {
              type: 'image',
              path: copied.path,
              bigThumbnailPath: copied.bigThumbnailPath,
              smallThumbnailPath: copied.smallThumbnailPath,
              width: copied.width,
              height: copied.height,
              prompt: lastPrompt,
              resolution: settings.imageResolution,
              generationParams: {
                mode: genMode,
                prompt: lastPrompt,
                model: 'fast',
                duration: 5,
                resolution: settings.imageResolution,
                fps: 24,
                audio: false,
                cameraMotion: 'none',
                imageAspectRatio: settings.aspectRatio,
                imageSteps: 4,
              },
              takes: [{
                path: copied.path,
                bigThumbnailPath: copied.bigThumbnailPath,
                smallThumbnailPath: copied.smallThumbnailPath,
                width: copied.width,
                height: copied.height,
                createdAt: Date.now(),
              }],
              activeTakeIndex: 0,
            })
          }
        }
      })()
    }
  }, [imagePaths, currentProjectId, isGenerating])
  
  const enqueueFromGenSpace = useCallback(
    async (payload: QueuePayload): Promise<boolean> => {
      const result = await enqueue(payload, {
        originatingProjectId: currentProjectId ?? undefined,
        source: 'genspace',
      })
      if (!result.ok) {
        setLocalError(createLocalGenerationError(result.error))
        return false
      }
      return true
    },
    [currentProjectId, enqueue],
  )

  const handleGenerate = async () => {
    if (mode === 'retake') {
      if (!retakeInput.videoPath || retakeInput.duration < 2) return
      retakeSubmissionRef.current = {
        prompt,
        input: {
          videoPath: retakeInput.videoPath,
          startTime: retakeInput.startTime,
          duration: retakeInput.duration,
          videoDuration: retakeInput.videoDuration,
        },
      }
      await submitRetake({
        videoPath: retakeInput.videoPath,
        startTime: retakeInput.startTime,
        duration: retakeInput.duration,
        prompt,
        mode: 'replace_audio_and_video',
      })
      return
    }

    if (!prompt.trim()) return

    // Save the prompt before generation starts
    setLastPrompt(prompt)

    if (mode === 'image') {
      // FLUX.2 [klein] 9B editing model: enqueue through the durable queue
      // (kind 'image_edit') with the optional reference image, so the edit
      // shows in the queue panel and doesn't grey out / block the UI while
      // it runs. The queue runner cooperates with the single-flight GPU slot
      // and routes the result into the project via the queue asset router,
      // just like z-image and video generations. Other image models use the
      // 'image' payload kind below.
      if (selectedImageSpec?.is_edit_model) {
        const dims = getImageDimensions({
          model: 'fast' as 'fast' | 'pro',
          duration: 5,
          videoResolution: settings.videoResolution,
          fps: 24,
          audio: false,
          cameraMotion: 'none',
          imageResolution: settings.imageResolution,
          imageAspectRatio: settings.aspectRatio,
          imageSteps: 4,
          variations: settings.variations,
        })
        const numImages = settings.variations || 1
        const editRequest = {
          prompt,
          width: dims.width,
          height: dims.height,
          numSteps: 4,
          numImages,
          referenceImages: inputImage ? [inputImage] : [],
        } as unknown as ApiRequestBodyOf<'generateImageEdit'>
        await enqueueFromGenSpace(
          { kind: 'image_edit', request: editRequest } as QueuePayload,
        )
        return
      }

      const dims = getImageDimensions({
        model: 'fast' as 'fast' | 'pro',
        duration: 5,
        videoResolution: settings.videoResolution,
        fps: 24,
        audio: false,
        cameraMotion: 'none',
        imageResolution: settings.imageResolution,
        imageAspectRatio: settings.aspectRatio,
        imageSteps: 4,
        variations: settings.variations,
      })
      const numImages = settings.variations || 1
      const imageRequest = {
        prompt,
        model: imageModelId,
        width: dims.width,
        height: dims.height,
        numSteps: 4,
        numImages,
      } as unknown as ApiRequestBodyOf<'generateImage'>
      await enqueueFromGenSpace(
        { kind: 'image', request: imageRequest } as QueuePayload,
      )
    } else {
      // Generate video (t2v if no image/audio, i2v if image, a2v if audio).
      // Enqueue so the user can line up multiple generations; the queue
      // panel shows progress and the asset router copies results into the
      // project as they complete.
      const imagePath = inputImage || null
      const audioPath = inputAudio || null
      const videoSettings = sanitizeVideoSettings(settings)
      const body: Record<string, unknown> = {
        prompt,
        model: videoSettings.model,
        duration: videoSettings.duration,
        resolution: videoSettings.videoResolution,
        fps: videoSettings.fps,
        audio: videoSettings.audio || false,
        cameraMotion: 'none',
        negativePrompt: (settings as { negativePrompt?: string }).negativePrompt ?? '',
        aspectRatio: videoSettings.aspectRatio || '16:9',
      }
      if (imagePath) body.imagePath = imagePath
      if (audioPath) body.audioPath = audioPath
      const videoRequest = body as unknown as ApiRequestBodyOf<'generateVideo'>

      // When a LoRA is selected, dispatch a `kind: 'lora'` payload instead of
      // a plain video request. The backend LoraInferenceHandler routes by
      // variant: standard stacks the adapter on a fast-video gen; union_control
      // runs the IC-LoRA control pipeline against the reference video; and
      // video_input_ic_lora feeds the raw reference video straight into a
      // user-trained IC-LoRA. LoRAs are local-only, so this branch is never
      // reached when API generations are forced (the selection is cleared).
      if (selectedLora && !forceApiGenerations) {
        if (selectedLora.variant === 'standard') {
          const payload = {
            kind: 'lora' as const,
            request: {
              variant: 'standard' as const,
              loraId: selectedLora.id,
              loraScale: loraScale,
              request: videoRequest,
            },
          } as unknown as QueuePayload
          await enqueueFromGenSpace(payload)
          return
        }
        if (!loraRefVideo) {
          setLocalError(
            createLocalGenerationError(
              'Attach a reference video before generating with this LoRA.',
            ),
          )
          return
        }
        if (selectedLora.variant === 'union_control') {
          const payload = {
            kind: 'lora' as const,
            request: {
              variant: 'union_control' as const,
              loraId: selectedLora.id,
              request: {
                video_path: loraRefVideo,
                conditioning_type: loraCondType,
                prompt,
                conditioning_strength: loraCondStrength,
                num_inference_steps: 30,
                cfg_guidance_scale: 1,
                negative_prompt: '',
                images: [],
                duration: loraDuration,
                preserve_audio: loraPreserveAudio,
                refine: loraRefine,
                resolution: loraResolution,
              },
            },
          } as unknown as QueuePayload
          await enqueueFromGenSpace(payload)
          return
        }
        // video_input_ic_lora
        const payload = {
          kind: 'lora' as const,
          request: {
            variant: 'video_input_ic_lora' as const,
            loraId: selectedLora.id,
            loraScale: loraScale,
            conditioningStrength: loraCondStrength,
            prompt,
            videoPath: loraRefVideo,
            negativePrompt: '',
            duration: loraDuration,
            preserveAudio: loraPreserveAudio,
            refine: loraRefine,
            resolution: loraResolution,
          },
        } as unknown as QueuePayload
        await enqueueFromGenSpace(payload)
        return
      }

      await enqueueFromGenSpace(
        { kind: 'video', request: videoRequest } as QueuePayload,
      )
    }
  }
  
  const handleDelete = (assetId: string) => {
    if (currentProjectId) {
      deleteAsset(currentProjectId, assetId)
    }
  }
  
  const handleDragStart = (e: React.DragEvent, asset: Asset) => {
    e.dataTransfer.setData('asset', JSON.stringify(asset))
    e.dataTransfer.setData('assetId', asset.id)
    e.dataTransfer.effectAllowed = 'copy'
  }
  
  const handleCreateVideo = (imageAsset: Asset) => {
    setMode('video')
    setInputImage(imageAsset.path)
    setPrompt(`${imageAsset.prompt || 'The scene comes to life...'}`)
  }

  const handleEditImage = (imageAsset: Asset) => {
    startGalleryImageEdit({
      assetPath: imageAsset.path,
      imageModelSpecs,
      setMode,
      setImageModelId,
      setInputImage,
      setPrompt,
      promptComposer: promptComposerRef.current,
    })
  }

  const handleRetake = (videoAsset: Asset) => {
    setMode('retake')
    setPrompt('')
    setActiveRetakeSource(null)
    setRetakeInitial({
      videoPath: videoAsset.path,
      duration: videoAsset.duration,
    })
    setRetakePanelKey((prev) => prev + 1)
  }

  // Asset-card "Apply LoRA": drop the clip into the reference-video slot and
  // surface the LoRA picker so the user can pick an adapter (union control or
  // video-input IC-LoRA). Standard (style) LoRAs don't need a reference video,
  // so the picker still lets them pick one — it just won't use the slot.
  const handleApplyLora = (videoAsset: Asset) => {
    if (forceApiGenerations) return
    setMode('video')
    setPrompt('')
    setLoraRefVideo(videoAsset.path)
    setActiveLoraSource(null)
    setLoraPickerOpen(true)
  }

  const handleSendToLora = (videoAsset: Asset) => {
    setSendToLoraAsset({ path: videoAsset.path, prompt: videoAsset.prompt })
  }

  const isRetakeMode = mode === 'retake'
  const loraNeedsRefVideo = selectedLora !== null
    && (selectedLora.variant === 'union_control' || selectedLora.variant === 'video_input_ic_lora')
  const hasCompatibleVideoSettings = mode !== 'video' || (
    !isLoadingVideoGenerationModelSpecs
    && videoModelSpecs.length > 0
    && resolveVideoGenerationOptions({
      settings,
      modelSpecs: videoModelSpecs,
      hasAudio: Boolean(inputAudio),
    }).hasCompatibleOptions
  )
  // A ref-video LoRA needs a reference clip before it can generate; a standard
  // LoRA stacks on a normal video gen so it only needs compatible video settings.
  // A union_control LoRA additionally needs its preprocessing models on disk
  // (depth / pose processors) — the download gate blocks generation until then.
  const loraReady = !loraNeedsRefVideo || !!loraRefVideo
  const unionModelsReady = selectedLora?.variant !== 'union_control' || icLoraModelGate.ready
  const canSubmit = isRetakeMode
    ? retakeInput.ready && !!retakeInput.videoPath && !isRetaking
    : mode === 'image'
      ? !!prompt.trim() && canGenerateImage
      : !!prompt.trim()
        && (selectedLora ? (loraReady && unionModelsReady) : hasCompatibleVideoSettings)
  const promptButtonLabel = isRetakeMode ? 'Retake' : 'Generate'
  const promptButtonIcon = isRetakeMode
    ? <Scissors className="h-3.5 w-3.5" />
    : <Sparkles className={`h-3.5 w-3.5 ${isGenerating ? 'animate-pulse' : ''}`} />
  const promptGenerating = isRetakeMode ? isRetaking : isGenerating

  // Auto-prompt assistant: available only for IC-LoRA variants that carry a
  // per-LoRA system prompt, when a reference video is attached and a Gemini
  // API key is configured. The one-shot handler fills the prompt box so the
  // user can review/edit before generating.
  const autoPromptAvailable =
    !!selectedLora
    && selectedLora.promptTemplate != null
    && !!loraRefVideo
    && appSettings.hasGeminiApiKey
    && !autoPromptLoading
  const handleAutoPrompt = useCallback(async () => {
    if (!selectedLora || !loraRefVideo) return
    setAutoPromptLoading(true)
    const result = await ApiClient.autoPrompt({ loraId: selectedLora.id, videoPath: loraRefVideo })
    setAutoPromptLoading(false)
    if (!result.ok) {
      const message = (result.error as { message?: string })?.message ?? 'Auto-prompt failed'
      logger.warn(`Auto-prompt failed (${message})`)
      return
    }
    const text = result.data.prompt.trim()
    if (text) setPrompt(text)
  }, [selectedLora, loraRefVideo])

  // Close size menu on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (sizeMenuRef.current && !sizeMenuRef.current.contains(e.target as Node)) {
        setShowSizeMenu(false)
      }
    }
    if (showSizeMenu) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showSizeMenu])

  const filteredAssets = showFavorites ? assets.filter(a => a.favorite) : assets
  const favoriteCount = assets.filter(a => a.favorite).length
  const isLibraryMode = mode === 'video' || mode === 'image'

  // Navigation for the asset preview modal
  const selectedIndex = selectedAsset ? filteredAssets.findIndex(a => a.id === selectedAsset.id) : -1
  const canGoPrev = selectedIndex > 0
  const canGoNext = selectedIndex >= 0 && selectedIndex < filteredAssets.length - 1

  const goToPrev = useCallback(() => {
    if (canGoPrev) setSelectedAsset(filteredAssets[selectedIndex - 1])
  }, [canGoPrev, filteredAssets, selectedIndex])

  const goToNext = useCallback(() => {
    if (canGoNext) setSelectedAsset(filteredAssets[selectedIndex + 1])
  }, [canGoNext, filteredAssets, selectedIndex])

  // Keyboard navigation for the preview modal
  useEffect(() => {
    if (!selectedAsset) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); goToPrev() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); goToNext() }
      else if (e.key === 'Escape' && !document.fullscreenElement) setSelectedAsset(null)
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [selectedAsset, goToPrev, goToNext])

  return (
    <div className="h-full relative bg-zinc-950">

      {/* Empty state */}
      {isLibraryMode && assets.length === 0 && !isGenerating && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
          <div className="w-24 h-24 rounded-2xl border-2 border-dashed border-zinc-700 flex items-center justify-center mb-4">
            <Sparkles className="h-10 w-10 text-zinc-600" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Start Creating</h3>
          <p className="text-zinc-500 max-w-md">
            Use the prompt bar below to generate images and videos.
            Drag assets into the input box to use them as references.
          </p>
        </div>
      )}

      {/* No favorites empty state */}
      {isLibraryMode && showFavorites && filteredAssets.length === 0 && assets.length > 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
          <Heart className="h-12 w-12 text-zinc-700 mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">No favorites yet</h3>
          <p className="text-zinc-500 text-sm">
            Click the heart icon on any asset to add it to your favorites.
          </p>
        </div>
      )}

      {/* Assets area — full width, no background, above the prompt bar */}
      {isLibraryMode && (assets.length > 0 || isGenerating) && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] flex flex-col px-4 pt-4">
          {/* Top bar */}
          <div className="flex items-center justify-end pb-2 gap-2">
            <button
              onClick={() => setShowFavorites(!showFavorites)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                showFavorites
                  ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
              }`}
            >
              <Heart className={`h-4 w-4 ${showFavorites ? 'fill-current' : ''}`} />
              Favorites
              {favoriteCount > 0 && (
                <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                  showFavorites ? 'bg-red-500/30 text-red-300' : 'bg-zinc-800 text-zinc-500'
                }`}>
                  {favoriteCount}
                </span>
              )}
            </button>

            <div ref={sizeMenuRef} className="relative">
              <button
                onClick={() => setShowSizeMenu(!showSizeMenu)}
                className={`p-2 rounded-md transition-colors ${
                  showSizeMenu ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
                }`}
              >
                {gallerySize === 'small' ? <GridSmallIcon className="h-4 w-4" /> :
                 gallerySize === 'medium' ? <GridMediumIcon className="h-4 w-4" /> :
                 <GridLargeIcon className="h-4 w-4" />}
              </button>

              {showSizeMenu && (
                <div className="absolute top-full mt-2 right-0 bg-zinc-800 border border-zinc-700 rounded-md p-2 min-w-[160px] shadow-xl z-50">
                  {([
                    { value: 'small' as GallerySize, label: 'Small', icon: GridSmallIcon },
                    { value: 'medium' as GallerySize, label: 'Medium', icon: GridMediumIcon },
                    { value: 'large' as GallerySize, label: 'Large', icon: GridLargeIcon },
                  ]).map(option => (
                    <button
                      key={option.value}
                      onClick={() => { setGallerySize(option.value); setShowSizeMenu(false) }}
                      className={`w-full flex items-center justify-between px-2 py-2.5 rounded-md transition-colors text-left ${gallerySize === option.value ? 'bg-white/20 hover:bg-white/25' : 'hover:bg-zinc-700'}`}
                    >
                      <div className="flex items-center gap-3">
                        <option.icon className={`h-4 w-4 ${gallerySize === option.value ? 'text-white' : 'text-zinc-500'}`} />
                        <span className={`text-sm ${gallerySize === option.value ? 'text-white font-medium' : 'text-zinc-400'}`}>
                          {option.label}
                        </span>
                      </div>
                      {gallerySize === option.value && (
                        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Assets grid — fills remaining space, scrollable */}
          <div className="overflow-y-auto overflow-x-hidden [scrollbar-gutter:stable] flex-1">
            <div className={`grid ${gallerySizeClasses[gallerySize]} gap-4`}>
              {isGenerating && (
                <div className="relative rounded-xl overflow-hidden bg-zinc-800 aspect-video">
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className="relative w-16 h-16 mb-3">
                      <div className="absolute inset-0 rounded-full border-2 border-blue-500/30" />
                      <div className="absolute inset-0 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
                      <div className="absolute inset-2 rounded-full bg-zinc-800 flex items-center justify-center">
                        <Sparkles className="h-6 w-6 text-blue-400" />
                      </div>
                    </div>
                    <p className="text-sm text-zinc-400">{statusMessage || 'Generating...'}</p>
                    {progress > 0 && (
                      <div className="w-32 h-1 bg-zinc-800 rounded-full mt-2 overflow-hidden">
                        <div className="h-full bg-blue-500 transition-all" style={{ width: `${progress}%` }} />
                      </div>
                    )}
                  </div>
                </div>
              )}
              {filteredAssets.map(asset => (
                <AssetCard
                  key={asset.id}
                  asset={asset}
                  onDelete={() => handleDelete(asset.id)}
                  onPlay={() => setSelectedAsset(asset)}
                  onDragStart={handleDragStart}
                  onEditImage={handleEditImage}
                  onCreateVideo={handleCreateVideo}
                  onRetake={handleRetake}
                  onApplyLora={!forceApiGenerations ? handleApplyLora : undefined}
                  onSendToLora={handleSendToLora}
                  onToggleFavorite={() => currentProjectId && toggleFavorite(currentProjectId, asset.id)}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {mode === 'retake' && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] px-4 pt-4 pb-4 flex flex-col overflow-hidden">
          <RetakePanel
            initialVideoPath={retakeInitial.videoPath}
            initialDuration={retakeInitial.duration}
            resetKey={retakePanelKey}
            fillHeight
            isProcessing={isRetaking}
            processingStatus={retakeStatus}
            onChange={(data) => setRetakeInput(data)}
          />
        </div>
      )}

      {/* Floating prompt panel — wider, responsive, centered */}
      <div className="absolute bottom-5 left-1/2 w-[min(700px,calc(100%-2rem))] -translate-x-1/2">

        <FreeApiKeyBubble
          forceApiGenerations={forceApiGenerations}
          hasLtxApiKey={appSettings.hasLtxApiKey}
          isGenerating={isGenerating}
        />

        {/* IC-LoRA model download gate — shown while a union_control LoRA is
            selected but its preprocessing models (depth / pose processors) aren't
            fully on disk yet. Covers the pose path that needs the DW processor +
            YOLOX person detector in addition to the union adapter. */}
        {unionGateActive && (icLoraModelGate.checking || !icLoraModelGate.ready) && (
          <ModelDownloadGate
            title="Download Required: IC-LoRA Resources"
            description="Pose / depth conditioning needs extra preprocessing models. Download them to use this LoRA."
            gate={icLoraModelGate}
          />
        )}

        {/* Prompt bar */}
        <PromptBar
          mode={mode}
          onModeChange={setMode}
          prompt={prompt}
          onPromptChange={setPrompt}
          onGenerate={handleGenerate}
          isGenerating={promptGenerating}
          canGenerate={canSubmit}
          buttonLabel={promptButtonLabel}
          buttonIcon={promptButtonIcon}
          inputImage={inputImage}
          onInputImageChange={setInputImage}
          inputAudio={inputAudio}
          onInputAudioChange={setInputAudio}
          settings={settings}
          onSettingsChange={(nextSettings) => setSettings(sanitizeVideoSettings(nextSettings))}
          videoModelSpecs={videoModelSpecs}
          videoSettingsMessage={videoSettingsMessage}
          imageModelSpecs={imageModelSpecs}
          selectedImageModelId={imageModelId}
          onSelectImageModel={setImageModelId}
          onRefreshImageModelSpecs={refreshImageModelSpecs}
          imageSettingsMessage={imageSettingsMessage}
          imageEditModelSelected={selectedImageSpec?.is_edit_model ?? false}
          selectedLora={selectedLora}
          forceApiGenerations={forceApiGenerations}
          loraScale={loraScale}
          onLoraScaleChange={setLoraScale}
          loraCondType={loraCondType}
          onLoraCondTypeChange={setLoraCondType}
          loraCondStrength={loraCondStrength}
          onLoraCondStrengthChange={setLoraCondStrength}
          loraRefVideo={loraRefVideo}
          onLoraRefVideoChange={setLoraRefVideo}
          loraDuration={loraDuration}
          onLoraDurationChange={setLoraDuration}
          loraResolution={loraResolution}
          onLoraResolutionChange={setLoraResolution}
          loraRefine={loraRefine}
          onLoraRefineChange={setLoraRefine}
          preserveAudio={loraPreserveAudio}
          onPreserveAudioChange={setLoraPreserveAudio}
          onAutoPrompt={handleAutoPrompt}
          autoPromptLoading={autoPromptLoading}
          autoPromptAvailable={autoPromptAvailable}
          autoPromptHasKey={appSettings.hasGeminiApiKey}
          loraPickerOpen={loraPickerOpen}
          onToggleLoraPicker={() => setLoraPickerOpen((v) => !v)}
          onClearLora={clearLora}
          onSelectLora={selectLora}
          onDeletedLora={(deletedId) => {
            if (selectedLora?.id === deletedId) clearLora()
          }}
          loraEntries={loraRegistry.entries}
          loraLoading={loraRegistry.loading}
          loraError={loraRegistry.error}
          onRefreshLora={() => void loraRegistry.refresh()}
          promptTextareaRef={promptComposerRef}
        />
      </div>
      
      {sendToLoraAsset && (
        <SendToLoraModal
          videoPath={sendToLoraAsset.path}
          suggestedCaption={sendToLoraAsset.prompt}
          originatingProjectId={currentProjectId}
          onClose={() => setSendToLoraAsset(null)}
        />
      )}

      {/* Asset preview modal */}
      {selectedAsset && (
        <div 
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
          onClick={() => setSelectedAsset(null)}
        >
          {/* Previous button */}
          <button
            onClick={(e) => { e.stopPropagation(); goToPrev() }}
            disabled={!canGoPrev}
            className={`absolute left-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full backdrop-blur-md transition-all ${
              canGoPrev
                ? 'bg-white/10 text-white hover:bg-white/20 cursor-pointer'
                : 'bg-white/5 text-zinc-600 cursor-default'
            }`}
          >
            <ChevronLeft className="h-6 w-6" />
          </button>

          {/* Next button */}
          <button
            onClick={(e) => { e.stopPropagation(); goToNext() }}
            disabled={!canGoNext}
            className={`absolute right-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full backdrop-blur-md transition-all ${
              canGoNext
                ? 'bg-white/10 text-white hover:bg-white/20 cursor-pointer'
                : 'bg-white/5 text-zinc-600 cursor-default'
            }`}
          >
            <ChevronRight className="h-6 w-6" />
          </button>

          {/* Content area */}
          <div className="relative max-w-5xl w-full max-h-full px-20 py-8" onClick={e => e.stopPropagation()}>
            {/* Top bar: counter + close */}
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm text-zinc-500 font-medium">
                {selectedIndex + 1} / {filteredAssets.length}
              </span>
              <button
                onClick={() => setSelectedAsset(null)}
                className="p-2 rounded-md text-zinc-400 hover:text-white transition-colors"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            {hasIcLoraViews && (
              <div className="flex items-center justify-center gap-1 mb-3">
                {([
                  ['result', 'Result'],
                  referenceVideoPath ? ['before-after', 'Before / After'] : null,
                  controlVideoPath ? ['control', 'Control'] : null,
                ].filter(Boolean) as ['result' | 'before-after' | 'control', string][]).map(([mode, label]) => (
                  <button
                    key={mode}
                    onClick={() => setResultViewMode(mode)}
                    className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                      resultViewMode === mode
                        ? 'bg-white text-black'
                        : 'bg-white/10 text-zinc-300 hover:bg-white/20'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}

            {resultViewMode === 'before-after' && referenceVideoPath ? (
              <div
                ref={comparisonViewerRef}
                className={`flex flex-col ${baFullscreen ? 'h-screen w-screen bg-black p-4' : ''}`}
              >
                <div className={`grid ${showControlInBa ? 'grid-cols-3' : 'grid-cols-2'} gap-3 ${baFullscreen ? 'flex-1 min-h-0' : 'max-h-[68vh]'}`}>
                  <div className="relative flex min-h-0 flex-col items-center">
                    <video
                      ref={baLeftRef}
                      src={pathToFileUrl(referenceVideoPath)}
                      onClick={toggleBaPlay}
                      className={`w-full min-h-0 rounded-xl object-contain bg-black cursor-pointer ${baFullscreen ? 'flex-1 h-full max-h-full' : 'max-h-[64vh]'}`}
                    />
                    <span className="text-xs text-zinc-500 mt-1.5">{showControlInBa ? 'Driving video' : 'Reference'}</span>
                    {!baPlaying && (
                      <button
                        onClick={toggleBaPlay}
                        className="absolute inset-0 flex items-center justify-center"
                        aria-label="Play both"
                      >
                        <span className="p-3 rounded-full bg-black/60 text-white">
                          <Play className="h-6 w-6" />
                        </span>
                      </button>
                    )}
                  </div>
                  {showControlInBa && controlVideoPath && (
                    <div className="relative flex min-h-0 flex-col items-center">
                      <video
                        ref={baControlRef}
                        src={pathToFileUrl(controlVideoPath)}
                        onClick={toggleBaPlay}
                        className={`w-full min-h-0 rounded-xl object-contain bg-black cursor-pointer ${baFullscreen ? 'flex-1 h-full max-h-full' : 'max-h-[64vh]'}`}
                      />
                      <span className="text-xs text-zinc-500 mt-1.5">
                        Control — {selectedAsset.generationParams?.icLoraConditioningType ?? 'conditioning'}
                      </span>
                    </div>
                  )}
                  <div className="relative flex min-h-0 flex-col items-center">
                    <video
                      ref={baRightRef}
                      key={selectedAsset.id}
                      src={pathToFileUrl(selectedAsset.path)}
                      onClick={toggleBaPlay}
                      onPlay={() => setBaPlaying(true)}
                      onPause={() => setBaPlaying(false)}
                      onTimeUpdate={(e) => setBaTime(e.currentTarget.currentTime)}
                      onLoadedMetadata={(e) => setBaDuration(e.currentTarget.duration)}
                      className={`w-full min-h-0 rounded-xl object-contain bg-black cursor-pointer ${baFullscreen ? 'flex-1 h-full max-h-full' : 'max-h-[64vh]'}`}
                    />
                    <span className="text-xs text-zinc-500 mt-1.5">Output</span>
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-3 px-1">
                  <button
                    onClick={toggleBaPlay}
                    className="p-2 rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors"
                    aria-label={baPlaying ? 'Pause' : 'Play'}
                  >
                    {baPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  </button>
                  <input
                    type="range"
                    min={0}
                    max={baDuration || 0}
                    step={0.05}
                    value={Math.min(baTime, baDuration || 0)}
                    onChange={(e) => seekBa(parseFloat(e.target.value))}
                    className="flex-1 accent-white"
                  />
                  <span className="text-xs text-zinc-400 tabular-nums whitespace-nowrap">
                    {formatTime(Math.min(baTime, baDuration || 0))} / {formatTime(baDuration || 0)}
                  </span>
                  <button
                    onClick={handleDownloadSideBySide}
                    disabled={baExporting}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white/10 text-zinc-200 hover:bg-white/20 text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Export a single side-by-side MP4 (reference | output)"
                  >
                    {baExporting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                    {baExporting ? 'Exporting…' : 'Export side-by-side'}
                  </button>
                  <button
                    onClick={() => void toggleBaFullscreen()}
                    className="inline-flex items-center gap-1.5 rounded-md bg-white/10 px-3 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:bg-white/20"
                    aria-label={baFullscreen ? 'Exit comparison fullscreen' : 'Enter comparison fullscreen'}
                    title={baFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                  >
                    {baFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
                    <span className="hidden sm:inline">{baFullscreen ? 'Exit fullscreen' : 'Fullscreen'}</span>
                  </button>
                </div>
                {baFullscreenError && (
                  <p className="mt-2 text-center text-xs text-amber-300" role="status">
                    {baFullscreenError}
                  </p>
                )}
              </div>
            ) : resultViewMode === 'control' && controlVideoPath ? (
              <div className="flex flex-col items-center">
                <video
                  src={pathToFileUrl(controlVideoPath)}
                  controls
                  autoPlay
                  loop
                  className="w-full rounded-xl object-contain max-h-[75vh] bg-black"
                />
                <span className="text-xs text-zinc-500 mt-1.5">
                  Control signal — {selectedAsset.generationParams?.icLoraConditioningType ?? 'conditioning'}
                </span>
              </div>
            ) : selectedAsset.type === 'video' ? (
              <video
                key={selectedAsset.id}
                src={pathToFileUrl(selectedAsset.path)}
                controls
                autoPlay
                className="w-full rounded-xl object-contain max-h-[75vh]"
              />
            ) : (
              <img
                key={selectedAsset.id}
                src={pathToFileUrl(selectedAsset.path)}
                alt=""
                className="w-full rounded-xl object-contain max-h-[75vh]"
              />
            )}
            <div className="mt-4 text-center">
              <div className="inline-flex items-start gap-2 max-w-full">
                <p className="text-zinc-300">{selectedAsset.prompt}</p>
                {selectedAsset.prompt && (
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(selectedAsset.prompt)
                      setCopiedPrompt(true)
                      setTimeout(() => setCopiedPrompt(false), 2000)
                    }}
                    className="shrink-0 p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors"
                    title="Copy prompt"
                  >
                    {copiedPrompt ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                  </button>
                )}
              </div>
              <p className="text-zinc-500 text-sm mt-1">
                {selectedAsset.resolution} • {selectedAsset.duration ? `${selectedAsset.duration}s` : 'Image'}
              </p>
            </div>
          </div>
        </div>
      )}

      {(error || localError) && (
        <GenerationErrorDialog
          error={(error || localError)!}
          onDismiss={() => {
            if (error) reset()
            if (localError) {
              setLocalError(null)
              resetRetake()
            }
          }}
        />
      )}
    </div>
  )
}
