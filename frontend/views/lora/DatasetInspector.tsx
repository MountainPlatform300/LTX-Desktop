import { useEffect, useState } from 'react'
import { AlertTriangle, Archive, ChevronDown, ChevronRight, Crop, Image as ImageIcon, Maximize2, Recycle, SlidersHorizontal, Sparkles, Trash2, Upload, Wand2, XCircle } from 'lucide-react'
import { Inspector } from '../studio/Inspector'
import { DatasetHealthMeter, PreflightChecklist } from '../../components/lora/DatasetHealth'
import { TrainingLivePeek } from '../../components/lora/TrainingLivePeek'
import { datasetHealth } from '../../lib/lora-quality'
import { countReadyPairs, derivePairs } from '../../lib/lora-pairs'
import type { StudioClip } from '../studio/studio-store'
import { STAGE_DOT, STAGE_TEXT, type Lifecycle } from './lifecycle'

/** Compact ETA: "45s", "3m", "1h 2m". */
function formatEta(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))}s`
  const mins = Math.round(seconds / 60)
  if (mins < 60) return `${mins}m`
  return `${Math.floor(mins / 60)}h ${mins % 60}m`
}

/** Compact elapsed: "12s", "3m 4s", "1h 2m". */
function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`
  const mins = Math.floor(seconds / 60)
  if (mins < 60) return `${mins}m ${Math.round(seconds % 60)}s`
  return `${Math.floor(mins / 60)}h ${mins % 60}m`
}

/**
 * Ticks once a second while `active` so we can show a live "Xm elapsed" timer
 * alongside the backend's phase/%/ETA. Reinforces "it's running" during long,
 * indeterminate prep phases (model load) where % is momentarily absent.
 */
function useElapsedSeconds(active: boolean, startedAt: string | null | undefined): number {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (!active || !startedAt) {
      setElapsed(0)
      return
    }
    const start = new Date(startedAt).getTime()
    if (Number.isNaN(start)) return
    const tick = () => setElapsed(Math.max(0, (Date.now() - start) / 1000))
    tick()
    const id = window.setInterval(tick, 1000)
    return () => window.clearInterval(id)
  }, [active, startedAt])
  return elapsed
}

function EditableField({
  label,
  value,
  placeholder,
  onCommit,
}: {
  label: string
  value: string
  placeholder: string
  onCommit: (next: string) => void
}) {
  const [draft, setDraft] = useState(value)
  // Keep the local draft in sync when the dataset changes underneath us.
  useEffect(() => setDraft(value), [value])

  const commit = () => {
    const trimmed = draft.trim()
    if (trimmed !== value) onCommit(trimmed)
  }

  return (
    <label className="block">
      <span className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</span>
      <input
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur()
        }}
        className="mt-0.5 w-full px-2 py-1.5 rounded-md bg-zinc-900 border border-zinc-800 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600"
      />
    </label>
  )
}

function SidebarAction({
  icon: Icon,
  label,
  onClick,
  disabled,
  title,
}: {
  icon: typeof Crop
  label: string
  onClick: () => void
  disabled?: boolean
  title?: string
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="w-full text-xs px-3 py-2 rounded-md border border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
    >
      <Icon className="h-3.5 w-3.5" /> {label}
    </button>
  )
}

export function DatasetInspector({
  datasetId,
  datasetName,
  datasetType,
  triggerWord,
  life,
  clips,
  selectedIds,
  editable,
  errorText,
  onRename,
  onSetTrigger,
  onSetType,
  onDelete,
  onArchive,
  onExport,
  trashCount,
  onOpenTrash,
  singleClip,
  canMakePair,
  onCaptionChange,
  onAutoCaption,
  onOpenClip,
  onEditClip,
  onFrameEditClip,
  onMakePairClip,
  onVariantClip,
  onCancelPreprocess,
  onCancelUpload,
  onViewSettings,
}: {
  datasetId: string | null
  datasetName: string
  datasetType: 'standard' | 'ic_lora'
  triggerWord: string | null
  life: Lifecycle
  clips: StudioClip[]
  selectedIds: Set<string>
  editable: boolean
  errorText: string | null
  onRename: (next: string) => void
  onSetTrigger: (next: string) => void
  onSetType: (type: 'standard' | 'ic_lora') => void
  onDelete: () => void
  onArchive: () => void
  onExport: () => void
  trashCount: number
  onOpenTrash: () => void
  singleClip: StudioClip | null
  canMakePair: boolean
  onCaptionChange: (clipId: string, caption: string) => void
  onAutoCaption: (id: string) => void
  onOpenClip: (id: string) => void
  onEditClip: (id: string) => void
  onFrameEditClip: (id: string) => void
  onMakePairClip: (id: string) => void
  onVariantClip: (id: string) => void
  /** Cancel an in-progress preprocessing run (captioning/preparing stages). */
  onCancelPreprocess: (preprocessedId: string) => void
  /** Cancel an in-progress upload (releases the provisioned GPU pod). */
  onCancelUpload: () => void
  /** Open the read-only training/prep settings modal (any stage). */
  onViewSettings: () => void
}) {
  const [showPreflight, setShowPreflight] = useState(false)
  const isIcLora = datasetType === 'ic_lora'
  const health = datasetHealth(clips.map((c) => ({ caption: c.caption, probe: c.probe })))
  const pairGroups = derivePairs(clips).pairs
  const readyPairs = countReadyPairs(pairGroups)
  const nothingSelected = selectedIds.size === 0
  // Cancel is offered while preprocessing (captioning/preparing/queued) and not
  // yet already cancelling.
  const prep = life.preprocessed
  const canCancelPrep =
    life.stage === 'preparing' && prep != null && !prep.cancelRequested
  // Cancel is also offered during the upload stage (acquiring pod / uploading
  // clips): the backend releases the provisioned GPU pod and stops before
  // preprocessing. Hidden once the cancel has been requested (shows
  // "Cancelling…" instead) and on local runs (no pod to reclaim, the backend
  // finalizes instantly so the flag never lingers).
  const canCancelUpload =
    life.stage === 'uploading' && !life.cancelRequested
  // Live "Xm elapsed" while busy — uses the preprocess start during prep and the
  // training job start during a run. Reinforces progress even when the backend's
  // % is momentarily unavailable (e.g. model-load phase).
  const elapsedStartedAt =
    life.stage === 'preparing'
      ? prep?.startedAt
      : life.training?.startedAt
  const elapsed = useElapsedSeconds(life.busy, elapsedStartedAt)

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-zinc-800 space-y-3">
        {editable ? (
          <>
            <EditableField label="Name" value={datasetName} placeholder="Dataset name" onCommit={onRename} />
            <EditableField
              label="Trigger word"
              value={triggerWord ?? ''}
              placeholder="optional, e.g. mychar"
              onCommit={onSetTrigger}
            />
          </>
        ) : (
          <div>
            <p className="text-sm font-medium text-white truncate">{datasetName}</p>
            {triggerWord && <p className="text-[11px] text-zinc-500">trigger &ldquo;{triggerWord}&rdquo;</p>}
          </div>
        )}

        <div className="space-y-1">
          <span className="text-[10px] uppercase tracking-wide text-zinc-500">Type</span>
          {editable ? (
            <div className="grid grid-cols-2 gap-1">
              {([
                { id: 'standard', label: 'Standard', title: 'Standard LoRA: learns a look/subject from individual clips' },
                { id: 'ic_lora', label: 'IC-LoRA', title: 'In-Context LoRA: trains input → output transformations' },
              ] as const).map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => onSetType(opt.id)}
                  title={opt.title}
                  className={`text-[11px] px-2 py-1 rounded border transition-colors ${
                    datasetType === opt.id
                      ? opt.id === 'ic_lora'
                        ? 'border-blue-500/50 text-blue-200 bg-blue-500/15'
                        : 'border-blue-500/50 text-blue-200 bg-blue-500/15'
                      : 'border-zinc-700 text-zinc-400 hover:border-zinc-600'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          ) : (
            <div>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded border ${
                  isIcLora
                    ? 'border-blue-500/40 text-blue-300 bg-blue-500/10'
                    : 'border-zinc-700 text-zinc-400 bg-zinc-800/60'
                }`}
                title={isIcLora ? 'In-Context LoRA: trains input → output transformations' : 'Standard LoRA: learns a look/subject from individual clips'}
              >
                {isIcLora ? 'IC-LoRA' : 'Standard LoRA'}
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between">
          <span className={`text-xs flex items-center gap-1.5 ${STAGE_TEXT[life.tone]}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${STAGE_DOT[life.tone]}`} />
            {life.label}
          </span>
          <div className="flex items-center gap-1">
            {(prep != null || life.training != null) && (
              <button
                onClick={onViewSettings}
                title="View training & preprocessing settings"
                className="h-7 w-7 flex items-center justify-center rounded-md text-zinc-500 hover:text-blue-300 hover:bg-zinc-800"
              >
                <SlidersHorizontal className="h-3.5 w-3.5" />
              </button>
            )}
            <button
              onClick={onExport}
              title="Export dataset for training or sharing"
              className="h-7 w-7 flex items-center justify-center rounded-md text-zinc-500 hover:text-blue-300 hover:bg-zinc-800"
            >
              <Upload className="h-3.5 w-3.5" />
            </button>
            {trashCount > 0 && (
              <button
                onClick={onOpenTrash}
                title={`Recycle bin — ${trashCount} clip${trashCount === 1 ? '' : 's'}`}
                className="h-7 pl-1.5 pr-2 flex items-center gap-1 rounded-md text-zinc-500 hover:text-amber-300 hover:bg-zinc-800 text-[11px] font-medium"
              >
                <Recycle className="h-3.5 w-3.5" />
                {trashCount}
              </button>
            )}
            <button
              onClick={onArchive}
              title="Archive dataset"
              className="h-7 w-7 flex items-center justify-center rounded-md text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800"
            >
              <Archive className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={onDelete}
              title="Delete dataset"
              className="h-7 w-7 flex items-center justify-center rounded-md text-zinc-500 hover:text-red-400 hover:bg-zinc-800"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {(life.detail || (life.busy && elapsed > 0)) && (
          <div className="space-y-1">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[11px] text-zinc-400 leading-snug truncate">
                {life.detail ?? '\u00A0'}
              </span>
              <span className="text-[11px] font-medium tabular-nums text-zinc-300 flex-shrink-0">
                {typeof life.percent === 'number'
                  ? `${life.percent}%${life.etaSeconds != null ? ` · ${formatEta(life.etaSeconds)} left` : ''}`
                  : life.busy && elapsed > 0
                    ? `${formatElapsed(elapsed)} elapsed`
                    : ''}
              </span>
            </div>
            {typeof life.percent === 'number' ? (
              <div className="h-1 w-full overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${Math.min(100, Math.max(0, life.percent))}%` }}
                />
              </div>
            ) : (
              // Indeterminate activity bar for phases without a measurable %
              // (model load, cloning, installing, uploading). Skipped during
              // training, which has its own detailed step progress in RunView.
              life.busy &&
              life.stage !== 'training' && (
                <div className="h-1 w-full overflow-hidden rounded-full bg-zinc-800 relative">
                  <div className="absolute inset-y-0 w-1/3 rounded-full bg-blue-500/70 animate-[lora-shimmer_1.1s_ease-in-out_infinite]" />
                </div>
              )
            )}
          </div>
        )}

        <TrainingLivePeek datasetId={datasetId} />

        {canCancelPrep && prep && (
          <button
            onClick={() => onCancelPreprocess(prep.id)}
            className="w-full text-xs px-3 py-2 rounded-md border border-zinc-700 text-zinc-200 hover:border-red-500/50 hover:bg-red-500/10 hover:text-red-300 flex items-center justify-center gap-2"
          >
            <XCircle className="h-3.5 w-3.5" /> Cancel preprocessing
          </button>
        )}
        {prep?.cancelRequested && life.stage === 'preparing' && (
          <p className="text-[11px] text-zinc-500 flex items-center gap-1.5">
            <XCircle className="h-3.5 w-3.5 animate-pulse" /> Cancelling…
          </p>
        )}

        {canCancelUpload && (
          <button
            onClick={onCancelUpload}
            className="w-full text-xs px-3 py-2 rounded-md border border-zinc-700 text-zinc-200 hover:border-red-500/50 hover:bg-red-500/10 hover:text-red-300 flex items-center justify-center gap-2"
          >
            <XCircle className="h-3.5 w-3.5" /> Cancel upload
          </button>
        )}
        {life.cancelRequested && life.stage === 'uploading' && (
          <p className="text-[11px] text-zinc-500 flex items-center gap-1.5">
            <XCircle className="h-3.5 w-3.5 animate-pulse" /> Cancelling…
          </p>
        )}

        {errorText && (
          /unavailable|out of stock|in-stock/i.test(errorText) ? (
            // GPU stock errors are recoverable user actions, not failures — show
            // an actionable amber callout pointing at the GPU picker, not a
            // red stack-trace-style message.
            <div className="flex items-start gap-1.5 rounded-md bg-amber-500/10 border border-amber-500/20 px-2.5 py-2">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0 mt-0.5" />
              <p className="text-[11px] text-amber-300 whitespace-pre-wrap break-words leading-snug">
                {errorText}
              </p>
            </div>
          ) : (
            <p className="text-[11px] text-red-400 whitespace-pre-wrap break-words font-mono">{errorText}</p>
          )
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        <Inspector clips={clips} selectedIds={selectedIds} />

        {singleClip && (
          <div className="px-4 pb-4 space-y-3">
            <EditableField
              label="Caption"
              value={singleClip.caption}
              placeholder="Describe this clip…"
              onCommit={(next) => onCaptionChange(singleClip.id, next)}
            />
            {editable && (
              <SidebarAction
                icon={Sparkles}
                label="Auto-caption (AI)"
                onClick={() => onAutoCaption(singleClip.id)}
                title="Generate a caption for this clip with AI"
              />
            )}
            <button
              onClick={() => onOpenClip(singleClip.id)}
              className="w-full text-xs px-3 py-2 rounded-md border border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800 flex items-center gap-2"
            >
              <Maximize2 className="h-3.5 w-3.5" /> Open
            </button>
            {editable && (
              <div className="space-y-2">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">Actions</span>
                {singleClip.kind === 'video' ? (
                  <>
                    <SidebarAction icon={Crop} label="Trim & crop" onClick={() => onEditClip(singleClip.id)} />
                    <SidebarAction icon={ImageIcon} label="Frame edit (AI)" onClick={() => onFrameEditClip(singleClip.id)} />
                    {isIcLora && <SidebarAction icon={Sparkles} label="Generate example (AI)" onClick={() => onMakePairClip(singleClip.id)} />}
                    <SidebarAction icon={Wand2} label="Variant" onClick={() => onVariantClip(singleClip.id)} />
                  </>
                ) : (
                  <>
                    <SidebarAction icon={Wand2} label="Animate (i2v)" onClick={() => onVariantClip(singleClip.id)} />
                    {isIcLora && (
                      <SidebarAction
                        icon={Sparkles}
                        label="Generate example (motion-lock)"
                        onClick={() => onMakePairClip(singleClip.id)}
                        disabled={!canMakePair}
                        title={canMakePair ? undefined : 'Add a video clip to use as the motion driver'}
                      />
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {nothingSelected && clips.length > 0 && (
          <div className="px-4 pb-4 space-y-2">
            <DatasetHealthMeter health={health} />
            {pairGroups.length > 0 && (
              <div className="flex items-center justify-between text-[11px] px-1">
                <span className="text-zinc-500" title="Training examples: input(s) → output">Examples</span>
                <span className={readyPairs === pairGroups.length ? 'text-emerald-400' : 'text-amber-400'}>
                  {readyPairs}/{pairGroups.length} ready
                </span>
              </div>
            )}
            <button
              onClick={() => setShowPreflight((v) => !v)}
              className="text-[11px] text-zinc-400 hover:text-zinc-200 flex items-center gap-1"
            >
              {showPreflight ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              Pre-flight checklist
            </button>
            {showPreflight && (
              <div className="bg-zinc-800/40 rounded-lg px-3 py-2.5">
                <PreflightChecklist clips={clips.map((c) => ({ caption: c.caption, probe: c.probe }))} triggerWord={triggerWord} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
