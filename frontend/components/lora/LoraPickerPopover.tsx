import { useEffect, useId, useRef, useState, useCallback } from 'react'
import { Sparkles, Check, Box, Layers, AlertCircle, RefreshCw, Download, Loader2, ArrowLeft, Pencil, RotateCcw, Trash2, FolderOpen, ChevronDown, ChevronRight, ArrowUpDown } from 'lucide-react'
import { ApiClient } from '../../lib/api-client'
import { logger } from '../../lib/logger'
import { Tooltip } from '../ui/tooltip'
import { confirmAction } from '../ui/confirm-dialog'
import type {
  LoraInferenceEntry,
  LoraInferenceConditioningType,
} from '../../hooks/use-lora-inference-registry'

// Variants a user can tag on an imported LoRA. Mirrors the backend
// `ImportedLoraVariantApi` — the official union_control checkpoint is never
// importable (it's fetched through the model-download flow), so only the two
// variants that route through a user-supplied weights file are offered.
const IMPORT_VARIANTS: { value: 'standard' | 'video_input_ic_lora'; label: string; hint: string }[] = [
  { value: 'standard', label: 'Style LoRA', hint: 'Text/image-to-video adapter' },
  { value: 'video_input_ic_lora', label: 'Reference IC-LoRA', hint: 'Conditioned on a reference video' },
]

// Compact chip text for the picker rows (kept short so each row stays a single
// line). The full label + conditioning type are surfaced in the prompt bar
// once a LoRA is selected, so the picker only needs to distinguish variants.
const VARIANT_CHIP: Record<LoraInferenceEntry['variant'], string> = {
  standard: 'Style',
  union_control: 'IC · Control',
  video_input_ic_lora: 'IC · Ref',
}

const TRASH_STORAGE_KEY = 'lora.libraryTrash.v1'

function loadTrashedIds(): Set<string> {
  try {
    const value = JSON.parse(window.localStorage.getItem(TRASH_STORAGE_KEY) ?? '[]')
    return new Set(Array.isArray(value) ? value.filter((id): id is string => typeof id === 'string') : [])
  } catch {
    return new Set()
  }
}

function saveTrashedIds(ids: Set<string>): void {
  try {
    window.localStorage.setItem(TRASH_STORAGE_KEY, JSON.stringify([...ids]))
  } catch {
    // Recoverable deletion is best-effort when browser storage is unavailable.
  }
}

export interface LoraPickerValue {
  entry: LoraInferenceEntry
  conditioningType: LoraInferenceConditioningType
}

interface LoraPickerPopoverProps {
  open: boolean
  selectedId: string | null
  conditioningType: LoraInferenceConditioningType
  entries: LoraInferenceEntry[]
  loading: boolean
  error: string | null
  onRefresh: () => void
  onSelect: (value: LoraPickerValue) => void
  onClose: () => void
  onDeleted?: (deletedId: string) => void
}

/**
 * Gen Space "Apply LoRA" picker. Lists every LoRA usable from the prompt bar:
 * the official LTX-2 union IC-LoRA (canny/depth/pose) and user-trained adapters
 * from completed training jobs. For `union_control` entries the user also picks
 * a conditioning signal; `standard` and `video_input_ic_lora` carry no control
 * signal (their conditioning is the prompt / a reference video respectively).
 *
 * The registry state is owned by the parent (GenSpace) so there's a single
 * fetch shared with the asset-card "IC-LoRA" action and the RunView bridge.
 */
export function LoraPickerPopover({
  open,
  selectedId,
  conditioningType,
  entries,
  loading,
  error,
  onRefresh,
  onSelect,
  onClose,
  onDeleted,
}: LoraPickerPopoverProps) {
  const ref = useRef<HTMLDivElement>(null)
  const panelId = useId()
  const importNameId = `${panelId}-import-name`
  const importTriggerId = `${panelId}-import-trigger`
  const importHfId = `${panelId}-import-hf`
  const importExampleId = `${panelId}-import-example`
  const editNameId = `${panelId}-edit-name`
  const editDescriptionId = `${panelId}-edit-description`
  const editTriggerId = `${panelId}-edit-trigger`
  const editTemplateId = `${panelId}-edit-template`

  // ----- Import-LoRA form state -----
  const [showImport, setShowImport] = useState(false)
  const [importPath, setImportPath] = useState<string | null>(null)
  const [importName, setImportName] = useState('')
  const [importTrigger, setImportTrigger] = useState('')
  const [importVariant, setImportVariant] = useState<'standard' | 'video_input_ic_lora'>('standard')
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState<string | null>(null)
  // Optional sources the backend uses to auto-configure an accurate per-LoRA
  // system prompt + trigger word (so the adapter actually activates instead of
  // silently no-op'ing from a prompt-structure mismatch). Collapsed by default
  // to keep the import form compact.
  const [showPromptConfig, setShowPromptConfig] = useState(false)
  const [importHfUrl, setImportHfUrl] = useState('')
  const [importExamplePrompt, setImportExamplePrompt] = useState('')
  // Outcome of the backend's prompt-profiling step (built-in profile / Gemini
  // from the HF card / example prompt). Surfaced inline so a profiling failure
  // (no Gemini key, HF page not found, Gemini couldn't parse) is never silent —
  // previously the import just fell back to the default template with no signal.
  const [profileOutcome, setProfileOutcome] = useState<{
    status: 'builtin' | 'configured' | 'skipped' | 'failed'
    message: string | null
  } | null>(null)

  // ----- Per-LoRA prompt-template editor -----
  // The auto-prompt assistant uses each LoRA's (auto-generated, editable) system
  // prompt. Editing lives in a sub-panel so the row list stays uncluttered.
  const [editingEntry, setEditingEntry] = useState<LoraInferenceEntry | null>(null)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [editTrigger, setEditTrigger] = useState('')
  const [editTemplate, setEditTemplate] = useState('')
  const [editSaving, setEditSaving] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)
  // ----- Imported-LoRA deletion -----
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [trashedIds, setTrashedIds] = useState<Set<string>>(loadTrashedIds)
  const [showTrash, setShowTrash] = useState(false)

  // ----- List sort -----
  // Applies within each section (Official / Imported / Trained / Unavailable).
  // 'available' surfaces ready-to-use adapters first within a section.
  const [sortMode, setSortMode] = useState<'name-asc' | 'name-desc' | 'available'>('name-asc')
  const [sortMenuOpen, setSortMenuOpen] = useState(false)

  const sortEntries = useCallback(
    (list: LoraInferenceEntry[]): LoraInferenceEntry[] => {
      const sorted = [...list]
      sorted.sort((a, b) => {
        if (sortMode === 'available') {
          // Available first, then by name. Keeps missing-weights rows from
          // crowding the top of a section.
          if (a.available !== b.available) return a.available ? -1 : 1
        }
        const cmp = a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
        return sortMode === 'name-desc' ? -cmp : cmp
      })
      return sorted
    },
    [sortMode],
  )

  const SORT_LABELS: Record<typeof sortMode, string> = {
    'name-asc': 'Name A–Z',
    'name-desc': 'Name Z–A',
    'available': 'Available first',
  }

  const resetImportForm = useCallback(() => {
    setImportPath(null)
    setImportName('')
    setImportTrigger('')
    setImportVariant('standard')
    setImporting(false)
    setImportError(null)
    setShowPromptConfig(false)
    setImportHfUrl('')
    setImportExamplePrompt('')
    setProfileOutcome(null)
  }, [])

  const openEditor = useCallback((entry: LoraInferenceEntry) => {
    setEditingEntry(entry)
    setEditName(entry.name)
    setEditDescription(entry.description ?? '')
    setEditTrigger(entry.triggerWord ?? '')
    setEditTemplate(entry.promptTemplate ?? '')
    setEditError(null)
  }, [])

  const closeEditor = useCallback(() => {
    setEditingEntry(null)
    setEditError(null)
  }, [])

  const persistEditorMetadata = useCallback(async (): Promise<string | null> => {
    if (!editingEntry) return null
    const trimmedName = editName.trim()
    const trimmedDescription = editDescription.trim() || null
    const nameChanged = editingEntry.kind === 'imported'
      && Boolean(trimmedName)
      && trimmedName !== editingEntry.name
    const descriptionChanged = trimmedDescription !== (editingEntry.description ?? null)
    if (editingEntry.kind === 'imported' && (nameChanged || descriptionChanged)) {
      const result = await ApiClient.updateImportedLora(editingEntry.id, {
        ...(nameChanged ? { name: trimmedName } : {}),
        ...(descriptionChanged ? { description: trimmedDescription } : {}),
      })
      return result.ok ? null : (result.error as { message?: string }).message ?? 'Metadata save failed'
    }
    if (editingEntry.kind === 'user_trained' && descriptionChanged) {
      const result = await ApiClient.updateTrainedLora(editingEntry.id, {
        description: trimmedDescription,
      })
      return result.ok ? null : (result.error as { message?: string }).message ?? 'Metadata save failed'
    }
    return null
  }, [editDescription, editName, editingEntry])

  const handleSaveTemplate = useCallback(async () => {
    if (!editingEntry || editSaving) return
    setEditSaving(true)
    setEditError(null)
    const metadataError = await persistEditorMetadata()
    if (metadataError) {
      setEditSaving(false)
      setEditError(metadataError)
      logger.warn(`LoRA metadata save failed (${metadataError})`)
      return
    }

    const templateChanged = editTemplate !== (editingEntry.promptTemplate ?? '')
    const result = await ApiClient.updateLoraPromptTemplate(editingEntry.id, {
      promptTemplate:
        templateChanged || editingEntry.promptTemplateCustomized
          ? editTemplate.trim() || null
          : null,
      triggerWord: editTrigger.trim() ? editTrigger : null,
    })
    setEditSaving(false)
    if (!result.ok) {
      const message = (result.error as { message?: string })?.message ?? 'Save failed'
      setEditError(message)
      logger.warn(`LoRA prompt-template save failed (${message})`)
      return
    }
    await onRefresh()
    closeEditor()
  }, [editingEntry, editSaving, editTemplate, editTrigger, onRefresh, closeEditor, persistEditorMetadata])

  const handleResetToDefault = useCallback(async () => {
    if (!editingEntry || editSaving) return
    if (!await confirmAction({
      title: 'Regenerate system prompt?',
      message: 'The current custom prompt will be replaced using the behavior description and trigger word.',
      confirmLabel: 'Regenerate',
    })) return
    setEditSaving(true)
    setEditError(null)
    const metadataError = await persistEditorMetadata()
    if (metadataError) {
      setEditSaving(false)
      setEditError(metadataError)
      return
    }
    const result = await ApiClient.updateLoraPromptTemplate(editingEntry.id, {
      promptTemplate: null,
      triggerWord: editTrigger.trim() || null,
    })
    setEditSaving(false)
    if (!result.ok) {
      const message = (result.error as { message?: string })?.message ?? 'Reset failed'
      setEditError(message)
      logger.warn(`LoRA prompt-template reset failed (${message})`)
      return
    }
    await onRefresh()
    // The refresh repopulates entries; reflect the regenerated default locally.
    const refreshed = result.data.entry as unknown as LoraInferenceEntry
    if (!refreshed.promptTemplate) {
      setEditError(
        'The backend did not return a generated system prompt. Your previous text was preserved; restart the app backend and try Regenerate again.',
      )
      setEditingEntry(refreshed)
      setEditName(refreshed.name)
      setEditDescription(refreshed.description ?? editDescription)
      setEditTrigger(refreshed.triggerWord ?? editTrigger)
      return
    }
    setEditingEntry(refreshed)
    setEditName(refreshed.name)
    setEditDescription(refreshed.description ?? '')
    setEditTrigger(refreshed.triggerWord ?? '')
    setEditTemplate(refreshed.promptTemplate ?? '')
  }, [
    editDescription,
    editingEntry,
    editSaving,
    editTrigger,
    onRefresh,
    persistEditorMetadata,
  ])

  const handleShowInFolder = useCallback(async (entry: LoraInferenceEntry) => {
    // Imported LoRAs are copied into app storage with a real on-disk path; the
    // IPC opens that file's containing folder (with the file selected) so the
    // user can see where their imported weights live.
    if (entry.kind !== 'imported' || !entry.localPath) return
    try {
      await window.electronAPI.showItemInFolder({ filePath: entry.localPath })
    } catch (err) {
      logger.warn(`Show LoRA in folder failed (${String(err)})`)
    }
  }, [])

  const handleDeleteImported = useCallback(async (entry: LoraInferenceEntry) => {
    if (entry.kind !== 'imported' || deletingId) return
    if (!await confirmAction({
      title: `Delete “${entry.name}” permanently?`,
      message: 'Its managed weights will be removed. This cannot be undone.',
      confirmLabel: 'Delete permanently',
      variant: 'destructive',
    })) {
      return
    }
    setDeletingId(entry.id)
    setDeleteError(null)
    const result = await ApiClient.deleteImportedLora(entry.id)
    setDeletingId(null)
    if (!result.ok) {
      const code = (result.error as { code?: string })?.code
      // DELETE returns 204 No Content, which the typed client surfaces as a
      // synthetic EMPTY_SUCCESS_RESPONSE "error" — that's success, not a
      // failure. 404 means it's already gone — treat as success and resync.
      if (code !== 'EMPTY_SUCCESS_RESPONSE' && result.status !== 404) {
        const message = (result.error as { message?: string })?.message ?? 'Remove failed'
        setDeleteError(message)
        logger.warn(`LoRA delete failed (${message})`)
        return
      }
    }
    setTrashedIds((current) => {
      const next = new Set(current)
      next.delete(entry.id)
      saveTrashedIds(next)
      return next
    })
    await onRefresh()
    onDeleted?.(entry.id)
  }, [deletingId, onRefresh, onDeleted])

  const moveToTrash = useCallback((entry: LoraInferenceEntry) => {
    if (entry.kind !== 'imported' && entry.kind !== 'user_trained') return
    setTrashedIds((current) => {
      const next = new Set(current)
      next.add(entry.id)
      saveTrashedIds(next)
      return next
    })
    onDeleted?.(entry.id)
  }, [onDeleted])

  const restoreFromTrash = useCallback((entryId: string) => {
    setTrashedIds((current) => {
      const next = new Set(current)
      next.delete(entryId)
      saveTrashedIds(next)
      return next
    })
  }, [])

  const deletePermanently = useCallback(async (entry: LoraInferenceEntry) => {
    if (deletingId) return
    if (entry.kind === 'imported') {
      await handleDeleteImported(entry)
      return
    }
    if (entry.kind !== 'user_trained') return
    if (!await confirmAction({
      title: `Delete “${entry.name}” permanently?`,
      message: 'The trained adapter will be removed. This cannot be undone.',
      confirmLabel: 'Delete permanently',
      variant: 'destructive',
    })) return
    setDeletingId(entry.id)
    setDeleteError(null)
    const result = await ApiClient.deleteTrainedLora(entry.id)
    setDeletingId(null)
    if (!result.ok) {
      const code = (result.error as { code?: string })?.code
      if (code !== 'EMPTY_SUCCESS_RESPONSE' && result.status !== 404) {
        setDeleteError((result.error as { message?: string })?.message ?? 'Delete failed')
        return
      }
    }
    restoreFromTrash(entry.id)
    await onRefresh()
    onDeleted?.(entry.id)
  }, [deletingId, handleDeleteImported, onDeleted, onRefresh, restoreFromTrash])

  const handlePickFile = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select LoRA weights',
      filters: [{ name: 'Safe LoRA weights', extensions: ['safetensors'] }],
    })
    if (paths && paths.length > 0) {
      const filePath = paths[0]
      setImportPath(filePath)
      // Pre-fill the name from the filename if the user hasn't typed one.
      const derivedName = filePath.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') ?? ''
      setImportName((prev) => prev || derivedName)
      setImportError(null)
    }
  }, [])

  const handleImport = useCallback(async () => {
    if (!importPath || !importName.trim() || importing) return
    setImporting(true)
    setImportError(null)
    setProfileOutcome(null)
    const result = await ApiClient.importLora({
      sourcePath: importPath,
      name: importName.trim(),
      variant: importVariant,
      triggerWord: importTrigger.trim() || null,
      huggingfaceUrl: importHfUrl.trim() || null,
      examplePrompt: importExamplePrompt.trim() || null,
    })
    setImporting(false)
    if (!result.ok) {
      const message = (result.error as { message?: string })?.message ?? 'Import failed'
      setImportError(message)
      logger.warn(`LoRA import failed (${message})`)
      return
    }
    const data = result.data
    const profileStatus = data.profileStatus
    const profileMessage = data.profileMessage ?? null
    // A failed profiling (no Gemini key / HF page not found / Gemini parse
    // failure) keeps the modal open with a visible reason instead of silently
    // closing on the default template — the user can fix the URL/key and retry,
    // or close manually. Success states close and select the LoRA.
    if (profileStatus === 'failed') {
      setProfileOutcome({ status: profileStatus, message: profileMessage })
      await onRefresh()
      return
    }
    const entry = data.entry as unknown as LoraInferenceEntry
    resetImportForm()
    setShowImport(false)
    await onRefresh()
    onSelect({ entry, conditioningType: 'canny' })
  }, [
    importPath,
    importName,
    importTrigger,
    importing,
    importVariant,
    importHfUrl,
    importExamplePrompt,
    onRefresh,
    onSelect,
    resetImportForm,
  ])

  // Close on outside click / escape.
  useEffect(() => {
    if (!open) return
    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
    ref.current?.focus()
    const onPointer = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('mousedown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onPointer)
      document.removeEventListener('keydown', onKey)
      if (previouslyFocused?.isConnected) previouslyFocused.focus()
    }
  }, [open, onClose])

  if (!open) return null

  const activeEntries = entries.filter((entry) => !trashedIds.has(entry.id))
  const trashedEntries = entries.filter(
    (entry) =>
      trashedIds.has(entry.id) &&
      (entry.kind === 'imported' || entry.kind === 'user_trained'),
  )
  const official = activeEntries.filter((e) => e.kind === 'official_union')
  const imported = activeEntries.filter((e) => e.kind === 'imported')
  const userTrained = activeEntries.filter((e) => e.kind === 'user_trained')
  const availableUser = userTrained.filter((e) => e.available)
  const unavailableUser = userTrained.filter((e) => !e.available)

  const renderRow = (entry: LoraInferenceEntry) => {
    const isSelected = entry.id === selectedId
    // User-trained adapters go missing when their weights file is deleted —
    // keep those disabled. The official union entry stays selectable even when
    // its checkpoint isn't downloaded yet, so the user can pick it and let the
    // Gen Space download gate fetch the union adapter + preprocessing models.
    const dim = !entry.available && entry.kind !== 'official_union'
    return (
      <div
        key={entry.id}
        className={`flex w-full items-center gap-1 rounded-lg transition-colors ${
          isSelected
            ? 'bg-blue-600/20 ring-1 ring-blue-500/40'
            : dim
              ? 'opacity-40 cursor-not-allowed hover:bg-transparent'
              : 'hover:bg-zinc-800/60'
        }`}
      >
        <button
          type="button"
          disabled={dim}
          aria-pressed={isSelected}
          aria-label={`${entry.name}, ${VARIANT_CHIP[entry.variant]}${entry.available ? '' : ', unavailable'}`}
          onClick={() =>
            onSelect({
              entry,
              // Conditioning type is chosen in the prompt bar toolbar; pass a
              // valid default so the toolbar starts from a supported value.
              conditioningType: entry.variant === 'union_control'
                ? entry.conditioningTypes.includes(conditioningType)
                  ? conditioningType
                  : entry.conditioningTypes[0]
                : 'canny',
            })
          }
          className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-2 text-left"
        >
          <span className="shrink-0">
            {entry.kind === 'official_union' ? (
              <Sparkles className="h-4 w-4 text-blue-400" />
            ) : (
              <Box className="h-4 w-4 text-zinc-400" />
            )}
          </span>
          <span className="text-sm text-zinc-100 font-medium truncate flex-1">{entry.name}</span>
          {!entry.available && (
            <AlertCircle className="h-3.5 w-3.5 text-amber-400/80 shrink-0" aria-hidden="true" />
          )}
          <span className="text-[10px] uppercase tracking-wide text-zinc-500 shrink-0">
            {VARIANT_CHIP[entry.variant]}
          </span>
          {isSelected && <Check className="h-3.5 w-3.5 text-blue-400 shrink-0" aria-hidden="true" />}
        </button>
        {(entry.variant !== 'standard' || entry.promptTemplate != null) && (
          <button
            type="button"
            title="Edit AI prompt template"
            aria-label={`Edit prompt template for ${entry.name}`}
            onClick={(e) => {
              e.stopPropagation()
              openEditor(entry)
            }}
            className="shrink-0 p-1 -m-1 rounded text-zinc-500 hover:text-blue-300 hover:bg-zinc-800/80 transition-colors"
          >
            <Pencil className="h-3.5 w-3.5" />
          </button>
        )}
        {entry.kind === 'imported' && entry.localPath && (
          <button
            type="button"
            title="Show in folder"
            aria-label={`Show ${entry.name} in folder`}
            onClick={(e) => {
              e.stopPropagation()
              void handleShowInFolder(entry)
            }}
            className="shrink-0 p-1 -m-1 rounded text-zinc-500 hover:text-blue-300 hover:bg-zinc-800/80 transition-colors"
          >
            <FolderOpen className="h-3.5 w-3.5" />
          </button>
        )}
        {(entry.kind === 'imported' || entry.kind === 'user_trained') && (
          <button
            type="button"
            title="Move to Trash"
            aria-label={`Move ${entry.name} to Trash`}
            disabled={deletingId === entry.id}
            onClick={(e) => {
              e.stopPropagation()
              moveToTrash(entry)
            }}
            className="shrink-0 p-1 -m-1 rounded text-zinc-500 hover:text-red-400 hover:bg-zinc-800/80 transition-colors disabled:opacity-60"
          >
            {deletingId === entry.id ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Trash2 className="h-3.5 w-3.5" />
            )}
          </button>
        )}
      </div>
    )
  }

  const editing = editingEntry !== null

  return (
    <div
      ref={ref}
      id={panelId}
      role="dialog"
      aria-label={showTrash ? 'LoRA Trash' : showImport ? 'Import LoRA' : editing ? 'Edit LoRA prompt' : 'Apply a LoRA'}
      tabIndex={-1}
      className={`fixed inset-x-2 bottom-20 flex max-h-[min(70dvh,35rem)] flex-col overflow-hidden rounded-xl border border-zinc-700/80 bg-zinc-900/95 shadow-2xl shadow-black/50 backdrop-blur focus:outline-none sm:absolute sm:inset-x-auto sm:bottom-full sm:left-0 sm:mb-2 ${
        editing ? 'sm:w-[560px]' : 'sm:w-[300px]'
      } z-50`}
    >
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-zinc-800/80">
        {showTrash ? (
          <button
            type="button"
            onClick={() => setShowTrash(false)}
            className="flex items-center gap-1.5 text-xs text-zinc-300 font-medium hover:text-white transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Trash
          </button>
        ) : showImport ? (
          <button
            type="button"
            onClick={() => { setShowImport(false); resetImportForm() }}
            className="flex items-center gap-1.5 text-xs text-zinc-300 font-medium hover:text-white transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back
          </button>
        ) : editing ? (
          <button
            type="button"
            onClick={closeEditor}
            className="flex items-center gap-1.5 text-xs text-zinc-300 font-medium hover:text-white transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back
          </button>
        ) : (
          <div className="flex items-center gap-2 text-xs text-zinc-300 font-medium">
            <Layers className="h-3.5 w-3.5 text-blue-400" />
            Apply a LoRA
          </div>
        )}
        {!showTrash && !showImport && !editing && (
          <button
            type="button"
            onClick={() => { setShowImport(true); setImportError(null) }}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-md bg-blue-600 px-2.5 py-1.5 text-[11px] font-medium text-white transition-colors hover:bg-blue-500"
          >
            <Download className="h-3.5 w-3.5" />
            Import
          </button>
        )}
      </div>

      {!showTrash && !showImport && !editing && (
        <div className="flex items-center gap-1.5 border-b border-zinc-800/80 px-3 py-2">
          <div className="relative min-w-0 flex-1">
              <button
                type="button"
                onClick={() => setSortMenuOpen((v) => !v)}
                aria-label={`Sort LoRAs: ${SORT_LABELS[sortMode]}`}
                aria-expanded={sortMenuOpen}
                className="flex h-7 w-full min-w-0 items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-800/50 px-2 text-[11px] text-zinc-300 transition-colors hover:bg-zinc-800 hover:text-white"
                title={`Sort: ${SORT_LABELS[sortMode]}`}
              >
                <ArrowUpDown className="h-3.5 w-3.5" />
                <span className="min-w-0 flex-1 truncate text-left">{SORT_LABELS[sortMode]}</span>
                <ChevronDown className="h-3 w-3 shrink-0 text-zinc-500" />
              </button>
              {sortMenuOpen && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setSortMenuOpen(false)}
                  />
                  <div role="menu" aria-label="Sort LoRAs" className="absolute left-0 top-full mt-1 z-50 w-40 rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl shadow-black/50 py-1">
                    {(Object.keys(SORT_LABELS) as (keyof typeof SORT_LABELS)[]).map((key) => (
                      <button
                        key={key}
                        type="button"
                        role="menuitemradio"
                        aria-checked={sortMode === key}
                        onClick={() => { setSortMode(key); setSortMenuOpen(false) }}
                        className={`w-full text-left px-2.5 py-1.5 text-[11px] flex items-center justify-between transition-colors ${
                          sortMode === key
                            ? 'text-blue-300 bg-blue-600/15'
                            : 'text-zinc-300 hover:bg-zinc-800/80'
                        }`}
                      >
                        {SORT_LABELS[key]}
                        {sortMode === key && <Check className="h-3 w-3" />}
                      </button>
                    ))}
                  </div>
                </>
              )}
          </div>
          <Tooltip content="Open LoRA Trash" side="bottom">
            <button
              type="button"
              onClick={() => setShowTrash(true)}
              aria-label={`Open LoRA Trash${trashedEntries.length ? `, ${trashedEntries.length} items` : ''}`}
              className="relative flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-white"
            >
              <Trash2 className="h-3.5 w-3.5" />
              {trashedEntries.length > 0 && (
                <span className="absolute -right-1 -top-1 min-w-3.5 rounded-full bg-red-500 px-1 text-center text-[8px] font-semibold leading-3.5 text-white">
                  {trashedEntries.length > 99 ? '99+' : trashedEntries.length}
                </span>
              )}
            </button>
          </Tooltip>
          <Tooltip content="Refresh LoRA list" side="bottom">
            <button
              type="button"
              onClick={onRefresh}
              aria-label="Refresh LoRA list"
              className="flex h-7 w-7 items-center justify-center rounded-md text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </Tooltip>
        </div>
      )}

      {showTrash ? (
        <div className="overflow-y-auto p-2 space-y-2">
          <p className="px-1 text-[10px] leading-relaxed text-zinc-500">
            Trashed LoRAs are hidden from Apply LoRA but their files remain until you delete them permanently.
          </p>
          {trashedEntries.length === 0 ? (
            <p className="py-6 text-center text-xs text-zinc-500">Trash is empty.</p>
          ) : (
            trashedEntries.map((entry) => (
              <div key={entry.id} className="rounded-lg border border-zinc-800 bg-zinc-800/30 p-2">
                <div className="flex items-center gap-2">
                  <Box className="h-4 w-4 text-zinc-500" />
                  <span className="min-w-0 flex-1 truncate text-xs font-medium text-zinc-200">
                    {entry.name}
                  </span>
                  <span className="text-[9px] uppercase text-zinc-600">
                    {entry.kind === 'imported' ? 'Imported' : 'Trained'}
                  </span>
                </div>
                <div className="mt-2 flex gap-1.5">
                  <button
                    type="button"
                    onClick={() => restoreFromTrash(entry.id)}
                    className="flex-1 rounded border border-zinc-700 px-2 py-1 text-[10px] text-zinc-200 hover:bg-zinc-800"
                  >
                    Restore
                  </button>
                  <button
                    type="button"
                    disabled={deletingId === entry.id}
                    onClick={() => void deletePermanently(entry)}
                    className="flex-1 rounded border border-red-500/40 px-2 py-1 text-[10px] text-red-300 hover:bg-red-500/10 disabled:opacity-50"
                  >
                    {deletingId === entry.id ? 'Deleting…' : 'Delete permanently'}
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      ) : showImport ? (
        <div className="overflow-y-auto p-3 space-y-3">
          <p className="text-[11px] text-zinc-500">
            Bring in a LoRA you got from elsewhere. The weights are copied into
            LTX Desktop so they stay available even if the original file moves.
          </p>

          <div className="space-y-1.5">
            <span className="block text-[10px] uppercase tracking-wider text-zinc-500">LoRA file</span>
            <button
              type="button"
              onClick={handlePickFile}
              className="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg border border-zinc-700 bg-zinc-800/60 hover:bg-zinc-800 text-xs text-zinc-300 transition-colors"
            >
              <Download className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate flex-1 text-left">
                {importPath ? importPath.split(/[\\/]/).pop() : 'Choose a .safetensors file'}
              </span>
            </button>
          </div>

          <div className="space-y-1.5">
            <label htmlFor={importNameId} className="text-[10px] uppercase tracking-wider text-zinc-500">Name</label>
            <input
              id={importNameId}
              type="text"
              value={importName}
              onChange={(e) => setImportName(e.target.value)}
              placeholder="e.g. My Style LoRA"
              className="w-full px-2.5 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="space-y-1.5">
            <label htmlFor={importTriggerId} className="text-[10px] uppercase tracking-wider text-zinc-500">Trigger word</label>
            <input
              id={importTriggerId}
              type="text"
              value={importTrigger}
              onChange={(e) => setImportTrigger(e.target.value)}
              placeholder="e.g. instant shave"
              spellCheck={false}
              className="w-full px-2.5 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
            />
            <p className="text-[10px] text-zinc-500 leading-relaxed">
              The token the LoRA was trained with. Pre-filled from the name — correct it if that's
              wrong. It drives the auto-prompt system prompt so the adapter actually activates.
            </p>
          </div>

          <fieldset className="space-y-1.5">
            <legend className="text-[10px] uppercase tracking-wider text-zinc-500">Type</legend>
            <div className="grid grid-cols-1 gap-1.5">
              {IMPORT_VARIANTS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  aria-pressed={importVariant === opt.value}
                  onClick={() => setImportVariant(opt.value)}
                  className={`flex items-center justify-between px-2.5 py-2 rounded-lg border text-left transition-colors ${
                    importVariant === opt.value
                      ? 'border-blue-500/60 bg-blue-600/15'
                      : 'border-zinc-700 bg-zinc-800/40 hover:bg-zinc-800'
                  }`}
                >
                  <div>
                    <div className="text-xs text-zinc-100 font-medium">{opt.label}</div>
                    <div className="text-[10px] text-zinc-500">{opt.hint}</div>
                  </div>
                  {importVariant === opt.value && <Check className="h-3.5 w-3.5 text-blue-300" />}
                </button>
              ))}
            </div>
          </fieldset>

          <div className="space-y-1.5">
            <button
              type="button"
              onClick={() => setShowPromptConfig((v) => !v)}
              aria-expanded={showPromptConfig}
              className="w-full flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-zinc-400 hover:text-zinc-200"
            >
              {showPromptConfig ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
              Auto-configure the system prompt (optional)
            </button>
            {showPromptConfig && (
              <div className="space-y-2.5 rounded-lg border border-zinc-700/70 bg-zinc-800/30 p-2.5">
                <p className="text-[10px] text-zinc-500 leading-relaxed">
                  A LoRA only activates if the prompt matches the structure and trigger word it was trained on —
                  information that isn't in the file. Paste either source and Gemini will configure an accurate
                  system prompt + trigger for it. You can always edit it later.
                </p>
                <div className="space-y-1">
                  <label htmlFor={importHfId} className="text-[10px] uppercase tracking-wider text-zinc-500">HuggingFace page URL</label>
                  <input
                    id={importHfId}
                    type="url"
                    value={importHfUrl}
                    onChange={(e) => setImportHfUrl(e.target.value)}
                    placeholder="https://huggingface.co/<org>/<repo>"
                    spellCheck={false}
                    className="w-full px-2.5 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor={importExampleId} className="text-[10px] uppercase tracking-wider text-zinc-500">
                    Example prompt <span className="normal-case text-zinc-600">(for LoRAs with no HF page)</span>
                  </label>
                  <textarea
                    id={importExampleId}
                    value={importExamplePrompt}
                    onChange={(e) => setImportExamplePrompt(e.target.value)}
                    placeholder="Paste a prompt the LoRA was trained on…"
                    rows={3}
                    spellCheck={false}
                    className="w-full px-2.5 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500 resize-none"
                  />
                </div>
              </div>
            )}
          </div>

          {importError && (
            <div role="alert" className="text-[11px] text-red-400 flex items-start gap-1.5">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span>{importError}</span>
            </div>
          )}

          {profileOutcome && (
            <div
              role="status"
              className={`text-[11px] flex items-start gap-1.5 rounded-lg border p-2 ${
                profileOutcome.status === 'failed'
                  ? 'text-amber-300 border-amber-500/40 bg-amber-500/10'
                  : 'text-emerald-300 border-emerald-500/40 bg-emerald-500/10'
              }`}
            >
              {profileOutcome.status === 'failed' ? (
                <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              ) : (
                <Check className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              )}
              <span>
                {profileOutcome.message ??
                  (profileOutcome.status === 'failed'
                    ? 'Could not auto-configure the system prompt. You can edit it manually after import.'
                    : 'System prompt auto-configured.')}
              </span>
            </div>
          )}

          <button
            type="button"
            onClick={handleImport}
            disabled={!importPath || !importName.trim() || importing}
            className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {importing ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                {importHfUrl.trim() || importExamplePrompt.trim() ? 'Configuring prompt…' : 'Importing...'}
              </>
            ) : (
              <>
                <Download className="h-3.5 w-3.5" />
                Import LoRA
              </>
            )}
          </button>
        </div>
      ) : editing && editingEntry ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
          <p className="text-[11px] text-zinc-500">
            Gemini Flash uses this system prompt to write a tailored prompt from
            the reference video. Generated prompts use the behavior description
            and exact trigger—not the display name.
          </p>

          <div className="space-y-1.5">
            <span className="block text-[10px] uppercase tracking-wider text-zinc-500">Name</span>
            {editingEntry.kind === 'imported' ? (
              <>
                <input
                  id={editNameId}
                  aria-label="LoRA name"
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  placeholder="e.g. Instant Shave"
                  className="w-full px-2.5 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
                />
                <p className="text-[10px] text-zinc-500 leading-relaxed">
                  The name is only a library label and never changes prompt instructions.
                </p>
              </>
            ) : (
              <p className="text-xs text-zinc-100 font-medium px-2.5 py-1.5 rounded-lg border border-zinc-800 bg-zinc-900/40">
                {editingEntry.name}
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <label htmlFor={editDescriptionId} className="text-[10px] uppercase tracking-wider text-zinc-500">
              What this LoRA does
            </label>
            <textarea
              id={editDescriptionId}
              value={editDescription}
              onChange={(event) => setEditDescription(event.target.value)}
              disabled={editingEntry.kind === 'official_union'}
              maxLength={500}
              rows={2}
              placeholder="e.g. Removes foreground people and reconstructs the hidden background"
              className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-800/60 px-2.5 py-1.5 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500 disabled:opacity-60"
            />
            <p className="text-[10px] text-zinc-500 leading-relaxed">
              Used to generate the default system prompt. Leave blank for conservative generic guidance.
            </p>
          </div>

          <div className="space-y-1.5">
            <label htmlFor={editTriggerId} className="text-[10px] uppercase tracking-wider text-zinc-500">Trigger word</label>
            <input
              id={editTriggerId}
              type="text"
              value={editTrigger}
              onChange={(e) => setEditTrigger(e.target.value)}
              placeholder="e.g. conehead"
              spellCheck={false}
              className="w-full px-2.5 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800/60 text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
            />
            <p className="text-[10px] text-zinc-500 leading-relaxed">
              Custom prompt text is never rewritten automatically. Regenerate below to safely rebuild it from this exact trigger.
            </p>
          </div>

          <div className="space-y-1.5">
            <label htmlFor={editTemplateId} className="text-[10px] uppercase tracking-wider text-zinc-500">System prompt</label>
            <textarea
              id={editTemplateId}
              value={editTemplate}
              onChange={(e) => setEditTemplate(e.target.value)}
              rows={6}
              spellCheck={false}
              className="max-h-[30dvh] min-h-28 w-full resize-y overflow-y-auto rounded-lg border border-zinc-700 bg-zinc-800/60 px-2.5 py-2 font-mono text-[11px] leading-relaxed text-zinc-200 focus:outline-none focus:border-blue-500"
            />
          </div>

          {editError && (
            <div role="alert" className="text-[11px] text-red-400 flex items-start gap-1.5">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span>{editError}</span>
            </div>
          )}
          </div>

          <div data-testid="prompt-editor-actions" className="flex shrink-0 items-center gap-2 border-t border-zinc-800 p-3">
            <button
              type="button"
              onClick={handleSaveTemplate}
              disabled={editSaving}
              className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {editSaving ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Check className="h-3.5 w-3.5" />
                  Save
                </>
              )}
            </button>
            <button
              type="button"
              onClick={handleResetToDefault}
              disabled={editSaving}
              title="Regenerate from the behavior description and trigger"
              className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-zinc-700 text-zinc-300 text-xs font-medium hover:bg-zinc-800 transition-colors disabled:opacity-50"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Regenerate
            </button>
          </div>
        </div>
      ) : (
        <div className="overflow-y-auto p-1.5 space-y-2">
          {error && (
            <div role="alert" className="text-[11px] text-amber-400/90 px-2 py-1.5">{error}</div>
          )}
          {deleteError && (
            <div role="alert" className="text-[11px] text-red-400 flex items-start gap-1.5 px-2 py-1.5">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span>{deleteError}</span>
            </div>
          )}

          {official.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-500 px-1.5 mb-0.5">
                Official
              </p>
              <div className="space-y-0.5">{sortEntries(official).map(renderRow)}</div>
            </div>
          )}

          {imported.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-500 px-1.5 mb-0.5">
                Imported
              </p>
              <div className="space-y-0.5">{sortEntries(imported).map(renderRow)}</div>
            </div>
          )}

          {availableUser.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-500 px-1.5 mb-0.5">
                Your trained LoRAs
              </p>
              <div className="space-y-0.5">{sortEntries(availableUser).map(renderRow)}</div>
            </div>
          )}

          {unavailableUser.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-600 px-1.5 mb-0.5">
                Unavailable
              </p>
              <div className="space-y-0.5">{sortEntries(unavailableUser).map(renderRow)}</div>
            </div>
          )}

          {!loading && !error && entries.length === 0 && (
            <div className="text-center py-6 px-3">
              <p className="text-sm text-zinc-400">No LoRAs yet</p>
              <p className="text-[11px] text-zinc-600 mt-1">
                Train a LoRA, import one, or download the official IC-LoRA to apply it here.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
