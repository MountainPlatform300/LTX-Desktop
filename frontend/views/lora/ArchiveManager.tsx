import { ArchiveRestore, Search, Trash2, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import type {
  LoraDataset,
  LoraTrainingJob,
} from '../../contexts/LoraTrainingContext'
import { confirmAction } from '../../components/ui/confirm-dialog'

type ArchivedItem =
  | { kind: 'dataset'; value: LoraDataset; subtitle: string }
  | { kind: 'run'; value: LoraTrainingJob; subtitle: string }

export function ArchiveManager({
  kind,
  datasets,
  runs,
  datasetNameForRun,
  onClose,
  onRestoreDataset,
  onRestoreRun,
  onDeleteDataset,
  onDeleteRun,
}: {
  kind: 'dataset' | 'run'
  datasets: LoraDataset[]
  runs: LoraTrainingJob[]
  datasetNameForRun: (run: LoraTrainingJob) => string | null
  onClose: () => void
  onRestoreDataset: (id: string) => Promise<void>
  onRestoreRun: (id: string) => Promise<void>
  onDeleteDataset: (id: string) => Promise<void>
  onDeleteRun: (id: string) => Promise<void>
}) {
  const [query, setQuery] = useState('')
  const [busyId, setBusyId] = useState<string | null>(null)
  const items = useMemo<ArchivedItem[]>(() => {
    const all: ArchivedItem[] = kind === 'dataset'
      ? datasets.map((value) => ({
          kind: 'dataset',
          value,
          subtitle: `${value.clips.filter((clip) => !clip.deletedAt).length} clips`,
        }))
      : runs.map((value) => ({
          kind: 'run',
          value,
          subtitle: datasetNameForRun(value)
            ? `Dataset: ${datasetNameForRun(value)}`
            : `Status: ${value.status}`,
        }))
    const needle = query.trim().toLowerCase()
    return needle
      ? all.filter((item) => item.value.name.toLowerCase().includes(needle))
      : all
  }, [datasetNameForRun, datasets, kind, query, runs])

  const restore = async (item: ArchivedItem) => {
    setBusyId(item.value.id)
    try {
      if (item.kind === 'dataset') await onRestoreDataset(item.value.id)
      else await onRestoreRun(item.value.id)
    } finally {
      setBusyId(null)
    }
  }

  const permanentlyDelete = async (item: ArchivedItem) => {
    const label = item.kind === 'dataset' ? 'dataset' : 'training run'
    if (!await confirmAction({
      title: `Permanently delete archived ${label}?`,
      message: 'This item and its stored data cannot be restored afterward.',
      confirmLabel: 'Delete permanently',
      variant: 'destructive',
    })) return
    setBusyId(item.value.id)
    try {
      if (item.kind === 'dataset') await onDeleteDataset(item.value.id)
      else await onDeleteRun(item.value.id)
    } finally {
      setBusyId(null)
    }
  }

  const noun = kind === 'dataset' ? 'datasets' : 'runs'
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={`Archived ${noun}`}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="flex max-h-[80vh] w-full max-w-xl flex-col overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <div>
            <h2 className="text-sm font-semibold text-white">Archived {noun}</h2>
            <p className="mt-0.5 text-[11px] text-zinc-500">
              Archived items keep their files and can be restored at any time.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-white"
            aria-label="Close archive"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <label className="mx-4 mt-3 flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-2">
          <Search className="h-3.5 w-3.5 text-zinc-500" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={`Search archived ${noun}`}
            className="min-w-0 flex-1 bg-transparent text-xs text-white outline-none placeholder:text-zinc-600"
          />
        </label>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {items.length === 0 ? (
            <p className="py-8 text-center text-xs text-zinc-600">
              {query ? 'No archived items match your search.' : `No archived ${noun}.`}
            </p>
          ) : (
            <div className="space-y-1.5">
              {items.map((item) => (
                <div
                  key={item.value.id}
                  className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-950/60 px-3 py-2.5"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-zinc-200">{item.value.name}</p>
                    <p className="mt-0.5 truncate text-[10px] text-zinc-500">{item.subtitle}</p>
                  </div>
                  <button
                    type="button"
                    disabled={busyId !== null}
                    onClick={() => void restore(item)}
                    className="inline-flex items-center gap-1 rounded-md bg-blue-600 px-2 py-1.5 text-[10px] font-medium text-white hover:bg-blue-500 disabled:opacity-40"
                  >
                    <ArchiveRestore className="h-3 w-3" />
                    Restore
                  </button>
                  <button
                    type="button"
                    disabled={busyId !== null}
                    onClick={() => void permanentlyDelete(item)}
                    className="rounded-md p-1.5 text-zinc-500 hover:bg-red-500/10 hover:text-red-400 disabled:opacity-40"
                    aria-label={`Permanently delete ${item.value.name}`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
