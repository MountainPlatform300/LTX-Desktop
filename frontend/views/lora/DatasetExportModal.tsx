import { useState } from 'react'
import { FileArchive, FolderOutput, Info, Loader2, X } from 'lucide-react'
import { useLoraTraining } from '../../contexts/LoraTrainingContext'
import { Tooltip } from '../../components/ui/tooltip'

export type ExportFormat = 'folder' | 'zip'

export interface IcLoraExportSettings {
  fps: number
  shortSide: number
  bucketFrames: number
  forbiddenWords: string[]
}

/** Supplementary files to write next to the always-included dataset. */
export interface ExportComponents {
  config: boolean
  readme: boolean
  manifest: boolean
  modelCard: boolean
}

export interface DatasetExportOptions {
  format: ExportFormat
  includeRejected: boolean
  profileId: string | null
  components: ExportComponents
  icLora?: IcLoraExportSettings
}

/**
 * Picks how to write a portable, trainer-ready dataset bundle:
 *
 * - **Folder** — a relocatable directory (`dataset.json` + `clips/` +
 *   `README.md` + `train_config.yaml` + `ltxdesktop.json`).
 * - **Zip** — the same, packed into a single shareable archive.
 *
 * Rejected clips are dropped by default (they never train); the toggle is
 * only offered when the collection actually has some.
 */
/** One include/exclude row with a label and a hover-help info icon. */
function CheckRow({
  checked,
  disabled,
  label,
  tip,
  onChange,
}: {
  checked: boolean
  disabled?: boolean
  label: string
  tip: string
  onChange: (value: boolean) => void
}) {
  return (
    <div className="flex items-center gap-2">
      <label
        className={`flex items-center gap-2.5 select-none flex-1 min-w-0 ${
          disabled ? 'cursor-default' : 'cursor-pointer'
        }`}
      >
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
          className="h-3.5 w-3.5 accent-blue-500 shrink-0 disabled:opacity-60"
        />
        <span className="text-xs text-zinc-300 truncate">{label}</span>
      </label>
      <Tooltip content={tip} side="left" wide>
        <Info className="h-3.5 w-3.5 text-zinc-500 hover:text-zinc-300 shrink-0" />
      </Tooltip>
    </div>
  )
}

export function DatasetExportModal({
  datasetName,
  totalClips,
  rejectedCount,
  isIcLora = false,
  busy,
  onClose,
  onExport,
}: {
  datasetName: string
  totalClips: number
  rejectedCount: number
  isIcLora?: boolean
  busy: boolean
  onClose: () => void
  onExport: (opts: DatasetExportOptions) => void
}) {
  const { profiles } = useLoraTraining()
  const [format, setFormat] = useState<ExportFormat>('folder')
  const [includeRejected, setIncludeRejected] = useState(false)
  const [profileId, setProfileId] = useState<string | null>(profiles[0]?.id ?? null)
  // IC-LoRA training-ready normalization. Defaults match a clean run.
  const [fps, setFps] = useState(25)
  const [shortSide, setShortSide] = useState(576)
  const [bucketFrames, setBucketFrames] = useState(49)
  const [forbiddenWords, setForbiddenWords] = useState('')
  const [components, setComponents] = useState<ExportComponents>({
    config: true,
    readme: true,
    manifest: true,
    modelCard: true,
  })

  const shipCount = includeRejected ? totalClips : totalClips - rejectedCount

  const buildOptions = (): DatasetExportOptions => ({
    format,
    includeRejected,
    profileId,
    components,
    ...(isIcLora
      ? {
          icLora: {
            fps,
            shortSide,
            bucketFrames,
            forbiddenWords: forbiddenWords
              .split(',')
              .map((w) => w.trim())
              .filter(Boolean),
          },
        }
      : {}),
  })

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={busy ? undefined : onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-md mx-4">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="text-base font-semibold text-white">Export dataset</h2>
          <button
            onClick={onClose}
            disabled={busy}
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-40"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4">
          <p className="text-xs text-zinc-400 leading-relaxed">
            Writes a self-contained, trainer-ready bundle for{' '}
            <span className="text-zinc-200 font-medium">{datasetName}</span> — a relative{' '}
            <code className="text-zinc-300">dataset.json</code> plus the clips, a ready-to-run{' '}
            <code className="text-zinc-300">train_config.yaml</code>, and a README with the exact
            commands. It can be re-imported into another LTX Desktop installation.
          </p>

          <div className="grid grid-cols-2 gap-2">
            {([
              { id: 'folder' as const, icon: FolderOutput, label: 'Folder', hint: 'Relocatable directory' },
              { id: 'zip' as const, icon: FileArchive, label: 'Zip archive', hint: 'Single shareable file' },
            ]).map((opt) => {
              const active = format === opt.id
              const Icon = opt.icon
              return (
                <button
                  key={opt.id}
                  onClick={() => setFormat(opt.id)}
                  disabled={busy}
                  className={`flex flex-col items-start gap-1 rounded-lg border p-3 text-left transition-colors disabled:opacity-50 ${
                    active
                      ? 'border-blue-500 bg-blue-500/10'
                      : 'border-zinc-700 bg-zinc-800/40 hover:border-zinc-600'
                  }`}
                >
                  <Icon className={`h-4 w-4 ${active ? 'text-blue-300' : 'text-zinc-400'}`} />
                  <span className="text-xs font-medium text-zinc-200">{opt.label}</span>
                  <span className="text-[11px] text-zinc-500">{opt.hint}</span>
                </button>
              )
            })}
          </div>

          {profiles.length > 0 && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-zinc-300">Training profile</label>
              <select
                value={profileId ?? ''}
                onChange={(e) => setProfileId(e.target.value || null)}
                disabled={busy}
                className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              >
                {profiles.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
              <p className="text-[11px] text-zinc-500">
                Seeds the bundled <code className="text-zinc-400">train_config.yaml</code> (rank, steps,
                validation, etc.). You'll still fill in the model paths.
              </p>
            </div>
          )}

          <div className="space-y-2">
            <p className="text-xs font-medium text-zinc-300">What to include</p>
            <div className="rounded-lg border border-zinc-800 bg-zinc-800/30 p-3 space-y-2">
              <CheckRow
                checked
                disabled
                label="Dataset (clips + dataset.json)"
                tip="The training data itself — the processed clips and the dataset.json the LTX-2 trainer reads. Always included."
                onChange={() => {}}
              />
              {([
                {
                  key: 'config' as const,
                  label: 'Training config (train_config.yaml)',
                  tip: 'A ready-to-run trainer config seeded from the selected training profile (rank, steps, validation…). You still fill in the model paths.',
                },
                {
                  key: 'readme' as const,
                  label: 'Training instructions (README.md)',
                  tip: 'A short README with the exact terminal commands to pre-compute latents and train this dataset with the LTX-2 trainer.',
                },
                {
                  key: 'manifest' as const,
                  label: 'Re-import manifest (ltxdesktop.json)',
                  tip: 'A sidecar the trainer ignores but LTX Desktop reads, so you can re-import this bundle later with captions, trigger word, IC-LoRA pairing and triage intact.',
                },
                {
                  key: 'modelCard' as const,
                  label: 'Model card (MODEL_CARD.md)',
                  tip: 'A fill-in-the-blanks Hugging Face model card for the trained LoRA — pre-filled with the type, trigger word and training settings. Complete the blanks and rename to README.md when you publish.',
                },
              ]).map((opt) => (
                <CheckRow
                  key={opt.key}
                  checked={components[opt.key]}
                  disabled={busy}
                  label={opt.label}
                  tip={opt.tip}
                  onChange={(v) => setComponents((prev) => ({ ...prev, [opt.key]: v }))}
                />
              ))}
            </div>
          </div>

          {isIcLora && (
            <div className="space-y-2.5 rounded-lg border border-zinc-800 bg-zinc-800/30 p-3">
              <p className="text-xs font-medium text-zinc-300">Training-ready normalization</p>
              <p className="text-[11px] text-zinc-500 leading-relaxed">
                Each pair's target + reference are re-encoded to one fps and resolution, trimmed to
                the bucket frame count, rotation baked in and audio stripped — so they align frame for
                frame. Pairs that can't comply (fps/size/length mismatch, or an empty/truncated/
                forbidden caption) are dropped and reported.
              </p>
              <div className="grid grid-cols-3 gap-2">
                <label className="space-y-1">
                  <span className="text-[11px] text-zinc-400">FPS</span>
                  <input
                    type="number" min={1} max={120} value={fps} disabled={busy}
                    onChange={(e) => setFps(Number(e.target.value) || 25)}
                    className="w-full px-2 py-1.5 bg-zinc-800 border border-zinc-700 rounded-md text-xs text-white focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[11px] text-zinc-400">Short side</span>
                  <input
                    type="number" min={32} max={2160} value={shortSide} disabled={busy}
                    onChange={(e) => setShortSide(Number(e.target.value) || 576)}
                    className="w-full px-2 py-1.5 bg-zinc-800 border border-zinc-700 rounded-md text-xs text-white focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[11px] text-zinc-400">Bucket frames</span>
                  <input
                    type="number" min={1} max={2049} value={bucketFrames} disabled={busy}
                    onChange={(e) => setBucketFrames(Number(e.target.value) || 49)}
                    className="w-full px-2 py-1.5 bg-zinc-800 border border-zinc-700 rounded-md text-xs text-white focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
                  />
                </label>
              </div>
              <label className="space-y-1 block">
                <span className="text-[11px] text-zinc-400">Forbidden caption words (comma-separated)</span>
                <input
                  type="text" value={forbiddenWords} disabled={busy}
                  placeholder="e.g. beard, stubble"
                  onChange={(e) => setForbiddenWords(e.target.value)}
                  className="w-full px-2 py-1.5 bg-zinc-800 border border-zinc-700 rounded-md text-xs text-white focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
                />
                <span className="text-[11px] text-zinc-500">
                  The trigger word is always rejected. Bucket frames should be 8k+1 (33, 41, 49…).
                </span>
              </label>
            </div>
          )}

          {rejectedCount > 0 && (
            <label className="flex items-center gap-2.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={includeRejected}
                onChange={(e) => setIncludeRejected(e.target.checked)}
                disabled={busy}
                className="h-3.5 w-3.5 accent-blue-500"
              />
              <span className="text-xs text-zinc-300">
                Include {rejectedCount} rejected clip{rejectedCount === 1 ? '' : 's'}
              </span>
            </label>
          )}

          <p className="text-[11px] text-zinc-500">
            {shipCount} clip{shipCount === 1 ? '' : 's'} will be exported.
          </p>
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex justify-end gap-2">
          <button
            onClick={onClose}
            disabled={busy}
            className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            onClick={() => onExport(buildOptions())}
            disabled={busy || shipCount <= 0}
            className="text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
          >
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {busy ? 'Exporting…' : 'Choose location…'}
          </button>
        </div>
      </div>
    </div>
  )
}
