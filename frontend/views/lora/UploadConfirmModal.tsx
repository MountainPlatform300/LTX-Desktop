import { useMemo, useState } from 'react'
import { AlertTriangle, Check, Cloud, Info, X } from 'lucide-react'
import { PreflightChecklist } from '../../components/lora/DatasetHealth'
import { HARD_MIN_CLIPS, clipWarnings } from '../../lib/lora-quality'
import { isPairTarget } from '../../lib/lora-pairs'
import type { LoraDatasetType } from '../../contexts/LoraTrainingContext'
import type { StudioClip, StudioStoreApi } from '../studio/studio-store'

// Why a candidate clip isn't "ready" to train on. The bar: a hard quality error
// (e.g. resolution too low) or a missing caption. Soft warnings (below-
// recommended size, long clip) advise elsewhere but don't gate the upload.
function notReadyReasons(clip: StudioClip, requireCaption: boolean): string[] {
  const reasons: string[] = []
  if (requireCaption && !(clip.caption ?? '').trim()) reasons.push('No caption')
  for (const w of clipWarnings(clip)) {
    if (w.level === 'error') reasons.push(w.text)
  }
  return reasons
}

function clipName(clip: StudioClip): string {
  return clip.localPath.split(/[\\/]/).pop() || clip.localPath
}

/**
 * Pre-flight confirmation shown before pushing a dataset to the GPU.
 *
 * Default-safe: only training-ready clips upload. Clips that aren't ready
 * (uncaptioned / quality errors) are held back unless the user explicitly opts
 * to include them. Holding back reuses the existing reject curation (the only
 * subset the upload API understands), so it's fully reversible from the gallery.
 */
export function UploadConfirmModal({
  datasetName,
  datasetType,
  triggerWord,
  providerLabel,
  store,
  onCancel,
  onConfirm,
}: {
  datasetName: string
  datasetType: LoraDatasetType
  triggerWord: string | null
  providerLabel: string
  store: StudioStoreApi
  onCancel: () => void
  onConfirm: (rejectIds: string[]) => void
}) {
  // Opt-in: by default not-ready clips are NOT uploaded.
  const [includeNotReady, setIncludeNotReady] = useState(false)
  const [uploading, setUploading] = useState(false)
  const isIc = datasetType === 'ic_lora'
  const isRunPod = providerLabel.toLowerCase().includes('runpod')
  const action = isRunPod ? 'Upload' : 'Stage'
  const actionLower = isRunPod ? 'upload' : 'stage'
  const actionGerund = isRunPod ? 'Uploading' : 'Staging'

  // Snapshot once on open: the gallery is behind the modal, so clips can't
  // change underneath us while it's up.
  const { live, alreadyRejected, candidates, notReady } = useMemo(() => {
    const all = store.getState().clips
    const live = all.filter((c) => !c.deletedAt)
    const alreadyRejected = live.filter((c) => c.triage === 'reject')
    const candidates = live.filter((c) => c.triage !== 'reject')
    const notReady = candidates
      .map((c) => ({
        clip: c,
        // IC-LoRA inputs are conditioning only; captions belong to outputs.
        reasons: notReadyReasons(c, !isIc || isPairTarget(c)),
      }))
      .filter((x) => x.reasons.length > 0)
    return { live, alreadyRejected, candidates, notReady }
  }, [store, isIc])

  const notReadyIds = useMemo(() => new Set(notReady.map((x) => x.clip.id)), [notReady])
  const included = useMemo(
    () => (includeNotReady ? candidates : candidates.filter((c) => !notReadyIds.has(c.id))),
    [includeNotReady, candidates, notReadyIds],
  )

  const onlyCaptionIssues = notReady.every((x) => x.reasons.length === 1 && x.reasons[0] === 'No caption')
  const blocked = included.length < HARD_MIN_CLIPS

  const handleUpload = () => {
    if (blocked || uploading) return
    setUploading(true)
    // Held-back clips are excluded by marking them rejected; including them
    // leaves curation untouched.
    const rejectIds = includeNotReady ? [] : notReady.map((x) => x.clip.id)
    onConfirm(rejectIds)
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={uploading ? undefined : onCancel} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-lg mx-4">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Cloud className="h-4 w-4 text-blue-400" />
            <div>
              <h2 className="text-base font-semibold text-white">
                {isRunPod ? 'Upload to RunPod' : 'Prepare local training workspace'}
              </h2>
              <p className="text-[11px] text-zinc-500">{datasetName}</p>
            </div>
          </div>
          <button
            onClick={onCancel}
            disabled={uploading}
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-40"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4 max-h-[64vh] overflow-y-auto">
          {/* Headline count: exactly what will be sent. */}
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-semibold text-white">{included.length}</span>
              <span className="text-sm text-zinc-400">
                ready clip{included.length === 1 ? '' : 's'} will {actionLower}
              </span>
              {alreadyRejected.length > 0 && (
                <span className="ml-auto text-[11px] text-zinc-500">{alreadyRejected.length} rejected, not sent</span>
              )}
            </div>
            {notReady.length > 0 && !includeNotReady && (
              <p className="text-[11px] text-zinc-500 mt-0.5">
                {notReady.length} not-ready clip{notReady.length === 1 ? '' : 's'} held back (see below)
              </p>
            )}
          </div>

          {/* Readiness checklist for the set that will actually ship. */}
          <div className="bg-zinc-800/40 rounded-lg px-3 py-3">
            <PreflightChecklist
              clips={included.map((c) => ({ caption: c.caption, probe: c.probe }))}
              triggerWord={triggerWord}
            />
          </div>

          {/* Not-ready clips: held back by default, with a clear opt-in. */}
          {notReady.length > 0 && (
            <div
              className={`rounded-lg border overflow-hidden ${
                includeNotReady ? 'border-blue-500/40 bg-blue-500/5' : 'border-amber-500/30 bg-amber-500/5'
              }`}
            >
              <div className="px-3 py-2.5">
                <div className="flex items-center gap-1.5">
                  <AlertTriangle className={`h-3.5 w-3.5 ${includeNotReady ? 'text-blue-300' : 'text-amber-300'}`} />
                  <span className={`text-xs font-medium ${includeNotReady ? 'text-blue-300' : 'text-amber-300'}`}>
                    {notReady.length} clip{notReady.length === 1 ? '' : 's'} not ready
                  </span>
                  <span
                    className={`ml-auto text-[10px] px-1.5 py-0.5 rounded font-medium ${
                      includeNotReady ? 'bg-blue-500/15 text-blue-300' : 'bg-zinc-700/60 text-zinc-300'
                    }`}
                  >
                    {includeNotReady ? `Will ${actionLower}` : `Won't ${actionLower}`}
                  </span>
                </div>
                <p className="text-[11px] text-zinc-400 mt-1">
                  {onlyCaptionIssues
                    ? `${notReady.length === 1 ? 'It has' : 'They have'} no caption.`
                    : `${notReady.length === 1 ? 'It has' : 'They have'} a missing caption or a quality issue.`}{' '}
                  {includeNotReady
                    ? `${actionGerund} them anyway.`
                    : `By default these are left out of this ${isRunPod ? 'upload' : 'preparation'}.`}
                </p>

                <label className="mt-2 flex items-start gap-2 cursor-pointer select-none rounded-md bg-zinc-900/60 px-2.5 py-2 hover:bg-zinc-900">
                  <input
                    type="checkbox"
                    checked={includeNotReady}
                    onChange={(e) => setIncludeNotReady(e.target.checked)}
                    className="mt-0.5 h-3.5 w-3.5 accent-blue-500"
                  />
                  <span className="flex-1">
                    <span className="text-xs text-zinc-200">
                      {action} {notReady.length === 1 ? 'it' : 'these'} anyway
                    </span>
                    {onlyCaptionIssues && !isIc && (
                      <span className="block text-[11px] text-zinc-500">
                        Uncaptioned clips are auto-captioned on the GPU during Prepare.
                      </span>
                    )}
                    {!includeNotReady && (
                      <span className="block text-[11px] text-zinc-500">
                        Left-out clips are marked rejected — reversible anytime from the gallery.
                      </span>
                    )}
                  </span>
                </label>
              </div>

              <div className="border-t border-zinc-800/60 divide-y divide-zinc-800/60 max-h-44 overflow-y-auto">
                {notReady.map(({ clip, reasons }) => (
                  <div key={clip.id} className={`flex items-center gap-2 px-3 py-2 ${includeNotReady ? '' : 'opacity-50'}`}>
                    <div className="min-w-0 flex-1">
                      <p
                        className={`text-[11px] truncate font-mono text-zinc-300 ${includeNotReady ? '' : 'line-through'}`}
                      >
                        {clipName(clip)}
                      </p>
                      <div className="flex flex-wrap gap-1 mt-0.5">
                        {reasons.map((r) => (
                          <span key={r} className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400">
                            {r}
                          </span>
                        ))}
                      </div>
                    </div>
                    {includeNotReady ? (
                      <Check className="h-3.5 w-3.5 text-blue-300 flex-shrink-0" />
                    ) : (
                      <span className="text-[10px] text-zinc-600 flex-shrink-0">excluded</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Things worth knowing before paying for GPU time. */}
          <div className="space-y-1.5 text-[11px] text-zinc-500">
            {blocked && (
              <p className="flex items-start gap-1.5 text-red-400">
                <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0 mt-px" />
                Need at least {HARD_MIN_CLIPS} ready clips to upload.
                {notReady.length > 0 && !includeNotReady ? ' Caption more clips or include the not-ready ones above.' : ''}
              </p>
            )}
            {isIc && (
              <p className="flex items-start gap-1.5">
                <Info className="h-3.5 w-3.5 flex-shrink-0 mt-px text-zinc-400" />
                IC-LoRA: each output's reference clip is {isRunPod ? 'uploaded' : 'staged'} alongside it and paired
                automatically.
              </p>
            )}
            <p className="flex items-start gap-1.5">
              <Info className="h-3.5 w-3.5 flex-shrink-0 mt-px text-zinc-400" />
              Clips and a dataset.json are copied to{' '}
              {isRunPod ? 'the RunPod training workspace' : 'the local training workspace'}. Track progress on the
              collection's status chip in the sidebar.
            </p>
            {isRunPod && (
              <p className="flex items-start gap-1.5 text-amber-300/80">
                <Info className="h-3.5 w-3.5 flex-shrink-0 mt-px" />
                RunPod billing begins when the pod starts and continues until it is stopped or terminated.
              </p>
            )}
          </div>
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex items-center justify-between gap-2">
          <span className="text-[11px] text-zinc-500">
            {live.length} clip{live.length === 1 ? '' : 's'} in collection
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onCancel}
              disabled={uploading}
              className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white disabled:opacity-40"
            >
              Cancel
            </button>
            <button
              onClick={handleUpload}
              disabled={blocked || included.length === 0 || uploading}
              className="text-xs px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
            >
              <Cloud className="h-3.5 w-3.5" />
              {uploading ? `${actionGerund}…` : `${action} ${included.length} clip${included.length === 1 ? '' : 's'}`}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
