import { useEffect, useMemo, useState } from 'react'
import { Check, Cloud, FileText, Film, FolderOpen, Loader2, X } from 'lucide-react'
import {
  useLoraTraining,
  type LoraTrainingJob,
  type PublicationMeta,
  type PublishPlatform,
} from '../../contexts/LoraTrainingContext'
import { useToast } from '../../contexts/ToastContext'
import { isImagePath, pathToFileUrl } from '../../lib/file-url'

type Step = 'platforms' | 'details' | 'examples' | 'review'
const STEPS: { id: Step; label: string }[] = [
  { id: 'platforms', label: 'Platforms' },
  { id: 'details', label: 'Details' },
  { id: 'examples', label: 'Examples' },
  { id: 'review', label: 'Review' },
]

const PLATFORM_LABEL: Record<PublishPlatform, string> = {
  huggingface: 'Hugging Face',
  civitai: 'Civitai',
  portable: 'Portable card',
}
const PLATFORM_HINT: Record<PublishPlatform, string> = {
  huggingface: 'README.md with model-card front-matter',
  civitai: 'Description + structured metadata',
  portable: 'Plain Markdown card (works anywhere)',
}
const ALL_PLATFORMS: PublishPlatform[] = ['huggingface', 'civitai', 'portable']
const LICENSES = ['other', 'apache-2.0', 'mit', 'cc-by-4.0', 'cc-by-nc-4.0', 'openrail', 'creativeml-openrail-m']

/**
 * Turns a finished run into a shareable model card + asset bundle. The example
 * gallery is sourced from the LoRA's own dataset clips — "how it was trained".
 * (A separate results gallery, where the user uploads videos of the LoRA's
 * output, is a planned follow-up.)
 */
export function PublishWizard({ job, onClose }: { job: LoraTrainingJob; onClose: () => void }) {
  const { datasets, preprocessed, publishPreview, publishExport } = useLoraTraining()
  const { addToast } = useToast()

  const dataset = useMemo(() => {
    const pre = preprocessed.find((p) => p.id === job.preprocessedId)
    return pre ? datasets.find((d) => d.id === pre.datasetId) ?? null : null
  }, [datasets, preprocessed, job.preprocessedId])

  // Showcase candidates: live (non-recycled) clips from the source dataset.
  const clips = useMemo(() => (dataset?.clips ?? []).filter((c) => !c.deletedAt), [dataset])

  const [step, setStep] = useState<Step>('platforms')
  const [platforms, setPlatforms] = useState<Set<PublishPlatform>>(new Set(['huggingface', 'portable']))
  const [meta, setMeta] = useState<PublicationMeta | null>(null)
  const [tagsText, setTagsText] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [cards, setCards] = useState<Record<string, string>>({})
  const [activeCard, setActiveCard] = useState<PublishPlatform | null>(null)
  const [previewing, setPreviewing] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch the server's suggested fields once so Details opens pre-filled.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      const res = await publishPreview(job.id, { platforms: ['portable'], examples: [] })
      if (cancelled) return
      if (res.ok) {
        setMeta(res.data.meta)
        setTagsText((res.data.meta.tags ?? []).join(', '))
      } else {
        setError(res.error)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [job.id, publishPreview])

  const platformList = useMemo(() => ALL_PLATFORMS.filter((p) => platforms.has(p)), [platforms])

  const examples = useMemo(
    () =>
      clips
        .filter((c) => selected.has(c.id))
        .map((c) => ({ mediaPath: c.localPath, caption: c.caption ?? '' })),
    [clips, selected],
  )

  const buildMeta = (): PublicationMeta | null => {
    if (!meta) return null
    return {
      ...meta,
      tags: tagsText
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean),
    }
  }

  const togglePlatform = (p: PublishPlatform) => {
    setPlatforms((prev) => {
      const next = new Set(prev)
      if (next.has(p)) next.delete(p)
      else next.add(p)
      return next
    })
  }

  const toggleClip = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const refreshPreview = async () => {
    const m = buildMeta()
    if (!m) return
    setPreviewing(true)
    setError(null)
    const res = await publishPreview(job.id, { platforms: platformList, meta: m, examples })
    setPreviewing(false)
    if (!res.ok) {
      setError(res.error)
      return
    }
    setCards(res.data.cards)
    setActiveCard((prev) => (prev && res.data.cards[prev] ? prev : platformList[0] ?? null))
  }

  const goNext = () => {
    const idx = STEPS.findIndex((s) => s.id === step)
    const next = STEPS[idx + 1]
    if (!next) return
    setStep(next.id)
    if (next.id === 'review') void refreshPreview()
  }
  const goBack = () => {
    const idx = STEPS.findIndex((s) => s.id === step)
    const prev = STEPS[idx - 1]
    if (prev) setStep(prev.id)
  }

  const handleExport = async () => {
    const m = buildMeta()
    if (!m) return
    const destPath = (await window.electronAPI?.showOpenDirectoryDialog?.({
      title: 'Choose a folder to write the publication into',
    })) ?? null
    if (!destPath) return
    setExporting(true)
    setError(null)
    const res = await publishExport(job.id, { destPath, platforms: platformList, meta: m, examples })
    setExporting(false)
    if (!res.ok) {
      setError(res.error)
      return
    }
    addToast({
      title: 'Publication ready',
      description: `${res.data.files.length} file(s) → ${res.data.publicationPath}`,
      variant: 'success',
      actionLabel: 'Reveal',
      onAction: () => window.electronAPI?.showItemInFolder?.({ filePath: res.data.publicationPath }),
    })
    onClose()
  }

  const canProceed =
    (step === 'platforms' && platformList.length > 0) ||
    (step === 'details' && !!buildMeta()?.title.trim()) ||
    step === 'examples'

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={exporting ? undefined : onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-2xl mx-4 flex flex-col max-h-[85vh]">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Cloud className="h-4 w-4 text-blue-400" />
            <div>
              <h2 className="text-base font-semibold text-white">Publish LoRA</h2>
              <p className="text-[11px] text-zinc-500">{job.name}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={exporting}
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-40"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-2 px-5 py-3 border-b border-zinc-800">
          {STEPS.map((s, i) => {
            const activeIdx = STEPS.findIndex((x) => x.id === step)
            const done = i < activeIdx
            const current = s.id === step
            return (
              <div key={s.id} className="flex items-center gap-2">
                <span
                  className={`flex items-center gap-1.5 text-[11px] ${
                    current ? 'text-white font-medium' : done ? 'text-blue-400' : 'text-zinc-500'
                  }`}
                >
                  <span
                    className={`h-4 w-4 rounded-full flex items-center justify-center text-[9px] ${
                      current ? 'bg-blue-600 text-white' : done ? 'bg-blue-500/20 text-blue-300' : 'bg-zinc-800 text-zinc-500'
                    }`}
                  >
                    {done ? <Check className="h-2.5 w-2.5" /> : i + 1}
                  </span>
                  {s.label}
                </span>
                {i < STEPS.length - 1 && <span className="text-zinc-700">›</span>}
              </div>
            )
          })}
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {step === 'platforms' && (
            <div className="space-y-2">
              <p className="text-xs text-zinc-400">Choose where you'll publish. We tailor the card to each.</p>
              {ALL_PLATFORMS.map((p) => (
                <label
                  key={p}
                  className="flex items-start gap-2.5 rounded-lg border border-zinc-800 bg-zinc-800/40 px-3 py-2.5 cursor-pointer hover:border-zinc-700"
                >
                  <input
                    type="checkbox"
                    checked={platforms.has(p)}
                    onChange={() => togglePlatform(p)}
                    className="mt-0.5 h-3.5 w-3.5 accent-blue-500"
                  />
                  <span>
                    <span className="block text-sm text-zinc-200">{PLATFORM_LABEL[p]}</span>
                    <span className="block text-[11px] text-zinc-500">{PLATFORM_HINT[p]}</span>
                  </span>
                </label>
              ))}
            </div>
          )}

          {step === 'details' &&
            (meta ? (
              <div className="space-y-3">
                <Field label="Title">
                  <input
                    value={meta.title}
                    onChange={(e) => setMeta({ ...meta, title: e.target.value })}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-2.5 py-1.5 text-sm text-zinc-100"
                  />
                </Field>
                <Field label="Summary">
                  <input
                    value={meta.summary}
                    onChange={(e) => setMeta({ ...meta, summary: e.target.value })}
                    placeholder="One-line description"
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-2.5 py-1.5 text-sm text-zinc-100"
                  />
                </Field>
                <Field label="Description">
                  <textarea
                    value={meta.description}
                    onChange={(e) => setMeta({ ...meta, description: e.target.value })}
                    rows={4}
                    placeholder="What does this LoRA do? When should people use it?"
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-2.5 py-1.5 text-sm text-zinc-100 resize-none"
                  />
                </Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Author">
                    <input
                      value={meta.author}
                      onChange={(e) => setMeta({ ...meta, author: e.target.value })}
                      placeholder="Your name / handle"
                      className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-2.5 py-1.5 text-sm text-zinc-100"
                    />
                  </Field>
                  <Field label="License">
                    <select
                      value={meta.license}
                      onChange={(e) => setMeta({ ...meta, license: e.target.value })}
                      className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-2.5 py-1.5 text-sm text-zinc-100"
                    >
                      {LICENSES.map((l) => (
                        <option key={l} value={l}>
                          {l}
                        </option>
                      ))}
                    </select>
                  </Field>
                </div>
                <Field label="Tags (comma-separated)">
                  <input
                    value={tagsText}
                    onChange={(e) => setTagsText(e.target.value)}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-2.5 py-1.5 text-sm text-zinc-100"
                  />
                </Field>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-xs text-zinc-500 py-8 justify-center">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading suggested details…
              </div>
            ))}

          {step === 'examples' && (
            <div className="space-y-3">
              <div>
                <p className="text-sm text-zinc-200">Training samples</p>
                <p className="text-[11px] text-zinc-500">
                  Pick clips from the dataset to show how it was trained. Results videos (the LoRA's output) can be
                  added later.
                </p>
              </div>
              {clips.length === 0 ? (
                <p className="text-xs text-zinc-600 border border-dashed border-zinc-700 rounded-lg py-6 text-center">
                  The source dataset isn't available, so there are no clips to showcase. You can still publish the
                  card without examples.
                </p>
              ) : (
                <div className="grid grid-cols-3 gap-2">
                  {clips.map((c) => {
                    const isSel = selected.has(c.id)
                    const poster = c.posterPath ?? (isImagePath(c.localPath) ? c.localPath : null)
                    return (
                      <button
                        key={c.id}
                        onClick={() => toggleClip(c.id)}
                        className={`relative aspect-video rounded-lg overflow-hidden border-2 group ${
                          isSel ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                        }`}
                      >
                        {poster ? (
                          <img src={pathToFileUrl(poster)} alt="" className="h-full w-full object-cover" />
                        ) : (
                          <div className="h-full w-full bg-zinc-800 flex items-center justify-center">
                            <Film className="h-5 w-5 text-zinc-600" />
                          </div>
                        )}
                        {isSel && (
                          <span className="absolute top-1 right-1 h-4 w-4 rounded-full bg-blue-600 flex items-center justify-center">
                            <Check className="h-2.5 w-2.5 text-white" />
                          </span>
                        )}
                        {c.caption?.trim() && (
                          <span className="absolute inset-x-0 bottom-0 bg-black/60 px-1.5 py-0.5 text-[9px] text-zinc-200 truncate text-left">
                            {c.caption}
                          </span>
                        )}
                      </button>
                    )
                  })}
                </div>
              )}
              {selected.size > 0 && (
                <p className="text-[11px] text-zinc-400">{selected.size} clip(s) selected for the gallery.</p>
              )}
            </div>
          )}

          {step === 'review' && (
            <div className="space-y-3">
              {previewing ? (
                <div className="flex items-center gap-2 text-xs text-zinc-500 py-8 justify-center">
                  <Loader2 className="h-4 w-4 animate-spin" /> Rendering preview…
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {platformList.map((p) => (
                      <button
                        key={p}
                        onClick={() => setActiveCard(p)}
                        className={`text-[11px] px-2 py-1 rounded-md flex items-center gap-1 ${
                          activeCard === p ? 'bg-blue-600 text-white' : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                        }`}
                      >
                        <FileText className="h-3 w-3" /> {PLATFORM_LABEL[p]}
                      </button>
                    ))}
                  </div>
                  <pre className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 text-[11px] text-zinc-300 whitespace-pre-wrap break-words max-h-[40vh] overflow-y-auto font-mono">
                    {activeCard ? cards[activeCard] ?? '' : ''}
                  </pre>
                  <p className="text-[11px] text-zinc-500">
                    Exporting writes the card(s){examples.length > 0 ? `, ${examples.length} example file(s),` : ''}{' '}
                    {job.localLoraPath ? 'and the .safetensors weights' : '(weights not found locally)'} to a folder
                    you choose.
                  </p>
                </>
              )}
            </div>
          )}

          {error && <p className="mt-3 text-xs text-red-400">{error}</p>}
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex items-center justify-between gap-2">
          <button
            onClick={step === 'platforms' ? onClose : goBack}
            disabled={exporting}
            className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white disabled:opacity-40"
          >
            {step === 'platforms' ? 'Cancel' : 'Back'}
          </button>
          {step === 'review' ? (
            <button
              onClick={() => void handleExport()}
              disabled={exporting || previewing || platformList.length === 0}
              className="text-xs px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 flex items-center gap-1.5"
            >
              {exporting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FolderOpen className="h-3.5 w-3.5" />}
              {exporting ? 'Exporting…' : 'Export to folder'}
            </button>
          ) : (
            <button
              onClick={goNext}
              disabled={!canProceed}
              className="text-xs px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40"
            >
              Next
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="block text-[11px] font-medium text-zinc-400 mb-1">{label}</span>
      {children}
    </label>
  )
}
