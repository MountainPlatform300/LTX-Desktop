import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertCircle, Check, Clock, ExternalLink, Image as ImageIcon, KeyRound, Loader2, Search, Video, X } from 'lucide-react'
import {
  useLoraTraining,
  type ClipInput,
  type PexelsMediaItem,
} from '../../contexts/LoraTrainingContext'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { ImportNormalizeOptions, useImportNormalizeSpec } from './ImportNormalizeOptions'
import { importSpecActive, normalizeImportInputs } from '../../lib/lora-import-normalize'

type MediaKind = PexelsMediaItem['kind']
type Orientation = '' | 'landscape' | 'portrait' | 'square'

const PER_PAGE = 24
// Cap concurrent downloads so a big multi-select doesn't hammer the CDN.
const DOWNLOAD_CONCURRENCY = 3

const itemKey = (item: PexelsMediaItem) => `${item.kind}:${item.id}`

// Stock-media browser for the LoRA trainer. Searches Pexels (BYOK) for photos
// and videos and downloads the chosen assets into the open collection via the
// same ClipInput handoff as the file importer. Pexels' API guidelines require
// a visible Pexels link + author credit, both rendered below.
export function PexelsBrowser({
  onClose,
  onAdd,
  normalizeOnAdd = true,
}: {
  onClose: () => void
  onAdd: (clips: ClipInput[]) => void | Promise<void>
  // When false, hand raw downloads back untouched — the host (e.g. the
  // create-dataset wizard) owns normalization so it isn't offered twice.
  normalizeOnAdd?: boolean
}) {
  const { searchPexels, downloadPexels, applyClipEdits } = useLoraTraining()
  const { settings } = useAppSettings()
  const [normSpec, setNormSpec] = useImportNormalizeSpec()
  const [normalizing, setNormalizing] = useState<{ done: number; total: number } | null>(null)

  const [media, setMedia] = useState<MediaKind>('video')
  const [orientation, setOrientation] = useState<Orientation>('')
  const [queryInput, setQueryInput] = useState('')
  const [activeQuery, setActiveQuery] = useState('')
  const [items, setItems] = useState<PexelsMediaItem[]>([])
  const [page, setPage] = useState(1)
  const [hasNext, setHasNext] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<Map<string, PexelsMediaItem>>(new Map())
  const [adding, setAdding] = useState<{ done: number; total: number } | null>(null)
  // Bumped on every new search so an in-flight stale page can't append.
  const searchToken = useRef(0)

  const runSearch = useCallback(
    async (opts: { query: string; media: MediaKind; orientation: Orientation; page: number; append: boolean }) => {
      if (!settings.hasPexelsApiKey) return
      const token = ++searchToken.current
      setLoading(true)
      setError(null)
      const res = await searchPexels({
        query: opts.query,
        media: opts.media,
        page: opts.page,
        perPage: PER_PAGE,
        orientation: opts.orientation,
      })
      if (token !== searchToken.current) return
      setLoading(false)
      if (!res.ok) {
        setError(res.error)
        if (!opts.append) setItems([])
        return
      }
      setItems((prev) => (opts.append ? [...prev, ...res.data.items] : res.data.items))
      setPage(res.data.page)
      setHasNext(res.data.hasNext)
    },
    [searchPexels, settings.hasPexelsApiKey],
  )

  // Open with the curated/popular feed so the grid isn't empty.
  useEffect(() => {
    if (settings.hasPexelsApiKey) {
      void runSearch({ query: '', media: 'video', orientation: '', page: 1, append: false })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const submitSearch = () => {
    const q = queryInput.trim()
    setActiveQuery(q)
    void runSearch({ query: q, media, orientation, page: 1, append: false })
  }

  const switchMedia = (next: MediaKind) => {
    if (next === media) return
    setMedia(next)
    void runSearch({ query: activeQuery, media: next, orientation, page: 1, append: false })
  }

  const switchOrientation = (next: Orientation) => {
    setOrientation(next)
    void runSearch({ query: activeQuery, media, orientation: next, page: 1, append: false })
  }

  const loadMore = () => {
    if (loading || !hasNext) return
    void runSearch({ query: activeQuery, media, orientation, page: page + 1, append: true })
  }

  const toggleSelect = (item: PexelsMediaItem) => {
    setSelected((prev) => {
      const next = new Map(prev)
      const key = itemKey(item)
      if (next.has(key)) next.delete(key)
      else next.set(key, item)
      return next
    })
  }

  const addSelected = async () => {
    const chosen = [...selected.values()]
    if (chosen.length === 0) return
    setError(null)
    setAdding({ done: 0, total: chosen.length })
    const clips: ClipInput[] = []
    let done = 0
    let failed = 0

    // Bounded-concurrency download pool.
    const queue = [...chosen]
    const worker = async () => {
      while (queue.length > 0) {
        const item = queue.shift()
        if (!item) break
        const res = await downloadPexels(item)
        done += 1
        setAdding({ done, total: chosen.length })
        if (res.ok) {
          clips.push({
            localPath: res.data.localPath,
            caption: '',
            origin: 'imported',
            probe: res.data.probe ?? undefined,
            durationSeconds: res.data.probe?.durationSeconds ?? undefined,
          })
        } else {
          failed += 1
          setError(res.error)
        }
      }
    }
    await Promise.all(
      Array.from({ length: Math.min(DOWNLOAD_CONCURRENCY, chosen.length) }, () => worker()),
    )
    setAdding(null)
    if (clips.length === 0) return

    let toAdd = clips
    if (normalizeOnAdd && importSpecActive(normSpec)) {
      setNormalizing({ done: 0, total: clips.length })
      const res = await normalizeImportInputs(clips, normSpec, applyClipEdits, {
        onProgress: setNormalizing,
      })
      setNormalizing(null)
      toAdd = res.inputs
      if (res.failures.length > 0) {
        failed += res.failures.length
        setError(`Some clips couldn't be normalized: ${res.failures.join('; ')}`)
      }
    }
    await onAdd(toAdd)
    if (failed === 0) onClose()
  }

  const selectedCount = selected.size

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-5xl mx-4 flex flex-col max-h-[88vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold text-white">Browse Pexels</h2>
            <a
              href="https://www.pexels.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[11px] text-teal-400 hover:text-teal-300 inline-flex items-center gap-1"
            >
              Photos & videos provided by Pexels <ExternalLink className="h-3 w-3" />
            </a>
          </div>
          <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        {!settings.hasPexelsApiKey ? (
          <div className="px-6 py-14 flex flex-col items-center text-center gap-3">
            <KeyRound className="h-7 w-7 text-amber-400" />
            <p className="text-sm text-zinc-300">A Pexels API key is required to browse stock media.</p>
            <p className="text-xs text-zinc-500 max-w-sm">
              Pexels offers a free API key. Add it in Settings, then reopen this browser.
            </p>
            <button
              onClick={() => window.dispatchEvent(new CustomEvent('open-settings', { detail: { tab: 'apiKeys' } }))}
              className="mt-1 text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white"
            >
              Open Settings
            </button>
          </div>
        ) : (
          <>
            {/* Controls */}
            <div className="px-5 py-3 border-b border-zinc-800 space-y-3">
              <div className="flex items-center gap-2">
                <div className="relative flex-1">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500" />
                  <input
                    value={queryInput}
                    onChange={(e) => setQueryInput(e.target.value)}
                    onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Enter') submitSearch() }}
                    placeholder={media === 'video' ? 'Search videos (e.g. ocean waves, city street)…' : 'Search photos (e.g. portrait, forest)…'}
                    className="w-full pl-8 pr-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <button
                  onClick={submitSearch}
                  className="text-xs px-3 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white whitespace-nowrap"
                >
                  Search
                </button>
              </div>
              <div className="flex items-center gap-4 flex-wrap">
                <div className="inline-flex rounded-lg bg-zinc-800 p-0.5">
                  <button
                    onClick={() => switchMedia('video')}
                    className={`text-xs px-3 py-1.5 rounded-md inline-flex items-center gap-1.5 transition-colors ${media === 'video' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}
                  >
                    <Video className="h-3.5 w-3.5" /> Videos
                  </button>
                  <button
                    onClick={() => switchMedia('photo')}
                    className={`text-xs px-3 py-1.5 rounded-md inline-flex items-center gap-1.5 transition-colors ${media === 'photo' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}
                  >
                    <ImageIcon className="h-3.5 w-3.5" /> Photos
                  </button>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-zinc-500">Orientation</span>
                  {(['', 'landscape', 'portrait', 'square'] as const).map((o) => (
                    <button
                      key={o || 'all'}
                      onClick={() => switchOrientation(o)}
                      className={`text-[11px] px-2 py-1 rounded-md capitalize transition-colors ${orientation === o ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200 bg-zinc-800/60'}`}
                    >
                      {o || 'any'}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Results */}
            <div className="flex-1 overflow-y-auto px-5 py-4">
              {error && (
                <div className="mb-3 flex items-start gap-2 px-3 py-2 rounded-lg bg-red-500/10 text-red-300 text-xs">
                  <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {items.length === 0 && loading ? (
                <div className="py-16 flex justify-center text-zinc-500">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : items.length === 0 ? (
                <div className="py-16 text-center text-xs text-zinc-600">No results. Try a different search.</div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                  {items.map((item) => {
                    const key = itemKey(item)
                    const isSelected = selected.has(key)
                    return (
                      <button
                        key={key}
                        onClick={() => toggleSelect(item)}
                        className={`group relative aspect-video rounded-lg overflow-hidden bg-zinc-800 border-2 transition-colors text-left ${isSelected ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'}`}
                      >
                        {item.previewUrl ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={item.previewUrl} alt={item.alt} loading="lazy" className="h-full w-full object-cover" />
                        ) : (
                          <div className="h-full w-full flex items-center justify-center text-zinc-600">
                            {item.kind === 'video' ? <Video className="h-5 w-5" /> : <ImageIcon className="h-5 w-5" />}
                          </div>
                        )}

                        {/* Selection check */}
                        <div className={`absolute top-1.5 left-1.5 h-5 w-5 rounded-full flex items-center justify-center transition-colors ${isSelected ? 'bg-blue-500 text-white' : 'bg-black/50 text-transparent group-hover:text-white/60'}`}>
                          <Check className="h-3 w-3" />
                        </div>

                        {/* Duration badge (videos) */}
                        {item.kind === 'video' && item.durationSeconds ? (
                          <span className="absolute top-1.5 right-1.5 text-[10px] px-1.5 py-0.5 rounded bg-black/60 text-white inline-flex items-center gap-1">
                            <Clock className="h-2.5 w-2.5" /> {Math.round(item.durationSeconds)}s
                          </span>
                        ) : null}

                        {/* Attribution footer */}
                        <div className="absolute inset-x-0 bottom-0 px-2 py-1.5 bg-gradient-to-t from-black/80 to-transparent">
                          <span className="text-[10px] text-zinc-300 truncate block">{item.author || 'Pexels'}</span>
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}

              {hasNext && items.length > 0 && (
                <div className="pt-4 flex justify-center">
                  <button
                    onClick={loadMore}
                    disabled={loading}
                    className="text-xs px-4 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:bg-zinc-800 disabled:opacity-50 inline-flex items-center gap-2"
                  >
                    {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                    Load more
                  </button>
                </div>
              )}
            </div>

            {/* Normalize options */}
            {normalizeOnAdd && (
              <div className="px-5 pt-3 border-t border-zinc-800">
                <ImportNormalizeOptions value={normSpec} onChange={setNormSpec} disabled={normalizing !== null} />
              </div>
            )}

            {/* Footer */}
            <div className="px-5 py-4 border-t border-zinc-800 flex items-center justify-between gap-2">
              <span className="text-xs text-zinc-500">
                {selectedCount > 0 ? `${selectedCount} selected` : 'Select photos or videos to add'}
              </span>
              <div className="flex items-center gap-2">
                <button onClick={onClose} className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white">Cancel</button>
                <button
                  onClick={() => void addSelected()}
                  disabled={selectedCount === 0 || adding !== null || normalizing !== null}
                  className="text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-1.5"
                >
                  {adding || normalizing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                  {adding
                    ? `Downloading ${adding.done}/${adding.total}…`
                    : normalizing
                      ? `Normalizing ${normalizing.done}/${normalizing.total}…`
                      : `Add ${selectedCount > 0 ? selectedCount : ''} to collection`}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
