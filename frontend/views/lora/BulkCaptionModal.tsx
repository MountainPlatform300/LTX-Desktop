import { useMemo, useState } from 'react'
import { Check, X } from 'lucide-react'
import type { StudioClip } from '../studio/studio-store'

type Op = 'prefix' | 'suffix' | 'replace' | 'set'

const OPS: Array<{ id: Op; label: string }> = [
  { id: 'prefix', label: 'Add prefix' },
  { id: 'suffix', label: 'Add suffix' },
  { id: 'replace', label: 'Find & replace' },
  { id: 'set', label: 'Set caption (overwrite)' },
]

// Compute the new caption for one clip under the chosen op. Returns null when
// the op leaves the caption unchanged (so we skip no-op writes).
function nextCaption(
  current: string,
  op: Op,
  opts: { text: string; find: string },
): string | null {
  const text = opts.text.trim()
  switch (op) {
    case 'prefix':
      if (!text) return null
      return current ? `${text} ${current}` : text
    case 'suffix':
      if (!text) return null
      return current ? `${current} ${text}` : text
    case 'replace': {
      if (!opts.find) return null
      const out = current.split(opts.find).join(text)
      return out === current ? null : out
    }
    case 'set':
      return text === current ? null : text
  }
}

// Bulk caption text editor: apply the same transform (prefix/suffix/find &
// replace/set) to every selected clip in one go. No backend —
// captions are local until persisted.
export function BulkCaptionModal({
  clips,
  isIcLora = false,
  onClose,
  onApply,
}: {
  clips: StudioClip[]
  isIcLora?: boolean
  onClose: () => void
  onApply: (updates: Array<{ id: string; caption: string }>) => void
}) {
  const [op, setOp] = useState<Op>('prefix')
  const [text, setText] = useState('')
  const [find, setFind] = useState('')

  const updates = useMemo(() => {
    const out: Array<{ id: string; caption: string }> = []
    for (const c of clips) {
      const next = nextCaption(c.caption, op, { text, find })
      if (next !== null) out.push({ id: c.id, caption: next })
    }
    return out
  }, [clips, op, text, find])

  const preview = useMemo(() => {
    const sample = clips[0]
    if (!sample) return null
    const next = nextCaption(sample.caption, op, { text, find })
    return { before: sample.caption || '(empty)', after: (next ?? sample.caption) || '(unchanged)' }
  }, [clips, op, text, find])

  const needsText = op === 'prefix' || op === 'suffix' || op === 'set'
  const canApply = updates.length > 0

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-lg mx-4">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="text-base font-semibold text-white">Edit captions · {clips.length} clips</h2>
          <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4">
          <div className="space-y-1.5">
            <label className="text-[11px] text-zinc-400">Operation</label>
            <div className="grid grid-cols-2 gap-1.5">
              {OPS.map((o) => (
                  <button
                    key={o.id}
                    onClick={() => setOp(o.id)}
                    className={`text-xs px-3 py-2 rounded-md border text-left transition-colors ${
                      op === o.id ? 'bg-blue-500/15 border-blue-500/40 text-white' : 'border-zinc-800 text-zinc-300 hover:border-zinc-700'
                    }`}
                  >
                    {o.label}
                  </button>
              ))}
            </div>
            <p className="text-[11px] text-amber-400/80">
              Don’t add the trigger word here — preprocessing adds it automatically.
              {isIcLora ? ' IC-LoRA captions should describe the edited result.' : ''}
            </p>
          </div>

          {op === 'replace' && (
            <div className="space-y-1">
              <label className="text-[11px] text-zinc-400">Find</label>
              <input
                value={find}
                onChange={(e) => setFind(e.target.value)}
                placeholder="text to find"
                className="w-full text-sm bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500/50"
              />
            </div>
          )}

          {(needsText || op === 'replace') && (
            <div className="space-y-1">
              <label className="text-[11px] text-zinc-400">
                {op === 'replace' ? 'Replace with' : op === 'set' ? 'Caption' : 'Text'}
              </label>
              <input
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={op === 'replace' ? 'replacement (blank to remove)' : op === 'set' ? 'new caption for all selected' : 'text to add'}
                className="w-full text-sm bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-blue-500/50"
              />
            </div>
          )}

          {preview && (
            <div className="space-y-1 text-[11px]">
              <div className="text-zinc-500">Preview (first clip)</div>
              <div className="rounded-md bg-zinc-950 border border-zinc-800 px-3 py-2 space-y-1">
                <div className="text-zinc-500 truncate">- {preview.before}</div>
                <div className="text-emerald-300 truncate">+ {preview.after}</div>
              </div>
            </div>
          )}

          <div className="text-[11px] text-zinc-500">{updates.length} of {clips.length} clips will change.</div>
        </div>

        <div className="px-5 py-4 border-t border-zinc-800 flex items-center justify-end gap-2">
          <button onClick={onClose} className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white">Cancel</button>
          <button
            onClick={() => { onApply(updates); onClose() }}
            disabled={!canApply}
            className="text-xs px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
          >
            <Check className="h-3.5 w-3.5" /> Apply to {updates.length}
          </button>
        </div>
      </div>
    </div>
  )
}
