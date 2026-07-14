import { useState } from 'react'
import {
  ArrowLeftRight,
  Crop,
  FolderPlus,
  Link2,
  Loader2,
  Maximize2,
  MessageSquare,
  Scissors,
  Sparkles,
  SlidersHorizontal,
  FlaskConical,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Unlink,
  Wand2,
  X,
} from 'lucide-react'

const CROP_ASPECTS: Array<{ label: string; ratio: [number, number] }> = [
  { label: '16:9', ratio: [16, 9] },
  { label: '9:16', ratio: [9, 16] },
  { label: '1:1', ratio: [1, 1] },
]

const NORMALIZE_OPTIONS: Array<[string, number | null]> = [
  ['Snap size + 24fps', 24],
  ['Snap size + 25fps', 25],
  ['Snap size + 30fps', 30],
  ['Snap size only', null],
]

const btn =
  'text-xs px-2.5 py-1.5 rounded-md text-zinc-200 hover:bg-zinc-700/70 disabled:opacity-40 flex items-center gap-1.5'
const menu =
  'absolute bottom-full mb-2 left-0 min-w-48 max-h-[60vh] overflow-y-auto rounded-md border border-zinc-700 bg-zinc-900 shadow-xl py-1'
const menuItem = 'w-full text-left text-xs px-3 py-1.5 text-zinc-200 hover:bg-zinc-800 flex items-center gap-2'
const menuHeader = 'px-3 pt-2 pb-1 text-[10px] uppercase tracking-wide text-zinc-500'

type Menu = 'caption' | 'edit' | 'examples'

// Floating, centered bulk-action bar shown while 2+ clips are selected. Common
// actions stay inline; related transforms (Crop/Trim/Normalize/Split) collapse
// under "Edit", and grouping (Group/Reverse/Ungroup) under "Examples", so the
// bar stays compact. The destructive Trash is icon-only and visually separated.
export function BulkActionsPill({
  selectedCount,
  editable,
  isIcLora,
  busyAction,
  onCaption,
  onCaptionTools,
  onFrameEdit,
  onGenerate,
  onCrop,
  onTrim,
  onNormalize,
  onGroup,
  onUngroup,
  onReverseRoles,
  hasPaired,
  canReverse,
  canSplit,
  onSceneSplit,
  onSegment,
  onKeep,
  onReject,
  onHoldout,
  onNewCollection,
  onRemove,
  onClear,
}: {
  selectedCount: number
  editable: boolean
  isIcLora: boolean
  busyAction: boolean
  onCaption: () => void
  onCaptionTools: () => void
  onFrameEdit: () => void
  onGenerate: () => void
  onCrop: (ratio: [number, number]) => void
  onTrim: () => void
  onNormalize: (fps: number | null) => void
  onGroup: () => void
  onUngroup: () => void
  onReverseRoles: () => void
  hasPaired: boolean
  /** Whether any selected example has both sides resolved (can be reversed). */
  canReverse: boolean
  /** Whether any selected clip is long enough that splitting it does something. */
  canSplit: boolean
  onSceneSplit: () => void
  onSegment: (seconds: number) => void
  onKeep: () => void
  onReject: () => void
  onHoldout: () => void
  onNewCollection: () => void
  onRemove: () => void
  onClear: () => void
}) {
  const [open, setOpen] = useState<Menu | null>(null)
  const toggle = (m: Menu) => setOpen((v) => (v === m ? null : m))
  const close = () => setOpen(null)

  // The grouping menu only earns a slot when at least one action applies.
  const showExamples = isIcLora && (selectedCount > 1 || hasPaired)

  return (
    <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-20">
      {open && <div className="fixed inset-0 z-0" onClick={close} />}
      <div className="relative z-10 flex items-center gap-1 rounded-xl border border-zinc-700 bg-zinc-900/95 backdrop-blur px-2 py-1.5 shadow-2xl">
        <span className="text-[11px] text-zinc-400 px-2 whitespace-nowrap">{selectedCount} selected</span>
        <div className="w-px h-5 bg-zinc-700/70" />

        <div className="relative">
          <button onClick={() => toggle('caption')} disabled={busyAction} className={btn} title="Caption actions for the selected clips">
            {busyAction ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <MessageSquare className="h-3.5 w-3.5" />}
            Captions
          </button>
          {open === 'caption' && (
            <div className={menu}>
              <button className={menuItem} onClick={() => { close(); onCaption() }}><Sparkles className="h-3.5 w-3.5" /> Auto-caption (AI)</button>
              <button className={menuItem} onClick={() => { close(); onCaptionTools() }}><MessageSquare className="h-3.5 w-3.5" /> Edit captions…</button>
            </div>
          )}
        </div>

        <button onClick={onNewCollection} className={btn} title="Create a new collection from the selected clips">
          <FolderPlus className="h-3.5 w-3.5" /> New collection
        </button>

        {editable && (
          <>
            <div className="w-px h-5 bg-zinc-700/70" />
            <button onClick={onKeep} disabled={busyAction} className={btn} title="Mark the selected clips as keepers">
              <ThumbsUp className="h-3.5 w-3.5" /> Keep
            </button>
            <button onClick={onReject} disabled={busyAction} className={btn} title="Mark the selected clips as rejected (excluded from training)">
              <ThumbsDown className="h-3.5 w-3.5" /> Reject
            </button>
            <button onClick={onHoldout} disabled={busyAction} className={btn} title="Reserve the selected clips for validation (held out of training)">
              <FlaskConical className="h-3.5 w-3.5" /> Hold out
            </button>
            <div className="w-px h-5 bg-zinc-700/70" />

            {/* Transforms — one menu keeps Crop / Trim / Normalize / Split tidy. */}
            <div className="relative">
              <button onClick={() => toggle('edit')} disabled={busyAction} className={btn} title="Crop, trim, normalize or split the selected clips (non-destructive)">
                <SlidersHorizontal className="h-3.5 w-3.5" /> Edit
              </button>
              {open === 'edit' && (
                <div className={menu}>
                  <div className={menuHeader}>Crop</div>
                  {CROP_ASPECTS.map((a) => (
                    <button key={a.label} className={menuItem} onClick={() => { close(); onCrop(a.ratio) }}>
                      <Crop className="h-3.5 w-3.5" /> {a.label}
                    </button>
                  ))}
                  <div className="my-1 h-px bg-zinc-800" />
                  <button className={menuItem} onClick={() => { close(); onTrim() }}>
                    <Scissors className="h-3.5 w-3.5" /> Trim start/end…
                  </button>
                  <div className="my-1 h-px bg-zinc-800" />
                  <div className={menuHeader}>Normalize</div>
                  {NORMALIZE_OPTIONS.map(([label, fps]) => (
                    <button key={label} className={menuItem} onClick={() => { close(); onNormalize(fps) }}>
                      <Maximize2 className="h-3.5 w-3.5" /> {label}
                    </button>
                  ))}
                  {canSplit && (
                    <>
                      <div className="my-1 h-px bg-zinc-800" />
                      <div className={menuHeader}>Split into clips</div>
                      <button className={menuItem} onClick={() => { close(); onSceneSplit() }}>
                        <Scissors className="h-3.5 w-3.5" /> By scene cuts
                      </button>
                      {([3, 5, 10] as const).map((s) => (
                        <button key={s} className={menuItem} onClick={() => { close(); onSegment(s) }}>
                          <Scissors className="h-3.5 w-3.5" /> Into {s}s segments
                        </button>
                      ))}
                    </>
                  )}
                </div>
              )}
            </div>

            <button onClick={onFrameEdit} disabled={busyAction} className={btn} title="Apply the same AI frame edit (FLUX.2 Klein or Nano Banana) to the first frame of every selected clip">
              <Wand2 className="h-3.5 w-3.5" /> Frame edit
            </button>

            <button onClick={onGenerate} disabled={busyAction} className={btn} title={isIcLora ? 'Generate a training example from each selected clip (runs in the background)' : 'Generate a variant for each selected clip (runs in the background)'}>
              <Sparkles className="h-3.5 w-3.5" /> {isIcLora ? 'Generate examples' : 'Generate variants'}
            </button>

            {/* Example grouping — one menu for Group / Reverse / Ungroup. */}
            {showExamples && (
              <div className="relative">
                <button onClick={() => toggle('examples')} disabled={busyAction} className={btn} title="Group these clips into a training example, or edit existing examples">
                  <Link2 className="h-3.5 w-3.5" /> Examples
                </button>
                {open === 'examples' && (
                  <div className={menu}>
                    {selectedCount > 1 && (
                      <button className={menuItem} onClick={() => { close(); onGroup() }} title="Link these clips into one training example: pick the output, the rest become inputs">
                        <Link2 className="h-3.5 w-3.5" /> Group into example
                      </button>
                    )}
                    {hasPaired && canReverse && (
                      <button className={menuItem} onClick={() => { close(); onReverseRoles() }} title="Swap inputs ↔ outputs in the selected example(s)">
                        <ArrowLeftRight className="h-3.5 w-3.5" /> Reverse input ↔ output
                      </button>
                    )}
                    {hasPaired && (
                      <button className={menuItem} onClick={() => { close(); onUngroup() }} title="Dissolve the selected training example(s) back into loose clips">
                        <Unlink className="h-3.5 w-3.5" /> Ungroup
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        )}

        <div className="w-px h-5 bg-zinc-700/70" />
        {editable && (
          <button onClick={onRemove} disabled={busyAction} title="Move the selected clips to the recycle bin" className="h-7 w-7 flex items-center justify-center rounded-md text-red-300 hover:bg-red-500/15 disabled:opacity-40">
            <Trash2 className="h-4 w-4" />
          </button>
        )}
        <button onClick={onClear} className="h-7 w-7 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800" title="Clear selection">
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}
