import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from 'react'
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  Cloud,
  FolderPlus,
  Import,
  Library,
  ListTodo,
  MessageSquare,
  Rocket,
  Wand2,
  X,
} from 'lucide-react'

export const TOUR_VERSION = 2
export const TOUR_DONE_KEY = `lora.tour.v${TOUR_VERSION}`
export const LEGACY_TOUR_DONE_KEYS = ['lora.tour.v1'] as const

export type TourDatasetType = 'standard' | 'ic_lora'

export type TourStep = {
  id: string
  // data-tour anchor to spotlight; omit for a centered concept card.
  selector?: string
  icon: typeof Wand2
  title: string
  body: string
}

function captionStep(datasetType: TourDatasetType | null): TourStep {
  if (datasetType === 'standard') {
    return {
      id: 'captions',
      selector: 'captions',
      icon: MessageSquare,
      title: '4 · Caption Standard clips',
      body: 'Describe only what appears in each target clip: subject, action, camera, and look. Auto-caption can help fill gaps. Never type the trigger into captions—Prepare injects the collection trigger exactly once.',
    }
  }
  if (datasetType === 'ic_lora') {
    return {
      id: 'captions',
      selector: 'captions',
      icon: MessageSquare,
      title: '4 · Caption IC-LoRA outputs',
      body: 'Caption every target/output before upload and describe the edited result only. Reference inputs do not need captions, and remote auto-caption is unavailable. Never type the trigger into captions—Prepare injects it once.',
    }
  }
  return {
    id: 'captions',
    selector: 'captions',
    icon: MessageSquare,
    title: '4 · Use the right caption contract',
    body: 'Standard captions describe each target clip. IC-LoRA captions describe target outputs only; references need no caption and targets must be complete before upload. Never add triggers to captions—Prepare injects the collection trigger once.',
  }
}

export function getTourSteps(datasetType: TourDatasetType | null): TourStep[] {
  return [
    {
      id: 'welcome',
      icon: Wand2,
      title: 'Welcome to LoRA Studio',
      body: 'Build a focused dataset, train on your Local GPU or RunPod, then use the finished adapter in Gen Space. This tour covers the current workflow in about a minute.',
    },
    {
      id: 'provider',
      selector: 'provider',
      icon: Cloud,
      title: '1 · Choose where training runs',
      body: 'Local GPU uses CUDA through WSL2 with no cloud charge; RunPod rents the GPU you select. Keep Training profile on Auto (recommended): sub-80 GB GPUs use conservative settings, 32 GB local is experimental, and 80 GB+ is the standard tier.',
    },
    {
      id: 'collection',
      selector: 'new-collection',
      icon: FolderPlus,
      title: '2 · Create the right collection',
      body: 'Choose Standard for a style, subject, or motion learned from individual clips. Choose IC-LoRA for an aligned input → output transformation. Record one optional trigger on the collection; do not copy it into captions.',
    },
    {
      id: 'media',
      selector: 'import',
      icon: Import,
      title: '3 · Add focused media',
      body: 'Import short, varied, high-quality clips or images. Remove blurry, watermarked, or off-concept examples, then use the readiness indicator to catch missing captions, invalid media, and incomplete IC-LoRA pairs.',
    },
    captionStep(datasetType),
    {
      id: 'train',
      selector: 'readiness',
      icon: Rocket,
      title: '5 · Check readiness, then train',
      body: 'Resolve the readiness warnings, then use the adjacent primary action to follow the collection lifecycle on the selected provider. Review the upload list and resolved hardware profile before paying for GPU time; recovery stays with the collection and run.',
    },
    {
      id: 'queue',
      selector: 'lora-queue',
      icon: ListTodo,
      title: '6 · Watch generated examples',
      body: 'Generated IC-LoRA examples and variants appear in Queue → LoRA. “Awaiting review” needs Review edits in its collection. “Cancel all LoRA jobs” is global across collections; training status remains in LoRA Studio.',
    },
    {
      id: 'library',
      selector: 'lora-library',
      icon: Library,
      title: '7 · Use the finished LoRA',
      body: 'Completed adapters appear in LoRA Library. Verify the exact recorded trigger—the app never guesses one from a name or filename—then choose Try in Gen Space. Apply LoRA uses that adapter and adds its verified trigger automatically.',
    },
  ]
}

export function shouldAutoStartTour(
  storage: Pick<Storage, 'getItem'>,
  hasWorkspaceContent: boolean,
): boolean {
  if (!hasWorkspaceContent) return false
  try {
    return storage.getItem(TOUR_DONE_KEY) !== '1'
  } catch {
    return false
  }
}

export function markTourComplete(storage: Pick<Storage, 'setItem' | 'removeItem'>): void {
  try {
    storage.setItem(TOUR_DONE_KEY, '1')
    for (const key of LEGACY_TOUR_DONE_KEYS) storage.removeItem(key)
  } catch {
    // Tour completion is best-effort when storage is unavailable.
  }
}

const FOCUSABLE_SELECTOR = [
  'button:not([disabled])',
  '[href]',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

// A lightweight coachmark tour: dims the screen, spotlights real UI elements by
// their `data-tour` anchor, and shows a positioned card. Steps without an anchor
// (or whose anchor isn't on screen yet) render as a centered concept card, so
// the tour still reads well on a brand-new, empty workspace.
export function GuidedTour({
  open,
  onClose,
  onOpenRecipes,
  datasetType = null,
}: {
  open: boolean
  onClose: () => void
  onOpenRecipes: () => void
  datasetType?: TourDatasetType | null
}) {
  const [index, setIndex] = useState(0)
  const [rect, setRect] = useState<DOMRect | null>(null)
  const cardRef = useRef<HTMLDivElement>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)
  const steps = useMemo(() => getTourSteps(datasetType), [datasetType])

  useEffect(() => {
    if (open) setIndex(0)
  }, [open])

  const step = steps[index] ?? steps[0]

  useEffect(() => {
    if (!open) return
    previousFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    cardRef.current?.querySelector<HTMLElement>(FOCUSABLE_SELECTOR)?.focus()
    return () => {
      if (previousFocusRef.current?.isConnected) previousFocusRef.current.focus()
    }
  }, [open])

  useLayoutEffect(() => {
    if (!open) return
    const update = () => {
      if (!step.selector) {
        setRect(null)
        return
      }
      const el = document.querySelector(`[data-tour="${step.selector}"]`)
      setRect(el ? el.getBoundingClientRect() : null)
    }
    update()
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    const poll = window.setInterval(update, 400)
    return () => {
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
      window.clearInterval(poll)
    }
  }, [open, step])

  if (!open) return null

  const finish = () => {
    markTourComplete(window.localStorage)
    onClose()
  }
  const next = () => (index < steps.length - 1 ? setIndex((i) => i + 1) : finish())
  const back = () => setIndex((i) => Math.max(0, i - 1))
  const isLast = index === steps.length - 1
  const Icon = step.icon

  const cardWidth = Math.min(360, Math.max(280, window.innerWidth - 24))
  const cardPos: CSSProperties = rect && window.innerWidth >= 640
    ? rect.bottom < window.innerHeight * 0.6
      ? { top: rect.bottom + 14, left: clampLeft(rect.left, cardWidth) }
      : { top: rect.top - 14, left: clampLeft(rect.left, cardWidth), transform: 'translateY(-100%)' }
    : { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    event.stopPropagation()
    if (event.key === 'Escape') {
      event.preventDefault()
      finish()
      return
    }
    if (event.key !== 'Tab') return
    const focusable = Array.from(
      cardRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR) ?? [],
    )
    if (focusable.length === 0) {
      event.preventDefault()
      cardRef.current?.focus()
      return
    }
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  return (
    <div className="fixed inset-0 z-[200]" onKeyDown={handleKeyDown}>
      {rect ? (
        <div
          aria-hidden="true"
          className="pointer-events-none fixed rounded-xl ring-2 ring-blue-400/80 transition-all duration-200"
          style={{
            top: rect.top - 6,
            left: rect.left - 6,
            width: rect.width + 12,
            height: rect.height + 12,
            boxShadow: '0 0 0 9999px rgba(9, 9, 11, 0.74)',
          }}
        />
      ) : (
        <div aria-hidden="true" className="fixed inset-0 bg-zinc-950/74 backdrop-blur-[1px]" />
      )}

      <div
        ref={cardRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="lora-tour-title"
        aria-describedby="lora-tour-body"
        tabIndex={-1}
        className="fixed max-h-[calc(100dvh-1.5rem)] w-[calc(100vw-1.5rem)] max-w-[360px] overflow-y-auto rounded-2xl border border-zinc-700 bg-zinc-900 shadow-2xl shadow-black/60"
        style={cardPos}
      >
        <div className="h-1 bg-gradient-to-r from-blue-500 via-blue-500 to-blue-500" />
        <div className="p-5">
          <div className="flex items-start gap-3">
            <div className="h-9 w-9 shrink-0 rounded-lg bg-gradient-to-br from-blue-500/20 to-blue-500/20 border border-blue-500/30 flex items-center justify-center">
              <Icon className="h-4 w-4 text-blue-300" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 id="lora-tour-title" className="text-sm font-semibold text-white">{step.title}</h3>
              <p className="mt-0.5 text-[10px] text-zinc-500">
                Step {index + 1} of {steps.length}
              </p>
            </div>
            <button
              type="button"
              onClick={finish}
              aria-label="Skip guided tour"
              className="h-7 w-7 -mt-1 -mr-1 flex items-center justify-center rounded-md text-zinc-500 hover:text-white hover:bg-zinc-800"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <p id="lora-tour-body" className="text-xs leading-relaxed text-zinc-400 mt-3">{step.body}</p>

          <div className="flex items-center justify-between mt-5">
            <div className="flex items-center gap-1.5">
              {steps.map((s, i) => (
                <span
                  key={s.id}
                  aria-hidden="true"
                  className={`h-1.5 rounded-full transition-all ${
                    i === index ? 'w-4 bg-blue-400' : 'w-1.5 bg-zinc-700'
                  }`}
                />
              ))}
            </div>
            <div className="flex items-center gap-1.5">
              {index > 0 && (
                <button
                  type="button"
                  onClick={back}
                  className="text-xs px-2.5 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white flex items-center gap-1"
                >
                  <ArrowLeft className="h-3.5 w-3.5" /> Back
                </button>
              )}
              <button
                type="button"
                onClick={next}
                className="text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white flex items-center gap-1.5"
              >
                {isLast ? (
                  <>
                    <Check className="h-3.5 w-3.5" /> Done
                  </>
                ) : (
                  <>
                    Next <ArrowRight className="h-3.5 w-3.5" />
                  </>
                )}
              </button>
            </div>
          </div>

          {isLast && (
            <button
              type="button"
              onClick={() => {
                finish()
                onOpenRecipes()
              }}
              className="mt-3 w-full text-xs px-3 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600 flex items-center justify-center gap-1.5"
            >
              <BookOpen className="h-3.5 w-3.5" /> Browse dataset recipes
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function clampLeft(left: number, width: number): number {
  return Math.min(Math.max(12, left), Math.max(12, window.innerWidth - width - 12))
}
