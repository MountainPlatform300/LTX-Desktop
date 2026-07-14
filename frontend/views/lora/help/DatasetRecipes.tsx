import {
  BookOpen,
  Clapperboard,
  FolderPlus,
  Lightbulb,
  Palette,
  Sparkles,
  User,
  Wand2,
  type LucideIcon,
} from 'lucide-react'
import { Dialog } from '../../../components/ui/dialog'

export const DATASET_RECIPES_VERSION = 2

type Recipe = {
  id: string
  icon: LucideIcon
  gradient: string
  title: string
  goal: string
  steps: string[]
  meta: string
}

export const DATASET_RECIPES: Recipe[] = [
  {
    id: 'style',
    icon: Palette,
    gradient: 'from-sky-500/25 to-blue-500/25 border-sky-500/30 text-sky-300',
    title: 'Style LoRA',
    goal: 'Teach the model a consistent look or aesthetic.',
    steps: [
      'Create a Standard collection with 15–40 clean clips that share the look.',
      'Vary subjects, settings, angles, and motion so the style is the common signal.',
      'Caption what is visible, including the look; do not put the collection trigger in captions.',
    ],
    meta: 'Diverse scenes · short clips · consistent visual treatment',
  },
  {
    id: 'character',
    icon: User,
    gradient: 'from-blue-500/25 to-blue-500/25 border-blue-500/30 text-blue-300',
    title: 'Character / subject LoRA',
    goal: 'Lock in one person, character, or object.',
    steps: [
      'Create a Standard collection with 10–30 clear views of the same subject.',
      'Vary pose, angle, lighting, action, and background; remove ambiguous examples.',
      'Describe each shot naturally. Store the optional trigger once on the collection.',
    ],
    meta: 'Consistency > quantity · vary pose/angle/background',
  },
  {
    id: 'edit',
    icon: Wand2,
    gradient: 'from-amber-500/25 to-orange-500/25 border-amber-500/30 text-amber-300',
    title: 'Edit / IC-LoRA (add or remove things)',
    goal: 'Teach a transformation: input → output.',
    steps: [
      'Create an IC-LoRA collection and build aligned reference input(s) → target output examples.',
      'Repeat one clear transformation across varied subjects, scenes, and camera motion.',
      'Caption every target output before upload; references do not need captions.',
      'Review paused edits from the collection before their video-generation step continues.',
    ],
    meta: 'Aligned pairs · one repeatable edit · target captions only',
  },
  {
    id: 'motion',
    icon: Clapperboard,
    gradient: 'from-emerald-500/25 to-teal-500/25 border-emerald-500/30 text-emerald-300',
    title: 'Motion / restyle',
    goal: 'Capture a movement, camera move, or re-render a look.',
    steps: [
      'Use a Standard collection of short clips centered on the motion or camera move.',
      'Keep unrelated appearance changes varied so motion remains the shared signal.',
      'Caption the visible action and camera movement clearly, without adding the trigger.',
    ],
    meta: 'Keep clips short and motion-focused',
  },
]

export const BEST_PRACTICES = [
  'Quality beats quantity: remove blurry, watermarked, duplicated, or off-concept media.',
  'Captions describe target media only. Prepare injects the normalized collection trigger exactly once.',
  'Standard can auto-caption missing targets during Prepare; IC-LoRA targets must be captioned before upload.',
  'Use the readiness indicator and review the exact upload list before starting paid GPU work.',
  'Keep Training profile on Auto unless you understand the override: 32 GB local is experimental, 48 GB+ is the safer cloud baseline, and 80 GB+ is the standard tier.',
  'Local GPU requires CUDA through WSL2. RunPod is billed while active; use Compute controls to stop or terminate unused pods.',
  'Completed adapters appear in LoRA Library. Use an exact verified trigger; the app does not infer one from a LoRA name or filename.',
]

// Accessible, responsive recipe dialog for common dataset-prep goals, plus a
// short best-practices list. Opened from Help, empty states, and the tour.
export function DatasetRecipes({
  open,
  onClose,
  onNewCollection,
  onStartTour,
}: {
  open: boolean
  onClose: () => void
  onNewCollection: () => void
  onStartTour: () => void
}) {
  if (!open) return null
  return (
    <Dialog
      title="Dataset recipes"
      onClose={onClose}
      className="max-w-[520px]"
      footer={
        <>
          <button
            type="button"
            onClick={onStartTour}
            className="text-xs px-3 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600 flex items-center justify-center gap-1.5"
          >
            <Sparkles className="h-3.5 w-3.5" /> Take the tour
          </button>
          <button
            type="button"
            onClick={onNewCollection}
            className="text-xs px-3 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white flex items-center justify-center gap-1.5"
          >
            <FolderPlus className="h-3.5 w-3.5" /> New collection
          </button>
        </>
      }
    >
      <div className="flex items-center gap-2.5">
        <div className="h-8 w-8 shrink-0 rounded-lg bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
          <BookOpen className="h-4 w-4 text-blue-300" />
        </div>
        <div>
          <p className="text-xs text-zinc-300">Pick the goal closest to yours.</p>
          <p className="text-[10px] text-zinc-600">Guidance v{DATASET_RECIPES_VERSION}</p>
        </div>
      </div>

      <div className="space-y-3">
        {DATASET_RECIPES.map((r) => {
          const Icon = r.icon
          return (
            <div key={r.id} className="rounded-xl border border-zinc-800 bg-zinc-950/40 p-4 hover:border-zinc-700 transition-colors">
              <div className="flex items-center gap-3">
                <div className={`h-9 w-9 shrink-0 rounded-lg bg-gradient-to-br border flex items-center justify-center ${r.gradient}`}>
                  <Icon className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                  <h3 className="text-sm font-semibold text-white">{r.title}</h3>
                  <p className="text-[11px] text-zinc-400">{r.goal}</p>
                </div>
              </div>
              <ul className="mt-3 space-y-1.5">
                {r.steps.map((s, i) => (
                  <li key={i} className="flex gap-2 text-[11px] text-zinc-300">
                    <span className="mt-0.5 h-4 w-4 shrink-0 rounded-full bg-zinc-800 text-zinc-400 text-[9px] font-semibold flex items-center justify-center">
                      {i + 1}
                    </span>
                    <span className="leading-relaxed">{s}</span>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-[10px] text-zinc-600 border-t border-zinc-800/80 pt-2">{r.meta}</p>
            </div>
          )
        })}

        <div className="rounded-xl border border-amber-500/20 bg-amber-500/[0.04] p-4">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="h-4 w-4 text-amber-300" />
            <h3 className="text-sm font-semibold text-amber-100">Best practices</h3>
          </div>
          <ul className="space-y-1.5">
            {BEST_PRACTICES.map((p, i) => (
              <li key={i} className="flex gap-2 text-[11px] text-zinc-300">
                <span className="text-amber-400/70">•</span>
                <span className="leading-relaxed">{p}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Dialog>
  )
}
