import { ErrorBoundary } from '../components/ErrorBoundary'
import { LoraWorkspace } from './lora/LoraWorkspace'

// The LoRA trainer is mounted as the "LoRA Trainer" tab in both Home (no
// projectId) and Project (scoped + embedded) views. The actual UI lives in
// `LoraWorkspace` — a gallery-first, Adobe-Bridge-style workspace. This thin
// wrapper preserves the export signature so the Home/Project wiring is
// untouched.
export function LoraTrainer({
  projectId = null,
  embedded = false,
}: { projectId?: string | null; embedded?: boolean } = {}) {
  return (
    <ErrorBoundary
      resetKey={`${projectId ?? 'home'}:${embedded ? 'embedded' : 'full'}`}
      title="LoRA Trainer could not be displayed"
    >
      <LoraWorkspace projectId={projectId} embedded={embedded} />
    </ErrorBoundary>
  )
}
