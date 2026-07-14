import {
  CLIP_MAX_SECONDS,
  CLIP_MIN_SECONDS,
  RECOMMENDED_MIN_DIM,
  aspectRatioKey,
  clipWarnings,
} from '../../lib/lora-quality'
import { derivePairs, groupMemberIds, pairReadiness } from '../../lib/lora-pairs'
import type { StudioClip } from './studio-store'

export type FacetTone = 'neutral' | 'warn' | 'error'

export interface Facet {
  id: string
  label: string
  tone: FacetTone
  ids: string[]
}

/**
 * Derive the left-rail "smart facets" — saved queries over the clip set the
 * user can click to select+filter a whole category at once (the bulk-prep
 * superpower). Pure: recompute whenever clips change. Only non-empty facets
 * are returned, most-actionable first.
 */
export function computeFacets(clips: StudioClip[]): Facet[] {
  const facets: Facet[] = []
  const collect = (pred: (c: StudioClip) => boolean): string[] =>
    clips.filter(pred).map((c) => c.id)

  const attention = clips
    .filter((c) => clipWarnings({ caption: c.caption, probe: c.probe }).some((w) => w.level === 'error'))
    .map((c) => c.id)
  if (attention.length) facets.push({ id: 'attention', label: 'Needs attention', tone: 'error', ids: attention })

  // Manual curation triage. Surfaced near the top so the user can jump to their
  // picks (or sweep up rejects) before uploading.
  const kept = collect((c) => c.triage === 'keep')
  if (kept.length) facets.push({ id: 'kept', label: 'Kept', tone: 'neutral', ids: kept })
  const rejected = collect((c) => c.triage === 'reject')
  if (rejected.length) facets.push({ id: 'rejected', label: 'Rejected', tone: 'warn', ids: rejected })
  const holdout = collect((c) => c.triage === 'holdout')
  if (holdout.length) facets.push({ id: 'holdout', label: 'Holdout', tone: 'warn', ids: holdout })

  // Pair facets: filter to all pair members, just the not-yet-ready ones, or
  // the clips that aren't in any example yet.
  const { pairs, looseClipIds } = derivePairs(clips)
  if (pairs.length) {
    const memberIds = (groups: typeof pairs): string[] => groups.flatMap(groupMemberIds)
    facets.push({ id: 'pairs', label: `Examples (${pairs.length})`, tone: 'neutral', ids: memberIds(pairs) })
    const incomplete = pairs.filter((g) => pairReadiness(g).tone !== 'ready')
    if (incomplete.length)
      facets.push({ id: 'incomplete-pairs', label: 'Incomplete examples', tone: 'warn', ids: memberIds(incomplete) })
    // Only meaningful once at least one example exists (otherwise everything is
    // ungrouped and the tag is noise).
    const ungrouped = clips.filter((c) => looseClipIds.has(c.id)).map((c) => c.id)
    if (ungrouped.length) facets.push({ id: 'ungrouped', label: 'Ungrouped', tone: 'neutral', ids: ungrouped })
  }

  const uncaptioned = collect((c) => !c.caption.trim())
  if (uncaptioned.length) facets.push({ id: 'uncaptioned', label: 'Uncaptioned', tone: 'warn', ids: uncaptioned })

  const short = collect((c) => (c.probe?.durationSeconds ?? Infinity) < CLIP_MIN_SECONDS)
  if (short.length) facets.push({ id: 'short', label: `Under ${CLIP_MIN_SECONDS}s`, tone: 'warn', ids: short })

  const long = collect((c) => (c.probe?.durationSeconds ?? 0) > CLIP_MAX_SECONDS)
  if (long.length) facets.push({ id: 'long', label: `Over ${CLIP_MAX_SECONDS}s`, tone: 'warn', ids: long })

  const lowRes = collect((c) => {
    if (!c.probe) return false
    const shortEdge = Math.min(c.probe.width, c.probe.height)
    return shortEdge > 0 && shortEdge < RECOMMENDED_MIN_DIM
  })
  if (lowRes.length) facets.push({ id: 'low-res', label: `Below ${RECOMMENDED_MIN_DIM}px`, tone: 'warn', ids: lowRes })

  // Aspect-ratio buckets (only meaningful when the set is mixed).
  const byAspect = new Map<string, string[]>()
  for (const c of clips) {
    if (!c.probe) continue
    const key = aspectRatioKey(c.probe.width, c.probe.height)
    const list = byAspect.get(key) ?? []
    list.push(c.id)
    byAspect.set(key, list)
  }
  if (byAspect.size > 1) {
    for (const [key, ids] of [...byAspect.entries()].sort((a, b) => b[1].length - a[1].length)) {
      facets.push({ id: `aspect:${key}`, label: key, tone: 'neutral', ids })
    }
  }

  return facets
}
