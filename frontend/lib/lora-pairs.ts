// Pure pairing derivation for the LoRA gallery.
//
// An IC-LoRA "set" is a before→after relationship: one or more *target*
// (edited / output) clips conditioned on one-or-more *reference* (control /
// input) clips. A target stores its references as paths — `referencePath`
// (the primary, from auto Make-pair) and/or `referencePaths` (the full list,
// from manual grouping). We never persist the reverse link, so groups are
// *derived* here at render time. A control clip is matched by its `sourcePath`
// or `localPath`, so we match a target's reference paths against either field
// of every clip. Targets sharing the exact same reference set are grouped
// together; a reference path with no matching clip is a "missing" control.

import { clipWarnings } from './lora-quality'
import type { StudioClip } from '../views/studio/studio-store'

export interface PairGroup {
  /** Stable id for the group (first resolved control's id, else first target's
   *  id) so React keys + selection stay stable across renders. */
  id: string
  /** Resolved reference/before clips (may be fewer than `referencePaths` when
   *  some referenced clips aren't in the dataset). */
  controls: StudioClip[]
  /** Every path the targets reference, including ones whose clip is missing,
   *  so the UI can show what's absent. */
  referencePaths: string[]
  /** One or more edited/after clips that share this reference set. */
  targets: StudioClip[]
}

export interface PairsResult {
  pairs: PairGroup[]
  /** Ids of clips that are NOT part of any pair (render as normal cards). */
  looseClipIds: Set<string>
}

export type ReadinessTone = 'ready' | 'warn' | 'error'

export interface PairReadiness {
  tone: ReadinessTone
  reasons: string[]
}

/** The effective reference paths for a clip: the full manual list when set,
 *  else the single primary reference, else none. */
export function effectiveReferences(clip: StudioClip): string[] {
  if (clip.referencePaths.length > 0) return clip.referencePaths
  return clip.referencePath ? [clip.referencePath] : []
}

/** A clip is a pair *target* when it references at least one other clip. */
export function isPairTarget(clip: StudioClip): boolean {
  return effectiveReferences(clip).length > 0
}

function controlMatches(clip: StudioClip, referencePath: string): boolean {
  return clip.localPath === referencePath || clip.sourcePath === referencePath
}

/**
 * Group clips into before→after sets derived from each target's references.
 * Targets that share the exact same reference set are grouped together; each
 * reference path is resolved to its control clip (may be missing).
 */
export function derivePairs(clips: StudioClip[]): PairsResult {
  const targets = clips.filter(isPairTarget)
  const looseClipIds = new Set(clips.map((c) => c.id))

  // Group targets by their (order-independent) set of reference paths.
  const byKey = new Map<string, { paths: string[]; targets: StudioClip[] }>()
  for (const t of targets) {
    const paths = effectiveReferences(t)
    const key = [...paths].sort().join('\u0000')
    const entry = byKey.get(key) ?? { paths, targets: [] }
    entry.targets.push(t)
    byKey.set(key, entry)
  }

  const pairs: PairGroup[] = []
  for (const { paths, targets: refTargets } of byKey.values()) {
    const controls = paths
      .map((p) => clips.find((c) => controlMatches(c, p)) ?? null)
      .filter((c): c is StudioClip => c != null)
    pairs.push({
      id: controls[0]?.id ?? refTargets[0].id,
      controls,
      referencePaths: paths,
      targets: refTargets,
    })
    for (const t of refTargets) looseClipIds.delete(t.id)
    for (const c of controls) looseClipIds.delete(c.id)
  }

  return { pairs, looseClipIds }
}

/** Every clip id that belongs to a group (controls + targets). */
export function groupMemberIds(group: PairGroup): string[] {
  return [...group.controls.map((c) => c.id), ...group.targets.map((t) => t.id)]
}

/**
 * Training-readiness of a set. Ready requires every referenced control to be
 * present and every target captioned, with no error-level quality warnings on
 * any member. A missing reference is an error — nothing to anchor against.
 */
export function pairReadiness(group: PairGroup): PairReadiness {
  const reasons: string[] = []
  let tone: ReadinessTone = 'ready'
  const bump = (next: ReadinessTone) => {
    if (next === 'error') tone = 'error'
    else if (next === 'warn' && tone !== 'error') tone = 'warn'
  }

  const missing = group.referencePaths.length - group.controls.length
  if (group.controls.length === 0) {
    return { tone: 'error', reasons: ['Input clip is missing from this dataset'] }
  }
  if (missing > 0) {
    bump('error')
    reasons.push(`${missing} input clip(s) missing from this dataset`)
  }
  // The released trainer conditions on a single reference. Legacy examples with
  // more than one input will have their extras dropped at export, so warn.
  if (group.controls.length > 1) {
    bump('warn')
    reasons.push('Multiple inputs — the trainer uses one; extras are dropped at export')
  }

  const members = [...group.controls, ...group.targets]
  for (const member of members) {
    if (clipWarnings({ caption: member.caption, probe: member.probe }).some((w) => w.level === 'error')) {
      bump('error')
      reasons.push('A clip has a quality error')
      break
    }
  }
  // Targets carry the edit's caption; require each to be captioned.
  if (group.targets.some((t) => !t.caption.trim())) {
    bump('warn')
    reasons.push('Edited clip needs a caption')
  }

  if (tone === 'ready') {
    const refLabel = group.controls.length === 1 ? 'input' : `${group.controls.length} inputs`
    reasons.push(`Output + ${refLabel} ready`)
  }
  return { tone, reasons }
}

/** Count how many derived pairs are fully ready. */
export function countReadyPairs(pairs: PairGroup[]): number {
  return pairs.filter((p) => pairReadiness(p).tone === 'ready').length
}
