import React, { createContext, useContext, useMemo } from 'react'
import { createStore, type StoreApi } from 'zustand/vanilla'
import { useStoreWithEqualityFn } from 'zustand/traditional'
import type { ClipEdits, ClipInput, ClipJob, LoraDataset, LoraDatasetClip } from '../../contexts/LoraTrainingContext'
import type { ClipProbeLike } from '../../lib/lora-quality'
import { isImagePath } from '../../lib/file-url'

export type ClipKind = 'video' | 'image'

/** Manual curation flag (null = unreviewed). `keep`/`reject` gate training
 *  eligibility; `holdout` reserves the clip for the in-training validation feed
 *  (excluded from training, reference staged for IC-LoRA validation). */
export type ClipTriage = 'keep' | 'reject' | 'holdout'

/**
 * Dataset Studio store (M0 skeleton).
 *
 * Mirrors the video-editor store split (vanilla zustand + context provider +
 * selector hook) so the curation workspace can grow domain selectors/actions
 * the same way. M0 only models the gallery's read surface — clips, their
 * generated preview assets, and selection — with no undo/persistence yet;
 * later milestones add the edit stack, history, and project bridge.
 */

export interface StudioClip {
  id: string
  /** Untouched original; edits always re-render from here. */
  sourcePath: string
  /** Currently rendered file (== sourcePath when pristine). Sprites + training
   *  use this, so preview jobs are keyed by `localPath`, not `sourcePath`. */
  localPath: string
  caption: string
  /** Still image vs. video clip. Inferred from `localPath`'s extension;
   *  drives gallery rendering (no scrub/duration for stills) and which AI
   *  actions apply (animate / motion-lock vs. trim/restyle). */
  kind: ClipKind
  origin: ClipInput['origin']
  /** Primary conditioning reference (the "before" for an auto-derived pair). */
  referencePath: string | null
  /** All references for a manually-grouped IC-LoRA set: one target (this clip)
   *  conditioned on one-or-more reference clips' `localPath`s. Empty for loose
   *  clips and auto-derived pairs (which use `referencePath`). The effective
   *  reference list is `referencePaths` when non-empty, else `[referencePath]`. */
  referencePaths: string[]
  /** For stills: the source video this frame came from, so a later
   *  motion-locked "Make pair" can use it as the driver automatically.
   *  Session-only lineage (not persisted); null when unknown. */
  driverPath: string | null
  /** Applied non-destructive edit stack (null = pristine). */
  edits: ClipEdits | null
  posterPath: string | null
  spritePath: string | null
  spriteTiles: number | null
  /** Cached measurement; null until probed. Shape matches `ClipProbeLike` so
   *  the shared quality heuristics (warnings, aspect, health) apply directly. */
  probe: ClipProbeLike | null
  /** Fallback duration when no probe yet (e.g. from the import metadata). */
  durationSeconds: number | null
  /** Manual keep/reject curation flag (null = unreviewed). Lets the user mark
   *  picks while scrubbing an import, then filter to kept clips before training.
   *  Rejected clips are excluded from readiness counts. */
  triage: ClipTriage | null
  /** Soft-delete timestamp (ISO-8601). When set, the clip lives in the
   *  dataset's recycle bin: hidden from the gallery and excluded from pairing,
   *  readiness, training and export until restored or permanently deleted.
   *  null = live clip. */
  deletedAt: string | null
  /** Session-only Nano Banana preview produced in the Generate-example wizard,
   *  kept so reopening the wizard for this clip restores the edited frame
   *  instead of forcing a (token-costing) regenerate. Keyed to the prompt +
   *  frame it was made for, so a changed prompt still reads as stale. Lives in
   *  memory for the session; not persisted to the dataset. null = none yet. */
  editPreview: EditFramePreview | null
}

/** A committed wizard frame edit, remembered per clip for the session. */
export interface EditFramePreview {
  /** Path to the edited PNG on disk (already persisted by `editFrame`). */
  path: string
  /** The edit prompt that produced it (used to detect a stale preview). */
  prompt: string
  /** The frame timestamp it was extracted from (0 for stills). */
  frameTimeSeconds: number
}

/** Result of an apply-edits render: a fresh derived file + its probe. */
export interface ClipEditResult {
  localPath: string
  probe: ClipProbeLike
  edits: ClipEdits
}

export interface StudioState {
  datasetId: string
  datasetName: string
  triggerWord: string | null
  clips: StudioClip[]
  selectedIds: Set<string>
}

export interface StudioActions {
  setClips: (clips: StudioClip[]) => void
  /** Append newly imported clips (keeps existing selection). */
  addClips: (clips: StudioClip[]) => void
  /** Add a still-image asset (e.g. an extracted/Nano-Banana-edited frame) and
   *  select it for review. `driverPath` records the source video for a later
   *  motion-locked pair. Returns the new clip's id. */
  addImageClip: (opts: { id?: string; localPath: string; caption?: string; driverPath?: string | null; probe?: ClipProbeLike | null }) => string
  addImageClips: (items: Array<{ localPath: string; caption?: string; driverPath?: string | null; probe?: ClipProbeLike | null }>) => void
  /** Permanently drop clips by id (and from the selection). Used for internal
   *  replace (scene-split) and emptying the recycle bin. User-facing "remove"
   *  should call `trashClips` instead so the action is recoverable. */
  removeClips: (ids: string[]) => void
  /** Soft-delete: move clips to the recycle bin (stamps `deletedAt`) and drop
   *  them from the selection. Recoverable via `restoreClips`. */
  trashClips: (ids: string[]) => void
  /** Restore clips from the recycle bin (clears `deletedAt`). */
  restoreClips: (ids: string[]) => void
  /** Set a single clip's caption (used by inline + bulk captioning). */
  setClipCaption: (id: string, caption: string) => void
  /** Set the keep/reject triage flag on one-or-more clips (null = clear). */
  setClipTriage: (ids: string[], triage: ClipTriage | null) => void
  /** Remember (or clear) the Generate-example wizard's committed frame edit for
   *  a clip, so reopening the wizard restores it instead of regenerating.
   *  Session-only. */
  setEditPreview: (clipId: string, preview: EditFramePreview | null) => void
  /** Set a clip's measured probe (after import / revert re-probe). */
  setClipProbe: (id: string, probe: ClipProbeLike) => void
  /** Update the dataset's trigger word (drives the missing-trigger facet). */
  setTriggerWord: (triggerWord: string | null) => void
  toggleSelect: (id: string, additive: boolean) => void
  setSelection: (ids: string[]) => void
  selectAll: () => void
  clearSelection: () => void
  /** Merge completed sprite-job results onto matching clips (by current path). */
  applySpriteResults: (jobs: ClipJob[]) => void
  /** Replace a clip with a freshly rendered derivative (trim/crop). Clears the
   *  stale preview so a new sprite is generated for the derived file. */
  applyEditResult: (clipId: string, result: ClipEditResult) => void
  /** Discard a clip's edits and point it back at the untouched source. */
  resetClipEdits: (clipId: string) => void
  /** Manually group clips into an IC-LoRA example: `targetId` is the output and
   *  `referenceIds` supplies its conditioning input (stored by `localPath`).
   *  The released trainer conditions on a single reference, so only the first
   *  resolvable input is kept. Overwrites the target's existing reference. */
  groupAsPair: (targetId: string, referenceIds: string[]) => void
  /** Clear pairing on the given clips (drops their references → loose again). */
  ungroupClips: (ids: string[]) => void
}

export type StudioStore = StudioState & StudioActions
export type StudioStoreApi = StoreApi<StudioStore>

export function clipFromDataset(clip: LoraDatasetClip): StudioClip {
  const localPath = clip.localPath
  const image = isImagePath(localPath)
  return {
    id: clip.id,
    sourcePath: clip.sourcePath ?? localPath,
    localPath,
    caption: clip.caption ?? '',
    kind: image ? 'image' : 'video',
    origin: clip.origin,
    referencePath: clip.referencePath ?? null,
    referencePaths: clip.referencePaths ?? [],
    driverPath: null,
    edits: clip.edits ?? null,
    // Stills render directly from the file; fall back to it as the poster.
    posterPath: clip.posterPath ?? (image ? localPath : null),
    spritePath: clip.spritePath ?? null,
    spriteTiles: clip.spriteTiles ?? null,
    probe: clip.probe ?? null,
    durationSeconds: clip.probe?.durationSeconds ?? clip.durationSeconds ?? null,
    triage: clip.triage ?? null,
    deletedAt: clip.deletedAt ?? null,
    editPreview: null,
  }
}

/** Build a studio clip from a freshly imported clip-input (no dataset id yet).
 *  Used by the add-clips sources (files, scene-split, Gen Space handoff). */
export function clipFromInput(input: ClipInput): StudioClip {
  const localPath = input.localPath
  const image = isImagePath(localPath)
  return {
    id: input.id ?? crypto.randomUUID(),
    sourcePath: input.sourcePath ?? localPath,
    localPath,
    caption: input.caption ?? '',
    kind: image ? 'image' : 'video',
    origin: input.origin,
    referencePath: input.referencePath ?? null,
    referencePaths: input.referencePaths ?? [],
    driverPath: null,
    edits: input.edits ?? null,
    posterPath: input.posterPath ?? (image ? localPath : null),
    spritePath: input.spritePath ?? null,
    spriteTiles: input.spriteTiles ?? null,
    probe: input.probe ?? null,
    durationSeconds: input.probe?.durationSeconds ?? input.durationSeconds ?? null,
    triage: input.triage ?? null,
    deletedAt: input.deletedAt ?? null,
    editPreview: null,
  }
}

/** Serialize a studio clip back into the dataset's clip-input shape so the
 *  curation (edits, captions, generated previews) persists across sessions. */
export function toClipInput(c: StudioClip): ClipInput {
  return {
    id: c.id,
    localPath: c.localPath,
    caption: c.caption,
    durationSeconds: c.probe?.durationSeconds ?? c.durationSeconds ?? null,
    referencePath: c.referencePath,
    referencePaths: c.referencePaths,
    origin: c.origin,
    probe: c.probe
      ? {
          durationSeconds: c.probe.durationSeconds,
          width: c.probe.width,
          height: c.probe.height,
          fps: c.probe.fps,
          frameCount: c.probe.frameCount,
          hasAudio: c.probe.hasAudio,
          videoCodec: c.probe.videoCodec ?? null,
        }
      : null,
    // Pristine clips store no separate source (localPath is the original).
    sourcePath: c.sourcePath === c.localPath ? null : c.sourcePath,
    edits: c.edits,
    posterPath: c.posterPath,
    spritePath: c.spritePath,
    spriteTiles: c.spriteTiles,
    triage: c.triage,
    deletedAt: c.deletedAt,
  }
}

export function createStudioStore(dataset: LoraDataset): StudioStoreApi {
  return createStore<StudioStore>()((set) => ({
    datasetId: dataset.id,
    datasetName: dataset.name,
    triggerWord: dataset.triggerWord ?? null,
    clips: dataset.clips.map(clipFromDataset),
    selectedIds: new Set<string>(),

    setClips: (clips) => set({ clips }),

    addClips: (incoming) =>
      set((prev) => {
        const existing = new Set(prev.clips.map((c) => c.id))
        const fresh = incoming.filter((c) => !existing.has(c.id))
        return fresh.length ? { clips: [...prev.clips, ...fresh] } : prev
      }),

    addImageClip: ({ id: requestedId, localPath, caption, driverPath, probe }) => {
      const id = requestedId ?? crypto.randomUUID()
      const clip: StudioClip = {
        id,
        sourcePath: localPath,
        localPath,
        caption: caption ?? '',
        kind: 'image',
        origin: 'ai_derived',
        referencePath: null,
        referencePaths: [],
        driverPath: driverPath ?? null,
        edits: null,
        posterPath: localPath,
        spritePath: null,
        spriteTiles: null,
        probe: probe ?? null,
        durationSeconds: probe?.durationSeconds ?? null,
        triage: null,
        deletedAt: null,
        editPreview: null,
      }
      set((prev) => (
        prev.clips.some((existing) => existing.id === id)
          ? prev
          : { clips: [...prev.clips, clip], selectedIds: new Set([id]) }
      ))
      return id
    },

    addImageClips: (items) =>
      set((prev) => {
        const fresh: StudioClip[] = items.map(({ localPath, caption, driverPath, probe }) => ({
          id: crypto.randomUUID(),
          sourcePath: localPath,
          localPath,
          caption: caption ?? '',
          kind: 'image',
          origin: 'ai_derived',
          referencePath: null,
          referencePaths: [],
          driverPath: driverPath ?? null,
          edits: null,
          posterPath: localPath,
          spritePath: null,
          spriteTiles: null,
          probe: probe ?? null,
          durationSeconds: probe?.durationSeconds ?? null,
          triage: null,
          deletedAt: null,
          editPreview: null,
        }))
        if (!fresh.length) return prev
        return { clips: [...prev.clips, ...fresh], selectedIds: new Set(fresh.map((c) => c.id)) }
      }),

    removeClips: (ids) =>
      set((prev) => {
        const drop = new Set(ids)
        const clips = prev.clips.filter((c) => !drop.has(c.id))
        if (clips.length === prev.clips.length) return prev
        const selectedIds = new Set([...prev.selectedIds].filter((id) => !drop.has(id)))
        return { clips, selectedIds }
      }),

    trashClips: (ids) =>
      set((prev) => {
        const target = new Set(ids)
        const deletedAt = new Date().toISOString()
        let changed = false
        const clips = prev.clips.map((c) => {
          if (!target.has(c.id) || c.deletedAt) return c
          changed = true
          return { ...c, deletedAt }
        })
        if (!changed) return prev
        const selectedIds = new Set([...prev.selectedIds].filter((id) => !target.has(id)))
        return { clips, selectedIds }
      }),

    restoreClips: (ids) =>
      set((prev) => {
        const target = new Set(ids)
        let changed = false
        const clips = prev.clips.map((c) => {
          if (!target.has(c.id) || !c.deletedAt) return c
          changed = true
          return { ...c, deletedAt: null }
        })
        return changed ? { clips } : prev
      }),

    setClipCaption: (id, caption) =>
      set((prev) => ({
        clips: prev.clips.map((c) => (c.id === id ? { ...c, caption } : c)),
      })),

    setClipTriage: (ids, triage) =>
      set((prev) => {
        const target = new Set(ids)
        return {
          clips: prev.clips.map((c) => (target.has(c.id) ? { ...c, triage } : c)),
        }
      }),

    setEditPreview: (clipId, preview) =>
      set((prev) => ({
        clips: prev.clips.map((c) => (c.id === clipId ? { ...c, editPreview: preview } : c)),
      })),

    setClipProbe: (id, probe) =>
      set((prev) => ({
        clips: prev.clips.map((c) =>
          c.id === id ? { ...c, probe, durationSeconds: probe.durationSeconds } : c,
        ),
      })),

    setTriggerWord: (triggerWord) => set({ triggerWord }),

    toggleSelect: (id, additive) =>
      set((prev) => {
        const next = new Set(additive ? prev.selectedIds : [])
        if (prev.selectedIds.has(id) && (additive || prev.selectedIds.size === 1)) {
          next.delete(id)
        } else {
          next.add(id)
        }
        return { selectedIds: next }
      }),

    setSelection: (ids) => set({ selectedIds: new Set(ids) }),

    selectAll: () => set((prev) => ({ selectedIds: new Set(prev.clips.map((c) => c.id)) })),

    clearSelection: () => set({ selectedIds: new Set<string>() }),

    applySpriteResults: (jobs) =>
      set((prev) => {
        // Sprites preview the *currently rendered* file, so jobs are keyed by
        // `localPath` (== sourcePath when pristine, the derivative after edits).
        // We apply the poster as soon as it lands (job still `running`) so the
        // card drops its spinner early, then merge the sprite when it finishes.
        const byPath = new Map<string, ClipJob>()
        for (const j of jobs) {
          if (j.status === 'failed' || (!j.posterPath && !j.spritePath)) continue
          const existing = byPath.get(j.sourcePath)
          // A job carrying a finished sprite supersedes a poster-only one.
          if (!existing || (j.spritePath && !existing.spritePath)) byPath.set(j.sourcePath, j)
        }
        if (byPath.size === 0) return prev
        let changed = false
        const clips = prev.clips.map((clip) => {
          const job = byPath.get(clip.localPath)
          if (!job) return clip
          const posterPath = clip.posterPath ?? job.posterPath ?? null
          const spritePath = job.spritePath ?? clip.spritePath
          const spriteTiles = job.spriteTiles ?? clip.spriteTiles
          if (posterPath === clip.posterPath && spritePath === clip.spritePath && spriteTiles === clip.spriteTiles) {
            return clip
          }
          changed = true
          return { ...clip, posterPath, spritePath, spriteTiles }
        })
        return changed ? { clips } : prev
      }),

    applyEditResult: (clipId, result) =>
      set((prev) => {
        let changed = false
        const clips = prev.clips.map((clip) => {
          if (clip.id !== clipId) return clip
          changed = true
          return {
            ...clip,
            localPath: result.localPath,
            edits: result.edits,
            probe: result.probe,
            durationSeconds: result.probe.durationSeconds,
            // The derived file differs from the old preview; drop the stale
            // sprite so the pipeline regenerates one for the new localPath.
            posterPath: null,
            spritePath: null,
            spriteTiles: null,
          }
        })
        return changed ? { clips } : prev
      }),

    resetClipEdits: (clipId) =>
      set((prev) => {
        let changed = false
        const clips = prev.clips.map((clip) => {
          if (clip.id !== clipId || (clip.edits == null && clip.localPath === clip.sourcePath)) return clip
          changed = true
          return {
            ...clip,
            localPath: clip.sourcePath,
            edits: null,
            probe: null,
            durationSeconds: null,
            posterPath: null,
            spritePath: null,
            spriteTiles: null,
          }
        })
        return changed ? { clips } : prev
      }),

    groupAsPair: (targetId, referenceIds) =>
      set((prev) => {
        const byId = new Map(prev.clips.map((c) => [c.id, c] as const))
        const refPaths = referenceIds
          .filter((id) => id !== targetId)
          .map((id) => byId.get(id)?.localPath)
          .filter((p): p is string => !!p)
        if (refPaths.length === 0 || !byId.has(targetId)) return prev
        // The released LTX-2 trainer conditions on a single reference per
        // example, so hard-cap to one input even if more were selected.
        const refPath = refPaths[0]
        const clips = prev.clips.map((c) =>
          c.id === targetId ? { ...c, referencePath: refPath, referencePaths: [refPath] } : c,
        )
        return { clips }
      }),

    ungroupClips: (ids) =>
      set((prev) => {
        const drop = new Set(ids)
        let changed = false
        const clips = prev.clips.map((c) => {
          if (!drop.has(c.id)) return c
          if (c.referencePath == null && c.referencePaths.length === 0) return c
          changed = true
          return { ...c, referencePath: null, referencePaths: [] }
        })
        return changed ? { clips } : prev
      }),
  }))
}

const StudioStoreContext = createContext<StudioStoreApi | null>(null)

export function StudioStoreProvider({
  store,
  children,
}: {
  store: StudioStoreApi
  children: React.ReactNode
}) {
  return <StudioStoreContext.Provider value={store}>{children}</StudioStoreContext.Provider>
}

function useStudioStoreApi(): StudioStoreApi {
  const store = useContext(StudioStoreContext)
  if (!store) throw new Error('useStudioStore must be used within StudioStoreProvider')
  return store
}

export function useStudioStore<T>(
  selector: (state: StudioStore) => T,
  equalityFn?: (a: T, b: T) => boolean,
): T {
  return useStoreWithEqualityFn(useStudioStoreApi(), selector, equalityFn)
}

export function useStudioActions(): StudioActions {
  const store = useStudioStoreApi()
  return useMemo<StudioActions>(() => {
    const {
      setClips,
      addClips,
      addImageClip,
      addImageClips,
      removeClips,
      trashClips,
      restoreClips,
      setClipCaption,
      setClipTriage,
      setEditPreview,
      setClipProbe,
      setTriggerWord,
      toggleSelect,
      setSelection,
      selectAll,
      clearSelection,
      applySpriteResults,
      applyEditResult,
      resetClipEdits,
      groupAsPair,
      ungroupClips,
    } = store.getState()
    return {
      setClips,
      addClips,
      addImageClip,
      addImageClips,
      removeClips,
      trashClips,
      restoreClips,
      setClipCaption,
      setClipTriage,
      setEditPreview,
      setClipProbe,
      setTriggerWord,
      toggleSelect,
      setSelection,
      selectAll,
      clearSelection,
      applySpriteResults,
      applyEditResult,
      resetClipEdits,
      groupAsPair,
      ungroupClips,
    }
  }, [store])
}
