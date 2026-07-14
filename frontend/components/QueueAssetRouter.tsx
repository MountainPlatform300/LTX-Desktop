import { useEffect, useRef } from 'react'
import { useQueue, type QueueItem } from '../contexts/QueueContext'
import { useProjects } from '../contexts/ProjectContext'
import { addVisualAssetToProject, addGenericAssetToProject } from '../lib/asset-copy'
import { logger } from '../lib/logger'

/**
 * Global watcher that copies completed queue items into their originating
 * project. Renders nothing — it just lives at the App level inside the
 * QueueProvider so it's mounted regardless of which view is active.
 *
 * Why this instead of inline logic in GenSpace:
 *  - Overnight queues complete while GenSpace may not be mounted; the
 *    asset still has to land in the right project.
 *  - With multi-project routing (each item carries its originating project
 *    id), the routing is naturally a global concern, not GenSpace's.
 *  - Single source of truth — there's no other watcher racing this one
 *    to copy the same file twice.
 *
 * Dedup strategy:
 *
 * The router only routes an item that it watched transition INTO
 * `completed` during this browser session. Items that the router sees
 * for the first time already in a terminal status (completed / failed /
 * cancelled) are assumed to have been routed in a previous session and
 * are silently marked as already-routed.
 *
 * Implementation:
 *  - `seenStatusRef` maps id -> last status the router observed for
 *    that item. Updated on every state change.
 *  - On first sight of an item (id not in the map), if the status is
 *    terminal we mark it as already-routed without copying. Otherwise
 *    we just remember its status; the next time it shows up completed
 *    we route.
 *  - On status change to completed (id was previously pending /
 *    running), we route once and mark as routed. Subsequent polls
 *    see the id already in `routedRef` and skip.
 *  - On routing failure (file copy or addAsset throw), the id is
 *    removed from `routedRef` so the next poll tick retries.
 *
 * Image note: the queue records a single `outputPath` per item. For
 * image requests with `numImages > 1` the backend stores the first
 * image path, so only that one is routed into the project today —
 * matching the single-output breadcrumb contract. A future enhancement
 * could carry the full image-path list on the item.
 */
export function QueueAssetRouter() {
  const { state } = useQueue()
  const { addAsset } = useProjects()
  const routedRef = useRef<Set<string>>(new Set())
  const seenStatusRef = useRef<Map<string, QueueItem['status']>>(new Map())

  useEffect(() => {
    for (const item of state.items) {
      const previousStatus = seenStatusRef.current.get(item.id)
      seenStatusRef.current.set(item.id, item.status)

      // First sight — decide whether to seed as already-routed (item
      // came in already terminal, must be from a prior session) or to
      // just record-and-watch (we'll route on the next status change).
      if (previousStatus === undefined) {
        if (
          item.status === 'completed' ||
          item.status === 'failed' ||
          item.status === 'cancelled'
        ) {
          routedRef.current.add(item.id)
        }
        continue
      }

      // Subsequent sights — only route on the running -> completed
      // transition. Failed / cancelled transitions don't produce an
      // asset, so nothing to route.
      if (previousStatus === 'completed' || item.status !== 'completed') continue
      if (!item.outputPath) continue
      if (!item.originatingProjectId) continue
      if (routedRef.current.has(item.id)) continue

      routedRef.current.add(item.id)
      void routeOne(item, addAsset, routedRef.current)
    }

    // Clean up records for items that have been removed from the
    // queue (e.g., user clicked "Clear completed"). Keeping stale
    // entries forever wouldn't break correctness but the maps grow
    // unboundedly across long sessions.
    const liveIds = new Set(state.items.map((i) => i.id))
    for (const id of seenStatusRef.current.keys()) {
      if (!liveIds.has(id)) {
        seenStatusRef.current.delete(id)
        routedRef.current.delete(id)
      }
    }
  }, [state.items, addAsset])

  return null
}

async function routeOne(
  item: QueueItem,
  addAsset: ReturnType<typeof useProjects>['addAsset'],
  routedSet: Set<string>,
): Promise<void> {
  const outputPath = item.outputPath
  const projectId = item.originatingProjectId
  if (!outputPath || !projectId) return

  const type = item.payload.kind === 'image' || item.payload.kind === 'image_edit' ? 'image' : 'video'
  try {
    const copied = await addVisualAssetToProject(outputPath, projectId, type)
    if (!copied) {
      throw new Error(`Could not persist generated ${type} to project storage`)
    }

    if (item.payload.kind === 'lora') {
      const req = item.payload.request
      // The LoRA generate request is a discriminated union (by `variant`);
      // pull the prompt + adapter id off whichever member fired so the asset
      // card shows what produced it.
      const variant = req.variant
      const loraId = req.loraId
      const isIcLora = variant === 'union_control' || variant === 'video_input_ic_lora'
      const prompt = variant === 'video_input_ic_lora' ? req.prompt : req.request.prompt

      // IC-LoRA: resolution/duration aren't on the request the way they are for
      // a standard video gen. Resolution comes from the actual output pixels
      // (copied.width/height); duration comes from the chosen output length
      // (payload `duration`, defaulting to 5s). Standard LoRA reuses the video
      // request's resolution/duration directly.
      const resolution = isIcLora ? `${copied.width}x${copied.height}` : req.request.resolution
      const duration = variant === 'video_input_ic_lora'
        ? (req.duration ?? 5)
        : variant === 'union_control'
          ? (req.request.duration ?? 5)
          : req.request.duration

      // Reference video (the "before" in a before/after view) comes straight
      // off the payload — the user-supplied reference path.
      const referenceVideoPath = variant === 'video_input_ic_lora'
        ? req.videoPath
        : variant === 'union_control'
          ? req.request.video_path
          : undefined

      // Union control produces a control video (canny/depth/pose) that the
      // backend copies to <output>_control.mp4 beside the generated output.
      // Copy it into project storage so the result viewer can surface it.
      let controlVideoPath: string | undefined
      if (variant === 'union_control' && outputPath) {
        const controlSrc = outputPath.replace(/\.mp4$/i, '_control.mp4')
        const copiedControl = await addGenericAssetToProject(controlSrc, projectId)
        if (copiedControl) controlVideoPath = copiedControl.path
      }

      // Whether the user asked to carry the reference's audio into the output.
      // The backend muxes it in; we just record it on the asset so the viewer
      // and any future audio-aware UI know the result has sound.
      const preserveAudio = variant === 'video_input_ic_lora'
        ? req.preserveAudio
        : variant === 'union_control'
          ? req.request.preserve_audio
          : false

      addAsset(projectId, {
        type: 'video',
        path: copied.path,
        bigThumbnailPath: copied.bigThumbnailPath,
        smallThumbnailPath: copied.smallThumbnailPath,
        width: copied.width,
        height: copied.height,
        prompt,
        resolution,
        duration,
        generationParams: {
          mode: 'lora',
          prompt,
          model: 'fast',
          duration,
          resolution,
          fps: 24,
          audio: preserveAudio,
          cameraMotion: 'none',
          loraId,
          loraVariant: variant,
          referenceVideoPath,
          controlVideoPath,
          icLoraConditioningType: variant === 'union_control'
            ? req.request.conditioning_type
            : undefined,
        },
        takes: [
          {
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          },
        ],
        activeTakeIndex: 0,
      })
    } else if (item.payload.kind === 'video') {
      const req = item.payload.request
      // Mode precedence mirrors GenSpace's pre-queue logic so assets
      // generated via the queue are indistinguishable from the legacy
      // synchronous path: audio takes priority, then image, then t2v.
      const mode = req.audioPath
        ? 'audio-to-video'
        : req.imagePath
          ? 'image-to-video'
          : 'text-to-video'

      addAsset(projectId, {
        type: 'video',
        path: copied.path,
        bigThumbnailPath: copied.bigThumbnailPath,
        smallThumbnailPath: copied.smallThumbnailPath,
        width: copied.width,
        height: copied.height,
        prompt: req.prompt,
        resolution: req.resolution,
        duration: req.duration,
        generationParams: {
          mode,
          prompt: req.prompt,
          model: req.model,
          duration: req.duration,
          resolution: req.resolution,
          fps: req.fps,
          audio: req.audio || false,
          cameraMotion: 'none',
          imageAspectRatio: req.aspectRatio || '16:9',
          imageSteps: 4,
          inputImageUrl: req.imagePath ?? undefined,
          inputAudioUrl: req.audioPath ?? undefined,
        },
        takes: [
          {
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          },
        ],
        activeTakeIndex: 0,
      })
    } else {
      const req = item.payload.request
      addAsset(projectId, {
        type: 'image',
        path: copied.path,
        bigThumbnailPath: copied.bigThumbnailPath,
        smallThumbnailPath: copied.smallThumbnailPath,
        width: copied.width,
        height: copied.height,
        prompt: req.prompt,
        resolution: `${req.width}x${req.height}`,
        generationParams: {
          mode: 'text-to-image',
          prompt: req.prompt,
          model: 'fast',
          duration: 5,
          resolution: `${req.width}x${req.height}`,
          fps: 24,
          audio: false,
          cameraMotion: 'none',
          imageSteps: req.numSteps,
        },
        takes: [
          {
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          },
        ],
        activeTakeIndex: 0,
      })
    }
  } catch (err) {
    routedSet.delete(item.id)
    logger.error(
      `Queue: failed to route item ${item.id} to project ${projectId}: ${
        err instanceof Error ? err.message : String(err)
      }`,
    )
  }
}
