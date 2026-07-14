import { useEffect } from 'react'
import { useLoraTraining } from '../../contexts/LoraTrainingContext'
import { logger } from '../../lib/logger'
import type { StudioStoreApi } from '../studio/studio-store'

const POLL_INTERVAL_MS = 1500
const MAX_POLLS = 80 // ~2 min ceiling so a stuck job never polls forever

/**
 * Drives the hover-scrub preview pipeline for whichever collection is open:
 * enqueues local sprite/filmstrip jobs for any clip lacking one, polls the
 * durable ledger, and merges results onto the store until every clip has a
 * sprite.
 *
 * Crucially it stays subscribed to the store, so clips that appear *after*
 * mount — trims (which rewrite `localPath` and clear the sprite) and
 * generate-example derivatives — get their sprite enqueued and polled too,
 * instead of only resolving on the next app launch.
 */
export function useSpritePipeline(store: StudioStoreApi | null): void {
  const { enqueueClipJobs, listClipJobs } = useLoraTraining()

  useEffect(() => {
    if (!store) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null
    let polling = false
    // Reset every time we enqueue fresh work so a new trim/derivative gets a
    // full polling window even if an earlier batch had exhausted the budget.
    let pollsSinceEnqueue = 0
    // Paths we've already asked the backend to build, so frequent store
    // updates (selection, sprite merges) don't re-enqueue the same clip.
    const enqueued = new Set<string>()

    const hasMissingSprite = () =>
      store.getState().clips.some((c) => c.spritePath == null)

    const poll = async () => {
      if (cancelled) {
        polling = false
        return
      }
      const result = await listClipJobs()
      if (cancelled) {
        polling = false
        return
      }
      if (result.ok) store.getState().applySpriteResults(result.data)
      pollsSinceEnqueue += 1
      if (hasMissingSprite() && pollsSinceEnqueue < MAX_POLLS) {
        timer = setTimeout(() => void poll(), POLL_INTERVAL_MS)
      } else {
        polling = false
      }
    }

    const ensurePolling = () => {
      // Don't restart once the budget for the current batch is spent — wait for
      // new work (which resets the counter) so a permanently stuck clip can't
      // poll forever on every unrelated store update.
      if (polling || cancelled || pollsSinceEnqueue >= MAX_POLLS) return
      if (!hasMissingSprite()) return
      polling = true
      void poll()
    }

    const sync = async () => {
      if (cancelled) return
      const needed = store
        .getState()
        .clips.filter((c) => c.spritePath == null)
        .map((c) => c.localPath)
        .filter((p) => !enqueued.has(p))
      if (needed.length > 0) {
        // Mark before awaiting so re-entrant store updates don't double-enqueue.
        for (const p of needed) enqueued.add(p)
        pollsSinceEnqueue = 0
        const enq = await enqueueClipJobs(needed)
        if (!enq.ok) {
          logger.warn('LoRA Studio: failed to enqueue sprite jobs')
          // Allow a later store change to retry these paths.
          for (const p of needed) enqueued.delete(p)
        }
      }
      ensurePolling()
    }

    void sync()
    const unsubscribe = store.subscribe(() => void sync())
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
      unsubscribe()
    }
  }, [store, enqueueClipJobs, listClipJobs])
}
