import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
//
// QueueItem and QueueState are re-exported from the generated OpenAPI types
// so renames or new fields on the backend show up as TS errors in the panel
// at compile time.

export type QueueState = ApiSuccessOf<'getQueueState'>
export type QueueItem = QueueState['items'][number]
export type QueueItemStatus = QueueItem['status']
export type QueueItemSource = QueueItem['source']
// Discriminated (by `kind`) video/image payload — the request snapshot the
// runner dispatches on. We intentionally derive it from the generated
// payload type inside QueueItem so the context stays self-contained and
// stays in sync with the backend union.
export type QueuePayload = QueueItem['payload']
export type VideoQueuePayload = Extract<QueuePayload, { kind: 'video' }>
export type ImageQueuePayload = Extract<QueuePayload, { kind: 'image' }>

// Live progress for the currently-running item (from /api/generation/progress).
// `progress` is 0-100; `phase` is a free-form string the backend uses as a
// debug breadcrumb ("loading_model" / "encoding_text" / "inference" / etc.);
// `currentStep` and `totalSteps` are populated during the inference phase
// so the UI can render "step 4/8" alongside the percentage. Null when no
// generation is running.
export interface RunningGenerationProgress {
  progress: number
  phase: string
  currentStep: number | null
  totalSteps: number | null
}

interface QueueContextValue {
  state: QueueState
  loading: boolean
  // Live progress of the running item, or null if nothing is running.
  // Refreshed by the same polling loop that drives queue state, but only
  // when the queue actually has a running item — no point hitting the
  // progress endpoint when we know it'll return idle.
  runningProgress: RunningGenerationProgress | null
  // Counts for the header badge — pending + running. These are the items
  // the user actually cares about ("how much work is in flight"); completed
  // and failed accumulate but don't drive the badge.
  activeCount: number
  pendingCount: number
  runningCount: number
  // Increase the active polling cadence while the panel is open so reorder /
  // status updates feel snappy. While closed, the cadence drops to keep the
  // backend cost ~zero for users who never look at the queue.
  setIsPanelOpen: (open: boolean) => void
  isPanelOpen: boolean
  // Force an immediate refetch — used after explicit mutations so the UI
  // doesn't lag a poll tick behind the action.
  refresh: () => Promise<void>
  // Mutations. Each one refreshes state on success so callers don't have to.
  enqueue: (payload: QueuePayload, opts?: EnqueueOptions) => Promise<{ ok: true; item: QueueItem } | { ok: false; error: string }>
  enqueueBatch: (items: { payload: QueuePayload; opts?: EnqueueOptions }[]) => Promise<{ ok: true; items: QueueItem[] } | { ok: false; error: string }>
  cancelPending: (itemId: string) => Promise<void>
  // Stop a *running* generation. Pending items should use `removeItem`
  // (gone, no breadcrumb); only items the runner has already claimed
  // should hit cancelRunning, which routes through /api/generate/cancel
  // so the inference loop can unwind cleanly. The runner observes the
  // cancellation and moves the item to `cancelled` for history.
  cancelRunning: () => Promise<void>
  removeItem: (itemId: string) => Promise<void>
  // Replace a pending item's payload (typo fix, parameter change). Only
  // valid while the item is pending — running and terminal items return
  // 409 from the backend; UI hides the affordance.
  updatePending: (itemId: string, payload: QueuePayload) => Promise<{ ok: true } | { ok: false; error: string }>
  // Re-enqueue a completed / cancelled / failed item's payload as a
  // fresh pending item, preserving the originating project routing
  // and source so the asset router still copies the new render into
  // the right project. The original item stays in the queue history
  // — re-queueing is additive, not destructive.
  requeue: (itemId: string) => Promise<{ ok: true; item: QueueItem } | { ok: false; error: string }>
  reorder: (itemIds: string[]) => Promise<void>
  pause: () => Promise<void>
  resume: () => Promise<void>
  clearCompleted: () => Promise<void>
  clearFailed: () => Promise<void>
}

export interface EnqueueOptions {
  originatingProjectId?: string | null
  source?: QueueItemSource
}

const EMPTY_STATE: QueueState = {
  items: [],
  paused: false,
  schemaVersion: 1,
}

const QueueContext = createContext<QueueContextValue | null>(null)

// ---------------------------------------------------------------------------
// Polling cadence
// ---------------------------------------------------------------------------
//
// The queue is small and the read endpoint is cheap, but we still want to
// avoid a constant 1Hz poll for users who never open the panel. Two cadences:
//   - PANEL_OPEN_INTERVAL_MS: when the side panel is open, prioritise UX
//     responsiveness — drag-reorder, pause/resume, completion all need to
//     feel immediate.
//   - PANEL_CLOSED_INTERVAL_MS: just frequent enough to keep the badge count
//     accurate when the user re-opens after a long absence.
const PANEL_OPEN_INTERVAL_MS = 1000
const PANEL_CLOSED_INTERVAL_MS = 4000

// When fetches fail consistently — e.g., the python backend is restarting,
// or it's an older build that doesn't yet have the /api/queue routes — we
// don't want to spam the logs once per tick. Log only the first failure and
// the recovery, and hold a short debounce on subsequent identical errors.
// `LOG_REPEAT_INTERVAL_MS` caps how often the same kind of failure shows up
// so a stuck backend produces ~1 log/min instead of 1/poll-tick.
const LOG_REPEAT_INTERVAL_MS = 60_000

export function QueueProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<QueueState>(EMPTY_STATE)
  const [loading, setLoading] = useState(true)
  const [isPanelOpen, setIsPanelOpenState] = useState(false)
  const [runningProgress, setRunningProgress] = useState<RunningGenerationProgress | null>(null)
  // Hold the latest cadence in a ref so the polling effect can read it
  // without re-subscribing every time the panel opens/closes.
  const intervalMsRef = useRef(PANEL_CLOSED_INTERVAL_MS)
  // Used to coalesce overlapping fetches (e.g., a manual refresh fired while
  // the periodic tick is mid-flight). Without this, racing responses can
  // briefly flicker the UI.
  const inFlightRef = useRef<Promise<void> | null>(null)
  // Debounce repeated identical failure logs so a stale or unreachable
  // backend doesn't spam one warning per poll tick. Tracks the last
  // logged status and the wall-clock time of that log.
  const lastFailureRef = useRef<{ status: string; loggedAt: number } | null>(null)
  const wasFailingRef = useRef(false)
  // Mirror of state.items "is anything running?" so the polling loop
  // can decide whether to fetch progress without re-subscribing on
  // every state change. Updated inside fetchState below.
  const hasRunningRef = useRef(false)

  const fetchProgress = useCallback(async (): Promise<void> => {
    // Skip when no item is running — /api/generation/progress would
    // return idle and we'd just churn the React state with null.
    if (!hasRunningRef.current) {
      setRunningProgress((prev) => (prev === null ? prev : null))
      return
    }
    const result = await ApiClient.getGenerationProgress()
    if (!result.ok) {
      // A transient progress fetch failure shouldn't blank the bar
      // mid-render; keep the last known value until the next tick
      // succeeds. The queue's own fetch is the source of truth for
      // running/completed transitions, so we won't get stuck on a
      // stale progress value beyond the actual generation.
      return
    }
    const data = result.data
    if (data.status === 'running') {
      setRunningProgress({
        progress: data.progress,
        phase: data.phase,
        currentStep: data.currentStep,
        totalSteps: data.totalSteps,
      })
    } else {
      setRunningProgress(null)
    }
  }, [])

  const fetchState = useCallback(async (): Promise<void> => {
    if (inFlightRef.current) {
      return inFlightRef.current
    }
    const promise = (async () => {
      const result = await ApiClient.getQueueState()
      if (result.ok) {
        setState(result.data)
        // Update the "is anything running" mirror so the next progress
        // tick knows whether to bother. Doing this inside the queue
        // fetch (rather than in a separate useEffect on state.items)
        // saves one render cycle of mismatch.
        hasRunningRef.current = result.data.items.some((i) => i.status === 'running')
        if (!hasRunningRef.current) {
          setRunningProgress((prev) => (prev === null ? prev : null))
        }
        if (wasFailingRef.current) {
          logger.info('Queue: state fetch recovered')
          wasFailingRef.current = false
          lastFailureRef.current = null
        }
      } else {
        // Don't blank out the panel on transient errors — the next tick
        // will retry. Log the first failure of a streak immediately, then
        // throttle identical follow-ups to LOG_REPEAT_INTERVAL_MS so a
        // sustained backend outage doesn't fill the log file. A different
        // status code (e.g., 4XX -> 5XX) resets the debounce so a regime
        // change still shows up promptly.
        const status = String(result.status)
        const now = Date.now()
        const last = lastFailureRef.current
        const isFirstOrChanged = !last || last.status !== status
        const isOverdue = !last || now - last.loggedAt >= LOG_REPEAT_INTERVAL_MS
        if (isFirstOrChanged || isOverdue) {
          logger.warn(`Queue: state fetch failed (status=${status})`)
          lastFailureRef.current = { status, loggedAt: now }
        }
        wasFailingRef.current = true
      }
    })().finally(() => {
      inFlightRef.current = null
      setLoading(false)
    })
    inFlightRef.current = promise
    return promise
  }, [])

  // Initial fetch + cadence loop.
  useEffect(() => {
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout> | null = null

    const tick = async () => {
      if (cancelled) return
      await fetchState()
      // Pair the queue fetch with a progress fetch when something's
      // running. fetchProgress short-circuits when nothing's running
      // so we don't hit /api/generation/progress unnecessarily.
      if (!cancelled && hasRunningRef.current) {
        await fetchProgress()
      }
      if (cancelled) return
      timeoutId = setTimeout(tick, intervalMsRef.current)
    }
    void tick()

    return () => {
      cancelled = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [fetchState, fetchProgress])

  const setIsPanelOpen = useCallback((open: boolean) => {
    setIsPanelOpenState(open)
    intervalMsRef.current = open ? PANEL_OPEN_INTERVAL_MS : PANEL_CLOSED_INTERVAL_MS
    // Force an immediate fetch on open so the panel doesn't render a stale
    // snapshot while the user waits for the next interval tick.
    if (open) void fetchState()
  }, [fetchState])

  const refresh = useCallback(async () => {
    await fetchState()
  }, [fetchState])

  // -------------------------------------------------------------------------
  // Mutations
  // -------------------------------------------------------------------------

  const enqueue = useCallback<QueueContextValue['enqueue']>(async (payload, opts) => {
    const result = await ApiClient.enqueueQueueItem({
      payload,
      originatingProjectId: opts?.originatingProjectId ?? null,
      source: opts?.source ?? 'genspace',
    })
    if (!result.ok) {
      const message = (result.error as { message?: string })?.message ?? 'Failed to enqueue item'
      return { ok: false, error: message }
    }
    await fetchState()
    return { ok: true, item: result.data }
  }, [fetchState])

  const enqueueBatch = useCallback<QueueContextValue['enqueueBatch']>(async (items) => {
    if (items.length === 0) {
      return { ok: false, error: 'Batch is empty' }
    }
    const result = await ApiClient.enqueueQueueBatch({
      items: items.map(({ payload, opts }) => ({
        payload,
        originatingProjectId: opts?.originatingProjectId ?? null,
        source: opts?.source ?? 'genspace',
      })),
    })
    if (!result.ok) {
      const message = (result.error as { message?: string })?.message ?? 'Failed to enqueue batch'
      return { ok: false, error: message }
    }
    await fetchState()
    return { ok: true, items: result.data }
  }, [fetchState])

  const cancelPending = useCallback(async (itemId: string) => {
    const result = await ApiClient.cancelQueueItem(itemId)
    if (!result.ok) {
      logger.warn(`Queue: cancel-pending failed for ${itemId}`)
    }
    await fetchState()
  }, [fetchState])

  const removeItem = useCallback(async (itemId: string) => {
    const result = await ApiClient.removeQueueItem(itemId)
    if (!result.ok) {
      logger.warn(`Queue: remove failed for ${itemId}`)
    }
    await fetchState()
  }, [fetchState])

  const cancelRunning = useCallback(async () => {
    // The cancel endpoint is global (one running generation at a time)
    // — we don't need an item id. The runner observes the cancellation
    // and calls `cancel_running` on the queue handler, which moves the
    // item to `cancelled`. We refresh to reflect that transition.
    const result = await ApiClient.cancelGeneration()
    if (!result.ok) {
      logger.warn('Queue: cancel-running failed')
    }
    await fetchState()
  }, [fetchState])

  const updatePending = useCallback<QueueContextValue['updatePending']>(async (itemId, payload) => {
    const result = await ApiClient.updateQueueItem(itemId, { payload })
    if (!result.ok) {
      const message = (result.error as { message?: string })?.message ?? 'Failed to update item'
      logger.warn(`Queue: update-pending failed for ${itemId} (${message})`)
      await fetchState()
      return { ok: false, error: message }
    }
    await fetchState()
    return { ok: true }
  }, [fetchState])

  const requeue = useCallback<QueueContextValue['requeue']>(async (itemId) => {
    // Look up the item to clone its payload snapshot. We do this from
    // the latest in-memory state rather than a fresh GET — the panel
    // is already polling, so the snapshot is at most one tick stale,
    // and the user explicitly clicked re-queue on a row they can see.
    // If the item has been cleared between render and click (unlikely
    // but possible), fall back to a fresh fetch so we don't fail
    // silently.
    let target: QueueItem | undefined = state.items.find((i) => i.id === itemId)
    if (!target) {
      const result = await ApiClient.getQueueItem(itemId)
      if (!result.ok) {
        return { ok: false, error: 'Item no longer exists' }
      }
      target = result.data
    }
    const enqueued = await enqueue(target.payload, {
      originatingProjectId: target.originatingProjectId ?? undefined,
      source: target.source,
    })
    return enqueued
  }, [state.items, enqueue])

  const reorder = useCallback(async (itemIds: string[]) => {
    const result = await ApiClient.reorderQueue({ itemIds })
    if (!result.ok) {
      logger.warn('Queue: reorder failed')
    }
    await fetchState()
  }, [fetchState])

  const pause = useCallback(async () => {
    const result = await ApiClient.pauseQueue()
    if (result.ok) {
      setState(result.data)
    } else {
      logger.warn('Queue: pause failed')
      await fetchState()
    }
  }, [fetchState])

  const resume = useCallback(async () => {
    const result = await ApiClient.resumeQueue()
    if (result.ok) {
      setState(result.data)
    } else {
      logger.warn('Queue: resume failed')
      await fetchState()
    }
  }, [fetchState])

  const clearCompleted = useCallback(async () => {
    await ApiClient.clearQueueCompleted()
    await fetchState()
  }, [fetchState])

  const clearFailed = useCallback(async () => {
    await ApiClient.clearQueueFailed()
    await fetchState()
  }, [fetchState])

  // -------------------------------------------------------------------------
  // Counts derived from the latest state. Memoized so re-renders triggered by
  // siblings don't recompute these on every parent render.
  // -------------------------------------------------------------------------

  const { pendingCount, runningCount, activeCount } = useMemo(() => {
    let pending = 0
    let running = 0
    for (const item of state.items) {
      if (item.status === 'pending') pending += 1
      else if (item.status === 'running') running += 1
    }
    return { pendingCount: pending, runningCount: running, activeCount: pending + running }
  }, [state.items])

  const value = useMemo<QueueContextValue>(() => ({
    state,
    loading,
    runningProgress,
    pendingCount,
    runningCount,
    activeCount,
    isPanelOpen,
    setIsPanelOpen,
    refresh,
    enqueue,
    enqueueBatch,
    cancelPending,
    cancelRunning,
    removeItem,
    updatePending,
    requeue,
    reorder,
    pause,
    resume,
    clearCompleted,
    clearFailed,
  }), [
    state,
    loading,
    runningProgress,
    pendingCount,
    runningCount,
    activeCount,
    isPanelOpen,
    setIsPanelOpen,
    refresh,
    enqueue,
    enqueueBatch,
    cancelPending,
    cancelRunning,
    removeItem,
    updatePending,
    requeue,
    reorder,
    pause,
    resume,
    clearCompleted,
    clearFailed,
  ])

  return <QueueContext.Provider value={value}>{children}</QueueContext.Provider>
}

export function useQueue(): QueueContextValue {
  const ctx = useContext(QueueContext)
  if (!ctx) {
    throw new Error('useQueue must be used within QueueProvider')
  }
  return ctx
}
