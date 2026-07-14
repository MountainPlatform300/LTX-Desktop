import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Cloud,
  Loader2,
  Play,
  RefreshCw,
  Square,
  Trash2,
} from 'lucide-react'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { ApiClient, type ApiSuccessOf } from '../../lib/api-client'
import { Tooltip } from '../../components/ui/tooltip'

// Refresh cadence for the pod list while the panel is mounted. RunPod's pod
// state changes asynchronously (stop/resume take a few seconds to land), so a
// short periodic refresh keeps cost + status honest without hammering the API.
const REFRESH_INTERVAL_MS = 20_000

type Pod = ApiSuccessOf<'listRunpodPods'>[number]

export interface PodWorkTarget {
  kind: 'dataset' | 'run'
  id: string
  label: string
  stage: string
}

export interface PodLifecycleInfo {
  autoStopAt: string | null
  autoStopDisabled: boolean
  releaseStatus: string | null
  releaseError: string | null
  workspacePolicy: 'primary_cache' | 'ephemeral_any_region'
}

type PodTone = 'running' | 'stopped' | 'other'

function podStatusTone(pod: Pod): PodTone {
  if (pod.running) return 'running'
  if (pod.desiredStatus === 'STOPPED') return 'stopped'
  return 'other'
}

const DOT_CLASS: Record<PodTone, string> = {
  running: 'bg-emerald-400',
  stopped: 'bg-amber-400',
  other: 'bg-zinc-500',
}

function formatCost(perHr: number | null | undefined): string {
  if (perHr == null) return ''
  return `$${perHr.toFixed(2)}/hr`
}

function formatTimestamp(d: Date): string {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function effectiveUptimeSeconds(pod: Pod, nowMs: number): number | null {
  const reported = pod.uptimeSeconds ?? null
  if (!pod.running || !pod.lastStartedAt) return reported
  const startedMs = Date.parse(pod.lastStartedAt)
  if (!Number.isFinite(startedMs)) return reported
  const sinceStart = Math.max(0, Math.floor((nowMs - startedMs) / 1000))
  return Math.max(reported ?? 0, sinceStart)
}

function formatRuntime(seconds: number | null): string {
  if (seconds == null) return 'Runtime unavailable'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}

function estimatedSpend(pod: Pod, nowMs: number): number | null {
  const uptime = effectiveUptimeSeconds(pod, nowMs)
  return uptime != null && pod.costPerHr != null ? (uptime / 3600) * pod.costPerHr : null
}

/**
 * Always-visible RunPod compute panel for the LoRA Trainer sidebar.
 *
 * Lists every pod on the account with live status + cost and lets the user
 * Stop (pause), Resume, or Terminate each — so a pod left running by a failed
 * or cancelled run can't keep billing silently. Terminate deletes pod + disk;
 * Stop keeps the disk (a small storage fee remains) and is reversible via
 * Resume. The panel only loads when a RunPod key is configured; otherwise it
 * shows a connect hint. `activePodIds` carries pod ids that have an in-progress
 * training job so Stop/Terminate on one of those warns before interrupting it.
 */
export function ComputePanel({
  activePodIds,
  workByPodId,
  lifecycleByPodId,
  onOpenWork,
  collapsed,
  onToggleCollapsed,
}: {
  activePodIds: Set<string>
  workByPodId: ReadonlyMap<string, PodWorkTarget>
  lifecycleByPodId: ReadonlyMap<string, PodLifecycleInfo>
  onOpenWork: (target: PodWorkTarget) => void
  collapsed: boolean
  onToggleCollapsed: () => void
}) {
  const { settings } = useAppSettings()
  const hasKey = settings.hasRunpodApiKey

  const [pods, setPods] = useState<Pod[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  // pod id -> in-flight action ('stop' | 'resume' | 'terminate'); disables the
  // row's buttons and shows a spinner on the active one so a double-click can't
  // fire two contradictory lifecycle calls on the same pod.
  const [busy, setBusy] = useState<Record<string, 'stop' | 'resume' | 'terminate'>>({})
  // pod id pending a terminate confirm (destructive — needs an explicit ok).
  const [confirmTerminate, setConfirmTerminate] = useState<string | null>(null)
  // Latest action outcome message (per pod), surfaced inline so a failed
  // stop/resume isn't silent.
  const [podMessage, setPodMessage] = useState<Record<string, string>>({})
  const [nowMs, setNowMs] = useState(() => Date.now())

  const mountedRef = useRef(true)
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const refresh = useCallback(async () => {
    if (!hasKey) return
    setLoading(true)
    setError(null)
    const res = await ApiClient.listRunpodPods()
    if (!mountedRef.current) return
    if (res.ok) {
      setPods(res.data)
      setUpdatedAt(new Date())
    } else {
      setError(res.error.message)
      setPods([])
    }
    setLoading(false)
  }, [hasKey])

  // Initial load + periodic refresh while mounted. The interval re-fetches
  // even on failure so a transient RunPod hiccup self-heals.
  useEffect(() => {
    if (!hasKey) return
    void refresh()
    const id = window.setInterval(() => void refresh(), REFRESH_INTERVAL_MS)
    return () => window.clearInterval(id)
  }, [hasKey, refresh])

  useEffect(() => {
    if (!pods.some((pod) => pod.running)) return
    const id = window.setInterval(() => setNowMs(Date.now()), 1_000)
    return () => window.clearInterval(id)
  }, [pods])

  const runAction = useCallback(
    async (
      podId: string,
      kind: 'stop' | 'resume' | 'terminate',
      call: () => Promise<
        | { ok: true; data: { ok: boolean; message: string } }
        | { ok: false; error: { code: string; message: string } }
      >,
    ) => {
      setBusy((b) => ({ ...b, [podId]: kind }))
      setPodMessage((m) => ({ ...m, [podId]: '' }))
      const res = await call()
      if (!mountedRef.current) return
      if (res.ok) {
        if (!res.data.ok) setPodMessage((m) => ({ ...m, [podId]: res.data.message }))
      } else {
        setPodMessage((m) => ({ ...m, [podId]: res.error.message }))
      }
      setBusy((b) => {
        const next = { ...b }
        delete next[podId]
        return next
      })
      setConfirmTerminate(null)
      // Refresh immediately so the row reflects the new state instead of
      // waiting up to REFRESH_INTERVAL_MS for the next tick.
      void refresh()
    },
    [refresh],
  )

  const onStop = (podId: string) =>
    runAction(podId, 'stop', () => ApiClient.stopRunpodPod(podId))
  const onResume = (podId: string) =>
    runAction(podId, 'resume', () => ApiClient.resumeRunpodPod(podId))
  const onTerminate = (podId: string) =>
    runAction(podId, 'terminate', () => ApiClient.terminateRunpodPod(podId))

  const openSettings = () =>
    window.dispatchEvent(new CustomEvent('open-settings', { detail: { tab: 'loraTrainer' } }))
  const idlePods = pods.filter(
    (pod) => pod.running && pod.createdByApp && !activePodIds.has(pod.id),
  )
  const idleSpend = idlePods.reduce(
    (total, pod) => total + (estimatedSpend(pod, nowMs) ?? 0),
    0,
  )

  // No RunPod key: nothing to list. Offer the connect entry point so the panel
  // is never a dead blank — the user can wire up RunPod without leaving the
  // trainer.
  if (!hasKey) {
    return (
      <div className="flex h-full min-h-0 flex-col overflow-hidden border-t border-zinc-800">
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-expanded={!collapsed}
          className="flex h-9 shrink-0 items-center gap-1.5 px-3 text-left hover:bg-zinc-800/70"
        >
          {collapsed ? <ChevronRight className="h-3.5 w-3.5 text-zinc-500" /> : <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />}
          <Cloud className="h-3.5 w-3.5 text-zinc-400" />
          <span className="text-[11px] font-semibold text-zinc-300">Compute</span>
        </button>
        {!collapsed && <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        <p className="mt-1.5 text-[10px] text-zinc-500 leading-relaxed">
          Connect RunPod to see and control your GPU pods.
        </p>
        <button
          onClick={openSettings}
          className="mt-2 w-full text-[11px] px-2 py-1.5 rounded-md border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600"
        >
          Add RunPod API key
        </button>
        </div>}
      </div>
    )
  }

  const runningOwnedPods = pods.filter((pod) => pod.running && pod.createdByApp)
  const cloudTone = idlePods.length > 0 ? 'text-amber-400' : 'text-emerald-400'
  const runningSpend = runningOwnedPods.reduce(
    (total, pod) => total + (estimatedSpend(pod, nowMs) ?? 0),
    0,
  )
  const runningRate = runningOwnedPods.reduce(
    (total, pod) => total + (pod.costPerHr ?? 0),
    0,
  )

  return (
    <div className="flex h-full min-h-0 flex-col border-t border-zinc-800">
      <div className="flex h-9 shrink-0 items-center justify-between px-2">
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-expanded={!collapsed}
          className={`flex min-w-0 flex-1 items-center gap-1.5 rounded px-1 py-1 text-left hover:bg-zinc-800/70 ${
            idlePods.length > 0
              ? 'bg-amber-500/[0.07]'
              : runningOwnedPods.length > 0
                ? 'bg-emerald-500/[0.06]'
                : ''
          }`}
        >
          {collapsed ? <ChevronRight className="h-3.5 w-3.5 text-zinc-500" /> : <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />}
          <span className="relative flex h-4 w-4 shrink-0 items-center justify-center">
            <Cloud className={`h-3.5 w-3.5 ${runningOwnedPods.length > 0 ? cloudTone : 'text-zinc-400'}`} />
            {runningOwnedPods.length > 0 && (
              <>
                <span className={`absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full ${idlePods.length > 0 ? 'bg-amber-400' : 'bg-emerald-400'}`} />
                <span className={`absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full motion-safe:animate-ping ${idlePods.length > 0 ? 'bg-amber-400' : 'bg-emerald-400'}`} />
              </>
            )}
          </span>
          <span className="text-[11px] font-semibold text-zinc-300">Compute</span>
          {runningOwnedPods.length === 0 && pods.length > 0 && (
            <span className="text-[10px] text-zinc-600">{pods.length} pod{pods.length === 1 ? '' : 's'}</span>
          )}
          {runningOwnedPods.length > 0 && (
            <span
              className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-300"
              title={`Estimated across running app pods since their current start · $${runningRate.toFixed(2)}/hr total`}
            >
              {runningOwnedPods.length} running{runningSpend > 0 ? ` · ~$${runningSpend.toFixed(2)}` : ''}
            </span>
          )}
          {idlePods.length > 0 && (
            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-medium text-amber-300">
              {idlePods.length} idle{idleSpend > 0 ? ` · ~$${idleSpend.toFixed(2)}` : ''}
            </span>
          )}
        </button>
        {!collapsed && (
        <div className="flex items-center gap-1.5">
          {updatedAt && (
            <span className="text-[10px] text-zinc-600 tabular-nums" title="Last refreshed">
              {formatTimestamp(updatedAt)}
            </span>
          )}
          <button
            onClick={() => void refresh()}
            disabled={loading}
            title="Refresh now"
            className="h-5 w-5 flex items-center justify-center rounded text-zinc-500 hover:text-white hover:bg-zinc-800 disabled:opacity-40"
          >
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCw className="h-3 w-3" />
            )}
          </button>
        </div>
        )}
      </div>

      {!collapsed && (
      <>
      {error && (
        <p className="px-3 pb-1.5 text-[10px] text-red-400/90 leading-relaxed">
          Couldn&rsquo;t load pods: {error}
        </p>
      )}

      <div className="overflow-y-auto px-1.5 pb-2 flex-1 min-h-0">
        {pods.length === 0 && !loading ? (
          <p className="px-2 py-2 text-[10px] text-zinc-600 leading-relaxed">
            No pods on your account.
          </p>
        ) : (
          <div className="space-y-1">
            {pods.map((pod) => (
              <PodRow
                key={pod.id}
                pod={pod}
                isActiveRun={activePodIds.has(pod.id)}
                work={workByPodId.get(pod.id)}
                lifecycle={lifecycleByPodId.get(pod.id)}
                nowMs={nowMs}
                busy={busy[pod.id]}
                message={podMessage[pod.id]}
                confirmTerminate={confirmTerminate === pod.id}
                onConfirmTerminate={() => setConfirmTerminate(pod.id)}
                onCancelTerminate={() => setConfirmTerminate(null)}
                onStop={() => onStop(pod.id)}
                onResume={() => onResume(pod.id)}
                onTerminate={() => onTerminate(pod.id)}
                onOpenWork={onOpenWork}
              />
            ))}
          </div>
        )}
      </div>
      </>
      )}
    </div>
  )
}

function PodRow({
  pod,
  isActiveRun,
  work,
  lifecycle,
  nowMs,
  busy,
  message,
  confirmTerminate,
  onConfirmTerminate,
  onCancelTerminate,
  onStop,
  onResume,
  onTerminate,
  onOpenWork,
}: {
  pod: Pod
  isActiveRun: boolean
  work?: PodWorkTarget
  lifecycle?: PodLifecycleInfo
  nowMs: number
  busy: 'stop' | 'resume' | 'terminate' | undefined
  message: string | undefined
  confirmTerminate: boolean
  onConfirmTerminate: () => void
  onCancelTerminate: () => void
  onStop: () => void
  onResume: () => void
  onTerminate: () => void
  onOpenWork: (target: PodWorkTarget) => void
}) {
  const tone = podStatusTone(pod)
  const cost = formatCost(pod.costPerHr)
  const uptime = effectiveUptimeSeconds(pod, nowMs)
  const spend = estimatedSpend(pod, nowMs)
  const idleBilling = pod.running && pod.createdByApp && !isActiveRun
  // Warn before Stop/Terminate when this pod is driving an in-progress run —
  // stopping interrupts the job (it'll fail/retry), terminating loses the pod.
  const interruptWarning = isActiveRun && pod.running
  const actionDisabled = busy !== undefined || !pod.createdByApp
  const autoStopMs = lifecycle?.autoStopAt ? Date.parse(lifecycle.autoStopAt) : NaN
  const autoStopSeconds = Number.isFinite(autoStopMs)
    ? Math.max(0, Math.ceil((autoStopMs - nowMs) / 1000))
    : null
  const autoStopLabel = autoStopSeconds != null
    ? `${Math.floor(autoStopSeconds / 60)}:${String(autoStopSeconds % 60).padStart(2, '0')}`
    : null

  return (
    <div className="rounded-md bg-zinc-800/40 px-2 py-1.5">
      <div className="flex items-center gap-1.5">
        <span
          className={`h-1.5 w-1.5 rounded-full shrink-0 ${DOT_CLASS[tone]} ${
            idleBilling ? 'bg-amber-400 motion-safe:animate-pulse' : ''
          }`}
        />
        <span className="text-[11px] text-zinc-200 truncate flex-1" title={pod.gpu || pod.name}>
          {pod.gpu || pod.name}
        </span>
        {pod.createdByApp && (
          <span
            className="shrink-0 text-[9px] leading-none px-1 py-0.5 rounded bg-blue-500/15 text-blue-300 border border-blue-500/30"
            title="Created by this app"
          >
            this app
          </span>
        )}
      </div>

      <div className="mt-0.5 flex items-center gap-1.5">
        <span className="text-[10px] text-zinc-500 tabular-nums">
          {pod.running
            ? `${formatRuntime(uptime)} · ${spend != null ? `estimated $${spend.toFixed(2)} so far · ` : ''}${cost}`
            : cost}
        </span>
        <span className="text-[10px] text-zinc-600 truncate" title={pod.id}>
          {pod.id.slice(0, 8)}
        </span>
      </div>

      {message && (
        <p className="mt-1 text-[10px] text-amber-400/90 leading-snug break-words">{message}</p>
      )}
      {idleBilling && (
        <p className="mt-1 text-[10px] font-medium text-amber-300">
          {lifecycle?.autoStopDisabled
            ? 'Idle and still billing · Auto-stop off'
            : autoStopLabel
              ? `Idle and still billing · Auto-${lifecycle?.workspacePolicy === 'primary_cache' ? 'stop' : 'terminate'} in ${autoStopLabel}`
              : 'Idle and still billing. Stop the pod to end GPU charges.'}
        </p>
      )}
      {lifecycle?.releaseStatus === 'failed' && (
        <p className="mt-1 text-[10px] text-red-300">
          Auto-stop failed: {lifecycle.releaseError || 'Unknown RunPod error'}. Use Stop below to retry.
        </p>
      )}
      {work && (
        <button
          type="button"
          onClick={() => onOpenWork(work)}
          className="mt-1 text-[10px] font-medium text-blue-400 hover:text-blue-300"
        >
          View {work.kind === 'run' ? 'run' : 'dataset'} · {work.label} ({work.stage})
        </button>
      )}
      {!pod.createdByApp && (
        <p className="mt-1 text-[10px] text-zinc-500">
          Managed outside LTX Desktop; lifecycle controls are disabled.
        </p>
      )}

      {confirmTerminate ? (
        <div className="mt-1.5 rounded border border-red-500/40 bg-red-500/[0.07] p-1.5">
          <p className="text-[10px] text-red-200 leading-snug">
            Terminate deletes the pod and its disk. {interruptWarning && 'It is running an active training job — stopping will interrupt it. '}
            Continue?
          </p>
          <div className="mt-1.5 flex gap-1.5">
            <button
              onClick={onTerminate}
              disabled={actionDisabled}
              className="flex-1 flex items-center justify-center gap-1 rounded bg-red-600 px-2 py-1 text-[10px] font-medium text-white hover:bg-red-500 disabled:opacity-40"
            >
              {busy === 'terminate' && <Loader2 className="h-3 w-3 animate-spin" />}
              Terminate
            </button>
            <button
              onClick={onCancelTerminate}
              disabled={actionDisabled}
              className="rounded border border-zinc-700 px-2 py-1 text-[10px] text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="mt-1.5 flex items-center gap-1.5">
          {pod.running ? (
            <ActionButton
              label={idleBilling ? 'Stop billing' : 'Stop'}
              icon={<Square className="h-3 w-3" />}
              busy={busy === 'stop'}
              disabled={actionDisabled}
              onClick={onStop}
              tooltip={
                idleBilling
                  ? 'Stop billing: pauses GPU charges. A small disk fee may remain.'
                  : 'Stop pauses GPU billing (a small disk fee remains). Resume later.'
              }
              warning={interruptWarning}
            />
          ) : (
            <ActionButton
              label="Resume"
              icon={<Play className="h-3 w-3" />}
              busy={busy === 'resume'}
              disabled={actionDisabled}
              onClick={onResume}
              tooltip="Resume restarts the pod and resumes GPU billing."
            />
          )}
          <ActionButton
            label="Terminate"
            icon={<Trash2 className="h-3 w-3" />}
            busy={busy === 'terminate'}
            disabled={actionDisabled}
            onClick={onConfirmTerminate}
            tone="danger"
            tooltip="Terminate deletes the pod and its disk permanently."
            warning={interruptWarning}
          />
        </div>
      )}
    </div>
  )
}

function ActionButton({
  label,
  icon,
  busy,
  disabled,
  onClick,
  tooltip,
  tone = 'neutral',
  warning = false,
}: {
  label: string
  icon: React.ReactNode
  busy: boolean
  disabled: boolean
  onClick: () => void
  tooltip: string
  tone?: 'neutral' | 'danger'
  warning?: boolean
}) {
  const base =
    'flex-1 flex items-center justify-center gap-1 rounded px-2 py-1 text-[10px] font-medium disabled:opacity-40'
  const toneClass =
    tone === 'danger'
      ? 'border border-red-500/40 text-red-300 hover:bg-red-500/10'
      : 'border border-zinc-700 text-zinc-300 hover:text-white hover:border-zinc-600'
  const button = (
    <button onClick={onClick} disabled={disabled} className={`${base} ${toneClass}`}>
      {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : icon}
      {label}
      {warning && <AlertTriangle className="h-3 w-3 text-amber-400" />}
    </button>
  )
  // Tooltip wraps the button so the cost framing + active-run warning are
  // discoverable on hover without crowding the compact row. `flex-1` on the
  // tooltip wrapper lets the action buttons share the row width equally.
  return <Tooltip wide side="top" content={tooltip} className="flex-1">{button}</Tooltip>
}
