import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react'

export type ToastVariant = 'info' | 'success' | 'warning'

export interface ToastInput {
  title: string
  description?: string
  variant?: ToastVariant
  /** Auto-dismiss after this many ms (default 6000; 0 disables). */
  durationMs?: number
  /** Optional action button. */
  actionLabel?: string
  onAction?: () => void
}

interface Toast extends ToastInput {
  id: string
}

interface ToastContextValue {
  addToast: (toast: ToastInput) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

const DEFAULT_DURATION_MS = 6000

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

  const dismiss = useCallback((id: string) => {
    setToasts((cur) => cur.filter((t) => t.id !== id))
    const timer = timers.current.get(id)
    if (timer) {
      clearTimeout(timer)
      timers.current.delete(id)
    }
  }, [])

  const addToast = useCallback((toast: ToastInput) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setToasts((cur) => [...cur, { ...toast, id }])
    const duration = toast.durationMs ?? DEFAULT_DURATION_MS
    if (duration > 0) {
      timers.current.set(id, setTimeout(() => dismiss(id), duration))
    }
  }, [dismiss])

  useEffect(() => {
    const map = timers.current
    return () => {
      map.forEach((t) => clearTimeout(t))
      map.clear()
    }
  }, [])

  const value = useMemo(() => ({ addToast }), [addToast])

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 w-80 max-w-[calc(100vw-2rem)] pointer-events-none">
        {toasts.map((t) => (
          <ToastCard key={t.id} toast={t} onDismiss={() => dismiss(t.id)} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}

function ToastCard({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  const Icon = toast.variant === 'success' ? CheckCircle2 : toast.variant === 'warning' ? AlertTriangle : Info
  const accent =
    toast.variant === 'success'
      ? 'text-emerald-400'
      : toast.variant === 'warning'
        ? 'text-amber-400'
        : 'text-violet-400'
  return (
    <div className="pointer-events-auto flex items-start gap-2.5 rounded-xl border border-zinc-700 bg-zinc-900/95 backdrop-blur shadow-2xl px-3.5 py-3">
      <Icon className={`h-4 w-4 shrink-0 mt-0.5 ${accent}`} />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-zinc-100">{toast.title}</p>
        {toast.description && <p className="text-[11px] text-zinc-400 mt-0.5">{toast.description}</p>}
        {toast.actionLabel && toast.onAction && (
          <button
            onClick={() => {
              toast.onAction?.()
              onDismiss()
            }}
            className="mt-1.5 text-[11px] font-medium text-violet-300 hover:text-violet-200"
          >
            {toast.actionLabel}
          </button>
        )}
      </div>
      <button onClick={onDismiss} className="shrink-0 h-5 w-5 rounded-md text-zinc-500 hover:text-white hover:bg-zinc-800 flex items-center justify-center">
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}
