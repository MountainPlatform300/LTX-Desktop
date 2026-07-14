import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Button } from './button'
import { Dialog } from './dialog'

export interface ConfirmDialogOptions {
  title: string
  message: ReactNode
  confirmLabel: string
  cancelLabel?: string
  variant?: 'default' | 'destructive'
}

type RequestConfirm = (options: ConfirmDialogOptions) => Promise<boolean>

let requestConfirm: RequestConfirm | null = null

export function confirmAction(options: ConfirmDialogOptions): Promise<boolean> {
  return requestConfirm?.(options) ?? Promise.resolve(false)
}

export function ConfirmDialogProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<ConfirmDialogOptions | null>(null)
  const resolverRef = useRef<((confirmed: boolean) => void) | null>(null)

  const settle = useCallback((confirmed: boolean) => {
    resolverRef.current?.(confirmed)
    resolverRef.current = null
    setPending(null)
  }, [])

  useEffect(() => {
    requestConfirm = (options) => new Promise<boolean>((resolve) => {
      resolverRef.current?.(false)
      resolverRef.current = resolve
      setPending(options)
    })
    return () => {
      if (requestConfirm) requestConfirm = null
      resolverRef.current?.(false)
      resolverRef.current = null
    }
  }, [])

  return (
    <>
      {children}
      {pending && (
        <Dialog
          title={pending.title}
          onClose={() => settle(false)}
          className="max-w-md"
          footer={(
            <>
              <Button variant="outline" autoFocus onClick={() => settle(false)}>
                {pending.cancelLabel ?? 'Cancel'}
              </Button>
              <Button
                onClick={() => settle(true)}
                className={
                  pending.variant === 'destructive'
                    ? 'bg-red-600 text-white hover:bg-red-500'
                    : 'bg-blue-600 text-white hover:bg-blue-500'
                }
              >
                {pending.confirmLabel}
              </Button>
            </>
          )}
        >
          <div className="flex items-start gap-3 text-sm leading-relaxed text-zinc-300">
            {pending.variant === 'destructive' && (
              <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
            )}
            <div>{pending.message}</div>
          </div>
        </Dialog>
      )}
    </>
  )
}
