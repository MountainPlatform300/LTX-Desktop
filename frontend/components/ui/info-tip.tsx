import React, { useEffect, useRef, useState, type ReactNode } from 'react'
import ReactDOM from 'react-dom'
import { Info } from 'lucide-react'
import { cn } from '@/lib/utils'

const GAP_PX = 8

/**
 * A small, clickable info affordance: a subtle ⓘ icon that toggles a dark
 * popover with an explanation. Unlike {@link Tooltip} (hover-only, white, not
 * interactive), this is a deliberate click target — used to keep a surface
 * decluttered while still offering "what does this do?" on demand.
 *
 * The popover is portaled to `document.body` so it renders above modals and is
 * never clipped by `overflow` ancestors. Closes on outside-click or Escape
 * (Escape uses capture so it fires even when a modal stops key propagation).
 */
export function InfoTip({
  content,
  label = 'More info',
  side = 'bottom',
  className,
}: {
  content: ReactNode
  /** Accessible label for the icon button. */
  label?: string
  side?: 'top' | 'bottom' | 'left' | 'right'
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const [style, setStyle] = useState<React.CSSProperties>({})
  const btnRef = useRef<HTMLButtonElement>(null)
  const popRef = useRef<HTMLDivElement>(null)

  const computeStyle = (): React.CSSProperties => {
    const rect = btnRef.current?.getBoundingClientRect()
    if (!rect) return {}
    switch (side) {
      case 'top':
        return { left: rect.left + rect.width / 2, top: rect.top - GAP_PX, transform: 'translate(-50%, -100%)' }
      case 'bottom':
        return { left: rect.left + rect.width / 2, top: rect.bottom + GAP_PX, transform: 'translate(-50%, 0)' }
      case 'left':
        return { left: rect.left - GAP_PX, top: rect.top + rect.height / 2, transform: 'translate(-100%, -50%)' }
      case 'right':
        return { left: rect.right + GAP_PX, top: rect.top + rect.height / 2, transform: 'translate(0, -50%)' }
    }
  }

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node
      if (btnRef.current?.contains(t) || popRef.current?.contains(t)) return
      setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey, true)
    }
  }, [open])

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        aria-label={label}
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation()
          setStyle(computeStyle())
          setOpen((v) => !v)
        }}
        className={cn(
          'inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full transition-colors',
          open ? 'text-blue-300' : 'text-zinc-600 hover:text-zinc-300',
          className,
        )}
      >
        <Info className="h-3.5 w-3.5" />
      </button>
      {open &&
        ReactDOM.createPortal(
          <div
            ref={popRef}
            role="tooltip"
            className="fixed z-[99999] max-w-[260px] rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-xs leading-snug text-zinc-200 shadow-xl shadow-black/40"
            style={style}
          >
            {content}
          </div>,
          document.body,
        )}
    </>
  )
}
