import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ChevronRight, type LucideIcon } from 'lucide-react'

export interface ContextMenuLeaf {
  type?: 'item'
  label: string
  icon?: LucideIcon
  onClick?: () => void
  disabled?: boolean
  danger?: boolean
  /** When present, the row opens a hover flyout of these items instead of
   *  firing `onClick`. One nesting level is supported. */
  children?: ContextMenuLeaf[]
}

export type ContextMenuItem = { type: 'separator' } | ContextMenuLeaf

// Lightweight right-click menu: renders at a fixed viewport position, clamps
// to the viewport edges, and closes on outside click / Escape / scroll.
export function ClipContextMenu({
  x,
  y,
  items,
  onClose,
}: {
  x: number
  y: number
  items: ContextMenuItem[]
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [pos, setPos] = useState({ x, y })
  const [openSub, setOpenSub] = useState<number | null>(null)
  // Flip submenu flyouts to the left when the menu sits in the right half of
  // the viewport so they don't overflow off-screen.
  const subToLeft = pos.x > window.innerWidth / 2

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const nextX = Math.min(x, window.innerWidth - rect.width - 8)
    const nextY = Math.min(y, window.innerHeight - rect.height - 8)
    setPos({ x: Math.max(8, nextX), y: Math.max(8, nextY) })
  }, [x, y])

  useEffect(() => {
    const close = () => onClose()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('scroll', close, true)
    window.addEventListener('resize', close)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('scroll', close, true)
      window.removeEventListener('resize', close)
      window.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  return createPortal(
    <div className="fixed inset-0 z-[70]" onClick={onClose} onContextMenu={(e) => { e.preventDefault(); onClose() }}>
      <div
        ref={ref}
        className="absolute min-w-48 rounded-lg border border-zinc-700 bg-zinc-900 py-1 shadow-2xl"
        style={{ left: pos.x, top: pos.y }}
        onClick={(e) => e.stopPropagation()}
      >
        {items.map((item, i) => {
          if (item.type === 'separator') {
            return <div key={`sep-${i}`} className="my-1 h-px bg-zinc-800" />
          }
          const Icon = item.icon
          const hasChildren = !!item.children?.length
          if (hasChildren) {
            return (
              <div
                key={item.label}
                className="relative"
                onMouseEnter={() => setOpenSub(i)}
                onMouseLeave={() => setOpenSub((cur) => (cur === i ? null : cur))}
              >
                <button
                  disabled={item.disabled}
                  className={`flex w-full items-center gap-2.5 px-3 py-1.5 text-left text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-30 disabled:cursor-not-allowed ${
                    openSub === i ? 'bg-zinc-800' : ''
                  }`}
                >
                  {Icon && <Icon className="h-3.5 w-3.5 shrink-0" />}
                  <span className="flex-1">{item.label}</span>
                  <ChevronRight className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                </button>
                {openSub === i && (
                  <div
                    className={`absolute top-0 min-w-44 rounded-lg border border-zinc-700 bg-zinc-900 py-1 shadow-2xl ${
                      subToLeft ? 'right-full mr-1' : 'left-full ml-1'
                    }`}
                  >
                    {item.children!.map((child) => {
                      const ChildIcon = child.icon
                      return (
                        <button
                          key={child.label}
                          disabled={child.disabled}
                          onClick={() => {
                            if (child.disabled) return
                            child.onClick?.()
                            onClose()
                          }}
                          className="flex w-full items-center gap-2.5 px-3 py-1.5 text-left text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-30 disabled:cursor-not-allowed"
                        >
                          {ChildIcon && <ChildIcon className="h-3.5 w-3.5 shrink-0" />}
                          {child.label}
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          }
          return (
            <button
              key={item.label}
              disabled={item.disabled}
              onClick={() => {
                if (item.disabled) return
                item.onClick?.()
                onClose()
              }}
              className={`flex w-full items-center gap-2.5 px-3 py-1.5 text-left text-xs disabled:opacity-30 disabled:cursor-not-allowed ${
                item.danger
                  ? 'text-red-300 hover:bg-red-500/10'
                  : 'text-zinc-200 hover:bg-zinc-800'
              }`}
            >
              {Icon && <Icon className="h-3.5 w-3.5 shrink-0" />}
              {item.label}
            </button>
          )
        })}
      </div>
    </div>,
    document.body,
  )
}
