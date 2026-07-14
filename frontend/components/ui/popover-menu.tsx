import { useEffect, useRef, useState, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

/**
 * Generalization of `SettingsDropdown`: same compact trigger + open-upward
 * popover + close-on-outside-click behavior, but accepts arbitrary `children`
 * instead of an option list, so it can host sliders, switches, and other custom
 * controls. Used by the GenSpace prompt bar's Strengths and Settings menus.
 *
 * The popover opens upward (`bottom-full`) so it sits above the bar, matching
 * `SettingsDropdown`. `align` controls which edge lines up with the trigger.
 */
export function PopoverMenu({
  trigger,
  children,
  title,
  align = 'left',
  triggerClassName,
  triggerTitle,
  popoverClassName,
  disabled,
}: {
  trigger: ReactNode
  children: ReactNode
  /** Optional small uppercase header rendered above the content. */
  title?: ReactNode
  align?: 'left' | 'right'
  triggerClassName?: string
  /** Hover tooltip on the trigger button (e.g. explains what the menu controls). */
  triggerTitle?: string
  popoverClassName?: string
  disabled?: boolean
}) {
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isOpen) return
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        title={triggerTitle}
        className={cn(
          'flex shrink-0 items-center gap-1 whitespace-nowrap px-2 py-1.5 rounded-md transition-colors',
          disabled
            ? 'opacity-40 cursor-not-allowed'
            : isOpen
              ? 'bg-zinc-700 hover:bg-zinc-700'
              : 'hover:bg-zinc-800',
          triggerClassName,
        )}
      >
        {trigger}
      </button>

      {isOpen && !disabled && (
        <div
          className={cn(
            'absolute bottom-full mb-2 bg-zinc-800 border border-zinc-700 rounded-md p-3 min-w-[180px] shadow-xl z-[9999]',
            align === 'right' ? 'right-0' : 'left-0',
            popoverClassName,
          )}
        >
          {title && (
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">{title}</div>
          )}
          {children}
        </div>
      )}
    </div>
  )
}
