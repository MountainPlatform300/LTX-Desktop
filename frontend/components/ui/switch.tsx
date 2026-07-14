import { cn } from '@/lib/utils'

/**
 * Small on/off toggle used inside the GenSpace prompt-bar Settings popover.
 * Accessible: role="switch" + aria-checked + keyboard Space/Enter toggle.
 */
export function Switch({
  checked,
  onChange,
  label,
  description,
  disabled,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  label: string
  description?: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between gap-3 rounded-md px-1.5 py-1.5 text-left transition-colors hover:bg-zinc-700/60 disabled:opacity-40 disabled:hover:bg-transparent"
    >
      <span className="flex flex-col leading-tight">
        <span className={cn('text-sm', checked ? 'text-white' : 'text-zinc-300')}>{label}</span>
        {description && <span className="text-[11px] text-zinc-500">{description}</span>}
      </span>
      <span
        className={cn(
          'relative h-5 w-9 shrink-0 rounded-full transition-colors',
          checked ? 'bg-blue-600' : 'bg-zinc-600',
        )}
      >
        <span
          className={cn(
            'absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform',
            checked ? 'translate-x-4' : 'translate-x-0.5',
          )}
        />
      </span>
    </button>
  )
}
