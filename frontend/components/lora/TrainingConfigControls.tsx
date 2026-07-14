import { useState, type ReactNode } from 'react'
import { ChevronDown, Info } from 'lucide-react'
import { Tooltip } from '../ui/tooltip'
import { cn } from '@/lib/utils'
import type { ConfigFieldDescriptor } from './trainingConfigFields'

const INPUT_CLASS =
  'w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500'

const AUTO_SENTINEL = '__auto__'

function FieldLabel({ label, help }: { label: string; help: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <label className="text-xs font-medium text-zinc-300">{label}</label>
      <Tooltip content={help} wide side="top">
        <Info className="h-3.5 w-3.5 text-zinc-500 hover:text-zinc-300 cursor-help" />
      </Tooltip>
    </div>
  )
}

// Segmented Auto / On / Off control backing a nullable boolean (null = Auto).
function TriBool({ value, onChange }: { value: boolean | null; onChange: (v: boolean | null) => void }) {
  const options: { v: boolean | null; label: string }[] = [
    { v: null, label: 'Auto' },
    { v: true, label: 'On' },
    { v: false, label: 'Off' },
  ]
  return (
    <div className="flex gap-1.5">
      {options.map((o) => (
        <button
          key={o.label}
          type="button"
          onClick={() => onChange(o.v)}
          className={cn(
            'flex-1 px-2 py-1.5 rounded-md text-xs border transition-colors',
            value === o.v
              ? 'border-blue-500 bg-blue-500/10 text-white'
              : 'border-zinc-700 text-zinc-400 hover:text-white',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/**
 * One declarative config control: renders the field's label + info tooltip and
 * the right input for its `control` type. `value`/`onChange` are intentionally
 * loosely typed (the descriptor drives coercion) so the editor can pass any
 * config key through a single map.
 */
export function ConfigField({
  field,
  value,
  onChange,
}: {
  field: ConfigFieldDescriptor
  value: unknown
  onChange: (value: unknown) => void
}) {
  const control = (() => {
    switch (field.control) {
      case 'number':
        return (
          <input
            type="number"
            value={value as number}
            min={field.min}
            max={field.max}
            step={field.step}
            onChange={(e) => onChange(Number(e.target.value))}
            className={INPUT_CLASS}
          />
        )
      case 'nullableNumber': {
        const isAuto = value === null || value === undefined
        return (
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1.5 text-xs text-zinc-400 shrink-0">
              <input
                type="checkbox"
                checked={isAuto}
                onChange={(e) => onChange(e.target.checked ? null : (field.min ?? 0))}
                className="accent-blue-500"
              />
              Auto
            </label>
            <input
              type="number"
              disabled={isAuto}
              value={isAuto ? '' : (value as number)}
              min={field.min}
              max={field.max}
              step={field.step}
              onChange={(e) => onChange(Number(e.target.value))}
              className={cn(INPUT_CLASS, isAuto && 'opacity-40')}
            />
          </div>
        )
      }
      case 'text':
        return (
          <input
            type="text"
            value={(value as string) ?? ''}
            placeholder={field.placeholder}
            spellCheck={false}
            onChange={(e) => onChange(e.target.value)}
            className={INPUT_CLASS}
          />
        )
      case 'nullableText':
        return (
          <input
            type="text"
            value={(value as string | null) ?? ''}
            placeholder={field.placeholder}
            spellCheck={false}
            onChange={(e) => onChange(e.target.value.trim() === '' ? null : e.target.value)}
            className={INPUT_CLASS}
          />
        )
      case 'toggle':
        return (
          <button
            type="button"
            onClick={() => onChange(!(value as boolean))}
            className={cn(
              'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
              value ? 'bg-blue-600' : 'bg-zinc-700',
            )}
          >
            <span
              className={cn(
                'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                value ? 'translate-x-6' : 'translate-x-1',
              )}
            />
          </button>
        )
      case 'triBool':
        return <TriBool value={(value as boolean | null) ?? null} onChange={onChange} />
      case 'enum':
        return (
          <select
            value={String(value)}
            onChange={(e) => onChange(e.target.value)}
            className={INPUT_CLASS}
          >
            {field.options?.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        )
      case 'nullableEnum':
        return (
          <select
            value={value === null || value === undefined ? AUTO_SENTINEL : String(value)}
            onChange={(e) => onChange(e.target.value === AUTO_SENTINEL ? null : e.target.value)}
            className={INPUT_CLASS}
          >
            <option value={AUTO_SENTINEL}>Auto (from preset)</option>
            {field.options?.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        )
      case 'stringList':
        return (
          <textarea
            value={((value as string[]) ?? []).join('\n')}
            spellCheck={false}
            rows={3}
            onChange={(e) =>
              onChange(
                e.target.value
                  .split('\n')
                  .map((s) => s.trim())
                  .filter((s) => s.length > 0),
              )
            }
            className={cn(INPUT_CLASS, 'resize-y font-mono text-[12px]')}
          />
        )
      case 'intList':
        return (
          <input
            type="text"
            value={((value as number[]) ?? []).join(', ')}
            spellCheck={false}
            onChange={(e) =>
              onChange(
                e.target.value
                  .split(',')
                  .map((s) => Number(s.trim()))
                  .filter((n) => Number.isFinite(n)),
              )
            }
            className={INPUT_CLASS}
          />
        )
    }
  })()

  // Toggles read better with the control inline next to the label.
  if (field.control === 'toggle') {
    return (
      <div className="flex items-center justify-between gap-3 py-0.5">
        <FieldLabel label={field.label} help={field.help} />
        {control}
      </div>
    )
  }

  return (
    <div className="space-y-1.5">
      <FieldLabel label={field.label} help={field.help} />
      {control}
    </div>
  )
}

/** A titled, collapsible group of config fields. */
export function CollapsibleSection({
  title,
  defaultOpen = false,
  children,
}: {
  title: string
  defaultOpen?: boolean
  children: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-zinc-800 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2.5 bg-zinc-800/50 hover:bg-zinc-800 transition-colors"
      >
        <span className="text-xs font-semibold text-zinc-200">{title}</span>
        <ChevronDown
          className={cn('h-4 w-4 text-zinc-400 transition-transform', open && 'rotate-180')}
        />
      </button>
      {open && <div className="px-3 py-3 space-y-3">{children}</div>}
    </div>
  )
}
