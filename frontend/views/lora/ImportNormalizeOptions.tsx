import { useCallback, useState } from 'react'
import { SlidersHorizontal } from 'lucide-react'
import {
  FPS_CHOICES,
  SHORT_SIDE_CHOICES,
  type ImportNormalizeSpec,
} from '../../lib/lora-import-normalize'
import { loadLoraUiPreferences, saveLoraUiPreferences } from '../../lib/lora-ui-persistence'

/** Remembers the user's last import-normalize choices across modals/sessions. */
export function useImportNormalizeSpec(): [ImportNormalizeSpec, (next: ImportNormalizeSpec) => void] {
  const [spec, setSpec] = useState<ImportNormalizeSpec>(
    () => loadLoraUiPreferences().importNormalize,
  )
  const update = useCallback((next: ImportNormalizeSpec) => {
    setSpec(next)
    saveLoraUiPreferences({ importNormalize: next })
  }, [])
  return [spec, update]
}

function Row({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center gap-2 min-h-[28px]">{children}</div>
}

/**
 * Compact "offer to normalize" panel for the import modals. All knobs default
 * off; enabling one re-encodes matching clips on import (trim long clips,
 * downscale to a short-side, force one fps). Resizing preserves aspect ratio
 * and never upscales; originals are kept (a normalized copy is used).
 */
export function ImportNormalizeOptions({
  value,
  onChange,
  disabled = false,
}: {
  value: ImportNormalizeSpec
  onChange: (next: ImportNormalizeSpec) => void
  disabled?: boolean
}) {
  const inputCls =
    'px-2 py-1 bg-zinc-800 border border-zinc-700 rounded-md text-xs text-white focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50'

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-800/30 p-3 space-y-2">
      <div className="flex items-center gap-1.5">
        <SlidersHorizontal className="h-3.5 w-3.5 text-zinc-400" />
        <span className="text-xs font-medium text-zinc-300">Normalize on import (optional)</span>
      </div>

      <Row>
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            className="h-3.5 w-3.5 accent-blue-500"
            checked={value.trim.enabled}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, trim: { ...value.trim, enabled: e.target.checked } })}
          />
          <span className="text-xs text-zinc-300">Trim to first</span>
        </label>
        <input
          type="number"
          min={1}
          step={1}
          value={value.trim.maxSeconds}
          disabled={disabled || !value.trim.enabled}
          onChange={(e) =>
            onChange({ ...value, trim: { ...value.trim, maxSeconds: Number(e.target.value) || 0 } })
          }
          className={`${inputCls} w-16`}
        />
        <span className="text-xs text-zinc-500">seconds</span>
      </Row>

      <Row>
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            className="h-3.5 w-3.5 accent-blue-500"
            checked={value.resolution.enabled}
            disabled={disabled}
            onChange={(e) =>
              onChange({ ...value, resolution: { ...value.resolution, enabled: e.target.checked } })
            }
          />
          <span className="text-xs text-zinc-300">Resize short side to</span>
        </label>
        <select
          value={value.resolution.shortSide}
          disabled={disabled || !value.resolution.enabled}
          onChange={(e) =>
            onChange({ ...value, resolution: { ...value.resolution, shortSide: Number(e.target.value) } })
          }
          className={inputCls}
        >
          {SHORT_SIDE_CHOICES.map((s) => (
            <option key={s} value={s}>
              {s}px
            </option>
          ))}
        </select>
      </Row>

      <Row>
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            className="h-3.5 w-3.5 accent-blue-500"
            checked={value.fps.enabled}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, fps: { ...value.fps, enabled: e.target.checked } })}
          />
          <span className="text-xs text-zinc-300">Set frame rate to</span>
        </label>
        <select
          value={value.fps.value}
          disabled={disabled || !value.fps.enabled}
          onChange={(e) => onChange({ ...value, fps: { ...value.fps, value: Number(e.target.value) } })}
          className={inputCls}
        >
          {FPS_CHOICES.map((f) => (
            <option key={f} value={f}>
              {f} fps
            </option>
          ))}
        </select>
      </Row>

      <p className="text-[11px] text-zinc-500 leading-relaxed">
        Resizing preserves aspect ratio and never upscales. Originals are kept — a normalized copy is
        used in the collection.
      </p>
    </div>
  )
}
