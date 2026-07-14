import { useMemo, useState } from 'react'
import { Copy, Plus, Trash2, X } from 'lucide-react'
import { Button } from '../ui/button'
import { confirmAction } from '../ui/confirm-dialog'
import { cn } from '@/lib/utils'
import {
  useLoraTraining,
  type LoraProfile,
  type LoraTrainingConfig,
} from '../../contexts/LoraTrainingContext'
import { ConfigField, CollapsibleSection } from './TrainingConfigControls'
import { PRIMARY_FIELDS, SECTIONS, SECTION_FIELDS } from './trainingConfigFields'

type DatasetType = NonNullable<LoraProfile['datasetTypes']>[number]
type Draft = {
  name: string
  description: string
  datasetTypes: DatasetType[]
  config: LoraTrainingConfig
}

// `preset` is carried through unchanged (it only seeds the acceleration
// fallback); the editor surfaces every other tunable knob.
function toDraft(profile: LoraProfile): Draft {
  return {
    name: profile.name,
    description: profile.description,
    datasetTypes: [...(profile.datasetTypes ?? ['standard', 'ic_lora'])],
    config: { ...profile.config },
  }
}

/**
 * Manage reusable training profiles: a left rail of saved profiles plus a
 * grouped, tooltip-rich editor for the selected one. Creating starts from an
 * existing profile as a template (so the full default config is inherited from
 * the server-seeded built-ins).
 */
export function TrainingProfileEditor({ onClose }: { onClose: () => void }) {
  const { profiles, createProfile, updateProfile, deleteProfile } = useLoraTraining()

  const [selectedId, setSelectedId] = useState<string | null>(profiles[0]?.id ?? null)
  const [draft, setDraft] = useState<Draft | null>(profiles[0] ? toDraft(profiles[0]) : null)
  // null selectedId + non-null draft => unsaved "new profile".
  const isCreating = selectedId === null && draft !== null
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const selectedProfile = profiles.find((profile) => profile.id === selectedId) ?? null
  const readonlyBuiltin = selectedProfile?.builtin ?? false

  const selectProfile = (profile: LoraProfile) => {
    setSelectedId(profile.id)
    setDraft(toDraft(profile))
    setError(null)
  }

  const startNew = () => {
    const template = profiles.find((p) => p.id === selectedId) ?? profiles[0]
    setSelectedId(null)
    setDraft({
      name: 'New profile',
      description: template?.description ?? '',
      datasetTypes: template
        ? [...(template.datasetTypes ?? ['standard', 'ic_lora'])]
        : ['standard', 'ic_lora'],
      config: template ? { ...template.config } : ({} as LoraTrainingConfig),
    })
    setError(null)
  }

  const reseedFrom = (profile: LoraProfile) => {
    setDraft((d) => (d ? { ...d, config: { ...profile.config } } : d))
  }

  const setConfigValue = (key: string, value: unknown) => {
    setDraft((d) => (d ? { ...d, config: { ...d.config, [key]: value } } : d))
  }

  const setName = (name: string) => setDraft((d) => (d ? { ...d, name } : d))

  const save = async () => {
    if (!draft) return
    if (!draft.name.trim()) {
      setError('Give this profile a name.')
      return
    }
    setSaving(true)
    setError(null)
    const result = isCreating
      ? await createProfile({
          name: draft.name.trim(),
          description: draft.description.trim(),
          datasetTypes: draft.datasetTypes,
          config: draft.config,
        })
      : await updateProfile(selectedId as string, {
          name: draft.name.trim(),
          description: draft.description.trim(),
          datasetTypes: draft.datasetTypes,
          config: draft.config,
        })
    setSaving(false)
    if (!result.ok) {
      setError(result.error)
      return
    }
    setSelectedId(result.data.id)
    setDraft(toDraft(result.data))
  }

  const duplicate = async () => {
    if (!draft) return
    setSaving(true)
    setError(null)
    const result = await createProfile({
      name: `${draft.name} copy`,
      description: draft.description,
      datasetTypes: draft.datasetTypes,
      config: draft.config,
    })
    setSaving(false)
    if (!result.ok) {
      setError(result.error)
      return
    }
    setSelectedId(result.data.id)
    setDraft(toDraft(result.data))
  }

  const remove = async () => {
    if (selectedId === null) return
    if (!await confirmAction({
      title: `Delete “${draft?.name ?? 'profile'}”?`,
      message: 'Existing runs keep their snapshotted settings, but this custom profile cannot be restored.',
      confirmLabel: 'Delete profile',
      variant: 'destructive',
    })) return
    await deleteProfile(selectedId)
    const next = profiles.find((p) => p.id !== selectedId) ?? null
    if (next) selectProfile(next)
    else {
      setSelectedId(null)
      setDraft(null)
    }
  }

  const templates = useMemo(() => profiles.filter((p) => p.id !== selectedId), [profiles, selectedId])
  const profileGroups = useMemo(() => [
    {
      label: 'Standard LoRA',
      profiles: profiles.filter((profile) =>
        (profile.datasetTypes ?? ['standard', 'ic_lora']).includes('standard'),
      ),
    },
    {
      label: 'IC-LoRA',
      profiles: profiles.filter(
        (profile) =>
          (profile.datasetTypes ?? ['standard', 'ic_lora']).includes('ic_lora')
          && !(profile.datasetTypes ?? ['standard', 'ic_lora']).includes('standard'),
      ),
    },
  ], [profiles])

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" onKeyDown={(e) => e.stopPropagation()}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-3xl mx-4 flex flex-col max-h-[82vh]">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="text-base font-semibold text-white">Training profiles</h2>
          <button onClick={onClose} className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex flex-1 min-h-0">
          {/* Profile list */}
          <div className="w-52 shrink-0 border-r border-zinc-800 flex flex-col">
            <button
              onClick={startNew}
              className="flex items-center gap-1.5 m-2 px-2.5 py-2 rounded-lg text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white justify-center"
            >
              <Plus className="h-3.5 w-3.5" /> New profile
            </button>
            <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-1">
              {profileGroups.map((group) => group.profiles.length > 0 && (
                <div key={group.label} className="space-y-1">
                  <div className="px-2.5 pt-2 text-[9px] font-semibold uppercase tracking-wider text-zinc-600">
                    {group.label}
                  </div>
                  {group.profiles.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => selectProfile(p)}
                      className={cn(
                        'w-full text-left px-2.5 py-2 rounded-lg text-xs transition-colors',
                        selectedId === p.id
                          ? 'bg-zinc-800 text-white'
                          : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50',
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate">{p.name}</span>
                        {p.builtin && <span className="text-[9px] text-zinc-500 shrink-0">built-in</span>}
                      </div>
                      {p.minVramGb != null && (
                        <span className="mt-0.5 block text-[9px] text-amber-400/80">{p.minVramGb} GB+ GPU</span>
                      )}
                    </button>
                  ))}
                </div>
              ))}
              {isCreating && (
                <div className="px-2.5 py-2 rounded-lg text-xs bg-zinc-800 text-white">
                  {draft?.name || 'New profile'} <span className="text-[10px] text-blue-400">· unsaved</span>
                </div>
              )}
            </div>
          </div>

          {/* Editor */}
          {draft ? (
            <div className="flex-1 min-w-0 flex flex-col">
              <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-300">Profile name</label>
                  <input
                    value={draft.name}
                    onChange={(e) => setName(e.target.value)}
                    disabled={readonlyBuiltin}
                    placeholder="e.g. Character LoRA"
                    className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-default disabled:opacity-70"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-300">Purpose</label>
                  <textarea
                    value={draft.description}
                    onChange={(event) => setDraft((current) => current ? { ...current, description: event.target.value } : current)}
                    disabled={readonlyBuiltin}
                    rows={2}
                    className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-xs text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-default disabled:opacity-70"
                  />
                </div>

                <fieldset disabled={readonlyBuiltin} className="space-y-1.5">
                  <legend className="text-xs font-medium text-zinc-300">Compatible datasets</legend>
                  <div className="flex gap-2">
                    {([
                      ['standard', 'Standard LoRA'],
                      ['ic_lora', 'IC-LoRA'],
                    ] as const).map(([type, label]) => (
                      <label key={type} className="flex items-center gap-1.5 text-xs text-zinc-300">
                        <input
                          type="checkbox"
                          checked={draft.datasetTypes.includes(type)}
                          onChange={(event) => setDraft((current) => {
                            if (!current) return current
                            const next = event.target.checked
                              ? [...current.datasetTypes, type]
                              : current.datasetTypes.filter((value) => value !== type)
                            return next.length > 0 ? { ...current, datasetTypes: next } : current
                          })}
                        />
                        {label}
                      </label>
                    ))}
                  </div>
                </fieldset>

                {readonlyBuiltin && (
                  <p className="rounded-lg border border-blue-500/20 bg-blue-500/5 px-3 py-2 text-[11px] text-blue-200/80">
                    Curated built-ins are read-only so Auto stays reliable. Duplicate this profile to customize it.
                  </p>
                )}

                {isCreating && templates.length > 0 && (
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-zinc-300">Start from</label>
                    <select
                      onChange={(e) => {
                        const t = templates.find((p) => p.id === e.target.value)
                        if (t) reseedFrom(t)
                      }}
                      defaultValue=""
                      className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="" disabled>Copy settings from…</option>
                      {templates.map((p) => (
                        <option key={p.id} value={p.id}>{p.name}</option>
                      ))}
                    </select>
                  </div>
                )}

                <fieldset disabled={readonlyBuiltin} className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    {PRIMARY_FIELDS.map((field) => (
                      <ConfigField
                        key={field.key}
                        field={field}
                        value={draft.config[field.key]}
                        onChange={(v) => setConfigValue(field.key, v)}
                      />
                    ))}
                  </div>

                  {SECTIONS.map((section) => (
                    <CollapsibleSection key={section.id} title={section.title} defaultOpen={section.defaultOpen}>
                      {SECTION_FIELDS[section.id].map((field) => (
                        <ConfigField
                          key={field.key}
                          field={field}
                          value={draft.config[field.key]}
                          onChange={(v) => setConfigValue(field.key, v)}
                        />
                      ))}
                    </CollapsibleSection>
                  ))}
                </fieldset>

                {error && <p className="text-xs text-red-400">{error}</p>}
              </div>

              <div className="px-5 py-4 border-t border-zinc-800 flex items-center justify-between gap-2">
                <div className="flex gap-2">
                  {!isCreating && (
                    <>
                      <Button variant="outline" className="border-zinc-700" onClick={() => void duplicate()} disabled={saving}>
                        <Copy className="h-3.5 w-3.5 mr-1.5" /> Duplicate
                      </Button>
                      {!readonlyBuiltin && (
                        <Button variant="outline" className="border-zinc-700 text-red-400 hover:text-red-300" onClick={() => void remove()} disabled={saving}>
                          <Trash2 className="h-3.5 w-3.5 mr-1.5" /> Delete
                        </Button>
                      )}
                    </>
                  )}
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" className="border-zinc-700" onClick={onClose}>Close</Button>
                  {!readonlyBuiltin && (
                    <Button className="bg-blue-600 hover:bg-blue-500" onClick={() => void save()} disabled={saving}>
                      {saving ? 'Saving…' : isCreating ? 'Create profile' : 'Save changes'}
                    </Button>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-sm text-zinc-500">
              Select a profile or create a new one.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
