import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { LoraInferenceEntry } from '../../hooks/use-lora-inference-registry'
import { LoraPickerPopover } from './LoraPickerPopover'

const ENTRY: LoraInferenceEntry = {
  id: 'style-one',
  name: 'Style One',
  kind: 'imported',
  variant: 'standard',
  available: true,
  conditioningTypes: [],
  localPath: 'C:\\loras\\style-one.safetensors',
  promptTemplate: 'Use Style One',
  promptTemplateCustomized: true,
  triggerWord: 'style-one',
}

afterEach(() => {
  cleanup()
  localStorage.clear()
})

describe('LoraPickerPopover', () => {
  it('exposes a responsive dialog without nested interactive controls', () => {
    const onSelect = vi.fn()
    const rendered = render(
      <LoraPickerPopover
        open
        selectedId={null}
        conditioningType="canny"
        entries={[ENTRY]}
        loading={false}
        error={null}
        onRefresh={vi.fn()}
        onSelect={onSelect}
        onClose={vi.fn()}
      />,
    )

    expect(screen.getByRole('dialog', { name: 'Apply a LoRA' })).toBeTruthy()
    expect(rendered.container.querySelector('button button')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Style One, Style' }))
    expect(onSelect).toHaveBeenCalledWith({ entry: ENTRY, conditioningType: 'canny' })
    expect(screen.getByRole('button', { name: 'Edit prompt template for Style One' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Show Style One in folder' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Move Style One to Trash' })).toBeTruthy()
  })

  it('moves a LoRA to recoverable Trash and restores it', () => {
    render(
      <LoraPickerPopover
        open
        selectedId={ENTRY.id}
        conditioningType="canny"
        entries={[ENTRY]}
        loading={false}
        error={null}
        onRefresh={vi.fn()}
        onSelect={vi.fn()}
        onClose={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Move Style One to Trash' }))
    expect(screen.queryByRole('button', { name: 'Style One, Style' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Open LoRA Trash, 1 items' }))
    expect(screen.getByRole('dialog', { name: 'LoRA Trash' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Restore' }))
    fireEvent.click(screen.getByRole('button', { name: 'Trash' }))
    expect(screen.getByRole('button', { name: 'Style One, Style' })).toBeTruthy()
  })

  it('never rewrites prompt substrings when the name or trigger changes', () => {
    const entry: LoraInferenceEntry = {
      ...ENTRY,
      id: 'single-letter',
      name: 'C',
      triggerWord: 'C',
      promptTemplate: 'Critical rules: always use CLEANPLATE.',
    }
    render(
      <LoraPickerPopover
        open
        selectedId={entry.id}
        conditioningType="canny"
        entries={[entry]}
        loading={false}
        error={null}
        onRefresh={vi.fn()}
        onSelect={vi.fn()}
        onClose={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit prompt template for C' }))
    fireEvent.change(screen.getByLabelText('LoRA name'), { target: { value: 'Clean Plate Test' } })
    fireEvent.change(screen.getByLabelText('Trigger word'), { target: { value: 'CLEANPLATE' } })

    expect((screen.getByLabelText('System prompt') as HTMLTextAreaElement).value)
      .toBe('Critical rules: always use CLEANPLATE.')
  })

  it('keeps the prompt editor recoverable when an IC-LoRA template is missing', () => {
    const entry: LoraInferenceEntry = {
      ...ENTRY,
      id: 'missing-template',
      name: 'Recoverable IC-LoRA',
      variant: 'video_input_ic_lora',
      promptTemplate: null,
      promptTemplateCustomized: false,
    }
    render(
      <LoraPickerPopover
        open
        selectedId={entry.id}
        conditioningType="canny"
        entries={[entry]}
        loading={false}
        error={null}
        onRefresh={vi.fn()}
        onSelect={vi.fn()}
        onClose={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', {
      name: 'Edit prompt template for Recoverable IC-LoRA',
    }))
    expect(screen.getByRole('dialog', { name: 'Edit LoRA prompt' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Regenerate' })).toBeTruthy()
  })

  it('closes on Escape and restores focus to the opener', () => {
    const opener = document.createElement('button')
    document.body.append(opener)
    opener.focus()
    const onClose = vi.fn()
    const rendered = render(
      <LoraPickerPopover
        open
        selectedId={ENTRY.id}
        conditioningType="canny"
        entries={[ENTRY]}
        loading={false}
        error={null}
        onRefresh={vi.fn()}
        onSelect={vi.fn()}
        onClose={onClose}
      />,
    )

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledOnce()
    rendered.unmount()
    expect(opener).toBe(document.activeElement)
    opener.remove()
  })
})
