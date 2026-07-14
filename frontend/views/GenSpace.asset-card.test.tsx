import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Asset } from '@/types/project-model'
import { AssetCard } from './GenSpace'

afterEach(cleanup)

describe('Gen Space image asset card', () => {
  it('starts image editing without opening the asset viewer', () => {
    const asset = {
      id: 'image-1',
      type: 'image',
      path: 'C:\\images\\source.png',
    } as Asset
    const onEditImage = vi.fn()
    const onPlay = vi.fn()
    const { container } = render(
      <AssetCard
        asset={asset}
        onDelete={vi.fn()}
        onPlay={onPlay}
        onDragStart={vi.fn()}
        onEditImage={onEditImage}
      />,
    )

    fireEvent.mouseEnter(container.firstElementChild!)
    fireEvent.click(screen.getByRole('button', { name: 'Edit image' }))

    expect(onEditImage).toHaveBeenCalledWith(asset)
    expect(onPlay).not.toHaveBeenCalled()
  })
})
