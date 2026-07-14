import React, { useState, useRef, useCallback, useEffect, useLayoutEffect } from 'react'
import ReactDOM from 'react-dom'
import { cn } from '@/lib/utils'

interface TooltipProps {
  content: React.ReactNode
  children: React.ReactNode
  /** Which side of the trigger the tooltip appears on. Default: 'top' */
  side?: 'top' | 'bottom' | 'left' | 'right'
  className?: string
  /**
   * Allow multi-line, wrapped help text. Drops `whitespace-nowrap` and caps
   * the width so long descriptions (e.g. config field help) read as a small
   * paragraph instead of one runaway line.
   */
  wide?: boolean
}

const DELAY_MS = 500
const GAP_PX = 6
const VIEWPORT_MARGIN_PX = 8

/**
 * Styled tooltip with 500ms show delay and instant hide.
 * Renders via a portal into document.body so it is never clipped
 * by overflow-hidden ancestors.
 */
export function Tooltip({ content, children, side = 'top', className, wide = false }: TooltipProps) {
  const [visible, setVisible] = useState(false)
  const [style, setStyle] = useState<React.CSSProperties>({})
  const wrapperRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  const computeStyle = useCallback(() => {
    const triggerRect = wrapperRef.current?.getBoundingClientRect()
    const tooltipRect = tooltipRef.current?.getBoundingClientRect()
    if (!triggerRect || !tooltipRect) return

    const viewportWidth = window.innerWidth
    const viewportHeight = window.innerHeight
    const spaces = {
      top: triggerRect.top - VIEWPORT_MARGIN_PX,
      bottom: viewportHeight - triggerRect.bottom - VIEWPORT_MARGIN_PX,
      left: triggerRect.left - VIEWPORT_MARGIN_PX,
      right: viewportWidth - triggerRect.right - VIEWPORT_MARGIN_PX,
    }
    let resolvedSide = side
    if (side === 'top' && spaces.top < tooltipRect.height + GAP_PX && spaces.bottom > spaces.top) {
      resolvedSide = 'bottom'
    } else if (side === 'bottom' && spaces.bottom < tooltipRect.height + GAP_PX && spaces.top > spaces.bottom) {
      resolvedSide = 'top'
    } else if (side === 'left' && spaces.left < tooltipRect.width + GAP_PX && spaces.right > spaces.left) {
      resolvedSide = 'right'
    } else if (side === 'right' && spaces.right < tooltipRect.width + GAP_PX && spaces.left > spaces.right) {
      resolvedSide = 'left'
    }

    let left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2
    let top = triggerRect.top + (triggerRect.height - tooltipRect.height) / 2
    switch (resolvedSide) {
      case 'top':
        top = triggerRect.top - tooltipRect.height - GAP_PX
        break
      case 'bottom':
        top = triggerRect.bottom + GAP_PX
        break
      case 'left':
        left = triggerRect.left - tooltipRect.width - GAP_PX
        break
      case 'right':
        left = triggerRect.right + GAP_PX
        break
    }
    left = Math.min(
      Math.max(left, VIEWPORT_MARGIN_PX),
      Math.max(VIEWPORT_MARGIN_PX, viewportWidth - tooltipRect.width - VIEWPORT_MARGIN_PX),
    )
    top = Math.min(
      Math.max(top, VIEWPORT_MARGIN_PX),
      Math.max(VIEWPORT_MARGIN_PX, viewportHeight - tooltipRect.height - VIEWPORT_MARGIN_PX),
    )
    setStyle({ left, top, visibility: 'visible' })
  }, [side])

  const handleMouseEnter = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      timerRef.current = null
      setVisible(true)
    }, DELAY_MS)
  }, [])

  const handleMouseLeave = useCallback(() => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null }
    setVisible(false)
    setStyle({})
  }, [])

  useLayoutEffect(() => {
    if (visible) computeStyle()
  }, [visible, content, wide, computeStyle])

  useEffect(() => {
    if (!visible) return
    const schedulePositionUpdate = () => {
      if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = requestAnimationFrame(() => {
        animationFrameRef.current = null
        computeStyle()
      })
    }
    window.addEventListener('resize', schedulePositionUpdate)
    window.addEventListener('scroll', schedulePositionUpdate, true)
    return () => {
      window.removeEventListener('resize', schedulePositionUpdate)
      window.removeEventListener('scroll', schedulePositionUpdate, true)
      if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current)
    }
  }, [visible, computeStyle])

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current)
    if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current)
  }, [])

  return (
    <div
      ref={wrapperRef}
      className={cn('inline-flex', className)}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleMouseEnter}
      onBlur={handleMouseLeave}
    >
      {children}
      {visible && ReactDOM.createPortal(
        <div
          ref={tooltipRef}
          role="tooltip"
          className={cn(
            'fixed z-[99999] px-2.5 py-1.5 bg-white text-zinc-800 text-xs font-medium rounded-md shadow-md pointer-events-none select-none',
            wide ? 'max-w-[260px] leading-snug whitespace-normal text-left' : 'whitespace-nowrap',
          )}
          style={{ visibility: 'hidden', ...style }}
        >
          {content}
        </div>,
        document.body,
      )}
    </div>
  )
}
