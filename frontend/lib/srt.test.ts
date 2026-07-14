import { describe, expect, it } from 'vitest'
import { parseSrt } from './srt'

describe('parseSrt', () => {
  it('extracts inert text from nested and malformed markup', () => {
    const cues = parseSrt(
      '1\n00:00:00,000 --> 00:00:01,000\n<<b>hello</b><img src=x onerror=alert(1)>',
    )

    expect(cues).toHaveLength(1)
    expect(cues[0].text).toBe('<hello')
    expect(cues[0].text).not.toContain('onerror')
  })
})
