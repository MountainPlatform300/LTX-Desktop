// @vitest-environment node
import { describe, expect, it } from 'vitest'
import { escapeDrawtextText } from './video-filter'

describe('escapeDrawtextText', () => {
  it('cannot close the quoted drawtext value or inject filter separators', () => {
    const escaped = escapeDrawtextText(
      "caption';textfile=C:\\secret,[in]evil[out]\nnext",
    )

    expect(escaped).not.toContain("'")
    expect(escaped).toContain('caption’\\;textfile=C\\:')
    expect(escaped).toContain('\\\\\\\\secret\\,\\[in\\]evil\\[out\\]')
    expect(escaped).toContain('\\nnext')
  })
})
