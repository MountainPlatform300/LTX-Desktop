// @vitest-environment node
import fs from 'fs'
import os from 'os'
import path from 'path'
import { afterEach, describe, expect, it } from 'vitest'
import { approvePath, validatePath } from './path-validation'

const temporaryPaths: string[] = []

function temporaryDirectory(): string {
  const created = fs.mkdtempSync(path.join(os.tmpdir(), 'ltx-path-test-'))
  temporaryPaths.push(created)
  return created
}

afterEach(() => {
  for (const temporary of temporaryPaths.splice(0)) {
    fs.rmSync(temporary, { recursive: true, force: true })
  }
})

describe('path validation', () => {
  it('rejects a symlink that escapes an allowed root', () => {
    const base = temporaryDirectory()
    const allowed = path.join(base, 'allowed')
    const outside = path.join(base, 'outside')
    fs.mkdirSync(allowed)
    fs.mkdirSync(outside)
    fs.writeFileSync(path.join(outside, 'secret.txt'), 'secret')
    fs.symlinkSync(outside, path.join(allowed, 'link'), 'junction')

    expect(() =>
      validatePath(path.join(allowed, 'link', 'secret.txt'), [allowed]),
    ).toThrow('Path not allowed')
  })

  it('does not treat an approved file as an approved directory', () => {
    const base = temporaryDirectory()
    const selectedFile = path.join(base, 'selected.txt')
    fs.writeFileSync(selectedFile, 'selected')
    approvePath(selectedFile)

    expect(validatePath(selectedFile, [])).toBe(fs.realpathSync.native(selectedFile))
    expect(() => validatePath(path.join(selectedFile, 'child'), [])).toThrow(
      'Path not allowed',
    )
  })
})
