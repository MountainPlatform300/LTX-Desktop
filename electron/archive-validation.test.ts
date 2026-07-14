import { describe, expect, it } from 'vitest'
import {
  validatePythonArchiveEntries,
  validatePythonArchiveTypes,
} from './archive-validation'

describe('Python archive validation', () => {
  it('accepts files under the expected root', () => {
    expect(() => validatePythonArchiveEntries(
      'python-embed/\npython-embed/python.exe\npython-embed/Lib/site-packages/example.py\n',
    )).not.toThrow()
  })

  it.each([
    '../outside',
    'python-embed/../../outside',
    '/absolute/path',
    'C:/absolute/path',
    'different-root/file.py',
  ])('rejects unsafe entry %s', (entry) => {
    expect(() => validatePythonArchiveEntries(entry)).toThrow()
  })

  it('rejects links and special files', () => {
    expect(() => validatePythonArchiveTypes(
      'drwxr-xr-x user/group 0 date python-embed/\n'
      + 'lrwxr-xr-x user/group 0 date python-embed/link -> ../../outside\n',
    )).toThrow('regular files and directories')
  })
})
