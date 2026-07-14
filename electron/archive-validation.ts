import path from 'path'

export function validatePythonArchiveEntries(listing: string): void {
  const entries = listing.split(/\r?\n/).filter(Boolean)
  if (entries.length === 0) {
    throw new Error('Python archive is empty')
  }

  for (const rawEntry of entries) {
    if (rawEntry.includes('\0')) {
      throw new Error('Python archive contains an invalid NUL byte')
    }
    const entry = rawEntry.replaceAll('\\', '/')
    if (
      entry.startsWith('/')
      || /^[A-Za-z]:/.test(entry)
      || entry.split('/').includes('..')
    ) {
      throw new Error(`Unsafe Python archive path: ${rawEntry}`)
    }
    const normalized = path.posix.normalize(entry).replace(/\/$/, '')
    if (normalized !== 'python-embed' && !normalized.startsWith('python-embed/')) {
      throw new Error(`Unexpected Python archive root: ${rawEntry}`)
    }
  }
}

export function validatePythonArchiveTypes(verboseListing: string): void {
  for (const line of verboseListing.split(/\r?\n/).filter(Boolean)) {
    const type = line[0]
    if (type !== '-' && type !== 'd') {
      throw new Error('Python archive may contain only regular files and directories')
    }
  }
}
