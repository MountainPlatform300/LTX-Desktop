import fs from 'fs'
import path from 'path'

const isWindows = process.platform === 'win32'

function canonicalize(p: string): string {
  let existing = path.resolve(p)
  const missingParts: string[] = []
  while (!fs.existsSync(existing)) {
    const parent = path.dirname(existing)
    if (parent === existing) break
    missingParts.unshift(path.basename(existing))
    existing = parent
  }
  const canonicalExisting = fs.existsSync(existing)
    ? fs.realpathSync.native(existing)
    : existing
  return path.resolve(canonicalExisting, ...missingParts)
}

function normalize(p: string): string {
  const canonical = canonicalize(p)
  return isWindows ? canonical.toLowerCase() : canonical
}

function stripFileUrl(fileUrl: string): string {
  let raw = fileUrl
  if (raw.startsWith('file:///')) raw = raw.slice(8)
  else if (raw.startsWith('file://')) raw = raw.slice(7)
  return decodeURIComponent(raw).replace(/\//g, path.sep)
}

const approvedPaths = new Map<string, boolean>()

export function approvePath(filePath: string): void {
  const canonical = canonicalize(filePath)
  approvedPaths.set(
    normalize(canonical),
    fs.existsSync(canonical) && fs.statSync(canonical).isDirectory(),
  )
}

export function validatePath(inputPath: string, allowedRoots: string[]): string {
  const cleaned = inputPath.startsWith('file://') ? stripFileUrl(inputPath) : inputPath
  const resolved = canonicalize(cleaned)
  const norm = normalize(resolved)

  for (const root of allowedRoots.map(normalize)) {
    if (norm === root || norm.startsWith(root + path.sep)) return resolved
  }

  let found = false
  approvedPaths.forEach((isDirectory, approved) => {
    if (norm === approved || (isDirectory && norm.startsWith(approved + path.sep))) {
      found = true
    }
  })
  if (found) return resolved

  throw new Error(`Path not allowed: ${inputPath}`)
}
