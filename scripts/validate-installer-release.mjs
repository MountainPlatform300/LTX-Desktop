import { createHash } from 'node:crypto'
import {
  createReadStream,
  readFileSync,
  readdirSync,
  statSync,
} from 'node:fs'
import { basename, join, resolve } from 'node:path'

const directory = resolve(process.argv[2] || 'release/evidence')
const version = process.argv[3]

if (!version || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
  throw new Error('Usage: node scripts/validate-installer-release.mjs <directory> <version>')
}

const maxAssetBytes = 2 * 1024 * 1024 * 1024
const expectedFiles = new Set([
  'LTX-Desktop-Setup.exe',
  'LTX-Desktop-arm64.dmg',
  'LTX-Desktop-x64.AppImage',
  'LTX-Desktop-x64.deb',
  'RELEASE_NOTES.md',
  'node-licenses.json',
  'python-requirements.txt',
  'sbom.spdx.json',
  `ltx-desktop-${version}.tar.gz`,
])

const runtimeContracts = [
  {
    manifestName: 'python-embed-win32.manifest.json',
    prefix: 'python-embed-win32',
    platform: 'win32',
    arch: 'x64',
  },
  {
    manifestName: 'python-embed-linux-x64.manifest.json',
    prefix: 'python-embed-linux-x64',
    platform: 'linux',
    arch: 'x64',
  },
]

async function hashFile(filePath, aggregateHash) {
  const hash = createHash('sha256')
  let size = 0
  for await (const chunk of createReadStream(filePath)) {
    hash.update(chunk)
    aggregateHash?.update(chunk)
    size += chunk.length
  }
  return { sha256: hash.digest('hex'), size }
}

for (const contract of runtimeContracts) {
  expectedFiles.add(contract.manifestName)
  const manifestPath = join(directory, contract.manifestName)
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'))
  if (
    manifest.schemaVersion !== 1
    || manifest.platform !== contract.platform
    || manifest.arch !== contract.arch
    || typeof manifest.depsHash !== 'string'
    || !/^[a-f0-9]{64}$/.test(manifest.depsHash)
    || !Array.isArray(manifest.parts)
    || manifest.parts.length === 0
  ) {
    throw new Error(`Invalid runtime manifest contract: ${contract.manifestName}`)
  }

  const archiveHash = createHash('sha256')
  let totalSize = 0
  for (const [index, part] of manifest.parts.entries()) {
    const expectedName = `${contract.prefix}.part${String(index + 1).padStart(3, '0')}`
    if (
      part.name !== expectedName
      || !Number.isSafeInteger(part.size)
      || part.size <= 0
      || part.size >= maxAssetBytes
      || typeof part.sha256 !== 'string'
      || !/^[a-f0-9]{64}$/.test(part.sha256)
    ) {
      throw new Error(`Invalid runtime part declaration: ${part.name ?? expectedName}`)
    }
    expectedFiles.add(part.name)
    const actual = await hashFile(join(directory, part.name), archiveHash)
    if (actual.size !== part.size || actual.sha256 !== part.sha256) {
      throw new Error(`Runtime part integrity mismatch: ${part.name}`)
    }
    totalSize += actual.size
  }

  if (
    totalSize !== manifest.totalSize
    || archiveHash.digest('hex') !== manifest.archiveSha256
  ) {
    throw new Error(`Runtime archive integrity mismatch: ${contract.manifestName}`)
  }
}

const actualFiles = readdirSync(directory, { withFileTypes: true })
  .filter((entry) => entry.isFile())
  .map((entry) => entry.name)
  .sort()
const unexpected = actualFiles.filter((name) => !expectedFiles.has(name))
const missing = [...expectedFiles].filter((name) => !actualFiles.includes(name))
if (unexpected.length || missing.length) {
  throw new Error(
    `Release asset contract mismatch. Missing: ${missing.join(', ') || 'none'}. `
    + `Unexpected: ${unexpected.join(', ') || 'none'}.`,
  )
}

for (const name of actualFiles) {
  const size = statSync(join(directory, name)).size
  if (size <= 0 || size >= maxAssetBytes) {
    throw new Error(`Release asset has invalid size: ${basename(name)} (${size} bytes)`)
  }
}

console.log(`Validated ${actualFiles.length} installer release assets and both runtime archives.`)
