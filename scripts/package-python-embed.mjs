import { createHash } from 'node:crypto'
import {
  createReadStream,
  createWriteStream,
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const projectDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const platformArg = process.argv.find((arg) => arg.startsWith('--platform='))
const archArg = process.argv.find((arg) => arg.startsWith('--arch='))
const outputArg = process.argv.find((arg) => arg.startsWith('--output='))
const partSizeArg = process.argv.find((arg) => arg.startsWith('--part-size='))
const platform = platformArg?.slice('--platform='.length) || process.platform
const arch = archArg?.slice('--arch='.length) || process.arch
const outputDir = resolve(projectDir, outputArg?.slice('--output='.length) || 'release/python')
const partSize = Number(partSizeArg?.slice('--part-size='.length) || 1_900_000_000)
const prefix = platform === 'win32'
  ? 'python-embed-win32'
  : `python-embed-${platform}-${arch}`
const archivePath = join(outputDir, `${prefix}.tar.gz`)
const manifestPath = join(outputDir, `${prefix}.manifest.json`)
const depsHash = readFileSync(join(projectDir, 'python-deps-hash.txt'), 'utf8').trim()

if (!Number.isSafeInteger(partSize) || partSize < 1024 * 1024) {
  throw new Error('--part-size must be an integer of at least 1 MiB')
}
if (
  !existsSync(join(projectDir, 'python-embed'))
  || !statSync(join(projectDir, 'python-embed')).isDirectory()
) {
  throw new Error('python-embed directory is missing')
}

mkdirSync(outputDir, { recursive: true })
rmSync(archivePath, { force: true })
rmSync(manifestPath, { force: true })
for (const name of readdirSync(outputDir)) {
  if (name.startsWith(`${prefix}.part`)) {
    rmSync(join(outputDir, name), { force: true })
  }
}

const tar = spawnSync(
  'tar',
  ['-czf', archivePath, '-C', projectDir, 'python-embed'],
  { stdio: 'inherit' },
)
if (tar.status !== 0) {
  throw new Error(`tar failed with exit code ${tar.status ?? 'unknown'}`)
}

const archiveHash = createHash('sha256')
const parts = []
let partIndex = 0
let currentSize = 0
let currentHash
let currentName
let currentStream

function openPart() {
  partIndex += 1
  currentSize = 0
  currentHash = createHash('sha256')
  currentName = `${prefix}.part${String(partIndex).padStart(3, '0')}`
  currentStream = createWriteStream(join(outputDir, currentName), { flags: 'wx' })
}

async function closePart() {
  if (!currentStream) return
  await new Promise((resolvePart, rejectPart) => {
    currentStream.once('error', rejectPart)
    currentStream.end(resolvePart)
  })
  parts.push({
    name: currentName,
    size: currentSize,
    sha256: currentHash.digest('hex'),
  })
  currentStream = undefined
}

for await (const chunk of createReadStream(archivePath)) {
  archiveHash.update(chunk)
  let offset = 0
  while (offset < chunk.length) {
    if (!currentStream) openPart()
    const remaining = partSize - currentSize
    const slice = chunk.subarray(offset, offset + remaining)
    if (!currentStream.write(slice)) {
      await new Promise((resolveDrain) => currentStream.once('drain', resolveDrain))
    }
    currentHash.update(slice)
    currentSize += slice.length
    offset += slice.length
    if (currentSize === partSize) {
      await closePart()
    }
  }
}
await closePart()

const manifest = {
  schemaVersion: 1,
  platform,
  arch,
  depsHash,
  totalSize: statSync(archivePath).size,
  archiveSha256: archiveHash.digest('hex'),
  parts,
}
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8')
writeFileSync(join(outputDir, 'python-deps-hash.txt'), `${depsHash}\n`, 'utf8')
copyFileSync(manifestPath, join(projectDir, 'python-runtime-manifest.json'))
rmSync(archivePath, { force: true })

console.log(`Created ${parts.length} verified parts and ${manifestPath}`)
