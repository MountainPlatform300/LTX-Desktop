import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const projectDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const outputPath = join(projectDir, 'python-deps-hash.txt')
const uvVersion = '0.11.25'
const baseInputs = [
  'backend/.python-version',
  'backend/pyproject.toml',
  'backend/uv.lock',
  'scripts/python-build-constraints.txt',
]

const args = new Set(process.argv.slice(2))
const platformArg = process.argv.find((arg) => arg.startsWith('--platform='))
const archArg = process.argv.find((arg) => arg.startsWith('--arch='))
const platform = platformArg?.slice('--platform='.length) || process.platform
const arch = archArg?.slice('--arch='.length) || process.arch
const inputs = [
  ...baseInputs,
  platform === 'win32'
    ? 'scripts/prepare-python.ps1'
    : 'scripts/prepare-python.sh',
]

const hash = createHash('sha256')
hash.update('ltx-desktop-python-environment-v1\0')
hash.update(`${platform}\0${arch}\0`)
hash.update(`uv\0${uvVersion}\0`)

for (const input of inputs) {
  const absolutePath = join(projectDir, input)
  hash.update(`${relative(projectDir, absolutePath).replaceAll('\\', '/')}\0`)
  hash.update(readFileSync(absolutePath))
  hash.update('\0')
}

const expected = `${hash.digest('hex')}\n`

if (args.has('--check')) {
  let actual = ''
  try {
    actual = readFileSync(outputPath, 'utf8')
  } catch {
    console.error('python-deps-hash.txt is missing. Run pnpm prepare:python or this script.')
    process.exit(1)
  }
  if (actual !== expected) {
    console.error(`python-deps-hash.txt is stale for ${platform}/${arch}. Regenerate it before packaging.`)
    process.exit(1)
  }
  console.log(`Verified Python dependency hash for ${platform}/${arch}: ${expected.trim()}`)
} else {
  writeFileSync(outputPath, expected, 'utf8')
  console.log(`Generated Python dependency hash for ${platform}/${arch}: ${expected.trim()}`)
}
