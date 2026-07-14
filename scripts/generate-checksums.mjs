import { createHash } from 'node:crypto'
import {
  createReadStream,
  readdirSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { join, relative, resolve } from 'node:path'

const directory = resolve(process.argv[2] || 'release/evidence')
const output = resolve(process.argv[3] || join(directory, 'SHA256SUMS'))

function listFiles(current) {
  return readdirSync(current, { withFileTypes: true })
    .flatMap((entry) => {
      const fullPath = join(current, entry.name)
      return entry.isDirectory() ? listFiles(fullPath) : [fullPath]
    })
}

function sha256(filePath) {
  return new Promise((resolveHash, rejectHash) => {
    const hash = createHash('sha256')
    const stream = createReadStream(filePath)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.on('error', rejectHash)
    stream.on('end', () => resolveHash(hash.digest('hex')))
  })
}

const files = listFiles(directory)
  .filter((filePath) => resolve(filePath) !== output && statSync(filePath).isFile())
  .sort()
const lines = []
for (const filePath of files) {
  const name = relative(directory, filePath).replaceAll('\\', '/')
  lines.push(`${await sha256(filePath)}  ${name}`)
}
writeFileSync(output, `${lines.join('\n')}\n`, 'utf8')
console.log(`Wrote SHA-256 checksums for ${files.length} files to ${output}`)
