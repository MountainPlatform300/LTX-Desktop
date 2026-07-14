import { execFile } from 'child_process'
import { createHash } from 'crypto'
import { app } from 'electron'
import fs from 'fs'
import http from 'http'
import https from 'https'
import { load as loadYaml } from 'js-yaml'
import path from 'path'
import { validatePythonArchiveEntries, validatePythonArchiveTypes } from './archive-validation'
import { isDev } from './config'
import { logger } from './logger'

export interface PythonSetupProgress {
  status: 'downloading' | 'extracting' | 'installing-extras' | 'complete' | 'error'
  percent: number
  downloadedBytes: number
  totalBytes: number
  speed: number
}

interface ArchiveManifest {
  schemaVersion: 1
  platform: string
  arch: string
  depsHash: string
  parts: { name: string; size: number; sha256: string }[]
  totalSize: number
  archiveSha256: string
}

// ── GitHub private repo authentication ────────────────────────────────
// Mirrors electron-updater: only sends GH_TOKEN when `private: true` is set
// in the publish config (app-update.yml). This prevents accidental token leaks
// for public repos.

let _authHeaders: Record<string, string> | null = null

function getAuthHeaders(url: URL): Record<string, string> {
  if (url.hostname !== 'github.com') return {}
  if (_authHeaders !== null) return _authHeaders

  _authHeaders = {}

  const configPath = isDev
    ? path.join(process.cwd(), 'dev-app-update.yml')
    : path.join(process.resourcesPath, 'app-update.yml')

  let isPrivate = false
  try {
    const config = loadYaml(fs.readFileSync(configPath, 'utf-8')) as Record<string, unknown>
    isPrivate = config?.private === true
  } catch { /* no config file — public repo */ }

  if (isPrivate) {
    const token = process.env.GH_TOKEN || process.env.GITHUB_TOKEN
    if (token) {
      _authHeaders = { authorization: `token ${token}` }
    }
  }

  return _authHeaders
}

function getBundledHashPath(): string {
  if (isDev) {
    return path.join(process.cwd(), 'python-deps-hash.txt')
  }
  return path.join(process.resourcesPath, 'python-deps-hash.txt')
}

function getBundledManifestPath(): string {
  if (isDev) {
    return path.join(process.cwd(), 'python-runtime-manifest.json')
  }
  return path.join(process.resourcesPath, 'python-runtime-manifest.json')
}

function getInstalledHashPath(): string {
  return path.join(app.getPath('userData'), 'python', 'deps-hash.txt')
}

function getInstalledArchiveHashPath(): string {
  return path.join(app.getPath('userData'), 'python', 'archive-sha256.txt')
}

/** Directory where python-embed lives at runtime. */
export function getPythonDir(): string {
  if (process.platform === 'win32' || process.platform === 'linux') {
    if (isDev) {
      return path.join(process.cwd(), 'python-embed')
    }
    return path.join(app.getPath('userData'), 'python')
  }
  // macOS: bundled in resources
  return path.join(process.resourcesPath, 'python')
}

/** Check whether the installed Python environment matches the signed app. */
export function isPythonReady(): { ready: boolean } {
  if (process.platform === 'darwin') {
    return { ready: true }
  }

  if (isDev) {
    return { ready: true }
  }

  const bundledHash = readHash(getBundledHashPath())
  const installedHash = readHash(getInstalledHashPath())
  const installedArchiveHash = readHash(getInstalledArchiveHashPath())
  if (!bundledHash) return { ready: false }
  try {
    const manifest = parseArchiveManifest(getBundledManifestPath(), bundledHash)
    return {
      ready: bundledHash === installedHash
        && manifest.archiveSha256 === installedArchiveHash,
    }
  } catch (error) {
    logger.error(`[python-setup] Invalid bundled runtime manifest: ${error}`)
    return { ready: false }
  }
}

function readHash(filePath: string): string | null {
  try {
    return fs.readFileSync(filePath, 'utf-8').trim()
  } catch {
    return null
  }
}

// ── Archive source resolution ─────────────────────────────────────────
// Production artifacts are hosted only on this fork's GitHub release.

function getPythonArchivePrefix(): string {
  if (process.platform === 'win32') return 'python-embed-win32'
  if (process.platform === 'linux') {
    if (process.arch === 'x64') return 'python-embed-linux-x64'
    if (process.arch === 'arm64') return 'python-embed-linux-arm64'
    throw new Error(`Unsupported Linux architecture: ${process.arch}`)
  }
  throw new Error(`Python download is not supported on ${process.platform}`)
}

function getArchiveBase(): string {
  // LTX_PYTHON_URL is a dev-only override for testing with local archives.
  // Disabled in production to prevent code injection into a signed app.
  if (isDev && process.env.LTX_PYTHON_URL) {
    return process.env.LTX_PYTHON_URL.replace(/^["']+|["']+$/g, '')
  }
  const version = app.getVersion()
  return `https://github.com/MountainPlatform300/LTX-Desktop/releases/download/v${version}`
}

function isLocalPath(source: string): boolean {
  return !source.startsWith('http://') && !source.startsWith('https://')
}

/**
 * Acquire the python-embed archive from a source (local, GitHub, or CDN).
 * Returns once the archive is written to archivePath.
 */
async function acquireArchive(
  base: string,
  archivePath: string,
  cleanupFiles: string[],
  onProgress: (progress: PythonSetupProgress) => void,
  expectedManifest?: ArchiveManifest,
): Promise<void> {
  if (isLocalPath(base) && base.endsWith('.tar.gz')) {
    await copyFileWithProgress(base, archivePath, 0, fs.statSync(base).size, onProgress)
  } else if (isLocalPath(base)) {
    await acquirePartsLocal(base, archivePath, cleanupFiles, onProgress, expectedManifest)
  } else if (base.includes('/releases/download/')) {
    // GitHub Releases — multi-part
    await acquirePartsRemote(base, archivePath, cleanupFiles, onProgress, expectedManifest)
  } else {
    // CDN or other URL — single file (content-length discovered from response)
    let lastTime = Date.now()
    let lastBytes = 0
    let speed = 0

    await downloadFileWithGlobalProgress(base, archivePath, 0, 0, (downloaded, totalBytes) => {
      const now = Date.now()
      const elapsed = (now - lastTime) / 1000
      if (elapsed >= 1) {
        speed = (downloaded - lastBytes) / elapsed
        lastTime = now
        lastBytes = downloaded
      }

      onProgress({
        status: 'downloading',
        percent: totalBytes > 0 ? Math.round((downloaded / totalBytes) * 100) : 0,
        downloadedBytes: downloaded,
        totalBytes,
        speed,
      })
    })
  }
}

/**
 * Download (or copy) python-embed archive and extract to userData/python/.
 * Downloads from the matching fork GitHub release.
 */
export async function downloadPythonEmbed(
  onProgress: (progress: PythonSetupProgress) => void
): Promise<void> {
  const destDir = path.join(app.getPath('userData'), 'python')
  const tempDir = path.join(app.getPath('userData'), 'python-tmp')
  const archivePath = path.join(app.getPath('userData'), `${getPythonArchivePrefix()}.tar.gz`)

  try {
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true })
    }
  } catch { /* ignore */ }

  fs.mkdirSync(tempDir, { recursive: true })

  const cleanupFiles: string[] = []

  try {
    const base = getArchiveBase()
    logger.info( `[python-setup] Archive base: ${base}`)

    const expectedManifest = parseArchiveManifest(
      getBundledManifestPath(),
      readHash(getBundledHashPath()) ?? undefined,
    )
    await acquireArchive(base, archivePath, cleanupFiles, onProgress, expectedManifest)

    // Extract
    onProgress({ status: 'extracting', percent: 100, downloadedBytes: 0, totalBytes: 0, speed: 0 })
    logger.info( `[python-setup] Extracting to: ${tempDir}`)
    await extractTarGz(archivePath, tempDir)

    // Move into place (archive has top-level `python-embed/` directory)
    const extractedInner = path.join(tempDir, 'python-embed')
    const extractedSource = fs.existsSync(extractedInner) ? extractedInner : tempDir

    if (fs.existsSync(destDir)) {
      fs.rmSync(destDir, { recursive: true, force: true })
    }
    fs.renameSync(extractedSource, destDir)

    // Record both dependency and archive identities only after verified extraction.
    const bundledHash = getBundledHashPath()
    if (fs.existsSync(bundledHash)) {
      fs.copyFileSync(bundledHash, path.join(destDir, 'deps-hash.txt'))
    }
    fs.writeFileSync(
      path.join(destDir, 'archive-sha256.txt'),
      `${expectedManifest.archiveSha256}\n`,
      'utf-8',
    )

    onProgress({ status: 'complete', percent: 100, downloadedBytes: 0, totalBytes: 0, speed: 0 })
    logger.info( '[python-setup] Python environment ready')
  } catch (err) {
    try { fs.rmSync(tempDir, { recursive: true, force: true }) } catch { /* ignore */ }
    try { fs.rmSync(destDir, { recursive: true, force: true }) } catch { /* ignore */ }
    throw err
  } finally {
    try { fs.unlinkSync(archivePath) } catch { /* ignore */ }
    for (const f of cleanupFiles) {
      try { fs.unlinkSync(f) } catch { /* ignore */ }
    }
    try { if (fs.existsSync(tempDir)) fs.rmSync(tempDir, { recursive: true, force: true }) } catch { /* ignore */ }
  }
}

// ── Multi-part: local directory ──────────────────────────────────────

function parseArchiveManifest(
  manifestPath: string,
  expectedDepsHash?: string,
): ArchiveManifest {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as ArchiveManifest
  const prefix = getPythonArchivePrefix()
  if (
    manifest.schemaVersion !== 1
    || manifest.platform !== process.platform
    || manifest.arch !== process.arch
    || !/^[a-f0-9]{64}$/.test(manifest.depsHash)
    || !/^[a-f0-9]{64}$/.test(manifest.archiveSha256)
    || !Number.isSafeInteger(manifest.totalSize)
    || manifest.totalSize <= 0
    || !Array.isArray(manifest.parts)
    || manifest.parts.length === 0
  ) {
    throw new Error('Invalid Python archive manifest')
  }
  if (expectedDepsHash && manifest.depsHash !== expectedDepsHash) {
    throw new Error('Python archive dependency hash does not match this app version')
  }
  for (const [index, part] of manifest.parts.entries()) {
    const expectedName = `${prefix}.part${String(index + 1).padStart(3, '0')}`
    if (
      part.name !== expectedName
      || !Number.isSafeInteger(part.size)
      || part.size <= 0
      || !/^[a-f0-9]{64}$/.test(part.sha256)
    ) {
      throw new Error('Invalid Python archive part metadata')
    }
  }
  if (manifest.parts.reduce((sum, part) => sum + part.size, 0) !== manifest.totalSize) {
    throw new Error('Python archive manifest size mismatch')
  }
  return manifest
}

function assertExpectedManifest(
  manifest: ArchiveManifest,
  expectedManifest?: ArchiveManifest,
): void {
  if (!expectedManifest) return
  if (JSON.stringify(manifest) !== JSON.stringify(expectedManifest)) {
    throw new Error('Python archive manifest does not match the signed application')
  }
}

function sha256File(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256')
    const input = fs.createReadStream(filePath)
    input.on('data', (chunk: Buffer) => hash.update(chunk))
    input.on('error', reject)
    input.on('end', () => resolve(hash.digest('hex')))
  })
}

async function verifyArchiveFile(
  filePath: string,
  expectedSize: number,
  expectedHash: string,
): Promise<void> {
  const actualSize = fs.statSync(filePath).size
  if (actualSize !== expectedSize) {
    throw new Error(`Python archive part has size ${actualSize}; expected ${expectedSize}`)
  }
  const actualHash = await sha256File(filePath)
  if (actualHash !== expectedHash) {
    throw new Error('Python archive integrity verification failed')
  }
}

async function acquirePartsLocal(
  dirPath: string,
  archivePath: string,
  cleanupFiles: string[],
  onProgress: (progress: PythonSetupProgress) => void,
  expectedManifest?: ArchiveManifest,
): Promise<void> {
  const manifestPath = path.join(dirPath, `${getPythonArchivePrefix()}.manifest.json`)
  const manifest = parseArchiveManifest(manifestPath, expectedManifest?.depsHash)
  assertExpectedManifest(manifest, expectedManifest)

  const partPaths: string[] = []
  let bytesSoFar = 0

  for (const part of manifest.parts) {
    const src = path.join(dirPath, part.name)
    const dest = path.join(app.getPath('userData'), part.name)
    partPaths.push(dest)
    cleanupFiles.push(dest)

    await copyFileWithProgress(src, dest, bytesSoFar, manifest.totalSize, onProgress)
    await verifyArchiveFile(dest, part.size, part.sha256)
    bytesSoFar += part.size
  }

  await concatenateParts(partPaths, archivePath)
  await verifyArchiveFile(archivePath, manifest.totalSize, manifest.archiveSha256)
}

// ── Multi-part: remote download ──────────────────────────────────────

async function acquirePartsRemote(
  baseUrl: string,
  archivePath: string,
  cleanupFiles: string[],
  onProgress: (progress: PythonSetupProgress) => void,
  expectedManifest?: ArchiveManifest,
): Promise<void> {
  // Fetch manifest
  const prefix = getPythonArchivePrefix()
  const manifestUrl = `${baseUrl}/${prefix}.manifest.json`
  const manifestDest = path.join(app.getPath('userData'), `${prefix}.manifest.json`)
  cleanupFiles.push(manifestDest)
  await downloadFileRaw(manifestUrl, manifestDest)
  const manifest = parseArchiveManifest(manifestDest, expectedManifest?.depsHash)
  assertExpectedManifest(manifest, expectedManifest)

  const partPaths: string[] = []
  let bytesSoFar = 0
  let lastTime = Date.now()
  let lastReportedBytes = 0
  let speed = 0

  for (const part of manifest.parts) {
    const partUrl = `${baseUrl}/${part.name}`
    const partDest = path.join(app.getPath('userData'), part.name)
    partPaths.push(partDest)
    cleanupFiles.push(partDest)

    await downloadFileWithGlobalProgress(
      partUrl,
      partDest,
      bytesSoFar,
      manifest.totalSize,
      (globalDownloaded, totalBytes) => {
        const now = Date.now()
        const elapsed = (now - lastTime) / 1000

        if (elapsed >= 1) {
          speed = (globalDownloaded - lastReportedBytes) / elapsed
          lastTime = now
          lastReportedBytes = globalDownloaded
        }

        onProgress({
          status: 'downloading',
          percent: Math.round((globalDownloaded / totalBytes) * 100),
          downloadedBytes: globalDownloaded,
          totalBytes,
          speed,
        })
      }
    )

    await verifyArchiveFile(partDest, part.size, part.sha256)
    bytesSoFar += part.size
  }

  await concatenateParts(partPaths, archivePath)
  await verifyArchiveFile(archivePath, manifest.totalSize, manifest.archiveSha256)
}

// ── File operations ──────────────────────────────────────────────────

function concatenateParts(parts: string[], dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const writeStream = fs.createWriteStream(dest)
    let i = 0

    function writeNext() {
      if (i >= parts.length) {
        writeStream.end(() => resolve())
        return
      }

      const readStream = fs.createReadStream(parts[i])
      i++

      readStream.on('error', (err) => {
        writeStream.destroy()
        reject(err)
      })

      readStream.on('end', writeNext)
      readStream.pipe(writeStream, { end: false })
    }

    writeStream.on('error', reject)
    writeNext()
  })
}

/** Copy a local file with progress relative to a global total. */
function copyFileWithProgress(
  source: string,
  dest: string,
  globalOffset: number,
  globalTotal: number,
  onProgress: (progress: PythonSetupProgress) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    let copiedBytes = 0

    const readStream = fs.createReadStream(source)
    const writeStream = fs.createWriteStream(dest)

    readStream.on('data', (chunk: Buffer) => {
      copiedBytes += chunk.length
      const totalDone = globalOffset + copiedBytes
      onProgress({
        status: 'downloading',
        percent: Math.round((totalDone / globalTotal) * 100),
        downloadedBytes: totalDone,
        totalBytes: globalTotal,
        speed: 0,
      })
    })

    readStream.on('error', reject)
    writeStream.on('error', reject)
    writeStream.on('finish', resolve)

    readStream.pipe(writeStream)
  })
}

/** Download a file without progress (used for manifest). */
function validateDownloadUrl(rawUrl: string): URL {
  const url = new URL(rawUrl)
  if (isDev) {
    if (url.protocol !== 'https:' && url.protocol !== 'http:') {
      throw new Error(`Unsupported download protocol: ${url.protocol}`)
    }
    return url
  }
  const trustedHost = url.hostname === 'github.com'
    || url.hostname.endsWith('.githubusercontent.com')
  if (url.protocol !== 'https:' || !trustedHost) {
    throw new Error(`Untrusted Python artifact URL: ${url.origin}`)
  }
  return url
}

function downloadFileRaw(url: string, dest: string, redirectCount = 0): Promise<void> {
  return new Promise((resolve, reject) => {
    if (redirectCount > 5) {
      reject(new Error('Too many redirects'))
      return
    }

    let validatedUrl: URL
    try {
      validatedUrl = validateDownloadUrl(url)
    } catch (error) {
      reject(error)
      return
    }
    const client = validatedUrl.protocol === 'https:' ? https : http
    const req = client.get(validatedUrl, { headers: getAuthHeaders(validatedUrl) }, (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume()
        const redirectUrl = new URL(res.headers.location, validatedUrl).toString()
        downloadFileRaw(redirectUrl, dest, redirectCount + 1).then(resolve).catch(reject)
        return
      }
      if (!res.statusCode || res.statusCode >= 400) {
        res.resume()
        reject(new Error(`Download failed: HTTP ${res.statusCode}`))
        return
      }

      const file = fs.createWriteStream(dest)
      res.pipe(file)
      file.on('finish', () => file.close(() => resolve()))
      file.on('error', (err) => { fs.unlink(dest, () => {}); reject(err) })
    })

    req.on('error', reject)
  })
}

/** Download a file, reporting progress as (globalDownloaded, globalTotal). */
function downloadFileWithGlobalProgress(
  url: string,
  dest: string,
  globalOffset: number,
  globalTotal: number,
  onProgress: (globalDownloaded: number, globalTotal: number) => void,
  redirectCount = 0
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (redirectCount > 5) {
      reject(new Error('Too many redirects'))
      return
    }

    let validatedUrl: URL
    try {
      validatedUrl = validateDownloadUrl(url)
    } catch (error) {
      reject(error)
      return
    }
    const client = validatedUrl.protocol === 'https:' ? https : http
    const req = client.get(validatedUrl, { headers: getAuthHeaders(validatedUrl) }, (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume()
        const redirectUrl = new URL(res.headers.location, validatedUrl).toString()
        downloadFileWithGlobalProgress(redirectUrl, dest, globalOffset, globalTotal, onProgress, redirectCount + 1)
          .then(resolve).catch(reject)
        return
      }
      if (!res.statusCode || res.statusCode >= 400) {
        res.resume()
        reject(new Error(`Download failed: HTTP ${res.statusCode}`))
        return
      }

      // If caller didn't know total, use content-length from response
      const effectiveTotal = globalTotal || parseInt(res.headers['content-length'] || '0', 10)

      let downloadedBytes = 0
      const file = fs.createWriteStream(dest)
      res.pipe(file)

      res.on('data', (chunk: Buffer) => {
        downloadedBytes += chunk.length
        onProgress(globalOffset + downloadedBytes, effectiveTotal)
      })

      file.on('finish', () => file.close(() => resolve()))
      file.on('error', (err) => { fs.unlink(dest, () => {}); reject(err) })
    })

    req.on('error', reject)
  })
}

function runTar(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile('tar', args, { maxBuffer: 64 * 1024 * 1024 }, (err, stdout, stderr) => {
      if (err) {
        reject(new Error(`tar failed: ${stderr || err.message}`))
        return
      }
      resolve(stdout)
    })
  })
}

/** Validate and extract a .tar.gz using the system tar command. */
async function extractTarGz(archive: string, destDir: string): Promise<void> {
  const entries = await runTar(['-tzf', archive])
  validatePythonArchiveEntries(entries)
  const verboseEntries = await runTar(['-tvzf', archive])
  validatePythonArchiveTypes(verboseEntries)
  await runTar(['-xzf', archive, '-C', destDir])
}
