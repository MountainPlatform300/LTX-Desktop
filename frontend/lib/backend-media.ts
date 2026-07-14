import { useEffect, useState } from 'react'
import { backendFetch } from './backend'

// Media served off the backend (validation samples, etc.) lives behind the
// same Bearer-auth middleware as the JSON API. `<video src>` / `<img src>`
// can't carry an Authorization header, so we fetch the bytes through
// `backendFetch` and hand the element a blob: URL. Blob URLs are cached per
// path so a re-render or sibling component doesn't re-fetch the same media,
// and revoked once nothing references them.

const blobCache = new Map<string, { url: string; refs: number }>()

async function resolveBlobUrl(path: string): Promise<string> {
  const cached = blobCache.get(path)
  if (cached) {
    cached.refs += 1
    return cached.url
  }
  const res = await backendFetch(path)
  if (!res.ok) {
    throw new Error(`backend media ${path} returned ${res.status}`)
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  blobCache.set(path, { url, refs: 1 })
  return url
}

function releaseBlobUrl(path: string): void {
  const entry = blobCache.get(path)
  if (!entry) return
  entry.refs -= 1
  if (entry.refs <= 0) {
    URL.revokeObjectURL(entry.url)
    blobCache.delete(path)
  }
}

/**
 * Resolve a backend-relative media path (e.g. `/api/lora/training/{id}/
 * validation-media?step=50&sampleIndex=1`) to a blob: URL usable as a
 * `<video src>` / `<img src>`. Returns null while loading or when the path is
 * absent; returns a stable error string only if the fetch fails. The blob is
 * ref-counted across callers and revoked when the last consumer unmounts.
 */
export function useBackendMediaUrl(path: string | null | undefined): {
  url: string | null
  error: string | null
} {
  const [url, setUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!path) {
      setUrl(null)
      setError(null)
      return
    }
    let active = true
    setUrl(null)
    setError(null)
    resolveBlobUrl(path)
      .then((u) => {
        if (active) setUrl(u)
      })
      .catch((e: unknown) => {
        if (active) setError(e instanceof Error ? e.message : 'Failed to load media')
      })
    return () => {
      active = false
      releaseBlobUrl(path)
    }
  }, [path])

  return { url, error }
}
