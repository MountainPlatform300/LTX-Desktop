let cached: { url: string; token: string } | null = null

export async function getBackendCredentials(): Promise<{ url: string; token: string }> {
  if (cached && cached.url) return cached
  // Only cache once the backend URL is actually available. An API call that
  // races ahead of backend startup gets an empty URL from Electron; caching
  // that would permanently route requests at the renderer (Vite) origin and
  // yield HTML instead of JSON. Re-resolve until a real URL exists so the
  // connection self-heals as soon as the backend is up.
  const next = await window.electronAPI.getBackend()
  if (next.url) cached = next
  return next
}

export function resetBackendCredentials(): void {
  cached = null
}

export async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  const { url, token } = await getBackendCredentials()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${url}${path}`, { ...init, headers })
}

export async function backendWsUrl(path: string): Promise<string> {
  const { url, token } = await getBackendCredentials()
  const ws = url.replace('http://', 'ws://')
  const sep = path.includes('?') ? '&' : '?'
  return `${ws}${path}${sep}token=${token}`
}
