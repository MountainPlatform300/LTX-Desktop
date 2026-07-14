import { app } from 'electron'
import fs from 'fs'
import path from 'path'

export interface AppState {
  analyticsEnabled?: boolean
  installationId?: string
  projectAssetsPath?: string
  // Persisted state of the local-training (WSL2) setup wizard, so it can resume
  // after the reboot that `wsl --install` requires. See electron/wsl-setup.ts.
  wslSetup?: {
    stage: string
    distro: string
    startedAt?: string
    updatedAt?: string
    error?: string | null
  }
  [key: string]: unknown
}

export function getAppStatePath(): string {
  return path.join(app.getPath('userData'), 'app_state.json')
}

export function readAppState(): AppState {
  const statePath = getAppStatePath()
  try {
    if (fs.existsSync(statePath)) {
      return JSON.parse(fs.readFileSync(statePath, 'utf-8')) as AppState
    }
  } catch (err) {
    console.warn('[app-state] failed to read app state:', err)
  }
  return {}
}

export function writeAppState(state: AppState): void {
  fs.writeFileSync(getAppStatePath(), JSON.stringify(state, null, 2))
}

let cachedProjectAssetsPath: string | null = null

export function getProjectAssetsPath(): string {
  if (cachedProjectAssetsPath) return cachedProjectAssetsPath
  const state = readAppState()
  if (state.projectAssetsPath) {
    cachedProjectAssetsPath = path.resolve(state.projectAssetsPath)
    return cachedProjectAssetsPath
  }
  const defaultPath = path.resolve(path.join(app.getPath('downloads'), 'Ltx Desktop Assets'))
  cachedProjectAssetsPath = defaultPath
  return defaultPath
}

export function setProjectAssetsPath(p: string): void {
  const resolvedPath = path.resolve(p)
  cachedProjectAssetsPath = resolvedPath
  const state = readAppState()
  state.projectAssetsPath = resolvedPath
  writeAppState(state)
}
