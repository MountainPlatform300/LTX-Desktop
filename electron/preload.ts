import { electronAPISchemas, type BackendHealthStatus, type WslSetupStateApi } from '../shared/electron-api-schema'
import { HF_GATING_ENABLED } from '../shared/feature-flags'

const { contextBridge, ipcRenderer, webUtils } = require('electron')

const api: Record<string, unknown> = {}

for (const key of Object.keys(electronAPISchemas)) {
  api[key] = (input?: unknown) => ipcRenderer.invoke(key, input)
}

api.onPythonSetupProgress = (cb: (data: unknown) => void) => {
  ipcRenderer.on('python-setup-progress', (_: unknown, data: unknown) => cb(data))
}

api.removePythonSetupProgress = () => {
  ipcRenderer.removeAllListeners('python-setup-progress')
}

api.onWslSetupProgress = (cb: (data: WslSetupStateApi) => void) => {
  const listener = (_: unknown, data: WslSetupStateApi) => cb(data)
  ipcRenderer.on('wsl-setup-progress', listener)
  return () => {
    ipcRenderer.removeListener('wsl-setup-progress', listener)
  }
}

api.onBackendHealthStatus = (cb: (data: BackendHealthStatus) => void) => {
  const listener = (_: unknown, data: BackendHealthStatus) => cb(data)
  ipcRenderer.on('backend-health-status', listener)
  return () => {
    ipcRenderer.removeListener('backend-health-status', listener)
  }
}

api.getPathForFile = (file: File) => webUtils.getPathForFile(file)

api.platform = process.platform

api.hfGatingEnabled = HF_GATING_ENABLED

contextBridge.exposeInMainWorld('electronAPI', api)

export {}
