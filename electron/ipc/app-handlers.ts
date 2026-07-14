import { app, dialog } from 'electron'
import path from 'path'
import fs from 'fs'
import { checkGPU } from '../gpu'
import { isPythonReady, downloadPythonEmbed } from '../python-setup'
import { getBackendHealthStatus, getBackendUrl, getAuthToken, getAdminToken, startPythonBackend } from '../python-backend'
import { probeWsl, getWslSetupState, startWslInstall, restartWindows, probeWslMemory, configureWslMemory, restartWsl, ensureWslMemoryReady } from '../wsl-setup'
import { getMainWindow } from '../window'
import { getAnalyticsState, setAnalyticsEnabled, sendAnalyticsEvent } from '../analytics'
import { logger } from '../logger'
import { handle } from './typed-handle'

function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / Math.pow(1024, i)
  return `${value >= 100 || i === 0 ? Math.round(value) : value.toFixed(1)} ${units[i]}`
}

function getModelsPath(): string {
  const modelsPath = path.join(app.getPath('userData'), 'models')
  if (!fs.existsSync(modelsPath)) {
    fs.mkdirSync(modelsPath, { recursive: true })
  }
  return modelsPath
}

function getSetupStatus(settingsPath: string): { needsSetup: boolean; needsLicense: boolean } {
  if (!fs.existsSync(settingsPath)) {
    return { needsSetup: true, needsLicense: true }
  }
  try {
    const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    return {
      needsSetup: !settings.setupComplete,
      needsLicense: !settings.licenseAccepted,
    }
  } catch {
    return { needsSetup: true, needsLicense: true }
  }
}

function markSetupComplete(settingsPath: string): void {
  let settings: Record<string, unknown> = {}

  try {
    if (fs.existsSync(settingsPath)) {
      settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    }
  } catch {
    settings = {}
  }

  settings.setupComplete = true
  settings.licenseAccepted = true
  settings.licenseAcceptedDate = new Date().toISOString()
  settings.setupDate = new Date().toISOString()

  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2))
}

function markLicenseAccepted(settingsPath: string): void {
  let settings: Record<string, unknown> = {}

  try {
    if (fs.existsSync(settingsPath)) {
      settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    }
  } catch {
    settings = {}
  }

  settings.licenseAccepted = true
  settings.licenseAcceptedDate = new Date().toISOString()

  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2))
}

export function registerAppHandlers(): void {
  handle('getBackend', () => {
    return { url: getBackendUrl() ?? '', token: getAuthToken() ?? '' }
  })

  handle('getModelsPath', () => {
    return getModelsPath()
  })

  handle('checkGpu', async () => {
    return await checkGPU()
  })

  handle('getAppInfo', () => {
    return {
      version: app.getVersion(),
      isPackaged: app.isPackaged,
      modelsPath: getModelsPath(),
      userDataPath: app.getPath('userData'),
    }
  })

  handle('getDownloadsPath', () => {
    return app.getPath('downloads')
  })

  handle('getAvailableDiskSpace', () => {
    // Free space on the drive that holds the models dir (where first-run
    // downloads land). `statfs` needs an existing path, so fall back to the
    // userData root if the models dir doesn't exist yet.
    const modelsPath = path.join(app.getPath('userData'), 'models')
    const target = fs.existsSync(modelsPath) ? modelsPath : app.getPath('userData')
    let availableBytes = 0
    try {
      const stats = fs.statfsSync(target)
      availableBytes = stats.bavail * stats.bsize
    } catch (e) {
      logger.error(`getAvailableDiskSpace: ${e}`)
    }
    return { availableBytes, label: formatBytes(availableBytes) }
  })

  handle('checkFirstRun', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    return getSetupStatus(settingsPath)
  })

  handle('acceptLicense', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    markLicenseAccepted(settingsPath)
    return true
  })

  handle('completeSetup', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    markSetupComplete(settingsPath)
    return true
  })

  handle('fetchLicenseText', async () => {
    const resp = await fetch('https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE')
    if (!resp.ok) {
      throw new Error(`Failed to fetch license (HTTP ${resp.status})`)
    }
    return await resp.text()
  })

  handle('getNoticesText', async () => {
    const noticesPath = path.join(app.getAppPath(), 'NOTICES.md')
    return fs.readFileSync(noticesPath, 'utf-8')
  })

  handle('getResourcePath', () => {
    if (!app.isPackaged) {
      return null
    }
    return process.resourcesPath
  })

  handle('checkPythonReady', () => {
    return isPythonReady()
  })

  handle('startPythonSetup', async () => {
    await downloadPythonEmbed((progress) => {
      getMainWindow()?.webContents.send('python-setup-progress', progress)
    })
  })

  handle('startPythonBackend', async () => {
    await startPythonBackend()
  })

  handle('getBackendHealthStatus', () => {
    return getBackendHealthStatus()
  })

  handle('probeWsl', () => {
    return probeWsl()
  })

  handle('getWslSetupState', () => {
    return getWslSetupState()
  })

  handle('startWslInstall', async () => {
    await startWslInstall()
  })

  handle('restartWindows', () => {
    return restartWindows()
  })

  handle('probeWslMemory', () => {
    return probeWslMemory()
  })

  handle('configureWslMemory', () => {
    return configureWslMemory()
  })

  handle('restartWsl', () => {
    return restartWsl()
  })

  handle('ensureWslMemoryReady', () => {
    return ensureWslMemoryReady()
  })

  handle('getAnalyticsState', () => {
    return getAnalyticsState()
  })

  handle('setAnalyticsEnabled', ({ enabled }) => {
    setAnalyticsEnabled(enabled)
  })

  handle('sendAnalyticsEvent', async ({ eventName, extraDetails }) => {
    await sendAnalyticsEvent(eventName, extraDetails)
  })

  handle('openModelsDirChangeDialog', async () => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return { success: false, error: 'No window' }

    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Select Models Directory',
      properties: ['openDirectory', 'createDirectory'],
    })
    if (result.canceled || !result.filePaths.length) return { success: false, error: 'cancelled' }

    const newDir = result.filePaths[0]
    const url = getBackendUrl()
    const auth = getAuthToken()
    const admin = getAdminToken()
    if (!url || !auth || !admin) return { success: false, error: 'Backend not ready' }

    const resp = await fetch(`${url}/api/settings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${auth}`,
        'X-Admin-Token': admin,
      },
      body: JSON.stringify({ modelsDir: newDir }),
    })
    if (!resp.ok) return { success: false, error: await resp.text() }

    return { success: true, path: newDir }
  })

}
