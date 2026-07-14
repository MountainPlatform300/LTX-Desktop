import {
  DEFAULT_IMPORT_NORMALIZE,
  FPS_CHOICES,
  SHORT_SIDE_CHOICES,
  type ImportNormalizeSpec,
} from './lora-import-normalize'

export type StoredPairView = 'combined' | 'sideBySide' | 'flat'
export type StoredPairFilter = 'all' | 'pairsOnly' | 'looseOnly' | 'incomplete'
export type StoredGalleryLayout = 'grid' | 'list'
export type LoraSidebarSection = 'datasets' | 'runs' | 'compute'

export interface LoraSidebarSectionSizes {
  datasets: number
  runs: number
  compute: number
}

export interface LoraUiPreferences {
  pairView: StoredPairView
  pairFilter: StoredPairFilter
  galleryLayout: StoredGalleryLayout
  importNormalize: ImportNormalizeSpec
  computePanePercent: number
  sidebarSectionSizes: LoraSidebarSectionSizes
  collapsedSidebarSections: LoraSidebarSection[]
  sidebarWidth: number
}

const STORAGE_KEY = 'lora.uiPrefs.v1' // gitleaks:allow (localStorage key, not a credential)
const LEGACY_PAIR_VIEW = 'lora.pairView'
const LEGACY_PAIR_FILTER = 'lora.pairFilter'
const LEGACY_LAYOUT = 'lora.galleryLayout'
const LEGACY_IMPORT_NORMALIZE = 'ltx.lora.importNormalize'

const DEFAULTS: LoraUiPreferences = {
  pairView: 'combined',
  pairFilter: 'all',
  galleryLayout: 'grid',
  importNormalize: DEFAULT_IMPORT_NORMALIZE,
  computePanePercent: 34,
  sidebarSectionSizes: { datasets: 42, runs: 24, compute: 34 },
  collapsedSidebarSections: [],
  sidebarWidth: 256,
}

function oneOf<T extends string>(
  value: unknown,
  allowed: readonly T[],
  fallback: T,
): T {
  return typeof value === 'string' && allowed.includes(value as T)
    ? (value as T)
    : fallback
}

function normalizeImportSpec(value: unknown): ImportNormalizeSpec {
  const record =
    typeof value === 'object' && value !== null
      ? (value as Record<string, unknown>)
      : {}
  const trim =
    typeof record.trim === 'object' && record.trim !== null
      ? (record.trim as Record<string, unknown>)
      : {}
  const resolution =
    typeof record.resolution === 'object' && record.resolution !== null
      ? (record.resolution as Record<string, unknown>)
      : {}
  const fps =
    typeof record.fps === 'object' && record.fps !== null
      ? (record.fps as Record<string, unknown>)
      : {}
  return {
    trim: {
      enabled:
        typeof trim.enabled === 'boolean'
          ? trim.enabled
          : DEFAULT_IMPORT_NORMALIZE.trim.enabled,
      maxSeconds:
        typeof trim.maxSeconds === 'number' && trim.maxSeconds > 0
          ? trim.maxSeconds
          : DEFAULT_IMPORT_NORMALIZE.trim.maxSeconds,
    },
    resolution: {
      enabled:
        typeof resolution.enabled === 'boolean'
          ? resolution.enabled
          : DEFAULT_IMPORT_NORMALIZE.resolution.enabled,
      shortSide:
        typeof resolution.shortSide === 'number' &&
        (SHORT_SIDE_CHOICES as readonly number[]).includes(resolution.shortSide)
          ? resolution.shortSide
          : DEFAULT_IMPORT_NORMALIZE.resolution.shortSide,
    },
    fps: {
      enabled:
        typeof fps.enabled === 'boolean'
          ? fps.enabled
          : DEFAULT_IMPORT_NORMALIZE.fps.enabled,
      value:
        typeof fps.value === 'number' &&
        (FPS_CHOICES as readonly number[]).includes(fps.value)
          ? fps.value
          : DEFAULT_IMPORT_NORMALIZE.fps.value,
    },
  }
}

function normalizePreferences(value: unknown): LoraUiPreferences {
  const record =
    typeof value === 'object' && value !== null
      ? (value as Record<string, unknown>)
      : {}
  const rawSizes =
    typeof record.sidebarSectionSizes === 'object' &&
    record.sidebarSectionSizes !== null
      ? (record.sidebarSectionSizes as Record<string, unknown>)
      : {}
  const size = (key: LoraSidebarSection, fallback: number) =>
    typeof rawSizes[key] === 'number' && Number.isFinite(rawSizes[key])
      ? Math.min(84, Math.max(8, rawSizes[key]))
      : fallback
  const collapsed = Array.isArray(record.collapsedSidebarSections)
    ? record.collapsedSidebarSections.filter(
      (item): item is LoraSidebarSection =>
        item === 'datasets' || item === 'runs' || item === 'compute',
    )
    : DEFAULTS.collapsedSidebarSections
  return {
    pairView: oneOf(
      record.pairView,
      ['combined', 'sideBySide', 'flat'],
      DEFAULTS.pairView,
    ),
    pairFilter: oneOf(
      record.pairFilter,
      ['all', 'pairsOnly', 'looseOnly', 'incomplete'],
      DEFAULTS.pairFilter,
    ),
    galleryLayout: oneOf(
      record.galleryLayout,
      ['grid', 'list'],
      DEFAULTS.galleryLayout,
    ),
    importNormalize: normalizeImportSpec(record.importNormalize),
    computePanePercent:
      typeof record.computePanePercent === 'number' &&
      Number.isFinite(record.computePanePercent)
        ? Math.min(60, Math.max(24, record.computePanePercent))
        : DEFAULTS.computePanePercent,
    sidebarSectionSizes: {
      datasets: size('datasets', DEFAULTS.sidebarSectionSizes.datasets),
      runs: size('runs', DEFAULTS.sidebarSectionSizes.runs),
      compute: size(
        'compute',
        typeof record.computePanePercent === 'number'
          ? Math.min(60, Math.max(24, record.computePanePercent))
          : DEFAULTS.sidebarSectionSizes.compute,
      ),
    },
    collapsedSidebarSections: [...new Set(collapsed)],
    sidebarWidth:
      typeof record.sidebarWidth === 'number' && Number.isFinite(record.sidebarWidth)
        ? Math.min(520, Math.max(220, Math.round(record.sidebarWidth)))
        : DEFAULTS.sidebarWidth,
  }
}

function readLegacy(storage: Storage): LoraUiPreferences {
  let importNormalize: unknown = null
  try {
    const raw = storage.getItem(LEGACY_IMPORT_NORMALIZE)
    importNormalize = raw ? JSON.parse(raw) : null
  } catch {
    importNormalize = null
  }
  return normalizePreferences({
    pairView: storage.getItem(LEGACY_PAIR_VIEW),
    pairFilter: storage.getItem(LEGACY_PAIR_FILTER),
    galleryLayout: storage.getItem(LEGACY_LAYOUT),
    importNormalize,
  })
}

export function loadLoraUiPreferences(): LoraUiPreferences {
  if (typeof window === 'undefined') return DEFAULTS
  const storage = window.localStorage
  try {
    const current = storage.getItem(STORAGE_KEY)
    if (current) return normalizePreferences(JSON.parse(current))

    const migrated = readLegacy(storage)
    storage.setItem(STORAGE_KEY, JSON.stringify(migrated))
    for (const key of [
      LEGACY_PAIR_VIEW,
      LEGACY_PAIR_FILTER,
      LEGACY_LAYOUT,
      LEGACY_IMPORT_NORMALIZE,
    ]) {
      storage.removeItem(key)
    }
    return migrated
  } catch {
    return DEFAULTS
  }
}

export function saveLoraUiPreferences(
  patch: Partial<LoraUiPreferences>,
): LoraUiPreferences {
  const next = normalizePreferences({ ...loadLoraUiPreferences(), ...patch })
  if (typeof window !== 'undefined') {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    } catch {
      // Preferences are best-effort; inaccessible storage must not break UI.
    }
  }
  return next
}
