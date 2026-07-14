const FIRST_TB_GB = 1000
const FIRST_TB_RATE = 0.07
const OVER_TB_RATE = 0.05

export function estimateRunpodStorageMonthlyUsd(sizeGb: number): number {
  const normalized = Math.max(0, sizeGb)
  return Math.min(normalized, FIRST_TB_GB) * FIRST_TB_RATE
    + Math.max(0, normalized - FIRST_TB_GB) * OVER_TB_RATE
}

export function formatRunpodStorageMonthlyUsd(sizeGb: number): string {
  return estimateRunpodStorageMonthlyUsd(sizeGb).toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}
