// Lightweight, greppable logger for the LoRA Studio. Renderer-side logs land
// in the DevTools console; prefixing every line with `[LoRA]` makes the whole
// dataset-prep pipeline easy to filter and trace when something goes wrong.

type LogArg = unknown

function fmt(scope: string): string {
  return `[LoRA] ${scope}`
}

export const loraLog = {
  info(scope: string, ...args: LogArg[]): void {
    console.info(fmt(scope), ...args)
  },
  warn(scope: string, ...args: LogArg[]): void {
    console.warn(fmt(scope), ...args)
  },
  error(scope: string, ...args: LogArg[]): void {
    console.error(fmt(scope), ...args)
  },
}
