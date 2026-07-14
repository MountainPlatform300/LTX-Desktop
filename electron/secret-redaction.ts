const REDACTED = '[REDACTED]'

const SECRET_PATTERNS: RegExp[] = [
  /\b([A-Z][A-Z0-9_]*(?:_KEY|_TOKEN|_SECRET|_PASSWORD|_CREDENTIAL)\s*=\s*)(?:'[^'\r\n]*'|"[^"\r\n]*"|[^\s;|]+)/gi,
  /\b(authorization\s*[:=]\s*(?:bearer|basic|key)\s+)[^\s,;]+/gi,
  /\b((?:x-goog-api-key|x-api-key)\s*[:=]\s*)[^\s,;]+/gi,
  /(["']?(?:access[_-]?token|refresh[_-]?token|api[_-]?key|password|secret)["']?\s*[:=]\s*)(["']).*?\2/gi,
  /\b((?:access[_-]?token|refresh[_-]?token|api[_-]?key|password|secret)\s*[:=]\s*)[^\s,;}\]]+/gi,
  /([?&](?:access_token|refresh_token|api_key|apikey|key|token|secret|password)=)[^&#\s]+/gi,
]

export function redactSecrets(
  text: string,
  knownSecrets: readonly (string | null | undefined)[] = [],
): string {
  let redacted = text
  for (const pattern of SECRET_PATTERNS) {
    redacted = redacted.replace(pattern, `$1${REDACTED}`)
  }
  const explicitSecrets = [...new Set(
    knownSecrets.filter((secret): secret is string => Boolean(secret && secret.length >= 4)),
  )].sort((left, right) => right.length - left.length)
  for (const secret of explicitSecrets) {
    redacted = redacted.replaceAll(secret, REDACTED)
  }
  return redacted
}
