/**
 * Converts a filesystem path to a properly encoded file:// URL.
 *
 *   /Users/me/my file.mp4   → file:///Users/me/my%20file.mp4
 *   C:\Users\me\video#1.mp4 → file:///C:/Users/me/video%231.mp4
 */
export function pathToFileUrl(filePath: string): string {
  // Normalize Windows separators
  let normalized = filePath.replace(/\\/g, '/')

  // Ensure leading slash (Windows drive letters like C:/ need one prepended)
  if (!normalized.startsWith('/')) {
    normalized = '/' + normalized
  }

  // Encode each path segment individually so we don't encode the slashes
  const encoded = normalized
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')

  return 'file://' + encoded
}

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp']

/** Heuristic: treat a path as a still image (vs. a video clip) by extension.
 *  Frame extracts and Nano Banana edits are PNGs, so they read as images. */
export function isImagePath(filePath: string): boolean {
  const lower = filePath.toLowerCase()
  return IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext))
}
