// What the workspace currently has focused in the center stage + inspector.
export type Selection =
  | { kind: 'dataset'; id: string }
  | { kind: 'run'; id: string }
  | { kind: 'library' }
  | null
