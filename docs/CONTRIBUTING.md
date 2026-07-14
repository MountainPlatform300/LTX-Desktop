# Contributing to LTX Desktop

Thanks for taking the time to contribute!

## Getting started (development)

Prereqs:

- Node.js
- `uv` (Python package manager)
- Python 3.13.12
- Git

Setup:

```bash
pnpm setup:dev
```

Run:

```bash
pnpm dev
```

Debug:

```bash
pnpm dev:debug
```

Typecheck:

```bash
pnpm typecheck
```

## What we accept right now

- Bug fixes and small improvements
- Documentation updates
- Small, targeted UI fixes

**Frontend policy:** the frontend is under active refactor. Please avoid large UI/state rewrites for now — open an issue first so we can align on the target direction.

## Proposing larger work

Before starting a larger change (especially frontend architecture/state), please open an issue with:

- The problem you’re trying to solve
- The proposed approach (1–2 paragraphs is fine)
- Scope (areas/files likely to change)
- Any UX or compatibility impact

Wait for maintainer alignment before investing in a major refactor.

## Checks

At minimum, run:

- Type checking:

```bash
pnpm typecheck
```

- Backend tests:

```bash
pnpm backend:test
```

- Frontend tests and production build:

```bash
pnpm test:frontend
pnpm build:frontend
```

## Security and privacy

Report vulnerabilities through
[private vulnerability reporting](https://github.com/MountainPlatform300/LTX-Desktop/security/advisories/new),
not a public issue. Never submit API keys, tokens, private media, model weights,
personal paths, pod identifiers, generated app data, or local environment files.
New network, archive, filesystem, IPC, or process behavior should include abuse
tests and explicit allowlists or bounds.

By contributing, you agree to follow the repository
[Code of Conduct](../CODE_OF_CONDUCT.md).
