# Documentation contract

`docs-app/src/docs.ts` is the single source of truth for architecture,
pipelines, features, commands, artifacts, dependencies, and guarantees.

Update it in the same change whenever a CLI entry point, pipeline stage,
contract, feature, output path, temporal guarantee, or validation strategy
changes. The change is incomplete until the docs app builds and the affected
feature page matches the code.

Do not maintain parallel architecture descriptions in `PROJECT_CONTEXT.md`,
`docs/ARCHITECTURE.md`, presentations, or report prose. They are historical
snapshots, not authoritative documentation.
