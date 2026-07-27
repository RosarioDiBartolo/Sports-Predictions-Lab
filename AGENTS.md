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

# Decision continuity

Before planning or implementing work, read the relevant feature entries and
guarantees in `docs-app/src/docs.ts`. Treat documented active decisions,
modeling constraints, temporal guarantees, validation requirements, and
promotion gates as binding project directives across chats and sessions.

Do not silently replace, weaken, bypass, or reinterpret those directives.
If a request appears to conflict with them, identify the conflict explicitly
and ask for a deliberate decision. When the user deliberately changes a
directive, update `docs-app/src/docs.ts` in the same change before treating the
new decision as authoritative.

Every predictive model or model variant must use the documented common
evaluation pipeline and satisfy its out-of-sample promotion gate before it can
replace an official model or be described as operational. A deliberate user
decision may override the gate only when the exception and its rationale are
recorded explicitly in `docs-app/src/docs.ts`; evaluation evidence remains
mandatory and must not be hidden.
