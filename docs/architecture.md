# Architecture

## Direction

The codebase uses compact domain packages so agents can work in small,
independent scopes. Business logic in the package root is forbidden.

Target layout:

```text
src/football_odds/
  core/          shared configuration, identifiers and temporal primitives
  data/          schema, repositories, releases, lineage and quarantine
  ingestion/     provider adapters, collection and reconciliation
  players/       lineup observations, player datasets and temporal feature store
  modeling/      common evaluation, neural model and archived baselines
  market/        bookmaker odds, calibration and market datasets
  strategies/    selection research, backtests and settlement
  enrichment/    venues, weather and other approved enrichments
  cli/           parser and thin domain commands
```

## Dependency direction

- Domain packages may depend on `core`.
- Provider adapters depend on data contracts, not on modeling.
- Player features depend on canonical data and may be consumed by modeling.
- Modeling emits versioned predictions; it does not own odds or strategy data.
- Strategies join versioned predictions to odds available at the decision
  cutoff.
- The CLI orchestrates domains but contains no model, ingestion or persistence
  logic.
- Circular imports between domain packages are prohibited.

## Code policy

- There are no compatibility wrappers or legacy import paths.
- `football_odds.__init__` does not re-export domain APIs.
- Each domain owns its implementation, tests and local `AGENTS.md`.
- Refactors remove more production lines than they add.
- New abstractions require at least two current consumers.
- No database or artifact is deleted as part of package reorganisation.

## Documentation

The retired docs web application has been removed. Markdown documents indexed
by `docs/README.md` are the only authoritative documentation.
