# Data platform

## Layers

- `data/raw/`: immutable provider assets with licences and acquisition context.
- Canonical SQLite: reconciled identities, matches, results, odds, lineups,
  player observations and enrichment.
- `data/processed/`: reproducible derived datasets and temporal feature stores.
- `reports/`: validation and evaluation evidence.

Raw assets are never overwritten by canonicalisation.

## Required capabilities

The platform must retain multi-provider collection, bookmaker odds,
reconciliation of teams/players/matches, enrichment, leakage-safe player
features, strategy research and common OOS evaluation.

Provider mappings preserve external ID, source, reconciliation method, quality
and ambiguity. Missing mappings and conflicts are quarantined rather than
guessed.

## Player observations

The canonical player-match grain is match, player and provider. It must preserve
team, starter/bench status, original and normalized position, formation grid,
shirt number, minutes, substitution timing, individual statistics, availability,
quality, source record and acquisition time.

Observed zero, missing and derived values are distinct.

Temporal team membership represents observed intervals, not assumed contracts.

## Odds

The canonical odds grain is match, bookmaker, provider, market, selection and
snapshot timestamp. Decimal odds as published are preserved. Normalized
probabilities and margins should be versioned transformations or views unless
materialization has a measured justification.

## Governance target

The canonical v2 design includes source assets, data releases, quarantine,
schema version, checksums, row counts and artifact lineage. Migration is
idempotent, resumable and additive; legacy tables are not physically dropped in
the first migration.

## Version-control boundary

Git does not store raw external datasets, model weights, execution logs or
mutable collector state. Those assets remain preserved in their data or
artifact storage and are referenced by lightweight, immutable manifests that
record source, licence, checksum, size, dataset or model version and lineage.

Curated Markdown reports and compact JSON evaluation evidence may be versioned
when they identify the producing configuration and inputs. Ignoring an asset in
Git is never authorization to delete or overwrite it.
