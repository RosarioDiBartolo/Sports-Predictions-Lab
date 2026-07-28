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

Historical Football-Data opening and closing labels have no verifiable
availability timestamp and are not eligible for cutoff-sensitive evaluation.
Prospective API-Football 1X2 collections preserve the provider `update`
timestamp, local UTC acquisition time, raw response checksum and normalized
selection rows under `data/raw/api_football_odds/`. These snapshots become
eligible only after reconciliation to a canonical fixture and validation that
their timestamp is not later than the prediction cutoff.

The CC BY-SA 4.0 BeatTheBookie hourly series is the historical cutoff-valid
source for the active gated comparison. Raw compressed files remain under
`data/raw/beat_the_bookie/`. Reconciliation requires the same calendar date and
final score plus high-confidence home/away team-name similarity with an
unambiguous runner-up margin. Rejected candidates are quarantined. The accepted
snapshot is the complete bookmaker H/D/A triplet sampled one hour before the
UTC kickoff; bookmaker identity follows the source generator's published
32-row mapping. The reconciled rows and immutable input checksums are recorded
in `reconciled_cutoff_snapshot.csv` and its manifest.

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
