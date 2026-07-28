# Operations

The public executable remains `odds-lab` during package migration.

## Pipeline stages

1. ingest provider assets into canonical identities, matches, results and odds;
2. collect and reconcile player and lineup observations;
3. enrich venues and weather;
4. build leakage-safe match and player temporal features;
5. train and evaluate the official neural lineup model;
6. join OOS predictions to bookmaker snapshots;
7. research and backtest frozen strategies;
8. predict future fixtures only from information available before the cutoff.

## Validation

Affected changes must run focused tests plus the common evaluation checks.
Database migrations additionally compare row counts, checksums, distinct
cardinalities, league/season/provider coverage, duplicates, orphan records,
mapping conflicts, zero versus missing and temporal leakage.

Training-ready matches require a final H/D/A result and exactly 11 distinct
starters per team with player identity and usable role information.

Training commands first execute the mandatory dataset preflight and expose a
run directory with fresh heartbeats. They must fail before model initialization
when the preflight is invalid. Agents monitor `run.json` and `events.jsonl`;
console output alone is not sufficient evidence of progress.

## Artifacts

Canonical database: `data/football_odds.sqlite3`.

Important processed artifacts include:

- `data/processed/player_training_ready.csv`;
- `data/processed/player_match_temporal_features.csv`;
- neural model, metadata, OOS predictions and metrics under
  `reports/modeling/neural_lineup_model/`;
- market and strategy evidence under `reports/`.

Artifacts from retired models remain versioned historical evidence.
