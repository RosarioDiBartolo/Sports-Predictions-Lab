# Models and evaluation

## Official operational model

As of 2026-07-28, the only official and operational predictive model is
`dixon_coles_shared_encoder_pooling`.

It is the compact neural confirmed-lineup model that:

- consumes leakage-safe temporal player features;
- uses a shared player encoder rather than identity-first embeddings;
- pools players separately by team and, where possible, department;
- represents confirmed starters and bench separately;
- applies a bounded correction to Dixon–Coles goal rates;
- produces normalized H/D/A probabilities.

## Deliberate gate exception

This designation is an explicit user decision and a deliberate exception to the
previous promotion gate. At the time of the decision, its out-of-sample
comparison with the previous official model was inconclusive.

Rationale: models that use player data are the only operational family wanted
by the project, and the decisive external benchmark is the bookmaker market.

The exception changes operational status, not the evidence:

- existing inconclusive or unfavorable results remain visible;
- every new version still runs through the common OOS evaluation pipeline;
- no evaluation record may be deleted or rewritten to imply that the old gate
  was passed;
- model, dataset, configuration, cutoff, fold and metric provenance remain
  mandatory.

## Benchmark

The decisive benchmark is the bookmaker market available at the prediction
cutoff. Evaluation must preserve the exact bookmaker, market, selection, odds
snapshot and timestamp used for comparison.

Required evidence includes:

- temporal train/test splits;
- Log Loss, Brier, RPS and ECE/calibration;
- results by season and league;
- paired bootstrap;
- leakage checks;
- comparison against bookmaker probabilities available at the cutoff;
- reproducible strategy backtests and economic results where applicable.

Accuracy remains secondary. Closing odds may be used as a research benchmark
but must not be presented as live-executable when they were unavailable at the
decision time.

## Non-operational models

The following are not operational models:

- `dixon_coles_gradient_boosting`;
- `sport_gradient_boosting`;
- Dixon–Coles without player data;
- regularized linear confirmed-lineup pooling;
- self-attention lineup variants unless a later explicit decision promotes one.

They remain only when needed as:

- structural or diagnostic baselines;
- technical components of the neural model;
- common evaluation dependencies;
- versioned historical artifacts.

Their artifacts, configurations, datasets, predictions and metrics must be
archived, not deleted.

## Temporal guarantees

- Current-match results and performance never enter pre-kickoff features.
- Matches sharing a timestamp are snapshotted before any of them update state.
- Missing, observed zero and derived fallback are distinct states.
- Every player feature carries availability, quality, source and fallback kind.
- The supervised sample size is the number of matches, not the number of player
  observations.

## Mandatory dataset preflight

Every training run must create a run identifier and finish an immutable
preflight before allocating training resources. A failed preflight forbids
training.

The preflight must record and validate:

- exact input paths, byte sizes, checksums, schema and dataset version;
- row counts, unique match/player counts and coverage by league, season and
  provider;
- target distribution and the rows assigned to every temporal train,
  validation and test fold;
- duplicate keys, orphan records, mapping conflicts and invalid identifiers;
- exactly 11 distinct starters per team for every training-ready match;
- missing, observed zero and derived fallback as separate states;
- NaN, infinity, impossible values and feature availability/quality coverage;
- cutoff ordering, same-timestamp handling and all leakage assertions;
- bookmaker snapshot coverage at the evaluation cutoff;
- model configuration, feature list, random seeds, software versions and
  estimated workload.

The canonical output is
`reports/modeling/neural_lineup_model/runs/<run_id>/preflight.json`. It is
written before training and never overwritten. The console must print a short
PASS/FAIL summary and the report path.

## Training observability

Every run writes into its own run directory:

- `run.json`: configuration, dataset fingerprint, status, start/end timestamps,
  process information and latest heartbeat;
- `events.jsonl`: append-only structured events;
- `training.log`: concise human-readable log;
- fold, epoch and final metric artifacts.

While work is active, logs are flushed at least once per minute and at every
fold or epoch boundary. Each heartbeat includes phase, fold/epoch, train and
validation loss when available, elapsed time, ETA and latest metrics. Terminal
status is exactly `completed`, `failed` or `cancelled`; failures include the
exception and last valid checkpoint.

An agent must be able to verify progress by reading the latest `run.json` and
tail of `events.jsonl` without parsing an entire console transcript. Two missed
heartbeat intervals are stale and require investigation.
