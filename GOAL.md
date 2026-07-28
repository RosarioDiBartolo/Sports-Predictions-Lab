# Current project goal

Updated: 2026-07-28

## Goal

Evaluate the official neural player model through controlled feature ablations,
add a deterministic reliability gate that reduces its Dixon-Coles rate
correction when player data are weak, and compare it with a gated tabular
residual challenger trained on the same inputs and targets.

## Status

In progress.

`dixon_coles_shared_encoder_pooling` remains the only official and operational
model. Both gated candidates, the ablation matrix, prospective timestamped-odds
collection and the observable comparison runner are implemented. The first
full preflight failed closed because the new 2026 bookmaker snapshot has no
overlap with the 2010/11-2024/25 training matches, so no candidate training has
started.

## Completion criteria

- Evaluate base history, detailed feature-store, bench and combined variants on
  identical out-of-sample matches through the common pipeline.
- Add a versioned, deterministic reliability score that attenuates the neural
  correction when its player inputs are incomplete or unreliable.
- Train a gradient-boosting residual challenger on the same feature groups,
  cross-fitted Dixon-Coles targets, temporal folds and reliability gate as the
  neural model.
- Compare Dixon-Coles, the hybrid diagnostic baseline, the current neural
  model, every ablation, the gated neural model and the gated tabular challenger
  using Log Loss, Brier, RPS, calibration, season stability and paired
  bootstrap.
- Report results separately for 2010/11-2017/18, 2018/19-2020/21 and
  2021/22-2024/25.
- Decide whether detailed-feature backfill is justified; do not perform it in
  this goal.
- Training performs the mandatory dataset preflight and produces observable,
  heartbeat-based run logs.
- `ruff`, `mypy` and the full `pytest` run pass with coverage at or above 90%.
- The canonical database checksum and all historical data/artifacts remain
  unchanged.

## Constraints

- Preserve raw/canonical data, provenance, licences, quarantine and historical
  evaluation evidence.
- No compatibility layer or obsolete runtime code.
- `dixon_coles_shared_encoder_pooling` remains the official operational model
  throughout experimentation; no candidate replaces its versioned artifact
  without accepted common-pipeline evidence.
- Bookmaker markets are the decisive model benchmark.
- Database v2 is the next separate phase; this goal does not migrate data.
- Self-attention, larger embeddings and historical backfill are out of scope.

## Next verification checkpoint

The immutable data preflight now passes. Preserve the accepted snapshot and
quarantine unchanged; start the monitored candidate comparison only in a later,
explicitly authorized long-running checkpoint.

## Discussion notes

- Historical backfill may start with 2015/16-2017/18 only if the mature-period
  ablations show a stable benefit from detailed feature-store inputs.
- The hybrid Dixon-Coles plus gradient boosting model is a diagnostic baseline,
  not the incumbent or an operational fallback.
- The new tabular model is a residual challenger, not a revival of a legacy
  hybrid runtime: it shares the gated player-data contract with the neural
  candidate and remains non-operational unless explicitly promoted later.
- API-Football prospective collection began on 2026-07-28. Historical
  Football-Data opening/closing labels remain ineligible for cutoff-sensitive
  evaluation because they have no availability timestamps.
- The free CC BY-SA BeatTheBookie series covers hourly pre-kickoff snapshots
  during 2015-2016 and overlaps the canonical lineup dataset; ambiguous fixture
  mappings remain quarantined rather than inferred.
- Candidate training is deliberately deferred after the data preflight because
  the full monitored comparison is too long for the current checkpoint.
- The long-running comparison will be split into independently preflighted
  candidate/ablation runs on Colab. Inputs are transferred as a checksum-verified
  bundle and code is checked out at an explicit tested commit.

## Decision log

- 2026-07-28: The ablation and reliability-gating program superseded the
  refactor-only goal, while retaining its quality and observability gates.
- 2026-07-28: The neural model remains the official operational model; the
  hybrid is evaluated only as a diagnostic baseline.
- 2026-07-28: A gated tabular residual model was added as a non-operational
  challenger to distinguish neural representation gains from tabular gains.
- 2026-07-28: The first common candidate run failed preflight solely because
  timestamped bookmaker coverage had zero overlap with the historical
  evaluation matches; the models were not initialized.
- 2026-07-28: Free CC BY-SA BeatTheBookie hourly odds were reconciled to 1,530
  completed canonical matches with valid lineups and results. The cutoff gate
  passed with 108,381 rows and 36,127 complete bookmaker markets; 429 uncertain
  mappings were quarantined. Training remained deliberately disabled.
- 2026-07-28: Markdown replaced the docs application as authoritative
  documentation.
- 2026-07-28: `dixon_coles_shared_encoder_pooling` became the only operational
  model by explicit exception; bookmaker markets became the decisive benchmark.
- 2026-07-28: Code organization became domain-scoped, subtraction-first and
  incompatible with legacy Python/CLI paths by design.
- 2026-07-28: Dataset preflight and heartbeat-based training observability became
  mandatory before every training run.
