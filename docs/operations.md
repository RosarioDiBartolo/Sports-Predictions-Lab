# Operations

The public executable remains `odds-lab` during package migration.
`model predict`, `strategy discover` and `strategy backtest` are reserved
commands and currently fail fast without reading project data.

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

The gated comparison runner writes immutable runs below
`reports/modeling/gated_comparison/runs/<run_id>/`. It evaluates the neural and
tabular residual candidates across the same four feature ablations only after
the common preflight passes. Prospective bookmaker snapshots must first be
reconciled to canonical matches and be available no later than the prediction
cutoff; an unreconciled or non-overlapping snapshot fails closed.
The historical BeatTheBookie reconciliation is run through
`football_odds.market.historical_snapshots.reconcile_beat_the_bookie`. It emits
only complete H/D/A markets one hour before kickoff, preserves ambiguous
mappings in `data/raw/beat_the_bookie/reconciliation_quarantine.csv`, and writes
the accepted snapshot plus checksum manifest beside the raw source.

## Remote candidate training

Remote training uses a tested Git commit and a portable bundle created only
from an immutable successful preflight:

```text
odds-lab --project-dir <project> model bundle \
  --preflight <successful-preflight.json> \
  --output-dir <new-bundle-directory>
```

The bundle preserves project-relative input paths and records byte sizes and
SHA-256 checksums in `training_bundle.manifest.json`. Verify it after transfer:

```text
odds-lab model verify-bundle --bundle-dir <bundle-directory>
```

`notebooks/03_gated_comparison_colab.ipynb` mounts Google Drive, clones the
repository at an explicit commit, verifies the bundle and starts one selected
candidate/ablation. Run outputs and heartbeats are written directly to Drive.
Every invocation performs a fresh immutable preflight; a previously accepted
preflight authorizes bundling but is not reused to bypass this check.

The comparison command is:

```text
odds-lab --project-dir <bundle-directory> model compare \
  --candidate dixon_coles_shared_encoder_pooling_gated \
  --ablation base --epochs 10 --max-iter 20 --device cuda \
  --run-root <persistent-run-directory>
```

Repeat `--candidate` or `--ablation` to select multiple stages. CUDA applies
only to the neural candidate; the scikit-learn tabular challenger remains
CPU-only. The default device is CPU, preserving local deterministic behavior.

After OOS evaluation has selected an ablation, fit a non-operational final
gated-neural artifact on all eligible cross-fitted residual targets:

```text
odds-lab --project-dir <bundle-directory> model fit-final-gated \
  --ablation combined --epochs 80 --device cuda \
  --code-version <tested-git-commit> \
  --run-root <persistent-final-run-directory>
```

This command performs a fresh mandatory preflight and writes
`artifacts/final_model/model.pt` plus `metadata.json` under a new immutable run
directory. The metadata binds the checkpoint to its dataset checksum, code
commit, feature ablation, seed, hyperparameters and deterministic reliability
gate. It does not promote or overwrite the official operational model.

## Artifacts

Canonical database: `data/football_odds.sqlite3`.

Important processed artifacts include:

- `data/processed/player_training_ready.csv`;
- `data/processed/player_match_temporal_features.csv`;
- neural model, metadata, OOS predictions and metrics under
  `reports/modeling/neural_lineup_model/`;
- market and strategy evidence under `reports/`.

Artifacts from retired models remain versioned historical evidence.
