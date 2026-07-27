# Sports Predictions Lab

Research platform for evaluating bookmaker probabilities and building
leakage-safe football modeling datasets.

## Run the canonical pipeline

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
odds-lab build all
```

The pipeline resolves one canonical dependency graph:

```text
01 ingest:    raw CSV → SQLite canonical tables
02 enrich:    teams + matches → venues + weather
03 analytics: canonical tables → analytics_dataset.csv + research tables
04 market:    matches + odds → calibration and market metrics
05 features:  canonical history → modeling_features.csv
06 baselines: modeling_features.csv → baseline predictions + metrics
07 model:     modeling_features.csv → model + OOS evidence
08 predict:   model + history + fixtures → upcoming_predictions.csv
```

Target individual outputs when needed:

```powershell
odds-lab build ingest
odds-lab build analytics
odds-lab build market
odds-lab build features
odds-lab build baselines
odds-lab build model
odds-lab build hybrid
odds-lab build edge
odds-lab sport-model
```

Use `--refresh` to refresh provider caches and `--rebuild-features` to ignore
an existing feature artifact.

`build all` runs the canonical production graph and deliberately excludes the
experimental `hybrid` and `edge` targets. Run those targets explicitly when
evaluating a candidate model or a frozen betting rule.

Predict a target-free fixture file after training the model:

```powershell
odds-lab predict --fixtures upcoming_fixtures.csv
```

The fixture CSV requires `date`, `season`, `league`, `home_team` and
`away_team`; `match_id` is optional. The sport-only model never consumes odds.
Its sports inputs include leakage-safe rolling shots, shots on target, corners,
cards, finishing rates, venue splits and exponentially weighted recent form.
The model report includes paired-bootstrap uncertainty and an explicit
promotion gate against the logistic sports baseline.

The execution manifest is written to `reports/pipeline_manifest.json`. For each
executed stage it records exact inputs, outputs, row counts and status.

## Living documentation

Architecture, features, commands, artifacts, dependencies and guarantees are
maintained in `docs-app/src/docs.ts`. It is both the LLM-readable source of
truth and the data source rendered by the documentation webapp.

## Quality

```powershell
pytest
ruff check src tests
mypy src
```
