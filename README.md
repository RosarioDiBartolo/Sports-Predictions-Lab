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
acquire → normalize → SQLite
                     ├→ analytics → market evaluation
                     └→ pre-match features → walk-forward baselines
```

Target individual outputs when needed:

```powershell
odds-lab build ingest
odds-lab build analytics
odds-lab build market
odds-lab build features
odds-lab build baselines
odds-lab build model
odds-lab sport-model
```

Use `--refresh` to refresh provider caches and `--rebuild-features` to ignore
an existing feature artifact.

Predict a target-free fixture file after training the model:

```powershell
odds-lab predict --fixtures upcoming_fixtures.csv
```

The fixture CSV requires `date`, `season`, `league`, `home_team` and
`away_team`; `match_id` is optional. The sport-only model never consumes odds.

The execution manifest is written to `reports/pipeline_manifest.json`.

## Living documentation

Architecture, features, commands, artifacts, dependencies and guarantees are
maintained in `docs-app/src/docs.ts`. It is both the LLM-readable source of
truth and the data source rendered by the documentation webapp.

`PROJECT_CONTEXT.md`, `docs/ARCHITECTURE.md` and presentations are historical
snapshots, not maintained documentation.

## Quality

```powershell
pytest
ruff check src tests
mypy src
```
