from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .analytics_dataset import save_analytics_dataset
from .baseline_modeling import BaselineResult, export_baseline_report
from .calibration import (
    calibration_table,
    expected_calibration_error,
    plot_calibration,
    to_long_calibration,
)
from .config import AnalysisConfig, BackfillConfig, ModelingConfig
from .data import load_all_seasons
from .database import ResearchDatabase
from .ingestion import IngestionPipeline, IngestionSummary
from .metrics import calculate_metrics, metrics_by_season
from .modeling_dataset import (
    build_fixture_features,
    build_prematch_features,
    load_canonical_matches,
)
from .modeling_reporting import export_modeling_report
from .odds import prepare_matches
from .predictive_model import (
    SportModelResult,
    export_sport_model,
    load_sport_model,
    predict_fixtures,
)
from .providers.football_data import FootballDataProvider
from .reporting import export_research_report


@dataclass
class AnalysisResult:
    matches: pd.DataFrame
    calibration: pd.DataFrame
    season_metrics: pd.DataFrame
    metrics: dict[str, float | int]


@dataclass
class ResearchResult:
    """Artifacts created by the extended research pipeline."""

    ingestion: IngestionSummary
    analytics: pd.DataFrame
    outputs: dict[str, Path]


@dataclass
class ModelingResult:
    """Leakage-safe feature dataset and its diagnostics."""

    features: pd.DataFrame
    outputs: dict[str, Path]


@dataclass
class BackfillResult:
    """Summary of a complete, idempotent multi-league import."""

    ingestion: IngestionSummary
    analytics: pd.DataFrame
    manifest: Path


@dataclass
class PipelineResult:
    """Artifacts and row counts produced by the unified stage graph."""

    completed_stages: list[str]
    artifacts: dict[str, Path]
    counts: dict[str, int]


@dataclass
class FixturePredictionResult:
    """Probabilities produced for target-free future fixtures."""

    predictions: pd.DataFrame
    output: Path


def run_analysis(
    config: AnalysisConfig | None = None, *, refresh: bool = False
) -> AnalysisResult:
    config = config or AnalysisConfig()
    raw = load_all_seasons(config, refresh=refresh)
    matches = prepare_matches(raw)
    long_frame = to_long_calibration(matches)
    table = calibration_table(long_frame, config.bin_width)
    metrics = calculate_metrics(matches)
    metrics["expected_calibration_error"] = expected_calibration_error(table)
    seasons = metrics_by_season(matches)
    save_outputs(config, matches, table, seasons, metrics)
    return AnalysisResult(matches, table, seasons, metrics)


def run_research_pipeline(
    config: AnalysisConfig | None = None, *, refresh: bool = False
) -> ResearchResult:
    """Run provider → database → analytics → report without changing the MVP."""
    config = config or AnalysisConfig()
    raw = load_all_seasons(config, refresh=refresh)
    database = ResearchDatabase(config.database_path)
    ingestion = IngestionPipeline(database).run(FootballDataProvider(raw))
    analytics = save_analytics_dataset(config, database)
    outputs = export_research_report(analytics, config.report_dir / "research")
    return ResearchResult(ingestion, analytics, outputs)


def run_backfill_pipeline(
    config: BackfillConfig | None = None, *, refresh: bool = False
) -> BackfillResult:
    """Backfill every configured league/season into one canonical database."""
    config = config or BackfillConfig()
    config.validate()
    database = ResearchDatabase(config.database_path)
    total_matches = total_selections = 0
    imported: list[dict[str, object]] = []
    for league in config.leagues:
        analysis = AnalysisConfig(
            league=league,
            seasons=config.seasons,
            project_dir=config.project_dir,
            database_name=config.database_name,
        )
        raw = load_all_seasons(analysis, refresh=refresh)
        summary = IngestionPipeline(database).run(FootballDataProvider(raw))
        total_matches += summary.matches
        total_selections += summary.odds_selections
        imported.append(
            {
                "league": league,
                "seasons": list(config.seasons),
                "source_rows": len(raw),
                "odds_selections_processed": summary.odds_selections,
            }
        )
    analytics_config = AnalysisConfig(
        project_dir=config.project_dir,
        database_name=config.database_name,
    )
    analytics = save_analytics_dataset(analytics_config, database)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = config.report_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": "football-data.co.uk",
                "leagues": list(config.leagues),
                "seasons": list(config.seasons),
                "source_matches_processed": total_matches,
                "odds_selections_processed": total_selections,
                "analytics_rows": len(analytics),
                "imports": imported,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return BackfillResult(
        IngestionSummary(total_matches, total_selections), analytics, manifest
    )


def run_modeling_pipeline(
    config: ModelingConfig | None = None, *, refresh: bool = False
) -> ModelingResult:
    """Compatibility wrapper for the canonical feature target."""
    config = config or ModelingConfig()
    result = run_unified_pipeline(
        config,
        targets=("features",),
        refresh=refresh,
        rebuild_features=True,
    )
    feature_path = result.artifacts["features"]
    features = pd.read_csv(feature_path, parse_dates=["date"])
    outputs = {
        key.removeprefix("modeling_"): value
        for key, value in result.artifacts.items()
        if key.startswith("modeling_")
    }
    outputs["features"] = feature_path
    return ModelingResult(features, outputs)


def run_unified_pipeline(
    config: ModelingConfig | None = None,
    *,
    targets: tuple[str, ...] = ("all",),
    refresh: bool = False,
    rebuild_features: bool = False,
) -> PipelineResult:
    """Resolve and execute canonical pipeline stages for the requested targets."""
    config = config or ModelingConfig()
    config.validate()
    requested = set(targets)
    allowed = {
        "all",
        "ingest",
        "analytics",
        "market",
        "features",
        "baselines",
        "model",
    }
    unknown = requested.difference(allowed)
    if unknown:
        raise ValueError(f"Target pipeline sconosciuti: {sorted(unknown)}")
    if "all" in requested:
        requested = allowed.difference({"all"})

    completed: list[str] = []
    artifacts: dict[str, Path] = {}
    counts: dict[str, int] = {}
    database_path = config.project_dir / "data" / "football_odds.sqlite3"
    database = ResearchDatabase(database_path)

    # All downstream stages depend on canonical ingestion.
    for league in config.leagues:
        analysis = AnalysisConfig(
            league=league,
            seasons=config.seasons,
            project_dir=config.project_dir,
        )
        raw = load_all_seasons(analysis, refresh=refresh)
        summary = IngestionPipeline(database).run(FootballDataProvider(raw))
        counts["matches_ingested"] = counts.get("matches_ingested", 0) + summary.matches
        counts["odds_selections_ingested"] = (
            counts.get("odds_selections_ingested", 0) + summary.odds_selections
        )
    completed.append("ingest")
    artifacts["database"] = database_path

    analysis_config = AnalysisConfig(project_dir=config.project_dir)
    if requested.intersection({"analytics", "market"}):
        analytics = save_analytics_dataset(analysis_config, database)
        counts["analytics_rows"] = len(analytics)
        completed.append("analytics")
        artifacts["analytics"] = analysis_config.processed_dir / "analytics_dataset.csv"
        report_paths = export_research_report(
            analytics, analysis_config.report_dir / "research"
        )
        artifacts.update(
            {f"research_{key}": value for key, value in report_paths.items()}
        )

    canonical = load_canonical_matches(
        database, leagues=config.leagues, seasons=config.seasons
    )
    counts["canonical_matches"] = len(canonical)

    if "market" in requested:
        market = canonical.dropna(
            subset=[
                "market_home_probability",
                "market_draw_probability",
                "market_away_probability",
            ]
        ).rename(
            columns={
                "market_home_probability": "p_home",
                "market_draw_probability": "p_draw",
                "market_away_probability": "p_away",
                "market_margin": "margin",
                "result": "FTR",
                "season": "Season",
            }
        )
        table = calibration_table(
            to_long_calibration(market), analysis_config.bin_width
        )
        metrics = calculate_metrics(market)
        metrics["expected_calibration_error"] = expected_calibration_error(table)
        seasons = metrics_by_season(market)
        save_outputs(analysis_config, market, table, seasons, metrics)
        completed.append("market")
        artifacts.update(
            {
                "market_metrics": analysis_config.report_dir / "metrics.json",
                "market_calibration": analysis_config.report_dir
                / "calibration_curve.png",
            }
        )

    feature_path = config.processed_dir / "modeling_features.csv"
    feature_manifest_path = config.processed_dir / "modeling_features.meta.json"
    feature_cache_key = {
        "leagues": list(config.leagues),
        "seasons": list(config.seasons),
        "rolling_windows": list(config.rolling_windows),
        "elo_initial_rating": config.elo_initial_rating,
        "elo_k_factor": config.elo_k_factor,
        "elo_home_advantage": config.elo_home_advantage,
        "elo_season_regression": config.elo_season_regression,
        "canonical_hash": int(pd.util.hash_pandas_object(canonical, index=False).sum()),
    }
    features: pd.DataFrame | None = None
    if requested.intersection({"features", "baselines", "model"}):
        cached_key = None
        if feature_manifest_path.exists():
            try:
                cached_key = json.loads(
                    feature_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                cached_key = None
        if (
            feature_path.exists()
            and not rebuild_features
            and cached_key == feature_cache_key
        ):
            features = pd.read_csv(feature_path, parse_dates=["date"])
        else:
            features = build_prematch_features(canonical, config)
            config.ensure_directories()
            features.to_csv(feature_path, index=False)
            feature_manifest_path.write_text(
                json.dumps(feature_cache_key, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        counts["feature_rows"] = len(features)
        completed.append("features")
        artifacts["features"] = feature_path
        artifacts["features_manifest"] = feature_manifest_path
        modeling_paths = export_modeling_report(features, config.report_dir)
        artifacts.update(
            {f"modeling_{key}": value for key, value in modeling_paths.items()}
        )

    if "baselines" in requested:
        if features is None:
            raise RuntimeError("Le baseline richiedono il dataset delle feature.")
        baseline = export_baseline_report(features, config.report_dir / "baselines")
        completed.append("baselines")
        artifacts.update(
            {f"baseline_{key}": value for key, value in baseline.outputs.items()}
        )

    if "model" in requested:
        if features is None:
            raise RuntimeError("Il modello richiede il dataset delle feature.")
        sport_model = export_sport_model(
            features,
            config.report_dir / "sport_model",
        )
        completed.append("model")
        counts["sport_model_oos_predictions"] = len(sport_model.predictions)
        artifacts.update(
            {
                f"sport_model_{key}": value
                for key, value in sport_model.outputs.items()
            }
        )

    manifest_path = config.report_dir.parent / "pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "completed_stages": completed,
                "counts": counts,
                "artifacts": {key: str(value) for key, value in artifacts.items()},
                "leagues": list(config.leagues),
                "seasons": list(config.seasons),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    artifacts["manifest"] = manifest_path
    return PipelineResult(completed, artifacts, counts)


def run_baseline_pipeline(config: ModelingConfig | None = None) -> BaselineResult:
    """Compatibility wrapper for the canonical baseline target."""
    config = config or ModelingConfig()
    result = run_unified_pipeline(config, targets=("baselines",))
    baseline_dir = config.report_dir / "baselines"
    return BaselineResult(
        metrics=pd.read_csv(baseline_dir / "baseline_metrics_by_season.csv"),
        predictions=pd.read_csv(baseline_dir / "baseline_predictions.csv"),
        outputs={
            key.removeprefix("baseline_"): value
            for key, value in result.artifacts.items()
            if key.startswith("baseline_")
        },
    )


def run_sport_model_pipeline(
    config: ModelingConfig | None = None,
) -> SportModelResult:
    """Build features, evaluate the candidate and persist its final estimator."""
    config = config or ModelingConfig()
    result = run_unified_pipeline(config, targets=("model",))
    model_dir = config.report_dir / "sport_model"
    metrics = pd.read_csv(model_dir / "sport_model_metrics_by_season.csv")
    predictions = pd.read_csv(model_dir / "sport_model_predictions.csv")
    outputs = {
        key.removeprefix("sport_model_"): value
        for key, value in result.artifacts.items()
        if key.startswith("sport_model_")
    }
    predictor = (
        load_sport_model(outputs["model"]) if "model" in outputs else None
    )
    return SportModelResult(metrics, predictions, outputs, predictor)


def run_fixture_prediction(
    fixtures_path: Path,
    config: ModelingConfig | None = None,
    *,
    model_path: Path | None = None,
    output_path: Path | None = None,
) -> FixturePredictionResult:
    """Replay canonical history and predict a target-free fixture CSV."""
    config = config or ModelingConfig()
    config.validate()
    model_path = model_path or (
        config.report_dir / "sport_model" / "sport_model.joblib"
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modello non trovato: {model_path}. Eseguire prima `odds-lab sport-model`."
        )
    fixtures = pd.read_csv(fixtures_path)
    database = ResearchDatabase(
        config.project_dir / "data" / "football_odds.sqlite3"
    )
    history = load_canonical_matches(
        database,
        leagues=config.leagues,
        seasons=config.seasons,
    )
    fixture_features = build_fixture_features(history, fixtures, config)
    predictions = predict_fixtures(
        load_sport_model(model_path),
        fixture_features,
    )
    output_path = output_path or (
        config.report_dir / "sport_model" / "upcoming_predictions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return FixturePredictionResult(predictions, output_path)


def save_outputs(
    config: AnalysisConfig,
    matches: pd.DataFrame,
    table: pd.DataFrame,
    seasons: pd.DataFrame,
    metrics: dict[str, float | int],
) -> None:
    config.ensure_directories()
    matches.to_csv(config.processed_dir / "matches.csv", index=False)
    table.to_csv(config.report_dir / "calibration_table.csv", index=False)
    seasons.to_csv(config.report_dir / "metrics_by_season.csv", index=False)
    Path(config.report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_calibration(table, config.report_dir / "calibration_curve.png")
