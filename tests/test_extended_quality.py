import json
import sys

import pandas as pd
import pytest
from test_provider_ingestion import sample_frame

from football_odds import cli
from football_odds.config import AnalysisConfig, ModelingConfig
from football_odds.database import ResearchDatabase
from football_odds.market import (
    calculate_metrics,
    calibration_table,
    encode_results,
    expected_calibration_error,
    find_odds_columns,
    metrics_by_season,
    plot_calibration,
    prepare_matches,
    to_long_calibration,
)
from football_odds.pipeline import (
    run_analysis,
    run_research_pipeline,
    run_unified_pipeline,
)
from football_odds.research import (
    analyze_predictions,
    build_analytics_dataset,
    compare_bookmakers,
    compare_leagues,
    export_research_report,
    save_analytics_dataset,
)
from football_odds.sources import (
    OddsRecord,
    download_season,
    load_all_seasons,
    season_url,
)


def test_config_paths_directories_and_validation(tmp_path):
    config = AnalysisConfig(project_dir=tmp_path)
    config.ensure_directories()
    assert config.raw_dir.is_dir()
    assert config.processed_dir.is_dir()
    assert config.report_dir.is_dir()
    assert config.database_path == tmp_path / "data" / "football_odds.sqlite3"
    config.validate()
    with pytest.raises(ValueError):
        AnalysisConfig(league="", project_dir=tmp_path).validate()
    with pytest.raises(ValueError):
        AnalysisConfig(seasons=(), project_dir=tmp_path).validate()
    with pytest.raises(ValueError):
        AnalysisConfig(bin_width=0, project_dir=tmp_path).validate()


def test_data_download_cache_and_load(monkeypatch, tmp_path):
    csv = sample_frame().to_csv(index=False).encode()

    class Response:
        content = csv

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "football_odds.sources.requests.get", lambda *args, **kwargs: Response()
    )
    destination = tmp_path / "raw" / "season.csv"
    downloaded = download_season("2425", "I1", destination)
    cached = download_season("2425", "I1", destination)
    assert len(downloaded) == len(cached) == 1
    assert season_url("2425", "I1").endswith("/2425/I1.csv")

    config = AnalysisConfig(seasons=("2425",), project_dir=tmp_path)
    all_data = load_all_seasons(config)
    assert all_data.loc[0, "Season"] == "2425"
    assert all_data.loc[0, "League"] == "I1"


def test_odds_preparation_and_errors():
    frame = sample_frame()
    prepared = prepare_matches(frame)
    assert len(prepared) == 1
    assert prepared.loc[0, "odds_source"] == "B365CH/B365CD/B365CA"
    assert prepared.loc[0, ["p_home", "p_draw", "p_away"]].sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        find_odds_columns(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError):
        prepare_matches(frame.drop(columns="HomeTeam"))


def test_calibration_metrics_and_plot(tmp_path):
    prepared = prepare_matches(sample_frame())
    long = to_long_calibration(prepared)
    table = calibration_table(long, 0.1)
    assert len(long) == 3
    assert expected_calibration_error(table) >= 0
    assert pd.isna(
        expected_calibration_error(
            pd.DataFrame({"observations": [], "calibration_error": []})
        )
    )
    destination = tmp_path / "calibration.png"
    plot_calibration(table, destination)
    assert destination.exists()

    assert encode_results(pd.Series(["H", "D", "A"])).shape == (3, 3)
    with pytest.raises(ValueError):
        encode_results(pd.Series(["X"]))
    metrics = calculate_metrics(prepared)
    assert metrics["matches"] == 1
    assert len(metrics_by_season(prepared)) == 1
    with pytest.raises(ValueError):
        calculate_metrics(prepared.iloc[0:0])


def test_database_errors_and_empty_analytics(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    assert database.upsert_league("I1", "Serie A", "Italy") > 0
    empty = build_analytics_dataset(database)
    assert empty.empty
    assert "roi" in empty

    invalid = OddsRecord(
        provider_match_id="missing",
        bookmaker="Book",
        market="1X2",
        odds={"H": 1.0},
        timestamp=None,
        timing="snapshot",
    )
    with pytest.raises(ValueError):
        database.add_odds("Provider", invalid)
    valid = OddsRecord(
        provider_match_id="missing",
        bookmaker="Book",
        market="1X2",
        odds={"H": 2.0, "D": 3.0, "A": 4.0},
        timestamp=None,
        timing="snapshot",
    )
    with pytest.raises(KeyError):
        database.add_odds("Provider", valid)


def test_comparisons_reporting_and_dataset_save(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    from football_odds.sources import FootballDataProvider, IngestionPipeline

    IngestionPipeline(database).run(FootballDataProvider(sample_frame()))
    config = AnalysisConfig(project_dir=tmp_path, seasons=("2425",))
    frame = save_analytics_dataset(config, database)
    duplicated = frame.copy()
    duplicated["bookmaker"] = "Second Book"
    combined = pd.concat([frame, duplicated], ignore_index=True)
    assert len(compare_bookmakers(combined)) == 2
    assert len(compare_leagues(combined)) == 1
    with pytest.raises(ValueError):
        analyze_predictions(frame.iloc[0:0])

    paths = export_research_report(frame, tmp_path / "reports")
    assert paths["dashboard"].exists()
    dashboard = paths["dashboard"].read_text(encoding="utf-8").lower()
    assert "plotly" in dashboard
    assert "che cos’è?" in dashboard
    assert "come si legge?" in dashboard
    assert "un esempio facile" in dashboard
    assert "una quota non è una previsione certa" in dashboard


def test_both_pipelines_and_cli(monkeypatch, tmp_path, capsys):
    frame = sample_frame()
    monkeypatch.setattr(
        "football_odds.pipeline.load_all_seasons", lambda *a, **k: frame
    )
    config = AnalysisConfig(project_dir=tmp_path, seasons=("2425",))
    original = run_analysis(config)
    research = run_research_pipeline(config)
    assert original.metrics["matches"] == 1
    assert research.ingestion.matches == 1

    monkeypatch.setattr(cli, "run_analysis", lambda *a, **k: original)
    monkeypatch.setattr(sys, "argv", ["odds-lab", "run", "--seasons", "2425"])
    cli.main()
    assert '"matches": 1' in capsys.readouterr().out

    monkeypatch.setattr(cli, "run_research_pipeline", lambda *a, **k: research)
    monkeypatch.setattr(sys, "argv", ["odds-lab", "research", "--seasons", "2425"])
    cli.main()
    output = capsys.readouterr().out
    assert "analytics_rows" in output


def test_unified_pipeline_runs_from_canonical_database(monkeypatch, tmp_path):
    frame = sample_frame()
    monkeypatch.setattr(
        "football_odds.pipeline.load_all_seasons", lambda *a, **k: frame
    )
    result = run_unified_pipeline(
        ModelingConfig(
            project_dir=tmp_path,
            leagues=("I1",),
            seasons=("2425",),
            rolling_windows=(2,),
        ),
        targets=("features",),
        rebuild_features=True,
    )
    assert result.completed_stages == ["ingest", "features"]
    assert result.counts["canonical_matches"] == 1
    assert result.artifacts["features"].exists()
    assert result.artifacts["manifest"].exists()
    manifest = json.loads(result.artifacts["manifest"].read_text(encoding="utf-8"))
    assert [stage["id"] for stage in manifest["stages"]] == [
        "01-ingest",
        "05-features",
    ]
    assert "sqlite:matches" in manifest["stages"][0]["outputs"]
    assert manifest["stages"][1]["rows"]["features"] == 1


def test_unified_pipeline_all_targets_and_feature_reuse(monkeypatch, tmp_path):
    frame = sample_frame()
    monkeypatch.setattr(
        "football_odds.pipeline.load_all_seasons", lambda *a, **k: frame
    )

    class Baseline:
        outputs = {"report": tmp_path / "baseline.md"}

    def baseline_export(*args, **kwargs):
        Baseline.outputs["report"].write_text("baseline", encoding="utf-8")
        return Baseline()

    monkeypatch.setattr(
        "football_odds.pipeline.export_baseline_report", baseline_export
    )
    config = ModelingConfig(
        project_dir=tmp_path,
        leagues=("I1",),
        seasons=("2425",),
        rolling_windows=(2,),
    )
    first = run_unified_pipeline(config, targets=("all",), rebuild_features=True)
    assert first.completed_stages == [
        "ingest",
        "analytics",
        "market",
        "features",
        "baselines",
        "model",
    ]
    assert first.artifacts["market_metrics"].exists()
    second = run_unified_pipeline(config, targets=("features",))
    assert second.counts["feature_rows"] == 1
    assert second.artifacts["features_manifest"].exists()
    changed = run_unified_pipeline(
        ModelingConfig(
            project_dir=tmp_path,
            leagues=("I1",),
            seasons=("2425",),
            rolling_windows=(3,),
        ),
        targets=("features",),
    )
    changed_features = pd.read_csv(changed.artifacts["features"])
    assert "home_points_3" in changed_features
    with pytest.raises(ValueError, match="sconosciuti"):
        run_unified_pipeline(config, targets=("unknown",))
