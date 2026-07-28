import numpy as np

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import FootballDataProvider, IngestionPipeline
from football_odds.market.research import (
    analyze_odds_ranges,
    analyze_predictions,
    build_analytics_dataset,
    compare_opening_closing,
)
from tests.ingestion.test_provider_ingestion import sample_frame


def _dataset(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    IngestionPipeline(database).run(FootballDataProvider(sample_frame()))
    return build_analytics_dataset(database, bin_width=0.1)


def test_analytics_dataset_has_one_row_per_selection(tmp_path):
    frame = _dataset(tmp_path)
    assert len(frame) == 6
    assert {
        "prediction_correct",
        "favorite",
        "favorite_won",
        "logloss_contribution",
        "brier_contribution",
        "roi",
    }.issubset(frame.columns)
    assert frame["favorite"].sum() == 2
    assert frame["favorite_won"].sum() == 2


def test_analyzer_outputs_research_metrics(tmp_path):
    frame = _dataset(tmp_path)
    metrics = analyze_predictions(frame)
    assert metrics["predictions"] == 6
    assert metrics["accuracy"] == 1.0
    assert metrics["sharpness"] >= 0
    assert np.isfinite(metrics["expected_calibration_error"])
    expected_signed_error = (
        frame["prediction_correct"].mean() - frame["implied_probability"].mean()
    )
    assert np.isclose(metrics["calibration_error"], expected_signed_error)
    closing = frame[frame["opening_or_closing"] == "closing"]
    true_probability = closing.loc[
        closing["prediction_correct"], "implied_probability"
    ].iloc[0]
    assert np.isclose(metrics["log_loss"], -np.log(true_probability), atol=0.1)
    assert closing["brier_contribution"].nunique() == 1


def test_odds_ranges_and_opening_closing(tmp_path):
    frame = _dataset(tmp_path)
    ranges = analyze_odds_ranges(frame)
    timing = compare_opening_closing(frame)
    assert ranges["predictions"].sum() == 6
    assert set(timing["timing"]) == {"opening", "closing"}
    assert timing["mean_absolute_probability_movement"].notna().all()
