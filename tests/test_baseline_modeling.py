import numpy as np
import pandas as pd
import pytest

from football_odds.baseline_modeling import (
    _league_frequencies,
    _sport_feature_allowlist,
    export_baseline_report,
    walk_forward_baselines,
)


def test_walk_forward_uses_only_later_seasons_for_testing(tmp_path):
    rows = []
    for season in ("2021", "2122", "2223"):
        for index, result in enumerate(("H", "D", "A") * 4):
            rows.append(
                {
                    "match_id": f"{season}-{index}",
                    "season": season,
                    "league": "I1",
                    "result": result,
                    "elo_difference": float(index - 5),
                    "home_points_5": np.nan if index == 0 else float(index % 3),
                    "home_goals": index % 4,
                    "away_goals": index % 2,
                    "market_home_probability": 0.45,
                    "market_draw_probability": 0.28,
                    "market_away_probability": 0.27,
                }
            )
    metrics, predictions = walk_forward_baselines(pd.DataFrame(rows))
    assert set(metrics["season"]) == {"2122", "2223"}
    assert set(metrics["model"]) == {
        "historical_frequency",
        "elo",
        "sport_features",
        "market_closing",
    }
    assert predictions["match_id"].str.startswith("2021").sum() == 0
    assert predictions.filter(like="probability_").sum(axis=1).round(10).eq(1).all()
    exported = export_baseline_report(pd.DataFrame(rows), tmp_path)
    assert exported.outputs["report"].exists()
    assert "market_closing" in exported.outputs["report"].read_text(encoding="utf-8")


def test_baselines_reject_incomplete_dataset():
    with pytest.raises(ValueError, match="Colonne baseline mancanti"):
        walk_forward_baselines(pd.DataFrame({"season": ["2425"]}))


def test_frequency_baseline_falls_back_for_unseen_league():
    train = pd.DataFrame({"league": ["I1", "I1", "I1"], "result": ["H", "D", "A"]})
    probabilities = _league_frequencies(train, pd.DataFrame({"league": ["NEW"]}))
    assert probabilities.shape == (1, 3)
    assert probabilities.sum() == pytest.approx(1.0)


def test_sport_feature_allowlist_rejects_unknown_numeric_columns():
    columns = pd.Index(["elo_difference", "home_points_5", "future_final_score"])
    assert _sport_feature_allowlist(columns) == [
        "elo_difference",
        "home_points_5",
    ]


def test_single_season_returns_stable_empty_outputs(tmp_path):
    features = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "season": "2425",
                "league": "I1",
                "result": "H",
                "elo_difference": 0.0,
            }
        ]
    )
    metrics, predictions = walk_forward_baselines(features)
    assert metrics.empty
    assert predictions.empty
    assert list(predictions.columns) == [
        "match_id",
        "season",
        "league",
        "result",
        "model",
        "probability_home",
        "probability_draw",
        "probability_away",
    ]
    assert export_baseline_report(features, tmp_path).outputs["report"].exists()


def test_logistic_models_wait_until_all_outcomes_exist():
    features = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "season": "2324",
                "league": "I1",
                "result": "H",
                "elo_difference": 0.0,
            },
            {
                "match_id": "m2",
                "season": "2425",
                "league": "I1",
                "result": "D",
                "elo_difference": 1.0,
            },
        ]
    )
    metrics, _ = walk_forward_baselines(features)
    assert set(metrics["model"]) == {"historical_frequency"}
