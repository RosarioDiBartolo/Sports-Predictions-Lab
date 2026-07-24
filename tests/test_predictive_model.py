import joblib
import numpy as np
import pandas as pd
import pytest

from football_odds.predictive_model import (
    export_sport_model,
    fit_sport_model,
    load_sport_model,
    predict_fixtures,
    walk_forward_sport_model,
)


def _feature_rows() -> pd.DataFrame:
    rows = []
    outcomes = ("H", "D", "A")
    for season_index, season in enumerate(("2021", "2122", "2223")):
        for index in range(18):
            rows.append(
                {
                    "match_id": f"{season}-{index}",
                    "date": f"20{20 + season_index}-08-{index + 1:02d}",
                    "season": season,
                    "league": "I1" if index % 2 else "E0",
                    "home_team": f"Home {index % 4}",
                    "away_team": f"Away {index % 5}",
                    "result": outcomes[index % 3],
                    "elo_difference": float((index % 7) * 20 - 60),
                    "elo_expected_home": 0.45 + (index % 5) * 0.03,
                    "home_matches_played": index,
                    "away_matches_played": index + 1,
                    "home_rest_days": np.nan if index == 0 else 7.0,
                    "away_rest_days": np.nan if index == 0 else 6.0,
                    "home_points_5": float(index % 3),
                    "away_points_5": float((index + 1) % 3),
                    "market_home_probability": 0.95 if index % 2 else 0.02,
                    "market_draw_probability": 0.03,
                    "market_away_probability": 0.02 if index % 2 else 0.95,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_sport_model_outputs_probabilities_without_market_inputs():
    features = _feature_rows()
    metrics, predictions = walk_forward_sport_model(features)
    assert set(metrics["season"]) == {"2122", "2223"}
    assert set(metrics["model"]) == {"sport_gradient_boosting"}
    assert predictions["match_id"].str.startswith("2021").sum() == 0
    assert predictions.filter(like="probability_").sum(axis=1).eq(1).all()

    changed_market = features.copy()
    changed_market[
        [
            "market_home_probability",
            "market_draw_probability",
            "market_away_probability",
        ]
    ] = changed_market[
        [
            "market_away_probability",
            "market_draw_probability",
            "market_home_probability",
        ]
    ].to_numpy()
    _, changed_predictions = walk_forward_sport_model(changed_market)
    probability_columns = [
        "probability_home",
        "probability_draw",
        "probability_away",
    ]
    pd.testing.assert_frame_equal(
        predictions[probability_columns],
        changed_predictions[probability_columns],
    )


def test_fit_and_predict_fixtures_rejects_targets_and_keeps_identity():
    model = fit_sport_model(_feature_rows())
    fixtures = _feature_rows().iloc[:2].drop(columns=["result"]).copy()
    predictions = predict_fixtures(model, fixtures)
    assert list(predictions["match_id"]) == list(fixtures["match_id"])
    assert predictions["predicted_result"].isin(("H", "D", "A")).all()
    assert predictions.filter(like="probability_").sum(axis=1).eq(1).all()

    with pytest.raises(ValueError, match="non deve contenere target"):
        predict_fixtures(model, _feature_rows().iloc[:1])


def test_sport_model_requires_complete_outcomes():
    incomplete = _feature_rows().loc[lambda frame: frame["result"].ne("A")]
    with pytest.raises(ValueError, match="H, D e A"):
        fit_sport_model(incomplete)


def test_export_load_and_stable_empty_artifacts(tmp_path):
    result = export_sport_model(_feature_rows(), tmp_path / "complete")
    assert result.outputs["model"].exists()
    assert result.outputs["report"].exists()
    assert "market_closing" in result.outputs["report"].read_text(encoding="utf-8")
    loaded = load_sport_model(result.outputs["model"])
    assert loaded.numeric_features == result.predictor.numeric_features

    incomplete = _feature_rows().loc[lambda frame: frame["result"].eq("H")]
    empty = export_sport_model(incomplete, tmp_path / "incomplete")
    assert empty.metrics.empty
    assert "model" not in empty.outputs
    assert "Modello non esportato" in empty.outputs["report"].read_text(
        encoding="utf-8"
    )

    invalid = tmp_path / "invalid.joblib"
    joblib.dump({"not": "a model"}, invalid)
    with pytest.raises(TypeError, match="non valido"):
        load_sport_model(invalid)
