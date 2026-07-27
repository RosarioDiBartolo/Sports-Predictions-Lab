import json

import numpy as np
import pandas as pd
import pytest

from football_odds.confirmed_lineup import (
    CONFIRMED_LINEUP_BASELINE_NAME,
    CONFIRMED_LINEUP_MODEL_NAME,
    export_confirmed_lineup_model,
    fit_confirmed_lineup_model,
    load_confirmed_lineup_model,
    walk_forward_confirmed_lineup_model,
)


def _features() -> pd.DataFrame:
    rows = []
    for season_index, season in enumerate(("2021", "2122", "2223")):
        for index in range(24):
            lineup_edge = (index % 6 - 2.5) / 2.5
            home_goals = max(0, int(round(1.3 + 0.5 * lineup_edge)))
            away_goals = max(0, int(round(1.1 - 0.3 * lineup_edge)))
            if index % 5 == 0:
                home_goals = 0
            if index % 7 == 0:
                away_goals = 2
            rows.append(
                {
                    "match_id": f"{season}-{index}",
                    "season": season,
                    "league": "I1",
                    "home_team": f"H{index % 6}",
                    "away_team": f"A{index % 7}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": (
                        "H" if home_goals > away_goals
                        else "A" if home_goals < away_goals
                        else "D"
                    ),
                    "elo_difference": float(index * 3 - 30),
                    "home_points_5": float(index % 3),
                    "away_points_5": float((index + season_index) % 3),
                    "home_confirmed_lineup_strength": lineup_edge,
                    "away_confirmed_lineup_strength": -lineup_edge,
                    "home_confirmed_lineup_continuity": (index % 11) / 10,
                    "away_confirmed_lineup_continuity": ((index + 3) % 11) / 10,
                    "market_home_probability": 0.99,
                    "market_draw_probability": 0.005,
                    "market_away_probability": 0.005,
                }
            )
    return pd.DataFrame(rows)


def test_confirmed_lineup_model_is_regularized_and_normalized():
    features = _features()
    predictor = fit_confirmed_lineup_model(features, alpha=50.0)
    home_rates, away_rates = predictor.goal_rates(features.iloc[:5])
    probabilities = predictor.predict_proba(features.iloc[:5])
    assert predictor.alpha == 50.0
    assert (home_rates > 0).all() and (away_rates > 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    base_home = predictor.base.home_goal_model.predict(features.iloc[:5])
    assert np.max(np.abs(np.log(home_rates / base_home))) <= 0.35 + 1e-9


def test_confirmed_lineup_walk_forward_uses_common_folds_and_metrics():
    metrics, predictions = walk_forward_confirmed_lineup_model(_features())
    assert set(metrics["season"]) == {"2122", "2223"}
    assert set(metrics["model"]) == {
        CONFIRMED_LINEUP_MODEL_NAME,
        CONFIRMED_LINEUP_BASELINE_NAME,
    }
    assert {"log_loss", "brier", "rps", "accuracy", "ece"}.issubset(metrics)
    fold_counts = metrics.pivot(index="season", columns="model", values="matches")
    assert (fold_counts.iloc[:, 0] == fold_counts.iloc[:, 1]).all()
    assert predictions["match_id"].str.startswith("2021").sum() == 0

    changed_market = _features()
    changed_market["market_home_probability"] = 0.001
    _, changed = walk_forward_confirmed_lineup_model(changed_market)
    pd.testing.assert_frame_equal(
        predictions.filter(like="probability_"),
        changed.filter(like="probability_"),
    )


def test_confirmed_lineup_export_cannot_bypass_promotion_gate(tmp_path):
    result = export_confirmed_lineup_model(_features(), tmp_path)
    metadata = json.loads(result.outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["official_model_unchanged"] is True
    assert {"rps_not_worse", "promoted"}.issubset(metadata["promotion"])
    assert result.outputs["report"].exists()
    assert load_confirmed_lineup_model(result.outputs["model"]).alpha == 25.0


def test_confirmed_lineup_contract_rejects_missing_inputs():
    with pytest.raises(ValueError, match="Nessuna feature"):
        fit_confirmed_lineup_model(
            _features().drop(
                columns=[
                    "home_confirmed_lineup_strength",
                    "away_confirmed_lineup_strength",
                    "home_confirmed_lineup_continuity",
                    "away_confirmed_lineup_continuity",
                ]
            )
        )
    predictor = fit_confirmed_lineup_model(_features())
    with pytest.raises(ValueError, match="Feature formazione confermata mancanti"):
        predictor.predict_proba(
            _features().drop(columns=["home_confirmed_lineup_strength"])
        )
