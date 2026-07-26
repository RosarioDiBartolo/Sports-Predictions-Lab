import numpy as np
import pandas as pd
import pytest

from football_odds.hybrid import (
    HYBRID_MODEL_NAME,
    HYBRID_WITHOUT_PLAYERS,
    _dc_tau,
    _fit_rho,
    export_hybrid_model,
    fit_hybrid_model,
    load_hybrid_model,
    walk_forward_hybrid_model,
)


def _features() -> pd.DataFrame:
    rows = []
    for season_index, season in enumerate(("2021", "2122", "2223")):
        for index in range(18):
            home_goals = index % 4
            away_goals = (index + season_index) % 3
            rows.append(
                {
                    "match_id": f"{season}-{index}",
                    "season": season,
                    "league": "I1",
                    "home_team": f"H{index % 5}",
                    "away_team": f"A{index % 6}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": "H" if home_goals > away_goals else "A"
                    if home_goals < away_goals else "D",
                    "elo_difference": float(index * 5 - 40),
                    "home_points_5": float(index % 3),
                    "away_points_5": float((index + 1) % 3),
                    "market_home_probability": 0.99,
                    "market_draw_probability": 0.005,
                    "market_away_probability": 0.005,
                }
            )
    return pd.DataFrame(rows)


def test_hybrid_is_walk_forward_sport_only_and_normalized():
    features = _features()
    metrics, predictions = walk_forward_hybrid_model(features)
    assert set(metrics["season"]) == {"2122", "2223"}
    assert set(metrics["model"]) == {HYBRID_MODEL_NAME}
    assert predictions["match_id"].str.startswith("2021").sum() == 0
    np.testing.assert_allclose(
        predictions.filter(like="probability_").sum(axis=1), 1.0
    )

    changed_market = features.copy()
    changed_market["market_home_probability"] = 0.001
    _, changed = walk_forward_hybrid_model(changed_market)
    pd.testing.assert_frame_equal(
        predictions.filter(like="probability_"),
        changed.filter(like="probability_"),
    )


def test_hybrid_models_home_and_away_goals_separately():
    predictor = fit_hybrid_model(_features())
    assert predictor.home_goal_model is not predictor.away_goal_model
    assert -0.2 <= predictor.rho <= 0.2
    probabilities = predictor.predict_proba(_features().iloc[:3])
    assert probabilities.shape == (3, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_hybrid_export_runs_same_gate_without_replacing_official(tmp_path):
    result = export_hybrid_model(_features(), tmp_path)
    metadata = __import__("json").loads(
        result.outputs["metadata"].read_text(encoding="utf-8")
    )
    assert result.outputs["model"].exists()
    assert result.outputs["report"].exists()
    assert metadata["official_model_unchanged"] is True
    assert "promotion" in metadata
    assert result.outputs["player_ablation_metrics"].exists()
    assert result.outputs["player_ablation_summary"].exists()
    assert result.outputs["player_ablation_bootstrap"].exists()
    assert load_hybrid_model(result.outputs["model"]).rho == result.predictor.rho
    assert {
        HYBRID_MODEL_NAME,
        "sport_gradient_boosting",
        "market_closing",
    }.issubset(set(pd.read_csv(result.outputs["comparison"])["model"]))


def test_hybrid_player_ablation_uses_identical_walk_forward_folds():
    features = _features()
    features["home_player_expected_attack"] = np.arange(len(features))
    features["away_player_expected_attack"] = np.arange(len(features))[::-1]
    with_players, _ = walk_forward_hybrid_model(features)
    without_players, _ = walk_forward_hybrid_model(
        features,
        include_player_features=False,
    )
    assert set(without_players["model"]) == {HYBRID_WITHOUT_PLAYERS}
    assert with_players[["season", "matches"]].equals(
        without_players[["season", "matches"]]
    )
    predictor = fit_hybrid_model(features, include_player_features=False)
    assert not any("_player_" in name for name in predictor.numeric_features)


def test_hybrid_rejects_incomplete_feature_contracts():
    predictor = fit_hybrid_model(_features())
    with pytest.raises(ValueError, match="Feature ibride mancanti"):
        predictor.predict_proba(_features().drop(columns=["home_team"]))
    with pytest.raises(ValueError, match="Colonne modello ibrido mancanti"):
        fit_hybrid_model(_features().drop(columns=["away_goals"]))
    missing_team = _features()
    missing_team.loc[0, "home_team"] = np.nan
    with pytest.raises(ValueError, match="squadre e gol completi"):
        fit_hybrid_model(missing_team)


def test_hybrid_low_score_fallbacks_are_stable():
    assert _dc_tau(2, 2, 1.2, 1.0, -0.1) == 1.0
    no_low_scores = pd.DataFrame({"home_goals": [2], "away_goals": [3]})
    assert _fit_rho(no_low_scores, np.array([1.2]), np.array([1.0])) == 0.0
    sparse_history = _features()
    sparse_history.loc[sparse_history["season"].eq("2021"), "result"] = "H"
    metrics, predictions = walk_forward_hybrid_model(sparse_history)
    assert "2122" not in set(metrics["season"])
    assert predictions["match_id"].str.startswith("2122").sum() == 0
