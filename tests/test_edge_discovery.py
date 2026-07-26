import json

import pandas as pd
import pytest

from football_odds.edge import (
    _bootstrap_roi,
    discover_edges,
    export_edge_discovery,
    prepare_edge_dataset,
)


def _inputs():
    predictions = []
    features = []
    analytics = []
    seasons = ("2021", "2122", "2223", "2324")
    for season_index, season in enumerate(seasons):
        for index in range(20):
            match_id = f"{season}-{index}"
            result = "H" if season_index >= 2 or index % 4 else "A"
            predictions.append(
                {
                    "match_id": match_id,
                    "season": season,
                    "league": "I1",
                    "result": result,
                    "probability_home": 0.70,
                    "probability_draw": 0.18,
                    "probability_away": 0.12,
                }
            )
            features.append(
                {
                    "match_id": match_id,
                    "elo_difference": 120.0,
                    "home_matches_played": 10,
                    "away_matches_played": 10,
                    "market_home_probability": 0.55,
                    "market_draw_probability": 0.25,
                    "market_away_probability": 0.20,
                }
            )
            for selection, odds in (("H", 1.8), ("D", 4.0), ("A", 5.0)):
                analytics.append(
                    {
                        "match_id": match_id,
                        "bookmaker": "Market Average",
                        "selection": selection,
                        "odds": odds,
                        "opening_or_closing": "closing",
                    }
                )
    return (
        pd.DataFrame(predictions),
        pd.DataFrame(features),
        pd.DataFrame(analytics),
    )


def test_edge_discovery_freezes_rule_and_promotes_only_on_holdout_evidence(tmp_path):
    predictions, features, analytics = _inputs()
    data = prepare_edge_dataset(predictions, features, analytics)
    assert len(data) == len(predictions)
    assert data["pick"].eq("H").all()
    result = export_edge_discovery(
        predictions,
        features,
        analytics,
        tmp_path,
        holdout_seasons=("2223", "2324"),
        minimum_discovery_bets=10,
        bootstrap_samples=200,
    )
    assert result.promoted
    assert result.selected_rule["holdout_seasons"] == ["2223", "2324"]
    assert result.summary.loc[
        result.summary["period"].eq("holdout"), "ci_low"
    ].iloc[0] > 0
    assert result.season_stability["roi"].gt(0).all()
    assert all(path.exists() for path in result.outputs.values())
    saved = json.loads(result.outputs["rule"].read_text(encoding="utf-8"))
    assert saved["promoted"] is True
    assert "Quote Market Average closing" in result.outputs["report"].read_text(
        encoding="utf-8"
    )


def test_edge_dataset_ignores_non_average_and_opening_odds():
    predictions, features, analytics = _inputs()
    extra = analytics.iloc[:1].copy()
    extra["bookmaker"] = "Maximum Odds"
    extra["odds"] = 99.0
    opening = analytics.iloc[:1].copy()
    opening["opening_or_closing"] = "opening"
    opening["odds"] = 88.0
    data = prepare_edge_dataset(
        predictions, features, pd.concat([analytics, extra, opening])
    )
    assert data.iloc[0]["odds_H"] == 1.8


def test_edge_discovery_rejects_invalid_analysis_contracts():
    with pytest.raises(ValueError, match="positivo"):
        _bootstrap_roi(pd.Series([1.0]).to_numpy(), samples=0)
    empty = _bootstrap_roi(pd.Series(dtype=float).to_numpy(), samples=10)
    assert pd.isna(empty["roi"])
    predictions, features, analytics = _inputs()
    data = prepare_edge_dataset(predictions, features, analytics)
    with pytest.raises(ValueError, match="holdout"):
        discover_edges(data, holdout_seasons=("9999",), bootstrap_samples=10)
