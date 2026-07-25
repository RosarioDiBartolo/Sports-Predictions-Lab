import pandas as pd
import pytest

from football_odds.config import ModelingConfig
from football_odds.database import ResearchDatabase
from football_odds.features import (
    build_fixture_features,
    build_prematch_features,
    load_canonical_matches,
    normalize_team_name,
    prepare_future_fixtures,
    prepare_modeling_matches,
)
from football_odds.sources import FootballDataProvider, IngestionPipeline


def match_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["01/08/2024", "08/08/2024", "15/08/2024"],
            "Season": ["2425", "2425", "2425"],
            "League": ["I1", "I1", "I1"],
            "HomeTeam": ["A", "B", "A"],
            "AwayTeam": ["B", "A", "B"],
            "FTHG": [2, 0, 1],
            "FTAG": [0, 1, 1],
            "FTR": ["H", "A", "D"],
            "HS": [12, 9, 11],
            "AS": [6, 10, 8],
            "HST": [5, 3, 4],
            "AST": [2, 5, 3],
            "HC": [7, 4, 6],
            "AC": [3, 8, 5],
            "HY": [2, 1, 3],
            "AY": [3, 2, 2],
            "HR": [0, 0, 1],
            "AR": [0, 1, 0],
        }
    )


def test_team_normalization_and_match_validation():
    assert normalize_team_name("  Man   United ") == "Man United"
    assert normalize_team_name("Man Utd", {"Man Utd": "Man United"}) == "Man United"
    with pytest.raises(ValueError):
        prepare_modeling_matches(match_frame().drop(columns="FTR"))


def test_features_are_strictly_prematch():
    matches = prepare_modeling_matches(match_frame())
    config = ModelingConfig(leagues=("I1",), seasons=("2425",), rolling_windows=(2,))
    features = build_prematch_features(matches, config)
    first = features.iloc[0]
    second = features.iloc[1]
    assert first["home_elo"] == 1500
    assert pd.isna(first["home_points_2"])
    assert second["home_matches_played"] == 1
    assert second["home_points_2"] == 0
    assert second["away_points_2"] == 3
    assert second["home_rest_days"] == 7
    assert second["home_shots_for_2"] == 6
    assert second["home_shots_against_2"] == 12
    assert pd.isna(second["home_venue_points_2"])
    third = features.iloc[2]
    assert third["home_venue_points_2"] == 3
    assert third["away_venue_points_2"] == 0
    assert third["home_points_ewm"] > 0
    assert "home_shots" not in features


def test_adding_future_match_does_not_change_past_features():
    base = prepare_modeling_matches(match_frame().iloc[:2])
    extended = prepare_modeling_matches(match_frame())
    config = ModelingConfig(leagues=("I1",), seasons=("2425",), rolling_windows=(2,))
    base_features = build_prematch_features(base, config)
    extended_features = build_prematch_features(extended, config).iloc[:2]
    pd.testing.assert_frame_equal(
        base_features.reset_index(drop=True),
        extended_features.reset_index(drop=True),
        check_dtype=False,
    )


def test_future_fixture_features_use_only_completed_history():
    history = prepare_modeling_matches(match_frame())
    fixtures = prepare_future_fixtures(
        pd.DataFrame(
            [
                {
                    "date": "2024-09-01",
                    "season": "2425",
                    "league": "I1",
                    "home_team": "A",
                    "away_team": "C",
                }
            ]
        )
    )
    config = ModelingConfig(leagues=("I1",), seasons=("2425",), rolling_windows=(2,))
    features = build_fixture_features(history, fixtures, config)
    assert len(features) == 1
    assert features.loc[0, "home_matches_played"] == 3
    assert features.loc[0, "away_matches_played"] == 0
    assert pd.isna(features.loc[0, "result"])

    later_history = pd.concat(
        [
            history,
            history.iloc[[0]].assign(
                match_id="later",
                date=pd.Timestamp("2024-10-01"),
                home_goals=9,
                away_goals=0,
                result="H",
            ),
        ],
        ignore_index=True,
    )
    later_features = build_fixture_features(later_history, fixtures, config)
    pd.testing.assert_series_equal(
        features.iloc[0],
        later_features.iloc[0],
        check_names=False,
    )


def test_future_fixture_validation_and_generated_identity():
    fixtures = prepare_future_fixtures(
        pd.DataFrame(
            [
                {
                    "date": "2025-08-20",
                    "season": "2526",
                    "league": "I1",
                    "home_team": "Inter",
                    "away_team": "Milan",
                }
            ]
        )
    )
    assert fixtures.loc[0, "match_id"].startswith("fixture|I1|2526|")
    assert pd.isna(fixtures.loc[0, "result"])
    with pytest.raises(ValueError, match="fixture mancanti"):
        prepare_future_fixtures(pd.DataFrame({"date": ["2025-08-20"]}))
    with pytest.raises(ValueError, match="target osservati"):
        prepare_future_fixtures(fixtures.assign(result="H"))


def test_simultaneous_matches_do_not_see_each_others_results():
    frame = pd.DataFrame(
        {
            "Date": ["01/08/2024", "01/08/2024"],
            "Season": ["2425", "2425"],
            "League": ["I1", "I1"],
            "HomeTeam": ["A", "C"],
            "AwayTeam": ["B", "A"],
            "FTHG": [2, 0],
            "FTAG": [0, 1],
            "FTR": ["H", "A"],
        }
    )
    features = build_prematch_features(
        prepare_modeling_matches(frame),
        ModelingConfig(leagues=("I1",), seasons=("2425",), rolling_windows=(2,)),
    )
    second = features.loc[features["home_team"].eq("C")].iloc[0]
    assert second["away_matches_played"] == 0
    assert pd.isna(second["away_points_2"])


def test_modeling_config_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        ModelingConfig(leagues=()).validate()
    with pytest.raises(ValueError):
        ModelingConfig(seasons=()).validate()
    with pytest.raises(ValueError):
        ModelingConfig(rolling_windows=(0,)).validate()
    with pytest.raises(ValueError):
        ModelingConfig(elo_k_factor=0).validate()
    with pytest.raises(ValueError):
        ModelingConfig(elo_season_regression=2).validate()


def test_canonical_database_is_modeling_source(tmp_path):
    database = ResearchDatabase(tmp_path / "canonical.sqlite3")
    raw = match_frame().rename(
        columns={
            "AvgCH": "B365CH",
            "AvgCD": "B365CD",
            "AvgCA": "B365CA",
        }
    )
    raw["B365CH"] = 2.0
    raw["B365CD"] = 3.2
    raw["B365CA"] = 4.0
    IngestionPipeline(database).run(FootballDataProvider(raw))
    canonical = load_canonical_matches(database, leagues=("I1",), seasons=("2425",))
    assert len(canonical) == 3
    assert canonical["match_id"].str.len().gt(0).all()
    assert canonical["market_home_probability"].notna().all()
