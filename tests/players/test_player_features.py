import numpy as np
import pandas as pd

from football_odds.players.prematch_features import (
    PLAYER_FEATURE_NAMES,
    build_prematch_player_features,
)


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-08-01",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 0,
                "result": "H",
            },
            {
                "match_id": "m2",
                "date": "2024-08-08",
                "home_team": "B",
                "away_team": "A",
                "home_goals": 1,
                "away_goals": 1,
                "result": "D",
            },
            {
                "match_id": "m3",
                "date": "2024-08-15",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 0,
                "away_goals": 1,
                "result": "A",
            },
        ]
    )


def _lineups(include_future: bool = True) -> pd.DataFrame:
    rows = []
    for match_id, home, away in (("m1", "A", "B"), ("m2", "B", "A")):
        for team, offset in ((home, 0), (away, 100)):
            for number in range(1, 12):
                rows.append(
                    {
                        "match_id": match_id,
                        "team": team,
                        "player_id": f"{team}-{offset + number}",
                        "lineup_role": "starter",
                    }
                )
            rows.append(
                {
                    "match_id": match_id,
                    "team": team,
                    "player_id": f"{team}-bench",
                    "lineup_role": "substitute",
                }
            )
    if include_future:
        for number in range(1, 12):
            rows.append(
                {
                    "match_id": "m3",
                    "team": "A",
                    "player_id": f"future-{number}",
                    "lineup_role": "starter",
                }
            )
    return pd.DataFrame(rows)


def test_player_features_are_prematch_and_have_fixed_contract():
    features = build_prematch_player_features(_matches(), _lineups())

    assert len(features) == 3
    assert set(features.columns) == {
        "match_id",
        *(
            f"{side}_{name}"
            for side in ("home", "away")
            for name in PLAYER_FEATURE_NAMES
        ),
    }
    assert features.loc[0].drop(labels="match_id").isna().all()
    assert features.loc[1, "away_player_expected_strength"] == 1.0
    assert 0.0 <= features.loc[2, "home_player_lineup_continuity"] <= 1.0


def test_current_and_future_lineups_cannot_change_earlier_snapshots():
    with_future = build_prematch_player_features(_matches(), _lineups(True))
    without_future = build_prematch_player_features(_matches(), _lineups(False))

    pd.testing.assert_frame_equal(with_future, without_future)


def test_simultaneous_matches_are_snapshotted_before_updates():
    matches = _matches()
    matches.loc[1, "date"] = matches.loc[0, "date"]
    features = build_prematch_player_features(matches, _lineups(False))

    assert np.isnan(features.loc[0, "home_player_expected_strength"])
    assert np.isnan(features.loc[1, "home_player_expected_strength"])


def test_equivalent_timezone_dates_are_snapshotted_together():
    matches = _matches().iloc[:2].copy()
    matches.loc[0, "date"] = "2024-08-01T12:00:00+00:00"
    matches.loc[1, "date"] = "2024-08-01T13:00:00+01:00"
    features = build_prematch_player_features(matches, _lineups(False))

    assert features.drop(columns="match_id").isna().all(axis=None)


def test_player_feature_contract_rejects_invalid_lookback():
    try:
        build_prematch_player_features(_matches(), _lineups(), lookback=0)
    except ValueError as error:
        assert "lookback" in str(error)
    else:
        raise AssertionError("lookback=0 deve essere rifiutato")
