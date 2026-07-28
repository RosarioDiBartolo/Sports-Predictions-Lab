import pandas as pd

from football_odds.players.feature_store import build_temporal_player_matrix


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "kickoff": "2024-01-01T15:00:00",
                "league": "I1",
                "team_id": 1,
                "player_id": "p",
                "lineup_role": "starter",
                "minutes_played": 0.0,
                "minute_in": None,
                "minute_out": None,
                "position_original": "Centre-Back",
                "position_normalized": "D",
                "quality": "reported",
                "source": "provider-a",
            },
            {
                "match_id": "m2",
                "kickoff": "2024-01-08T15:00:00",
                "league": "I1",
                "team_id": 2,
                "player_id": "p",
                "lineup_role": "bench",
                "minutes_played": None,
                "minute_in": None,
                "minute_out": None,
                "position_original": None,
                "position_normalized": None,
                "quality": "reported",
                "source": "provider-b",
            },
        ]
    )


def test_current_match_is_not_visible_and_zero_is_not_missing():
    matrix = build_temporal_player_matrix(_rows())
    assert matrix.loc[0, "current_lineup_role"] == "starter"
    assert matrix.loc[1, "current_lineup_role"] == "bench"
    assert matrix.loc[0, "current_position_original"] == "Centre-Back"
    assert not matrix.loc[0, "observations_available"]
    assert matrix.loc[1, "mean_minutes_available"]
    assert matrix.loc[1, "mean_minutes_value"] == 0.0


def test_simultaneous_matches_update_only_after_all_snapshots():
    rows = pd.concat([_rows().iloc[[0]], _rows().iloc[[0]].assign(match_id="m1b")])
    matrix = build_temporal_player_matrix(rows)
    assert matrix["observations_available"].eq(False).all()


def test_equivalent_timezone_kickoffs_are_simultaneous():
    rows = pd.concat(
        [
            _rows().iloc[[0]].assign(kickoff="2024-01-01T12:00:00+00:00"),
            _rows()
            .iloc[[0]]
            .assign(match_id="m1b", kickoff="2024-01-01T13:00:00+01:00"),
        ],
        ignore_index=True,
    )
    matrix = build_temporal_player_matrix(rows)
    assert matrix["observations_available"].eq(False).all()


def test_team_change_is_an_observed_interval_not_a_contract():
    matrix = build_temporal_player_matrix(_rows())
    assert matrix.loc[1, "team_change_value"]
    assert matrix.loc[1, "team_change_fallback_kind"] == "not_contract"


def test_bench_without_timing_remains_unknown():
    rows = _rows()
    rows.loc[0, "lineup_role"] = "bench"
    rows.loc[0, "minutes_played"] = None
    matrix = build_temporal_player_matrix(rows)
    assert not matrix.loc[1, "mean_minutes_available"]
    assert matrix.loc[1, "sub_entry_rate_fallback_kind"] == "unknown_without_timing"
