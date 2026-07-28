import json

import numpy as np
import pandas as pd
import torch

from football_odds.modeling.neural import (
    DEPARTMENTS,
    NEURAL_FEATURE_NAMES,
    SharedPlayerEncoder,
    build_player_tensor_dataset,
    fit_neural_lineup_encoder,
    walk_forward_neural_lineup_model,
)


def _starters(prefix: str) -> str:
    positions = ("G", "D", "D", "D", "D", "M", "M", "M", "M", "F", "F")
    return json.dumps(
        [
            {"player_id": f"{prefix}-{index}", "position": position}
            for index, position in enumerate(positions)
        ]
    )


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2023-01-01T12:00:00Z",
                "season": "2223",
                "league": "I1",
                "home_team": "Home",
                "away_team": "Away",
                "home_goals": 2,
                "away_goals": 0,
                "result": "H",
            },
            {
                "match_id": "m2",
                "date": "2023-01-08T12:00:00Z",
                "season": "2223",
                "league": "I1",
                "home_team": "Home",
                "away_team": "Away",
                "home_goals": 0,
                "away_goals": 1,
                "result": "A",
            },
        ]
    )


def _lineups() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": match_id,
                "home_starters": _starters("home"),
                "away_starters": _starters("away"),
                "home_bench": "[]",
                "away_bench": "[]",
            }
            for match_id in ("m1", "m2")
        ]
    )


def test_real_minutes_and_bench_history_are_temporal_and_separate():
    promoted = json.loads(_starters("home"))
    promoted[-1] = {
        "player_id": "bench-player",
        "position": "F",
        "formation_grid": "reported:centre-forward",
    }
    first_bench = json.dumps(
        [
            {
                "player_id": "bench-player",
                "position": "F",
                "formation_grid": "reported:centre-forward",
                "timing_observed": 1,
                "minutes_played": 30,
                "minute_in": 60,
                "minute_out": None,
            }
        ]
    )
    lineups = _lineups()
    lineups.loc[0, "home_bench"] = first_bench
    lineups.loc[1, "home_starters"] = json.dumps(promoted)

    tensor = build_player_tensor_dataset(_matches(), lineups)
    feature = {name: index for index, name in enumerate(NEURAL_FEATURE_NAMES)}

    assert tensor.bench_mask[0, 0].sum() == 1
    assert tensor.bench_players[0, 0, 0, feature["observed"]] == 0
    player = tensor.players[1, 0, -1]
    assert player[feature["start_rate"]] == 0
    assert player[feature["bench_rate"]] == 1
    assert player[feature["bench_entry_rate"]] == 1
    np.testing.assert_allclose(player[feature["average_minutes"]], 1 / 3)
    assert player[feature["minutes_real_coverage"]] == 1
    assert player[feature["original_position_quality"]] == 1


def test_canonical_temporal_store_is_joined_with_quality_and_position():
    temporal = pd.DataFrame(
        [
            {
                "match_id": "m2",
                "player_id": "home-0",
                "team_id": "home-team",
                "current_lineup_role": "starter",
                "current_position_original": "Goalkeeper",
                "current_position_normalized": "G",
                "mean_minutes_value": 54.0,
                "mean_minutes_available": True,
                "mean_minutes_quality": "reported",
                "mean_minutes_fallback_kind": "none",
                "position_original_value": "Goalkeeper",
                "position_original_available": True,
                "position_original_quality": "reported",
                "position_original_fallback_kind": "none",
            }
        ]
    )
    tensor = build_player_tensor_dataset(
        _matches(),
        _lineups(),
        temporal_features=temporal,
    )
    feature = {name: index for index, name in enumerate(NEURAL_FEATURE_NAMES)}
    player = tensor.players[1, 0, 0]
    np.testing.assert_allclose(
        player[feature["fs_mean_minutes_value"]],
        0.6,
    )
    assert player[feature["fs_mean_minutes_available"]] == 1
    assert player[feature["fs_mean_minutes_quality"]] == 1
    assert player[feature["fs_position_goalkeeper"]] == 1
    assert player[feature["fs_store_available"]] == 1


def test_canonical_temporal_store_rejects_duplicate_player_match():
    temporal = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "player_id": "home-0",
                "team_id": "home-team",
                "current_lineup_role": "starter",
                "current_position_original": "Goalkeeper",
                "current_position_normalized": "G",
            },
            {
                "match_id": "m1",
                "player_id": "home-0",
                "team_id": "home-team",
                "current_lineup_role": "starter",
                "current_position_original": "Goalkeeper",
                "current_position_normalized": "G",
            },
        ]
    )
    with np.testing.assert_raises_regex(ValueError, "duplicati"):
        build_player_tensor_dataset(
            _matches(),
            _lineups(),
            temporal_features=temporal,
        )


def test_canonical_temporal_store_supplies_confirmed_bench_separately():
    temporal = pd.DataFrame(
        [
            {
                "match_id": "m2",
                "player_id": "home-0",
                "team_id": "home-team",
                "current_lineup_role": "starter",
                "current_position_original": "Goalkeeper",
                "current_position_normalized": "G",
            },
            {
                "match_id": "m2",
                "player_id": "reserve-1",
                "team_id": "home-team",
                "current_lineup_role": "bench",
                "current_position_original": "Centre-Forward",
                "current_position_normalized": "F",
            },
        ]
    )
    tensor = build_player_tensor_dataset(
        _matches(),
        _lineups(),
        temporal_features=temporal,
    )
    feature = {name: index for index, name in enumerate(NEURAL_FEATURE_NAMES)}
    assert tensor.bench_mask[1, 0].sum() == 1
    assert tensor.bench_players[1, 0, 0, feature["fs_store_available"]] == 1


def test_player_tensor_is_leakage_safe_and_keeps_individuals():
    tensor = build_player_tensor_dataset(_matches(), _lineups())
    assert tensor.players.shape == (2, 2, 11, len(NEURAL_FEATURE_NAMES))
    assert tensor.departments.shape == (2, 2, 11)
    assert tensor.bench_players.shape[:3] == (2, 2, 12)
    assert tensor.bench_mask.sum() == 0
    assert tensor.players[0, :, :, 4].sum() == 0
    assert tensor.players[1, :, :, 4].sum() == 22
    assert tensor.players[1, 0, :, 0].mean() == 1.0
    assert tensor.players[1, 1, :, 0].mean() == 0.0
    feature = {name: index for index, name in enumerate(NEURAL_FEATURE_NAMES)}
    assert tensor.players[1, 0, :, feature["career_goals_for"]].mean() == 2.0
    assert tensor.players[1, 0, :, feature["career_goals_against"]].mean() == 0.0
    assert tensor.players[1, 0, :, feature["team_experience_share"]].mean() == 1.0
    assert tensor.players[1, 0, :, feature["league_experience_share"]].mean() == 1.0
    assert tensor.players[1, 0, :, feature["role_stability"]].mean() == 1.0
    np.testing.assert_allclose(
        tensor.players[1, 0, :, feature["lineup_familiarity"]],
        np.log(2.0),
    )

    changed = _matches()
    changed.loc[1, ["home_goals", "away_goals", "result"]] = [7, 0, "H"]
    changed_tensor = build_player_tensor_dataset(changed, _lineups())
    np.testing.assert_allclose(tensor.players[1], changed_tensor.players[1])


def test_shared_encoder_pools_departments_and_bounds_corrections():
    model = SharedPlayerEncoder(len(NEURAL_FEATURE_NAMES), embedding_dim=32)
    assert sum(isinstance(module, torch.nn.Linear) for module in model.encoder) == 2
    players = torch.zeros((3, 2, 11, len(NEURAL_FEATURE_NAMES)))
    bench_players = torch.zeros((3, 2, 12, len(NEURAL_FEATURE_NAMES)))
    bench_departments = torch.full((3, 2, 12), -1)
    bench_mask = torch.zeros((3, 2, 12))
    departments = torch.tensor(
        np.tile(
            np.asarray(
                [
                    0,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    0,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                ]
            ).reshape(2, 11),
            (3, 1, 1),
        )
    )
    output = model(
        players,
        departments,
        bench_players,
        bench_departments,
        bench_mask,
    )
    assert output.shape == (3, 2)
    assert torch.max(torch.abs(output)) <= 0.35
    assert len(DEPARTMENTS) == 4


def test_neural_training_is_deterministic_and_has_no_identity_input():
    tensor = build_player_tensor_dataset(_matches(), _lineups())
    players = np.concatenate([tensor.players] * 4)
    departments = np.concatenate([tensor.departments] * 4)
    targets = np.zeros((len(players), 2), dtype=np.float32)
    first = fit_neural_lineup_encoder(players, departments, targets, epochs=2, seed=7)
    second = fit_neural_lineup_encoder(players, departments, targets, epochs=2, seed=7)
    np.testing.assert_allclose(
        first.corrections(
            players,
            departments,
            tensor.bench_players.repeat(4, axis=0),
            tensor.bench_departments.repeat(4, axis=0),
            tensor.bench_mask.repeat(4, axis=0),
        ),
        second.corrections(
            players,
            departments,
            tensor.bench_players.repeat(4, axis=0),
            tensor.bench_departments.repeat(4, axis=0),
            tensor.bench_mask.repeat(4, axis=0),
        ),
        atol=1e-7,
    )
    assert "player_id" not in NEURAL_FEATURE_NAMES


def test_neural_walk_forward_uses_cross_fitted_base_rates():
    rows = []
    lineup_rows = []
    for season_index, season in enumerate(("2021", "2122", "2223")):
        for index in range(9):
            home_goals = index % 3
            away_goals = (index * 2 + season_index) % 3
            match_id = f"{season}-{index}"
            rows.append(
                {
                    "match_id": match_id,
                    "date": f"20{20 + season_index}-01-{index + 1:02d}",
                    "season": season,
                    "league": "I1",
                    "home_team": f"H{index % 3}",
                    "away_team": f"A{index % 3}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": (
                        "H"
                        if home_goals > away_goals
                        else "A"
                        if home_goals < away_goals
                        else "D"
                    ),
                    "elo_difference": float(index - 4),
                    "home_points_5": float(index % 3),
                    "away_points_5": float((index + 1) % 3),
                }
            )
            lineup_rows.append(
                {
                    "match_id": match_id,
                    "home_starters": _starters(f"h-{index % 3}"),
                    "away_starters": _starters(f"a-{index % 3}"),
                    "home_bench": "[]",
                    "away_bench": "[]",
                }
            )
    base = pd.DataFrame(rows)
    tensor = build_player_tensor_dataset(base, pd.DataFrame(lineup_rows))
    metrics, predictions = walk_forward_neural_lineup_model(
        tensor,
        base,
        embedding_dim=8,
        epochs=1,
    )
    assert set(metrics["season"]) == {"2223"}
    assert predictions["match_id"].str.startswith("2021").sum() == 0
    assert predictions["match_id"].str.startswith("2122").sum() == 0
    np.testing.assert_allclose(
        predictions.filter(like="probability_").sum(axis=1),
        1.0,
    )
