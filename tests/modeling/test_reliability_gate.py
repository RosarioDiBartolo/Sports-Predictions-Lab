import numpy as np
import pandas as pd

from football_odds.modeling.gated_neural import (
    NeuralFeatureAblation,
    ablation_inputs,
    gated_prediction_frame,
    walk_forward_gated_neural_lineup_model,
)
from football_odds.modeling.neural import (
    BASE_FEATURE_COUNT,
    NEURAL_FEATURE_NAMES,
    PlayerTensorDataset,
)
from football_odds.modeling.reliability import (
    attenuate_corrections,
    reliability_scores,
)


def _starters() -> np.ndarray:
    values = np.zeros((2, 2, 11, len(NEURAL_FEATURE_NAMES)), dtype=np.float32)
    index = {name: position for position, name in enumerate(NEURAL_FEATURE_NAMES)}
    values[..., index["log_squad_observations"]] = np.log1p(5)
    values[..., index["fs_store_available"]] = 1
    values[..., index["fs_observations_value"]] = np.log1p(5)
    values[..., index["fs_observations_quality"]] = 0.7
    values[..., index["fs_mean_minutes_available"]] = 1
    values[..., index["fs_mean_minute_out_available"]] = 1
    return values


def test_reliability_is_conservative_and_deterministic():
    starters = _starters()
    starters[1, 1, :5, -1] = 0
    index = {name: position for position, name in enumerate(NEURAL_FEATURE_NAMES)}
    starters[1, 1, :5, index["fs_mean_minutes_available"]] = 0

    first = reliability_scores(starters, NEURAL_FEATURE_NAMES)
    second = reliability_scores(starters, NEURAL_FEATURE_NAMES)

    np.testing.assert_array_equal(first.match, second.match)
    assert first.match[0] == 1
    np.testing.assert_allclose(first.match[1], 6 / 11)
    np.testing.assert_allclose(first.team[1, 1], 6 / 11)
    assert first.team[1, 0] == 1


def test_gate_zero_falls_back_and_one_preserves_bounded_correction():
    corrections = np.array([[0.35, -0.35], [0.1, -0.2]])
    actual = attenuate_corrections(corrections, np.array([0.0, 1.0]))
    np.testing.assert_allclose(actual[0], 0)
    np.testing.assert_allclose(actual[1], corrections[1])


def test_gated_prediction_artifact_exposes_components_and_pre_post_corrections():
    starters = _starters()[:1]
    gate = reliability_scores(starters, NEURAL_FEATURE_NAMES)
    before = np.array([[0.2, -0.1]])
    after = attenuate_corrections(before, gate.match)
    matches = pd.DataFrame(
        [{"match_id": "m1", "season": "2425", "league": "I1", "result": "H"}]
    )

    artifact = gated_prediction_frame(
        matches,
        np.array([[0.5, 0.3, 0.2]]),
        "candidate",
        NeuralFeatureAblation.COMBINED,
        gate,
        before,
        after,
    )

    assert artifact.loc[0, "feature_ablation"] == "combined"
    assert artifact.loc[0, "match_reliability"] == 1
    assert artifact.loc[0, "home_history_depth"] == 1
    assert artifact.loc[0, "away_timing_coverage"] == 1
    assert artifact.loc[0, "home_correction_before_gate"] == 0.2
    assert artifact.loc[0, "away_correction_after_gate"] == -0.1


def test_ablation_masks_only_the_selected_feature_groups():
    players = _starters()
    bench = np.ones((2, 2, 12, len(NEURAL_FEATURE_NAMES)), dtype=np.float32)
    mask = np.ones((2, 2, 12), dtype=np.float32)
    tensor = PlayerTensorDataset(
        matches=None,  # type: ignore[arg-type]
        players=players,
        departments=np.zeros((2, 2, 11), dtype=np.int64),
        bench_players=bench,
        bench_departments=np.zeros((2, 2, 12), dtype=np.int64),
        bench_mask=mask,
    )

    base_players, _, base_mask = ablation_inputs(
        tensor,
        NeuralFeatureAblation.BASE,
    )
    assert not base_players[..., :BASE_FEATURE_COUNT].sum() == 0
    assert base_players[..., BASE_FEATURE_COUNT:].sum() == 0
    assert base_mask.sum() == 0

    store_players, _, store_mask = ablation_inputs(tensor, "feature_store")
    np.testing.assert_array_equal(store_players, players)
    assert store_mask.sum() == 0

    bench_players, _, bench_mask = ablation_inputs(tensor, "bench")
    assert bench_players[..., BASE_FEATURE_COUNT:].sum() == 0
    assert bench_mask.sum() > 0

    combined_players, _, combined_mask = ablation_inputs(tensor, "combined")
    np.testing.assert_array_equal(combined_players, players)
    np.testing.assert_array_equal(combined_mask, mask)


def test_gated_walk_forward_returns_the_common_empty_contract_without_folds():
    players = _starters()[:1]
    tensor = PlayerTensorDataset(
        matches=pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "season": "2425",
                    "league": "I1",
                    "result": "H",
                    "home_goals": 1,
                    "away_goals": 0,
                }
            ]
        ),
        players=players,
        departments=np.zeros((1, 2, 11), dtype=np.int64),
        bench_players=np.zeros(
            (1, 2, 12, len(NEURAL_FEATURE_NAMES)), dtype=np.float32
        ),
        bench_departments=np.zeros((1, 2, 12), dtype=np.int64),
        bench_mask=np.zeros((1, 2, 12), dtype=np.float32),
    )
    rates = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "season": "2425",
                "home_rate": 1.2,
                "away_rate": 0.9,
                "rho": -0.05,
            }
        ]
    )

    metrics, predictions = walk_forward_gated_neural_lineup_model(
        tensor,
        pd.DataFrame(),
        rates=rates,
    )

    assert metrics.empty
    assert list(predictions.columns[-3:]) == [
        "probability_home",
        "probability_draw",
        "probability_away",
    ]
