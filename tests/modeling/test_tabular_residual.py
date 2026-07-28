from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from football_odds.modeling.tabular_residual import (
    CORRECTION_BOUND,
    TABULAR_RESIDUAL_MODEL_NAME,
    TabularAblation,
    TabularPlayerData,
    build_tabular_features,
    fit_gated_tabular_residual,
    residual_targets,
    tabular_prediction_frame,
    walk_forward_gated_tabular_residual,
)


def _data(matches: int = 30) -> TabularPlayerData:
    rng = np.random.default_rng(9)
    starters = rng.normal(size=(matches, 2, 11, 4))
    bench = rng.normal(size=(matches, 2, 3, 4))
    departments = np.tile(np.arange(11) % 5, (matches, 2, 1))
    bench_departments = np.tile(np.arange(3), (matches, 2, 1))
    bench_mask = np.ones((matches, 2, 3))
    bench_mask[::2, :, -1] = 0
    return TabularPlayerData(
        starters,
        departments,
        bench,
        bench_departments,
        bench_mask,
        ("history", "form", "fs_available", "fs_quality"),
        ("history", "form"),
    )


def test_ablation_groups_are_independent_and_fixed():
    data = _data(4)
    base = build_tabular_features(data, TabularAblation.BASE)
    store = build_tabular_features(data, TabularAblation.FEATURE_STORE)
    bench = build_tabular_features(data, TabularAblation.BENCH)
    combined = build_tabular_features(data, TabularAblation.COMBINED)

    assert base.shape[1] == 18 * (2 * 2 + 1)
    assert store.shape[1] == 18 * (2 * 4 + 1)
    assert bench.shape[1] == base.shape[1] * 2
    assert combined.shape[1] == store.shape[1] * 2
    changed = TabularPlayerData(
        data.starters,
        data.starter_departments,
        data.bench + 100,
        data.bench_departments,
        data.bench_mask,
        data.feature_names,
        data.base_feature_names,
    )
    np.testing.assert_allclose(base, build_tabular_features(changed, "base"))
    assert not np.allclose(bench, build_tabular_features(changed, "bench"))


def test_targets_use_cross_fitted_base_rate_contract():
    targets = residual_targets([2, 0], [1, 3], [1.5, 0.8], [1.1, 1.7])
    np.testing.assert_allclose(
        targets,
        np.log(
            (np.asarray([[2, 1], [0, 3]]) + 0.5)
            / (np.asarray([[1.5, 1.1], [0.8, 1.7]]) + 0.5)
        ),
    )
    with pytest.raises(ValueError, match="rates positive"):
        residual_targets([1], [0], [0], [1])


def test_training_is_deterministic_and_gate_attenuates_clipped_corrections():
    data = _data()
    targets = np.column_stack(
        [
            data.starters[:, 0, :, 0].mean(axis=1),
            data.starters[:, 1, :, 1].mean(axis=1),
        ]
    )
    first = fit_gated_tabular_residual(data, targets, seed=7, max_iter=20)
    second = fit_gated_tabular_residual(data, targets, seed=7, max_iter=20)
    gate = np.linspace(0, 1, len(targets))
    pre, post = first.corrections(data, gate)
    other_pre, other_post = second.corrections(data, gate)

    np.testing.assert_allclose(pre, other_pre)
    np.testing.assert_allclose(post, other_post)
    np.testing.assert_allclose(post, pre * gate[:, None])
    assert np.abs(pre).max() <= CORRECTION_BOUND
    np.testing.assert_allclose(post[0], 0)
    np.testing.assert_allclose(post[-1], pre[-1])

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        first.corrections(data, np.full(len(targets), 2.0))


def test_prediction_artifact_records_gate_and_pre_post_corrections():
    matches = pd.DataFrame(
        {"match_id": ["a", "b"], "season": ["2024", "2024"], "result": ["H", "D"]}
    )
    pre = np.asarray([[0.2, -0.1], [0.1, 0.3]])
    post = np.asarray([[0.0, 0.0], [0.05, 0.15]])
    artifact = tabular_prediction_frame(
        matches,
        [1.4, 1.2],
        [1.0, 1.1],
        [-0.05, 0.05],
        pre,
        post,
        [0, 0.5],
        feature_ablation="combined",
    )

    assert set(
        [
            "home_correction_pre_gate",
            "away_correction_pre_gate",
            "home_correction_post_gate",
            "away_correction_post_gate",
            "reliability_gate",
        ]
    ).issubset(artifact)
    assert artifact["model"].eq(TABULAR_RESIDUAL_MODEL_NAME).all()
    assert artifact["feature_ablation"].eq("combined").all()
    np.testing.assert_allclose(
        artifact[["probability_home", "probability_draw", "probability_away"]].sum(
            axis=1
        ),
        1,
    )
    with pytest.raises(ValueError, match="align"):
        tabular_prediction_frame(
            matches,
            [1.4, 1.2],
            [1.0, 1.1],
            [-0.05],
            pre,
            post,
            [0, 0.5],
            feature_ablation="combined",
        )


def test_walk_forward_consumes_supplied_cross_fitted_rates():
    data = _data(18)
    seasons = np.repeat(["2021", "2022", "2023"], 6)
    matches = pd.DataFrame(
        {
            "match_id": [f"m{index}" for index in range(18)],
            "season": seasons,
            "league": "I1",
            "home_goals": np.arange(18) % 3,
            "away_goals": (np.arange(18) + 1) % 3,
        }
    )
    matches["result"] = np.where(
        matches["home_goals"] > matches["away_goals"],
        "H",
        np.where(matches["home_goals"] == matches["away_goals"], "D", "A"),
    )
    common = SimpleNamespace(
        matches=matches,
        players=data.starters,
        departments=data.starter_departments,
        bench_players=data.bench,
        bench_departments=data.bench_departments,
        bench_mask=data.bench_mask,
    )
    rates = matches[["match_id", "season"]].copy()
    rates["home_rate"] = 1.3
    rates["away_rate"] = 1.1
    rates["rho"] = np.linspace(-0.1, 0.1, len(rates))

    metrics, predictions = walk_forward_gated_tabular_residual(
        common,
        rates,
        np.linspace(0, 1, len(matches)),
        data.feature_names,
        data.base_feature_names,
        max_iter=5,
    )

    assert set(predictions["season"]) == {"2022", "2023"}
    assert predictions["feature_ablation"].eq("combined").all()
    assert not metrics.empty
