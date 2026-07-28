"""Deterministic tabular residual challenger for Dixon-Coles rates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .dixon_coles import score_probabilities
from .evaluation import metrics_by_season

TABULAR_RESIDUAL_MODEL_NAME = "dixon_coles_gated_tabular_residual"
CORRECTION_BOUND = 0.35
DEPARTMENT_COUNT = 5
NumericVector = Sequence[float] | np.ndarray | pd.Series


class PlayerTensorContract(Protocol):
    """Structural subset of the common player tensor dataset."""

    matches: pd.DataFrame
    players: np.ndarray
    departments: np.ndarray
    bench_players: np.ndarray
    bench_departments: np.ndarray
    bench_mask: np.ndarray


class TabularAblation(str, Enum):
    """Controlled player-input groups available to the challenger."""

    BASE = "base"
    FEATURE_STORE = "feature_store"
    BENCH = "bench"
    COMBINED = "combined"

    @property
    def uses_feature_store(self) -> bool:
        return self in {self.FEATURE_STORE, self.COMBINED}

    @property
    def uses_bench(self) -> bool:
        return self in {self.BENCH, self.COMBINED}


@dataclass(frozen=True)
class TabularPlayerData:
    """Player tensors prepared by the common temporal feature pipeline."""

    starters: np.ndarray
    starter_departments: np.ndarray
    bench: np.ndarray
    bench_departments: np.ndarray
    bench_mask: np.ndarray
    feature_names: tuple[str, ...]
    base_feature_names: tuple[str, ...]

    def validate(self) -> None:
        matches = self.starters.shape[0]
        if self.starters.ndim != 4 or self.starters.shape[1] != 2:
            raise ValueError("starters must have shape (matches, 2, players, features)")
        if self.starter_departments.shape != self.starters.shape[:3]:
            raise ValueError("starter_departments shape does not match starters")
        if self.bench.ndim != 4 or self.bench.shape[0:2] != (matches, 2):
            raise ValueError("bench must have shape (matches, 2, players, features)")
        if self.bench.shape[-1] != self.starters.shape[-1]:
            raise ValueError("starter and bench feature counts differ")
        if self.bench_departments.shape != self.bench.shape[:3]:
            raise ValueError("bench_departments shape does not match bench")
        if self.bench_mask.shape != self.bench.shape[:3]:
            raise ValueError("bench_mask shape does not match bench")
        if len(self.feature_names) != self.starters.shape[-1]:
            raise ValueError("feature_names does not match player tensors")
        unknown = set(self.base_feature_names).difference(self.feature_names)
        if unknown:
            raise ValueError(f"Unknown base features: {sorted(unknown)}")


def _selected_indices(
    data: TabularPlayerData, ablation: TabularAblation
) -> np.ndarray:
    selected = (
        data.feature_names
        if ablation.uses_feature_store
        else data.base_feature_names
    )
    lookup = {name: index for index, name in enumerate(data.feature_names)}
    return np.asarray([lookup[name] for name in selected], dtype=int)


def _pool(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return fixed mean/std/availability aggregates for a player pool."""

    valid = mask.astype(bool)
    counts = valid.sum(axis=1, keepdims=True).astype(float)
    divisor = np.maximum(counts, 1.0)
    masked = np.where(valid[..., None], values, 0.0)
    mean = masked.sum(axis=1) / divisor
    centered = np.where(valid[..., None], values - mean[:, None, :], 0.0)
    std = np.sqrt((centered * centered).sum(axis=1) / divisor)
    return np.concatenate([mean, std, counts], axis=1)


def _team_aggregates(
    values: np.ndarray,
    departments: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    pools = [_pool(values, mask)]
    for department in range(DEPARTMENT_COUNT):
        pools.append(_pool(values, mask & (departments == department)))
    return np.concatenate(pools, axis=1)


def build_tabular_features(
    data: TabularPlayerData,
    ablation: TabularAblation | str,
) -> np.ndarray:
    """Build fixed home, away and difference aggregates for an ablation."""

    data.validate()
    variant = TabularAblation(ablation)
    indices = _selected_indices(data, variant)
    starter_mask = np.ones(data.starters.shape[:3], dtype=bool)
    team_parts = []
    for side in range(2):
        parts = [
            _team_aggregates(
                np.take(data.starters[:, side], indices, axis=-1),
                data.starter_departments[:, side],
                starter_mask[:, side],
            )
        ]
        if variant.uses_bench:
            parts.append(
                _team_aggregates(
                    np.take(data.bench[:, side], indices, axis=-1),
                    data.bench_departments[:, side],
                    data.bench_mask[:, side].astype(bool),
                )
            )
        team_parts.append(np.concatenate(parts, axis=1))
    return np.concatenate(
        [team_parts[0], team_parts[1], team_parts[0] - team_parts[1]], axis=1
    )


def residual_targets(
    home_goals: NumericVector,
    away_goals: NumericVector,
    home_base_rates: NumericVector,
    away_base_rates: NumericVector,
) -> np.ndarray:
    """Create the two residual targets from cross-fitted Dixon-Coles rates."""

    goals = np.column_stack([home_goals, away_goals]).astype(float)
    rates = np.column_stack([home_base_rates, away_base_rates]).astype(float)
    if goals.shape != rates.shape or goals.ndim != 2:
        raise ValueError("goals and base rates must contain the same matches")
    if not np.isfinite(goals).all() or not np.isfinite(rates).all():
        raise ValueError("goals and base rates must be finite")
    if (goals < 0).any() or (rates <= 0).any():
        raise ValueError("goals must be non-negative and rates positive")
    return np.log((goals + 0.5) / (rates + 0.5))


@dataclass
class GatedTabularResidual:
    """Two deterministic residual regressors sharing one feature contract."""

    home: HistGradientBoostingRegressor
    away: HistGradientBoostingRegressor
    ablation: TabularAblation

    def corrections(
        self, data: TabularPlayerData, reliability: NumericVector
    ) -> tuple[np.ndarray, np.ndarray]:
        features = build_tabular_features(data, self.ablation)
        gate = np.asarray(reliability, dtype=float)
        if gate.shape != (len(features),):
            raise ValueError("reliability must contain one score per match")
        if not np.isfinite(gate).all() or ((gate < 0) | (gate > 1)).any():
            raise ValueError("reliability scores must be finite and within [0, 1]")
        raw = np.column_stack(
            [self.home.predict(features), self.away.predict(features)]
        )
        clipped = np.clip(raw, -CORRECTION_BOUND, CORRECTION_BOUND)
        return clipped, clipped * gate[:, None]


def fit_gated_tabular_residual(
    data: TabularPlayerData,
    targets: np.ndarray,
    ablation: TabularAblation | str = TabularAblation.COMBINED,
    *,
    seed: int = 42,
    max_iter: int = 100,
) -> GatedTabularResidual:
    """Fit the non-operational challenger on precomputed residual targets."""

    features = build_tabular_features(data, ablation)
    target_array = np.asarray(targets, dtype=float)
    if target_array.shape != (len(features), 2):
        raise ValueError("targets must have shape (matches, 2)")
    if not np.isfinite(target_array).all():
        raise ValueError("targets must be finite")

    def estimator() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=max_iter,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )

    home, away = estimator(), estimator()
    home.fit(features, target_array[:, 0])
    away.fit(features, target_array[:, 1])
    return GatedTabularResidual(home, away, TabularAblation(ablation))


def tabular_prediction_frame(
    matches: pd.DataFrame,
    home_base_rates: NumericVector,
    away_base_rates: NumericVector,
    rho: float | NumericVector,
    corrections_pre_gate: np.ndarray,
    corrections_post_gate: np.ndarray,
    reliability: NumericVector,
    *,
    feature_ablation: TabularAblation | str,
) -> pd.DataFrame:
    """Build the candidate prediction artifact without persisting or promoting it."""

    home_rates = np.asarray(home_base_rates, dtype=float)
    away_rates = np.asarray(away_base_rates, dtype=float)
    pre = np.asarray(corrections_pre_gate, dtype=float)
    post = np.asarray(corrections_post_gate, dtype=float)
    gate = np.asarray(reliability, dtype=float)
    size = len(matches)
    rho_values = np.asarray(rho, dtype=float)
    if rho_values.ndim == 0:
        rho_values = np.full(size, float(rho_values))
    if (
        home_rates.shape != (size,)
        or away_rates.shape != (size,)
        or pre.shape != (size, 2)
        or post.shape != (size, 2)
        or gate.shape != (size,)
        or rho_values.shape != (size,)
    ):
        raise ValueError("prediction inputs must align with matches")
    if not np.isfinite(rho_values).all():
        raise ValueError("rho must be finite")
    corrected_home = home_rates * np.exp(post[:, 0])
    corrected_away = away_rates * np.exp(post[:, 1])
    probabilities = np.vstack(
        [
            score_probabilities(home, away, match_rho)
            for home, away, match_rho in zip(
                corrected_home, corrected_away, rho_values, strict=True
            )
        ]
    )
    artifact = matches.reset_index(drop=True).copy()
    artifact["model"] = TABULAR_RESIDUAL_MODEL_NAME
    artifact["feature_ablation"] = TabularAblation(feature_ablation).value
    artifact[["probability_home", "probability_draw", "probability_away"]] = (
        probabilities
    )
    artifact["home_base_rate"] = home_rates
    artifact["away_base_rate"] = away_rates
    artifact["home_correction_pre_gate"] = pre[:, 0]
    artifact["away_correction_pre_gate"] = pre[:, 1]
    artifact["home_correction_post_gate"] = post[:, 0]
    artifact["away_correction_post_gate"] = post[:, 1]
    artifact["reliability_gate"] = gate
    return artifact


def _slice_player_data(
    tensor_data: PlayerTensorContract,
    indices: np.ndarray,
    feature_names: tuple[str, ...],
    base_feature_names: tuple[str, ...],
) -> TabularPlayerData:
    return TabularPlayerData(
        starters=tensor_data.players[indices],
        starter_departments=tensor_data.departments[indices],
        bench=tensor_data.bench_players[indices],
        bench_departments=tensor_data.bench_departments[indices],
        bench_mask=tensor_data.bench_mask[indices],
        feature_names=feature_names,
        base_feature_names=base_feature_names,
    )


def walk_forward_gated_tabular_residual(
    tensor_data: PlayerTensorContract,
    rates: pd.DataFrame,
    reliability: NumericVector,
    feature_names: tuple[str, ...],
    base_feature_names: tuple[str, ...],
    *,
    ablation: TabularAblation | str = TabularAblation.COMBINED,
    seed: int = 42,
    max_iter: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate temporal folds using supplied cross-fitted Dixon-Coles rates."""

    variant = TabularAblation(ablation)
    matches = tensor_data.matches.copy()
    matches["season"] = matches["season"].astype(str).str.zfill(4)
    supplied_rates = rates.copy()
    supplied_rates["season"] = supplied_rates["season"].astype(str).str.zfill(4)
    required_rates = {"match_id", "season", "home_rate", "away_rate", "rho"}
    missing = required_rates.difference(supplied_rates.columns)
    if missing:
        raise ValueError(f"Missing cross-fitted rate columns: {sorted(missing)}")
    if supplied_rates.duplicated(["match_id", "season"]).any():
        raise ValueError("Cross-fitted rates contain duplicate matches")
    matches = matches.merge(
        supplied_rates[list(required_rates)],
        on=["match_id", "season"],
        how="left",
        validate="one_to_one",
    )
    gate = np.asarray(reliability, dtype=float)
    if gate.shape != (len(matches),):
        raise ValueError("reliability must align with the common tensor dataset")

    predictions = []
    for season in sorted(matches["season"].unique())[1:]:
        train_mask = matches["season"].lt(season) & matches["home_rate"].notna()
        test_mask = matches["season"].eq(season) & matches["home_rate"].notna()
        if not train_mask.any() or not test_mask.any():
            continue
        train_indices = np.flatnonzero(train_mask.to_numpy())
        test_indices = np.flatnonzero(test_mask.to_numpy())
        training = matches.iloc[train_indices]
        targets = residual_targets(
            training["home_goals"],
            training["away_goals"],
            training["home_rate"],
            training["away_rate"],
        )
        predictor = fit_gated_tabular_residual(
            _slice_player_data(
                tensor_data, train_indices, feature_names, base_feature_names
            ),
            targets,
            variant,
            seed=seed,
            max_iter=max_iter,
        )
        test_data = _slice_player_data(
            tensor_data, test_indices, feature_names, base_feature_names
        )
        before, after = predictor.corrections(test_data, gate[test_indices])
        test = matches.iloc[test_indices]
        predictions.append(
            tabular_prediction_frame(
                test,
                test["home_rate"],
                test["away_rate"],
                test["rho"],
                before,
                after,
                gate[test_indices],
                feature_ablation=variant,
            )
        )
    artifact = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(
            columns=[
                "match_id",
                "season",
                "league",
                "result",
                "model",
                "feature_ablation",
                "probability_home",
                "probability_draw",
                "probability_away",
            ]
        )
    )
    return metrics_by_season(artifact), artifact
