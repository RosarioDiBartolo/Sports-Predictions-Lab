"""Controlled ablations and reliability gating for the neural candidate."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from .dixon_coles import score_probabilities
from .evaluation import PROBABILITY_COLUMNS, metrics_by_season
from .neural import (
    BASE_FEATURE_COUNT,
    NEURAL_LINEUP_MODEL_NAME,
    PlayerTensorDataset,
    cross_fitted_dixon_coles_rates,
    fit_neural_lineup_encoder,
)
from .reliability import ReliabilityScores, attenuate_corrections, reliability_scores

GATED_NEURAL_LINEUP_MODEL_NAME = "dixon_coles_shared_encoder_pooling_gated"


class NeuralFeatureAblation(str, Enum):
    """Controlled player-input groups used by the common comparison."""

    BASE = "base"
    FEATURE_STORE = "feature_store"
    BENCH = "bench"
    COMBINED = "combined"

    @property
    def includes_feature_store(self) -> bool:
        return self in {self.FEATURE_STORE, self.COMBINED}

    @property
    def includes_bench(self) -> bool:
        return self in {self.BENCH, self.COMBINED}


def ablation_inputs(
    tensor_data: PlayerTensorDataset,
    ablation: NeuralFeatureAblation | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return copied player inputs with excluded groups zeroed."""
    selected = NeuralFeatureAblation(ablation)
    players = tensor_data.players.copy()
    bench_players = tensor_data.bench_players.copy()
    bench_mask = tensor_data.bench_mask.copy()
    if not selected.includes_feature_store:
        players[..., BASE_FEATURE_COUNT:] = 0.0
        bench_players[..., BASE_FEATURE_COUNT:] = 0.0
    if not selected.includes_bench:
        bench_players.fill(0.0)
        bench_mask.fill(0.0)
    return players, bench_players, bench_mask


def gated_prediction_frame(
    matches: pd.DataFrame,
    probabilities: np.ndarray,
    model: str,
    ablation: NeuralFeatureAblation,
    gate: ReliabilityScores,
    corrections_before_gate: np.ndarray,
    corrections_after_gate: np.ndarray,
) -> pd.DataFrame:
    """Build the auditable prediction contract for a gated fold."""
    frame = matches[["match_id", "season", "league", "result"]].copy()
    frame["model"] = model
    frame["feature_ablation"] = ablation.value
    frame[list(PROBABILITY_COLUMNS)] = probabilities
    for side_index, side in enumerate(("home", "away")):
        frame[f"{side}_history_depth"] = gate.history_depth[:, side_index]
        frame[f"{side}_timing_coverage"] = gate.timing_coverage[:, side_index]
        frame[f"{side}_reliable_starters"] = gate.reliable_starters[:, side_index]
        frame[f"{side}_team_reliability"] = gate.team[:, side_index]
        frame[f"{side}_correction_before_gate"] = corrections_before_gate[
            :, side_index
        ]
        frame[f"{side}_correction_after_gate"] = corrections_after_gate[:, side_index]
    frame["match_reliability"] = gate.match
    return frame


def _fold_scores(scores: ReliabilityScores, indices: np.ndarray) -> ReliabilityScores:
    return ReliabilityScores(
        history_depth=scores.history_depth[indices],
        timing_coverage=scores.timing_coverage[indices],
        reliable_starters=scores.reliable_starters[indices],
        team=scores.team[indices],
        match=scores.match[indices],
    )


def walk_forward_gated_neural_lineup_model(
    tensor_data: PlayerTensorDataset,
    base_features: pd.DataFrame,
    *,
    embedding_dim: int = 32,
    epochs: int = 80,
    rates: pd.DataFrame | None = None,
    ablation: NeuralFeatureAblation | str = NeuralFeatureAblation.COMBINED,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate an ablation of the non-operational gated neural candidate."""
    selected = NeuralFeatureAblation(ablation)
    players, bench_players, bench_mask = ablation_inputs(tensor_data, selected)
    scores = reliability_scores(tensor_data.players, tensor_data.feature_names)
    matches = tensor_data.matches.copy()
    matches["season"] = matches["season"].astype(str).str.zfill(4)
    if rates is None:
        rates = cross_fitted_dixon_coles_rates(base_features, matches)
    matches = matches.merge(rates, on=["match_id", "season"], how="left")
    predictions = []
    for season in sorted(matches["season"].unique())[1:]:
        training_mask = matches["season"].lt(season) & matches["home_rate"].notna()
        test_mask = matches["season"].eq(season) & matches["home_rate"].notna()
        if not training_mask.any() or not test_mask.any():
            continue
        training_indices = np.flatnonzero(training_mask.to_numpy())
        test_indices = np.flatnonzero(test_mask.to_numpy())
        training = matches.iloc[training_indices]
        targets = np.column_stack(
            [
                np.log(
                    (training["home_goals"].to_numpy(float) + 0.5)
                    / (training["home_rate"].to_numpy(float) + 0.5)
                ),
                np.log(
                    (training["away_goals"].to_numpy(float) + 0.5)
                    / (training["away_rate"].to_numpy(float) + 0.5)
                ),
            ]
        ).astype(np.float32)
        predictor = fit_neural_lineup_encoder(
            players[training_indices],
            tensor_data.departments[training_indices],
            targets,
            bench_players=bench_players[training_indices],
            bench_departments=tensor_data.bench_departments[training_indices],
            bench_mask=bench_mask[training_indices],
            embedding_dim=embedding_dim,
            epochs=epochs,
            device=device,
        )
        before = predictor.corrections(
            players[test_indices],
            tensor_data.departments[test_indices],
            bench_players[test_indices],
            tensor_data.bench_departments[test_indices],
            bench_mask[test_indices],
        )
        fold_gate = _fold_scores(scores, test_indices)
        after = attenuate_corrections(before, fold_gate.match)
        test = matches.iloc[test_indices]
        home_rates = test["home_rate"].to_numpy(float) * np.exp(after[:, 0])
        away_rates = test["away_rate"].to_numpy(float) * np.exp(after[:, 1])
        probabilities = np.vstack(
            [
                score_probabilities(home, away, rho)
                for home, away, rho in zip(
                    home_rates,
                    away_rates,
                    test["rho"].to_numpy(float),
                    strict=True,
                )
            ]
        )
        model = (
            GATED_NEURAL_LINEUP_MODEL_NAME
            if selected is NeuralFeatureAblation.COMBINED
            else f"{NEURAL_LINEUP_MODEL_NAME}_{selected.value}_gated"
        )
        predictions.append(
            gated_prediction_frame(
                test,
                probabilities,
                model,
                selected,
                fold_gate,
                before,
                after,
            )
        )
    frame = (
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
                *PROBABILITY_COLUMNS,
            ]
        )
    )
    return metrics_by_season(frame), frame
