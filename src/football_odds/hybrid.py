"""Sport-only Dixon-Coles goal model blended with gradient boosting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import lgamma
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .models import (
    OUTCOMES,
    PROBABILITY_COLUMNS,
    SportModelResult,
    SportOnlyPredictor,
    _probability_metrics,
    _promotion_evidence,
    _sport_feature_allowlist,
    _validated_training_data,
    _weighted_summary,
    fit_sport_model,
    paired_log_loss_bootstrap,
    prediction_error_diagnostics,
    walk_forward_baselines,
    walk_forward_sport_model,
)

HYBRID_MODEL_NAME = "dixon_coles_gradient_boosting"
GOAL_IDENTITY_FEATURES = ("home_team", "away_team", "league")


def _goal_estimator(numeric_features: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "identity",
                OneHotEncoder(handle_unknown="ignore"),
                list(GOAL_IDENTITY_FEATURES),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", transformer),
            ("model", PoissonRegressor(alpha=1.0, max_iter=500)),
        ]
    )


def _poisson_mass(goals: np.ndarray, rate: float) -> np.ndarray:
    return np.exp(goals * np.log(rate) - rate - np.vectorize(lgamma)(goals + 1))


def _dc_tau(
    home_goals: int,
    away_goals: int,
    home_rate: float,
    away_rate: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_rate * away_rate * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_rate * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_rate * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _score_probabilities(
    home_rate: float,
    away_rate: float,
    rho: float,
    *,
    maximum_goals: int = 10,
) -> np.ndarray:
    goals = np.arange(maximum_goals + 1)
    matrix = np.outer(
        _poisson_mass(goals, max(home_rate, 1e-6)),
        _poisson_mass(goals, max(away_rate, 1e-6)),
    )
    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
        matrix[home_goals, away_goals] *= max(
            _dc_tau(home_goals, away_goals, home_rate, away_rate, rho),
            1e-9,
        )
    probabilities = np.asarray(
        [np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()]
    )
    return probabilities / probabilities.sum()


def _fit_rho(
    data: pd.DataFrame,
    home_rates: np.ndarray,
    away_rates: np.ndarray,
) -> float:
    low_score = (
        data["home_goals"].isin((0, 1)) & data["away_goals"].isin((0, 1))
    ).to_numpy()
    if not low_score.any():
        return 0.0
    home_goals = data["home_goals"].to_numpy(dtype=int)[low_score]
    away_goals = data["away_goals"].to_numpy(dtype=int)[low_score]
    candidates = np.linspace(-0.2, 0.2, 81)
    likelihoods = []
    for rho in candidates:
        corrections = np.asarray(
            [
                _dc_tau(hg, ag, hr, ar, float(rho))
                for hg, ag, hr, ar in zip(
                    home_goals,
                    away_goals,
                    home_rates[low_score],
                    away_rates[low_score],
                    strict=True,
                )
            ]
        )
        likelihoods.append(
            np.log(np.clip(corrections, 1e-9, None)).sum()
        )
    return float(candidates[int(np.argmax(likelihoods))])


@dataclass
class HybridPredictor:
    """Serializable goal/GB ensemble with a fixed sport-only blend."""

    home_goal_model: Pipeline
    away_goal_model: Pipeline
    gradient_boosting: SportOnlyPredictor
    numeric_features: tuple[str, ...]
    trained_seasons: tuple[str, ...]
    rho: float
    goal_weight: float = 0.55

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        required = {*self.numeric_features, *GOAL_IDENTITY_FEATURES}
        missing = required.difference(features.columns)
        if missing:
            raise ValueError(f"Feature ibride mancanti: {sorted(missing)}")
        home_rates = np.clip(self.home_goal_model.predict(features), 1e-6, 8.0)
        away_rates = np.clip(self.away_goal_model.predict(features), 1e-6, 8.0)
        goal_probabilities = np.vstack(
            [
                _score_probabilities(home_rate, away_rate, self.rho)
                for home_rate, away_rate in zip(
                    home_rates, away_rates, strict=True
                )
            ]
        )
        boosted = self.gradient_boosting.predict_proba(features)
        blended = self.goal_weight * goal_probabilities + (
            1.0 - self.goal_weight
        ) * boosted
        return blended / blended.sum(axis=1, keepdims=True)


def fit_hybrid_model(features: pd.DataFrame) -> HybridPredictor:
    """Fit goal rates, Dixon-Coles dependence and the existing sport model."""
    data, numeric_features = _validated_training_data(features)
    required = {"home_team", "away_team", "home_goals", "away_goals"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonne modello ibrido mancanti: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Il modello ibrido richiede squadre e gol completi.")
    reviewed_features = _sport_feature_allowlist(data.columns)
    home_model = _goal_estimator(reviewed_features)
    away_model = _goal_estimator(reviewed_features)
    home_model.fit(data, data["home_goals"])
    away_model.fit(data, data["away_goals"])
    home_rates = np.clip(home_model.predict(data), 1e-6, 8.0)
    away_rates = np.clip(away_model.predict(data), 1e-6, 8.0)
    return HybridPredictor(
        home_goal_model=home_model,
        away_goal_model=away_model,
        gradient_boosting=fit_sport_model(data),
        numeric_features=tuple(numeric_features),
        trained_seasons=tuple(sorted(data["season"].unique())),
        rho=_fit_rho(data, home_rates, away_rates),
    )


def walk_forward_hybrid_model(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the hybrid candidate using only seasons before each test fold."""
    data, _ = _validated_training_data(features)
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for season in sorted(data["season"].unique())[1:]:
        training = data.loc[data["season"].lt(season)]
        test = data.loc[data["season"].eq(season)]
        if set(training["result"]) != set(OUTCOMES) or test.empty:
            continue
        predictor = fit_hybrid_model(training)
        probabilities = predictor.predict_proba(test)
        metrics.append(
            {
                "season": season,
                "model": HYBRID_MODEL_NAME,
                **_probability_metrics(test["result"], probabilities),
                "calibrated": predictor.gradient_boosting.calibrator is not None,
            }
        )
        frame = test[["match_id", "season", "league", "result"]].copy()
        frame["model"] = HYBRID_MODEL_NAME
        frame[list(PROBABILITY_COLUMNS)] = probabilities
        predictions.append(frame)
    metric_columns = [
        "season",
        "model",
        "matches",
        "log_loss",
        "brier",
        "accuracy",
        "ece",
        "calibrated",
    ]
    prediction_columns = [
        "match_id",
        "season",
        "league",
        "result",
        "model",
        *PROBABILITY_COLUMNS,
    ]
    return (
        pd.DataFrame(metrics, columns=metric_columns),
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=prediction_columns),
    )


def export_hybrid_model(
    features: pd.DataFrame,
    destination: Path,
) -> SportModelResult:
    """Persist the hybrid candidate and evidence against the official model."""
    destination.mkdir(parents=True, exist_ok=True)
    metrics, predictions = walk_forward_hybrid_model(features)
    official_metrics, official_predictions = walk_forward_sport_model(features)
    baseline_metrics, _ = walk_forward_baselines(features)
    market_metrics = baseline_metrics.loc[
        baseline_metrics["model"].eq("market_closing")
    ]
    references = pd.concat([official_metrics, market_metrics], ignore_index=True)
    comparison = pd.concat(
        [metrics, references], ignore_index=True
    ).drop(columns=["calibrated"], errors="ignore")
    summary = _weighted_summary(comparison)
    bootstrap = paired_log_loss_bootstrap(predictions, official_predictions)
    promotion = _promotion_evidence(
        _weighted_summary(metrics),
        _weighted_summary(official_metrics),
        metrics,
        official_metrics,
        bootstrap,
        candidate_model=HYBRID_MODEL_NAME,
        baseline_model="sport_gradient_boosting",
    )
    predictor = fit_hybrid_model(features)

    metrics_path = destination / "hybrid_metrics_by_season.csv"
    predictions_path = destination / "hybrid_predictions.csv"
    comparison_path = destination / "hybrid_comparison.csv"
    bootstrap_path = destination / "hybrid_bootstrap.json"
    metadata_path = destination / "hybrid_model.meta.json"
    model_path = destination / "hybrid_model.joblib"
    report_path = destination / "HYBRID_MODEL_REPORT.md"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(comparison_path, index=False)
    bootstrap_path.write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    joblib.dump(predictor, model_path)
    diagnostics = prediction_error_diagnostics(predictions, features)
    diagnostic_paths: dict[str, Path] = {}
    for name, table in diagnostics.items():
        path = destination / f"hybrid_error_by_{name}.csv"
        table.to_csv(path, index=False)
        diagnostic_paths[name] = path
    metadata = {
        "model_name": HYBRID_MODEL_NAME,
        "algorithm": "Dixon-Coles + PoissonRegressor + HistGradientBoostingClassifier",
        "official_model_unchanged": True,
        "goal_weight": predictor.goal_weight,
        "rho": predictor.rho,
        "trained_seasons": list(predictor.trained_seasons),
        "sport_features": list(predictor.numeric_features),
        "excluded_inputs": ["odds", "market probabilities", "final targets"],
        "bootstrap_vs_official": bootstrap,
        "promotion": promotion,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Candidato ibrido Dixon–Coles + gradient boosting",
        "",
        "Il modello ufficiale resta `sport_gradient_boosting`. Il candidato è "
        "promuovibile solo dal gate walk-forward.",
        "",
        "## Confronto fuori campione",
        "",
        "| Modello | Match | Log Loss | Brier | Accuracy | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['model']} | {int(row['matches'])} | "
            f"{row['log_loss']:.4f} | {row['brier']:.4f} | "
            f"{row['accuracy']:.2%} | {row['ece']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Promotion gate",
            "",
            f"- IC 95% favorevole: {promotion['significant_log_loss']}.",
            f"- Stagioni vinte: {promotion['season_wins']}/"
            f"{len(metrics)} (richieste {promotion['required_season_wins']}).",
            f"- Brier non peggiore: {promotion['brier_not_worse']}.",
            f"- ECE non peggiore: {promotion['ece_not_worse']}.",
            f"- Verdetto: {'PROMOSSO' if promotion['promoted'] else 'NON PROMOSSO'}.",
            "",
            "Le quote closing sono utilizzate esclusivamente come benchmark.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "comparison": comparison_path,
        "bootstrap": bootstrap_path,
        "metadata": metadata_path,
        "model": model_path,
        "report": report_path,
        **{f"error_by_{name}": path for name, path in diagnostic_paths.items()},
    }
    return SportModelResult(metrics, predictions, outputs, predictor)


def load_hybrid_model(path: Path) -> HybridPredictor:
    """Load and validate one persisted hybrid candidate."""
    predictor = joblib.load(path)
    if not isinstance(predictor, HybridPredictor):
        raise TypeError(f"Artefatto modello ibrido non valido: {path}")
    return predictor
