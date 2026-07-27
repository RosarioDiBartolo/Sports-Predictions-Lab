"""Regularized confirmed-lineup corrections for Dixon-Coles goal rates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .hybrid import HybridPredictor, _score_probabilities, fit_hybrid_model
from .models import (
    OUTCOMES,
    PROBABILITY_COLUMNS,
    _probability_metrics,
    _promotion_evidence,
    _weighted_summary,
    paired_log_loss_bootstrap,
)

CONFIRMED_LINEUP_MODEL_NAME = "dixon_coles_confirmed_lineup_pooling"
CONFIRMED_LINEUP_BASELINE_NAME = "dixon_coles_without_confirmed_lineup"
LINEUP_FEATURE_MARKER = "_confirmed_lineup_"


def confirmed_lineup_feature_columns(columns: pd.Index) -> list[str]:
    """Return the reviewed numeric inputs produced from official lineups."""
    return sorted(
        column
        for column in columns
        if isinstance(column, str) and LINEUP_FEATURE_MARKER in column
    )


def _correction_estimator(alpha: float) -> Pipeline:
    if alpha <= 0:
        raise ValueError("alpha deve essere positivo.")
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _base_rates(
    predictor: HybridPredictor,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    home = np.clip(predictor.home_goal_model.predict(features), 1e-6, 8.0)
    away = np.clip(predictor.away_goal_model.predict(features), 1e-6, 8.0)
    return home, away


@dataclass
class ConfirmedLineupPredictor:
    """Dixon-Coles with bounded, regularized lineup adjustments."""

    base: HybridPredictor
    home_correction: Pipeline
    away_correction: Pipeline
    lineup_features: tuple[str, ...]
    trained_seasons: tuple[str, ...]
    alpha: float
    maximum_log_correction: float = 0.35

    def goal_rates(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = set(self.lineup_features).difference(features.columns)
        if missing:
            raise ValueError(
                f"Feature formazione confermata mancanti: {sorted(missing)}"
            )
        base_home, base_away = _base_rates(self.base, features)
        lineup = features[list(self.lineup_features)]
        home_delta = np.clip(
            self.home_correction.predict(lineup),
            -self.maximum_log_correction,
            self.maximum_log_correction,
        )
        away_delta = np.clip(
            self.away_correction.predict(lineup),
            -self.maximum_log_correction,
            self.maximum_log_correction,
        )
        return (
            np.clip(base_home * np.exp(home_delta), 1e-6, 8.0),
            np.clip(base_away * np.exp(away_delta), 1e-6, 8.0),
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        home_rates, away_rates = self.goal_rates(features)
        return np.vstack(
            [
                _score_probabilities(home, away, self.base.rho)
                for home, away in zip(home_rates, away_rates, strict=True)
            ]
        )


@dataclass
class ConfirmedLineupModelResult:
    """Evaluation artifacts and the fitted confirmed-lineup candidate."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    outputs: dict[str, Path]
    predictor: ConfirmedLineupPredictor


def fit_confirmed_lineup_model(
    features: pd.DataFrame,
    *,
    alpha: float = 25.0,
    maximum_log_correction: float = 0.35,
) -> ConfirmedLineupPredictor:
    """Fit a no-lineup Dixon-Coles base and two shrinkage corrections."""
    lineup_features = confirmed_lineup_feature_columns(features.columns)
    if not lineup_features:
        raise ValueError(
            "Nessuna feature *_confirmed_lineup_* disponibile per il training."
        )
    required = {
        "season",
        "home_goals",
        "away_goals",
        "result",
        *lineup_features,
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Colonne modello lineup mancanti: {sorted(missing)}")
    data = features.loc[features["result"].isin(OUTCOMES)].copy()
    if data.empty or data[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("Il training lineup richiede gol finali completi.")

    base_frame = data.drop(columns=lineup_features)
    base = fit_hybrid_model(base_frame, include_player_features=False)
    base_home, base_away = _base_rates(base, base_frame)
    home_target = np.log((data["home_goals"].to_numpy(float) + 0.5) / (base_home + 0.5))
    away_target = np.log((data["away_goals"].to_numpy(float) + 0.5) / (base_away + 0.5))
    home_correction = _correction_estimator(alpha)
    away_correction = _correction_estimator(alpha)
    home_correction.fit(data[lineup_features], home_target)
    away_correction.fit(data[lineup_features], away_target)
    return ConfirmedLineupPredictor(
        base=base,
        home_correction=home_correction,
        away_correction=away_correction,
        lineup_features=tuple(lineup_features),
        trained_seasons=tuple(sorted(data["season"].astype(str).unique())),
        alpha=alpha,
        maximum_log_correction=maximum_log_correction,
    )


def _baseline_probabilities(
    predictor: ConfirmedLineupPredictor,
    features: pd.DataFrame,
) -> np.ndarray:
    home_rates, away_rates = _base_rates(predictor.base, features)
    return np.vstack(
        [
            _score_probabilities(home, away, predictor.base.rho)
            for home, away in zip(home_rates, away_rates, strict=True)
        ]
    )


def _rps(results: pd.Series, probabilities: np.ndarray) -> float:
    indices = results.map({value: index for index, value in enumerate(OUTCOMES)})
    observed = np.eye(len(OUTCOMES))[indices.to_numpy(dtype=int)]
    return float(
        np.mean(
            np.sum(
                (np.cumsum(probabilities, axis=1)[:, :-1]
                 - np.cumsum(observed, axis=1)[:, :-1])
                ** 2,
                axis=1,
            )
            / (len(OUTCOMES) - 1)
        )
    )


def walk_forward_confirmed_lineup_model(
    features: pd.DataFrame,
    *,
    alpha: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate lineup correction and its no-lineup base on identical folds."""
    required = {"season", "match_id", "league", "result"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Colonne valutazione lineup mancanti: {sorted(missing)}")
    data = features.loc[features["result"].isin(OUTCOMES)].copy()
    data["season"] = data["season"].astype(str).str.zfill(4)
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for season in sorted(data["season"].unique())[1:]:
        training = data.loc[data["season"].lt(season)]
        test = data.loc[data["season"].eq(season)]
        if set(training["result"]) != set(OUTCOMES) or test.empty:
            continue
        predictor = fit_confirmed_lineup_model(training, alpha=alpha)
        variants = (
            (CONFIRMED_LINEUP_MODEL_NAME, predictor.predict_proba(test)),
            (CONFIRMED_LINEUP_BASELINE_NAME, _baseline_probabilities(predictor, test)),
        )
        for model_name, probabilities in variants:
            metrics.append(
                {
                    "season": season,
                    "model": model_name,
                    **_probability_metrics(test["result"], probabilities),
                    "rps": _rps(test["result"], probabilities),
                    "calibrated": False,
                }
            )
            frame = test[["match_id", "season", "league", "result"]].copy()
            frame["model"] = model_name
            frame[list(PROBABILITY_COLUMNS)] = probabilities
            predictions.append(frame)
    metric_columns = [
        "season", "model", "matches", "log_loss", "brier", "rps",
        "accuracy", "ece", "calibrated",
    ]
    prediction_columns = [
        "match_id", "season", "league", "result", "model",
        *PROBABILITY_COLUMNS,
    ]
    return (
        pd.DataFrame(metrics, columns=metric_columns),
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=prediction_columns),
    )


def export_confirmed_lineup_model(
    features: pd.DataFrame,
    destination: Path,
    *,
    alpha: float = 25.0,
) -> ConfirmedLineupModelResult:
    """Persist the candidate only after the common walk-forward evaluation."""
    destination.mkdir(parents=True, exist_ok=True)
    metrics, all_predictions = walk_forward_confirmed_lineup_model(
        features, alpha=alpha
    )
    candidate_metrics = metrics.loc[
        metrics["model"].eq(CONFIRMED_LINEUP_MODEL_NAME)
    ].copy()
    baseline_metrics = metrics.loc[
        metrics["model"].eq(CONFIRMED_LINEUP_BASELINE_NAME)
    ].copy()
    candidate_predictions = all_predictions.loc[
        all_predictions["model"].eq(CONFIRMED_LINEUP_MODEL_NAME)
    ].copy()
    baseline_predictions = all_predictions.loc[
        all_predictions["model"].eq(CONFIRMED_LINEUP_BASELINE_NAME)
    ].copy()
    bootstrap = paired_log_loss_bootstrap(
        candidate_predictions, baseline_predictions
    )
    summary = _weighted_summary(metrics)
    summary["rps"] = [
        float(
            np.average(
                metrics.loc[metrics["model"].eq(model), "rps"],
                weights=metrics.loc[metrics["model"].eq(model), "matches"],
            )
        )
        for model in summary["model"]
    ]
    promotion = _promotion_evidence(
        summary,
        summary,
        candidate_metrics,
        baseline_metrics,
        bootstrap,
        candidate_model=CONFIRMED_LINEUP_MODEL_NAME,
        baseline_model=CONFIRMED_LINEUP_BASELINE_NAME,
    )
    candidate_rps = summary.loc[
        summary["model"].eq(CONFIRMED_LINEUP_MODEL_NAME), "rps"
    ]
    baseline_rps = summary.loc[
        summary["model"].eq(CONFIRMED_LINEUP_BASELINE_NAME), "rps"
    ]
    rps_not_worse = bool(
        not candidate_rps.empty
        and not baseline_rps.empty
        and candidate_rps.iloc[0] <= baseline_rps.iloc[0]
    )
    promotion["rps_not_worse"] = rps_not_worse
    promotion["promoted"] = bool(promotion["promoted"] and rps_not_worse)
    predictor = fit_confirmed_lineup_model(features, alpha=alpha)

    paths = {
        "metrics": destination / "confirmed_lineup_metrics_by_season.csv",
        "predictions": destination / "confirmed_lineup_predictions.csv",
        "comparison": destination / "confirmed_lineup_comparison.csv",
        "bootstrap": destination / "confirmed_lineup_bootstrap.json",
        "metadata": destination / "confirmed_lineup_model.meta.json",
        "model": destination / "confirmed_lineup_model.joblib",
        "report": destination / "CONFIRMED_LINEUP_MODEL_REPORT.md",
    }
    metrics.to_csv(paths["metrics"], index=False)
    candidate_predictions.to_csv(paths["predictions"], index=False)
    summary.to_csv(paths["comparison"], index=False)
    paths["bootstrap"].write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "model_name": CONFIRMED_LINEUP_MODEL_NAME,
        "algorithm": "Dixon-Coles + regularized confirmed-lineup pooling correction",
        "official_model_unchanged": True,
        "alpha": alpha,
        "maximum_log_correction": predictor.maximum_log_correction,
        "lineup_features": list(predictor.lineup_features),
        "trained_seasons": list(predictor.trained_seasons),
        "excluded_inputs": [
            "odds",
            "market probabilities",
            "current-match performance",
        ],
        "bootstrap_vs_no_lineup": bootstrap,
        "promotion": promotion,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(predictor, paths["model"])
    report = [
        "# Dixon-Coles + correzione lineup confermata",
        "",
        "Il candidato non sostituisce il modello ufficiale senza promotion gate OOS.",
        "",
        "## Valutazione walk-forward",
        "",
        "| Modello | Match | Log Loss | Brier | RPS | Accuracy | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        report.append(
            f"| {row['model']} | {int(row['matches'])} | "
            f"{row['log_loss']:.4f} | {row['brier']:.4f} | "
            f"{row['rps']:.4f} | {row['accuracy']:.2%} | {row['ece']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Promotion gate",
            "",
            f"- IC 95% Log Loss favorevole: {promotion['significant_log_loss']}.",
            f"- Stagioni vinte: {promotion['season_wins']}/"
            f"{len(candidate_metrics)}.",
            f"- Brier non peggiore: {promotion['brier_not_worse']}.",
            f"- RPS non peggiore: {promotion['rps_not_worse']}.",
            f"- ECE non peggiore: {promotion['ece_not_worse']}.",
            f"- Verdetto: {'PROMOSSO' if promotion['promoted'] else 'NON PROMOSSO'}.",
        ]
    )
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    return ConfirmedLineupModelResult(
        candidate_metrics,
        candidate_predictions,
        paths,
        predictor,
    )


def load_confirmed_lineup_model(path: Path) -> ConfirmedLineupPredictor:
    predictor = joblib.load(path)
    if not isinstance(predictor, ConfirmedLineupPredictor):
        raise TypeError(f"Artefatto modello lineup non valido: {path}")
    return predictor
