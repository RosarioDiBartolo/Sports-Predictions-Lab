from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .baseline_modeling import (
    OUTCOMES,
    _probability_metrics,
    _sport_feature_allowlist,
    walk_forward_baselines,
)

MODEL_NAME = "sport_gradient_boosting"
PROBABILITY_COLUMNS = (
    "probability_home",
    "probability_draw",
    "probability_away",
)


@dataclass
class SportOnlyPredictor:
    """Serializable target-free 1X2 predictor."""

    estimator: Pipeline
    numeric_features: tuple[str, ...]
    trained_seasons: tuple[str, ...]
    calibrator: LogisticRegression | None = None
    calibration_season: str | None = None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        required = {*self.numeric_features, "league"}
        missing = required.difference(features.columns)
        if missing:
            raise ValueError(f"Feature sportive mancanti: {sorted(missing)}")
        probabilities = _aligned_probabilities(self.estimator, features)
        if self.calibrator is not None:
            probabilities = _aligned_probabilities(
                self.calibrator,
                np.log(np.clip(probabilities, 1e-15, 1.0)),
            )
        probabilities = np.clip(probabilities, 1e-15, 1.0)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


@dataclass
class SportModelResult:
    """Evaluation, predictions and persisted artifacts for the candidate."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    outputs: dict[str, Path]
    predictor: SportOnlyPredictor | None


def _make_estimator(numeric_features: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric_features,
            ),
            (
                "league",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["league"],
            ),
        ]
    )
    return Pipeline(
        [
            ("features", transformer),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=200,
                    max_leaf_nodes=7,
                    min_samples_leaf=40,
                    l2_regularization=5.0,
                    random_state=42,
                ),
            ),
        ]
    )


def _aligned_probabilities(estimator: Any, features: Any) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = list(estimator.classes_)
    return probabilities[:, [classes.index(outcome) for outcome in OUTCOMES]]


def _validated_training_data(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = {"season", "league", "result", "elo_difference"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Colonne modello mancanti: {sorted(missing)}")
    data = features.loc[features["result"].isin(OUTCOMES)].copy()
    if set(data["result"]) != set(OUTCOMES):
        raise ValueError("Il training deve contenere gli esiti H, D e A.")
    data["season"] = data["season"].astype(str).str.zfill(4)
    numeric_features = _sport_feature_allowlist(data.columns)
    if not numeric_features:
        raise ValueError("Non sono disponibili feature sportive approvate.")
    return data, numeric_features


def fit_sport_model(features: pd.DataFrame) -> SportOnlyPredictor:
    """Fit the candidate and calibrate it on the latest historical season."""
    data, numeric_features = _validated_training_data(features)
    seasons = tuple(sorted(data["season"].unique()))
    calibrator: LogisticRegression | None = None
    calibration_season: str | None = None

    if len(seasons) >= 2:
        candidate_season = seasons[-1]
        calibration = data.loc[data["season"].eq(candidate_season)]
        inner_training = data.loc[data["season"].lt(candidate_season)]
        if (
            set(inner_training["result"]) == set(OUTCOMES)
            and set(calibration["result"]) == set(OUTCOMES)
        ):
            inner_estimator = _make_estimator(numeric_features)
            inner_estimator.fit(inner_training, inner_training["result"])
            raw = _aligned_probabilities(inner_estimator, calibration)
            calibrator = LogisticRegression(max_iter=2000, C=1.0)
            calibrator.fit(
                np.log(np.clip(raw, 1e-15, 1.0)),
                calibration["result"],
            )
            calibration_season = candidate_season

    estimator = _make_estimator(numeric_features)
    estimator.fit(data, data["result"])
    return SportOnlyPredictor(
        estimator=estimator,
        numeric_features=tuple(numeric_features),
        trained_seasons=seasons,
        calibrator=calibrator,
        calibration_season=calibration_season,
    )


def predict_fixtures(
    predictor: SportOnlyPredictor,
    fixture_features: pd.DataFrame,
) -> pd.DataFrame:
    """Return target-free fixture probabilities in stable H/D/A order."""
    if "result" in fixture_features and fixture_features["result"].notna().any():
        raise ValueError("La previsione fixture non deve contenere target osservati.")
    probabilities = predictor.predict_proba(fixture_features)
    identity = [
        column
        for column in (
            "match_id",
            "date",
            "season",
            "league",
            "home_team",
            "away_team",
        )
        if column in fixture_features
    ]
    output = fixture_features[identity].reset_index(drop=True).copy()
    output[list(PROBABILITY_COLUMNS)] = probabilities
    output["predicted_result"] = np.asarray(OUTCOMES)[probabilities.argmax(axis=1)]
    output["confidence"] = probabilities.max(axis=1)
    return output


def walk_forward_sport_model(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each season with a model fitted only on earlier seasons."""
    data, _ = _validated_training_data(features)
    seasons = sorted(data["season"].unique())
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for season in seasons[1:]:
        training = data.loc[data["season"].lt(season)]
        test = data.loc[data["season"].eq(season)]
        if set(training["result"]) != set(OUTCOMES) or test.empty:
            continue
        predictor = fit_sport_model(training)
        probabilities = predictor.predict_proba(test)
        metric_rows.append(
            {
                "season": season,
                "model": MODEL_NAME,
                **_probability_metrics(test["result"], probabilities),
                "calibrated": predictor.calibrator is not None,
            }
        )
        identity = test[["match_id", "season", "league", "result"]].copy()
        identity["model"] = MODEL_NAME
        identity[list(PROBABILITY_COLUMNS)] = probabilities
        prediction_rows.append(identity)

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
    metrics = pd.DataFrame(metric_rows, columns=metric_columns)
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame(columns=prediction_columns)
    )
    return metrics, predictions


def _weighted_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in metrics.groupby("model", sort=False):
        weights = group["matches"].astype(float)
        rows.append(
            {
                "model": model,
                "matches": int(weights.sum()),
                **{
                    metric: float(np.average(group[metric], weights=weights))
                    for metric in ("log_loss", "brier", "accuracy", "ece")
                },
            }
        )
    return pd.DataFrame(rows)


def export_sport_model(
    features: pd.DataFrame,
    destination: Path,
) -> SportModelResult:
    """Persist walk-forward evidence and the final train-on-all model."""
    destination.mkdir(parents=True, exist_ok=True)
    training_error: str | None = None
    try:
        metrics, predictions = walk_forward_sport_model(features)
    except ValueError as error:
        training_error = str(error)
        metrics = pd.DataFrame(
            columns=[
                "season",
                "model",
                "matches",
                "log_loss",
                "brier",
                "accuracy",
                "ece",
                "calibrated",
            ]
        )
        predictions = pd.DataFrame(
            columns=[
                "match_id",
                "season",
                "league",
                "result",
                "model",
                *PROBABILITY_COLUMNS,
            ]
        )
    metrics_path = destination / "sport_model_metrics_by_season.csv"
    predictions_path = destination / "sport_model_predictions.csv"
    comparison_path = destination / "sport_model_comparison.csv"
    report_path = destination / "SPORT_MODEL_REPORT.md"
    metadata_path = destination / "sport_model.meta.json"
    model_path = destination / "sport_model.joblib"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    baseline_metrics, _ = walk_forward_baselines(features)
    references = baseline_metrics.loc[
        baseline_metrics["model"].isin(("sport_features", "market_closing"))
    ].copy()
    comparison = pd.concat(
        [metrics.drop(columns=["calibrated"], errors="ignore"), references],
        ignore_index=True,
    )
    summary = _weighted_summary(comparison) if not comparison.empty else pd.DataFrame()
    summary.to_csv(comparison_path, index=False)

    predictor: SportOnlyPredictor | None = None
    try:
        predictor = fit_sport_model(features)
        joblib.dump(predictor, model_path)
    except ValueError as error:
        training_error = str(error)

    metadata = {
        "model_name": MODEL_NAME,
        "algorithm": "HistGradientBoostingClassifier",
        "target": list(OUTCOMES),
        "training_rows": int(len(features)),
        "trained_seasons": list(predictor.trained_seasons) if predictor else [],
        "calibration_season": predictor.calibration_season if predictor else None,
        "sport_features": list(predictor.numeric_features) if predictor else [],
        "excluded_inputs": ["odds", "market probabilities", "final targets"],
        "training_error": training_error,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "# Modello predittivo sport-only",
        "",
        "Il candidato usa esclusivamente feature sportive pre-partita. "
        "Le quote closing compaiono solo come benchmark esterno.",
        "",
        "## Confronto walk-forward",
        "",
        "| Modello | Match OOS | Log Loss | Brier | Accuracy | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        report_lines.append(
            f"| {row['model']} | {int(row['matches'])} | "
            f"{row['log_loss']:.4f} | {row['brier']:.4f} | "
            f"{row['accuracy']:.2%} | {row['ece']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Garanzie",
            "",
            "- Ogni stagione è prevista usando soltanto stagioni precedenti.",
            "- Le feature sono allowlistate; quote e target finali "
            "non entrano nel modello.",
            "- La calibrazione usa l’ultima stagione interna al training, mai il test.",
            "- Il modello finale è addestrato su tutto lo storico disponibile.",
        ]
    )
    if training_error:
        report_lines.extend(["", f"Modello non esportato: {training_error}"])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    outputs = {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "comparison": comparison_path,
        "report": report_path,
        "metadata": metadata_path,
    }
    if predictor is not None:
        outputs["model"] = model_path
    return SportModelResult(metrics, predictions, outputs, predictor)


def load_sport_model(path: Path) -> SportOnlyPredictor:
    """Load and validate one persisted sport-only predictor."""
    predictor = joblib.load(path)
    if not isinstance(predictor, SportOnlyPredictor):
        raise TypeError(f"Artefatto modello non valido: {path}")
    return predictor
