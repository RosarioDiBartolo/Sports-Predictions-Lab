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


def paired_log_loss_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Estimate uncertainty of candidate-minus-reference paired log loss."""
    if samples <= 0:
        raise ValueError("samples deve essere positivo.")
    keys = ["match_id", "season"]
    probability_columns = list(PROBABILITY_COLUMNS)
    left = candidate[keys + ["result", *probability_columns]].rename(
        columns={column: f"{column}_candidate" for column in probability_columns}
    )
    right = reference[keys + probability_columns].rename(
        columns={column: f"{column}_reference" for column in probability_columns}
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        return {
            "matches": 0,
            "mean_log_loss_difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_candidate_better": float("nan"),
            "verdict": "insufficient_data",
        }
    indices = paired["result"].map(
        {outcome: index for index, outcome in enumerate(OUTCOMES)}
    ).to_numpy()
    candidate_probabilities = paired[
        [f"{column}_candidate" for column in probability_columns]
    ].to_numpy(dtype=float)
    reference_probabilities = paired[
        [f"{column}_reference" for column in probability_columns]
    ].to_numpy(dtype=float)
    row_indices = np.arange(len(paired))
    candidate_loss = -np.log(
        np.clip(candidate_probabilities[row_indices, indices], 1e-15, 1.0)
    )
    reference_loss = -np.log(
        np.clip(reference_probabilities[row_indices, indices], 1e-15, 1.0)
    )
    differences = candidate_loss - reference_loss
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=float)
    for sample in range(samples):
        selection = rng.integers(0, len(differences), size=len(differences))
        bootstrap_means[sample] = float(differences[selection].mean())
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "matches": int(len(differences)),
        "mean_log_loss_difference": float(differences.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "probability_candidate_better": float((bootstrap_means < 0).mean()),
        "verdict": "candidate_better" if ci_high < 0 else "inconclusive",
    }


def _diagnostic_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(group_column, observed=True, dropna=False):
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        rows.append(
            {
                group_column: str(value),
                **_probability_metrics(group["result"], probabilities),
            }
        )
    return pd.DataFrame(rows)


def prediction_error_diagnostics(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Segment OOS errors by league, result, experience and confidence."""
    if predictions.empty:
        empty = pd.DataFrame(
            columns=["segment", "matches", "log_loss", "brier", "accuracy", "ece"]
        )
        return {
            "league": empty.copy(),
            "result": empty.copy(),
            "experience": empty.copy(),
            "confidence": empty.copy(),
        }
    context_columns = [
        column
        for column in ("match_id", "home_matches_played", "away_matches_played")
        if column in features
    ]
    context = features[context_columns].drop_duplicates("match_id")
    data = predictions.merge(context, on="match_id", how="left")
    played = data[["home_matches_played", "away_matches_played"]].min(axis=1)
    data["experience"] = pd.cut(
        played,
        bins=[-np.inf, 4, 14, np.inf],
        labels=["0-4", "5-14", "15+"],
    )
    confidence = data[list(PROBABILITY_COLUMNS)].max(axis=1)
    data["confidence"] = pd.cut(
        confidence,
        bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
        include_lowest=True,
    )
    return {
        "league": _diagnostic_table(data, "league"),
        "result": _diagnostic_table(data, "result"),
        "experience": _diagnostic_table(data, "experience"),
        "confidence": _diagnostic_table(data, "confidence"),
    }


def _promotion_evidence(
    candidate_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    bootstrap: dict[str, float | int | str],
) -> dict[str, bool | int]:
    if "model" not in candidate_summary or "model" not in baseline_summary:
        return {
            "promoted": False,
            "season_wins": 0,
            "required_season_wins": 0,
            "significant_log_loss": False,
            "brier_not_worse": False,
            "ece_not_worse": False,
        }
    candidate_row = candidate_summary.loc[
        candidate_summary["model"].eq(MODEL_NAME)
    ]
    baseline_row = baseline_summary.loc[
        baseline_summary["model"].eq("sport_features")
    ]
    season_pairs = candidate_metrics.merge(
        baseline_metrics.loc[baseline_metrics["model"].eq("sport_features")],
        on="season",
        suffixes=("_candidate", "_baseline"),
    )
    season_wins = int(
        (
            season_pairs["log_loss_candidate"]
            < season_pairs["log_loss_baseline"]
        ).sum()
    )
    required_wins = len(season_pairs) // 2 + 1
    if candidate_row.empty or baseline_row.empty:
        return {
            "promoted": False,
            "season_wins": season_wins,
            "required_season_wins": required_wins,
            "significant_log_loss": False,
            "brier_not_worse": False,
            "ece_not_worse": False,
        }
    candidate_values = candidate_row.iloc[0]
    baseline_values = baseline_row.iloc[0]
    significant = bool(
        bootstrap.get("verdict") == "candidate_better"
    )
    brier_not_worse = bool(
        candidate_values["brier"] <= baseline_values["brier"]
    )
    ece_not_worse = bool(candidate_values["ece"] <= baseline_values["ece"])
    promoted = bool(
        significant
        and season_wins >= required_wins
        and brier_not_worse
        and ece_not_worse
    )
    return {
        "promoted": promoted,
        "season_wins": season_wins,
        "required_season_wins": required_wins,
        "significant_log_loss": significant,
        "brier_not_worse": brier_not_worse,
        "ece_not_worse": ece_not_worse,
    }


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
    reference_metrics_path = (
        destination / "sport_model_reference_metrics_by_season.csv"
    )
    predictions_path = destination / "sport_model_predictions.csv"
    comparison_path = destination / "sport_model_comparison.csv"
    bootstrap_path = destination / "sport_model_bootstrap.json"
    report_path = destination / "SPORT_MODEL_REPORT.md"
    metadata_path = destination / "sport_model.meta.json"
    model_path = destination / "sport_model.joblib"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    baseline_metrics, baseline_predictions = walk_forward_baselines(features)
    references = baseline_metrics.loc[
        baseline_metrics["model"].isin(("sport_features", "market_closing"))
    ].copy()
    references.to_csv(reference_metrics_path, index=False)
    comparison = pd.concat(
        [metrics.drop(columns=["calibrated"], errors="ignore"), references],
        ignore_index=True,
    )
    summary = _weighted_summary(comparison) if not comparison.empty else pd.DataFrame()
    summary.to_csv(comparison_path, index=False)
    candidate_summary = (
        _weighted_summary(metrics) if not metrics.empty else pd.DataFrame()
    )
    baseline_summary = (
        _weighted_summary(baseline_metrics)
        if not baseline_metrics.empty
        else pd.DataFrame()
    )
    logistic_predictions = baseline_predictions.loc[
        baseline_predictions["model"].eq("sport_features")
    ]
    bootstrap = paired_log_loss_bootstrap(
        predictions,
        logistic_predictions,
    )
    bootstrap_path.write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    diagnostics = prediction_error_diagnostics(predictions, features)
    diagnostic_paths: dict[str, Path] = {}
    for name, table in diagnostics.items():
        path = destination / f"sport_model_error_by_{name}.csv"
        table.to_csv(path, index=False)
        diagnostic_paths[name] = path
    promotion = _promotion_evidence(
        candidate_summary,
        baseline_summary,
        metrics,
        baseline_metrics,
        bootstrap,
    )
    season_comparison = metrics.merge(
        references.loc[references["model"].eq("sport_features")],
        on="season",
        suffixes=("_candidate", "_logistic"),
    )
    season_comparison["log_loss_delta"] = (
        season_comparison["log_loss_candidate"]
        - season_comparison["log_loss_logistic"]
    )

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
        "bootstrap": bootstrap,
        "promotion": promotion,
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
            "## Evidenza statistica",
            "",
            "Differenza Log Loss candidato − logistica: "
            f"{bootstrap['mean_log_loss_difference']:.4f} "
            f"(IC 95% {bootstrap['ci_low']:.4f}, {bootstrap['ci_high']:.4f}).",
            f"Stagioni vinte: {promotion['season_wins']}/"
            f"{len(metrics)}; richieste: {promotion['required_season_wins']}.",
            "Verdetto di promozione: "
            f"{'PROMOSSO' if promotion['promoted'] else 'NON PROMOSSO'}.",
            "",
            "## Stabilità stagionale",
            "",
            "| Stagione | Log Loss candidato | Log Loss logistica | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in season_comparison.to_dict("records"):
        report_lines.append(
            f"| {row['season']} | {row['log_loss_candidate']:.4f} | "
            f"{row['log_loss_logistic']:.4f} | "
            f"{row['log_loss_delta']:+.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Segmenti più difficili",
            "",
            "| Dimensione | Segmento | Match | Log Loss |",
            "|---|---|---:|---:|",
        ]
    )
    for name, table in diagnostics.items():
        if table.empty:
            continue
        group_column = str(table.columns[0])
        worst = table.sort_values("log_loss", ascending=False).iloc[0]
        report_lines.append(
            f"| {name} | {worst[group_column]} | "
            f"{int(worst['matches'])} | {worst['log_loss']:.4f} |"
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
        "reference_metrics": reference_metrics_path,
        "predictions": predictions_path,
        "comparison": comparison_path,
        "bootstrap": bootstrap_path,
        "report": report_path,
        "metadata": metadata_path,
        **{
            f"error_by_{name}": path
            for name, path in diagnostic_paths.items()
        },
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
