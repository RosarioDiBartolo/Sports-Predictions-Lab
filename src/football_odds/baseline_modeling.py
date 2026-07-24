from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUTCOMES = ("H", "D", "A")
BASE_SPORT_FEATURES = (
    "home_elo",
    "away_elo",
    "elo_difference",
    "elo_expected_home",
    "home_matches_played",
    "away_matches_played",
    "home_rest_days",
    "away_rest_days",
)


def _sport_feature_allowlist(columns: pd.Index) -> list[str]:
    """Return only reviewed pre-match features, never arbitrary numeric fields."""
    allowed = list(BASE_SPORT_FEATURES)
    prefixes = (
        "home_points_",
        "away_points_",
        "home_goals_for_",
        "away_goals_for_",
        "home_goals_against_",
        "away_goals_against_",
        "home_opponent_elo_",
        "away_opponent_elo_",
    )
    for column in columns:
        if not isinstance(column, str):
            continue
        prefix = next((value for value in prefixes if column.startswith(value)), None)
        if prefix is not None and column.removeprefix(prefix).isdigit():
            allowed.append(column)
    return [column for column in allowed if column in columns]


@dataclass
class BaselineResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    outputs: dict[str, Path]


def _probability_metrics(
    actual: pd.Series, probabilities: np.ndarray
) -> dict[str, float]:
    indices = actual.map(
        {label: index for index, label in enumerate(OUTCOMES)}
    ).to_numpy()
    clipped = np.clip(probabilities, 1e-15, 1.0)
    one_hot = np.eye(len(OUTCOMES))[indices]
    confidence = clipped.max(axis=1)
    correct = clipped.argmax(axis=1) == indices
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(
        np.mean(bins == bucket)
        * abs(
            float(correct[bins == bucket].mean())
            - float(confidence[bins == bucket].mean())
        )
        for bucket in np.unique(bins)
    )
    return {
        "matches": float(len(actual)),
        "log_loss": float(-np.log(clipped[np.arange(len(indices)), indices]).mean()),
        "brier": float(np.square(clipped - one_hot).sum(axis=1).mean()),
        "accuracy": float(correct.mean()),
        "ece": float(ece),
    }


def _league_frequencies(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    global_counts = train["result"].value_counts().reindex(OUTCOMES, fill_value=0) + 1
    global_probabilities = global_counts.to_numpy(dtype=float) / global_counts.sum()
    by_league = (
        train.groupby(["league", "result"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=OUTCOMES, fill_value=0)
    )
    output = []
    for league in test["league"]:
        if league not in by_league.index:
            output.append(global_probabilities)
            continue
        counts = by_league.loc[league].to_numpy(dtype=float) + global_probabilities * 20
        output.append(counts / counts.sum())
    return np.asarray(output)


def _logistic_probabilities(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
) -> np.ndarray:
    transformers = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("league", OneHotEncoder(handle_unknown="ignore"), ["league"]),
        ]
    )
    model = Pipeline(
        [
            ("features", transformers),
            ("model", LogisticRegression(max_iter=2000, C=0.5)),
        ]
    )
    model.fit(train[numeric_features + ["league"]], train["result"])
    raw = model.predict_proba(test[numeric_features + ["league"]])
    classes = list(model.named_steps["model"].classes_)
    return raw[:, [classes.index(outcome) for outcome in OUTCOMES]]


def walk_forward_baselines(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baselines on each season using strictly earlier seasons."""
    required = {"season", "league", "result", "elo_difference"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Colonne baseline mancanti: {sorted(missing)}")
    data = features.copy()
    data["season"] = data["season"].astype(str).str.zfill(4)
    seasons = sorted(data["season"].unique())
    full_numeric = _sport_feature_allowlist(data.columns)
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for season in seasons[1:]:
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        if train.empty or test.empty:
            continue
        candidates = {"historical_frequency": _league_frequencies(train, test)}
        if set(train["result"]) == set(OUTCOMES):
            candidates["elo"] = _logistic_probabilities(train, test, ["elo_difference"])
            candidates["sport_features"] = _logistic_probabilities(
                train, test, full_numeric
            )
        market_columns = [
            "market_home_probability",
            "market_draw_probability",
            "market_away_probability",
        ]
        if set(market_columns).issubset(test.columns):
            market = test[market_columns].to_numpy(dtype=float)
            valid = np.isfinite(market).all(axis=1)
            if valid.any():
                candidates["market_closing"] = market
        for name, probabilities in candidates.items():
            valid = np.isfinite(probabilities).all(axis=1)
            match_metrics = _probability_metrics(
                test.loc[valid, "result"], probabilities[valid]
            )
            rows.append({"season": season, "model": name, **match_metrics})
            predictions = test.loc[
                valid, ["match_id", "season", "league", "result"]
            ].copy()
            predictions["model"] = name
            prediction_columns = [
                "probability_home",
                "probability_draw",
                "probability_away",
            ]
            predictions[prediction_columns] = probabilities[valid]
            prediction_rows.append(predictions)
    metric_columns = [
        "season",
        "model",
        "matches",
        "log_loss",
        "brier",
        "accuracy",
        "ece",
    ]
    prediction_columns = [
        "match_id",
        "season",
        "league",
        "result",
        "model",
        "probability_home",
        "probability_draw",
        "probability_away",
    ]
    metrics = pd.DataFrame(rows, columns=metric_columns)
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame(columns=prediction_columns)
    )
    return metrics, predictions


def export_baseline_report(
    features: pd.DataFrame,
    destination: Path,
) -> BaselineResult:
    destination.mkdir(parents=True, exist_ok=True)
    metrics, predictions = walk_forward_baselines(features)
    metrics_path = destination / "baseline_metrics_by_season.csv"
    predictions_path = destination / "baseline_predictions.csv"
    report_path = destination / "BASELINE_REPORT.md"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    aggregate = pd.DataFrame()
    if not metrics.empty:
        aggregate = (
            metrics.groupby("model")
            .apply(
                lambda group: pd.Series(
                    {
                        metric: np.average(group[metric], weights=group["matches"])
                        for metric in ("log_loss", "brier", "accuracy", "ece")
                    }
                ),
                include_groups=False,
            )
            .sort_values("log_loss")
        )
    table_lines = [
        "| Modello | Log Loss | Brier | Accuracy | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, row in aggregate.iterrows():
        table_lines.append(
            f"| {model} | {row['log_loss']:.4f} | {row['brier']:.4f} | "
            f"{row['accuracy']:.2%} | {row['ece']:.4f} |"
        )
    report_path.write_text(
        "# Baseline walk-forward\n\n"
        "Ogni stagione è valutata usando esclusivamente stagioni precedenti. "
        "Le quote di mercato sono closing medie private del margine.\n\n"
        + "\n".join(table_lines)
        + "\n",
        encoding="utf-8",
    )
    return BaselineResult(
        metrics,
        predictions,
        {
            "metrics": metrics_path,
            "predictions": predictions_path,
            "report": report_path,
        },
    )
