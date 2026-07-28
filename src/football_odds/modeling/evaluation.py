"""Common out-of-sample probability evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

OUTCOMES = ("H", "D", "A")
PROBABILITY_COLUMNS = (
    "probability_home",
    "probability_draw",
    "probability_away",
)


def probability_metrics(
    results: pd.Series, probabilities: np.ndarray
) -> dict[str, float | int]:
    labels = pd.Categorical(results, categories=OUTCOMES).codes
    encoded = np.eye(3)[labels]
    cumulative_error = np.cumsum(probabilities, axis=1)[:, :-1] - np.cumsum(
        encoded, axis=1
    )[:, :-1]
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for bin_index in range(10):
        mask = bins == bin_index
        if mask.any():
            ece += mask.mean() * abs(
                confidence[mask].mean() - (predicted[mask] == labels[mask]).mean()
            )
    return {
        "matches": len(results),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1))),
        "rps": float(np.mean(np.sum(cumulative_error**2, axis=1) / 2.0)),
        "accuracy": float(np.mean(predicted == labels)),
        "ece": float(ece),
    }


def metrics_by_season(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (season, model), group in predictions.groupby(["season", "model"], sort=True):
        rows.append(
            {
                "season": season,
                "model": model,
                **probability_metrics(
                    group["result"], group[list(PROBABILITY_COLUMNS)].to_numpy(float)
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_log_loss_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    joined = candidate.merge(
        reference, on="match_id", suffixes=("_candidate", "_reference")
    )
    if joined.empty:
        return {"matches": 0, "verdict": "insufficient_data"}
    outcome_index = (
        joined["result_candidate"].map({"H": 0, "D": 1, "A": 2}).to_numpy(int)
    )
    candidate_p = joined[
        [f"{name}_candidate" for name in PROBABILITY_COLUMNS]
    ].to_numpy(float)
    reference_p = joined[
        [f"{name}_reference" for name in PROBABILITY_COLUMNS]
    ].to_numpy(float)
    row = np.arange(len(joined))
    differences = -np.log(np.clip(candidate_p[row, outcome_index], 1e-15, 1)) + np.log(
        np.clip(reference_p[row, outcome_index], 1e-15, 1)
    )
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            differences[rng.integers(0, len(differences), len(differences))].mean()
            for _ in range(samples)
        ]
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "matches": len(joined),
        "mean_log_loss_difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_candidate_better": float(np.mean(means < 0)),
        "verdict": (
            "candidate_better"
            if high < 0
            else "candidate_worse"
            if low > 0
            else "inconclusive"
        ),
    }
