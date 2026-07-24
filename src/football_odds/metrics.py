from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

RESULT_INDEX = {"H": 0, "D": 1, "A": 2}
PROBABILITY_COLUMNS = ["p_home", "p_draw", "p_away"]


def encode_results(results: pd.Series) -> np.ndarray:
    indices = results.map(RESULT_INDEX)
    if indices.isna().any():
        raise ValueError("Sono presenti esiti diversi da H, D o A.")
    return np.eye(3, dtype=float)[indices.astype(int).to_numpy()]


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean(np.sum((probabilities - y_true) ** 2, axis=1)))


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("Non è possibile calcolare metriche su dati vuoti.")
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy()
    actual = frame["FTR"].map(RESULT_INDEX).astype(int).to_numpy()
    y_true = encode_results(frame["FTR"])
    return {
        "matches": len(frame),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == actual)),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1, 2])),
        "brier_score": multiclass_brier_score(y_true, probabilities),
        "average_margin": float(frame["margin"].mean()),
    }


def metrics_by_season(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in frame.groupby("Season", sort=True):
        rows.append({"season": season, **calculate_metrics(group)})
    return pd.DataFrame(rows)
