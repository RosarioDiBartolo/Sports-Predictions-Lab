"""Pure odds normalization, calibration and market metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

matplotlib.use("Agg")

ODDS_CANDIDATES = (
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
)


def find_odds_columns(
    frame: pd.DataFrame,
    candidates: tuple[tuple[str, str, str], ...] = ODDS_CANDIDATES,
) -> tuple[str, str, str]:
    """Restituisce la prima terna 1-X-2 completa disponibile."""
    for columns in candidates:
        if all(column in frame.columns for column in columns):
            return columns
    raise ValueError("Nessuna terna completa di quote 1-X-2 disponibile.")


def remove_margin(odds: pd.DataFrame) -> pd.DataFrame:
    """Converte tre quote decimali in probabilità fair proporzionali."""
    numeric = odds.apply(pd.to_numeric, errors="coerce")
    raw = 1.0 / numeric
    overround = raw.sum(axis=1)
    fair = raw.div(overround, axis=0)
    fair.columns = ["p_home", "p_draw", "p_away"]
    fair["overround"] = overround
    fair["margin"] = overround - 1.0
    return fair


def prepare_matches(frame: pd.DataFrame) -> pd.DataFrame:
    home_col, draw_col, away_col = find_odds_columns(frame)
    identity = ["Date", "HomeTeam", "AwayTeam", "FTR", "Season", "League"]
    missing = [column for column in identity if column not in frame.columns]
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti: {missing}")

    selected = identity + [home_col, draw_col, away_col]
    clean = frame[selected].copy()
    odds_columns = [home_col, draw_col, away_col]
    clean[odds_columns] = clean[odds_columns].apply(pd.to_numeric, errors="coerce")
    valid = (
        clean["FTR"].isin(("H", "D", "A"))
        & clean[odds_columns].notna().all(axis=1)
        & (clean[odds_columns] > 1).all(axis=1)
    )
    clean = clean.loc[valid].reset_index(drop=True)
    probabilities = remove_margin(clean[odds_columns])
    clean = pd.concat([clean, probabilities], axis=1)
    clean["odds_source"] = "/".join(odds_columns)
    clean["Date"] = pd.to_datetime(clean["Date"], dayfirst=True, errors="coerce")
    return clean


def to_long_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("H", "Home", "p_home"),
        ("D", "Draw", "p_draw"),
        ("A", "Away", "p_away"),
    )
    parts = []
    for result, outcome, probability in definitions:
        parts.append(
            pd.DataFrame(
                {
                    "predicted_probability": frame[probability],
                    "occurred": (frame["FTR"] == result).astype(int),
                    "outcome": outcome,
                    "season": frame["Season"],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def calibration_table(
    long_frame: pd.DataFrame, bin_width: float = 0.05
) -> pd.DataFrame:
    bins = np.append(np.arange(0, 1, bin_width), 1.0)
    data = long_frame.copy()
    data["bin"] = pd.cut(
        data["predicted_probability"],
        bins=np.unique(bins),
        include_lowest=True,
    )
    table = (
        data.groupby("bin", observed=True)
        .agg(
            observations=("occurred", "size"),
            predicted_probability=("predicted_probability", "mean"),
            actual_frequency=("occurred", "mean"),
        )
        .reset_index()
    )
    table["calibration_error"] = (
        table["actual_frequency"] - table["predicted_probability"]
    )
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    total = table["observations"].sum()
    if total == 0:
        return float("nan")
    weighted = table["observations"] * table["calibration_error"].abs()
    return float(weighted.sum() / total)


def plot_calibration(table: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.plot([0, 1], [0, 1], "--", label="Calibrazione perfetta")
    axis.plot(
        table["predicted_probability"],
        table["actual_frequency"],
        marker="o",
        label="Quote bookmaker",
    )
    axis.set(
        xlabel="Probabilità implicita del bookmaker",
        ylabel="Frequenza reale dell'evento",
        title="Curva di calibrazione 1-X-2",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


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
