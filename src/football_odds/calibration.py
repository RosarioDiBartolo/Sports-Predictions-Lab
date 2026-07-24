from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
