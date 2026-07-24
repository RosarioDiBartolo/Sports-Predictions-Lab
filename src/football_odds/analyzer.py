from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ODDS_RANGE_EDGES


def analyze_predictions(frame: pd.DataFrame) -> dict[str, float | int]:
    """Calculate binary selection-level research metrics."""
    if frame.empty:
        raise ValueError("Il dataset analitico è vuoto.")
    actual = frame["prediction_correct"].astype(float)
    probability = frame["implied_probability"]
    calibration = (
        frame.groupby("calibration_bin", observed=True)
        .agg(
            observations=("prediction_correct", "size"),
            predicted=("implied_probability", "mean"),
            actual=("prediction_correct", "mean"),
        )
        .reset_index()
    )
    absolute_error = (calibration["predicted"] - calibration["actual"]).abs()
    ece = np.average(absolute_error, weights=calibration["observations"])
    favorite_rows = frame[frame["favorite"]]
    return {
        "predictions": len(frame),
        "accuracy": float(favorite_rows["favorite_won"].mean()),
        "log_loss": float(frame["logloss_contribution"].mean()),
        "brier_score": float(frame["brier_contribution"].mean()),
        "calibration_error": float(actual.mean() - probability.mean()),
        "expected_calibration_error": float(ece),
        "average_overround": float(frame["margin"].mean()),
        "sharpness": float(probability.var(ddof=0)),
    }


def compare_bookmakers(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank bookmakers on the common set of available predictions."""
    bookmakers = frame["bookmaker"].nunique()
    if bookmakers > 1:
        coverage = frame.groupby(
            ["match_id", "market", "selection", "opening_or_closing"]
        )["bookmaker"].nunique()
        shared = coverage[coverage == bookmakers].index
        indexed = frame.set_index(
            ["match_id", "market", "selection", "opening_or_closing"]
        )
        common = indexed[indexed.index.isin(shared)].reset_index()
        if not common.empty:
            frame = common
    rows = []
    for bookmaker, group in frame.groupby("bookmaker", sort=True):
        rows.append({"bookmaker": bookmaker, **analyze_predictions(group)})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["rank"] = result["log_loss"].rank(method="min").astype(int)
    return result.sort_values(["rank", "bookmaker"]).reset_index(drop=True)


def analyze_odds_ranges(
    frame: pd.DataFrame,
    edges: tuple[float, ...] = ODDS_RANGE_EDGES,
) -> pd.DataFrame:
    """Aggregate calibration and flat-stake ROI by decimal-odds range."""
    data = frame.copy()
    labels = [
        f"{left:.2f}-{right:.2f}" if np.isfinite(right) else f"{left:.2f}+"
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    ]
    data["odds_range"] = pd.cut(
        data["odds"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    result = (
        data.groupby("odds_range", observed=True)
        .agg(
            predictions=("prediction_correct", "size"),
            implied_probability=("implied_probability", "mean"),
            actual_frequency=("prediction_correct", "mean"),
            roi=("roi", "mean"),
        )
        .reset_index()
    )
    result["calibration_error"] = (
        result["actual_frequency"] - result["implied_probability"]
    )
    return result


def compare_leagues(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the same metrics for every available league."""
    return pd.DataFrame(
        [
            {"league": league, **analyze_predictions(group)}
            for league, group in frame.groupby("league", sort=True)
        ]
    )


def compare_opening_closing(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare timing metrics and average probability movement."""
    rows = []
    key = ["match_id", "bookmaker", "market", "selection"]
    pivot = frame.pivot_table(
        index=key,
        columns="opening_or_closing",
        values="implied_probability",
        aggfunc="last",
    )
    movement = (
        (pivot["closing"] - pivot["opening"]).abs().mean()
        if {"opening", "closing"}.issubset(pivot.columns)
        else float("nan")
    )
    for timing, group in frame[
        frame["opening_or_closing"].isin(["opening", "closing"])
    ].groupby("opening_or_closing"):
        rows.append(
            {
                "timing": timing,
                **analyze_predictions(group),
                "mean_absolute_probability_movement": float(movement),
            }
        )
    return pd.DataFrame(rows)
