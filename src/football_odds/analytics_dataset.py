from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .database import ResearchDatabase

ANALYTICS_QUERY = """
SELECT
    m.match_id,
    b.bookmaker_name AS bookmaker,
    p.provider_name AS provider,
    o.market,
    o.selection,
    o.decimal_odds AS odds,
    o.implied_probability,
    r.result,
    o.margin,
    o.timestamp,
    o.opening_or_closing,
    m.season,
    l.league_name AS league,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    r.home_goals,
    r.away_goals
FROM odds o
JOIN matches m ON m.match_id = o.match_id
JOIN bookmakers b ON b.bookmaker_id = o.bookmaker_id
JOIN providers p ON p.provider_id = o.provider_id
JOIN leagues l ON l.league_id = m.league_id
JOIN teams ht ON ht.team_id = m.home_team_id
JOIN teams at ON at.team_id = m.away_team_id
LEFT JOIN match_results r ON r.match_id = m.match_id
"""


def build_analytics_dataset(
    database: ResearchDatabase | Path | str,
    *,
    bin_width: float = 0.05,
) -> pd.DataFrame:
    """Build one analysis row per bookmaker selection and snapshot."""
    repository = (
        database
        if isinstance(database, ResearchDatabase)
        else ResearchDatabase(database)
    )
    repository.initialize()
    with repository.connect() as connection:
        frame = pd.read_sql_query(ANALYTICS_QUERY, connection)
    fixed = frame["opening_or_closing"].isin(["opening", "closing"])
    canonical = frame.loc[fixed].drop_duplicates(
        [
            "match_id",
            "bookmaker",
            "market",
            "selection",
            "opening_or_closing",
        ],
        keep="last",
    )
    frame = pd.concat([canonical, frame.loc[~fixed]], ignore_index=True)
    if frame.empty:
        return frame.assign(
            prediction_correct=pd.Series(dtype=bool),
            favorite=pd.Series(dtype=bool),
            favorite_won=pd.Series(dtype=bool),
            calibration_bin=pd.Series(dtype=str),
            logloss_contribution=pd.Series(dtype=float),
            brier_contribution=pd.Series(dtype=float),
            roi=pd.Series(dtype=float),
        )

    frame["prediction_correct"] = frame["selection"] == frame["result"]
    snapshot_keys = [
        "match_id",
        "bookmaker",
        "market",
        "timestamp",
        "opening_or_closing",
    ]
    maximum = frame.groupby(snapshot_keys, dropna=False)[
        "implied_probability"
    ].transform("max")
    frame["favorite"] = frame["implied_probability"].eq(maximum)
    frame["favorite_won"] = frame["favorite"] & frame["prediction_correct"]
    edges = np.append(np.arange(0, 1, bin_width), 1.0)
    frame["calibration_bin"] = pd.cut(
        frame["implied_probability"],
        bins=np.unique(edges),
        include_lowest=True,
    ).astype(str)
    actual = frame["prediction_correct"].astype(float)
    probability = frame["implied_probability"].clip(1e-15, 1 - 1e-15)
    true_probability = probability.where(frame["prediction_correct"])
    frame["logloss_contribution"] = -np.log(
        true_probability.groupby(
            [frame[key] for key in snapshot_keys], dropna=False
        ).transform("max")
    )
    squared_error = (probability - actual) ** 2
    frame["brier_contribution"] = squared_error.groupby(
        [frame[key] for key in snapshot_keys], dropna=False
    ).transform("sum")
    frame["roi"] = np.where(frame["prediction_correct"], frame["odds"] - 1.0, -1.0)
    return frame


def save_analytics_dataset(
    config: AnalysisConfig,
    database: ResearchDatabase | None = None,
) -> pd.DataFrame:
    """Build and persist the replaceable analytics layer."""
    frame = build_analytics_dataset(
        database or ResearchDatabase(config.database_path),
        bin_width=config.bin_width,
    )
    config.ensure_directories()
    frame.to_csv(config.processed_dir / "analytics_dataset.csv", index=False)
    return frame
