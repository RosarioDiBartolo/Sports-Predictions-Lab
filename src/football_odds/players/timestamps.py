"""Timestamp normalization shared by leakage-safe player pipelines."""

from __future__ import annotations

import pandas as pd


def utc_instants(values: pd.Series) -> pd.Series:
    """Parse timestamp-like values as comparable UTC instants."""
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError("I kickoff devono essere valorizzati.")
    return parsed
