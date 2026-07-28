"""Minimal Dixon–Coles baseline required by the neural model."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IDENTITY_FEATURES = ("home_team", "away_team", "league")
EXCLUDED = {"match_id", "date", "season", "result", "home_goals", "away_goals"}


def _estimator(numeric: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric,
                        ),
                        (
                            "identity",
                            OneHotEncoder(handle_unknown="ignore"),
                            list(IDENTITY_FEATURES),
                        ),
                    ]
                ),
            ),
            ("model", PoissonRegressor(alpha=1.0, max_iter=500)),
        ]
    )


def _tau(home: int, away: int, home_rate: float, away_rate: float, rho: float) -> float:
    if (home, away) == (0, 0):
        return 1 - home_rate * away_rate * rho
    if (home, away) == (0, 1):
        return 1 + home_rate * rho
    if (home, away) == (1, 0):
        return 1 + away_rate * rho
    if (home, away) == (1, 1):
        return 1 - rho
    return 1.0


def score_probabilities(
    home_rate: float, away_rate: float, rho: float, maximum_goals: int = 10
) -> np.ndarray:
    goals = np.arange(maximum_goals + 1)
    def poisson(rate: float) -> np.ndarray:
        return np.exp(
            goals * np.log(max(rate, 1e-6))
            - rate
            - np.vectorize(lgamma)(goals + 1)
        )
    matrix = np.outer(poisson(home_rate), poisson(away_rate))
    for home, away in ((0, 0), (0, 1), (1, 0), (1, 1)):
        matrix[home, away] *= max(_tau(home, away, home_rate, away_rate, rho), 1e-9)
    result = np.asarray(
        [np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()]
    )
    return result / result.sum()


@dataclass
class DixonColes:
    home: Pipeline
    away: Pipeline
    rho: float

    def rates(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.clip(self.home.predict(frame), 1e-6, 8.0),
            np.clip(self.away.predict(frame), 1e-6, 8.0),
        )


def fit_dixon_coles(frame: pd.DataFrame) -> DixonColes:
    required = {*IDENTITY_FEATURES, "home_goals", "away_goals", "result"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Dixon-Coles columns: {sorted(missing)}")
    numeric = [
        name
        for name in frame.select_dtypes(include=[np.number]).columns
        if name not in EXCLUDED and "odds" not in name and "probability" not in name
    ]
    home = _estimator(numeric)
    away = _estimator(numeric)
    home.fit(frame, frame["home_goals"])
    away.fit(frame, frame["away_goals"])
    home_rate, away_rate = (
        np.clip(home.predict(frame), 1e-6, 8),
        np.clip(away.predict(frame), 1e-6, 8),
    )
    low = frame["home_goals"].isin((0, 1)) & frame["away_goals"].isin((0, 1))
    candidates = np.linspace(-0.2, 0.2, 81)
    rho = 0.0
    if low.any():
        likelihood = [
            sum(
                np.log(max(_tau(int(h), int(a), hr, ar, float(candidate)), 1e-9))
                for h, a, hr, ar in zip(
                    frame.loc[low, "home_goals"],
                    frame.loc[low, "away_goals"],
                    home_rate[low],
                    away_rate[low],
                    strict=True,
                )
            )
            for candidate in candidates
        ]
        rho = float(candidates[int(np.argmax(likelihood))])
    return DixonColes(home, away, rho)
