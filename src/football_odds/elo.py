from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EloSettings:
    """Parameters controlling one Elo rating pool."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 65.0
    season_regression: float = 0.25


class EloRatings:
    """Stateful Elo engine whose values are read before each update."""

    def __init__(self, settings: EloSettings | None = None) -> None:
        self.settings = settings or EloSettings()
        self._ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        """Return a team's current rating, initializing it when unseen."""
        return self._ratings.setdefault(team, self.settings.initial_rating)

    def expected_home(self, home_team: str, away_team: str) -> float:
        """Return expected home score after applying home advantage."""
        difference = (
            self.rating(home_team)
            + self.settings.home_advantage
            - self.rating(away_team)
        )
        return 1.0 / (1.0 + 10.0 ** (-difference / 400.0))

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> tuple[float, float]:
        """Update both ratings and return their new values."""
        home_rating = self.rating(home_team)
        away_rating = self.rating(away_team)
        expected = self.expected_home(home_team, away_team)
        actual = 1.0 if home_goals > away_goals else 0.0
        if home_goals == away_goals:
            actual = 0.5
        goal_multiplier = 1.0 + math.log1p(abs(home_goals - away_goals))
        change = self.settings.k_factor * goal_multiplier * (actual - expected)
        self._ratings[home_team] = home_rating + change
        self._ratings[away_team] = away_rating - change
        return self._ratings[home_team], self._ratings[away_team]

    def regress_to_mean(self) -> None:
        """Shrink every known rating between seasons."""
        weight = self.settings.season_regression
        base = self.settings.initial_rating
        self._ratings = {
            team: (1.0 - weight) * rating + weight * base
            for team, rating in self._ratings.items()
        }
