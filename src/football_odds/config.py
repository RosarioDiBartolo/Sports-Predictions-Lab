from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SEASONS = ("1819", "1920", "2021", "2122", "2223", "2324", "2425")
DEFAULT_LEAGUES = ("I1", "E0", "SP1", "D1", "F1")

LEAGUES = {
    "I1": {"name": "Serie A", "country": "Italy"},
    "E0": {"name": "Premier League", "country": "England"},
    "SP1": {"name": "La Liga", "country": "Spain"},
    "D1": {"name": "Bundesliga", "country": "Germany"},
    "F1": {"name": "Ligue 1", "country": "France"},
    "CL": {"name": "Champions League", "country": "Europe"},
    "EL": {"name": "Europa League", "country": "Europe"},
}

BOOKMAKER_ODDS_COLUMNS = {
    "Market Average": {
        "closing": ("AvgCH", "AvgCD", "AvgCA"),
        "opening": ("AvgH", "AvgD", "AvgA"),
    },
    "Bet365": {
        "closing": ("B365CH", "B365CD", "B365CA"),
        "opening": ("B365H", "B365D", "B365A"),
    },
    "Pinnacle": {
        "closing": ("PSCH", "PSCD", "PSCA"),
        "opening": ("PSH", "PSD", "PSA"),
    },
    "Maximum Odds": {
        "closing": ("MaxCH", "MaxCD", "MaxCA"),
        "opening": ("MaxH", "MaxD", "MaxA"),
    },
}

ODDS_RANGE_EDGES = (
    1.01,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
    1.60,
    1.70,
    1.80,
    1.90,
    2.00,
    2.50,
    3.00,
    4.00,
    5.00,
    float("inf"),
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Parametri dell'analisi, senza stato globale nascosto."""

    league: str = "I1"
    seasons: tuple[str, ...] = DEFAULT_SEASONS
    bin_width: float = 0.05
    project_dir: Path = field(default_factory=Path.cwd)
    database_name: str = "football_odds.sqlite3"

    @property
    def raw_dir(self) -> Path:
        return self.project_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_dir / "data" / "processed"

    @property
    def report_dir(self) -> Path:
        return self.project_dir / "reports"

    @property
    def database_path(self) -> Path:
        return self.project_dir / "data" / self.database_name

    def ensure_directories(self) -> None:
        for directory in (self.raw_dir, self.processed_dir, self.report_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.league.strip():
            raise ValueError("Il codice del campionato non può essere vuoto.")
        if not self.seasons:
            raise ValueError("Serve almeno una stagione.")
        if not 0 < self.bin_width <= 1:
            raise ValueError("bin_width deve essere compreso tra 0 e 1.")


@dataclass(frozen=True)
class ModelingConfig:
    """Central configuration for leakage-safe football features."""

    leagues: tuple[str, ...] = DEFAULT_LEAGUES
    seasons: tuple[str, ...] = DEFAULT_SEASONS
    project_dir: Path = field(default_factory=Path.cwd)
    rolling_windows: tuple[int, ...] = (5, 10)
    elo_initial_rating: float = 1500.0
    elo_k_factor: float = 20.0
    elo_home_advantage: float = 65.0
    elo_season_regression: float = 0.25
    team_aliases: dict[str, str] = field(default_factory=dict)

    @property
    def processed_dir(self) -> Path:
        return self.project_dir / "data" / "processed"

    @property
    def report_dir(self) -> Path:
        return self.project_dir / "reports" / "modeling"

    def ensure_directories(self) -> None:
        """Create output folders used by the modeling pipeline."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Reject configurations that could produce ambiguous features."""
        if not self.leagues:
            raise ValueError("Serve almeno un campionato.")
        if not self.seasons:
            raise ValueError("Serve almeno una stagione.")
        if not self.rolling_windows or any(
            window <= 0 for window in self.rolling_windows
        ):
            raise ValueError("Le finestre rolling devono essere positive.")
        if self.elo_k_factor <= 0:
            raise ValueError("elo_k_factor deve essere positivo.")
        if not 0 <= self.elo_season_regression <= 1:
            raise ValueError("elo_season_regression deve essere tra 0 e 1.")


@dataclass(frozen=True)
class BackfillConfig:
    """Configuration for the reproducible multi-league research backfill."""

    leagues: tuple[str, ...] = DEFAULT_LEAGUES
    seasons: tuple[str, ...] = DEFAULT_SEASONS
    project_dir: Path = field(default_factory=Path.cwd)
    database_name: str = "football_odds.sqlite3"

    @property
    def raw_dir(self) -> Path:
        return self.project_dir / "data" / "raw"

    @property
    def report_dir(self) -> Path:
        return self.project_dir / "reports" / "backfill"

    @property
    def database_path(self) -> Path:
        return self.project_dir / "data" / self.database_name

    def validate(self) -> None:
        unknown = set(self.leagues).difference(LEAGUES)
        if not self.leagues or unknown:
            raise ValueError(f"Campionati non supportati: {sorted(unknown)}")
        if not self.seasons:
            raise ValueError("Serve almeno una stagione.")
