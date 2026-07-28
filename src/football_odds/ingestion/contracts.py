"""Acquisition, provider normalization and canonical ingestion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import pandas as pd
import requests

from ..core.config import BOOKMAKER_ODDS_COLUMNS, LEAGUES, AnalysisConfig

if TYPE_CHECKING:
    from ..data.repository import ResearchDatabase

OddsTiming = Literal["opening", "closing", "snapshot"]


@dataclass(frozen=True)
class MatchRecord:
    """Provider-neutral representation of one football match."""

    provider_match_id: str
    date: datetime
    season: str
    league_code: str
    home_team: str
    away_team: str
    home_goals: int | None = None
    away_goals: int | None = None
    result: str | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    home_yellow_cards: int | None = None
    away_yellow_cards: int | None = None
    home_red_cards: int | None = None
    away_red_cards: int | None = None


@dataclass(frozen=True)
class OddsRecord:
    """One normalized 1-X-2 market snapshot."""

    provider_match_id: str
    bookmaker: str
    market: str
    odds: dict[str, float]
    timestamp: datetime | None
    timing: OddsTiming


BASE_URL = "https://www.football-data.co.uk/mmz4281"


def season_url(season: str, league: str) -> str:
    return f"{BASE_URL}/{season}/{league}.csv"


def download_season(
    season: str,
    league: str,
    destination: Path,
    *,
    refresh: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """Scarica una stagione oppure legge la copia locale già disponibile."""
    if destination.exists() and not refresh:
        return pd.read_csv(destination)

    response = requests.get(
        season_url(season, league),
        timeout=timeout,
        headers={"User-Agent": "football-odds-lab/0.1"},
    )
    response.raise_for_status()
    frame = pd.read_csv(BytesIO(response.content))
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def load_all_seasons(config: AnalysisConfig, *, refresh: bool = False) -> pd.DataFrame:
    config.validate()
    config.ensure_directories()
    frames: list[pd.DataFrame] = []

    for season in config.seasons:
        destination = config.raw_dir / f"{config.league}_{season}.csv"
        frame = download_season(
            season, config.league, destination, refresh=refresh
        ).copy()
        frame["Season"] = season
        frame["League"] = config.league
        frames.append(frame)

    if not frames:
        raise ValueError("Nessuna stagione da caricare.")
    return pd.concat(frames, ignore_index=True, sort=False)


class DataProvider(Protocol):
    """Contract implemented by every external data source plugin."""

    @property
    def name(self) -> str:
        """Stable provider name used by the mapping table."""
        ...

    def matches(self) -> list[MatchRecord]:
        """Return validated, provider-neutral match records."""
        ...

    def odds(self) -> list[OddsRecord]:
        """Return normalized market snapshots."""
        ...


class FootballDataProvider:
    """Adapter from Football-Data CSV columns to domain records."""

    name = "Football-Data.co.uk"

    def __init__(
        self,
        frame: pd.DataFrame,
        bookmaker_columns: dict[
            str, dict[str, tuple[str, str, str]]
        ] = BOOKMAKER_ODDS_COLUMNS,
    ) -> None:
        self.frame = frame.copy()
        self.bookmaker_columns = bookmaker_columns
        self._validate()

    def _validate(self) -> None:
        required = {"Date", "HomeTeam", "AwayTeam", "Season", "League"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Colonne Football-Data mancanti: {sorted(missing)}")

    @staticmethod
    def _date(value: object, time_value: object | None = None) -> datetime:
        combined = str(value)
        if time_value is not None and not pd.isna(time_value):
            combined = f"{combined} {time_value}"
        parsed = pd.to_datetime(combined, dayfirst=True, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Data partita non valida: {value}")
        return parsed.to_pydatetime()

    @classmethod
    def _external_id(cls, row: pd.Series) -> str:
        identity = "|".join(
            (
                str(row["League"]),
                str(row["Season"]),
                cls._date(row["Date"]).date().isoformat(),
                str(row["HomeTeam"]),
                str(row["AwayTeam"]),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    def matches(self) -> list[MatchRecord]:
        """Convert every valid row into a match record."""
        records = []
        for _, row in self.frame.iterrows():
            records.append(
                MatchRecord(
                    provider_match_id=self._external_id(row),
                    date=self._date(row["Date"], row.get("Time")),
                    season=str(row["Season"]),
                    league_code=str(row["League"]),
                    home_team=str(row["HomeTeam"]),
                    away_team=str(row["AwayTeam"]),
                    home_goals=self._optional_int(row.get("FTHG")),
                    away_goals=self._optional_int(row.get("FTAG")),
                    result=self._optional_result(row.get("FTR")),
                    home_shots=self._optional_int(row.get("HS")),
                    away_shots=self._optional_int(row.get("AS")),
                    home_shots_on_target=self._optional_int(row.get("HST")),
                    away_shots_on_target=self._optional_int(row.get("AST")),
                    home_corners=self._optional_int(row.get("HC")),
                    away_corners=self._optional_int(row.get("AC")),
                    home_yellow_cards=self._optional_int(row.get("HY")),
                    away_yellow_cards=self._optional_int(row.get("AY")),
                    home_red_cards=self._optional_int(row.get("HR")),
                    away_red_cards=self._optional_int(row.get("AR")),
                )
            )
        return records

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return None if pd.isna(value) else int(value)

    @staticmethod
    def _optional_result(value: object) -> str | None:
        return str(value) if value in {"H", "D", "A"} else None

    def odds(self) -> list[OddsRecord]:
        """Extract every available opening and closing 1-X-2 snapshot."""
        records = []
        for _, row in self.frame.iterrows():
            provider_match_id = self._external_id(row)
            for bookmaker, timings in self.bookmaker_columns.items():
                for timing, columns in timings.items():
                    if not all(column in self.frame.columns for column in columns):
                        continue
                    values = pd.to_numeric(row[list(columns)], errors="coerce")
                    if values.isna().any() or (values <= 1).any():
                        continue
                    records.append(
                        OddsRecord(
                            provider_match_id=provider_match_id,
                            bookmaker=bookmaker,
                            market="1X2",
                            odds=dict(
                                zip(
                                    ("H", "D", "A"),
                                    map(float, values),
                                    strict=True,
                                )
                            ),
                            # Football-Data labels opening/closing but does not
                            # publish the observation instant for these columns.
                            timestamp=None,
                            timing=timing,  # type: ignore[arg-type]
                        )
                    )
        return records


@dataclass(frozen=True)
class IngestionSummary:
    """Counts produced by one provider ingestion."""

    matches: int
    odds_selections: int


class IngestionPipeline:
    """Validation → normalization → master database pipeline."""

    def __init__(
        self,
        database: ResearchDatabase,
        leagues: dict[str, dict[str, str]] = LEAGUES,
    ) -> None:
        self.database = database
        self.leagues = leagues

    def run(self, provider: DataProvider) -> IngestionSummary:
        """Ingest a provider through its common interface."""
        matches = provider.matches()
        snapshots = provider.odds()
        with self.database.batch():
            self.database.initialize()
            for match in matches:
                metadata = self.leagues.get(
                    match.league_code,
                    {"name": match.league_code, "country": "Unknown"},
                )
                self.database.upsert_match(
                    provider.name,
                    match,
                    league_name=metadata["name"],
                    country=metadata["country"],
                )
            selections = sum(
                self.database.add_odds(provider.name, snapshot)
                for snapshot in snapshots
            )
        return IngestionSummary(matches=len(matches), odds_selections=selections)
