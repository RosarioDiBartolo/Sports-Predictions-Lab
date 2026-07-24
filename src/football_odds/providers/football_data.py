from __future__ import annotations

import hashlib
from datetime import datetime

import pandas as pd

from football_odds.config import BOOKMAKER_ODDS_COLUMNS
from football_odds.domain import MatchRecord, OddsRecord


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
