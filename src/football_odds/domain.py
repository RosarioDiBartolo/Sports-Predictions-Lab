from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

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


@dataclass(frozen=True)
class OddsRecord:
    """One normalized 1-X-2 market snapshot."""

    provider_match_id: str
    bookmaker: str
    market: str
    odds: dict[str, float]
    timestamp: datetime | None
    timing: OddsTiming
