"""Provider-neutral records persisted by the canonical data domain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

OddsTiming = Literal["opening", "closing", "snapshot"]


@dataclass(frozen=True)
class MatchRecord:
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
    provider_match_id: str
    bookmaker: str
    market: str
    odds: dict[str, float]
    timestamp: datetime | None
    timing: OddsTiming
