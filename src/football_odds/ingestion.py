from __future__ import annotations

from dataclasses import dataclass

from .config import LEAGUES
from .database import ResearchDatabase
from .providers.base import DataProvider


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
