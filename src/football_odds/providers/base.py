from __future__ import annotations

from typing import Protocol

from football_odds.domain import MatchRecord, OddsRecord


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
