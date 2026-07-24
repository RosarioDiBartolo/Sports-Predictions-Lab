"""Provider plugins implementing a shared ingestion contract."""

from .base import DataProvider
from .football_data import FootballDataProvider

__all__ = ["DataProvider", "FootballDataProvider"]
