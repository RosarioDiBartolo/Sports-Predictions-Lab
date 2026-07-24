from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from .config import AnalysisConfig

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
