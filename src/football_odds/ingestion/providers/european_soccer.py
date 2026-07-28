"""Import the Open Database licensed European Soccer Database."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...data.contracts import MatchRecord
from ...data.repository import ResearchDatabase
from ...players.observations import ingest_fixture_lineups

PROVIDER = "European Soccer Database"
PLAYER_NAMESPACE = uuid.UUID("60def30b-983b-4fe9-8540-f595d90dfeb1")
DEFAULT_SEASONS = (
    "2010/2011",
    "2011/2012",
    "2012/2013",
    "2013/2014",
    "2014/2015",
    "2015/2016",
)
COUNTRIES = {
    "England": ("E0", "Premier League"),
    "France": ("F1", "Ligue 1"),
    "Germany": ("D1", "Bundesliga"),
    "Italy": ("I1", "Serie A"),
    "Spain": ("SP1", "La Liga"),
}


@dataclass(frozen=True)
class EuropeanSoccerImportResult:
    matches_seen: int
    matches_imported: int
    already_imported: int
    incomplete_lineups: int
    player_observations: int


def _season_code(value: str) -> str:
    start, end = value.split("/")
    return f"{start[-2:]}{end[-2:]}"


def _position(y_coordinate: Any) -> str:
    y = float(y_coordinate)
    if y <= 1:
        return "G"
    if y <= 5:
        return "D"
    if y <= 8:
        return "M"
    return "F"


def _lineup(
    row: sqlite3.Row,
    *,
    side: str,
    team_id: int,
    team_name: str,
    player_names: dict[int, str],
) -> dict[str, Any]:
    players = []
    for number in range(1, 12):
        player_id = int(row[f"{side}_player_{number}"])
        x = row[f"{side}_player_X{number}"]
        y = row[f"{side}_player_Y{number}"]
        players.append(
            {
                "player": {
                    "id": player_id,
                    "name": player_names.get(player_id, f"Player {player_id}"),
                    "pos": _position(y),
                    "grid": f"{int(float(y))}:{int(float(x))}",
                }
            }
        )
    return {
        "team": {"id": team_id, "name": team_name},
        "startXI": players,
        "substitutes": [],
    }


def import_european_soccer_database(
    project_dir: Path,
    *,
    source_path: Path | None = None,
    database_path: Path | None = None,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
) -> EuropeanSoccerImportResult:
    """Import complete starting elevens with roles derived from tactical Y."""
    source = (
        source_path
        or project_dir
        / "data"
        / "raw"
        / "external"
        / "european-soccer-database"
        / "database.sqlite"
    )
    if not source.exists():
        raise FileNotFoundError(f"European Soccer Database non trovato: {source}")
    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    source_connection = sqlite3.connect(source)
    source_connection.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in seasons)
    rows = source_connection.execute(
        f"""
        SELECT m.*, c.name AS country_name,
               ht.team_long_name AS home_team_name,
               at.team_long_name AS away_team_name
        FROM Match m
        JOIN Country c ON c.id=m.country_id
        JOIN Team ht ON ht.team_api_id=m.home_team_api_id
        JOIN Team at ON at.team_api_id=m.away_team_api_id
        WHERE c.name IN ({",".join("?" for _ in COUNTRIES)})
          AND m.season IN ({placeholders})
        ORDER BY m.date, m.match_api_id
        """,
        (*COUNTRIES, *seasons),
    ).fetchall()
    player_names = {
        int(row["player_api_id"]): str(row["player_name"])
        for row in source_connection.execute(
            "SELECT player_api_id, player_name FROM Player"
        )
    }
    source_connection.close()

    imported = already = incomplete = observations = 0
    observed_at = datetime.now(timezone.utc).isoformat()
    with database.batch() as connection:
        provider_id = database._lookup_id(  # noqa: SLF001
            connection, "providers", "provider_id", "provider_name", PROVIDER
        )
        existing = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT provider_match_id
                FROM provider_match_mapping
                WHERE provider_id=?
                """,
                (provider_id,),
            )
        }
        for row in rows:
            fixture_id = str(row["match_api_id"])
            if fixture_id in existing:
                already += 1
                continue
            required = [
                row[f"{side}_player_{number}"]
                for side in ("home", "away")
                for number in range(1, 12)
            ] + [
                row[f"{side}_player_Y{number}"]
                for side in ("home", "away")
                for number in range(1, 12)
            ]
            if any(value is None for value in required):
                incomplete += 1
                continue
            country = str(row["country_name"])
            league_code, league_name = COUNTRIES[country]
            home_name = str(row["home_team_name"])
            away_name = str(row["away_team_name"])
            home_goals = int(row["home_team_goal"])
            away_goals = int(row["away_team_goal"])
            result = (
                "H"
                if home_goals > away_goals
                else "A"
                if away_goals > home_goals
                else "D"
            )
            database.upsert_match(
                PROVIDER,
                MatchRecord(
                    provider_match_id=fixture_id,
                    date=datetime.fromisoformat(str(row["date"])),
                    season=_season_code(str(row["season"])),
                    league_code=league_code,
                    home_team=home_name,
                    away_team=away_name,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    result=result,
                ),
                league_name,
                country,
            )
            for external_id, name in (
                (int(row["home_team_api_id"]), home_name),
                (int(row["away_team_api_id"]), away_name),
            ):
                internal_team_id = int(
                    connection.execute(
                        "SELECT team_id FROM teams WHERE team_name=?", (name,)
                    ).fetchone()[0]
                )
                connection.execute(
                    """
                    INSERT INTO provider_team_mapping VALUES (
                        ?, ?, ?, ?, 'normalized_exact', ?
                    )
                    ON CONFLICT(provider_id, provider_team_id) DO NOTHING
                    """,
                    (
                        provider_id,
                        str(external_id),
                        internal_team_id,
                        name,
                        observed_at,
                    ),
                )
            ingest_fixture_lineups(
                database,
                provider_fixture_id=fixture_id,
                lineups=[
                    _lineup(
                        row,
                        side="home",
                        team_id=int(row["home_team_api_id"]),
                        team_name=home_name,
                        player_names=player_names,
                    ),
                    _lineup(
                        row,
                        side="away",
                        team_id=int(row["away_team_api_id"]),
                        team_name=away_name,
                        player_names=player_names,
                    ),
                ],
                observed_at=observed_at,
                provider_name=PROVIDER,
                player_namespace=PLAYER_NAMESPACE,
            )
            imported += 1
            observations += 22
    return EuropeanSoccerImportResult(
        matches_seen=len(rows),
        matches_imported=imported,
        already_imported=already,
        incomplete_lineups=incomplete,
        player_observations=observations,
    )
