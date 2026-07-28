"""Importer for the explicitly licensed StatsBomb Open Data repository."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ...data.repository import ResearchDatabase
from ...players.observations import ingest_fixture_lineups
from ...players.reconciliation import normalize_team_name

PROVIDER = "StatsBomb Open Data"
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
PLAYER_NAMESPACE = uuid.UUID("7a2eb233-2dd3-45bd-9b6c-b7be18612756")
TARGETS = {
    ("SP1", "1819"): (11, 4),
    ("SP1", "1920"): (11, 42),
    ("SP1", "2021"): (11, 90),
    ("D1", "2324"): (9, 281),
    ("F1", "2122"): (7, 108),
    ("F1", "2223"): (7, 235),
}
TEAM_ALIASES = {
    "athletic club": "ath bilbao",
    "atletico madrid": "ath madrid",
    "bayer leverkusen": "leverkusen",
    "borussia dortmund": "dortmund",
    "borussia monchengladbach": "m gladbach",
    "celta vigo": "celta",
    "clermont foot": "clermont",
    "darmstadt 98": "darmstadt",
    "deportivo alaves": "alaves",
    "eintracht frankfurt": "ein frankfurt",
    "espanyol": "espanol",
    "fsv mainz 05": "mainz",
    "levante ud": "levante",
    "ogc nice": "nice",
    "olympique de marseille": "marseille",
    "paris saint germain": "paris sg",
    "rayo vallecano": "vallecano",
    "real betis": "betis",
    "real sociedad": "sociedad",
    "real valladolid": "valladolid",
    "saint etienne": "st etienne",
    "stade brestois": "brest",
    "stade de reims": "reims",
    "vfb stuttgart": "stuttgart",
}


@dataclass(frozen=True)
class StatsBombImportResult:
    matches_seen: int
    matches_mapped: int
    matches_imported: int
    unresolved: int
    cache_hits: int


def _json(
    project_dir: Path,
    relative: str,
    *,
    request: Any,
) -> tuple[Any, bool]:
    path = project_dir / "data" / "cache" / "statsbomb_open" / relative
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    response = request(f"{BASE_URL}/{relative}", timeout=30)
    response.raise_for_status()
    payload = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return payload, False


def _canonical_index(
    database: ResearchDatabase,
    league: str,
    season: str,
) -> dict[tuple[str, str, str], tuple[str, int, int]]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT
                m.match_id, date(m.date) AS match_date,
                m.home_team_id, ht.team_name AS home_team,
                m.away_team_id, at.team_name AS away_team
            FROM matches m
            JOIN leagues l ON l.league_id=m.league_id
            JOIN teams ht ON ht.team_id=m.home_team_id
            JOIN teams at ON at.team_id=m.away_team_id
            WHERE l.league_code=? AND m.season=?
            """,
            (league, season),
        ).fetchall()
    return {
        (
            str(row["match_date"]),
            normalize_team_name(str(row["home_team"])),
            normalize_team_name(str(row["away_team"])),
        ): (
            str(row["match_id"]),
            int(row["home_team_id"]),
            int(row["away_team_id"]),
        )
        for row in rows
    }


def _team(match: dict[str, Any], side: str) -> tuple[str, int, str]:
    value = match[f"{side}_team"]
    normalized = normalize_team_name(str(value[f"{side}_team_name"]))
    return (
        TEAM_ALIASES.get(normalized, normalized),
        int(value[f"{side}_team_id"]),
        str(value[f"{side}_team_name"]),
    )


def _convert_lineup(team: dict[str, Any]) -> dict[str, Any]:
    starters: list[dict[str, Any]] = []
    substitutes: list[dict[str, Any]] = []
    for item in team.get("lineup") or []:
        positions = list(item.get("positions") or [])
        starting = next(
            (
                position
                for position in positions
                if position.get("start_reason") == "Starting XI"
            ),
            None,
        )
        observed = starting or (positions[0] if positions else {})
        player = {
            "player": {
                "id": item["player_id"],
                "name": item["player_name"],
                "number": item.get("jersey_number"),
                "pos": observed.get("position"),
            }
        }
        (starters if starting else substitutes).append(player)
    return {
        "team": {"id": team["team_id"], "name": team["team_name"]},
        "startXI": starters,
        "substitutes": substitutes,
    }


def import_statsbomb_open_data(
    project_dir: Path,
    *,
    database_path: Path | None = None,
    request: Any = requests.get,
) -> StatsBombImportResult:
    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    seen = mapped = imported = unresolved = cache_hits = 0
    observed_at = datetime.now(timezone.utc).isoformat()
    for (league, season), (competition_id, season_id) in TARGETS.items():
        matches, hit = _json(
            project_dir,
            f"matches/{competition_id}/{season_id}.json",
            request=request,
        )
        cache_hits += int(hit)
        canonical = _canonical_index(database, league, season)
        for match in matches:
            seen += 1
            home_name, home_external, home_provider_name = _team(match, "home")
            away_name, away_external, away_provider_name = _team(match, "away")
            resolved = canonical.get((str(match["match_date"]), home_name, away_name))
            if resolved is None:
                unresolved += 1
                continue
            match_id, home_id, away_id = resolved
            provider_match_id = str(match["match_id"])
            with database.batch() as connection:
                provider_id = database._lookup_id(  # noqa: SLF001
                    connection,
                    "providers",
                    "provider_id",
                    "provider_name",
                    PROVIDER,
                )
                connection.execute(
                    """
                    INSERT INTO provider_match_mapping VALUES (?, ?, ?)
                    ON CONFLICT(provider_id, provider_match_id) DO NOTHING
                    """,
                    (provider_id, provider_match_id, match_id),
                )
                for external_id, internal_id, name in (
                    (home_external, home_id, home_provider_name),
                    (away_external, away_id, away_provider_name),
                ):
                    connection.execute(
                        """
                        INSERT INTO provider_team_mapping VALUES (
                            ?, ?, ?, ?, 'normalized_exact', ?
                        )
                        ON CONFLICT(provider_id, provider_team_id) DO NOTHING
                        """,
                        (provider_id, str(external_id), internal_id, name, observed_at),
                    )
            mapped += 1
            payload, hit = _json(
                project_dir,
                f"lineups/{provider_match_id}.json",
                request=request,
            )
            cache_hits += int(hit)
            try:
                ingest_fixture_lineups(
                    database,
                    provider_fixture_id=provider_match_id,
                    lineups=[_convert_lineup(team) for team in payload],
                    observed_at=observed_at,
                    provider_name=PROVIDER,
                    player_namespace=PLAYER_NAMESPACE,
                )
            except ValueError:
                unresolved += 1
                continue
            imported += 1
    return StatsBombImportResult(
        matches_seen=seen,
        matches_mapped=mapped,
        matches_imported=imported,
        unresolved=unresolved,
        cache_hits=cache_hits,
    )
