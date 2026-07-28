"""Import the CC BY-SA 4.0 Premier League 2024/25 match reports."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...data.repository import ResearchDatabase
from ...players.observations import ingest_fixture_lineups
from .european_football_games import _existing_match, _normalized

PROVIDER = "Premier League 2024-2025 Data"
PLAYER_NAMESPACE = uuid.UUID("11bbc1fa-f131-4627-b0d7-6aec8a3587ef")


@dataclass(frozen=True)
class PremierLeagueReportsImportResult:
    matches_seen: int
    matches_imported: int
    already_imported: int
    incomplete_lineups: int
    reconciliation_unresolved: int
    player_observations: int


def _role(value: str) -> str:
    tokens = {token.strip().upper() for token in value.split(",")}
    if "GK" in tokens:
        return "G"
    if tokens & {"DF", "CB", "LB", "RB", "WB"}:
        return "D"
    if tokens & {"MF", "CM", "DM", "AM", "LM", "RM"}:
        return "M"
    if tokens & {"FW", "LW", "RW", "CF", "ST"}:
        return "F"
    raise ValueError(f"Ruolo non riconosciuto: {value}")


def _lineup(
    report: dict[str, Any],
    *,
    side: str,
) -> dict[str, Any]:
    team = report["teams"][side]
    raw_players = report["lineups"][side]["players"][:11]
    stats = {
        str(row.get("Player")): str(row.get("Pos") or "")
        for row in report["player_stats"][side]
    }
    players = []
    for item in raw_players:
        name = str(item["name"])
        reported_position = stats.get(name, "")
        players.append(
            {
                "player": {
                    "id": _normalized(name),
                    "name": name,
                    "pos": _role(reported_position),
                    "grid": f"reported:{reported_position}",
                    "number": item.get("number"),
                }
            }
        )
    return {
        "team": {"id": str(team["team_id"]), "name": str(team["name"])},
        "formation": report["lineups"][side].get("formation"),
        "startXI": players,
        "substitutes": [],
    }


def import_premier_league_reports(
    project_dir: Path,
    *,
    source_path: Path | None = None,
    database_path: Path | None = None,
) -> PremierLeagueReportsImportResult:
    """Import only complete, uniquely reconciled historical starting elevens."""
    source = source_path or (
        project_dir
        / "data"
        / "raw"
        / "external"
        / "premier-league-2024-2025-data"
        / "match_reports.jsonl"
    )
    if not source.exists():
        raise FileNotFoundError(f"Report Premier League non trovato: {source}")
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    imported = already = incomplete = unresolved = observations = 0
    observed_at = datetime.now(timezone.utc).isoformat()
    with database.batch() as connection:
        provider_id = database._lookup_id(  # noqa: SLF001
            connection, "providers", "provider_id", "provider_name", PROVIDER
        )
        existing = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT provider_match_id FROM provider_match_mapping
                WHERE provider_id=?
                """,
                (provider_id,),
            )
        }
        for item in rows:
            fixture_id = str(item.get("id") or "")
            if fixture_id in existing:
                already += 1
                continue
            report = item.get("match_report") or {}
            try:
                date = datetime.fromisoformat(str(report["date"]))
                home = report["teams"]["home"]
                away = report["teams"]["away"]
                raw_lineups = report["lineups"]
                stats = report["player_stats"]
                valid = all(
                    len(raw_lineups[side]["players"]) >= 11
                    and len(
                        {player["name"] for player in raw_lineups[side]["players"][:11]}
                    )
                    == 11
                    and all(
                        str(player["name"])
                        in {
                            str(row.get("Player"))
                            for row in stats[side]
                            if row.get("Pos")
                        }
                        for player in raw_lineups[side]["players"][:11]
                    )
                    for side in ("home", "away")
                )
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                incomplete += 1
                continue
            canonical = _existing_match(
                connection,
                league_code="E0",
                match_date=date,
                home_name=str(home["name"]),
                away_name=str(away["name"]),
                home_goals=int(home["score"]),
                away_goals=int(away["score"]),
            )
            if canonical is None:
                unresolved += 1
                continue
            connection.execute(
                "INSERT INTO provider_match_mapping VALUES (?, ?, ?)",
                (provider_id, fixture_id, str(canonical["match_id"])),
            )
            for side, team_id in (
                ("home", int(canonical["home_team_id"])),
                ("away", int(canonical["away_team_id"])),
            ):
                team = report["teams"][side]
                connection.execute(
                    """
                    INSERT INTO provider_team_mapping VALUES (
                        ?, ?, ?, ?, 'normalized_exact', ?
                    ) ON CONFLICT(provider_id, provider_team_id) DO UPDATE SET
                        internal_team_id=excluded.internal_team_id,
                        provider_team_name=excluded.provider_team_name,
                        mapping_method=excluded.mapping_method,
                        observed_at=excluded.observed_at
                    """,
                    (
                        provider_id,
                        str(team["team_id"]),
                        team_id,
                        str(team["name"]),
                        observed_at,
                    ),
                )
            ingest_fixture_lineups(
                database,
                provider_fixture_id=fixture_id,
                lineups=[
                    _lineup(report, side="home"),
                    _lineup(report, side="away"),
                ],
                observed_at=observed_at,
                provider_name=PROVIDER,
                player_namespace=PLAYER_NAMESPACE,
            )
            imported += 1
            observations += 22
    return PremierLeagueReportsImportResult(
        matches_seen=len(rows),
        matches_imported=imported,
        already_imported=already,
        incomplete_lineups=incomplete,
        reconciliation_unresolved=unresolved,
        player_observations=observations,
    )
