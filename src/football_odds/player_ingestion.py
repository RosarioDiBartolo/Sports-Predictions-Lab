"""Canonical ingestion of API-Football historical lineups."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import ResearchDatabase
from .player_coverage import ApiFootballClient, load_env_value

PROVIDER = "API-Football"
PLAYER_NAMESPACE = uuid.UUID("f26863c6-fb55-47bd-b70b-b6ea6ef1ec24")


@dataclass(frozen=True)
class LineupImportResult:
    fixtures: int
    lineups: int
    players: int
    requests_made: int


@dataclass(frozen=True)
class LineupBackfillResult:
    eligible_fixtures: int
    already_complete: int
    attempted_fixtures: int
    imported_fixtures: int
    lineups: int
    players: int
    failures: tuple[dict[str, str], ...]
    requests_made: int


def _player_items(lineup: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        *[("starter", item) for item in lineup.get("startXI") or []],
        *[("substitute", item) for item in lineup.get("substitutes") or []],
    ]


def _require_complete_lineup(lineup: dict[str, Any]) -> None:
    team = lineup.get("team") or {}
    starters = list(lineup.get("startXI") or [])
    if not team.get("name"):
        raise ValueError("Lineup senza squadra.")
    if len(starters) != 11:
        raise ValueError(
            f"Lineup incompleta per {team['name']}: {len(starters)} titolari."
        )
    for _, item in _player_items(lineup):
        player = item.get("player") or {}
        if player.get("id") is None or not player.get("name"):
            raise ValueError(f"Giocatore senza identità in {team['name']}.")


def ingest_fixture_lineups(
    database: ResearchDatabase,
    *,
    provider_fixture_id: str,
    lineups: list[dict[str, Any]],
    observed_at: str,
) -> tuple[int, int]:
    """Atomically replace a mapped fixture's historical lineups."""
    if len(lineups) != 2:
        raise ValueError(
            f"Attese due lineup per fixture {provider_fixture_id}, "
            f"ricevute {len(lineups)}."
        )
    for lineup in lineups:
        _require_complete_lineup(lineup)

    players_seen: set[str] = set()
    with database.batch() as connection:
        provider_id = database._lookup_id(  # noqa: SLF001
            connection, "providers", "provider_id", "provider_name", PROVIDER
        )
        mapping = connection.execute(
            """
            SELECT internal_match_id FROM provider_match_mapping
            WHERE provider_id=? AND provider_match_id=?
            """,
            (provider_id, provider_fixture_id),
        ).fetchone()
        if mapping is None:
            raise KeyError(
                f"Fixture API-Football non mappata: {provider_fixture_id}"
            )
        match_id = str(mapping[0])
        match_date = str(
            connection.execute(
                "SELECT date FROM matches WHERE match_id=?", (match_id,)
            ).fetchone()[0]
        )

        for lineup in lineups:
            team = lineup["team"]
            provider_team_id = str(team.get("id") or "")
            team_row = connection.execute(
                """
                SELECT internal_team_id
                FROM provider_team_mapping
                WHERE provider_id=? AND provider_team_id=?
                """,
                (provider_id, provider_team_id),
            ).fetchone()
            if team_row is None:
                team_row = connection.execute(
                    "SELECT team_id FROM teams WHERE team_name=?",
                    (team["name"],),
                ).fetchone()
            if team_row is None:
                raise KeyError(f"Squadra canonica non trovata: {team['name']}")
            team_id = int(team_row[0])
            connection.execute(
                """
                INSERT INTO fixture_lineups (
                    match_id, team_id, provider_id, formation,
                    coach_provider_id, lineup_kind, observed_at, published_at
                ) VALUES (?, ?, ?, ?, ?, 'confirmed_historical', ?, NULL)
                ON CONFLICT(match_id, team_id, provider_id, lineup_kind)
                DO UPDATE SET
                    formation=excluded.formation,
                    coach_provider_id=excluded.coach_provider_id,
                    observed_at=excluded.observed_at,
                    published_at=NULL
                """,
                (
                    match_id,
                    team_id,
                    provider_id,
                    lineup.get("formation"),
                    str((lineup.get("coach") or {}).get("id") or ""),
                    observed_at,
                ),
            )
            lineup_id = int(
                connection.execute(
                    """
                    SELECT lineup_id FROM fixture_lineups
                    WHERE match_id=? AND team_id=? AND provider_id=?
                      AND lineup_kind='confirmed_historical'
                    """,
                    (match_id, team_id, provider_id),
                ).fetchone()[0]
            )
            connection.execute(
                "DELETE FROM lineup_players WHERE lineup_id=?", (lineup_id,)
            )
            for role, item in _player_items(lineup):
                player = item["player"]
                provider_player_id = str(player["id"])
                mapped = connection.execute(
                    """
                    SELECT internal_player_id FROM provider_player_mapping
                    WHERE provider_id=? AND provider_player_id=?
                    """,
                    (provider_id, provider_player_id),
                ).fetchone()
                player_id = (
                    str(mapped[0])
                    if mapped
                    else str(
                        uuid.uuid5(
                            PLAYER_NAMESPACE,
                            f"{PROVIDER}:{provider_player_id}",
                        )
                    )
                )
                connection.execute(
                    """
                    INSERT INTO players (player_id, player_name)
                    VALUES (?, ?)
                    ON CONFLICT(player_id) DO UPDATE SET
                        player_name=excluded.player_name
                    """,
                    (player_id, player["name"]),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO provider_player_mapping
                    (provider_id, provider_player_id, internal_player_id)
                    VALUES (?, ?, ?)
                    """,
                    (provider_id, provider_player_id, player_id),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO team_memberships (
                        player_id, team_id, provider_id, valid_from, observed_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (player_id, team_id, provider_id, match_date[:10], observed_at),
                )
                connection.execute(
                    """
                    INSERT INTO lineup_players (
                        lineup_id, player_id, lineup_role, position,
                        formation_grid, shirt_number
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lineup_id,
                        player_id,
                        role,
                        player.get("pos"),
                        player.get("grid"),
                        player.get("number"),
                    ),
                )
                players_seen.add(player_id)
    return len(lineups), len(players_seen)


def import_api_football_lineups(
    project_dir: Path,
    *,
    fixture_ids: tuple[str, ...],
    database_path: Path | None = None,
    client: ApiFootballClient | None = None,
    observed_at: str | None = None,
) -> LineupImportResult:
    """Fetch selected mapped fixtures and persist their historical lineups."""
    if not fixture_ids:
        raise ValueError("Specificare almeno un fixture ID.")
    key = os.getenv("API_FOOTBALL_KEY") or load_env_value(
        project_dir / ".env", "API_FOOTBALL_KEY"
    )
    active = client or ApiFootballClient(key or "")
    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    timestamp = observed_at or datetime.now(timezone.utc).isoformat()
    total_lineups = 0
    total_players: set[str] = set()
    for fixture_id in fixture_ids:
        payload = active.get("fixtures/lineups", fixture=fixture_id)
        lineups, _ = ingest_fixture_lineups(
            database,
            provider_fixture_id=fixture_id,
            lineups=payload,
            observed_at=timestamp,
        )
        total_lineups += lineups
        with database.connect() as connection:
            total_players.update(
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT lp.player_id
                    FROM lineup_players lp
                    JOIN fixture_lineups fl USING(lineup_id)
                    JOIN provider_match_mapping pm
                      ON pm.internal_match_id=fl.match_id
                     AND pm.provider_id=fl.provider_id
                    WHERE pm.provider_match_id=?
                    """,
                    (fixture_id,),
                )
            )
    return LineupImportResult(
        fixtures=len(fixture_ids),
        lineups=total_lineups,
        players=len(total_players),
        requests_made=active.requests_made,
    )


def backfill_api_football_lineups(
    project_dir: Path,
    *,
    league: str = "I1",
    seasons: tuple[str, ...] = ("2223", "2324", "2425"),
    limit: int | None = None,
    database_path: Path | None = None,
    client: ApiFootballClient | None = None,
    observed_at: str | None = None,
) -> LineupBackfillResult:
    """Import missing mapped lineups in chronological, safely resumable batches."""
    if not seasons:
        raise ValueError("Specificare almeno una stagione.")
    if limit is not None and limit < 1:
        raise ValueError("limit deve essere positivo.")
    key = os.getenv("API_FOOTBALL_KEY") or load_env_value(
        project_dir / ".env", "API_FOOTBALL_KEY"
    )
    active = client or ApiFootballClient(key or "")
    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    placeholders = ",".join("?" for _ in seasons)
    with database.connect() as connection:
        rows = connection.execute(
            f"""
            SELECT
                pm.provider_match_id,
                m.match_id,
                m.date,
                COUNT(DISTINCT fl.team_id) AS stored_lineups
            FROM provider_match_mapping pm
            JOIN providers p ON p.provider_id=pm.provider_id
            JOIN matches m ON m.match_id=pm.internal_match_id
            JOIN leagues l ON l.league_id=m.league_id
            LEFT JOIN fixture_lineups fl
              ON fl.match_id=m.match_id
             AND fl.provider_id=p.provider_id
             AND fl.lineup_kind='confirmed_historical'
            WHERE p.provider_name=?
              AND l.league_code=?
              AND m.season IN ({placeholders})
            GROUP BY pm.provider_match_id, m.match_id, m.date
            ORDER BY m.date, m.match_id
            """,
            (PROVIDER, league, *seasons),
        ).fetchall()
    eligible = len(rows)
    complete = sum(int(row["stored_lineups"]) == 2 for row in rows)
    pending = [row for row in rows if int(row["stored_lineups"]) != 2]
    if limit is not None:
        pending = pending[:limit]

    timestamp = observed_at or datetime.now(timezone.utc).isoformat()
    imported = 0
    total_lineups = 0
    players_seen: set[str] = set()
    failures: list[dict[str, str]] = []
    for row in pending:
        fixture_id = str(row["provider_match_id"])
        try:
            payload = active.get("fixtures/lineups", fixture=fixture_id)
            lineups, _ = ingest_fixture_lineups(
                database,
                provider_fixture_id=fixture_id,
                lineups=payload,
                observed_at=timestamp,
            )
        except (KeyError, RuntimeError, ValueError) as error:
            failures.append({"fixture_id": fixture_id, "error": str(error)})
            continue
        imported += 1
        total_lineups += lineups
        with database.connect() as connection:
            players_seen.update(
                str(item[0])
                for item in connection.execute(
                    """
                    SELECT lp.player_id
                    FROM lineup_players lp
                    JOIN fixture_lineups fl USING(lineup_id)
                    WHERE fl.match_id=?
                      AND fl.lineup_kind='confirmed_historical'
                    """,
                    (str(row["match_id"]),),
                )
            )
    return LineupBackfillResult(
        eligible_fixtures=eligible,
        already_complete=complete,
        attempted_fixtures=len(pending),
        imported_fixtures=imported,
        lineups=total_lineups,
        players=len(players_seen),
        failures=tuple(failures),
        requests_made=active.requests_made,
    )
