"""Conservative reconciliation of API-Football fixtures to canonical matches."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import ResearchDatabase
from .player_coverage import ApiFootballClient, load_env_value

PROVIDER = "API-Football"
LEAGUE_CODES = {39: "E0", 140: "SP1", 135: "I1", 78: "D1", 61: "F1"}
IGNORED_TOKENS = {"ac", "afc", "as", "cf", "fc", "calcio", "hellas"}


@dataclass(frozen=True)
class ReconciliationResult:
    fixtures_seen: int
    fixtures_mapped: int
    already_mapped: int
    unresolved: tuple[dict[str, object], ...]
    requests_made: int = 0


def normalize_team_name(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[a-z0-9]+", ascii_name.casefold())
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        if len(tokens[index]) == 1:
            end = index
            while end < len(tokens) and len(tokens[end]) == 1:
                end += 1
            acronym = "".join(tokens[index:end])
            if acronym in IGNORED_TOKENS:
                index = end
                continue
        if tokens[index] not in IGNORED_TOKENS:
            normalized.append(tokens[index])
        index += 1
    return " ".join(normalized)


def _team_candidates(
    connection: Any, league_code: str, provider_name: str
) -> dict[str, list[tuple[int, str]]]:
    rows = connection.execute(
        """
        SELECT DISTINCT t.team_id, t.team_name
        FROM teams t
        JOIN matches m
          ON t.team_id IN (m.home_team_id, m.away_team_id)
        JOIN leagues l ON l.league_id=m.league_id
        WHERE l.league_code=?
        """,
        (league_code,),
    ).fetchall()
    result: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        result.setdefault(normalize_team_name(str(row["team_name"])), []).append(
            (int(row["team_id"]), str(row["team_name"]))
        )
    return result


def _resolve_team(
    connection: Any,
    *,
    provider_id: int,
    provider_team: dict[str, Any],
    candidates: dict[str, list[tuple[int, str]]],
    observed_at: str,
) -> int | None:
    provider_team_id = str(provider_team["id"])
    existing = connection.execute(
        """
        SELECT internal_team_id FROM provider_team_mapping
        WHERE provider_id=? AND provider_team_id=?
        """,
        (provider_id, provider_team_id),
    ).fetchone()
    if existing:
        return int(existing[0])
    matches = candidates.get(normalize_team_name(str(provider_team["name"])), [])
    if len(matches) != 1:
        return None
    internal_team_id = matches[0][0]
    connection.execute(
        """
        INSERT INTO provider_team_mapping (
            provider_id, provider_team_id, internal_team_id,
            provider_team_name, mapping_method, observed_at
        ) VALUES (?, ?, ?, ?, 'normalized_exact', ?)
        """,
        (
            provider_id,
            provider_team_id,
            internal_team_id,
            provider_team["name"],
            observed_at,
        ),
    )
    return internal_team_id


def reconcile_fixtures(
    database: ResearchDatabase,
    fixtures: list[dict[str, Any]],
    *,
    observed_at: str,
) -> ReconciliationResult:
    mapped = 0
    already = 0
    unresolved: list[dict[str, object]] = []
    with database.batch() as connection:
        provider_id = database._lookup_id(  # noqa: SLF001
            connection, "providers", "provider_id", "provider_name", PROVIDER
        )
        candidate_cache: dict[str, dict[str, list[tuple[int, str]]]] = {}
        for item in fixtures:
            fixture = item.get("fixture") or {}
            league = item.get("league") or {}
            teams = item.get("teams") or {}
            fixture_id = str(fixture.get("id"))
            league_id = int(league.get("id") or 0)
            league_code = LEAGUE_CODES.get(league_id)
            if not league_code:
                unresolved.append(
                    {"fixture_id": fixture_id, "reason": "unsupported_league"}
                )
                continue
            existing = connection.execute(
                """
                SELECT 1 FROM provider_match_mapping
                WHERE provider_id=? AND provider_match_id=?
                """,
                (provider_id, fixture_id),
            ).fetchone()
            if existing:
                already += 1
                continue
            candidates = candidate_cache.setdefault(
                league_code,
                _team_candidates(connection, league_code, PROVIDER),
            )
            home_id = _resolve_team(
                connection,
                provider_id=provider_id,
                provider_team=teams.get("home") or {},
                candidates=candidates,
                observed_at=observed_at,
            )
            away_id = _resolve_team(
                connection,
                provider_id=provider_id,
                provider_team=teams.get("away") or {},
                candidates=candidates,
                observed_at=observed_at,
            )
            if home_id is None or away_id is None:
                unresolved.append(
                    {"fixture_id": fixture_id, "reason": "team_unresolved"}
                )
                continue
            match = connection.execute(
                """
                SELECT m.match_id
                FROM matches m JOIN leagues l ON l.league_id=m.league_id
                WHERE l.league_code=? AND date(m.date)=date(?)
                  AND m.home_team_id=? AND m.away_team_id=?
                """,
                (league_code, fixture.get("date"), home_id, away_id),
            ).fetchall()
            if len(match) != 1:
                unresolved.append(
                    {
                        "fixture_id": fixture_id,
                        "reason": "match_not_unique" if match else "match_not_found",
                    }
                )
                continue
            connection.execute(
                """
                INSERT INTO provider_match_mapping (
                    provider_id, provider_match_id, internal_match_id
                ) VALUES (?, ?, ?)
                """,
                (provider_id, fixture_id, str(match[0]["match_id"])),
            )
            mapped += 1
    return ReconciliationResult(
        fixtures_seen=len(fixtures),
        fixtures_mapped=mapped,
        already_mapped=already,
        unresolved=tuple(unresolved),
    )


def reconcile_api_football(
    project_dir: Path,
    *,
    leagues: tuple[int, ...],
    seasons: tuple[int, ...],
    database_path: Path | None = None,
    client: ApiFootballClient | None = None,
) -> ReconciliationResult:
    key = os.getenv("API_FOOTBALL_KEY") or load_env_value(
        project_dir / ".env", "API_FOOTBALL_KEY"
    )
    active = client or ApiFootballClient(key or "")
    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    fixtures: list[dict[str, Any]] = []
    for league in leagues:
        if league not in LEAGUE_CODES:
            raise ValueError(f"Lega API-Football non supportata: {league}")
        for season in seasons:
            fixtures.extend(active.get("fixtures", league=league, season=season))
    result = reconcile_fixtures(
        database,
        fixtures,
        observed_at=datetime.now(timezone.utc).isoformat(),
    )
    return ReconciliationResult(
        fixtures_seen=result.fixtures_seen,
        fixtures_mapped=result.fixtures_mapped,
        already_mapped=result.already_mapped,
        unresolved=result.unresolved,
        requests_made=active.requests_made,
    )
