"""Quota-aware orchestration for historical player data providers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..data.repository import ResearchDatabase
from .coverage import ApiFootballClient, load_env_value
from .observations import backfill_api_football_lineups
from .reconciliation import LEAGUE_CODES, reconcile_fixtures

API_SEASONS = {
    "1819": 2018,
    "1920": 2019,
    "2021": 2020,
    "2122": 2021,
    "2223": 2022,
    "2324": 2023,
    "2425": 2024,
}
API_FOOTBALL_FREE_SEASONS = ("2223", "2324", "2425")


@dataclass(frozen=True)
class PlayerCollectionResult:
    request_budget: int
    requests_made: int
    fixture_batches: int
    fixtures_seen: int
    fixtures_mapped: int
    unresolved_fixtures: int
    lineup_attempts: int
    lineups_imported: int
    lineup_failures: int
    manifest: Path


def _mapping_counts(
    database: ResearchDatabase,
) -> dict[tuple[str, str], tuple[int, int]]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT
                l.league_code, m.season,
                COUNT(DISTINCT m.match_id) AS matches,
                COUNT(DISTINCT pm.internal_match_id) AS mapped
            FROM matches m
            JOIN leagues l ON l.league_id=m.league_id
            LEFT JOIN providers p ON p.provider_name='API-Football'
            LEFT JOIN provider_match_mapping pm
              ON pm.provider_id=p.provider_id
             AND pm.internal_match_id=m.match_id
            GROUP BY l.league_code, m.season
            """
        ).fetchall()
    return {
        (str(row["league_code"]), str(row["season"])): (
            int(row["matches"]),
            int(row["mapped"]),
        )
        for row in rows
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _cached_fixtures(
    project_dir: Path,
    *,
    league: str,
    season: str,
    api_league: int,
    api_season: int,
    client: ApiFootballClient,
) -> tuple[list[dict[str, Any]], bool]:
    path = (
        project_dir
        / "data"
        / "cache"
        / "api_football"
        / "fixtures"
        / f"{league}_{season}.json"
    )
    if path.exists():
        return list(json.loads(path.read_text(encoding="utf-8"))), True
    fixtures = client.get("fixtures", league=api_league, season=api_season)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(fixtures, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return fixtures, False


def _league_missing_lineups(
    database: ResearchDatabase,
    *,
    league: str,
    seasons: tuple[str, ...],
) -> int:
    placeholders = ",".join("?" for _ in seasons)
    with database.connect() as connection:
        return int(
            connection.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT m.match_id
                    FROM matches m
                    JOIN leagues l ON l.league_id=m.league_id
                    LEFT JOIN fixture_lineups fl
                      ON fl.match_id=m.match_id
                     AND fl.lineup_kind IN (
                         'confirmed_historical', 'confirmed_timestamped'
                     )
                    WHERE l.league_code=? AND m.season IN ({placeholders})
                    GROUP BY m.match_id
                    HAVING COUNT(DISTINCT fl.team_id) != 2
                )
                """,
                (league, *seasons),
            ).fetchone()[0]
        )


def collect_api_football_player_data(
    project_dir: Path,
    *,
    leagues: tuple[str, ...] = ("E0", "SP1", "D1", "F1", "I1"),
    seasons: tuple[str, ...] = tuple(API_SEASONS),
    request_budget: int = 100,
    database_path: Path | None = None,
    client: ApiFootballClient | None = None,
) -> PlayerCollectionResult:
    """Map missing fixture batches, then spend the remaining budget on lineups."""
    if request_budget < 1:
        raise ValueError("request_budget deve essere positivo.")
    unknown_leagues = sorted(set(leagues) - set(LEAGUE_CODES.values()))
    if unknown_leagues:
        raise ValueError(f"Leghe non supportate: {', '.join(unknown_leagues)}")
    unknown_seasons = sorted(set(seasons) - set(API_SEASONS))
    if unknown_seasons:
        raise ValueError(f"Stagioni non supportate: {', '.join(unknown_seasons)}")

    key = os.getenv("API_FOOTBALL_KEY") or load_env_value(
        project_dir / ".env", "API_FOOTBALL_KEY"
    )
    active = client or ApiFootballClient(key or "")
    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    manifest_path = (
        project_dir / "reports" / "player_data" / "dataset" / "collector_state.json"
    )
    started_at = datetime.now(timezone.utc).isoformat()
    state: dict[str, Any] = {
        "provider": "API-Football",
        "started_at": started_at,
        "updated_at": started_at,
        "status": "running",
        "request_budget": request_budget,
        "requests_made": 0,
        "fixture_batches": [],
        "lineup_batches": [],
        "skipped_seasons": sorted(set(seasons) - set(API_FOOTBALL_FREE_SEASONS)),
    }
    _write_manifest(manifest_path, state)

    accessible_seasons = tuple(
        season for season in seasons if season in API_FOOTBALL_FREE_SEASONS
    )
    inverse_leagues = {code: api_id for api_id, code in LEAGUE_CODES.items()}
    counts = _mapping_counts(database)
    fixtures_seen = 0
    fixtures_mapped = 0
    unresolved = 0
    fixture_batches = 0
    for league in leagues:
        for season in accessible_seasons:
            total, mapped = counts.get((league, season), (0, 0))
            if total == 0 or mapped >= total or active.requests_made >= request_budget:
                continue
            fixtures, cache_hit = _cached_fixtures(
                project_dir,
                league=league,
                season=season,
                api_league=inverse_leagues[league],
                api_season=API_SEASONS[season],
                client=active,
            )
            result = reconcile_fixtures(
                database,
                fixtures,
                observed_at=datetime.now(timezone.utc).isoformat(),
            )
            fixture_batches += 1
            fixtures_seen += result.fixtures_seen
            fixtures_mapped += result.fixtures_mapped
            unresolved += len(result.unresolved)
            state["requests_made"] = active.requests_made
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            state["fixture_batches"].append(
                {
                    "league": league,
                    "season": season,
                    "fixtures_seen": result.fixtures_seen,
                    "fixtures_mapped": result.fixtures_mapped,
                    "already_mapped": result.already_mapped,
                    "unresolved": len(result.unresolved),
                    "cache_hit": cache_hit,
                }
            )
            _write_manifest(manifest_path, state)

    lineup_attempts = 0
    lineups_imported = 0
    lineup_failures = 0
    for league in leagues:
        if not accessible_seasons:
            break
        if (
            _league_missing_lineups(
                database,
                league=league,
                seasons=accessible_seasons,
            )
            == 0
        ):
            continue
        remaining = request_budget - active.requests_made
        if remaining <= 0:
            break
        result = backfill_api_football_lineups(
            project_dir,
            league=league,
            seasons=accessible_seasons,
            limit=remaining,
            database_path=database.path,
            client=active,
        )
        lineup_attempts += result.attempted_fixtures
        lineups_imported += result.imported_fixtures
        lineup_failures += len(result.failures)
        state["requests_made"] = active.requests_made
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["lineup_batches"].append(
            {
                "league": league,
                "attempted": result.attempted_fixtures,
                "imported": result.imported_fixtures,
                "failures": list(result.failures),
            }
        )
        _write_manifest(manifest_path, state)

    state["status"] = (
        "budget_exhausted" if active.requests_made >= request_budget else "idle"
    )
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, state)
    return PlayerCollectionResult(
        request_budget=request_budget,
        requests_made=active.requests_made,
        fixture_batches=fixture_batches,
        fixtures_seen=fixtures_seen,
        fixtures_mapped=fixtures_mapped,
        unresolved_fixtures=unresolved,
        lineup_attempts=lineup_attempts,
        lineups_imported=lineups_imported,
        lineup_failures=lineup_failures,
        manifest=manifest_path,
    )
