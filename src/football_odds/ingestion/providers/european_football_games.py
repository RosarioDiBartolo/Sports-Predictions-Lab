"""Import the CC BY 4.0 European Football Games starting elevens."""

from __future__ import annotations

import csv
import sqlite3
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ...data.repository import ResearchDatabase
from ...players.observations import ingest_fixture_lineups
from ..contracts import MatchRecord

PROVIDER = "European Football Games"
PLAYER_NAMESPACE = uuid.UUID("78c97ef4-dd80-4e61-97ba-59270b496930")
DEFAULT_SEASONS = ("2016/2017", "2017/2018", "2018/2019")
LEAGUES = {
    "Premier League": ("E0", "Premier League", "England"),
    "Ligue 1": ("F1", "Ligue 1", "France"),
    "Bundesliga": ("D1", "Bundesliga", "Germany"),
    "Serie A": ("I1", "Serie A", "Italy"),
    "Primera División": ("SP1", "La Liga", "Spain"),
}
TEAM_NOISE = {
    "ac",
    "afc",
    "as",
    "borussia",
    "calcio",
    "cf",
    "club",
    "fc",
    "girondins",
    "ogc",
    "olympique",
    "sm",
    "stade",
}
TEAM_ALIASES = {"nizza": "nice", "munchen": "munich"}


@dataclass(frozen=True)
class EuropeanFootballGamesImportResult:
    matches_seen: int
    matches_imported: int
    already_imported: int
    incomplete_lineups: int
    player_observations: int
    roles_from_history: int
    roles_from_slot: int
    matches_reconciled: int
    reconciliation_unresolved: int


def _normalized(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _season_code(value: str) -> str:
    start, end = value.split("/")
    return f"{start[-2:]}{end[-2:]}"


def _team_key(value: str) -> str:
    tokens = _normalized(value).replace("'", " ").split()
    return " ".join(
        TEAM_ALIASES.get(token, token) for token in tokens if token not in TEAM_NOISE
    )


def _historical_roles(source: Path) -> dict[str, str]:
    """Learn a conservative modal role by name from tactical coordinates."""
    if not source.exists():
        return {}
    connection = sqlite3.connect(source)
    connection.row_factory = sqlite3.Row
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    names = {
        int(row["player_api_id"]): str(row["player_name"])
        for row in connection.execute("SELECT player_api_id, player_name FROM Player")
    }
    rows = connection.execute("SELECT * FROM Match")
    for row in rows:
        for side in ("home", "away"):
            for number in range(1, 12):
                candidate = row[f"{side}_player_{number}"]
                if candidate is None:
                    continue
                y = row[f"{side}_player_Y{number}"]
                player_name = names.get(int(candidate))
                if y is None or player_name is None:
                    continue
                coordinate = float(y)
                role = (
                    "G"
                    if coordinate <= 1
                    else "D"
                    if coordinate <= 5
                    else "M"
                    if coordinate <= 8
                    else "F"
                )
                counts[_normalized(player_name)][role] += 1
    connection.close()
    return {name: counter.most_common(1)[0][0] for name, counter in counts.items()}


def _slot_role(slot: int) -> str:
    # The source orders the XI from attack to goalkeeper.
    if slot == 10:
        return "G"
    if slot >= 6:
        return "D"
    if slot >= 2:
        return "M"
    return "F"


def _existing_match(
    connection: sqlite3.Connection,
    *,
    league_code: str,
    match_date: datetime,
    home_name: str,
    away_name: str,
    home_goals: int,
    away_goals: int,
) -> sqlite3.Row | None:
    """Conservatively match a provider row to an existing canonical fixture."""
    candidates = connection.execute(
        """
        SELECT m.match_id, m.home_team_id, m.away_team_id,
               ht.team_name AS home_name, at.team_name AS away_name
        FROM matches m
        JOIN leagues l ON l.league_id=m.league_id
        JOIN teams ht ON ht.team_id=m.home_team_id
        JOIN teams at ON at.team_id=m.away_team_id
        JOIN match_results r ON r.match_id=m.match_id
        WHERE l.league_code=?
          AND date(m.date) BETWEEN date(?, '-1 day') AND date(?, '+1 day')
          AND r.home_goals=? AND r.away_goals=?
        """,
        (
            league_code,
            match_date.isoformat(),
            match_date.isoformat(),
            home_goals,
            away_goals,
        ),
    ).fetchall()
    scored = [
        (
            SequenceMatcher(
                None, _team_key(home_name), _team_key(str(row["home_name"]))
            ).ratio()
            + SequenceMatcher(
                None, _team_key(away_name), _team_key(str(row["away_name"]))
            ).ratio(),
            row,
        )
        for row in candidates
    ]
    scored.sort(key=lambda item: item[0])
    if not scored or scored[-1][0] < 1.20:
        return None
    if len(scored) > 1 and scored[-1][0] - scored[-2][0] < 0.20:
        return None
    return scored[-1][1]


def _lineup(
    row: dict[str, str],
    *,
    side: str,
    team_name: str,
    learned_roles: dict[str, str],
) -> tuple[dict[str, Any], int, int]:
    players: list[dict[str, Any]] = []
    learned = fallback = 0
    for slot in range(11):
        name = row[f"{side} player {slot}"].strip()
        historical = learned_roles.get(_normalized(name))
        role = historical or _slot_role(slot)
        method = "history" if historical else "slot"
        learned += int(historical is not None)
        fallback += int(historical is None)
        players.append(
            {
                "player": {
                    "id": _normalized(name),
                    "name": name,
                    "pos": role,
                    "grid": f"derived-{method}:{slot}",
                }
            }
        )
    return (
        {
            "team": {"id": _normalized(team_name), "name": team_name},
            "startXI": players,
            "substitutes": [],
        },
        learned,
        fallback,
    )


def import_european_football_games(
    project_dir: Path,
    *,
    source_path: Path | None = None,
    database_path: Path | None = None,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
) -> EuropeanFootballGamesImportResult:
    """Import complete XIs while exposing whether every role was inferred."""
    source = source_path or (
        project_dir
        / "data"
        / "raw"
        / "external"
        / "european-football-games"
        / "data.csv"
    )
    if not source.exists():
        raise FileNotFoundError(f"European Football Games non trovato: {source}")
    role_source = (
        project_dir
        / "data"
        / "raw"
        / "external"
        / "european-soccer-database"
        / "database.sqlite"
    )
    roles = _historical_roles(role_source)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["season"] in seasons and row["league"] in LEAGUES
        ]

    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    imported = already = incomplete = observations = learned = fallback = reconciled = 0
    unresolved = 0
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
        overlap_seasons: dict[tuple[str, str], bool] = {}
        for row in rows:
            key = (LEAGUES[row["league"]][0], _season_code(row["season"]))
            if key in overlap_seasons:
                continue
            overlap_seasons[key] = (
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM matches m
                    JOIN leagues l ON l.league_id=m.league_id
                    WHERE l.league_code=? AND m.season=?
                      AND NOT EXISTS (
                        SELECT 1 FROM provider_match_mapping pm
                        WHERE pm.internal_match_id=m.match_id
                          AND pm.provider_id=?
                      )
                    """,
                    (*key, provider_id),
                ).fetchone()[0]
                > 0
            )
        for row in rows:
            league_code, league_name, country = LEAGUES[row["league"]]
            try:
                match_date = datetime.strptime(row["date"], "%d.%m.%Y")
                home_goals = int(float(row["home goals"]))
                away_goals = int(float(row["away goals"]))
            except (TypeError, ValueError):
                incomplete += 1
                continue
            fixture_id = (
                f"{row['season']}:{league_code}:{row['date']}:"
                f"{_normalized(row['home name'])}:{_normalized(row['away name'])}"
            )
            if fixture_id in existing:
                already += 1
                continue
            names = [
                row.get(f"{side} player {slot}", "").strip()
                for side in ("home", "away")
                for slot in range(11)
            ]
            if any(not name for name in names) or len(set(names)) != 22:
                incomplete += 1
                continue
            home_name = row["home name"].strip()
            away_name = row["away name"].strip()
            result = (
                "H"
                if home_goals > away_goals
                else "A"
                if away_goals > home_goals
                else "D"
            )
            canonical = _existing_match(
                connection,
                league_code=league_code,
                match_date=match_date,
                home_name=home_name,
                away_name=away_name,
                home_goals=home_goals,
                away_goals=away_goals,
            )
            if canonical is None:
                if overlap_seasons[(league_code, _season_code(row["season"]))]:
                    unresolved += 1
                    continue
                database.upsert_match(
                    PROVIDER,
                    MatchRecord(
                        provider_match_id=fixture_id,
                        date=match_date,
                        season=_season_code(row["season"]),
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
                canonical = connection.execute(
                    """
                    SELECT m.match_id, m.home_team_id, m.away_team_id
                    FROM provider_match_mapping pm
                    JOIN matches m ON m.match_id=pm.internal_match_id
                    WHERE pm.provider_id=? AND pm.provider_match_id=?
                    """,
                    (provider_id, fixture_id),
                ).fetchone()
                if canonical is None:
                    raise RuntimeError(
                        f"Fixture appena importata non trovata: {fixture_id}"
                    )
            else:
                connection.execute(
                    """
                    INSERT INTO provider_match_mapping VALUES (?, ?, ?)
                    """,
                    (provider_id, fixture_id, str(canonical["match_id"])),
                )
                reconciled += 1
            for team_name, internal_team_id in (
                (home_name, int(canonical["home_team_id"])),
                (away_name, int(canonical["away_team_id"])),
            ):
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
                        _normalized(team_name),
                        internal_team_id,
                        team_name,
                        observed_at,
                    ),
                )
            home, home_learned, home_fallback = _lineup(
                row, side="home", team_name=home_name, learned_roles=roles
            )
            away, away_learned, away_fallback = _lineup(
                row, side="away", team_name=away_name, learned_roles=roles
            )
            ingest_fixture_lineups(
                database,
                provider_fixture_id=fixture_id,
                lineups=[home, away],
                observed_at=observed_at,
                provider_name=PROVIDER,
                player_namespace=PLAYER_NAMESPACE,
            )
            imported += 1
            observations += 22
            learned += home_learned + away_learned
            fallback += home_fallback + away_fallback
    return EuropeanFootballGamesImportResult(
        matches_seen=len(rows),
        matches_imported=imported,
        already_imported=already,
        incomplete_lineups=incomplete,
        player_observations=observations,
        roles_from_history=learned,
        roles_from_slot=fallback,
        matches_reconciled=reconciled,
        reconciliation_unresolved=unresolved,
    )
