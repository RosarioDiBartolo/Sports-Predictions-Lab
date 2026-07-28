"""Fill remaining lineup gaps from a CC0 multi-league match-report dataset."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ...data.repository import ResearchDatabase
from ...players.observations import ingest_fixture_lineups
from .european_football_games import _existing_match, _normalized
from .premier_league_reports import _role

PROVIDER = "Football Matches from Europe and South America"
PLAYER_NAMESPACE = uuid.UUID("f50df00b-15db-40b0-a13a-38d6f5399a98")
COUNTRIES = {
    "ENG": "E0",
    "ESP": "SP1",
    "ITA": "I1",
    "GER": "D1",
    "FRA": "F1",
}


@dataclass(frozen=True)
class MatchReportsOpenImportResult:
    matches_seen: int
    matches_imported: int
    already_imported: int
    incomplete_lineups: int
    reconciliation_unresolved: int
    player_observations: int


def _lineup(
    row: dict[str, Any],
    *,
    side: str,
) -> dict[str, Any]:
    players = []
    for number in range(1, 12):
        name = str(row[f"starting_name_{side}{number}"])
        position = str(row[f"starting_position_{side}{number}"])
        players.append(
            {
                "player": {
                    "id": _normalized(name),
                    "name": name,
                    "pos": _role(position),
                    "grid": f"reported:{position}",
                }
            }
        )
    team_key = "Home" if side == "home" else "Away"
    formation_key = f"formation_{side}"
    return {
        "team": {
            "id": _normalized(str(row[team_key])),
            "name": str(row[team_key]),
        },
        "formation": (
            None if pd.isna(row.get(formation_key)) else str(row[formation_key])
        ),
        "startXI": players,
        "substitutes": [],
    }


def _missing_targets(database: ResearchDatabase) -> tuple[set[str], set[str]]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            WITH complete AS (
                SELECT fl.match_id, fl.team_id
                FROM fixture_lineups fl
                JOIN lineup_players lp USING(lineup_id)
                WHERE fl.lineup_kind IN (
                    'confirmed_historical', 'confirmed_timestamped'
                ) AND lp.lineup_role='starter'
                GROUP BY fl.lineup_id
                HAVING COUNT(*)=11
                  AND SUM(
                    CASE WHEN COALESCE(TRIM(lp.position), '')=''
                         THEN 1 ELSE 0 END
                  )=0
            )
            SELECT m.match_id, date(m.date) AS match_date
            FROM matches m
            WHERE NOT EXISTS (
                SELECT 1 FROM complete c
                WHERE c.match_id=m.match_id AND c.team_id=m.home_team_id
            ) OR NOT EXISTS (
                SELECT 1 FROM complete c
                WHERE c.match_id=m.match_id AND c.team_id=m.away_team_id
            )
            """
        ).fetchall()
    return (
        {str(row["match_id"]) for row in rows},
        {str(row["match_date"]) for row in rows},
    )


def import_match_reports_open_data(
    project_dir: Path,
    *,
    source_path: Path | None = None,
    database_path: Path | None = None,
) -> MatchReportsOpenImportResult:
    """Import complete report lineups only for currently incomplete fixtures."""
    source = source_path or (
        project_dir
        / "data"
        / "raw"
        / "external"
        / "football-match-reports-cc0"
        / "games.csv"
    )
    if not source.exists():
        raise FileNotFoundError(f"Dataset report CC0 non trovato: {source}")
    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    missing_match_ids, missing_dates = _missing_targets(database)
    rows = pd.read_csv(source, low_memory=False)
    rows = rows.assign(
        _parsed_date=pd.to_datetime(
            rows["Date"],
            format="mixed",
            errors="coerce",
        )
    )
    rows = rows[
        rows["Country"].isin(COUNTRIES)
        & rows["_parsed_date"].dt.date.astype(str).isin(missing_dates)
    ]
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
        for raw in rows.to_dict("records"):
            fixture_id = str(raw["ID"])
            if fixture_id in existing:
                already += 1
                continue
            required = [
                raw.get(f"starting_{field}_{side}{number}")
                for side in ("home", "away")
                for number in range(1, 12)
                for field in ("name", "position")
            ]
            if any(pd.isna(value) or not str(value).strip() for value in required):
                incomplete += 1
                continue
            date = pd.Timestamp(raw["_parsed_date"]).to_pydatetime()
            league_code = COUNTRIES[str(raw["Country"])]
            match = _existing_match(
                connection,
                league_code=league_code,
                match_date=date,
                home_name=str(raw["Home"]),
                away_name=str(raw["Away"]),
                home_goals=int(raw["HomeGoals"]),
                away_goals=int(raw["AwayGoals"]),
            )
            if match is None:
                unresolved += 1
                continue
            if str(match["match_id"]) not in missing_match_ids:
                continue
            connection.execute(
                "INSERT INTO provider_match_mapping VALUES (?, ?, ?)",
                (provider_id, fixture_id, str(match["match_id"])),
            )
            for side, team_id in (
                ("Home", int(match["home_team_id"])),
                ("Away", int(match["away_team_id"])),
            ):
                team_name = str(raw[side])
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
                        team_id,
                        team_name,
                        observed_at,
                    ),
                )
            ingest_fixture_lineups(
                database,
                provider_fixture_id=fixture_id,
                lineups=[
                    _lineup(raw, side="home"),
                    _lineup(raw, side="away"),
                ],
                observed_at=observed_at,
                provider_name=PROVIDER,
                player_namespace=PLAYER_NAMESPACE,
            )
            imported += 1
            observations += 22
    return MatchReportsOpenImportResult(
        matches_seen=len(rows),
        matches_imported=imported,
        already_imported=already,
        incomplete_lineups=incomplete,
        reconciliation_unresolved=unresolved,
        player_observations=observations,
    )
