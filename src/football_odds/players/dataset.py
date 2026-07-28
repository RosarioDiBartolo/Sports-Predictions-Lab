"""Build and audit the canonical post-lineup training dataset."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.repository import ResearchDatabase

PROVIDER_PRIORITY = {
    "Lega Serie A SDP": 0,
    "API-Football": 1,
    "StatsBomb Open Data": 2,
    "Football Data from Transfermarkt": 3,
    "Premier League 2024-2025 Data": 4,
    "Football Matches from Europe and South America": 5,
    "European Football Games": 6,
    "European Soccer Database": 7,
}
WARMUP_SEASONS = ("0910",)


@dataclass(frozen=True)
class PlayerDatasetResult:
    matches_seen: int
    training_ready: int
    quarantined: int
    player_observations: int
    reasons: dict[str, int]
    outputs: dict[str, Path]


def _lineups_by_match(database: ResearchDatabase) -> dict[str, list[dict[str, Any]]]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT
                fl.match_id, fl.lineup_id, fl.team_id, p.provider_name,
                SUM(CASE WHEN lp.lineup_role='starter' THEN 1 ELSE 0 END) AS starters,
                SUM(
                    CASE WHEN lp.lineup_role='starter'
                         AND COALESCE(TRIM(lp.position), '')=''
                         THEN 1 ELSE 0 END
                ) AS starters_without_position,
                SUM(CASE WHEN lp.lineup_role='substitute' THEN 1 ELSE 0 END) AS bench
            FROM fixture_lineups fl
            JOIN providers p ON p.provider_id=fl.provider_id
            LEFT JOIN lineup_players lp ON lp.lineup_id=fl.lineup_id
            WHERE fl.lineup_kind IN ('confirmed_historical', 'confirmed_timestamped')
            GROUP BY fl.match_id, fl.lineup_id, fl.team_id, p.provider_name
            """
        ).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["match_id"]), []).append(dict(row))
    return grouped


def _players(
    connection: sqlite3.Connection,
    lineup_id: int,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT
                lp.player_id,
                pl.player_name,
                lp.position,
                lp.lineup_role,
                lp.formation_grid,
                lp.shirt_number,
                stats.minute_in,
                stats.minute_out,
                stats.minutes_played,
                stats.player_id IS NOT NULL AS timing_observed
            FROM lineup_players lp
            JOIN players pl ON pl.player_id=lp.player_id
            JOIN fixture_lineups fl ON fl.lineup_id=lp.lineup_id
            LEFT JOIN player_match_lineup_stats stats
              ON stats.match_id=fl.match_id
             AND stats.team_id=fl.team_id
             AND stats.player_id=lp.player_id
             AND stats.provider_id=fl.provider_id
            WHERE lp.lineup_id=?
            ORDER BY lp.lineup_role DESC, lp.player_id
            """,
            (lineup_id,),
        )
    ]


def _choose_lineup(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    complete = [row for row in candidates if int(row["starters"] or 0) == 11]
    if not complete:
        return None
    return min(
        complete,
        key=lambda row: (
            PROVIDER_PRIORITY.get(str(row["provider_name"]), 99),
            -int(row["bench"] or 0),
            int(row["lineup_id"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0])
        if rows
        else [
            "match_id",
            "date",
            "season",
            "league",
            "home_team",
            "away_team",
            "result",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _feature_census(database: ResearchDatabase) -> dict[str, Any]:
    """Audit raw player-match fields before defining temporal fallbacks."""
    with database.connect() as connection:
        total = dict(
            connection.execute(
                """
                SELECT
                    COUNT(*) AS raw_rows,
                    COUNT(DISTINCT lp.player_id || ':' || fl.match_id)
                        AS distinct_player_matches,
                    SUM(lp.lineup_role='starter') AS starters,
                    SUM(lp.lineup_role='substitute') AS bench,
                    SUM(COALESCE(TRIM(lp.position), '')<>'') AS position,
                    SUM(COALESCE(TRIM(lp.formation_grid), '')<>'')
                        AS original_position,
                    SUM(lp.shirt_number IS NOT NULL) AS shirt_number
                FROM lineup_players lp
                JOIN fixture_lineups fl USING(lineup_id)
                """
            ).fetchone()
        )
        timing = dict(
            connection.execute(
                """
                SELECT
                    COUNT(*) AS rows,
                    SUM(minute_in IS NOT NULL) AS minute_in,
                    SUM(minute_out IS NOT NULL) AS minute_out,
                    SUM(minutes_played>0) AS positive_minutes
                FROM player_match_lineup_stats
                """
            ).fetchone()
        )
        membership = dict(
            connection.execute(
                """
                SELECT
                    COUNT(*) AS rows,
                    COUNT(DISTINCT player_id) AS players,
                    SUM(valid_to IS NOT NULL) AS bounded_intervals
                FROM team_memberships
                """
            ).fetchone()
        )
        providers = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    p.provider_name AS provider,
                    COUNT(*) AS rows,
                    SUM(lp.lineup_role='starter') AS starters,
                    SUM(lp.lineup_role='substitute') AS bench,
                    SUM(COALESCE(TRIM(lp.position), '')<>'') AS position,
                    SUM(COALESCE(TRIM(lp.formation_grid), '')<>'')
                        AS original_position,
                    SUM(lp.shirt_number IS NOT NULL) AS shirt_number
                FROM lineup_players lp
                JOIN fixture_lineups fl USING(lineup_id)
                JOIN providers p USING(provider_id)
                GROUP BY p.provider_name
                ORDER BY rows DESC, provider
                """
            )
        ]
    return {
        "raw_observations": total,
        "real_timing": timing,
        "team_memberships": membership,
        "by_provider": providers,
        "fallback_contract": {
            "minutes": "real_when_available_else_starter_90_bench_unknown",
            "shared_minutes": "real_when_available_else_shared_starts_proxy",
            "original_position": "formation_grid_when_available_else_department",
            "bench_entry": "minute_in_when_available_else_unknown",
            "quality_indicators_required": True,
        },
    }


def _collect_match(
    match: dict[str, Any],
    home: dict[str, Any] | None,
    away: dict[str, Any] | None,
    home_players: list[dict[str, Any]],
    away_players: list[dict[str, Any]],
    ready: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    reasons: Counter[str],
    coverage: Counter[tuple[str, str, str]],
) -> None:
    rejected: list[str] = []
    if match["result"] not in {"H", "D", "A"}:
        rejected.append("invalid_result")
    if home is None:
        rejected.append("missing_home_lineup")
    if away is None:
        rejected.append("missing_away_lineup")
    starters = [
        row for row in home_players + away_players if row["lineup_role"] == "starter"
    ]
    if home and int(home["starters_without_position"] or 0):
        rejected.append("home_starter_role_missing")
    if away and int(away["starters_without_position"] or 0):
        rejected.append("away_starter_role_missing")
    if len({row["player_id"] for row in starters}) != len(starters):
        rejected.append("duplicate_starter")

    base = {
        "match_id": match["match_id"],
        "date": match["date"],
        "season": match["season"],
        "league": match["league"],
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "result": match["result"],
    }
    league_season = (str(match["league"]), str(match["season"]))
    coverage[(*league_season, "total")] += 1
    if rejected:
        unique_reasons = sorted(set(rejected))
        reasons.update(unique_reasons)
        quarantine.append({**base, "reasons": unique_reasons})
        return

    assert home is not None and away is not None
    coverage[(*league_season, "ready")] += 1
    ready.append(
        {
            **base,
            "home_lineup_provider": home["provider_name"],
            "away_lineup_provider": away["provider_name"],
            "home_starters": json.dumps(
                [row for row in home_players if row["lineup_role"] == "starter"],
                ensure_ascii=False,
            ),
            "away_starters": json.dumps(
                [row for row in away_players if row["lineup_role"] == "starter"],
                ensure_ascii=False,
            ),
            "home_bench": json.dumps(
                [row for row in home_players if row["lineup_role"] == "substitute"],
                ensure_ascii=False,
            ),
            "away_bench": json.dumps(
                [row for row in away_players if row["lineup_role"] == "substitute"],
                ensure_ascii=False,
            ),
            "home_bench_count": int(home["bench"] or 0),
            "away_bench_count": int(away["bench"] or 0),
        }
    )


def build_player_dataset(
    project_dir: Path,
    *,
    database_path: Path | None = None,
) -> PlayerDatasetResult:
    """Export valid 22-starter matches and quarantine every rejected match."""
    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    lineups = _lineups_by_match(database)
    with database.connect() as connection:
        warmup_placeholders = ",".join("?" for _ in WARMUP_SEASONS)
        matches = connection.execute(
            f"""
            SELECT
                m.match_id, m.date, m.season, l.league_code AS league,
                ht.team_id AS home_team_id, ht.team_name AS home_team,
                at.team_id AS away_team_id, at.team_name AS away_team,
                mr.result
            FROM matches m
            JOIN leagues l ON l.league_id=m.league_id
            JOIN teams ht ON ht.team_id=m.home_team_id
            JOIN teams at ON at.team_id=m.away_team_id
            LEFT JOIN match_results mr ON mr.match_id=m.match_id
            WHERE m.season NOT IN ({warmup_placeholders})
            ORDER BY m.date, m.match_id
            """,
            WARMUP_SEASONS,
        ).fetchall()
        player_observations = int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT lp.player_id || ':' || fl.match_id)
                FROM lineup_players lp
                JOIN fixture_lineups fl USING(lineup_id)
                """
            ).fetchone()[0]
        )

    ready: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    coverage: Counter[tuple[str, str, str]] = Counter()
    with database.connect() as player_connection:
        for match_row in matches:
            match = dict(match_row)
            candidates = lineups.get(str(match["match_id"]), [])
            home = _choose_lineup(
                [row for row in candidates if row["team_id"] == match["home_team_id"]]
            )
            away = _choose_lineup(
                [row for row in candidates if row["team_id"] == match["away_team_id"]]
            )
            home_players = (
                _players(player_connection, int(home["lineup_id"])) if home else []
            )
            away_players = (
                _players(player_connection, int(away["lineup_id"])) if away else []
            )
            _collect_match(
                match,
                home,
                away,
                home_players,
                away_players,
                ready,
                quarantine,
                reasons,
                coverage,
            )

    output_dir = project_dir / "reports" / "player_data" / "dataset"
    dataset_path = project_dir / "data" / "processed" / "player_training_ready.csv"
    quarantine_path = output_dir / "quarantine.jsonl"
    coverage_path = output_dir / "coverage.json"
    feature_census_path = output_dir / "player_feature_census.json"
    report_path = output_dir / "PLAYER_DATASET_REPORT.md"
    _write_csv(dataset_path, ready)
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quarantine),
        encoding="utf-8",
    )
    dimensions = sorted({(key[0], key[1]) for key in coverage})
    coverage_rows = [
        {
            "league": league,
            "season": season,
            "matches": coverage[(league, season, "total")],
            "training_ready": coverage[(league, season, "ready")],
        }
        for league, season in dimensions
    ]
    coverage_path.write_text(
        json.dumps(
            {
                "matches_seen": len(matches),
                "training_ready": len(ready),
                "quarantined": len(quarantine),
                "player_observations": player_observations,
                "reasons": dict(sorted(reasons.items())),
                "by_league_season": coverage_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    feature_census = _feature_census(database)
    feature_census_path.write_text(
        json.dumps(feature_census, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    reason_lines = (
        "\n".join(f"- `{reason}`: {count}" for reason, count in sorted(reasons.items()))
        or "- Nessun caso in quarantena."
    )
    report_path.write_text(
        "# Player dataset\n\n"
        f"- Match canonici esaminati: **{len(matches)}**\n"
        f"- Match `training_ready`: **{len(ready)}**\n"
        f"- Match in quarantena: **{len(quarantine)}**\n\n"
        f"- Osservazioni giocatore-partita: **{player_observations}**\n\n"
        "## Contratto minimo\n\n"
        "Due squadre canoniche, due lineup confermate, 11 titolari per squadra, "
        "ID e ruolo per ogni titolare, risultato H/D/A e nessun titolare duplicato.\n\n"
        "La panchina confermata è esportata separatamente dai titolari quando "
        "disponibile. Copertura e fallback delle feature individuali sono censiti "
        "in `player_feature_census.json`.\n\n"
        "## Quarantena\n\n"
        f"{reason_lines}\n",
        encoding="utf-8",
    )
    return PlayerDatasetResult(
        matches_seen=len(matches),
        training_ready=len(ready),
        quarantined=len(quarantine),
        player_observations=player_observations,
        reasons=dict(sorted(reasons.items())),
        outputs={
            "dataset": dataset_path,
            "quarantine": quarantine_path,
            "coverage": coverage_path,
            "feature_census": feature_census_path,
            "report": report_path,
        },
    )
