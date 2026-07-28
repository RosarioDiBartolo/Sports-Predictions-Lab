"""Import recent starting elevens from the CC0 Transfermarkt Kaggle dataset."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from ...data.repository import ResearchDatabase
from ...players.observations import ingest_fixture_lineups
from .european_football_games import _team_key

PROVIDER = "Football Data from Transfermarkt"
PLAYER_NAMESPACE = uuid.UUID("008ce233-0fb1-4371-a354-97c9ee987a29")
COMPETITIONS = {
    "GB1": "E0",
    "ES1": "SP1",
    "IT1": "I1",
    "L1": "D1",
    "FR1": "F1",
}
DEFAULT_SEASONS = tuple(range(2018, 2025))


@dataclass(frozen=True)
class TransfermarktOpenImportResult:
    matches_seen: int
    matches_imported: int
    already_imported: int
    incomplete_lineups: int
    reconciliation_unresolved: int
    player_observations: int
    result_discrepancies: int
    lineups_reconstructed: int


def _season_code(year: int) -> str:
    return f"{year % 100:02}{(year + 1) % 100:02}"


def _role(value: str) -> str:
    normalized = value.casefold()
    if "keeper" in normalized:
        return "G"
    if "back" in normalized or "defender" in normalized or "sweeper" in normalized:
        return "D"
    if "midfield" in normalized:
        return "M"
    if "winger" in normalized or "forward" in normalized or "striker" in normalized:
        return "F"
    raise ValueError(f"Posizione Transfermarkt non riconosciuta: {value}")


def _shirt_number(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _canonical_index(
    database: ResearchDatabase,
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT m.match_id, date(m.date) AS match_date,
                   l.league_code, m.home_team_id, m.away_team_id,
                   ht.team_name AS home_name, at.team_name AS away_name,
                   r.home_goals, r.away_goals
            FROM matches m
            JOIN leagues l ON l.league_id=m.league_id
            JOIN teams ht ON ht.team_id=m.home_team_id
            JOIN teams at ON at.team_id=m.away_team_id
            JOIN match_results r ON r.match_id=m.match_id
            WHERE m.season IN (
                '1819', '1920', '2021', '2122', '2223', '2324', '2425'
            )
            """
        ).fetchall()
    result: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[
            (
                str(row["league_code"]),
                str(row["match_date"]),
                int(row["home_goals"]),
                int(row["away_goals"]),
            )
        ].append(dict(row))
    return result


def _resolve_match(
    index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    *,
    league_code: str,
    date: str,
    home_name: str,
    away_name: str,
    home_goals: int,
    away_goals: int,
) -> dict[str, Any] | None:
    parsed = datetime.fromisoformat(date)
    candidates: list[dict[str, Any]] = []
    for offset in (-1, 0, 1):
        candidate_date = (parsed + timedelta(days=offset)).date().isoformat()
        candidates.extend(
            index.get(
                (league_code, candidate_date, home_goals, away_goals),
                [],
            )
        )
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


def _resolve_by_team_ids(
    index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    *,
    league_code: str,
    date: str,
    home_goals: int,
    away_goals: int,
    home_team_id: int,
    away_team_id: int,
) -> dict[str, Any] | None:
    parsed = datetime.fromisoformat(date)
    matches: list[dict[str, Any]] = []
    for offset in (-1, 0, 1):
        candidate_date = (parsed + timedelta(days=offset)).date().isoformat()
        matches.extend(
            row
            for row in index.get(
                (league_code, candidate_date, home_goals, away_goals),
                [],
            )
            if int(row["home_team_id"]) == home_team_id
            and int(row["away_team_id"]) == away_team_id
        )
    return matches[0] if len(matches) == 1 else None


def _team_date_index(
    canonical: dict[tuple[str, str, int, int], list[dict[str, Any]]],
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    result: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for rows in canonical.values():
        for row in rows:
            match_id = str(row["match_id"])
            if match_id in seen:
                continue
            seen.add(match_id)
            result[
                (
                    str(row["league_code"]),
                    str(row["match_date"]),
                    int(row["home_team_id"]),
                    int(row["away_team_id"]),
                )
            ].append(row)
    return result


def _resolve_by_team_date(
    index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    *,
    league_code: str,
    date: str,
    home_team_id: int,
    away_team_id: int,
) -> dict[str, Any] | None:
    parsed = datetime.fromisoformat(date)
    matches: list[dict[str, Any]] = []
    for offset in (-1, 0, 1):
        candidate_date = (parsed + timedelta(days=offset)).date().isoformat()
        matches.extend(
            index.get(
                (league_code, candidate_date, home_team_id, away_team_id),
                [],
            )
        )
    return matches[0] if len(matches) == 1 else None


def _lineup(
    *,
    club_id: int,
    club_name: str,
    formation: Any,
    players: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "team": {"id": club_id, "name": club_name},
        "formation": None if pd.isna(formation) else str(formation),
        "startXI": [
            {
                "player": {
                    "id": int(player["player_id"]),
                    "name": str(player["player_name"]),
                    "pos": _role(str(player["position"])),
                    "grid": (
                        f"reconstructed:{player['position']}"
                        if player.get("reconstructed")
                        else f"reported:{player['position']}"
                    ),
                    "number": _shirt_number(player.get("number")),
                }
            }
            for player in players
        ],
        "substitutes": [],
    }


def _reconstructed_starters(
    *,
    base: Path,
    game_ids: set[int],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    appearances_path = base / "appearances.csv"
    events_path = base / "game_events.csv"
    players_path = base / "players.csv"
    if not all(path.exists() for path in (appearances_path, events_path, players_path)):
        return {}
    appearances: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        appearances_path,
        chunksize=500_000,
        low_memory=False,
    ):
        selected = chunk[chunk["game_id"].isin(game_ids)]
        if not selected.empty:
            appearances.append(selected)
    if not appearances:
        return {}
    appearance_rows = pd.concat(appearances, ignore_index=True)
    substitution_rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(events_path, chunksize=500_000, low_memory=False):
        selected = chunk[
            chunk["game_id"].isin(game_ids) & chunk["type"].eq("Substitutions")
        ]
        if not selected.empty:
            substitution_rows.append(selected)
    substitutions = (
        pd.concat(substitution_rows, ignore_index=True)
        if substitution_rows
        else pd.DataFrame(columns=["game_id", "club_id", "player_in_id"])
    )
    player_ids = {int(value) for value in appearance_rows["player_id"]}
    players = pd.read_csv(
        players_path,
        usecols=["player_id", "sub_position"],
        low_memory=False,
    )
    positions = {
        int(row["player_id"]): str(row["sub_position"])
        for row in players[players["player_id"].isin(player_ids)].to_dict("records")
        if not pd.isna(row.get("sub_position"))
        and str(row.get("sub_position") or "").strip()
    }
    result: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for (game_id, club_id), group in appearance_rows.groupby(
        ["game_id", "player_club_id"]
    ):
        game_id_value = int(str(game_id))
        club_id_value = int(str(club_id))
        subbed_in = {
            int(value)
            for value in substitutions[
                substitutions["game_id"].eq(game_id_value)
                & substitutions["club_id"].eq(club_id_value)
            ]["player_in_id"].dropna()
        }
        starters = group[~group["player_id"].isin(subbed_in)]
        if len(starters) != 11:
            continue
        records = starters.to_dict("records")
        if any(int(row["player_id"]) not in positions for row in records):
            continue
        result[(game_id_value, club_id_value)] = [
            {
                **{str(key): value for key, value in row.items()},
                "position": positions[int(row["player_id"])],
                "number": None,
                "reconstructed": True,
            }
            for row in records
        ]
    return result


def import_transfermarkt_open_data(
    project_dir: Path,
    *,
    games_path: Path | None = None,
    lineups_path: Path | None = None,
    database_path: Path | None = None,
    seasons: tuple[int, ...] = DEFAULT_SEASONS,
) -> TransfermarktOpenImportResult:
    """Import complete XIs for uniquely reconciled top-five fixtures."""
    base = project_dir / "data" / "raw" / "external" / "transfermarkt-player-scores"
    games_source = games_path or base / "games.csv"
    lineups_source = lineups_path or base / "game_lineups.csv"
    if not games_source.exists() or not lineups_source.exists():
        raise FileNotFoundError(
            f"Dataset Transfermarkt incompleto: {games_source}, {lineups_source}"
        )
    games = pd.read_csv(games_source, low_memory=False)
    games = games[
        games["competition_id"].isin(COMPETITIONS) & games["season"].isin(seasons)
    ].copy()
    game_ids = {int(value) for value in games["game_id"]}
    starters: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for chunk in pd.read_csv(lineups_source, chunksize=500_000, low_memory=False):
        selected = chunk[
            chunk["game_id"].isin(game_ids) & chunk["type"].eq("starting_lineup")
        ]
        for row in selected.to_dict("records"):
            starters[(int(row["game_id"]), int(row["club_id"]))].append(row)
    incomplete_game_ids = {
        int(game["game_id"])
        for game in games.to_dict("records")
        if len(
            starters.get(
                (int(game["game_id"]), int(game["home_club_id"])),
                [],
            )
        )
        != 11
        or len(
            starters.get(
                (int(game["game_id"]), int(game["away_club_id"])),
                [],
            )
        )
        != 11
    }
    reconstructed = _reconstructed_starters(
        base=base,
        game_ids=incomplete_game_ids,
    )

    database = ResearchDatabase(
        database_path or project_dir / "data" / "football_odds.sqlite3"
    )
    database.initialize()
    canonical = _canonical_index(database)
    canonical_by_team_date = _team_date_index(canonical)
    imported = already = incomplete = unresolved = observations = discrepancies = 0
    reconstructed_matches = 0
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
        team_mappings = {
            str(row["provider_team_id"]): int(row["internal_team_id"])
            for row in connection.execute(
                """
                SELECT provider_team_id, internal_team_id
                FROM provider_team_mapping
                WHERE provider_id=?
                """,
                (provider_id,),
            )
        }
        for game in games.to_dict("records"):
            fixture_id = str(int(game["game_id"]))
            if fixture_id in existing:
                already += 1
                continue
            home_club_id = int(game["home_club_id"])
            away_club_id = int(game["away_club_id"])
            home_players = starters.get((int(game["game_id"]), home_club_id), [])
            away_players = starters.get((int(game["game_id"]), away_club_id), [])
            used_reconstruction = False
            if len(home_players) != 11:
                home_players = reconstructed.get(
                    (int(game["game_id"]), home_club_id),
                    home_players,
                )
                used_reconstruction = len(home_players) == 11
            if len(away_players) != 11:
                away_players = reconstructed.get(
                    (int(game["game_id"]), away_club_id),
                    away_players,
                )
                used_reconstruction = used_reconstruction or len(away_players) == 11
            all_players = [*home_players, *away_players]
            if (
                len(home_players) != 11
                or len(away_players) != 11
                or len({int(row["player_id"]) for row in all_players}) != 22
                or any(
                    not str(row.get("position") or "").strip() for row in all_players
                )
            ):
                incomplete += 1
                continue
            league_code = COMPETITIONS[str(game["competition_id"])]
            mapped_home = team_mappings.get(str(home_club_id))
            mapped_away = team_mappings.get(str(away_club_id))
            match = (
                _resolve_by_team_ids(
                    canonical,
                    league_code=league_code,
                    date=str(game["date"]),
                    home_goals=int(game["home_club_goals"]),
                    away_goals=int(game["away_club_goals"]),
                    home_team_id=mapped_home,
                    away_team_id=mapped_away,
                )
                if mapped_home is not None and mapped_away is not None
                else None
            )
            if match is None and mapped_home is not None and mapped_away is not None:
                match = _resolve_by_team_date(
                    canonical_by_team_date,
                    league_code=league_code,
                    date=str(game["date"]),
                    home_team_id=mapped_home,
                    away_team_id=mapped_away,
                )
                discrepancies += int(match is not None)
            if match is None:
                match = _resolve_match(
                    canonical,
                    league_code=league_code,
                    date=str(game["date"]),
                    home_name=str(game["home_club_name"]),
                    away_name=str(game["away_club_name"]),
                    home_goals=int(game["home_club_goals"]),
                    away_goals=int(game["away_club_goals"]),
                )
            if match is None:
                unresolved += 1
                continue
            connection.execute(
                "INSERT INTO provider_match_mapping VALUES (?, ?, ?)",
                (provider_id, fixture_id, str(match["match_id"])),
            )
            for club_id, club_name, team_id in (
                (
                    home_club_id,
                    str(game["home_club_name"]),
                    int(match["home_team_id"]),
                ),
                (
                    away_club_id,
                    str(game["away_club_name"]),
                    int(match["away_team_id"]),
                ),
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
                        str(club_id),
                        team_id,
                        club_name,
                        observed_at,
                    ),
                )
                team_mappings[str(club_id)] = team_id
            ingest_fixture_lineups(
                database,
                provider_fixture_id=fixture_id,
                lineups=[
                    _lineup(
                        club_id=home_club_id,
                        club_name=str(game["home_club_name"]),
                        formation=game.get("home_club_formation"),
                        players=home_players,
                    ),
                    _lineup(
                        club_id=away_club_id,
                        club_name=str(game["away_club_name"]),
                        formation=game.get("away_club_formation"),
                        players=away_players,
                    ),
                ],
                observed_at=observed_at,
                provider_name=PROVIDER,
                player_namespace=PLAYER_NAMESPACE,
            )
            imported += 1
            observations += 22
            reconstructed_matches += int(used_reconstruction)
    return TransfermarktOpenImportResult(
        matches_seen=len(games),
        matches_imported=imported,
        already_imported=already,
        incomplete_lineups=incomplete,
        reconciliation_unresolved=unresolved,
        player_observations=observations,
        result_discrepancies=discrepancies,
        lineups_reconstructed=reconstructed_matches,
    )
