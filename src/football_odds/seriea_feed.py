"""Read-only audit client for the public Lega Serie A SDP JSON feed."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .database import ResearchDatabase
from .player_reconciliation import normalize_team_name

BASE_URL = "https://api-sdp.legaseriea.it/v1/serie-a/football"
COMPETITION_ID = (
    "serie-a::Football_Competition::ec93b94f74294dc98ab5bcfd67fc0d88"
)
DEFAULT_SEASONS = ("2022/2023", "2023/2024", "2024/2025")
PROVIDER = "Lega Serie A SDP"
PLAYER_NAMESPACE = uuid.UUID("543f76ad-f31d-4510-aa47-8da0922a7b22")


class SerieAFeedClient:
    """Small throttled client; the upstream feed is public but undocumented."""

    def __init__(
        self,
        *,
        request: Callable[..., Any] = requests.get,
        minimum_interval: float = 0.25,
    ) -> None:
        self.request = request
        self.minimum_interval = minimum_interval
        self._last_request_at: float | None = None
        self.requests_made = 0

    def get(self, path: str) -> dict[str, Any]:
        if self._last_request_at is not None and self.minimum_interval > 0:
            wait = self.minimum_interval - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
        response = self.request(
            f"{BASE_URL}/{path.lstrip('/')}",
            params={"locale": "en-GB"},
            headers={
                "User-Agent": "Sports-Predictions-Lab/0.1",
                "Accept": "application/json",
            },
            timeout=30,
        )
        self._last_request_at = time.monotonic()
        self.requests_made += 1
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Il feed Lega Serie A non ha restituito un oggetto JSON.")
        return payload

    def seasons(self) -> list[dict[str, Any]]:
        payload = self.get(f"competitions/{COMPETITION_ID}/seasons")
        return list(payload.get("seasons") or [])

    def matches(self, season_id: str) -> list[dict[str, Any]]:
        return list(self.get(f"seasons/{season_id}/matches").get("matches") or [])

    def lineup(self, season_id: str, match_id: str) -> dict[str, Any]:
        return self.get(f"seasons/{season_id}/matches/{match_id}/lineups")


@dataclass(frozen=True)
class SerieAFeedAuditResult:
    summary: pd.DataFrame
    matches: pd.DataFrame
    players: pd.DataFrame
    outputs: dict[str, Path]
    requests_made: int


@dataclass(frozen=True)
class SerieAFeedBackfillResult:
    feed_matches: int
    mapped_matches: int
    already_complete: int
    imported_matches: int
    unresolved: tuple[dict[str, str], ...]
    requests_made: int


def _sample_positions(size: int, count: int) -> list[int]:
    if size <= count:
        return list(range(size))
    if count == 1:
        return [size // 2]
    return sorted(
        {round(index * (size - 1) / (count - 1)) for index in range(count)}
    )


def _allocate_samples(total: int, groups: int) -> list[int]:
    base, remainder = divmod(total, groups)
    return [base + (index < remainder) for index in range(groups)]


def _event(player: dict[str, Any], event_type: str) -> dict[str, Any] | None:
    return next(
        (
            event
            for event in player.get("events") or []
            if event.get("type") == event_type
        ),
        None,
    )


def _minute(event: dict[str, Any] | None) -> float | None:
    if event is None:
        return None
    return float(event.get("time") or 0) + float(event.get("additionalTime") or 0)


def _player_rows(
    lineup: dict[str, Any],
    *,
    season: str,
    match_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side in ("home", "away"):
        team = lineup.get(side) or {}
        for lineup_role, key in (("starter", "fielded"), ("bench", "benched")):
            for player in team.get(key) or []:
                substitution_in = _event(player, "substitution-in")
                substitution_out = _event(player, "substitution-out")
                player_name = (
                    player.get("displayName")
                    or " ".join(
                        filter(
                            None,
                            (
                                player.get("mediaFirstName"),
                                player.get("mediaLastName"),
                            ),
                        )
                    )
                    or player.get("shirtName")
                    or player.get("providerId")
                    or "unknown-player"
                )
                start = 0.0 if lineup_role == "starter" else _minute(substitution_in)
                end = _minute(substitution_out) or 90.0
                minutes = (
                    max(0.0, min(90.0, end) - min(90.0, start))
                    if start is not None
                    else 0.0
                )
                rows.append(
                    {
                        "season": season,
                        "match_id": match_id,
                        "side": side,
                        "team_id": team.get("teamId"),
                        "team": team.get("mediaName") or team.get("officialName"),
                        "player_id": (
                            player.get("playerId")
                            or player.get("providerId")
                            or (
                                f"missing::{team.get('teamId')}::"
                                f"{normalize_team_name(str(player_name))}"
                            )
                        ),
                        "player": player_name,
                        "role": player.get("roleLabel"),
                        "lineup_role": lineup_role,
                        "entered": substitution_in is not None,
                        "left": substitution_out is not None,
                        "minute_in": _minute(substitution_in),
                        "minute_out": _minute(substitution_out),
                        "minutes_estimated": minutes,
                    }
                )
    return rows


def audit_seriea_feed(
    client: SerieAFeedClient,
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    sample_matches: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Audit a time-distributed match sample without persisting model data."""
    if sample_matches < len(seasons):
        raise ValueError("Il campione deve includere almeno una partita per stagione.")
    catalogue = {
        str(item["seasonName"]): str(item["seasonId"])
        for item in client.seasons()
        if item.get("seasonName") and item.get("seasonId")
    }
    missing = set(seasons).difference(catalogue)
    if missing:
        raise ValueError(f"Stagioni Lega Serie A non trovate: {sorted(missing)}")

    allocations = _allocate_samples(sample_matches, len(seasons))
    match_rows: list[dict[str, object]] = []
    player_rows: list[dict[str, object]] = []
    raw: list[dict[str, Any]] = []
    for season, count in zip(seasons, allocations, strict=True):
        season_id = catalogue[season]
        matches = sorted(
            client.matches(season_id),
            key=lambda item: (str(item.get("matchDateUtc")), str(item.get("matchId"))),
        )
        for position in _sample_positions(len(matches), count):
            match = matches[position]
            match_id = str(match["matchId"])
            lineup = client.lineup(season_id, match_id)
            raw.append({"season": season, "match": match, "lineup": lineup})
            home = lineup.get("home") or {}
            away = lineup.get("away") or {}
            rows = _player_rows(lineup, season=season, match_id=match_id)
            player_rows.extend(rows)
            match_rows.append(
                {
                    "season": season,
                    "match_id": match_id,
                    "date": match.get("matchDateUtc"),
                    "home": home.get("mediaName") or home.get("officialName"),
                    "away": away.get("mediaName") or away.get("officialName"),
                    "home_starters": len(home.get("fielded") or []),
                    "away_starters": len(away.get("fielded") or []),
                    "home_bench": len(home.get("benched") or []),
                    "away_bench": len(away.get("benched") or []),
                    "formation_available": bool(
                        home.get("tacticalFormation") and away.get("tacticalFormation")
                    ),
                }
            )

    matches_frame = pd.DataFrame(match_rows)
    players_frame = pd.DataFrame(player_rows)
    summaries: list[dict[str, object]] = []
    for season in seasons:
        match_mask = matches_frame["season"].eq(season)
        player_mask = players_frame["season"].eq(season)
        season_matches = matches_frame.loc[match_mask]
        season_players = players_frame.loc[player_mask]
        summaries.append(
            {
                "season": season,
                "sampled_matches": len(season_matches),
                "complete_starting_xi_rate": float(
                    (
                        season_matches["home_starters"].eq(11)
                        & season_matches["away_starters"].eq(11)
                    ).mean()
                ),
                "bench_rate": float(
                    (
                        season_matches["home_bench"].gt(0)
                        & season_matches["away_bench"].gt(0)
                    ).mean()
                ),
                "formation_rate": float(
                    season_matches["formation_available"].mean()
                ),
                "player_id_rate": float(season_players["player_id"].notna().mean()),
                "role_rate": float(season_players["role"].notna().mean()),
                "substitutions": int(season_players["entered"].sum()),
            }
        )
    return pd.DataFrame(summaries), matches_frame, players_frame, raw


def export_seriea_feed_audit(
    project_dir: Path,
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    sample_matches: int = 10,
    client: SerieAFeedClient | None = None,
) -> SerieAFeedAuditResult:
    """Run and persist the public-feed pilot, including its raw evidence."""
    active = client or SerieAFeedClient()
    summary, matches, players, raw = audit_seriea_feed(
        active,
        seasons=seasons,
        sample_matches=sample_matches,
    )
    destination = project_dir / "reports" / "player_data" / "seriea_feed"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": destination / "seriea_feed_summary.csv",
        "matches": destination / "seriea_feed_matches.csv",
        "players": destination / "seriea_feed_players.csv",
        "raw": destination / "seriea_feed_sample.json",
        "metadata": destination / "seriea_feed.meta.json",
        "report": destination / "SERIEA_FEED_AUDIT.md",
    }
    summary.to_csv(outputs["summary"], index=False)
    matches.to_csv(outputs["matches"], index=False)
    players.to_csv(outputs["players"], index=False)
    outputs["raw"].write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    outputs["metadata"].write_text(
        json.dumps(
            {
                "provider": "Lega Serie A public SDP feed",
                "seasons": list(seasons),
                "sample_matches": sample_matches,
                "requests_made": active.requests_made,
                "modeling_data_changed": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    outputs["report"].write_text(
        "# Audit feed pubblico Lega Serie A\n\n"
        f"Campione: {len(matches)} partite; richieste: {active.requests_made}.\n\n"
        "| Stagione | Match | XI completi | Panchina | Formazione | "
        "ID | Ruolo | Subentri |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row.season} | {int(row.sampled_matches)} | "
            f"{row.complete_starting_xi_rate:.0%} | {row.bench_rate:.0%} | "
            f"{row.formation_rate:.0%} | {row.player_id_rate:.0%} | "
            f"{row.role_rate:.0%} | {int(row.substitutions)} |"
            for row in summary.itertuples(index=False)
        )
        + "\n\nI minuti sono ricostruiti dagli eventi substitution-in/out e limitati "
        "ai 90 minuti regolamentari. Il feed resta non documentato e non modifica "
        "il dataset modellistico durante il pilot.\n",
        encoding="utf-8",
    )
    return SerieAFeedAuditResult(
        summary, matches, players, outputs, active.requests_made
    )


def _canonical_matches(database: ResearchDatabase) -> dict[tuple[str, str, str], str]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT m.match_id, substr(m.date, 1, 10), ht.team_name, at.team_name
            FROM matches m
            JOIN leagues l ON l.league_id=m.league_id
            JOIN teams ht ON ht.team_id=m.home_team_id
            JOIN teams at ON at.team_id=m.away_team_id
            WHERE l.league_code='I1'
            """
        ).fetchall()
    return {
        (
            str(row[1]),
            normalize_team_name(str(row[2])),
            normalize_team_name(str(row[3])),
        ): str(row[0])
        for row in rows
    }


def _feed_team(match: dict[str, Any], side: str) -> tuple[str, str]:
    team = match.get(side) or {}
    return (
        str(team.get("teamId") or ""),
        str(team.get("mediaName") or team.get("officialName") or ""),
    )


def _persist_lineup(
    database: ResearchDatabase,
    *,
    match_id: str,
    lineup: dict[str, Any],
) -> None:
    with database.batch() as connection:
        provider_id = database._lookup_id(  # noqa: SLF001
            connection, "providers", "provider_id", "provider_name", PROVIDER
        )
        match_teams = connection.execute(
            """
            SELECT home_team_id, away_team_id, substr(date, 1, 10)
            FROM matches WHERE match_id=?
            """,
            (match_id,),
        ).fetchone()
        for side, team_id in (
            ("home", int(match_teams[0])),
            ("away", int(match_teams[1])),
        ):
            team = lineup[side]
            connection.execute(
                """
                INSERT INTO provider_team_mapping (
                    provider_id, provider_team_id, internal_team_id,
                    provider_team_name, mapping_method, observed_at
                ) VALUES (?, ?, ?, ?, 'normalized_exact', CURRENT_TIMESTAMP)
                ON CONFLICT(provider_id, provider_team_id) DO NOTHING
                """,
                (
                    provider_id,
                    str(team["teamId"]),
                    team_id,
                    team.get("mediaName") or team.get("officialName"),
                ),
            )
            connection.execute(
                """
                INSERT INTO fixture_lineups (
                    match_id, team_id, provider_id, formation, lineup_kind,
                    observed_at
                ) VALUES (?, ?, ?, ?, 'confirmed_historical', CURRENT_TIMESTAMP)
                ON CONFLICT(match_id, team_id, provider_id, lineup_kind)
                DO UPDATE SET formation=excluded.formation,
                              observed_at=excluded.observed_at
                """,
                (match_id, team_id, provider_id, team.get("tacticalFormation")),
            )
            lineup_id = connection.execute(
                """
                SELECT lineup_id FROM fixture_lineups
                WHERE match_id=? AND team_id=? AND provider_id=?
                  AND lineup_kind='confirmed_historical'
                """,
                (match_id, team_id, provider_id),
            ).fetchone()[0]
            connection.execute(
                "DELETE FROM lineup_players WHERE lineup_id=?", (lineup_id,)
            )
            connection.execute(
                """
                DELETE FROM player_match_lineup_stats
                WHERE match_id=? AND team_id=? AND provider_id=?
                """,
                (match_id, team_id, provider_id),
            )
            for row in _player_rows(
                {"home": team, "away": {}},
                season="",
                match_id=match_id,
            ):
                external_id = str(row["player_id"])
                player_id = str(
                    uuid.uuid5(PLAYER_NAMESPACE, f"{PROVIDER}:{external_id}")
                )
                player = next(
                    item
                    for key in ("fielded", "benched")
                    for item in team.get(key) or []
                    if str(item.get("playerId")) == external_id
                )
                connection.execute(
                    """
                    INSERT INTO players (player_id, player_name, nationality)
                    VALUES (?, ?, ?)
                    ON CONFLICT(player_id) DO UPDATE SET
                        player_name=excluded.player_name,
                        nationality=excluded.nationality
                    """,
                    (player_id, row["player"], player.get("nationalityIsoCode")),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO provider_player_mapping
                    VALUES (?, ?, ?)
                    """,
                    (provider_id, external_id, player_id),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO team_memberships (
                        player_id, team_id, provider_id, valid_from, observed_at
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (player_id, team_id, provider_id, str(match_teams[2])),
                )
                database_role = (
                    "starter" if row["lineup_role"] == "starter" else "substitute"
                )
                connection.execute(
                    """
                    INSERT INTO lineup_players (
                        lineup_id, player_id, lineup_role, position, shirt_number
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        lineup_id,
                        player_id,
                        database_role,
                        row["role"],
                        player.get("bibNumber"),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO player_match_lineup_stats
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_id,
                        team_id,
                        player_id,
                        provider_id,
                        row["lineup_role"],
                        row["role"],
                        row["minute_in"],
                        row["minute_out"],
                        row["minutes_estimated"],
                    ),
                )


def backfill_seriea_feed(
    project_dir: Path,
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    limit: int | None = None,
    client: SerieAFeedClient | None = None,
) -> SerieAFeedBackfillResult:
    """Map and import the complete feed, safely skipping completed matches."""
    active = client or SerieAFeedClient()
    database = ResearchDatabase(project_dir / "data" / "football_odds.sqlite3")
    database.initialize()
    canonical = _canonical_matches(database)
    catalogue = {
        str(item["seasonName"]): str(item["seasonId"])
        for item in active.seasons()
    }
    feed: list[tuple[str, dict[str, Any]]] = []
    for season in seasons:
        season_id = catalogue[season]
        feed.extend((season_id, match) for match in active.matches(season_id))
    unresolved: list[dict[str, str]] = []
    mapped: list[tuple[str, str, str]] = []
    for season_id, match in feed:
        _, home = _feed_team(match, "home")
        _, away = _feed_team(match, "away")
        key = (
            str(match.get("matchDateUtc"))[:10],
            normalize_team_name(home),
            normalize_team_name(away),
        )
        internal_id = canonical.get(key)
        if internal_id is None:
            unresolved.append(
                {"match_id": str(match.get("matchId")), "reason": "match_not_found"}
            )
            continue
        mapped.append((season_id, str(match["matchId"]), internal_id))
    with database.batch() as connection:
        provider_id = database._lookup_id(  # noqa: SLF001
            connection, "providers", "provider_id", "provider_name", PROVIDER
        )
        for _, feed_match_id, internal_id in mapped:
            connection.execute(
                """
                INSERT INTO provider_match_mapping
                (provider_id, provider_match_id, internal_match_id)
                VALUES (?, ?, ?)
                ON CONFLICT(provider_id, provider_match_id) DO NOTHING
                """,
                (provider_id, feed_match_id, internal_id),
            )
        complete = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT match_id FROM fixture_lineups
                WHERE provider_id=? AND lineup_kind='confirmed_historical'
                GROUP BY match_id HAVING COUNT(DISTINCT team_id)=2
                """,
                (provider_id,),
            )
        }
    pending = [item for item in mapped if item[2] not in complete]
    if limit is not None:
        pending = pending[:limit]
    for season_id, feed_match_id, internal_id in pending:
        _persist_lineup(
            database,
            match_id=internal_id,
            lineup=active.lineup(season_id, feed_match_id),
        )
    return SerieAFeedBackfillResult(
        feed_matches=len(feed),
        mapped_matches=len(mapped),
        already_complete=len(mapped) - len([x for x in mapped if x[2] not in complete]),
        imported_matches=len(pending),
        unresolved=tuple(unresolved),
        requests_made=active.requests_made,
    )
