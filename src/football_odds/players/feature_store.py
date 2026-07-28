"""Canonical nullable observations and leakage-safe player-match features."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ..core.providers import TRANSFERMARKT_PROVIDER
from ..data.repository import ResearchDatabase
from .timestamps import utc_instants


@dataclass(frozen=True)
class FeatureStoreResult:
    observations_upserted: int
    conflicts: int
    rows: int
    outputs: dict[str, Path]


def _position(value: object) -> str | None:
    text = str(value or "").casefold()
    if "keeper" in text:
        return "G"
    if any(word in text for word in ("back", "defender", "sweeper")):
        return "D"
    if "midfield" in text:
        return "M"
    if any(word in text for word in ("winger", "forward", "striker")):
        return "F"
    return None


def _upsert_transfermarkt(
    database: ResearchDatabase, base: Path, quarantine: list[dict[str, Any]]
) -> int:
    appearances = base / "appearances.csv"
    events = base / "game_events.csv"
    lineups = base / "game_lineups.csv"
    if not all(path.exists() for path in (appearances, events, lineups)):
        return 0
    with database.connect() as connection:
        provider_id = connection.execute(
            "SELECT provider_id FROM providers WHERE provider_name=?",
            (TRANSFERMARKT_PROVIDER,),
        ).fetchone()
        if provider_id is None:
            return 0
        pid = int(provider_id[0])
        mappings = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                """SELECT provider_match_id, internal_match_id
                   FROM provider_match_mapping WHERE provider_id=?""",
                (pid,),
            )
        }
        players = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                """SELECT provider_player_id, internal_player_id
                   FROM provider_player_mapping WHERE provider_id=?""",
                (pid,),
            )
        }
        teams = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                """SELECT provider_team_id, internal_team_id
                   FROM provider_team_mapping WHERE provider_id=?""",
                (pid,),
            )
        }
    game_ids = {int(value) for value in mappings}
    facts: dict[tuple[int, int], dict[str, Any]] = {}
    for chunk in pd.read_csv(lineups, chunksize=250_000, low_memory=False):
        for row in chunk[chunk["game_id"].isin(game_ids)].to_dict("records"):
            key = (int(row["game_id"]), int(row["player_id"]))
            facts[key] = {
                "club_id": int(row["club_id"]),
                "lineup_role": (
                    "starter" if row["type"] == "starting_lineup" else "bench"
                ),
                "position_original": row.get("position"),
                "formation_grid": None,
                "shirt_number": row.get("number"),
                "bench_available": 1,
            }
    for chunk in pd.read_csv(appearances, chunksize=250_000, low_memory=False):
        for row in chunk[chunk["game_id"].isin(game_ids)].to_dict("records"):
            key = (int(row["game_id"]), int(row["player_id"]))
            fact = facts.setdefault(
                key,
                {
                    "club_id": int(row["player_club_id"]),
                    "lineup_role": None,
                    "position_original": None,
                    "formation_grid": None,
                    "shirt_number": None,
                    "bench_available": None,
                },
            )
            fact["minutes_played"] = float(row["minutes_played"])
            fact["statistics"] = {
                name: row.get(name)
                for name in ("goals", "assists", "yellow_cards", "red_cards")
                if not pd.isna(row.get(name))
            }
    for chunk in pd.read_csv(events, chunksize=250_000, low_memory=False):
        selected = chunk[
            chunk["game_id"].isin(game_ids) & chunk["type"].eq("Substitutions")
        ]
        for row in selected.to_dict("records"):
            minute = float(row["minute"])
            outgoing = (int(row["game_id"]), int(row["player_id"]))
            incoming_id = row.get("player_in_id")
            if outgoing in facts:
                facts[outgoing]["minute_out"] = minute
                facts[outgoing]["substitution_exit"] = 1
            if not pd.isna(incoming_id):
                incoming = (int(row["game_id"]), int(incoming_id))
                if incoming in facts:
                    facts[incoming]["minute_in"] = minute
                    facts[incoming]["substitution_entry"] = 1
    acquired = datetime.now(timezone.utc).isoformat()
    rows: list[tuple[Any, ...]] = []
    for (game_id, external_player), fact in facts.items():
        player_id = players.get(str(external_player))
        team_id = teams.get(str(fact["club_id"]))
        if player_id is None or team_id is None:
            continue
        minute_in = fact.get("minute_in")
        minute_out = fact.get("minute_out")
        minutes = fact.get("minutes_played")
        if minute_in is not None and minute_out is not None and minute_in > minute_out:
            quarantine.append(
                {
                    "provider": TRANSFERMARKT_PROVIDER,
                    "game_id": game_id,
                    "player_id": external_player,
                    "reason": "minute_in_after_minute_out",
                }
            )
            continue
        rows.append(
            (
                mappings[str(game_id)],
                team_id,
                player_id,
                pid,
                str(game_id),
                str(external_player),
                fact.get("lineup_role"),
                fact.get("position_original"),
                _position(fact.get("position_original")),
                fact.get("formation_grid"),
                fact.get("shirt_number"),
                fact.get("bench_available"),
                minutes,
                minute_in,
                minute_out,
                fact.get("substitution_entry"),
                fact.get("substitution_exit"),
                json.dumps(fact.get("statistics"), ensure_ascii=False)
                if fact.get("statistics")
                else None,
                acquired,
                "reported",
                f"{game_id}:{external_player}",
            )
        )
    with database.batch() as connection:
        connection.executemany(
            """INSERT INTO player_match_observations (
              match_id,team_id,player_id,provider_id,provider_match_id,
              provider_player_id,lineup_role,position_original,position_normalized,
              formation_grid,shirt_number,bench_available,minutes_played,minute_in,
              minute_out,substitution_entry,substitution_exit,player_statistics_json,
              acquired_at,quality,source_record_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(match_id,player_id,provider_id) DO UPDATE SET
              team_id=excluded.team_id, lineup_role=excluded.lineup_role,
              position_original=excluded.position_original,
              position_normalized=excluded.position_normalized,
              shirt_number=excluded.shirt_number,
              bench_available=excluded.bench_available,
              minutes_played=excluded.minutes_played,
              minute_in=excluded.minute_in, minute_out=excluded.minute_out,
              substitution_entry=excluded.substitution_entry,
              substitution_exit=excluded.substitution_exit,
              player_statistics_json=excluded.player_statistics_json,
              acquired_at=excluded.acquired_at, quality=excluded.quality""",
            rows,
        )
    return len(rows)


def build_temporal_player_matrix(observations: pd.DataFrame) -> pd.DataFrame:
    """Snapshot player histories before updating every kickoff-time batch."""
    required = {
        "match_id",
        "kickoff",
        "league",
        "team_id",
        "player_id",
        "lineup_role",
        "minutes_played",
        "minute_in",
        "minute_out",
        "position_original",
        "position_normalized",
        "quality",
        "source",
    }
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Missing temporal observation fields: {sorted(missing)}")
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    ordered = observations.assign(
        _kickoff_instant=utc_instants(observations["kickoff"])
    ).sort_values(["_kickoff_instant", "match_id", "player_id"])
    for _, simultaneous in ordered.groupby("_kickoff_instant", sort=False):
        for row in simultaneous.to_dict("records"):
            prior = history[str(row["player_id"])]
            starts = sum(item["lineup_role"] == "starter" for item in prior)
            benches = sum(item["lineup_role"] == "bench" for item in prior)
            real_minutes = [
                item["minutes_played"]
                for item in prior
                if pd.notna(item["minutes_played"])
            ]
            entry = [item["minute_in"] for item in prior if pd.notna(item["minute_in"])]
            exit_ = [
                item["minute_out"] for item in prior if pd.notna(item["minute_out"])
            ]
            positions = [
                item["position_normalized"]
                for item in prior
                if pd.notna(item["position_normalized"])
            ]
            dominant_position = (
                max(set(positions), key=positions.count) if positions else None
            )
            fallback_minutes = [
                90.0 if item["lineup_role"] == "starter" else None for item in prior
            ]
            minutes = real_minutes or [x for x in fallback_minutes if x is not None]
            last = prior[-1] if prior else None
            features: dict[str, tuple[Any, bool, str, str, str]] = {
                "observations": (
                    len(prior),
                    bool(prior),
                    "reported",
                    "canonical",
                    "none",
                ),
                "starts": (starts, bool(prior), "reported", "canonical", "none"),
                "benches": (benches, bool(prior), "reported", "canonical", "none"),
                "start_rate": (
                    starts / len(prior) if prior else None,
                    bool(prior),
                    "derived",
                    "canonical",
                    "none",
                ),
                "bench_rate": (
                    benches / len(prior) if prior else None,
                    bool(prior),
                    "derived",
                    "canonical",
                    "none",
                ),
                "sub_entry_rate": (
                    sum(pd.notna(item["minute_in"]) for item in prior) / len(prior)
                    if prior
                    else None,
                    bool(prior),
                    "derived",
                    "canonical",
                    "unknown_without_timing",
                ),
                "mean_minutes": (
                    sum(minutes) / len(minutes) if minutes else None,
                    bool(minutes),
                    "reported" if real_minutes else "proxy",
                    "canonical",
                    "none" if real_minutes else "starter_90",
                ),
                "recent_minutes": (
                    sum(minutes[-5:]) / len(minutes[-5:]) if minutes else None,
                    bool(minutes),
                    "reported" if real_minutes else "proxy",
                    "canonical",
                    "none" if real_minutes else "starter_90",
                ),
                "mean_minute_in": (
                    sum(entry) / len(entry) if entry else None,
                    bool(entry),
                    "reported",
                    "canonical",
                    "unknown_without_timing",
                ),
                "mean_minute_out": (
                    sum(exit_) / len(exit_) if exit_ else None,
                    bool(exit_),
                    "reported",
                    "canonical",
                    "unknown_without_timing",
                ),
                "days_since_appearance": (
                    (
                        pd.Timestamp(row["kickoff"]) - pd.Timestamp(last["kickoff"])
                    ).total_seconds()
                    / 86_400
                    if last
                    else None,
                    bool(last),
                    "derived",
                    "canonical",
                    "none",
                ),
                "team_experience": (
                    sum(item["team_id"] == row["team_id"] for item in prior),
                    bool(prior),
                    "observed_interval",
                    "canonical",
                    "not_contract",
                ),
                "league_experience": (
                    sum(item["league"] == row["league"] for item in prior),
                    bool(prior),
                    "observed",
                    "canonical",
                    "none",
                ),
                "team_change": (
                    bool(last and last["team_id"] != row["team_id"]),
                    bool(prior),
                    "observed_interval",
                    "canonical",
                    "not_contract",
                ),
                "position_original": (
                    last["position_original"] if last else None,
                    bool(last),
                    "reported",
                    last["source"] if last else "none",
                    "unknown",
                ),
                "position_normalized": (
                    last["position_normalized"] if last else None,
                    bool(last),
                    "normalized",
                    last["source"] if last else "none",
                    "department",
                ),
                "role_stability": (
                    positions.count(dominant_position) / len(positions)
                    if positions
                    else None,
                    bool(positions),
                    "derived",
                    "canonical",
                    "department",
                ),
                "player_statistics_observations": (
                    sum(pd.notna(item.get("player_statistics_json")) for item in prior),
                    bool(prior),
                    "reported",
                    "canonical",
                    "none",
                ),
            }
            result = {
                "match_id": row["match_id"],
                "player_id": row["player_id"],
                "team_id": row["team_id"],
                "kickoff": row["kickoff"],
                "current_lineup_role": row["lineup_role"],
                "current_position_original": row["position_original"],
                "current_position_normalized": row["position_normalized"],
                "current_observation_quality": row["quality"],
                "current_source": row["source"],
            }
            for name, (value, available, quality, source, fallback) in features.items():
                result[f"{name}_value"] = value
                result[f"{name}_available"] = available
                result[f"{name}_quality"] = quality
                result[f"{name}_source"] = source
                result[f"{name}_fallback_kind"] = fallback
            output.append({str(key): value for key, value in result.items()})
        for row in simultaneous.to_dict("records"):
            history[str(row["player_id"])].append(cast(dict[str, Any], row))
    return pd.DataFrame(output)


def build_player_feature_store(
    project_dir: Path, *, database_path: Path | None = None
) -> FeatureStoreResult:
    database = ResearchDatabase(
        database_path or project_dir / "data/football_odds.sqlite3"
    )
    database.initialize()
    quarantine: list[dict[str, Any]] = []
    before = _coverage(database)
    upserted = _upsert_transfermarkt(
        database,
        project_dir / "data/raw/external/transfermarkt-player-scores",
        quarantine,
    )
    after = _coverage(database)
    with database.connect() as connection:
        observations = pd.read_sql_query(
            """SELECT o.*, m.date AS kickoff, l.league_code AS league,
                      p.provider_name AS source
               FROM player_match_observations o JOIN matches m USING(match_id)
               JOIN leagues l USING(league_id) JOIN providers p USING(provider_id)""",
            connection,
        )
    matrix = (
        build_temporal_player_matrix(observations)
        if not observations.empty
        else pd.DataFrame()
    )
    out = project_dir / "reports/player_data/feature_store"
    out.mkdir(parents=True, exist_ok=True)
    matrix_path = project_dir / "data/processed/player_match_temporal_features.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_path, index=False)
    coverage_path = out / "coverage_before_after.json"
    coverage_path.write_text(
        json.dumps({"before": before, "after": after}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    census_path = project_dir / "reports/player_data/dataset/player_feature_census.json"
    census_path.parent.mkdir(parents=True, exist_ok=True)
    census_path.write_text(
        json.dumps(
            {
                "canonical_player_match_observations": after,
                "temporal_feature_rows": len(matrix),
                "conflicts_quarantined": len(quarantine),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    quarantine_path = out / "quarantine.jsonl"
    quarantine_path.write_text(
        "".join(json.dumps(x) + "\n" for x in quarantine), encoding="utf-8"
    )
    fallback_path = out / "fallbacks.json"
    fallback_path.write_text(
        json.dumps(
            {
                "minutes": "reported_else_starter_90; bench_without_timing_unknown",
                "shared_minutes": "reported_overlap_else_co_start_proxy",
                "position": "original_else_normalized_else_department",
                "substitution": "reported_timing_else_unknown",
                "membership": "observed_interval_not_contract",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = out / "FEATURE_STORE_REPORT.md"
    report_path.write_text(
        f"# Player feature store\n\n- Canonical observations: **{after['rows']}**\n"
        f"- Temporal rows: **{len(matrix)}**\n- Conflicts quarantined: "
        f"**{len(quarantine)}**\n",
        encoding="utf-8",
    )
    return FeatureStoreResult(
        upserted,
        len(quarantine),
        len(matrix),
        {
            "matrix": matrix_path,
            "coverage": coverage_path,
            "quarantine": quarantine_path,
            "fallbacks": fallback_path,
            "report": report_path,
        },
    )


def _coverage(database: ResearchDatabase) -> dict[str, Any]:
    fields = (
        "lineup_role",
        "position_original",
        "position_normalized",
        "formation_grid",
        "shirt_number",
        "bench_available",
        "minutes_played",
        "minute_in",
        "minute_out",
        "substitution_entry",
        "substitution_exit",
        "player_statistics_json",
    )
    with database.connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS rows, "
            + ", ".join(f"SUM({field} IS NOT NULL) AS {field}" for field in fields)
            + " FROM player_match_observations"
        ).fetchone()
        providers = [
            dict(item)
            for item in connection.execute(
                """SELECT p.provider_name AS provider, l.league_code AS league,
                      m.season, COUNT(*) AS rows,
                      SUM(o.minutes_played IS NOT NULL) AS minutes_played,
                      SUM(o.minute_in IS NOT NULL) AS minute_in,
                      SUM(o.minute_out IS NOT NULL) AS minute_out
               FROM player_match_observations o JOIN providers p USING(provider_id)
               JOIN matches m USING(match_id) JOIN leagues l USING(league_id)
               GROUP BY p.provider_name, l.league_code, m.season
               ORDER BY p.provider_name, l.league_code, m.season"""
            )
        ]
    result = dict(row)
    result["by_provider"] = providers
    return result
