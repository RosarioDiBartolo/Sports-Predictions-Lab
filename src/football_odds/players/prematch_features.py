"""Leakage-safe aggregate player features derived from historical lineups."""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from ..data.repository import ResearchDatabase
from .timestamps import utc_instants

PLAYER_FEATURE_NAMES = (
    "player_expected_strength",
    "player_expected_attack",
    "player_expected_defense",
    "player_bench_quality",
    "player_lineup_continuity",
    "player_lineup_uncertainty",
)

LINEUP_HISTORY_QUERY = """
WITH valid_lineups AS (
    SELECT
        fl.*,
        ROW_NUMBER() OVER (
            PARTITION BY fl.match_id, fl.team_id
            ORDER BY
                CASE p.provider_name
                    WHEN 'Lega Serie A SDP' THEN 0
                    WHEN 'API-Football' THEN 1
                    ELSE 2
                END,
                fl.lineup_id
        ) AS provider_rank
    FROM fixture_lineups fl
    JOIN providers p ON p.provider_id=fl.provider_id
    WHERE fl.lineup_kind='confirmed_historical'
      AND (
          SELECT COUNT(*)
          FROM lineup_players starters
          WHERE starters.lineup_id=fl.lineup_id
            AND starters.lineup_role='starter'
      )=11
)
SELECT
    fl.match_id,
    t.team_name AS team,
    lp.player_id,
    lp.lineup_role,
    lp.position
FROM valid_lineups fl
JOIN teams t ON t.team_id = fl.team_id
JOIN lineup_players lp ON lp.lineup_id = fl.lineup_id
WHERE fl.provider_rank=1
ORDER BY fl.match_id, t.team_name, lp.player_id
"""


def load_historical_lineups(database: ResearchDatabase) -> pd.DataFrame:
    """Load historical lineup facts without treating them as pre-match knowledge."""
    database.initialize()
    with database.connect() as connection:
        return pd.read_sql_query(LINEUP_HISTORY_QUERY, connection)


def _weighted_average(
    probabilities: dict[str, float],
    values: dict[str, float],
    *,
    limit: int | None = None,
) -> float:
    ordered = sorted(probabilities, key=probabilities.get, reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    weight = sum(probabilities[player] for player in ordered)
    if weight <= 0:
        return float("nan")
    return float(
        sum(probabilities[player] * values[player] for player in ordered) / weight
    )


def _team_snapshot(history: deque[dict[str, object]]) -> dict[str, float]:
    if not history:
        return {name: float("nan") for name in PLAYER_FEATURE_NAMES}

    players = sorted(
        {
            player
            for match in history
            for player in (
                set(match["starters"]) | set(match["bench"])  # type: ignore[arg-type]
            )
        }
    )
    denominator = float(len(history))
    starter_probability: dict[str, float] = {}
    bench_probability: dict[str, float] = {}
    strength: dict[str, float] = {}
    attack: dict[str, float] = {}
    defense: dict[str, float] = {}
    for player in players:
        starts = [
            match
            for match in history
            if player in match["starters"]  # type: ignore[operator]
        ]
        starter_probability[player] = len(starts) / denominator
        bench_probability[player] = (
            sum(player in match["bench"] for match in history) / denominator  # type: ignore[operator]
        )
        if starts:
            strength[player] = float(
                np.mean([float(match["points"]) / 3.0 for match in starts])
            )
            attack[player] = float(
                np.mean([float(match["goals_for"]) for match in starts])
            )
            defense[player] = float(
                -np.mean([float(match["goals_against"]) for match in starts])
            )
        else:
            strength[player] = attack[player] = defense[player] = 0.0

    previous_starters = set(history[-1]["starters"])  # type: ignore[arg-type]
    continuity = (
        sum(starter_probability.get(player, 0.0) for player in previous_starters)
        / len(previous_starters)
        if previous_starters
        else float("nan")
    )
    uncertainty = float(
        np.mean(
            [
                -(
                    probability * math.log(probability)
                    + (1.0 - probability) * math.log(1.0 - probability)
                )
                / math.log(2.0)
                if 0.0 < probability < 1.0
                else 0.0
                for probability in starter_probability.values()
            ]
        )
    )
    return {
        "player_expected_strength": _weighted_average(starter_probability, strength),
        "player_expected_attack": _weighted_average(starter_probability, attack),
        "player_expected_defense": _weighted_average(starter_probability, defense),
        "player_bench_quality": _weighted_average(bench_probability, strength, limit=5),
        "player_lineup_continuity": float(continuity),
        "player_lineup_uncertainty": uncertainty,
    }


def build_prematch_player_features(
    matches: pd.DataFrame,
    lineups: pd.DataFrame,
    *,
    lookback: int = 10,
) -> pd.DataFrame:
    """Create six home/away aggregates before observing each match's lineup."""
    if lookback < 1:
        raise ValueError("lookback deve essere almeno 1.")
    required_matches = {
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    }
    required_lineups = {"match_id", "team", "player_id", "lineup_role"}
    missing_matches = required_matches.difference(matches.columns)
    missing_lineups = required_lineups.difference(lineups.columns)
    if missing_matches or missing_lineups:
        raise ValueError(
            "Contratto player feature incompleto: "
            f"matches={sorted(missing_matches)}, lineups={sorted(missing_lineups)}"
        )

    facts: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"starters": set(), "bench": set()}
    )
    for row in lineups.itertuples(index=False):
        bucket = "starters" if row.lineup_role == "starter" else "bench"
        facts[(str(row.match_id), str(row.team))][bucket].add(str(row.player_id))

    histories: dict[str, deque[dict[str, object]]] = defaultdict(
        lambda: deque(maxlen=lookback)
    )
    output: list[dict[str, object]] = []
    ordered = matches.assign(_date_instant=utc_instants(matches["date"])).sort_values(
        ["_date_instant", "match_id"]
    )
    for _, simultaneous in ordered.groupby("_date_instant", sort=False):
        for row in simultaneous.itertuples(index=False):
            values: dict[str, object] = {"match_id": str(row.match_id)}
            for side in ("home", "away"):
                team = str(getattr(row, f"{side}_team"))
                snapshot = _team_snapshot(histories[team])
                values.update(
                    {f"{side}_{name}": value for name, value in snapshot.items()}
                )
            output.append(values)

        for row in simultaneous.itertuples(index=False):
            if (
                row.result not in ("H", "D", "A")
                or pd.isna(row.home_goals)
                or pd.isna(row.away_goals)
            ):
                continue
            for side, opponent in (("home", "away"), ("away", "home")):
                team = str(getattr(row, f"{side}_team"))
                lineup = facts.get((str(row.match_id), team))
                if not lineup or len(lineup["starters"]) != 11:
                    continue
                goals_for = int(getattr(row, f"{side}_goals"))
                goals_against = int(getattr(row, f"{opponent}_goals"))
                points = 3 if goals_for > goals_against else 0
                if goals_for == goals_against:
                    points = 1
                histories[team].append(
                    {
                        "starters": frozenset(lineup["starters"]),
                        "bench": frozenset(lineup["bench"]),
                        "points": points,
                        "goals_for": goals_for,
                        "goals_against": goals_against,
                    }
                )
    return pd.DataFrame(output)


def attach_prematch_player_features(
    matches: pd.DataFrame,
    base_features: pd.DataFrame,
    database: ResearchDatabase,
    *,
    lookback: int = 10,
) -> pd.DataFrame:
    """Attach chronological player aggregates to the canonical feature matrix."""
    player_features = build_prematch_player_features(
        matches,
        load_historical_lineups(database),
        lookback=lookback,
    )
    return base_features.merge(
        player_features,
        on="match_id",
        how="left",
        validate="one_to_one",
    )
