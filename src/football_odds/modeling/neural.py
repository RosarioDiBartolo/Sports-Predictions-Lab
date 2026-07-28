"""Shared neural player encoder with department pooling over confirmed lineups."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dixon_coles import fit_dixon_coles, score_probabilities
from .evaluation import (
    PROBABILITY_COLUMNS,
    metrics_by_season,
    paired_log_loss_bootstrap,
)

NEURAL_LINEUP_MODEL_NAME = "dixon_coles_shared_encoder_pooling"
DEPARTMENTS = ("goalkeeper", "defense", "midfield", "attack")
POOL_DEPARTMENTS = (*DEPARTMENTS, "unknown")
BASE_NEURAL_FEATURE_NAMES = (
    "career_strength",
    "career_goal_difference",
    "log_starts",
    "log_days_since",
    "observed",
    "recent_strength",
    "recent_goal_difference",
    "home_indicator",
    "career_goals_for",
    "career_goals_against",
    "recent_goals_for",
    "recent_goals_against",
    "opponent_strength",
    "team_experience_share",
    "league_experience_share",
    "role_stability",
    "lineup_familiarity",
    "log_squad_observations",
    "start_rate",
    "bench_rate",
    "bench_entry_rate",
    "average_minutes",
    "recent_minutes",
    "minute_in_mean",
    "minute_out_mean",
    "minutes_real_coverage",
    "original_position_quality",
    "shared_minutes",
    "shared_minutes_coverage",
)
TEMPORAL_NUMERIC_FIELDS = (
    "observations",
    "starts",
    "benches",
    "start_rate",
    "bench_rate",
    "sub_entry_rate",
    "mean_minutes",
    "recent_minutes",
    "mean_minute_in",
    "mean_minute_out",
    "days_since_appearance",
    "team_experience",
    "league_experience",
    "team_change",
    "role_stability",
    "player_statistics_observations",
)
GRANULAR_POSITIONS = (
    "goalkeeper",
    "centre_back",
    "full_back",
    "defensive_midfield",
    "central_midfield",
    "attacking_midfield",
    "wide_midfield",
    "winger",
    "second_striker",
    "centre_forward",
    "other",
)
TEMPORAL_FEATURE_NAMES = (
    tuple(
        f"fs_{field}_{suffix}"
        for field in TEMPORAL_NUMERIC_FIELDS
        for suffix in ("value", "available", "quality", "fallback")
    )
    + tuple(f"fs_position_{position}" for position in GRANULAR_POSITIONS)
    + (
        "fs_position_available",
        "fs_position_quality",
        "fs_position_fallback",
        "fs_store_available",
    )
)
NEURAL_FEATURE_NAMES = BASE_NEURAL_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
DEPARTMENT_INDEX = {name: index for index, name in enumerate(POOL_DEPARTMENTS)}
MAX_CONFIRMED_BENCH = 12
BASE_FEATURE_COUNT = len(BASE_NEURAL_FEATURE_NAMES)

QUALITY_SCORES = {
    "reported": 1.0,
    "observed": 1.0,
    "normalized": 0.9,
    "derived": 0.8,
    "observed_interval": 0.7,
    "proxy": 0.5,
}


def _position_bucket(value: object) -> str:
    position = str(value or "").strip().lower()
    if position == "goalkeeper":
        return "goalkeeper"
    if position in {"centre-back", "sweeper"}:
        return "centre_back"
    if position in {"right-back", "left-back"}:
        return "full_back"
    if position == "defensive midfield":
        return "defensive_midfield"
    if position in {"central midfield", "midfield"}:
        return "central_midfield"
    if position == "attacking midfield":
        return "attacking_midfield"
    if position in {"right midfield", "left midfield"}:
        return "wide_midfield"
    if position in {"right winger", "left winger"}:
        return "winger"
    if position == "second striker":
        return "second_striker"
    if position == "centre-forward":
        return "centre_forward"
    return "other"


def _department(position: object) -> str:
    bucket = _position_bucket(position)
    if bucket == "goalkeeper":
        return "goalkeeper"
    if bucket in {"centre_back", "full_back"}:
        return "defense"
    if bucket in {
        "defensive_midfield",
        "central_midfield",
        "attacking_midfield",
        "wide_midfield",
    }:
        return "midfield"
    if bucket in {"winger", "second_striker", "centre_forward"}:
        return "attack"
    return "unknown"


def _scaled_temporal_value(field: str, value: object) -> float:
    if pd.isna(value):
        return 0.0
    numeric = float(cast(Any, value))
    if field in {
        "observations",
        "starts",
        "benches",
        "team_experience",
        "league_experience",
        "player_statistics_observations",
    }:
        return float(np.log1p(max(numeric, 0.0)))
    if field in {
        "mean_minutes",
        "recent_minutes",
        "mean_minute_in",
        "mean_minute_out",
    }:
        return numeric / 90.0
    if field == "days_since_appearance":
        return float(np.log1p(min(max(numeric, 0.0), 365.0)))
    return numeric


def _temporal_vector(row: dict[str, Any]) -> np.ndarray:
    values: list[float] = []
    for field in TEMPORAL_NUMERIC_FIELDS:
        available = bool(row.get(f"{field}_available", False))
        quality = str(row.get(f"{field}_quality") or "")
        fallback = str(row.get(f"{field}_fallback_kind") or "none")
        values.extend(
            (
                _scaled_temporal_value(field, row.get(f"{field}_value")),
                float(available),
                QUALITY_SCORES.get(quality, 0.0),
                float(fallback != "none"),
            )
        )
    position = _position_bucket(row.get("position_original_value"))
    values.extend(float(position == name) for name in GRANULAR_POSITIONS)
    position_available = bool(row.get("position_original_available", False))
    values.extend(
        (
            float(position_available),
            QUALITY_SCORES.get(
                str(row.get("position_original_quality") or ""),
                0.0,
            ),
            float(str(row.get("position_original_fallback_kind") or "none") != "none"),
            1.0,
        )
    )
    return np.asarray(values, dtype=np.float32)


def _temporal_indexes(
    temporal_features: pd.DataFrame | None,
    match_ids: set[str],
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], str],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    if temporal_features is None or temporal_features.empty:
        return {}, {}, {}
    required = {
        "match_id",
        "player_id",
        "team_id",
        "current_lineup_role",
        "current_position_original",
        "current_position_normalized",
    }
    missing = required.difference(temporal_features.columns)
    if missing:
        raise ValueError(f"Feature store temporale incompleto: {sorted(missing)}")
    selected = temporal_features.loc[
        temporal_features["match_id"].astype(str).isin(match_ids)
    ].copy()
    if selected.duplicated(["match_id", "player_id"]).any():
        raise ValueError("Feature store con duplicati match_id/player_id.")
    vectors: dict[tuple[str, str], np.ndarray] = {}
    teams: dict[tuple[str, str], str] = {}
    benches: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in selected.to_dict("records"):
        row = cast(dict[str, Any], raw)
        match_id = str(row["match_id"])
        player_id = str(row["player_id"])
        team_id = str(row["team_id"])
        vectors[(match_id, player_id)] = _temporal_vector(row)
        teams[(match_id, player_id)] = team_id
        if str(row.get("current_lineup_role")) == "bench":
            original = row.get("current_position_original")
            benches[(match_id, team_id)].append(
                {
                    "player_id": player_id,
                    "position": row.get("current_position_normalized"),
                    "formation_grid": (
                        f"reported:{original}" if pd.notna(original) else None
                    ),
                }
            )
    return vectors, teams, dict(benches)


@dataclass
class PlayerTensorDataset:
    matches: pd.DataFrame
    players: np.ndarray
    departments: np.ndarray
    bench_players: np.ndarray
    bench_departments: np.ndarray
    bench_mask: np.ndarray
    feature_names: tuple[str, ...] = NEURAL_FEATURE_NAMES


def _position_quality(player: dict[str, Any]) -> float:
    original = str(player.get("formation_grid") or "").strip().lower()
    if original.startswith("reported:"):
        return 1.0
    if original.startswith("reconstructed:"):
        return 0.6
    if original:
        return 0.8
    return 0.4 if str(player.get("position") or "").strip() else 0.0


def _players(
    value: object,
    side: str,
    *,
    expected: int | None = None,
) -> list[dict[str, Any]]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or (
        expected is not None and len(parsed) != expected
    ):
        raise ValueError(
            f"La lineup {side} deve contenere "
            f"{expected if expected is not None else 'una lista di'} giocatori."
        )
    result = []
    for player in parsed:
        if not isinstance(player, dict) or not player.get("player_id"):
            raise ValueError(f"ID giocatore mancante nella lineup {side}.")
        try:
            department = _department(player.get("position"))
        except ValueError:
            if expected is not None:
                raise
            department = "unknown"
        result.append(
            {
                "player_id": str(player["player_id"]),
                "department": department,
                "position_quality": _position_quality(player),
                "timing_observed": bool(player.get("timing_observed", False)),
                "minutes_played": player.get("minutes_played"),
                "minute_in": player.get("minute_in"),
                "minute_out": player.get("minute_out"),
            }
        )
    if len({player["player_id"] for player in result}) != len(result):
        raise ValueError(f"ID duplicati nella lineup {side}.")
    return result


def _snapshot(
    player_id: str,
    kickoff: pd.Timestamp,
    state: dict[str, dict[str, Any]],
    shared_starts: dict[tuple[str, str], int],
    shared_minutes: dict[tuple[str, str], tuple[float, int]],
    *,
    home: bool,
    team: str,
    league: str,
    department: str,
    teammates: list[str],
    position_quality: float,
) -> list[float]:
    history = state.get(player_id)
    if history is None:
        values = {name: 0.0 for name in NEURAL_FEATURE_NAMES}
        values["home_indicator"] = float(home)
        values["original_position_quality"] = position_quality
        return [values[name] for name in NEURAL_FEATURE_NAMES]
    starts = int(history["starts"])
    observations = int(history["observations"])
    start_denominator = max(starts, 1)
    days = max((kickoff - history["last_date"]).total_seconds() / 86400, 0.0)
    recent_points = list(history["recent_points"])
    recent_gd = list(history["recent_goal_difference"])
    teammate_keys = [
        (min(player_id, teammate), max(player_id, teammate))
        for teammate in teammates
        if teammate != player_id
    ]
    familiarity = [shared_starts.get(key, 0) for key in teammate_keys]
    real_shared = [
        shared_minutes[key] for key in teammate_keys if key in shared_minutes
    ]
    real_minutes_count = int(history["real_minutes_count"])
    values = {name: 0.0 for name in NEURAL_FEATURE_NAMES}
    values.update(
        {
            "career_strength": float(history["points"]) / (3.0 * start_denominator),
            "career_goal_difference": (
                float(history["goal_difference"]) / start_denominator
            ),
            "log_starts": float(np.log1p(starts)),
            "log_days_since": float(np.log1p(min(days, 365.0))),
            "observed": 1.0,
            "recent_strength": (
                float(np.mean(recent_points) / 3.0) if recent_points else 0.0
            ),
            "recent_goal_difference": (float(np.mean(recent_gd)) if recent_gd else 0.0),
            "home_indicator": float(home),
            "career_goals_for": float(history["goals_for"]) / start_denominator,
            "career_goals_against": (
                float(history["goals_against"]) / start_denominator
            ),
            "recent_goals_for": (
                float(np.mean(history["recent_goals_for"])) if starts else 0.0
            ),
            "recent_goals_against": (
                float(np.mean(history["recent_goals_against"])) if starts else 0.0
            ),
            "opponent_strength": (
                (float(history["opponent_elo_sum"]) / start_denominator - 1500.0)
                / 400.0
                if starts
                else 0.0
            ),
            "team_experience_share": (
                float(history["team_observations"].get(team, 0)) / observations
            ),
            "league_experience_share": (
                float(history["league_observations"].get(league, 0)) / observations
            ),
            "role_stability": (
                float(history["department_observations"].get(department, 0))
                / observations
            ),
            "lineup_familiarity": (
                float(np.mean(np.log1p(familiarity))) if familiarity else 0.0
            ),
            "log_squad_observations": float(np.log1p(observations)),
            "start_rate": starts / observations,
            "bench_rate": float(history["bench"]) / observations,
            "bench_entry_rate": (
                float(history["entries"]) / history["entry_observed"]
                if history["entry_observed"]
                else 0.0
            ),
            "average_minutes": float(history["minutes_sum"]) / observations / 90.0,
            "recent_minutes": (
                float(np.mean(history["recent_minutes"])) / 90.0
                if history["recent_minutes"]
                else 0.0
            ),
            "minute_in_mean": (
                float(history["minute_in_sum"]) / history["minute_in_count"] / 90.0
                if history["minute_in_count"]
                else 0.0
            ),
            "minute_out_mean": (
                float(history["minute_out_sum"]) / history["minute_out_count"] / 90.0
                if history["minute_out_count"]
                else 0.0
            ),
            "minutes_real_coverage": real_minutes_count / observations,
            "original_position_quality": position_quality,
            "shared_minutes": (
                float(np.mean([total / count for total, count in real_shared])) / 90.0
                if real_shared
                else 0.0
            ),
            "shared_minutes_coverage": len(real_shared) / max(len(teammate_keys), 1),
        }
    )
    return [float(values[name]) for name in NEURAL_FEATURE_NAMES]


def build_player_tensor_dataset(
    match_features: pd.DataFrame,
    lineup_dataset: pd.DataFrame,
    *,
    temporal_features: pd.DataFrame | None = None,
    recent_matches: int = 5,
) -> PlayerTensorDataset:
    """Build 2×11 player tensors before updating state with the current match."""
    if recent_matches < 1:
        raise ValueError("recent_matches deve essere positivo.")
    required = {
        "match_id",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    }
    lineup_required = {
        "match_id",
        "home_starters",
        "away_starters",
        "home_bench",
        "away_bench",
    }
    missing = required.difference(match_features.columns)
    lineup_missing = lineup_required.difference(lineup_dataset.columns)
    if missing or lineup_missing:
        raise ValueError(
            "Contratto tensore incompleto: "
            f"matches={sorted(missing)}, lineups={sorted(lineup_missing)}"
        )
    data = match_features.merge(
        lineup_dataset[
            [
                "match_id",
                "home_starters",
                "away_starters",
                "home_bench",
                "away_bench",
            ]
        ],
        on="match_id",
        how="inner",
        validate="one_to_one",
    ).copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, format="mixed")
    data = data.sort_values(["date", "match_id"]).reset_index(drop=True)
    temporal, temporal_teams, temporal_benches = _temporal_indexes(
        temporal_features,
        set(data["match_id"].astype(str)),
    )
    tensors = np.zeros((len(data), 2, 11, len(NEURAL_FEATURE_NAMES)), dtype=np.float32)
    departments = np.full((len(data), 2, 11), -1, dtype=np.int64)
    bench_tensors = np.zeros(
        (len(data), 2, MAX_CONFIRMED_BENCH, len(NEURAL_FEATURE_NAMES)),
        dtype=np.float32,
    )
    bench_departments = np.full(
        (len(data), 2, MAX_CONFIRMED_BENCH),
        -1,
        dtype=np.int64,
    )
    bench_mask = np.zeros(
        (len(data), 2, MAX_CONFIRMED_BENCH),
        dtype=np.float32,
    )
    state: dict[str, dict[str, Any]] = {}
    shared_starts: dict[tuple[str, str], int] = defaultdict(int)
    shared_minutes: dict[tuple[str, str], tuple[float, int]] = {}

    for _, simultaneous in data.groupby("date", sort=False):
        parsed_starters: dict[tuple[int, int], list[dict[str, Any]]] = {}
        parsed_bench: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row_index in simultaneous.index:
            row = data.loc[row_index]
            for side_index, side in enumerate(("home", "away")):
                starters = _players(
                    row[f"{side}_starters"],
                    f"{side} starters",
                    expected=11,
                )
                bench = _players(
                    row[f"{side}_bench"],
                    f"{side} bench",
                )
                match_id = str(row["match_id"])
                starter_ids = [player["player_id"] for player in starters]
                team_ids = [
                    temporal_teams[(match_id, player_id)]
                    for player_id in starter_ids
                    if (match_id, player_id) in temporal_teams
                ]
                if team_ids:
                    team_id = max(set(team_ids), key=team_ids.count)
                    known = {player["player_id"] for player in starters + bench}
                    additions = temporal_benches.get((match_id, team_id), [])
                    bench.extend(
                        _players(
                            json.dumps(
                                [
                                    player
                                    for player in additions
                                    if player["player_id"] not in known
                                ]
                            ),
                            f"{side} feature-store bench",
                        )
                    )
                bench = bench[:MAX_CONFIRMED_BENCH]
                parsed_starters[(int(row_index), side_index)] = starters
                parsed_bench[(int(row_index), side_index)] = bench
                for player_index, player in enumerate(starters):
                    vector = _snapshot(
                        player["player_id"],
                        pd.Timestamp(row["date"]),
                        state,
                        shared_starts,
                        shared_minutes,
                        home=side == "home",
                        team=str(row[f"{side}_team"]),
                        league=str(row["league"]),
                        department=player["department"],
                        teammates=starter_ids,
                        position_quality=float(player["position_quality"]),
                    )
                    canonical = temporal.get(
                        (str(row["match_id"]), player["player_id"])
                    )
                    if canonical is not None:
                        vector[BASE_FEATURE_COUNT:] = canonical
                    tensors[row_index, side_index, player_index] = vector
                    departments[row_index, side_index, player_index] = (
                        DEPARTMENT_INDEX.get(player["department"], -1)
                    )
                for player_index, player in enumerate(bench):
                    vector = _snapshot(
                        player["player_id"],
                        pd.Timestamp(row["date"]),
                        state,
                        shared_starts,
                        shared_minutes,
                        home=side == "home",
                        team=str(row[f"{side}_team"]),
                        league=str(row["league"]),
                        department=player["department"],
                        teammates=starter_ids,
                        position_quality=float(player["position_quality"]),
                    )
                    canonical = temporal.get(
                        (str(row["match_id"]), player["player_id"])
                    )
                    if canonical is not None:
                        vector[BASE_FEATURE_COUNT:] = canonical
                    bench_tensors[row_index, side_index, player_index] = vector
                    bench_departments[row_index, side_index, player_index] = (
                        DEPARTMENT_INDEX.get(player["department"], -1)
                    )
                    bench_mask[row_index, side_index, player_index] = 1.0
        for row_index in simultaneous.index:
            row = data.loc[row_index]
            if row["result"] not in {"H", "D", "A"}:
                continue
            for side_index, (side, opponent) in enumerate(
                (("home", "away"), ("away", "home"))
            ):
                goals_for = float(row[f"{side}_goals"])
                goals_against = float(row[f"{opponent}_goals"])
                opponent_elo = float(row.get(f"{opponent}_elo", 1500.0))
                if not np.isfinite(opponent_elo):
                    opponent_elo = 1500.0
                points = 3.0 if goals_for > goals_against else 0.0
                if goals_for == goals_against:
                    points = 1.0
                starters = parsed_starters[(int(row_index), side_index)]
                bench = parsed_bench[(int(row_index), side_index)]
                lineup = [(player, True) for player in starters]
                lineup += [(player, False) for player in bench]
                actual_minutes: dict[str, float] = {}
                for player, is_starter in lineup:
                    history = state.setdefault(
                        player["player_id"],
                        {
                            "observations": 0,
                            "starts": 0,
                            "bench": 0,
                            "entries": 0,
                            "entry_observed": 0,
                            "points": 0.0,
                            "goal_difference": 0.0,
                            "goals_for": 0.0,
                            "goals_against": 0.0,
                            "opponent_elo_sum": 0.0,
                            "last_date": row["date"],
                            "recent_points": deque(maxlen=recent_matches),
                            "recent_goal_difference": deque(maxlen=recent_matches),
                            "recent_goals_for": deque(maxlen=recent_matches),
                            "recent_goals_against": deque(maxlen=recent_matches),
                            "minutes_sum": 0.0,
                            "real_minutes_count": 0,
                            "recent_minutes": deque(maxlen=recent_matches),
                            "minute_in_sum": 0.0,
                            "minute_in_count": 0,
                            "minute_out_sum": 0.0,
                            "minute_out_count": 0,
                            "position_quality_sum": 0.0,
                            "team_observations": defaultdict(int),
                            "league_observations": defaultdict(int),
                            "department_observations": defaultdict(int),
                        },
                    )
                    history["observations"] += 1
                    history["starts"] += int(is_starter)
                    history["bench"] += int(not is_starter)
                    timing_observed = bool(player["timing_observed"])
                    minutes = (
                        float(player["minutes_played"])
                        if timing_observed and player["minutes_played"] is not None
                        else (90.0 if is_starter else 0.0)
                    )
                    minutes = float(np.clip(minutes, 0.0, 120.0))
                    history["minutes_sum"] += minutes
                    history["recent_minutes"].append(minutes)
                    history["position_quality_sum"] += float(player["position_quality"])
                    if timing_observed:
                        history["real_minutes_count"] += 1
                        actual_minutes[player["player_id"]] = minutes
                    if not is_starter and timing_observed:
                        history["entry_observed"] += 1
                        history["entries"] += int(
                            player["minute_in"] is not None or minutes > 0
                        )
                    if player["minute_in"] is not None:
                        history["minute_in_sum"] += float(player["minute_in"])
                        history["minute_in_count"] += 1
                    if player["minute_out"] is not None:
                        history["minute_out_sum"] += float(player["minute_out"])
                        history["minute_out_count"] += 1
                    if is_starter:
                        history["points"] += points
                        history["goal_difference"] += goals_for - goals_against
                        history["goals_for"] += goals_for
                        history["goals_against"] += goals_against
                        history["opponent_elo_sum"] += opponent_elo
                        history["recent_points"].append(points)
                        history["recent_goal_difference"].append(
                            goals_for - goals_against
                        )
                        history["recent_goals_for"].append(goals_for)
                        history["recent_goals_against"].append(goals_against)
                    history["last_date"] = row["date"]
                    history["team_observations"][str(row[f"{side}_team"])] += 1
                    history["league_observations"][str(row["league"])] += 1
                    history["department_observations"][player["department"]] += 1
                ids = sorted(player["player_id"] for player in starters)
                for first_index, first in enumerate(ids):
                    for second in ids[first_index + 1 :]:
                        shared_starts[(first, second)] += 1
                participants = sorted(actual_minutes)
                for first_index, first in enumerate(participants):
                    for second in participants[first_index + 1 :]:
                        key = (first, second)
                        total, count = shared_minutes.get(key, (0.0, 0))
                        shared_minutes[key] = (
                            total + min(actual_minutes[first], actual_minutes[second]),
                            count + 1,
                        )
    return PlayerTensorDataset(
        matches=data.drop(
            columns=[
                "home_starters",
                "away_starters",
                "home_bench",
                "away_bench",
            ]
        ),
        players=tensors,
        departments=departments,
        bench_players=bench_tensors,
        bench_departments=bench_departments,
        bench_mask=bench_mask,
    )


class SharedPlayerEncoder(nn.Module):
    """One shared encoder, department pooling and a bounded two-rate head."""

    def __init__(
        self,
        input_dim: int,
        *,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.25,
        maximum_log_correction: float = 0.35,
    ) -> None:
        super().__init__()
        self.maximum_log_correction = maximum_log_correction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )
        pooled_dim = 2 * 2 * len(POOL_DEPARTMENTS) * embedding_dim
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        players: torch.Tensor,
        departments: torch.Tensor,
        bench_players: torch.Tensor,
        bench_departments: torch.Tensor,
        bench_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(players)
        encoded_bench = self.encoder(bench_players)
        pools = []
        for side in range(2):
            for department in range(len(POOL_DEPARTMENTS)):
                mask = (departments[:, side] == department).unsqueeze(-1)
                count = mask.sum(dim=1).clamp(min=1)
                pools.append((encoded[:, side] * mask).sum(dim=1) / count)
            for department in range(len(POOL_DEPARTMENTS)):
                mask = (
                    (bench_departments[:, side] == department)
                    * bench_mask[:, side].bool()
                ).unsqueeze(-1)
                count = mask.sum(dim=1).clamp(min=1)
                pools.append((encoded_bench[:, side] * mask).sum(dim=1) / count)
        pooled = torch.cat(pools, dim=1)
        return torch.tanh(self.head(pooled)) * self.maximum_log_correction


@dataclass
class NeuralLineupPredictor:
    model: SharedPlayerEncoder
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    embedding_dim: int
    maximum_log_correction: float
    device: str = "cpu"

    def corrections(
        self,
        players: np.ndarray,
        departments: np.ndarray,
        bench_players: np.ndarray,
        bench_departments: np.ndarray,
        bench_mask: np.ndarray,
    ) -> np.ndarray:
        normalized = (players - self.feature_mean) / self.feature_scale
        normalized_bench = (bench_players - self.feature_mean) / self.feature_scale
        self.model.eval()
        device = torch.device(self.device)
        with torch.no_grad():
            result = self.model(
                torch.as_tensor(normalized, dtype=torch.float32, device=device),
                torch.as_tensor(departments, dtype=torch.long, device=device),
                torch.as_tensor(
                    normalized_bench, dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    bench_departments, dtype=torch.long, device=device
                ),
                torch.as_tensor(bench_mask, dtype=torch.float32, device=device),
            )
        return result.cpu().numpy()


def fit_neural_lineup_encoder(
    players: np.ndarray,
    departments: np.ndarray,
    targets: np.ndarray,
    *,
    bench_players: np.ndarray | None = None,
    bench_departments: np.ndarray | None = None,
    bench_mask: np.ndarray | None = None,
    embedding_dim: int = 32,
    epochs: int = 80,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> NeuralLineupPredictor:
    if len(players) != len(targets) or not len(players):
        raise ValueError("Training neurale vuoto o disallineato.")
    if bench_players is None:
        bench_players = np.zeros(
            (len(players), 2, MAX_CONFIRMED_BENCH, players.shape[-1]),
            dtype=np.float32,
        )
    if bench_departments is None:
        bench_departments = np.full(
            (len(players), 2, MAX_CONFIRMED_BENCH),
            -1,
            dtype=np.int64,
        )
    if bench_mask is None:
        bench_mask = np.zeros(
            (len(players), 2, MAX_CONFIRMED_BENCH),
            dtype=np.float32,
        )
    torch.manual_seed(seed)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    flat = players.reshape(-1, players.shape[-1])
    valid_bench = bench_players[bench_mask.astype(bool)]
    if len(valid_bench):
        flat = np.concatenate([flat, valid_bench], axis=0)
    mean = flat.mean(axis=0, keepdims=True).reshape(1, 1, 1, -1)
    scale = flat.std(axis=0, keepdims=True).reshape(1, 1, 1, -1)
    scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = (players - mean) / scale
    dataset = TensorDataset(
        torch.as_tensor(normalized, dtype=torch.float32),
        torch.as_tensor(departments, dtype=torch.long),
        torch.as_tensor(
            (bench_players - mean) / scale,
            dtype=torch.float32,
        ),
        torch.as_tensor(bench_departments, dtype=torch.long),
        torch.as_tensor(bench_mask, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    model = SharedPlayerEncoder(
        players.shape[-1],
        embedding_dim=embedding_dim,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.SmoothL1Loss()
    model.train()
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0
    for _ in range(epochs):
        epoch_loss = 0.0
        for (
            batch_players,
            batch_departments,
            batch_bench_players,
            batch_bench_departments,
            batch_bench_mask,
            batch_targets,
        ) in loader:
            batch_players = batch_players.to(selected_device)
            batch_departments = batch_departments.to(selected_device)
            batch_bench_players = batch_bench_players.to(selected_device)
            batch_bench_departments = batch_bench_departments.to(selected_device)
            batch_bench_mask = batch_bench_mask.to(selected_device)
            batch_targets = batch_targets.to(selected_device)
            optimizer.zero_grad()
            loss = loss_function(
                model(
                    batch_players,
                    batch_departments,
                    batch_bench_players,
                    batch_bench_departments,
                    batch_bench_mask,
                ),
                batch_targets,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch_players)
        epoch_loss /= len(dataset)
        if best_loss - epoch_loss > early_stopping_min_delta:
            best_loss = epoch_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break
    model.load_state_dict(best_state)
    return NeuralLineupPredictor(
        model=model,
        feature_mean=mean,
        feature_scale=scale,
        embedding_dim=embedding_dim,
        maximum_log_correction=model.maximum_log_correction,
        device=str(selected_device),
    )


def cross_fitted_dixon_coles_rates(
    base_features: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Generate rates for each season from a base fitted only on earlier seasons."""
    base = base_features.copy()
    target = matches[["match_id", "season"]].copy()
    base["season"] = base["season"].astype(str).str.zfill(4)
    target["season"] = target["season"].astype(str).str.zfill(4)
    rows = []
    for season in sorted(target["season"].unique()):
        training = base.loc[base["season"].lt(season)]
        test_ids = set(target.loc[target["season"].eq(season), "match_id"])
        test = base.loc[base["match_id"].isin(test_ids)]
        if training.empty or test.empty or set(training["result"]) != {"H", "D", "A"}:
            continue
        predictor = fit_dixon_coles(training)
        home_rate, away_rate = predictor.rates(test)
        frame = test[["match_id", "season"]].copy()
        frame["home_rate"] = home_rate
        frame["away_rate"] = away_rate
        frame["rho"] = predictor.rho
        rows.append(frame)
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=["match_id", "season", "home_rate", "away_rate", "rho"]
        )
    )


def _prediction_frame(
    matches: pd.DataFrame,
    probabilities: np.ndarray,
    model: str,
) -> pd.DataFrame:
    frame = matches[["match_id", "season", "league", "result"]].copy()
    frame["model"] = model
    frame[list(PROBABILITY_COLUMNS)] = probabilities
    return frame


def walk_forward_neural_lineup_model(
    tensor_data: PlayerTensorDataset,
    base_features: pd.DataFrame,
    *,
    embedding_dim: int = 32,
    epochs: int = 80,
    rates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = tensor_data.matches.copy()
    matches["season"] = matches["season"].astype(str).str.zfill(4)
    if rates is None:
        rates = cross_fitted_dixon_coles_rates(base_features, matches)
    matches = matches.merge(rates, on=["match_id", "season"], how="left")
    all_predictions = []
    for season in sorted(matches["season"].unique())[1:]:
        training_mask = matches["season"].lt(season) & matches["home_rate"].notna()
        test_mask = matches["season"].eq(season) & matches["home_rate"].notna()
        if not training_mask.any() or not test_mask.any():
            continue
        training_indices = np.flatnonzero(training_mask.to_numpy())
        test_indices = np.flatnonzero(test_mask.to_numpy())
        training = matches.iloc[training_indices]
        targets = np.column_stack(
            [
                np.log(
                    (training["home_goals"].to_numpy(float) + 0.5)
                    / (training["home_rate"].to_numpy(float) + 0.5)
                ),
                np.log(
                    (training["away_goals"].to_numpy(float) + 0.5)
                    / (training["away_rate"].to_numpy(float) + 0.5)
                ),
            ]
        ).astype(np.float32)
        predictor = fit_neural_lineup_encoder(
            tensor_data.players[training_indices],
            tensor_data.departments[training_indices],
            targets,
            bench_players=tensor_data.bench_players[training_indices],
            bench_departments=tensor_data.bench_departments[training_indices],
            bench_mask=tensor_data.bench_mask[training_indices],
            embedding_dim=embedding_dim,
            epochs=epochs,
        )
        corrections = predictor.corrections(
            tensor_data.players[test_indices],
            tensor_data.departments[test_indices],
            tensor_data.bench_players[test_indices],
            tensor_data.bench_departments[test_indices],
            tensor_data.bench_mask[test_indices],
        )
        test = matches.iloc[test_indices]
        home_rates = test["home_rate"].to_numpy(float) * np.exp(corrections[:, 0])
        away_rates = test["away_rate"].to_numpy(float) * np.exp(corrections[:, 1])
        probabilities = np.vstack(
            [
                score_probabilities(home, away, rho)
                for home, away, rho in zip(
                    home_rates,
                    away_rates,
                    test["rho"].to_numpy(float),
                    strict=True,
                )
            ]
        )
        all_predictions.append(
            _prediction_frame(test, probabilities, NEURAL_LINEUP_MODEL_NAME)
        )
    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame(
            columns=[
                "match_id",
                "season",
                "league",
                "result",
                "model",
                *PROBABILITY_COLUMNS,
            ]
        )
    )
    return metrics_by_season(predictions), predictions


def export_neural_lineup_model(
    tensor_data: PlayerTensorDataset,
    base_features: pd.DataFrame,
    market_predictions: pd.DataFrame,
    destination: Path,
    *,
    embedding_dim: int = 32,
    epochs: int = 80,
) -> dict[str, Path]:
    """Evaluate and persist the official neural model against the market."""
    destination.mkdir(parents=True, exist_ok=True)
    normalized_matches = tensor_data.matches.copy()
    normalized_matches["season"] = normalized_matches["season"].astype(str).str.zfill(4)
    rates = cross_fitted_dixon_coles_rates(base_features, normalized_matches)
    neural_metrics, neural_predictions = walk_forward_neural_lineup_model(
        tensor_data,
        base_features,
        embedding_dim=embedding_dim,
        epochs=epochs,
        rates=rates,
    )
    ids = set(neural_predictions["match_id"])
    baseline_matches = normalized_matches.merge(
        rates,
        on=["match_id", "season"],
        how="inner",
    )
    baseline_probabilities = np.vstack(
        [
            score_probabilities(home, away, rho)
            for home, away, rho in zip(
                baseline_matches["home_rate"],
                baseline_matches["away_rate"],
                baseline_matches["rho"],
                strict=True,
            )
        ]
    )
    baseline = _prediction_frame(
        baseline_matches,
        baseline_probabilities,
        "dixon_coles_without_confirmed_lineup",
    )
    references = {
        "dixon_coles": baseline.loc[baseline["match_id"].isin(ids)],
    }
    if not market_predictions.empty:
        market = market_predictions.copy()
        market["season"] = market["season"].astype(str).str.zfill(4)
        references["bookmaker_market"] = market.loc[market["match_id"].isin(ids)]
    common_ids = set(neural_predictions["match_id"])
    for reference in references.values():
        common_ids.intersection_update(reference["match_id"])
    neural_predictions = neural_predictions.loc[
        neural_predictions["match_id"].isin(common_ids)
    ].copy()
    references = {
        name: reference.loc[reference["match_id"].isin(common_ids)].copy()
        for name, reference in references.items()
    }
    neural_metrics = metrics_by_season(neural_predictions)
    evidence = {
        name: paired_log_loss_bootstrap(neural_predictions, reference)
        for name, reference in references.items()
    }
    comparison_predictions = pd.concat(
        [neural_predictions, *references.values()],
        ignore_index=True,
    )
    comparison_metrics = metrics_by_season(comparison_predictions)
    paths = {
        "metrics": destination / "neural_lineup_metrics_by_season.csv",
        "predictions": destination / "neural_lineup_predictions.csv",
        "comparison": destination / "neural_lineup_comparison.csv",
        "evidence": destination / "neural_lineup_bootstrap.json",
        "feature_store_audit": destination / "neural_feature_store_audit.json",
        "metadata": destination / "neural_lineup_model.meta.json",
        "model": destination / "neural_lineup_model.pt",
        "report": destination / "NEURAL_LINEUP_MODEL_REPORT.md",
    }
    neural_metrics.to_csv(paths["metrics"], index=False)
    neural_predictions.to_csv(paths["predictions"], index=False)
    comparison_metrics.to_csv(paths["comparison"], index=False)
    paths["evidence"].write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    feature_index = {
        name: index for index, name in enumerate(tensor_data.feature_names)
    }
    valid_bench = tensor_data.bench_mask.astype(bool)
    audit = {
        "matches": len(tensor_data.matches),
        "starter_observations": int(np.prod(tensor_data.players.shape[:3])),
        "confirmed_bench_observations": int(tensor_data.bench_mask.sum()),
        "matches_with_confirmed_bench": int(
            (tensor_data.bench_mask.sum(axis=(1, 2)) > 0).sum()
        ),
        "maximum_confirmed_bench_per_team": MAX_CONFIRMED_BENCH,
        "feature_names": list(tensor_data.feature_names),
        "starter_history_observed_rate": float(
            tensor_data.players[..., feature_index["observed"]].mean()
        ),
        "starter_real_minutes_coverage_mean": float(
            tensor_data.players[..., feature_index["minutes_real_coverage"]].mean()
        ),
        "starter_canonical_store_rate": float(
            tensor_data.players[..., feature_index["fs_store_available"]].mean()
        ),
        "starter_canonical_minutes_available_rate": float(
            tensor_data.players[..., feature_index["fs_mean_minutes_available"]].mean()
        ),
        "bench_history_observed_rate": (
            float(
                tensor_data.bench_players[..., feature_index["observed"]][
                    valid_bench
                ].mean()
            )
            if valid_bench.any()
            else 0.0
        ),
        "bench_canonical_store_rate": (
            float(
                tensor_data.bench_players[..., feature_index["fs_store_available"]][
                    valid_bench
                ].mean()
            )
            if valid_bench.any()
            else 0.0
        ),
        "bench_canonical_minutes_available_rate": (
            float(
                tensor_data.bench_players[
                    ..., feature_index["fs_mean_minutes_available"]
                ][valid_bench].mean()
            )
            if valid_bench.any()
            else 0.0
        ),
        "fallbacks": {
            "minutes": ("real_when_available_else_starter_90_bench_0_with_indicator"),
            "shared_minutes": "real_overlap_when_available_else_shared_starts",
            "position": "original_when_available_else_department",
            "bench_entry": "real_timing_when_available_else_unknown_with_indicator",
            "membership": "chronological_team_observation_share",
        },
    }
    paths["feature_store_audit"].write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata = {
        "model_name": NEURAL_LINEUP_MODEL_NAME,
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "feature_names": list(tensor_data.feature_names),
        "departments": list(POOL_DEPARTMENTS),
        "identity_embedding": False,
        "separate_confirmed_bench_pooling": True,
        "maximum_confirmed_bench_per_team": MAX_CONFIRMED_BENCH,
        "feature_store_audit": str(paths["feature_store_audit"]),
        "cross_fitted_dixon_coles_targets": True,
        "automatic_promotion": False,
        "comparisons": evidence,
    }
    final_matches = normalized_matches.merge(
        rates,
        on=["match_id", "season"],
        how="left",
    )
    final_mask = final_matches["home_rate"].notna().to_numpy()
    final_targets = np.column_stack(
        [
            np.log(
                (final_matches.loc[final_mask, "home_goals"].to_numpy(float) + 0.5)
                / (final_matches.loc[final_mask, "home_rate"].to_numpy(float) + 0.5)
            ),
            np.log(
                (final_matches.loc[final_mask, "away_goals"].to_numpy(float) + 0.5)
                / (final_matches.loc[final_mask, "away_rate"].to_numpy(float) + 0.5)
            ),
        ]
    ).astype(np.float32)
    final_predictor = fit_neural_lineup_encoder(
        tensor_data.players[final_mask],
        tensor_data.departments[final_mask],
        final_targets,
        bench_players=tensor_data.bench_players[final_mask],
        bench_departments=tensor_data.bench_departments[final_mask],
        bench_mask=tensor_data.bench_mask[final_mask],
        embedding_dim=embedding_dim,
        epochs=epochs,
    )
    torch.save(
        {
            "state_dict": final_predictor.model.state_dict(),
            "feature_mean": final_predictor.feature_mean,
            "feature_scale": final_predictor.feature_scale,
            "feature_names": tensor_data.feature_names,
            "embedding_dim": embedding_dim,
            "maximum_log_correction": final_predictor.maximum_log_correction,
        },
        paths["model"],
    )
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = (
        comparison_metrics.groupby("model", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "matches": int(group["matches"].sum()),
                    **{
                        name: float(np.average(group[name], weights=group["matches"]))
                        for name in ("log_loss", "brier", "rps", "accuracy", "ece")
                    },
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    lines = [
        "# Encoder neurale condiviso + pooling",
        "",
        "Variante sperimentale: nessuna promozione automatica.",
        "",
        f"Feature store temporale: {len(tensor_data.feature_names)} feature "
        "individuali con titolari e panchina confermata in pool separati. "
        "Valori reali, fallback e copertura sono descritti in "
        "`neural_feature_store_audit.json`.",
        "",
        "| Modello | Match | Log Loss | Brier | RPS | Accuracy | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['model']} | {int(row['matches'])} | "
            f"{row['log_loss']:.4f} | {row['brier']:.4f} | "
            f"{row['rps']:.4f} | {row['accuracy']:.2%} | {row['ece']:.4f} |"
        )
    lines.extend(["", "## Bootstrap appaiato", ""])
    for name, values in evidence.items():
        lines.append(
            f"- Contro {name}: Δ Log Loss "
            f"{values['mean_log_loss_difference']:.5f}, "
            f"IC 95% [{values['ci_low']:.5f}, {values['ci_high']:.5f}]."
        )
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
