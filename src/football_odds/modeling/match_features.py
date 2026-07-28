"""Leakage-safe Elo, rolling history and pre-match features."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.config import ModelingConfig
from ..data.repository import ResearchDatabase


@dataclass(frozen=True)
class EloSettings:
    """Parameters controlling one Elo rating pool."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 65.0
    season_regression: float = 0.25


class EloRatings:
    """Stateful Elo engine whose values are read before each update."""

    def __init__(self, settings: EloSettings | None = None) -> None:
        self.settings = settings or EloSettings()
        self._ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        """Return a team's current rating, initializing it when unseen."""
        return self._ratings.setdefault(team, self.settings.initial_rating)

    def expected_home(self, home_team: str, away_team: str) -> float:
        """Return expected home score after applying home advantage."""
        difference = (
            self.rating(home_team)
            + self.settings.home_advantage
            - self.rating(away_team)
        )
        return 1.0 / (1.0 + 10.0 ** (-difference / 400.0))

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> tuple[float, float]:
        """Update both ratings and return their new values."""
        home_rating = self.rating(home_team)
        away_rating = self.rating(away_team)
        expected = self.expected_home(home_team, away_team)
        actual = 1.0 if home_goals > away_goals else 0.0
        if home_goals == away_goals:
            actual = 0.5
        goal_multiplier = 1.0 + math.log1p(abs(home_goals - away_goals))
        change = self.settings.k_factor * goal_multiplier * (actual - expected)
        self._ratings[home_team] = home_rating + change
        self._ratings[away_team] = away_rating - change
        return self._ratings[home_team], self._ratings[away_team]

    def regress_to_mean(self) -> None:
        """Shrink every known rating between seasons."""
        weight = self.settings.season_regression
        base = self.settings.initial_rating
        self._ratings = {
            team: (1.0 - weight) * rating + weight * base
            for team, rating in self._ratings.items()
        }


RAW_PERFORMANCE_COLUMNS = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
}
MATCH_PERFORMANCE_COLUMNS = tuple(RAW_PERFORMANCE_COLUMNS.values())
ROLLING_PERFORMANCE_FIELDS = (
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "yellow_cards",
    "red_cards",
    "shot_conversion",
    "shot_accuracy",
)
HISTORY_PERFORMANCE_FIELDS = (
    *ROLLING_PERFORMANCE_FIELDS,
    "goal_difference",
    "shots_on_target_difference",
)

CANONICAL_MATCH_QUERY = """
SELECT
    m.match_id,
    m.date,
    m.season,
    l.league_code AS league,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    r.home_goals,
    r.away_goals,
    r.result,
    r.home_shots,
    r.away_shots,
    r.home_shots_on_target,
    r.away_shots_on_target,
    r.home_corners,
    r.away_corners,
    r.home_yellow_cards,
    r.away_yellow_cards,
    r.home_red_cards,
    r.away_red_cards,
    AVG(CASE WHEN o.selection = 'H' THEN o.implied_probability END)
        AS market_home_probability,
    AVG(CASE WHEN o.selection = 'D' THEN o.implied_probability END)
        AS market_draw_probability,
    AVG(CASE WHEN o.selection = 'A' THEN o.implied_probability END)
        AS market_away_probability,
    AVG(o.margin) AS market_margin
FROM matches m
JOIN leagues l ON l.league_id = m.league_id
JOIN teams ht ON ht.team_id = m.home_team_id
JOIN teams at ON at.team_id = m.away_team_id
JOIN match_results r ON r.match_id = m.match_id
LEFT JOIN odds o
    ON o.match_id = m.match_id
    AND o.market = '1X2'
    AND o.opening_or_closing = 'closing'
GROUP BY
    m.match_id, m.date, m.season, l.league_code,
    ht.team_name, at.team_name,
    r.home_goals, r.away_goals, r.result,
    r.home_shots, r.away_shots,
    r.home_shots_on_target, r.away_shots_on_target,
    r.home_corners, r.away_corners,
    r.home_yellow_cards, r.away_yellow_cards,
    r.home_red_cards, r.away_red_cards
ORDER BY m.date, l.league_code, m.match_id
"""


def load_canonical_matches(
    database: ResearchDatabase,
    *,
    leagues: tuple[str, ...] | None = None,
    seasons: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Read the provider-neutral match view used by every downstream stage."""
    database.initialize()
    with database.connect() as connection:
        frame = pd.read_sql_query(
            CANONICAL_MATCH_QUERY, connection, parse_dates=["date"]
        )
    if leagues:
        frame = frame.loc[frame["league"].isin(leagues)]
    if seasons:
        frame = frame.loc[frame["season"].astype(str).isin(seasons)]
    valid = frame["result"].isin(("H", "D", "A")) & frame[
        ["home_goals", "away_goals"]
    ].notna().all(axis=1)
    frame = frame.loc[valid].copy()
    frame["home_goals"] = frame["home_goals"].astype(int)
    frame["away_goals"] = frame["away_goals"].astype(int)
    return frame.reset_index(drop=True)


def normalize_team_name(name: object, aliases: dict[str, str] | None = None) -> str:
    """Normalize whitespace and apply an optional provider alias map."""
    normalized = " ".join(str(name).strip().split())
    return (aliases or {}).get(normalized, normalized)


def prepare_modeling_matches(
    frame: pd.DataFrame,
    *,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Validate and chronologically order one provider-neutral match table."""
    required = {
        "Date",
        "Season",
        "League",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonne modellistiche mancanti: {sorted(missing)}")
    optional_odds = [
        column for column in ("AvgCH", "AvgCD", "AvgCA") if column in frame.columns
    ]
    available_performance = [
        column for column in RAW_PERFORMANCE_COLUMNS if column in frame.columns
    ]
    data = frame[list(required) + optional_odds + available_performance].copy()
    data["date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data["home_team"] = data["HomeTeam"].map(
        lambda value: normalize_team_name(value, aliases)
    )
    data["away_team"] = data["AwayTeam"].map(
        lambda value: normalize_team_name(value, aliases)
    )
    data["home_goals"] = pd.to_numeric(data["FTHG"], errors="coerce")
    data["away_goals"] = pd.to_numeric(data["FTAG"], errors="coerce")
    for source in available_performance:
        data[RAW_PERFORMANCE_COLUMNS[source]] = pd.to_numeric(
            data[source], errors="coerce"
        )
    if len(optional_odds) == 3:
        closing = data[optional_odds].apply(pd.to_numeric, errors="coerce")
        inverse = 1.0 / closing
        total = inverse.sum(axis=1)
        for source, target in zip(
            optional_odds,
            (
                "market_home_probability",
                "market_draw_probability",
                "market_away_probability",
            ),
            strict=True,
        ):
            data[target] = inverse[source] / total
    valid = (
        data["date"].notna()
        & data["FTR"].isin(("H", "D", "A"))
        & data[["home_goals", "away_goals"]].notna().all(axis=1)
        & data["home_team"].ne(data["away_team"])
    )
    data = data.loc[valid].rename(
        columns={"Season": "season", "League": "league", "FTR": "result"}
    )
    data["home_goals"] = data["home_goals"].astype(int)
    data["away_goals"] = data["away_goals"].astype(int)
    data = data.drop(
        columns=[
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            *optional_odds,
            *available_performance,
        ]
    )
    data["match_id"] = (
        data["league"].astype(str)
        + "|"
        + data["season"].astype(str)
        + "|"
        + data["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + data["home_team"]
        + "|"
        + data["away_team"]
    )
    data = data.drop_duplicates("match_id")
    return data.sort_values(["date", "league", "match_id"]).reset_index(drop=True)


def prepare_future_fixtures(
    frame: pd.DataFrame,
    *,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Validate the target-free canonical fixture contract."""
    required = {"date", "season", "league", "home_team", "away_team"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonne fixture mancanti: {sorted(missing)}")
    target_columns = {"result", "home_goals", "away_goals"}.intersection(frame.columns)
    if target_columns and frame[list(target_columns)].notna().any().any():
        raise ValueError("Le fixture future non devono contenere target osservati.")
    columns = [*required]
    if "match_id" in frame.columns:
        columns.append("match_id")
    data = frame[columns].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["home_team"] = data["home_team"].map(
        lambda value: normalize_team_name(value, aliases)
    )
    data["away_team"] = data["away_team"].map(
        lambda value: normalize_team_name(value, aliases)
    )
    invalid = (
        data["date"].isna()
        | data["season"].isna()
        | data["league"].isna()
        | data["home_team"].eq("")
        | data["away_team"].eq("")
        | data["home_team"].eq(data["away_team"])
    )
    if invalid.any():
        raise ValueError("Le fixture devono avere data, squadre e campionato validi.")
    if "match_id" not in data:
        data["match_id"] = (
            "fixture|"
            + data["league"].astype(str)
            + "|"
            + data["season"].astype(str)
            + "|"
            + data["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            + "|"
            + data["home_team"]
            + "|"
            + data["away_team"]
        )
    if data["match_id"].isna().any() or data["match_id"].duplicated().any():
        raise ValueError("Ogni fixture deve avere un match_id univoco.")
    data["home_goals"] = np.nan
    data["away_goals"] = np.nan
    data["result"] = pd.NA
    return data.sort_values(["date", "league", "match_id"]).reset_index(drop=True)


@dataclass
class _TeamHistory:
    dates: list[pd.Timestamp] = field(default_factory=list)
    goals_for: deque[float] = field(default_factory=deque)
    goals_against: deque[float] = field(default_factory=deque)
    points: deque[float] = field(default_factory=deque)
    opponent_elo: deque[float] = field(default_factory=deque)
    performance: dict[str, deque[float]] = field(
        default_factory=lambda: {
            field_name: deque() for field_name in HISTORY_PERFORMANCE_FIELDS
        }
    )


def _rolling_mean(values: deque[float], window: int) -> float:
    selected = np.asarray(list(values)[-window:], dtype=float)
    finite = selected[np.isfinite(selected)]
    return float(finite.mean()) if len(finite) else float("nan")


def _weighted_mean(values: deque[float], alpha: float = 0.35) -> float:
    selected = np.asarray(values, dtype=float)
    finite = np.isfinite(selected)
    if not finite.any():
        return float("nan")
    weights = np.power(1.0 - alpha, np.arange(len(selected))[::-1])
    return float(np.average(selected[finite], weights=weights[finite]))


def _numeric(row: pd.Series, column: str) -> float:
    value = row.get(column)
    return float(value) if pd.notna(value) else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return numerator / denominator


def _performance_values(
    row: pd.Series,
    side: str,
    goals_for: int,
    goals_against: int,
) -> dict[str, float]:
    opponent = "away" if side == "home" else "home"
    shots_for = _numeric(row, f"{side}_shots")
    shots_against = _numeric(row, f"{opponent}_shots")
    shots_on_target_for = _numeric(row, f"{side}_shots_on_target")
    shots_on_target_against = _numeric(row, f"{opponent}_shots_on_target")
    return {
        "shots_for": shots_for,
        "shots_against": shots_against,
        "shots_on_target_for": shots_on_target_for,
        "shots_on_target_against": shots_on_target_against,
        "corners_for": _numeric(row, f"{side}_corners"),
        "corners_against": _numeric(row, f"{opponent}_corners"),
        "yellow_cards": _numeric(row, f"{side}_yellow_cards"),
        "red_cards": _numeric(row, f"{side}_red_cards"),
        "shot_conversion": _ratio(float(goals_for), shots_for),
        "shot_accuracy": _ratio(shots_on_target_for, shots_for),
        "goal_difference": float(goals_for - goals_against),
        "shots_on_target_difference": (shots_on_target_for - shots_on_target_against),
    }


def build_prematch_features(
    matches: pd.DataFrame,
    config: ModelingConfig | None = None,
    *,
    include_unplayed: bool = False,
) -> pd.DataFrame:
    """Build features first, then update state only from completed matches."""
    config = config or ModelingConfig()
    config.validate()
    settings = EloSettings(
        initial_rating=config.elo_initial_rating,
        k_factor=config.elo_k_factor,
        home_advantage=config.elo_home_advantage,
        season_regression=config.elo_season_regression,
    )
    engines: dict[str, EloRatings] = defaultdict(lambda: EloRatings(settings))
    histories: dict[tuple[str, str], _TeamHistory] = defaultdict(_TeamHistory)
    venue_histories: dict[tuple[str, str, str], _TeamHistory] = defaultdict(
        _TeamHistory
    )
    current_season: dict[str, str] = {}
    output: list[dict[str, object]] = []

    ordered = matches.sort_values(["date", "league", "match_id"])
    for _, simultaneous in ordered.groupby("date", sort=False):
        pending: list[
            tuple[
                pd.Series,
                EloRatings,
                str,
                str,
                pd.Timestamp,
                _TeamHistory,
                _TeamHistory,
                _TeamHistory,
                _TeamHistory,
                float,
                float,
            ]
        ] = []
        for _, row in simultaneous.iterrows():
            league = str(row["league"])
            season = str(row["season"])
            engine = engines[league]
            if league in current_season and current_season[league] != season:
                engine.regress_to_mean()
            current_season[league] = season

            home = str(row["home_team"])
            away = str(row["away_team"])
            date = pd.Timestamp(row["date"])
            home_history = histories[(league, home)]
            away_history = histories[(league, away)]
            home_venue_history = venue_histories[(league, home, "home")]
            away_venue_history = venue_histories[(league, away, "away")]
            home_elo = engine.rating(home)
            away_elo = engine.rating(away)
            row_values = {
                str(key): value
                for key, value in row.to_dict().items()
                if key not in MATCH_PERFORMANCE_COLUMNS
            }
            feature_row: dict[str, object] = {
                **row_values,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_difference": home_elo + settings.home_advantage - away_elo,
                "elo_expected_home": engine.expected_home(home, away),
                "home_matches_played": len(home_history.dates),
                "away_matches_played": len(away_history.dates),
                "home_rest_days": _rest_days(home_history, date),
                "away_rest_days": _rest_days(away_history, date),
                "home_points_ewm": _weighted_mean(home_history.points),
                "away_points_ewm": _weighted_mean(away_history.points),
                "home_goal_difference_ewm": _weighted_mean(
                    home_history.performance["goal_difference"]
                ),
                "away_goal_difference_ewm": _weighted_mean(
                    away_history.performance["goal_difference"]
                ),
                "home_shots_on_target_difference_ewm": _weighted_mean(
                    home_history.performance["shots_on_target_difference"]
                ),
                "away_shots_on_target_difference_ewm": _weighted_mean(
                    away_history.performance["shots_on_target_difference"]
                ),
            }
            for window in config.rolling_windows:
                for prefix, history in (
                    ("home", home_history),
                    ("away", away_history),
                ):
                    feature_row[f"{prefix}_points_{window}"] = _rolling_mean(
                        history.points, window
                    )
                    feature_row[f"{prefix}_goals_for_{window}"] = _rolling_mean(
                        history.goals_for, window
                    )
                    feature_row[f"{prefix}_goals_against_{window}"] = _rolling_mean(
                        history.goals_against, window
                    )
                    feature_row[f"{prefix}_opponent_elo_{window}"] = _rolling_mean(
                        history.opponent_elo, window
                    )
                    for field_name in ROLLING_PERFORMANCE_FIELDS:
                        feature_row[f"{prefix}_{field_name}_{window}"] = _rolling_mean(
                            history.performance[field_name],
                            window,
                        )
                feature_row[f"home_venue_points_{window}"] = _rolling_mean(
                    home_venue_history.points, window
                )
                feature_row[f"home_venue_goal_difference_{window}"] = _rolling_mean(
                    home_venue_history.performance["goal_difference"],
                    window,
                )
                feature_row[f"away_venue_points_{window}"] = _rolling_mean(
                    away_venue_history.points, window
                )
                feature_row[f"away_venue_goal_difference_{window}"] = _rolling_mean(
                    away_venue_history.performance["goal_difference"],
                    window,
                )
            result = row.get("result")
            completed = bool(
                pd.notna(result)
                and result in ("H", "D", "A")
                and pd.notna(row.get("home_goals"))
                and pd.notna(row.get("away_goals"))
            )
            if completed or include_unplayed:
                output.append(feature_row)
            if completed:
                pending.append(
                    (
                        row,
                        engine,
                        home,
                        away,
                        date,
                        home_history,
                        away_history,
                        home_venue_history,
                        away_venue_history,
                        home_elo,
                        away_elo,
                    )
                )

        for (
            row,
            engine,
            home,
            away,
            date,
            home_history,
            away_history,
            home_venue_history,
            away_venue_history,
            home_elo,
            away_elo,
        ) in pending:
            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            home_points = 3 if home_goals > away_goals else 0
            away_points = 3 if away_goals > home_goals else 0
            if home_goals == away_goals:
                home_points = away_points = 1
            home_performance = _performance_values(row, "home", home_goals, away_goals)
            away_performance = _performance_values(row, "away", away_goals, home_goals)
            _append_history(
                home_history,
                date,
                home_goals,
                away_goals,
                home_points,
                away_elo,
                max(config.rolling_windows),
                home_performance,
            )
            _append_history(
                away_history,
                date,
                away_goals,
                home_goals,
                away_points,
                home_elo,
                max(config.rolling_windows),
                away_performance,
            )
            _append_history(
                home_venue_history,
                date,
                home_goals,
                away_goals,
                home_points,
                away_elo,
                max(config.rolling_windows),
                home_performance,
            )
            _append_history(
                away_venue_history,
                date,
                away_goals,
                home_goals,
                away_points,
                home_elo,
                max(config.rolling_windows),
                away_performance,
            )
            engine.update(home, away, home_goals, away_goals)

    return pd.DataFrame(output)


def build_fixture_features(
    completed_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    config: ModelingConfig | None = None,
) -> pd.DataFrame:
    """Replay completed history and return target-free pre-match fixture rows."""
    config = config or ModelingConfig()
    prepared = prepare_future_fixtures(fixtures, aliases=config.team_aliases)
    fixture_ids = set(prepared["match_id"].astype(str))
    if completed_matches["match_id"].astype(str).isin(fixture_ids).any():
        raise ValueError("I match_id delle fixture devono essere nuovi.")
    combined = pd.concat([completed_matches, prepared], ignore_index=True, sort=False)
    features = build_prematch_features(combined, config, include_unplayed=True)
    return (
        features.loc[features["match_id"].astype(str).isin(fixture_ids)]
        .sort_values(["date", "league", "match_id"])
        .reset_index(drop=True)
    )


def _rest_days(history: _TeamHistory, date: pd.Timestamp) -> float:
    if not history.dates:
        return float("nan")
    return float((date - history.dates[-1]).days)


def _append_history(
    history: _TeamHistory,
    date: pd.Timestamp,
    goals_for: int,
    goals_against: int,
    points: int,
    opponent_elo: float,
    maximum_window: int,
    performance: dict[str, float],
) -> None:
    history.dates.append(date)
    history.goals_for.append(goals_for)
    history.goals_against.append(goals_against)
    history.points.append(points)
    history.opponent_elo.append(opponent_elo)
    for field_name, value in performance.items():
        if field_name not in history.performance:
            history.performance[field_name] = deque()
        history.performance[field_name].append(value)
    for values in (
        history.goals_for,
        history.goals_against,
        history.points,
        history.opponent_elo,
        *history.performance.values(),
    ):
        while len(values) > maximum_window:
            values.popleft()
