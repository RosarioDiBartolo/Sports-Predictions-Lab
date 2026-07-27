"""Regularized confirmed-lineup corrections for Dixon-Coles goal rates."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .hybrid import (
    HYBRID_MODEL_NAME,
    HybridPredictor,
    _score_probabilities,
    fit_hybrid_model,
    walk_forward_hybrid_model,
)
from .models import (
    OUTCOMES,
    PROBABILITY_COLUMNS,
    _probability_metrics,
    _promotion_evidence,
    _weighted_summary,
    paired_log_loss_bootstrap,
)

CONFIRMED_LINEUP_MODEL_NAME = "dixon_coles_confirmed_lineup_pooling"
CONFIRMED_LINEUP_BASELINE_NAME = "dixon_coles_without_confirmed_lineup"
LINEUP_FEATURE_MARKER = "_confirmed_lineup_"
DEPARTMENTS = ("goalkeeper", "defense", "midfield", "attack")


def _department(position: object) -> str:
    value = str(position or "").strip().upper()
    if value == "ALL":
        return "unknown"
    if value in {"G", "GK", "GOALKEEPER"} or "GOALKEEPER" in value:
        return "goalkeeper"
    if (
        value.startswith("D")
        or value in {"CB", "LB", "RB", "LWB", "RWB"}
        or "BACK" in value
        or "DEFENDER" in value
    ):
        return "defense"
    if (
        value.startswith("M")
        or value in {"CM", "DM", "AM", "LM", "RM", "CDM", "CAM"}
        or "MIDFIELD" in value
    ):
        return "midfield"
    if (
        value.startswith(("F", "A"))
        or value in {"ST", "CF", "LW", "RW", "SS"}
        or "FORWARD" in value
        or "WING" in value
    ):
        return "attack"
    raise ValueError(f"Posizione titolare non riconosciuta: {position!r}")


def _parse_starters(value: object, *, side: str) -> list[dict[str, str]]:
    try:
        players = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON titolari {side} non valido.") from error
    if not isinstance(players, list) or len(players) != 11:
        raise ValueError(f"La lineup {side} deve contenere esattamente 11 titolari.")
    normalized = [
        {
            "player_id": str(player["player_id"]),
            "department": _department(player.get("position")),
        }
        for player in players
        if isinstance(player, dict) and player.get("player_id")
    ]
    if len(normalized) != 11 or len({row["player_id"] for row in normalized}) != 11:
        raise ValueError(f"La lineup {side} contiene ID mancanti o duplicati.")
    return normalized


def _player_values(
    player_id: str,
    kickoff: pd.Timestamp,
    states: dict[str, dict[str, Any]],
) -> dict[str, float]:
    state = states.get(player_id)
    if not state:
        return {
            "strength": 0.0,
            "goal_difference": 0.0,
            "experience": 0.0,
            "days_since": float("nan"),
            "observed": 0.0,
        }
    starts = max(int(state["starts"]), 1)
    return {
        "strength": float(state["points"]) / (3.0 * starts),
        "goal_difference": float(state["goal_difference"]) / starts,
        "experience": float(np.log1p(starts)),
        "days_since": float((kickoff - state["last_date"]).total_seconds() / 86400),
        "observed": 1.0,
    }


def _pool_lineup(
    players: list[dict[str, str]],
    kickoff: pd.Timestamp,
    player_states: dict[str, dict[str, Any]],
    pair_starts: dict[tuple[str, str], int],
    previous_starters: set[str],
) -> dict[str, float]:
    values: list[dict[str, Any]] = [
        {
            **_player_values(player["player_id"], kickoff, player_states),
            "department": player["department"],
        }
        for player in players
    ]
    pooled: dict[str, float] = {}
    for name in ("strength", "goal_difference", "experience", "days_since"):
        observed = np.asarray(
            [
                float(row[name])
                for row in values
                if np.isfinite(float(row[name]))
            ],
            dtype=float,
        )
        pooled[f"{name}_mean"] = (
            float(observed.mean()) if observed.size else float("nan")
        )
        pooled[f"{name}_std"] = (
            float(observed.std()) if observed.size else float("nan")
        )
    pooled["observation_coverage"] = float(
        np.mean([row["observed"] for row in values])
    )
    pooled["position_coverage"] = float(
        np.mean([row["department"] != "unknown" for row in values])
    )
    current = {player["player_id"] for player in players}
    pooled["continuity"] = len(current & previous_starters) / 11.0
    shared_counts = []
    for first, second in combinations(sorted(current), 2):
        shared_counts.append(pair_starts.get((first, second), 0))
    pooled["shared_experience"] = float(np.mean(shared_counts))
    for department in DEPARTMENTS:
        department_values = [
            row for row in values if row["department"] == department
        ]
        pooled[f"{department}_count"] = float(len(department_values))
        pooled[f"{department}_coverage"] = (
            float(np.mean([row["observed"] for row in department_values]))
            if department_values
            else 0.0
        )
        for name in ("strength", "goal_difference", "experience"):
            observations = np.asarray(
                [row[name] for row in department_values], dtype=float
            )
            pooled[f"{department}_{name}_mean"] = (
                float(observations.mean())
                if observations.size
                else float("nan")
            )
            pooled[f"{department}_{name}_std"] = (
                float(observations.std())
                if observations.size
                else float("nan")
            )
    return pooled


def build_confirmed_lineup_pooling_features(
    match_features: pd.DataFrame,
    lineup_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Attach confirmed-lineup pools built strictly before each kickoff."""
    match_required = {
        "match_id", "date", "season", "league", "home_team", "away_team",
        "home_goals", "away_goals", "result",
    }
    lineup_required = {"match_id", "home_starters", "away_starters"}
    match_missing = match_required.difference(match_features.columns)
    lineup_missing = lineup_required.difference(lineup_dataset.columns)
    if match_missing or lineup_missing:
        raise ValueError(
            "Contratto pooling incompleto: "
            f"matches={sorted(match_missing)}, lineups={sorted(lineup_missing)}"
        )
    if lineup_dataset["match_id"].duplicated().any():
        raise ValueError("Il dataset lineup contiene match_id duplicati.")
    lineup_columns = ["match_id", "home_starters", "away_starters"]
    data = match_features.merge(
        lineup_dataset[lineup_columns],
        on="match_id",
        how="inner",
        validate="one_to_one",
    ).copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, format="mixed")
    data = data.sort_values(["date", "match_id"]).reset_index(drop=True)

    player_states: dict[str, dict[str, Any]] = {}
    pair_starts: dict[tuple[str, str], int] = defaultdict(int)
    previous_by_team: dict[str, set[str]] = defaultdict(set)
    snapshots: list[dict[str, float | str]] = []
    for _, simultaneous in data.groupby("date", sort=False):
        parsed: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in simultaneous.itertuples(index=False):
            values: dict[str, float | str] = {"match_id": str(row.match_id)}
            side_pools: dict[str, dict[str, float]] = {}
            for side in ("home", "away"):
                team = str(getattr(row, f"{side}_team"))
                players = _parse_starters(
                    getattr(row, f"{side}_starters"), side=side
                )
                parsed[(str(row.match_id), side)] = players
                side_pools[side] = _pool_lineup(
                    players,
                    pd.Timestamp(str(row.date)),
                    player_states,
                    pair_starts,
                    previous_by_team[team],
                )
                values.update(
                    {
                        f"{side}{LINEUP_FEATURE_MARKER}{name}": value
                        for name, value in side_pools[side].items()
                    }
                )
            shared_names = set(side_pools["home"]) & set(side_pools["away"])
            for name in shared_names:
                values[f"diff{LINEUP_FEATURE_MARKER}{name}"] = (
                    side_pools["home"][name] - side_pools["away"][name]
                )
            values[f"home{LINEUP_FEATURE_MARKER}attack_vs_defense"] = (
                side_pools["home"]["attack_strength_mean"]
                - side_pools["away"]["defense_strength_mean"]
            )
            values[f"away{LINEUP_FEATURE_MARKER}attack_vs_defense"] = (
                side_pools["away"]["attack_strength_mean"]
                - side_pools["home"]["defense_strength_mean"]
            )
            snapshots.append(values)

        for row in simultaneous.itertuples(index=False):
            if (
                row.result not in OUTCOMES
                or pd.isna(row.home_goals)
                or pd.isna(row.away_goals)
            ):
                continue
            for side, opponent in (("home", "away"), ("away", "home")):
                players = parsed[(str(row.match_id), side)]
                team = str(getattr(row, f"{side}_team"))
                goals_for = float(getattr(row, f"{side}_goals"))
                goals_against = float(getattr(row, f"{opponent}_goals"))
                points = 3.0 if goals_for > goals_against else 0.0
                if goals_for == goals_against:
                    points = 1.0
                player_ids = {player["player_id"] for player in players}
                for player_id in player_ids:
                    state = player_states.setdefault(
                        player_id,
                        {
                            "starts": 0,
                            "points": 0.0,
                            "goal_difference": 0.0,
                            "last_date": row.date,
                        },
                    )
                    state["starts"] += 1
                    state["points"] += points
                    state["goal_difference"] += goals_for - goals_against
                    state["last_date"] = row.date
                for first, second in combinations(sorted(player_ids), 2):
                    pair_starts[(first, second)] += 1
                previous_by_team[team] = player_ids
    pooled = pd.DataFrame(snapshots)
    return data.drop(columns=["home_starters", "away_starters"]).merge(
        pooled,
        on="match_id",
        how="left",
        validate="one_to_one",
    )


def confirmed_lineup_feature_columns(columns: pd.Index) -> list[str]:
    """Return the reviewed numeric inputs produced from official lineups."""
    return sorted(
        column
        for column in columns
        if isinstance(column, str) and LINEUP_FEATURE_MARKER in column
    )


def _correction_estimator(alpha: float) -> Pipeline:
    if alpha <= 0:
        raise ValueError("alpha deve essere positivo.")
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _base_rates(
    predictor: HybridPredictor,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    home = np.clip(predictor.home_goal_model.predict(features), 1e-6, 8.0)
    away = np.clip(predictor.away_goal_model.predict(features), 1e-6, 8.0)
    return home, away


@dataclass
class ConfirmedLineupPredictor:
    """Dixon-Coles with bounded, regularized lineup adjustments."""

    base: HybridPredictor
    home_correction: Pipeline
    away_correction: Pipeline
    lineup_features: tuple[str, ...]
    trained_seasons: tuple[str, ...]
    alpha: float
    maximum_log_correction: float = 0.35

    def goal_rates(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = set(self.lineup_features).difference(features.columns)
        if missing:
            raise ValueError(
                f"Feature formazione confermata mancanti: {sorted(missing)}"
            )
        base_home, base_away = _base_rates(self.base, features)
        lineup = features[list(self.lineup_features)]
        home_delta = np.clip(
            self.home_correction.predict(lineup),
            -self.maximum_log_correction,
            self.maximum_log_correction,
        )
        away_delta = np.clip(
            self.away_correction.predict(lineup),
            -self.maximum_log_correction,
            self.maximum_log_correction,
        )
        return (
            np.clip(base_home * np.exp(home_delta), 1e-6, 8.0),
            np.clip(base_away * np.exp(away_delta), 1e-6, 8.0),
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        home_rates, away_rates = self.goal_rates(features)
        return np.vstack(
            [
                _score_probabilities(home, away, self.base.rho)
                for home, away in zip(home_rates, away_rates, strict=True)
            ]
        )


@dataclass
class ConfirmedLineupModelResult:
    """Evaluation artifacts and the fitted confirmed-lineup candidate."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    outputs: dict[str, Path]
    predictor: ConfirmedLineupPredictor


def fit_confirmed_lineup_model(
    features: pd.DataFrame,
    *,
    base_features: pd.DataFrame | None = None,
    alpha: float = 25.0,
    maximum_log_correction: float = 0.35,
) -> ConfirmedLineupPredictor:
    """Fit a no-lineup Dixon-Coles base and two shrinkage corrections."""
    lineup_features = confirmed_lineup_feature_columns(features.columns)
    if not lineup_features:
        raise ValueError(
            "Nessuna feature *_confirmed_lineup_* disponibile per il training."
        )
    required = {
        "season",
        "home_goals",
        "away_goals",
        "result",
        *lineup_features,
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Colonne modello lineup mancanti: {sorted(missing)}")
    data = features.loc[features["result"].isin(OUTCOMES)].copy()
    if data.empty or data[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("Il training lineup richiede gol finali completi.")

    correction_frame = data.drop(columns=lineup_features)
    base_training = (
        base_features.copy()
        if base_features is not None
        else correction_frame.copy()
    )
    base = fit_hybrid_model(base_training, include_player_features=False)
    base_home, base_away = _base_rates(base, correction_frame)
    home_target = np.log((data["home_goals"].to_numpy(float) + 0.5) / (base_home + 0.5))
    away_target = np.log((data["away_goals"].to_numpy(float) + 0.5) / (base_away + 0.5))
    home_correction = _correction_estimator(alpha)
    away_correction = _correction_estimator(alpha)
    home_correction.fit(data[lineup_features], home_target)
    away_correction.fit(data[lineup_features], away_target)
    return ConfirmedLineupPredictor(
        base=base,
        home_correction=home_correction,
        away_correction=away_correction,
        lineup_features=tuple(lineup_features),
        trained_seasons=tuple(sorted(data["season"].astype(str).unique())),
        alpha=alpha,
        maximum_log_correction=maximum_log_correction,
    )


def _baseline_probabilities(
    predictor: ConfirmedLineupPredictor,
    features: pd.DataFrame,
) -> np.ndarray:
    home_rates, away_rates = _base_rates(predictor.base, features)
    return np.vstack(
        [
            _score_probabilities(home, away, predictor.base.rho)
            for home, away in zip(home_rates, away_rates, strict=True)
        ]
    )


def _rps(results: pd.Series, probabilities: np.ndarray) -> float:
    indices = results.map({value: index for index, value in enumerate(OUTCOMES)})
    observed = np.eye(len(OUTCOMES))[indices.to_numpy(dtype=int)]
    return float(
        np.mean(
            np.sum(
                (np.cumsum(probabilities, axis=1)[:, :-1]
                 - np.cumsum(observed, axis=1)[:, :-1])
                ** 2,
                axis=1,
            )
            / (len(OUTCOMES) - 1)
        )
    )


def _metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, model), group in predictions.groupby(["season", "model"]):
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        rows.append(
            {
                "season": str(season),
                "model": str(model),
                **_probability_metrics(group["result"], probabilities),
                "rps": _rps(group["result"], probabilities),
                "calibrated": False,
            }
        )
    return pd.DataFrame(rows)


def walk_forward_confirmed_lineup_model(
    features: pd.DataFrame,
    *,
    base_features: pd.DataFrame | None = None,
    alpha: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate lineup correction and its no-lineup base on identical folds."""
    required = {"season", "match_id", "league", "result"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Colonne valutazione lineup mancanti: {sorted(missing)}")
    data = features.loc[features["result"].isin(OUTCOMES)].copy()
    data["season"] = data["season"].astype(str).str.zfill(4)
    full_history = (
        base_features.copy()
        if base_features is not None
        else data.drop(
            columns=confirmed_lineup_feature_columns(data.columns),
            errors="ignore",
        )
    )
    full_history["season"] = full_history["season"].astype(str).str.zfill(4)
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for season in sorted(data["season"].unique())[1:]:
        training = data.loc[data["season"].lt(season)]
        test = data.loc[data["season"].eq(season)]
        if set(training["result"]) != set(OUTCOMES) or test.empty:
            continue
        base_training = full_history.loc[full_history["season"].lt(season)]
        predictor = fit_confirmed_lineup_model(
            training,
            base_features=base_training,
            alpha=alpha,
        )
        variants = (
            (CONFIRMED_LINEUP_MODEL_NAME, predictor.predict_proba(test)),
            (CONFIRMED_LINEUP_BASELINE_NAME, _baseline_probabilities(predictor, test)),
        )
        for model_name, probabilities in variants:
            metrics.append(
                {
                    "season": season,
                    "model": model_name,
                    **_probability_metrics(test["result"], probabilities),
                    "rps": _rps(test["result"], probabilities),
                    "calibrated": False,
                }
            )
            frame = test[["match_id", "season", "league", "result"]].copy()
            frame["model"] = model_name
            frame[list(PROBABILITY_COLUMNS)] = probabilities
            predictions.append(frame)
    metric_columns = [
        "season", "model", "matches", "log_loss", "brier", "rps",
        "accuracy", "ece", "calibrated",
    ]
    prediction_columns = [
        "match_id", "season", "league", "result", "model",
        *PROBABILITY_COLUMNS,
    ]
    return (
        pd.DataFrame(metrics, columns=metric_columns),
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=prediction_columns),
    )


def export_confirmed_lineup_model(
    features: pd.DataFrame,
    destination: Path,
    *,
    base_features: pd.DataFrame | None = None,
    alpha: float = 25.0,
) -> ConfirmedLineupModelResult:
    """Persist the candidate only after the common walk-forward evaluation."""
    destination.mkdir(parents=True, exist_ok=True)
    metrics, all_predictions = walk_forward_confirmed_lineup_model(
        features, base_features=base_features, alpha=alpha
    )
    candidate_metrics = metrics.loc[
        metrics["model"].eq(CONFIRMED_LINEUP_MODEL_NAME)
    ].copy()
    baseline_metrics = metrics.loc[
        metrics["model"].eq(CONFIRMED_LINEUP_BASELINE_NAME)
    ].copy()
    candidate_predictions = all_predictions.loc[
        all_predictions["model"].eq(CONFIRMED_LINEUP_MODEL_NAME)
    ].copy()
    baseline_predictions = all_predictions.loc[
        all_predictions["model"].eq(CONFIRMED_LINEUP_BASELINE_NAME)
    ].copy()
    bootstrap = paired_log_loss_bootstrap(
        candidate_predictions, baseline_predictions
    )
    official_metrics = pd.DataFrame()
    official_predictions = pd.DataFrame()
    bootstrap_official: dict[str, float | int | str] = {
        "matches": 0,
        "verdict": "insufficient_data",
    }
    if base_features is not None:
        _, official_all = walk_forward_hybrid_model(base_features)
        official_predictions = official_all.loc[
            official_all["match_id"].isin(candidate_predictions["match_id"])
        ].copy()
        official_predictions["model"] = HYBRID_MODEL_NAME
        official_metrics = _metrics_from_predictions(official_predictions)
        bootstrap_official = paired_log_loss_bootstrap(
            candidate_predictions,
            official_predictions,
        )
    comparison_metrics = pd.concat(
        [metrics, official_metrics],
        ignore_index=True,
    )
    summary = _weighted_summary(comparison_metrics)
    summary["rps"] = [
        float(
            np.average(
                comparison_metrics.loc[
                    comparison_metrics["model"].eq(model), "rps"
                ],
                weights=comparison_metrics.loc[
                    comparison_metrics["model"].eq(model), "matches"
                ],
            )
        )
        for model in summary["model"]
    ]
    promotion = _promotion_evidence(
        summary,
        summary,
        candidate_metrics,
        baseline_metrics,
        bootstrap,
        candidate_model=CONFIRMED_LINEUP_MODEL_NAME,
        baseline_model=CONFIRMED_LINEUP_BASELINE_NAME,
    )
    candidate_rps = summary.loc[
        summary["model"].eq(CONFIRMED_LINEUP_MODEL_NAME), "rps"
    ]
    baseline_rps = summary.loc[
        summary["model"].eq(CONFIRMED_LINEUP_BASELINE_NAME), "rps"
    ]
    rps_not_worse = bool(
        not candidate_rps.empty
        and not baseline_rps.empty
        and candidate_rps.iloc[0] <= baseline_rps.iloc[0]
    )
    promotion["rps_not_worse"] = rps_not_worse
    promotion["promoted"] = bool(promotion["promoted"] and rps_not_worse)
    predictor = fit_confirmed_lineup_model(
        features,
        base_features=base_features,
        alpha=alpha,
    )

    paths = {
        "metrics": destination / "confirmed_lineup_metrics_by_season.csv",
        "predictions": destination / "confirmed_lineup_predictions.csv",
        "comparison": destination / "confirmed_lineup_comparison.csv",
        "bootstrap": destination / "confirmed_lineup_bootstrap.json",
        "metadata": destination / "confirmed_lineup_model.meta.json",
        "model": destination / "confirmed_lineup_model.joblib",
        "report": destination / "CONFIRMED_LINEUP_MODEL_REPORT.md",
    }
    metrics.to_csv(paths["metrics"], index=False)
    candidate_predictions.to_csv(paths["predictions"], index=False)
    summary.to_csv(paths["comparison"], index=False)
    paths["bootstrap"].write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "model_name": CONFIRMED_LINEUP_MODEL_NAME,
        "algorithm": "Dixon-Coles + regularized confirmed-lineup pooling correction",
        "official_model_unchanged": True,
        "alpha": alpha,
        "maximum_log_correction": predictor.maximum_log_correction,
        "lineup_features": list(predictor.lineup_features),
        "trained_seasons": list(predictor.trained_seasons),
        "base_training_matches": int(
            len(base_features) if base_features is not None else len(features)
        ),
        "lineup_training_matches": int(len(features)),
        "excluded_inputs": [
            "odds",
            "market probabilities",
            "current-match performance",
        ],
        "bootstrap_vs_no_lineup": bootstrap,
        "bootstrap_vs_official": bootstrap_official,
        "promotion": promotion,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(predictor, paths["model"])
    report = [
        "# Dixon-Coles + correzione lineup confermata",
        "",
        "Il candidato non sostituisce il modello ufficiale senza promotion gate OOS.",
        "",
        "## Valutazione walk-forward",
        "",
        "| Modello | Match | Log Loss | Brier | RPS | Accuracy | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        report.append(
            f"| {row['model']} | {int(row['matches'])} | "
            f"{row['log_loss']:.4f} | {row['brier']:.4f} | "
            f"{row['rps']:.4f} | {row['accuracy']:.2%} | {row['ece']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Promotion gate",
            "",
            f"- IC 95% Log Loss favorevole: {promotion['significant_log_loss']}.",
            f"- Stagioni vinte: {promotion['season_wins']}/"
            f"{len(candidate_metrics)}.",
            f"- Brier non peggiore: {promotion['brier_not_worse']}.",
            f"- RPS non peggiore: {promotion['rps_not_worse']}.",
            f"- ECE non peggiore: {promotion['ece_not_worse']}.",
            f"- Verdetto: {'PROMOSSO' if promotion['promoted'] else 'NON PROMOSSO'}.",
        ]
    )
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    return ConfirmedLineupModelResult(
        candidate_metrics,
        candidate_predictions,
        paths,
        predictor,
    )


def load_confirmed_lineup_model(path: Path) -> ConfirmedLineupPredictor:
    predictor = joblib.load(path)
    if not isinstance(predictor, ConfirmedLineupPredictor):
        raise TypeError(f"Artefatto modello lineup non valido: {path}")
    return predictor
