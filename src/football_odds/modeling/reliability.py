"""Deterministic reliability gate for player-informed rate corrections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HISTORY_OBSERVATION_THRESHOLD = 5.0
RELIABLE_QUALITY_THRESHOLD = 0.7


@dataclass(frozen=True)
class ReliabilityScores:
    """Team-level components and the conservative match-level score."""

    history_depth: np.ndarray
    timing_coverage: np.ndarray
    reliable_starters: np.ndarray
    team: np.ndarray
    match: np.ndarray


def reliability_scores(
    starters: np.ndarray,
    feature_names: tuple[str, ...],
) -> ReliabilityScores:
    """Score input reliability without learning from outcomes.

    ``starters`` has shape ``(matches, teams, 11, features)``. Team scores use
    the weakest component; match scores use the weaker of the two teams.
    """
    if starters.ndim != 4 or starters.shape[1:3] != (2, 11):
        raise ValueError("Il gate richiede tensori (partite, 2, 11, feature).")
    index = {name: position for position, name in enumerate(feature_names)}
    required = {
        "log_squad_observations",
        "fs_store_available",
        "fs_observations_value",
        "fs_observations_quality",
        "fs_mean_minutes_available",
        "fs_mean_minute_in_available",
        "fs_mean_minute_out_available",
    }
    missing = required.difference(index)
    if missing:
        raise ValueError(f"Feature del gate mancanti: {sorted(missing)}")

    observations = np.expm1(
        np.maximum(starters[..., index["log_squad_observations"]], 0.0)
    )
    history_depth = np.clip(
        observations / HISTORY_OBSERVATION_THRESHOLD,
        0.0,
        1.0,
    ).mean(axis=2)

    minutes = starters[..., index["fs_mean_minutes_available"]] > 0.5
    substitution = np.maximum(
        starters[..., index["fs_mean_minute_in_available"]],
        starters[..., index["fs_mean_minute_out_available"]],
    ) > 0.5
    timing_coverage = (minutes & substitution).mean(axis=2)

    store = starters[..., index["fs_store_available"]] > 0.5
    canonical_observations = np.expm1(
        np.maximum(starters[..., index["fs_observations_value"]], 0.0)
    )
    quality = starters[..., index["fs_observations_quality"]]
    reliable_starters = (
        store
        & (canonical_observations >= HISTORY_OBSERVATION_THRESHOLD)
        & (quality >= RELIABLE_QUALITY_THRESHOLD)
    ).mean(axis=2)

    team = np.minimum.reduce(
        [history_depth, timing_coverage, reliable_starters]
    )
    match = team.min(axis=1)
    return ReliabilityScores(
        history_depth=history_depth,
        timing_coverage=timing_coverage,
        reliable_starters=reliable_starters,
        team=team,
        match=match,
    )


def attenuate_corrections(
    bounded_corrections: np.ndarray,
    match_reliability: np.ndarray,
) -> np.ndarray:
    """Attenuate already-bounded home/away log-rate corrections."""
    if bounded_corrections.ndim != 2 or bounded_corrections.shape[1] != 2:
        raise ValueError("Le correzioni devono avere forma (partite, 2).")
    reliability = np.asarray(match_reliability, dtype=float)
    if reliability.shape != (len(bounded_corrections),):
        raise ValueError("Gate e correzioni non sono allineati.")
    if not np.isfinite(reliability).all() or np.any(
        (reliability < 0.0) | (reliability > 1.0)
    ):
        raise ValueError("Il reliability score deve essere finito e in [0, 1].")
    return bounded_corrections * reliability[:, None]
