"""Observable, fail-closed runner for the two gated model candidates."""

from __future__ import annotations

import hashlib
import json
import platform
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

CANDIDATE_NAMES = (
    "dixon_coles_shared_encoder_pooling_gated",
    "dixon_coles_gated_tabular_residual",
)
ABLATIONS = ("base", "feature_store", "bench", "combined")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _append_event(run_dir: Path, event: dict[str, object]) -> None:
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), **event}) + "\n")


def _lineup_checks(lineups: pd.DataFrame) -> dict[str, object]:
    required = {
        "match_id",
        "season",
        "league",
        "home_starters",
        "away_starters",
    }
    if missing := required.difference(lineups.columns):
        return {"passed": False, "missing_columns": sorted(missing)}
    invalid = 0
    duplicate_players = 0
    for row in lineups.itertuples():
        for side in ("home_starters", "away_starters"):
            try:
                players = json.loads(getattr(row, side))
                identifiers = [str(player["player_id"]) for player in players]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
                continue
            if len(identifiers) != 11:
                invalid += 1
            if len(set(identifiers)) != len(identifiers):
                duplicate_players += 1
    return {
        "passed": invalid == 0 and duplicate_players == 0,
        "matches": len(lineups),
        "invalid_team_lineups": invalid,
        "team_lineups_with_duplicate_players": duplicate_players,
    }


def _reconciled_snapshots(project: Path) -> list[Path]:
    candidates = [
        *(project / "data/raw/api_football_odds").glob("*.csv"),
        *(project / "data/raw/beat_the_bookie").glob("*snapshot.csv"),
    ]
    reconciled = []
    for path in candidates:
        if "match_id" in pd.read_csv(path, nrows=0).columns:
            reconciled.append(path)
    return sorted(reconciled, key=lambda path: path.stat().st_mtime)


def _temporal_checks(
    path: Path,
    eligible_matches: set[str],
) -> dict[str, object]:
    required = {
        "match_id",
        "player_id",
        "team_id",
        "kickoff",
        "current_lineup_role",
        "current_observation_quality",
        "current_source",
    }
    if not path.is_file():
        return {"passed": False, "missing_columns": sorted(required)}
    header = pd.read_csv(path, nrows=0)
    missing = required.difference(header.columns)
    if missing:
        return {"passed": False, "missing_columns": sorted(missing)}
    availability = [name for name in header.columns if name.endswith("_available")]
    fallback = [name for name in header.columns if name.endswith("_fallback_kind")]
    quality = [name for name in header.columns if name.endswith("_quality")]
    rows = 0
    duplicate_keys = 0
    orphan_rows = 0
    invalid_identifiers = 0
    invalid_kickoffs = 0
    available_values = 0
    availability_values = 0
    fallback_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()
    usecols = list(required) + availability + fallback + quality
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
        rows += len(chunk)
        identifiers = chunk[["match_id", "player_id", "team_id"]].astype(str)
        invalid_identifiers += int(
            identifiers.apply(lambda column: column.str.strip().eq(""))
            .any(axis=1)
            .sum()
        )
        orphan_rows += int(
            (~identifiers["match_id"].isin(eligible_matches)).sum()
        )
        invalid_kickoffs += int(
            pd.to_datetime(chunk["kickoff"], utc=True, errors="coerce").isna().sum()
        )
        for key in identifiers.itertuples(index=False, name=None):
            if key in seen:
                duplicate_keys += 1
            seen.add(key)
        for column in availability:
            values = chunk[column].astype(str).str.lower()
            availability_values += len(values)
            available_values += int(values.isin({"true", "1", "1.0"}).sum())
        for column in fallback:
            for value, count in chunk[column].fillna("missing").value_counts().items():
                fallback_counts[str(value)] = fallback_counts.get(str(value), 0) + int(
                    count
                )
        for column in quality:
            for value, count in chunk[column].fillna("missing").value_counts().items():
                quality_counts[str(value)] = quality_counts.get(str(value), 0) + int(
                    count
                )
    return {
        "passed": duplicate_keys == 0
        and orphan_rows == 0
        and invalid_identifiers == 0
        and invalid_kickoffs == 0,
        "rows": rows,
        "unique_match_player_team_keys": len(seen),
        "duplicate_keys": duplicate_keys,
        "orphan_rows": orphan_rows,
        "invalid_identifiers": invalid_identifiers,
        "invalid_kickoffs": invalid_kickoffs,
        "availability_rate": (
            available_values / availability_values if availability_values else None
        ),
        "fallback_counts": fallback_counts,
        "quality_counts": quality_counts,
    }


def _fold_manifest(matches: pd.DataFrame, common: set[str]) -> list[dict[str, object]]:
    eligible = matches.loc[matches["match_id"].astype(str).isin(common)].copy()
    eligible["season"] = eligible["season"].astype(str).str.zfill(4)
    rows = []
    for season in sorted(eligible["season"].unique()):
        train_ids = sorted(
            eligible.loc[eligible["season"].lt(season), "match_id"].astype(str)
        )
        test_ids = sorted(
            eligible.loc[eligible["season"].eq(season), "match_id"].astype(str)
        )
        rows.append(
            {
                "test_season": season,
                "train_matches": len(train_ids),
                "test_matches": len(test_ids),
                "train_match_ids_sha256": hashlib.sha256(
                    "\n".join(train_ids).encode()
                ).hexdigest(),
                "test_match_ids_sha256": hashlib.sha256(
                    "\n".join(test_ids).encode()
                ).hexdigest(),
            }
        )
    return rows


def create_preflight(
    project: Path,
    run_dir: Path,
    *,
    epochs: int,
    embedding_dim: int,
    seed: int,
    max_iter: int = 100,
    device: str = "cpu",
    candidates: tuple[str, ...] = CANDIDATE_NAMES,
    ablations: tuple[str, ...] = ABLATIONS,
) -> dict[str, Any]:
    """Write the immutable preflight report for the common candidate run."""
    paths = {
        "matches": project / "data/processed/modeling_features_all.csv",
        "lineups": project / "data/processed/player_training_ready.csv",
        "temporal": project / "data/processed/player_match_temporal_features.csv",
    }
    snapshots = _reconciled_snapshots(project)
    checks: list[dict[str, object]] = []
    for name, path in paths.items():
        checks.append(
            {
                "name": f"input_exists:{name}",
                "passed": path.is_file(),
                "path": str(path),
            }
        )
    inputs = [_fingerprint(path) for path in paths.values() if path.is_file()]
    inputs.extend(_fingerprint(snapshot) for snapshot in snapshots)
    matches = (
        pd.read_csv(paths["matches"]) if paths["matches"].is_file() else pd.DataFrame()
    )
    lineups = (
        pd.read_csv(paths["lineups"]) if paths["lineups"].is_file() else pd.DataFrame()
    )
    checks.append({"name": "lineups", **_lineup_checks(lineups)})
    required_match_columns = {
        "match_id",
        "date",
        "season",
        "league",
        "home_goals",
        "away_goals",
        "result",
    }
    checks.append(
        {
            "name": "match_schema",
            "passed": required_match_columns.issubset(matches.columns),
            "missing_columns": sorted(
                required_match_columns.difference(matches.columns)
            ),
        }
    )
    if not matches.empty and "match_id" in matches:
        checks.append(
            {
                "name": "unique_match_ids",
                "passed": not matches["match_id"].duplicated().any(),
                "duplicates": int(matches["match_id"].duplicated().sum()),
            }
        )
        numeric = matches.select_dtypes(include=[np.number])
        checks.append(
            {
                "name": "no_infinite_match_values",
                "passed": not np.isinf(numeric.to_numpy(float)).any(),
                "nan_values": int(numeric.isna().sum().sum()),
            }
        )
        outcomes = matches["result"].value_counts(dropna=False).to_dict()
        checks.append(
            {
                "name": "target_distribution",
                "passed": set(outcomes).issubset({"H", "D", "A"})
                and {"H", "D", "A"}.issubset(outcomes),
                "counts": {str(key): int(value) for key, value in outcomes.items()},
            }
        )
    common = (
        set(matches.get("match_id", pd.Series(dtype=str)).astype(str))
        & set(lineups.get("match_id", pd.Series(dtype=str)).astype(str))
    )
    seasons = sorted(
        lineups.loc[lineups["match_id"].astype(str).isin(common), "season"]
        .astype(str)
        .str.zfill(4)
        .unique()
    ) if not lineups.empty and "match_id" in lineups else []
    checks.append(
        {
            "name": "temporal_folds",
            "passed": len(seasons) >= 3,
            "seasons": seasons,
            "eligible_matches": len(common),
        }
    )
    checks.append(
        {
            "name": "temporal_feature_store",
            **_temporal_checks(paths["temporal"], common),
        }
    )
    fold_manifest = _fold_manifest(matches, common) if not matches.empty else []
    snapshot_rows = (
        pd.concat((pd.read_csv(path) for path in snapshots), ignore_index=True)
        if snapshots
        else pd.DataFrame()
    )
    timestamp_valid = (
        snapshot_rows.get("provider_updated_at", pd.Series(dtype=str))
        .pipe(pd.to_datetime, utc=True, errors="coerce")
        .notna()
    )
    cutoff = pd.to_datetime(
        snapshot_rows.get("prediction_cutoff", pd.Series(dtype=str)),
        utc=True,
        errors="coerce",
    )
    updated = pd.to_datetime(
        snapshot_rows.get("provider_updated_at", pd.Series(dtype=str)),
        utc=True,
        errors="coerce",
    )
    cutoff_valid = cutoff.notna() & updated.le(cutoff)
    complete_markets = (
        snapshot_rows.groupby(["match_id", "bookmaker"])["selection"].nunique().eq(3)
        if {"match_id", "bookmaker", "selection"}.issubset(snapshot_rows.columns)
        else pd.Series(dtype=bool)
    )
    reconciled = (
        set(snapshot_rows["match_id"].astype(str)) & common
        if "match_id" in snapshot_rows
        else set()
    )
    checks.append(
        {
            "name": "bookmaker_snapshot_at_cutoff",
            "passed": bool(reconciled)
            and bool(timestamp_valid.all())
            and bool(cutoff_valid.all())
            and bool(complete_markets.all()),
            "snapshots": [str(path) for path in snapshots],
            "timestamped_rows": int(timestamp_valid.sum()),
            "cutoff_valid_rows": int(cutoff_valid.sum()),
            "complete_match_bookmaker_markets": int(complete_markets.sum()),
            "eligible_match_overlap": len(reconciled),
            "required": "reconciled H/D/A snapshot timestamped at or before cutoff",
        }
    )
    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "run_id": run_dir.name,
        "created_at": _now(),
        "passed": passed,
        "inputs": inputs,
        "dataset_version": inputs[0]["sha256"] if inputs else None,
        "models": list(candidates),
        "ablations": list(ablations),
        "config": {
            "epochs": epochs,
            "embedding_dim": embedding_dim,
            "seed": seed,
            "max_iter": max_iter,
            "device": device,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "estimated_candidate_fits": len(candidates) * len(ablations),
        },
        "folds": fold_manifest,
        "checks": checks,
    }
    _write_json(run_dir / "preflight.json", report)
    return report


@contextmanager
def _heartbeat(run_dir: Path, state: dict[str, Any]) -> Iterator[None]:
    stop = threading.Event()
    lock = threading.Lock()

    def pulse() -> None:
        while not stop.wait(30):
            with lock:
                state["latest_heartbeat"] = _now()
                _write_json(run_dir / "run.json", state)
                _append_event(
                    run_dir,
                    {
                        "event": "heartbeat",
                        "phase": state.get("phase"),
                        "model": state.get("model"),
                        "ablation": state.get("ablation"),
                    },
                )

    thread = threading.Thread(target=pulse, name="candidate-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def _save_outputs(
    destination: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(destination / "metrics_by_season.csv", index=False)
    predictions.to_csv(destination / "predictions.csv", index=False)


def run_candidate_experiment(
    project: Path,
    *,
    epochs: int = 80,
    embedding_dim: int = 32,
    seed: int = 42,
    max_iter: int = 100,
    device: str = "cpu",
    candidates: tuple[str, ...] = CANDIDATE_NAMES,
    ablations: tuple[str, ...] = ABLATIONS,
    run_root: Path | None = None,
) -> Path:
    """Preflight, then train both candidates through identical temporal folds."""
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    unknown_candidates = set(candidates).difference(CANDIDATE_NAMES)
    unknown_ablations = set(ablations).difference(ABLATIONS)
    if unknown_candidates:
        raise ValueError(f"Unknown candidates: {sorted(unknown_candidates)}")
    if unknown_ablations:
        raise ValueError(f"Unknown ablations: {sorted(unknown_ablations)}")
    if not candidates or not ablations:
        raise ValueError("At least one candidate and ablation are required.")
    root = run_root or project / "reports/modeling/gated_comparison/runs"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "training.log").write_text("", encoding="utf-8")
    preflight = create_preflight(
        project,
        run_dir,
        epochs=epochs,
        embedding_dim=embedding_dim,
        seed=seed,
        max_iter=max_iter,
        device=device,
        candidates=candidates,
        ablations=ablations,
    )
    state: dict[str, Any] = {
        "run_id": run_id,
        "status": "running" if preflight["passed"] else "failed",
        "phase": "preflight",
        "started_at": _now(),
        "latest_heartbeat": _now(),
        "models": list(candidates),
        "ablations": list(ablations),
        "device": device,
    }
    _write_json(run_dir / "run.json", state)
    _append_event(run_dir, {"event": "preflight", "passed": preflight["passed"]})
    with (run_dir / "training.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"Preflight {'PASS' if preflight['passed'] else 'FAIL'}: "
            f"{run_dir / 'preflight.json'}\n"
        )
    if not preflight["passed"]:
        state["ended_at"] = _now()
        _write_json(run_dir / "run.json", state)
        _append_event(run_dir, {"event": "terminal", "status": "failed"})
        return run_dir

    try:
        from .gated_neural import walk_forward_gated_neural_lineup_model
        from .neural import (
            BASE_NEURAL_FEATURE_NAMES,
            build_player_tensor_dataset,
            cross_fitted_dixon_coles_rates,
        )
        from .reliability import reliability_scores
        from .tabular_residual import walk_forward_gated_tabular_residual

        matches = pd.read_csv(project / "data/processed/modeling_features_all.csv")
        lineups = pd.read_csv(project / "data/processed/player_training_ready.csv")
        temporal = pd.read_csv(
            project / "data/processed/player_match_temporal_features.csv"
        )
        tensors = build_player_tensor_dataset(
            matches, lineups, temporal_features=temporal
        )
        rates = cross_fitted_dixon_coles_rates(matches, tensors.matches)
        gate = reliability_scores(tensors.players, tensors.feature_names).match
        with _heartbeat(run_dir, state):
            for candidate in candidates:
                for ablation in ablations:
                    state.update(
                        phase="training", model=candidate, ablation=ablation
                    )
                    state["latest_heartbeat"] = _now()
                    _write_json(run_dir / "run.json", state)
                    _append_event(
                        run_dir,
                        {
                            "event": "stage_start",
                            "model": candidate,
                            "ablation": ablation,
                        },
                    )
                    started = time.monotonic()
                    if candidate == CANDIDATE_NAMES[0]:
                        metrics, predictions = (
                            walk_forward_gated_neural_lineup_model(
                                tensors,
                                matches,
                                embedding_dim=embedding_dim,
                                epochs=epochs,
                                rates=rates,
                                ablation=ablation,
                                device=device,
                            )
                        )
                    else:
                        metrics, predictions = walk_forward_gated_tabular_residual(
                            tensors,
                            rates,
                            gate,
                            tensors.feature_names,
                            BASE_NEURAL_FEATURE_NAMES,
                            ablation=ablation,
                            seed=seed,
                            max_iter=max_iter,
                        )
                    _save_outputs(
                        run_dir / "artifacts" / candidate / ablation,
                        metrics,
                        predictions,
                    )
                    _append_event(
                        run_dir,
                        {
                            "event": "stage_completed",
                            "model": candidate,
                            "ablation": ablation,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                    )
        state.update(status="completed", phase="completed", ended_at=_now())
    except BaseException as error:
        state.update(
            status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            phase="terminal",
            ended_at=_now(),
            error=repr(error),
        )
        raise
    finally:
        state["latest_heartbeat"] = _now()
        _write_json(run_dir / "run.json", state)
        _append_event(run_dir, {"event": "terminal", "status": state["status"]})
    return run_dir
