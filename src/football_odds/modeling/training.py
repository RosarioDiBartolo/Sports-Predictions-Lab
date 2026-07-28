"""Observable, fail-closed training lifecycle for the operational model."""

from __future__ import annotations

import hashlib
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .neural import build_player_tensor_dataset, export_neural_lineup_model


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fingerprint(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def run_training(project: Path, *, epochs: int, embedding_dim: int) -> Path:
    """Preflight inputs, then train while exposing durable run state."""
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = project / "reports/modeling/neural_lineup_model/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "matches": project / "data/processed/modeling_features.csv",
        "lineups": project / "data/processed/player_training_ready.csv",
        "temporal": project / "data/processed/player_match_temporal_features.csv",
        "market": project / "data/processed/market_predictions.csv",
    }
    required = (paths["matches"], paths["lineups"])
    checks: list[dict[str, object]] = []
    for path in required:
        checks.append({"name": f"exists:{path.name}", "passed": path.is_file()})
    inputs = [_fingerprint(path) for path in paths.values() if path.is_file()]
    matches = (
        pd.read_csv(paths["matches"]) if paths["matches"].is_file() else pd.DataFrame()
    )
    lineups = (
        pd.read_csv(paths["lineups"]) if paths["lineups"].is_file() else pd.DataFrame()
    )
    if not lineups.empty and {"match_id", "team", "lineup_role"}.issubset(lineups):
        starters = (
            lineups.loc[lineups["lineup_role"].eq("starter")]
            .groupby(["match_id", "team"])["player_id"]
            .nunique()
        )
        checks.append(
            {
                "name": "exactly_11_distinct_starters",
                "passed": bool(starters.eq(11).all()),
            }
        )
    numeric = matches.select_dtypes(include=[np.number])
    checks.append(
        {
            "name": "finite_numeric_features",
            "passed": bool(np.isfinite(numeric).all().all()),
        }
    )
    passed = all(bool(check["passed"]) for check in checks)
    preflight = {
        "run_id": run_id,
        "created_at": _now(),
        "passed": passed,
        "inputs": inputs,
        "checks": checks,
    }
    _write_json(run_dir / "preflight.json", preflight)
    state: dict[str, Any] = {
        "run_id": run_id,
        "status": "running" if passed else "failed",
        "started_at": _now(),
        "latest_heartbeat": _now(),
        "config": {"epochs": epochs, "embedding_dim": embedding_dim},
        "python": platform.python_version(),
    }
    _write_json(run_dir / "run.json", state)
    (run_dir / "events.jsonl").write_text(
        json.dumps({"at": _now(), "event": "preflight", "passed": passed}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "training.log").write_text(
        f"Preflight {'PASS' if passed else 'FAIL'}: {run_dir / 'preflight.json'}\n",
        encoding="utf-8",
    )
    if not passed:
        raise ValueError(f"Dataset preflight fallito: {run_dir / 'preflight.json'}")
    try:
        temporal = (
            pd.read_csv(paths["temporal"]) if paths["temporal"].is_file() else None
        )
        market = (
            pd.read_csv(paths["market"])
            if paths["market"].is_file()
            else pd.DataFrame()
        )
        tensors = build_player_tensor_dataset(
            matches, lineups, temporal_features=temporal
        )
        export_neural_lineup_model(
            tensors,
            matches,
            market,
            run_dir / "artifacts",
            epochs=epochs,
            embedding_dim=embedding_dim,
        )
        state.update(status="completed", ended_at=_now(), latest_heartbeat=_now())
    except BaseException as error:
        state.update(
            status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            ended_at=_now(),
            error=repr(error),
        )
        raise
    finally:
        _write_json(run_dir / "run.json", state)
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"at": _now(), "event": "terminal", "status": state["status"]}
                )
                + "\n"
            )
    return run_dir
