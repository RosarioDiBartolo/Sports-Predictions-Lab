"""Observable final fit for the non-operational gated neural candidate."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .candidate_experiment import (
    _append_event,
    _heartbeat,
    _now,
    _write_json,
    create_preflight,
)
from .gated_neural import (
    GATED_NEURAL_LINEUP_MODEL_NAME,
    NeuralFeatureAblation,
    fit_final_gated_neural_lineup_model,
)
from .neural import build_player_tensor_dataset, cross_fitted_dixon_coles_rates
from .reliability import (
    HISTORY_OBSERVATION_THRESHOLD,
    RELIABLE_QUALITY_THRESHOLD,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase(run_dir: Path, state: dict[str, Any], name: str) -> None:
    state.update(phase=name, latest_heartbeat=_now())
    _write_json(run_dir / "run.json", state)
    _append_event(run_dir, {"event": "phase", "phase": name})


def fit_final_gated_candidate(
    project: Path,
    *,
    code_version: str,
    epochs: int = 80,
    embedding_dim: int = 32,
    seed: int = 42,
    device: str = "cpu",
    ablation: str = "combined",
    run_root: Path | None = None,
) -> Path:
    """Preflight, fit on all eligible history and save a versioned checkpoint."""
    selected = NeuralFeatureAblation(ablation)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    root = run_root or project / "reports/modeling/gated_final/runs"
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
        device=device,
        candidates=(GATED_NEURAL_LINEUP_MODEL_NAME,),
        ablations=(selected.value,),
    )
    state: dict[str, Any] = {
        "run_id": run_id,
        "status": "running" if preflight["passed"] else "failed",
        "phase": "preflight",
        "started_at": _now(),
        "latest_heartbeat": _now(),
        "model": GATED_NEURAL_LINEUP_MODEL_NAME,
        "ablation": selected.value,
        "device": device,
        "code_version": code_version,
    }
    _write_json(run_dir / "run.json", state)
    _append_event(run_dir, {"event": "preflight", "passed": preflight["passed"]})
    if not preflight["passed"]:
        state.update(ended_at=_now())
        _write_json(run_dir / "run.json", state)
        _append_event(run_dir, {"event": "terminal", "status": "failed"})
        raise ValueError(f"Final-fit preflight failed: {run_dir / 'preflight.json'}")

    started = time.monotonic()
    try:
        with _heartbeat(run_dir, state):
            _phase(run_dir, state, "loading_inputs")
            matches = pd.read_csv(project / "data/processed/modeling_features_all.csv")
            lineups = pd.read_csv(project / "data/processed/player_training_ready.csv")
            temporal = pd.read_csv(
                project / "data/processed/player_match_temporal_features.csv"
            )
            _phase(run_dir, state, "building_tensors")
            tensors = build_player_tensor_dataset(
                matches, lineups, temporal_features=temporal
            )
            _phase(run_dir, state, "cross_fitting_rates")
            rates = cross_fitted_dixon_coles_rates(matches, tensors.matches)
            _phase(run_dir, state, "final_fit")
            predictor, training_matches = fit_final_gated_neural_lineup_model(
                tensors,
                matches,
                embedding_dim=embedding_dim,
                epochs=epochs,
                rates=rates,
                ablation=selected,
                seed=seed,
                device=device,
            )
            artifact_dir = run_dir / "artifacts/final_model"
            artifact_dir.mkdir(parents=True)
            model_path = artifact_dir / "model.pt"
            predictor.model.to("cpu")
            torch.save(
                {
                    "state_dict": predictor.model.state_dict(),
                    "feature_mean": predictor.feature_mean,
                    "feature_scale": predictor.feature_scale,
                    "feature_names": tensors.feature_names,
                    "embedding_dim": embedding_dim,
                    "maximum_log_correction": predictor.maximum_log_correction,
                    "model_name": GATED_NEURAL_LINEUP_MODEL_NAME,
                    "feature_ablation": selected.value,
                },
                model_path,
            )
            metadata = {
                "model_name": GATED_NEURAL_LINEUP_MODEL_NAME,
                "operational": False,
                "feature_ablation": selected.value,
                "training_matches": training_matches,
                "epochs": epochs,
                "embedding_dim": embedding_dim,
                "seed": seed,
                "device": device,
                "code_version": code_version,
                "dataset_version": preflight["dataset_version"],
                "preflight": str(run_dir / "preflight.json"),
                "model_sha256": _sha256(model_path),
                "reliability_gate": {
                    "history_observation_threshold": HISTORY_OBSERVATION_THRESHOLD,
                    "reliable_quality_threshold": RELIABLE_QUALITY_THRESHOLD,
                },
            }
            (artifact_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
        state.update(
            status="completed",
            phase="completed",
            ended_at=_now(),
            elapsed_seconds=time.monotonic() - started,
            artifact=str(model_path),
        )
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
