"""Read-only evaluation of frozen, versioned prediction artifacts."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from .evaluation import (
    metrics_by_season,
    paired_log_loss_bootstrap,
    probability_metrics,
)

PROBABILITIES = ("probability_home", "probability_draw", "probability_away")


def evaluate_frozen(predictions_path: Path, reference_path: Path, output: Path) -> Path:
    predictions = pd.read_csv(predictions_path)
    reference = pd.read_csv(reference_path)
    required = {
        "match_id",
        "season",
        "result",
        "model",
        "model_version",
        "dataset_version",
        "prediction_cutoff",
        *PROBABILITIES,
    }
    for name, frame in (("predictions", predictions), ("reference", reference)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name}: colonne mancanti {sorted(missing)}")
    if set(predictions["match_id"]) != set(reference["match_id"]):
        raise ValueError("Predizioni e riferimento devono coprire gli stessi match.")
    run_dir = output / f"evaluation-{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "candidate": probability_metrics(
            predictions["result"], predictions[list(PROBABILITIES)].to_numpy(float)
        ),
        "reference": probability_metrics(
            reference["result"], reference[list(PROBABILITIES)].to_numpy(float)
        ),
        "bootstrap": paired_log_loss_bootstrap(predictions, reference),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    metrics_by_season(pd.concat([predictions, reference], ignore_index=True)).to_csv(
        run_dir / "metrics_by_season.csv", index=False
    )
    return run_dir
