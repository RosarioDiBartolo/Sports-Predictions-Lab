import json

import pandas as pd

from football_odds.modeling.candidate_experiment import run_candidate_experiment


def test_failed_preflight_is_immutable_and_does_not_start_training(tmp_path):
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-01-01",
                "season": "2324",
                "league": "I1",
                "home_goals": 1,
                "away_goals": 0,
                "result": "H",
            }
        ]
    ).to_csv(processed / "modeling_features_all.csv", index=False)
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "season": "2324",
                "league": "I1",
                "home_starters": "[]",
                "away_starters": "[]",
            }
        ]
    ).to_csv(processed / "player_training_ready.csv", index=False)
    pd.DataFrame({"match_id": ["m1"]}).to_csv(
        processed / "player_match_temporal_features.csv", index=False
    )

    run_dir = run_candidate_experiment(
        tmp_path,
        epochs=1,
        candidates=("dixon_coles_gated_tabular_residual",),
        ablations=("base",),
    )

    preflight = json.loads((run_dir / "preflight.json").read_text())
    state = json.loads((run_dir / "run.json").read_text())
    assert not preflight["passed"]
    assert state["status"] == "failed"
    assert preflight["models"] == ["dixon_coles_gated_tabular_residual"]
    assert preflight["ablations"] == ["base"]
    assert not (run_dir / "artifacts").exists()
