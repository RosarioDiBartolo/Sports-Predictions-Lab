from test_modeling_dataset import match_frame

import numpy as np
import pandas as pd

from football_odds.core.config import ModelingConfig
from football_odds.modeling.evaluation import probability_metrics
from football_odds.modeling.match_features import (
    build_prematch_features,
    prepare_modeling_matches,
)


def test_probability_metrics_calculates_ranked_probability_score():
    perfect = probability_metrics(pd.Series(["H"]), np.array([[1.0, 0.0, 0.0]]))
    assert perfect["rps"] == 0.0
    manual = probability_metrics(pd.Series(["D"]), np.array([[0.5, 0.5, 0.0]]))
    assert manual["rps"] == 0.125
from football_odds.modeling.reports import (
    export_modeling_report,
    modeling_diagnostics,
)


def test_modeling_diagnostics_and_export(tmp_path):
    matches = prepare_modeling_matches(match_frame())
    config = ModelingConfig(leagues=("I1",), seasons=("2425",), rolling_windows=(2,))
    features = build_prematch_features(matches, config)
    diagnostics = modeling_diagnostics(features)
    assert "elo_patterns" in diagnostics
    assert diagnostics["league_diagnostics"].iloc[0]["matches"] == 3
    outputs = export_modeling_report(features, tmp_path)
    assert outputs["report"].exists()
    assert outputs["elo_chart"].exists()
