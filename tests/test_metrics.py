import numpy as np
import pandas as pd

from football_odds.metrics import calculate_metrics, multiclass_brier_score


def test_perfect_brier_score_is_zero():
    actual = np.eye(3)
    assert multiclass_brier_score(actual, actual) == 0.0


def test_calculate_metrics_on_perfect_predictions():
    frame = pd.DataFrame(
        {
            "FTR": ["H", "D", "A"],
            "p_home": [1.0, 0.0, 0.0],
            "p_draw": [0.0, 1.0, 0.0],
            "p_away": [0.0, 0.0, 1.0],
            "margin": [0.05, 0.05, 0.05],
        }
    )
    metrics = calculate_metrics(frame)
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] == 0.0
