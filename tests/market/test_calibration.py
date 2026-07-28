import numpy as np
import pandas as pd

from football_odds.market.calibration import (
    calibration_table,
    expected_calibration_error,
)


def test_expected_calibration_error_weighted_average():
    table = pd.DataFrame(
        {
            "observations": [75, 25],
            "calibration_error": [0.10, -0.20],
        }
    )
    assert np.isclose(expected_calibration_error(table), 0.125)


def test_calibration_table_counts_all_rows():
    long_frame = pd.DataFrame(
        {
            "predicted_probability": [0.1, 0.2, 0.8],
            "occurred": [0, 1, 1],
        }
    )
    table = calibration_table(long_frame, 0.1)
    assert table["observations"].sum() == 3
