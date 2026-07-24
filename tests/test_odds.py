import numpy as np
import pandas as pd

from football_odds.odds import find_odds_columns, remove_margin


def test_find_odds_columns_prefers_closing_average():
    frame = pd.DataFrame(columns=["AvgCH", "AvgCD", "AvgCA", "B365H", "B365D", "B365A"])
    assert find_odds_columns(frame) == ("AvgCH", "AvgCD", "AvgCA")


def test_remove_margin_returns_probabilities_summing_to_one():
    odds = pd.DataFrame([[2.0, 3.4, 4.0]], columns=["H", "D", "A"])
    result = remove_margin(odds)
    assert np.isclose(result.loc[0, ["p_home", "p_draw", "p_away"]].sum(), 1.0)
    assert np.isclose(result.loc[0, "margin"], (1 / 2 + 1 / 3.4 + 1 / 4) - 1)
