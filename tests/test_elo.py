import pytest

from football_odds.features import EloRatings, EloSettings


def test_elo_reads_before_update_and_is_zero_sum():
    elo = EloRatings(EloSettings(home_advantage=0))
    assert elo.rating("A") == 1500
    assert elo.expected_home("A", "B") == pytest.approx(0.5)
    home, away = elo.update("A", "B", 2, 0)
    assert home > 1500
    assert away < 1500
    assert home + away == pytest.approx(3000)


def test_elo_draw_and_season_regression():
    elo = EloRatings(EloSettings(home_advantage=0, season_regression=0.5))
    elo.update("A", "B", 1, 0)
    before = elo.rating("A")
    elo.update("A", "B", 0, 0)
    assert elo.rating("A") < before
    changed = elo.rating("A")
    elo.regress_to_mean()
    assert abs(elo.rating("A") - 1500) < abs(changed - 1500)
