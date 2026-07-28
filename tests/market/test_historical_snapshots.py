import pandas as pd

from football_odds.market.historical_snapshots import (
    BOOKMAKERS,
    _cutoff_rows,
    _reconcile,
)


def test_reconciliation_requires_score_team_confidence_and_margin():
    canonical = pd.DataFrame(
        [
            {
                "match_id": "canonical",
                "date": "2016-01-02",
                "home_team": "AC Milan",
                "away_team": "Torino",
                "home_goals": 1,
                "away_goals": 0,
            }
        ]
    )
    source = pd.DataFrame(
        [
            {
                "match_id": 10,
                "day": "2016-01-02",
                "home_team": "AC Mailand",
                "away_team": "FC Turin",
                "home_key": "mailand",
                "away_key": "turin",
                "home_goals": 1,
                "away_goals": 0,
                "match_datetime": pd.Timestamp("2016-01-02T20:00:00Z"),
                "series_file": "odds_series.csv.gz",
            }
        ]
    )

    reconciled, quarantine = _reconcile(canonical, source)

    assert len(reconciled) == 1
    assert quarantine.empty
    assert reconciled.iloc[0]["source_match_id"] == "10"


def test_cutoff_rows_require_complete_hda_and_are_strictly_prematch(tmp_path):
    values = {"match_id": [10]}
    for index in range(1, len(BOOKMAKERS) + 1):
        for selection in ("home", "draw", "away"):
            values[f"{selection}_b{index}_70"] = [2.0]
    values["away_b2_70"] = [float("nan")]
    pd.DataFrame(values).to_csv(
        tmp_path / "odds_series.csv.gz", index=False, compression="gzip"
    )
    reconciled = pd.DataFrame(
        [
            {
                "source_match_id": "10",
                "match_id": "canonical",
                "fixture_kickoff": pd.Timestamp("2016-01-02T20:00:00Z"),
                "series_file": "odds_series.csv.gz",
                "home_similarity": 90.0,
                "away_similarity": 90.0,
                "runner_up_margin": 20.0,
            }
        ]
    )

    rows = _cutoff_rows(tmp_path, reconciled)

    assert rows["match_id"].eq("canonical").all()
    assert rows.groupby("bookmaker")["selection"].nunique().eq(3).all()
    assert BOOKMAKERS[1] not in set(rows["bookmaker"])
    updated = pd.to_datetime(rows["provider_updated_at"], utc=True)
    cutoff = pd.to_datetime(rows["prediction_cutoff"], utc=True)
    assert updated.lt(cutoff).all()
