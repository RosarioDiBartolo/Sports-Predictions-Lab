import pandas as pd

from football_odds.database import ResearchDatabase
from football_odds.sources import FootballDataProvider, IngestionPipeline


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["10/01/2025"],
            "Season": ["2425"],
            "League": ["I1"],
            "HomeTeam": ["Home FC"],
            "AwayTeam": ["Away FC"],
            "FTHG": [2],
            "FTAG": [1],
            "FTR": ["H"],
            "HS": [14],
            "AS": [8],
            "HST": [6],
            "AST": [3],
            "HC": [7],
            "AC": [4],
            "HY": [2],
            "AY": [3],
            "HR": [0],
            "AR": [1],
            "B365H": [2.0],
            "B365D": [3.4],
            "B365A": [4.0],
            "B365CH": [1.9],
            "B365CD": [3.5],
            "B365CA": [4.2],
        }
    )


def test_football_data_provider_extracts_match_and_two_snapshots():
    provider = FootballDataProvider(sample_frame())
    assert len(provider.matches()) == 1
    snapshots = provider.odds()
    assert len(snapshots) == 2
    assert {snapshot.timing for snapshot in snapshots} == {"opening", "closing"}
    assert all(snapshot.bookmaker == "Bet365" for snapshot in snapshots)
    match = provider.matches()[0]
    assert match.home_shots == 14
    assert match.away_shots_on_target == 3
    assert match.away_red_cards == 1


def test_provider_preserves_kickoff_time():
    frame = sample_frame()
    frame["Time"] = "20:45"
    provider = FootballDataProvider(frame)
    assert provider.matches()[0].date.hour == 20
    closing = [row for row in provider.odds() if row.timing == "closing"]
    assert closing[0].timestamp is None


def test_ingestion_pipeline_uses_provider_contract(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    summary = IngestionPipeline(database).run(FootballDataProvider(sample_frame()))
    assert summary.matches == 1
    assert summary.odds_selections == 6
