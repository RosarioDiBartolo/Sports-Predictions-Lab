from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import MatchRecord
from football_odds.ingestion.providers.match_reports_open import (
    import_match_reports_open_data,
)


def test_imports_cc0_report_only_for_missing_fixture(tmp_path: Path) -> None:
    database_path = tmp_path / "football.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match(
        "Football-Data",
        MatchRecord(
            provider_match_id="sp1-1",
            date=datetime(2020, 12, 13),
            season="2021",
            league_code="SP1",
            home_team="Elche",
            away_team="Granada",
            home_goals=0,
            away_goals=1,
            result="A",
        ),
        "La Liga",
        "Spain",
    )
    row: dict[str, object] = {
        "ID": 4777,
        "Country": "ESP",
        "Date": "12/13/2020",
        "Home": "Elche",
        "Away": "Granada",
        "HomeGoals": 0,
        "AwayGoals": 1,
        "formation_home": "4-4-2",
        "formation_away": "4-3-3",
    }
    positions = ["GK", *["DF"] * 4, *["MF"] * 4, *["FW"] * 2]
    for side, prefix in (("home", "Home"), ("away", "Away")):
        for number, position in enumerate(positions, start=1):
            row[f"starting_name_{side}{number}"] = f"{prefix} {number}"
            row[f"starting_position_{side}{number}"] = position
    source = tmp_path / "games.csv"
    pd.DataFrame([row]).to_csv(source, index=False)

    result = import_match_reports_open_data(
        tmp_path,
        source_path=source,
        database_path=database_path,
    )

    assert result.matches_imported == 1
    assert result.player_observations == 22
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM lineup_players WHERE lineup_role='starter'"
            ).fetchone()[0]
            == 22
        )
