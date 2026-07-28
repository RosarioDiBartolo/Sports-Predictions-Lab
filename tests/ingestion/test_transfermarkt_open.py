from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import MatchRecord
from football_odds.ingestion.providers.transfermarkt_open import (
    import_transfermarkt_open_data,
)


def test_imports_complete_transfermarkt_starting_elevens(tmp_path: Path) -> None:
    database_path = tmp_path / "football.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match(
        "Football-Data",
        MatchRecord(
            provider_match_id="existing",
            date=datetime(2024, 8, 16),
            season="2425",
            league_code="E0",
            home_team="Man United",
            away_team="Fulham",
            home_goals=1,
            away_goals=0,
            result="H",
        ),
        "Premier League",
        "England",
    )
    games_path = tmp_path / "games.csv"
    pd.DataFrame(
        [
            {
                "game_id": 1,
                "competition_id": "GB1",
                "season": 2024,
                "date": "2024-08-16",
                "home_club_id": 10,
                "away_club_id": 20,
                "home_club_goals": 1,
                "away_club_goals": 0,
                "home_club_name": "Manchester United",
                "away_club_name": "Fulham FC",
                "home_club_formation": "4-2-3-1",
                "away_club_formation": "4-4-2",
            }
        ]
    ).to_csv(games_path, index=False)
    positions = [
        "Goalkeeper",
        *["Centre-Back"] * 4,
        *["Central Midfield"] * 4,
        *["Centre-Forward"] * 2,
    ]
    lineup_rows = []
    for club_id, prefix in ((10, "Home"), (20, "Away")):
        for index, position in enumerate(positions):
            lineup_rows.append(
                {
                    "game_id": 1,
                    "club_id": club_id,
                    "player_id": club_id * 100 + index,
                    "player_name": f"{prefix} {index}",
                    "type": "starting_lineup",
                    "position": position,
                    "number": index + 1,
                }
            )
    lineups_path = tmp_path / "lineups.csv"
    pd.DataFrame(lineup_rows).to_csv(lineups_path, index=False)

    result = import_transfermarkt_open_data(
        tmp_path,
        games_path=games_path,
        lineups_path=lineups_path,
        database_path=database_path,
        seasons=(2024,),
    )

    assert result.matches_imported == 1
    assert result.player_observations == 22
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
        assert (
            connection.execute(
                """
            SELECT COUNT(*) FROM lineup_players
            WHERE lineup_role='starter' AND position IN ('G', 'D', 'M', 'F')
            """
            ).fetchone()[0]
            == 22
        )
