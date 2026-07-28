from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import MatchRecord
from football_odds.ingestion.providers.premier_league_reports import (
    import_premier_league_reports,
)


def test_imports_reported_roles_into_existing_fixture(tmp_path: Path) -> None:
    database_path = tmp_path / "football.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match(
        "Football-Data",
        MatchRecord(
            provider_match_id="e0-2425-1",
            date=datetime(2024, 8, 18),
            season="2425",
            league_code="E0",
            home_team="Brentford",
            away_team="Crystal Palace",
            home_goals=2,
            away_goals=1,
            result="H",
        ),
        "Premier League",
        "England",
    )
    names = {
        "home": [f"Home {index}" for index in range(11)],
        "away": [f"Away {index}" for index in range(11)],
    }
    report = {
        "id": "8",
        "match_report": {
            "date": "2024-08-18",
            "teams": {
                "home": {
                    "name": "Brentford",
                    "score": 2,
                    "team_id": "home",
                },
                "away": {
                    "name": "Crystal Palace",
                    "score": 1,
                    "team_id": "away",
                },
            },
            "lineups": {
                side: {
                    "formation": "4-4-2",
                    "players": [
                        {"name": name, "number": index + 1}
                        for index, name in enumerate(names[side])
                    ],
                }
                for side in ("home", "away")
            },
            "player_stats": {
                side: [
                    {
                        "Player": name,
                        "Pos": (
                            "GK"
                            if index == 0
                            else "DF"
                            if index < 5
                            else "MF"
                            if index < 9
                            else "FW"
                        ),
                    }
                    for index, name in enumerate(names[side])
                ]
                for side in ("home", "away")
            },
        },
    }
    source = tmp_path / "reports.jsonl"
    source.write_text(json.dumps(report) + "\n", encoding="utf-8")

    result = import_premier_league_reports(
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
                """
            SELECT COUNT(*) FROM lineup_players
            WHERE lineup_role='starter' AND position IN ('G', 'D', 'M', 'F')
            """
            ).fetchone()[0]
            == 22
        )
