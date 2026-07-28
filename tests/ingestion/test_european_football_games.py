from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import MatchRecord
from football_odds.ingestion.providers.european_football_games import (
    import_european_football_games,
)


def test_imports_complete_starting_elevens_with_derived_roles(
    tmp_path: Path,
) -> None:
    source = tmp_path / "games.csv"
    fields = [
        "season",
        "league",
        "date",
        "home name",
        "away name",
        "home goals",
        "away goals",
        *[f"home player {slot}" for slot in range(11)],
        *[f"away player {slot}" for slot in range(11)],
    ]
    row = {
        "season": "2018/2019",
        "league": "Premier League",
        "date": "10.08.2018",
        "home name": "Home FC",
        "away name": "Away FC",
        "home goals": "2",
        "away goals": "1",
        **{f"home player {slot}": f"Home Player {slot}" for slot in range(11)},
        **{f"away player {slot}": f"Away Player {slot}" for slot in range(11)},
    }
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    database_path = tmp_path / "football.sqlite3"
    result = import_european_football_games(
        tmp_path,
        source_path=source,
        database_path=database_path,
        seasons=("2018/2019",),
    )

    assert result.matches_imported == 1
    assert result.player_observations == 22
    assert result.roles_from_slot == 22
    with ResearchDatabase(database_path).connect() as connection:
        roles = connection.execute(
            """
            SELECT lp.position, lp.formation_grid
            FROM lineup_players lp
            ORDER BY lp.formation_grid
            """
        ).fetchall()
    assert len(roles) == 22
    assert {str(row["position"]) for row in roles} == {"G", "D", "M", "F"}
    assert all(str(row["formation_grid"]).startswith("derived-slot:") for row in roles)


def test_reconciles_existing_fixture_instead_of_creating_duplicate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "games.csv"
    fields = [
        "season",
        "league",
        "date",
        "home name",
        "away name",
        "home goals",
        "away goals",
        *[f"home player {slot}" for slot in range(11)],
        *[f"away player {slot}" for slot in range(11)],
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "season": "2018/2019",
                "league": "Premier League",
                "date": "10.08.2018",
                "home name": "Manchester United",
                "away name": "Leicester City",
                "home goals": "2",
                "away goals": "1",
                **{f"home player {slot}": f"Home Player {slot}" for slot in range(11)},
                **{f"away player {slot}": f"Away Player {slot}" for slot in range(11)},
            }
        )
    database_path = tmp_path / "football.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match(
        "Football-Data",
        MatchRecord(
            provider_match_id="existing",
            date=datetime(2018, 8, 10),
            season="1819",
            league_code="E0",
            home_team="Man United",
            away_team="Leicester",
            home_goals=2,
            away_goals=1,
            result="H",
        ),
        "Premier League",
        "England",
    )

    result = import_european_football_games(
        tmp_path,
        source_path=source,
        database_path=database_path,
        seasons=("2018/2019",),
    )

    assert result.matches_reconciled == 1
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1


def test_quarantines_homonyms_without_merging_player_identities(tmp_path: Path) -> None:
    source = tmp_path / "games.csv"
    fields = [
        "season",
        "league",
        "date",
        "home name",
        "away name",
        "home goals",
        "away goals",
        *[f"home player {slot}" for slot in range(11)],
        *[f"away player {slot}" for slot in range(11)],
    ]
    row = {
        "season": "2018/2019",
        "league": "Premier League",
        "date": "10.08.2018",
        "home name": "Home FC",
        "away name": "Away FC",
        "home goals": "1",
        "away goals": "1",
        **{f"home player {slot}": f"Home Player {slot}" for slot in range(11)},
        **{f"away player {slot}": f"Away Player {slot}" for slot in range(11)},
    }
    row["home player 5"] = "Alex Smith"
    row["away player 6"] = "Alex Smith"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    database_path = tmp_path / "football.sqlite3"
    first = import_european_football_games(
        tmp_path,
        source_path=source,
        database_path=database_path,
        seasons=("2018/2019",),
    )
    quarantine_path = (
        tmp_path / "data/quarantine/european_football_games_player_identities.jsonl"
    )
    first_content = quarantine_path.read_text(encoding="utf-8")
    second = import_european_football_games(
        tmp_path,
        source_path=source,
        database_path=database_path,
        seasons=("2018/2019",),
    )

    assert first.matches_imported == 0
    assert first.ambiguous_player_identities == 2
    assert second.ambiguous_player_identities == 2
    assert quarantine_path.read_text(encoding="utf-8") == first_content
    assert {json.loads(line)["team"] for line in first_content.splitlines()} == {
        "Home FC",
        "Away FC",
    }
    with ResearchDatabase(database_path).connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
        assert (
                connection.execute(
                    "SELECT COUNT(*) FROM provider_player_mapping"
                ).fetchone()[0]
            == 0
        )
