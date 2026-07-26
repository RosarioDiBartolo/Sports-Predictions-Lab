import sqlite3
from dataclasses import replace
from datetime import datetime

from football_odds.database import ResearchDatabase
from football_odds.sources import MatchRecord, OddsRecord


def _match(provider_id: str = "provider-1") -> MatchRecord:
    return MatchRecord(
        provider_match_id=provider_id,
        date=datetime(2025, 1, 10, 20, 45),
        season="2425",
        league_code="I1",
        home_team="Home FC",
        away_team="Away FC",
        home_goals=2,
        away_goals=1,
        result="H",
    )


def test_database_creates_provider_independent_match_id(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    first = database.upsert_match("Provider A", _match("a-10"), "Serie A", "Italy")
    second = database.upsert_match("Provider B", _match("b-99"), "Serie A", "Italy")
    assert first == second
    assert first not in {"a-10", "b-99"}

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM provider_match_mapping"
            ).fetchone()[0]
            == 2
        )


def test_database_persists_match_performance_and_migrates_legacy_schema(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    record = replace(
        _match(),
        home_shots=14,
        away_shots=8,
        home_shots_on_target=6,
        away_shots_on_target=3,
        home_corners=7,
        away_corners=4,
        home_yellow_cards=2,
        away_yellow_cards=3,
        home_red_cards=0,
        away_red_cards=1,
    )
    database.upsert_match("Provider", record, "Serie A", "Italy")
    with database.connect() as connection:
        row = connection.execute("SELECT * FROM match_results").fetchone()
    assert row["home_shots"] == 14
    assert row["away_shots_on_target"] == 3
    assert row["away_red_cards"] == 1


def test_match_identity_ignores_provider_kickoff_time_precision(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    first = database.upsert_match("Provider A", _match("a"), "Serie A", "Italy")
    second_match = replace(_match("b"), date=datetime(2025, 1, 10, 21, 0))
    second = database.upsert_match("Provider B", second_match, "Serie A", "Italy")
    assert first == second


def test_odds_from_different_providers_do_not_overwrite_each_other(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    database.upsert_match("Provider A", _match("a"), "Serie A", "Italy")
    database.upsert_match("Provider B", _match("b"), "Serie A", "Italy")
    for provider, provider_match_id, home_odds in (
        ("Provider A", "a", 2.0),
        ("Provider B", "b", 2.1),
    ):
        database.add_odds(
            provider,
            OddsRecord(
                provider_match_id=provider_match_id,
                bookmaker="Book",
                market="1X2",
                odds={"H": home_odds, "D": 3.4, "A": 4.0},
                timestamp=None,
                timing="closing",
            ),
        )
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT provider_id, COUNT(*) FROM odds GROUP BY provider_id"
        ).fetchall()
    assert sorted(row[1] for row in rows) == [3, 3]


def test_database_normalizes_odds_and_keeps_snapshots(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    database.upsert_match("Provider", _match(), "Serie A", "Italy")
    opening = OddsRecord(
        provider_match_id="provider-1",
        bookmaker="Book",
        market="1X2",
        odds={"H": 2.0, "D": 3.4, "A": 4.0},
        timestamp=None,
        timing="opening",
    )
    closing = OddsRecord(
        provider_match_id="provider-1",
        bookmaker="Book",
        market="1X2",
        odds={"H": 1.9, "D": 3.5, "A": 4.2},
        timestamp=datetime(2025, 1, 10, 20, 45),
        timing="closing",
    )
    assert database.add_odds("Provider", opening) == 3
    assert database.add_odds("Provider", closing) == 3

    with database.connect() as connection:
        rows = connection.execute("SELECT * FROM odds").fetchall()
        assert len(rows) == 6
        opening_rows = [row for row in rows if row["opening_or_closing"] == "opening"]
        assert round(sum(row["implied_probability"] for row in opening_rows), 10) == 1


def test_schema_contains_only_pipeline_owned_tables(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    with database.connect() as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert names.difference({"sqlite_sequence"}) == {
        "bookmakers",
        "leagues",
        "match_results",
        "matches",
        "odds",
        "players",
        "provider_player_mapping",
        "provider_match_mapping",
        "provider_team_mapping",
        "providers",
        "fixture_lineups",
        "lineup_players",
        "player_match_lineup_stats",
        "team_memberships",
        "team_venues",
        "teams",
        "weather_observations",
    }


def test_player_schema_enforces_temporal_and_lineup_contracts(tmp_path):
    database = ResearchDatabase(tmp_path / "players.sqlite3")
    database.initialize()
    match_id = database.upsert_match(
        "API-Football", _match("fixture-1"), "Serie A", "Italy"
    )
    with database.connect() as connection:
        provider_id = connection.execute(
            "SELECT provider_id FROM providers WHERE provider_name='API-Football'"
        ).fetchone()[0]
        team_id = connection.execute(
            "SELECT team_id FROM teams WHERE team_name='Home FC'"
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO players VALUES (?, ?, ?, ?)",
            ("player-uuid", "Player One", "2000-01-01", "Italy"),
        )
        connection.execute(
            "INSERT INTO provider_player_mapping VALUES (?, ?, ?)",
            (provider_id, "123", "player-uuid"),
        )
        connection.execute(
            """
            INSERT INTO team_memberships (
                player_id, team_id, provider_id, valid_from, observed_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("player-uuid", team_id, provider_id, "2024-07-01", "2025-01-01"),
        )
        lineup_id = connection.execute(
            """
            INSERT INTO fixture_lineups (
                match_id, team_id, provider_id, formation, lineup_kind,
                observed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                team_id,
                provider_id,
                "4-3-3",
                "confirmed_historical",
                "2026-07-25",
            ),
        ).lastrowid
        connection.execute(
            """
            INSERT INTO lineup_players (
                lineup_id, player_id, lineup_role, position
            ) VALUES (?, ?, ?, ?)
            """,
            (lineup_id, "player-uuid", "starter", "M"),
        )
        row = connection.execute(
            """
            SELECT fl.lineup_kind, lp.lineup_role
            FROM fixture_lineups fl
            JOIN lineup_players lp USING(lineup_id)
            """
        ).fetchone()
    assert tuple(row) == ("confirmed_historical", "starter")


def test_initialize_migrates_legacy_integer_player_identity(tmp_path):
    path = tmp_path / "legacy-players.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE players (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO players (player_id, player_name) VALUES (7, 'Player Seven')"
        )

    ResearchDatabase(path).initialize()

    with sqlite3.connect(path) as connection:
        columns = {
            row[1]: row[2] for row in connection.execute("PRAGMA table_info(players)")
        }
        row = connection.execute(
            "SELECT player_id, player_name FROM players"
        ).fetchone()
    assert columns == {
        "player_id": "TEXT",
        "player_name": "TEXT",
        "date_of_birth": "TEXT",
        "nationality": "TEXT",
    }
    assert row == ("7", "Player Seven")


def test_initialize_migrates_legacy_odds_with_provider_provenance(tmp_path):
    database = ResearchDatabase(tmp_path / "legacy.sqlite3")
    database.initialize()
    match_id = database.upsert_match(
        "Football-Data.co.uk", _match(), "Serie A", "Italy"
    )
    with database.connect() as connection:
        connection.execute("DROP TABLE odds")
        connection.execute(
            """
            CREATE TABLE odds (
                odds_id INTEGER PRIMARY KEY,
                match_id TEXT NOT NULL,
                bookmaker_id INTEGER NOT NULL,
                market TEXT NOT NULL,
                selection TEXT NOT NULL,
                decimal_odds REAL NOT NULL,
                implied_probability_raw REAL NOT NULL,
                implied_probability REAL NOT NULL,
                margin REAL NOT NULL,
                timestamp TEXT,
                opening_or_closing TEXT NOT NULL,
                UNIQUE(
                    match_id, bookmaker_id, market, selection, timestamp,
                    opening_or_closing
                )
            )
            """
        )
        connection.execute(
            "INSERT INTO bookmakers(bookmaker_name) VALUES ('Legacy Book')"
        )
        bookmaker_id = connection.execute(
            "SELECT bookmaker_id FROM bookmakers WHERE bookmaker_name='Legacy Book'"
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO odds VALUES (
                1, ?, ?, '1X2', 'H', 2.0, 0.5, 0.45, 0.1, '', 'opening'
            )
            """,
            (match_id, bookmaker_id),
        )
    database.initialize()
    with database.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(odds)")}
        provider_id = connection.execute("SELECT provider_id FROM odds").fetchone()[0]
    assert "provider_id" in columns
    assert provider_id is not None
