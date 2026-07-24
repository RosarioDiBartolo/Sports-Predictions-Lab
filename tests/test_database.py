from dataclasses import replace
from datetime import datetime

from football_odds.database import ResearchDatabase
from football_odds.domain import MatchRecord, OddsRecord


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


def test_future_ready_tables_exist(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    with database.connect() as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {
        "players",
        "player_match_stats",
        "lineups",
        "injuries",
        "transfers",
        "player_ratings",
        "team_ratings",
        "elo_history",
        "team_form",
        "weather",
        "referees",
    }.issubset(names)


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
