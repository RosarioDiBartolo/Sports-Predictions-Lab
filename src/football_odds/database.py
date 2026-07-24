from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .domain import MatchRecord, OddsRecord

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS leagues (
    league_id INTEGER PRIMARY KEY,
    league_code TEXT NOT NULL UNIQUE,
    league_name TEXT NOT NULL,
    country TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    team_name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS bookmakers (
    bookmaker_id INTEGER PRIMARY KEY,
    bookmaker_name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS providers (
    provider_id INTEGER PRIMARY KEY,
    provider_name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    season TEXT NOT NULL,
    league_id INTEGER NOT NULL REFERENCES leagues(league_id),
    home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    UNIQUE(date, league_id, home_team_id, away_team_id)
);
CREATE TABLE IF NOT EXISTS match_results (
    match_id TEXT PRIMARY KEY REFERENCES matches(match_id) ON DELETE CASCADE,
    home_goals INTEGER,
    away_goals INTEGER,
    result TEXT CHECK(result IN ('H', 'D', 'A') OR result IS NULL)
);
CREATE TABLE IF NOT EXISTS provider_match_mapping (
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    provider_match_id TEXT NOT NULL,
    internal_match_id TEXT NOT NULL REFERENCES matches(match_id),
    PRIMARY KEY(provider_id, provider_match_id)
);
CREATE TABLE IF NOT EXISTS odds (
    odds_id INTEGER PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    bookmaker_id INTEGER NOT NULL REFERENCES bookmakers(bookmaker_id),
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    decimal_odds REAL NOT NULL CHECK(decimal_odds > 1),
    implied_probability_raw REAL NOT NULL,
    implied_probability REAL NOT NULL,
    margin REAL NOT NULL,
    timestamp TEXT,
    opening_or_closing TEXT NOT NULL
        CHECK(opening_or_closing IN ('opening', 'closing', 'snapshot')),
    UNIQUE(
        match_id, bookmaker_id, provider_id, market, selection, timestamp,
        opening_or_closing
    )
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY, player_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS player_match_stats (
    player_id INTEGER REFERENCES players(player_id),
    match_id TEXT REFERENCES matches(match_id),
    minutes REAL, goals REAL, assists REAL,
    PRIMARY KEY(player_id, match_id)
);
CREATE TABLE IF NOT EXISTS lineups (
    match_id TEXT REFERENCES matches(match_id),
    player_id INTEGER REFERENCES players(player_id),
    team_id INTEGER REFERENCES teams(team_id),
    starter INTEGER, position TEXT,
    PRIMARY KEY(match_id, player_id)
);
CREATE TABLE IF NOT EXISTS injuries (
    injury_id INTEGER PRIMARY KEY, player_id INTEGER REFERENCES players(player_id),
    start_date TEXT, end_date TEXT, description TEXT
);
CREATE TABLE IF NOT EXISTS transfers (
    transfer_id INTEGER PRIMARY KEY, player_id INTEGER REFERENCES players(player_id),
    from_team_id INTEGER REFERENCES teams(team_id),
    to_team_id INTEGER REFERENCES teams(team_id), transfer_date TEXT, fee REAL
);
CREATE TABLE IF NOT EXISTS player_ratings (
    player_id INTEGER REFERENCES players(player_id), rating_date TEXT,
    rating REAL, source TEXT, PRIMARY KEY(player_id, rating_date, source)
);
CREATE TABLE IF NOT EXISTS team_ratings (
    team_id INTEGER REFERENCES teams(team_id), rating_date TEXT,
    rating REAL, source TEXT, PRIMARY KEY(team_id, rating_date, source)
);
CREATE TABLE IF NOT EXISTS elo_history (
    team_id INTEGER REFERENCES teams(team_id), rating_date TEXT,
    elo REAL, PRIMARY KEY(team_id, rating_date)
);
CREATE TABLE IF NOT EXISTS team_form (
    team_id INTEGER REFERENCES teams(team_id), as_of_date TEXT,
    window INTEGER, points REAL, PRIMARY KEY(team_id, as_of_date, window)
);
CREATE TABLE IF NOT EXISTS league_table_history (
    league_id INTEGER REFERENCES leagues(league_id),
    team_id INTEGER REFERENCES teams(team_id),
    as_of_date TEXT, position INTEGER, points INTEGER,
    PRIMARY KEY(league_id, team_id, as_of_date)
);
CREATE TABLE IF NOT EXISTS schedule (
    match_id TEXT PRIMARY KEY REFERENCES matches(match_id), kickoff TEXT, venue TEXT
);
CREATE TABLE IF NOT EXISTS travel_distance (
    match_id TEXT PRIMARY KEY REFERENCES matches(match_id), distance_km REAL
);
CREATE TABLE IF NOT EXISTS weather (
    match_id TEXT PRIMARY KEY REFERENCES matches(match_id),
    temperature_c REAL, precipitation_mm REAL, wind_kph REAL
);
CREATE TABLE IF NOT EXISTS team_venues (
    team_id INTEGER PRIMARY KEY REFERENCES teams(team_id),
    venue_name TEXT,
    latitude REAL,
    longitude REAL,
    source TEXT NOT NULL,
    source_id TEXT,
    display_name TEXT,
    resolved INTEGER NOT NULL DEFAULT 1,
    query TEXT
);
CREATE TABLE IF NOT EXISTS weather_observations (
    match_id TEXT PRIMARY KEY REFERENCES matches(match_id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL,
    temperature_c REAL NOT NULL,
    precipitation_mm REAL NOT NULL,
    wind_kph REAL NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS referees (
    referee_id INTEGER PRIMARY KEY, referee_name TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_odds_analysis
ON odds(bookmaker_id, market, opening_or_closing);
CREATE INDEX IF NOT EXISTS idx_matches_league_season
ON matches(league_id, season);
"""


class ResearchDatabase:
    """SQLite repository for the provider-neutral master database."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._active_connection: sqlite3.Connection | None = None

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a transaction with named-column access."""
        if self._active_connection is not None:
            yield self._active_connection
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def batch(self) -> Iterator[sqlite3.Connection]:
        """Reuse one atomic transaction for a high-volume ingestion block."""
        if self._active_connection is not None:
            yield self._active_connection
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        self._active_connection = connection
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self._active_connection = None
            connection.close()

    def initialize(self) -> None:
        """Create all current and future-ready tables idempotently."""
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate_odds_provider(connection)

    @staticmethod
    def _migrate_odds_provider(connection: sqlite3.Connection) -> None:
        """Upgrade legacy odds rows so provenance participates in identity."""
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(odds)").fetchall()
        }
        if "provider_id" in columns:
            return
        connection.execute(
            "INSERT OR IGNORE INTO providers (provider_name) VALUES (?)",
            ("Football-Data.co.uk",),
        )
        provider_id = int(
            connection.execute(
                "SELECT provider_id FROM providers WHERE provider_name=?",
                ("Football-Data.co.uk",),
            ).fetchone()[0]
        )
        connection.execute("ALTER TABLE odds RENAME TO odds_legacy")
        connection.execute(
            """
            CREATE TABLE odds (
                odds_id INTEGER PRIMARY KEY,
                match_id TEXT NOT NULL
                    REFERENCES matches(match_id) ON DELETE CASCADE,
                bookmaker_id INTEGER NOT NULL
                    REFERENCES bookmakers(bookmaker_id),
                provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
                market TEXT NOT NULL,
                selection TEXT NOT NULL,
                decimal_odds REAL NOT NULL CHECK(decimal_odds > 1),
                implied_probability_raw REAL NOT NULL,
                implied_probability REAL NOT NULL,
                margin REAL NOT NULL,
                timestamp TEXT,
                opening_or_closing TEXT NOT NULL
                    CHECK(opening_or_closing IN ('opening', 'closing', 'snapshot')),
                UNIQUE(
                    match_id, bookmaker_id, provider_id, market, selection,
                    timestamp, opening_or_closing
                )
            )
            """
        )
        connection.execute(
            """
            INSERT INTO odds (
                odds_id, match_id, bookmaker_id, provider_id, market, selection,
                decimal_odds, implied_probability_raw, implied_probability,
                margin, timestamp, opening_or_closing
            )
            SELECT
                odds_id, match_id, bookmaker_id, ?, market, selection,
                decimal_odds, implied_probability_raw, implied_probability,
                margin, timestamp, opening_or_closing
            FROM odds_legacy
            """,
            (provider_id,),
        )
        connection.execute("DROP TABLE odds_legacy")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_odds_match
            ON odds(match_id, market, opening_or_closing, timestamp)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_odds_analysis
            ON odds(bookmaker_id, market, opening_or_closing)
            """
        )

    @staticmethod
    def _lookup_id(
        connection: sqlite3.Connection,
        table: str,
        id_column: str,
        name_column: str,
        value: str,
    ) -> int:
        connection.execute(
            f"INSERT OR IGNORE INTO {table} ({name_column}) VALUES (?)",
            (value,),
        )
        row = connection.execute(
            f"SELECT {id_column} FROM {table} WHERE {name_column} = ?",
            (value,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Impossibile creare {table}: {value}")
        return int(row[0])

    def upsert_league(self, code: str, name: str, country: str) -> int:
        """Create or update a league and return its internal ID."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO leagues (league_code, league_name, country)
                VALUES (?, ?, ?)
                ON CONFLICT(league_code) DO UPDATE SET
                    league_name = excluded.league_name,
                    country = excluded.country
                """,
                (code, name, country),
            )
            return int(
                connection.execute(
                    "SELECT league_id FROM leagues WHERE league_code = ?", (code,)
                ).fetchone()[0]
            )

    def upsert_match(
        self,
        provider_name: str,
        record: MatchRecord,
        league_name: str,
        country: str,
    ) -> str:
        """Store a match while keeping its UUID independent of provider IDs."""
        with self.connect() as connection:
            provider_id = self._lookup_id(
                connection, "providers", "provider_id", "provider_name", provider_name
            )
            existing = connection.execute(
                """
                SELECT internal_match_id FROM provider_match_mapping
                WHERE provider_id = ? AND provider_match_id = ?
                """,
                (provider_id, record.provider_match_id),
            ).fetchone()
            if existing:
                match_id = str(existing[0])
                connection.execute(
                    "UPDATE matches SET date=?, season=? WHERE match_id=?",
                    (record.date.isoformat(), record.season, match_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO leagues (league_code, league_name, country)
                    VALUES (?, ?, ?)
                    ON CONFLICT(league_code) DO UPDATE SET
                        league_name=excluded.league_name, country=excluded.country
                    """,
                    (record.league_code, league_name, country),
                )
                league_id = int(
                    connection.execute(
                        "SELECT league_id FROM leagues WHERE league_code=?",
                        (record.league_code,),
                    ).fetchone()[0]
                )
                home_id = self._lookup_id(
                    connection, "teams", "team_id", "team_name", record.home_team
                )
                away_id = self._lookup_id(
                    connection, "teams", "team_id", "team_name", record.away_team
                )
                natural = connection.execute(
                    """
                    SELECT match_id FROM matches
                    WHERE date(date)=date(?) AND league_id=?
                      AND home_team_id=? AND away_team_id=?
                    """,
                    (record.date.isoformat(), league_id, home_id, away_id),
                ).fetchone()
                match_id = str(natural[0]) if natural else str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT OR IGNORE INTO matches
                    (match_id, date, season, league_id, home_team_id, away_team_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_id,
                        record.date.isoformat(),
                        record.season,
                        league_id,
                        home_id,
                        away_id,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO provider_match_mapping
                    (provider_id, provider_match_id, internal_match_id)
                    VALUES (?, ?, ?)
                    """,
                    (provider_id, record.provider_match_id, match_id),
                )
            connection.execute(
                """
                INSERT INTO match_results (match_id, home_goals, away_goals, result)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(match_id) DO UPDATE SET
                    home_goals=excluded.home_goals,
                    away_goals=excluded.away_goals,
                    result=excluded.result
                """,
                (match_id, record.home_goals, record.away_goals, record.result),
            )
            return match_id

    def add_odds(self, provider_name: str, record: OddsRecord) -> int:
        """Normalize and store all selections in one odds snapshot."""
        if not record.odds or any(value <= 1 for value in record.odds.values()):
            raise ValueError("Le quote decimali devono essere maggiori di 1.")
        with self.connect() as connection:
            provider_id = self._lookup_id(
                connection, "providers", "provider_id", "provider_name", provider_name
            )
            mapping = connection.execute(
                """
                SELECT internal_match_id FROM provider_match_mapping
                WHERE provider_id=? AND provider_match_id=?
                """,
                (provider_id, record.provider_match_id),
            ).fetchone()
            if mapping is None:
                raise KeyError(
                    f"Partita provider non mappata: {record.provider_match_id}"
                )
            bookmaker_id = self._lookup_id(
                connection,
                "bookmakers",
                "bookmaker_id",
                "bookmaker_name",
                record.bookmaker,
            )
            raw = {selection: 1.0 / value for selection, value in record.odds.items()}
            overround = sum(raw.values())
            margin = overround - 1.0
            timestamp = record.timestamp.isoformat() if record.timestamp else ""
            if record.timing in {"opening", "closing"}:
                connection.execute(
                    """
                    DELETE FROM odds
                    WHERE match_id=? AND bookmaker_id=? AND provider_id=?
                      AND market=?
                      AND opening_or_closing=?
                    """,
                    (
                        str(mapping[0]),
                        bookmaker_id,
                        provider_id,
                        record.market,
                        record.timing,
                    ),
                )
            inserted = 0
            for selection, decimal_odds in record.odds.items():
                cursor = connection.execute(
                    """
                    INSERT OR REPLACE INTO odds (
                        match_id, bookmaker_id, provider_id, market, selection,
                        decimal_odds, implied_probability_raw,
                        implied_probability, margin, timestamp,
                        opening_or_closing
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(mapping[0]),
                        bookmaker_id,
                        provider_id,
                        record.market,
                        selection,
                        decimal_odds,
                        raw[selection],
                        raw[selection] / overround,
                        margin,
                        timestamp,
                        record.timing,
                    ),
                )
                inserted += cursor.rowcount
            return inserted

    def teams_missing_venues(self) -> list[sqlite3.Row]:
        """Return home teams not yet resolved, including their league country."""
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT DISTINCT t.team_id, t.team_name, l.country
                FROM matches m
                JOIN teams t ON t.team_id = m.home_team_id
                JOIN leagues l ON l.league_id = m.league_id
                LEFT JOIN team_venues v ON v.team_id = t.team_id
                WHERE v.team_id IS NULL OR v.resolved = 0
                ORDER BY l.country, t.team_name
                """
            ).fetchall()

    def upsert_team_venue(
        self,
        *,
        team_id: int,
        venue_name: str,
        latitude: float,
        longitude: float,
        source: str,
        source_id: str = "",
        display_name: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO team_venues (
                    team_id, venue_name, latitude, longitude, source,
                    source_id, display_name, resolved, query
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, NULL)
                ON CONFLICT(team_id) DO UPDATE SET
                    venue_name=excluded.venue_name,
                    latitude=excluded.latitude,
                    longitude=excluded.longitude,
                    source=excluded.source,
                    source_id=excluded.source_id,
                    display_name=excluded.display_name,
                    resolved=1,
                    query=NULL
                """,
                (
                    team_id,
                    venue_name,
                    latitude,
                    longitude,
                    source,
                    source_id,
                    display_name,
                ),
            )

    def record_unresolved_venue(self, team_id: int, query: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO team_venues (team_id, source, resolved, query)
                VALUES (?, 'OpenStreetMap Nominatim', 0, ?)
                ON CONFLICT(team_id) DO UPDATE SET resolved=0, query=excluded.query
                """,
                (team_id, query),
            )

    def matches_missing_weather(self, *, limit: int | None = None) -> list[sqlite3.Row]:
        query = """
            SELECT m.match_id, m.date, v.latitude, v.longitude
            FROM matches m
            JOIN team_venues v ON v.team_id = m.home_team_id AND v.resolved = 1
            LEFT JOIN weather_observations w ON w.match_id = m.match_id
            WHERE w.match_id IS NULL
            ORDER BY m.date, m.match_id
        """
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)
        with self.connect() as connection:
            return connection.execute(query, parameters).fetchall()

    def upsert_weather(
        self,
        *,
        match_id: str,
        observed_at: str,
        temperature_c: float,
        precipitation_mm: float,
        wind_kph: float,
        source: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO weather_observations (
                    match_id, observed_at, temperature_c,
                    precipitation_mm, wind_kph, source
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id) DO UPDATE SET
                    observed_at=excluded.observed_at,
                    temperature_c=excluded.temperature_c,
                    precipitation_mm=excluded.precipitation_mm,
                    wind_kph=excluded.wind_kph,
                    source=excluded.source,
                    fetched_at=CURRENT_TIMESTAMP
                """,
                (
                    match_id,
                    observed_at,
                    temperature_c,
                    precipitation_mm,
                    wind_kph,
                    source,
                ),
            )
