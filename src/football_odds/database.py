from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .sources import MatchRecord, OddsRecord

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
    result TEXT CHECK(result IN ('H', 'D', 'A') OR result IS NULL),
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    home_corners INTEGER,
    away_corners INTEGER,
    home_yellow_cards INTEGER,
    away_yellow_cards INTEGER,
    home_red_cards INTEGER,
    away_red_cards INTEGER
);
CREATE TABLE IF NOT EXISTS provider_match_mapping (
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    provider_match_id TEXT NOT NULL,
    internal_match_id TEXT NOT NULL REFERENCES matches(match_id),
    PRIMARY KEY(provider_id, provider_match_id)
);
CREATE TABLE IF NOT EXISTS provider_team_mapping (
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    provider_team_id TEXT NOT NULL,
    internal_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    provider_team_name TEXT NOT NULL,
    mapping_method TEXT NOT NULL CHECK(
        mapping_method IN ('normalized_exact', 'manual')
    ),
    observed_at TEXT NOT NULL,
    PRIMARY KEY(provider_id, provider_team_id)
);
CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    player_name TEXT NOT NULL,
    date_of_birth TEXT,
    nationality TEXT
);
CREATE TABLE IF NOT EXISTS provider_player_mapping (
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    provider_player_id TEXT NOT NULL,
    internal_player_id TEXT NOT NULL REFERENCES players(player_id),
    PRIMARY KEY(provider_id, provider_player_id)
);
CREATE TABLE IF NOT EXISTS team_memberships (
    membership_id INTEGER PRIMARY KEY,
    player_id TEXT NOT NULL REFERENCES players(player_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    observed_at TEXT NOT NULL,
    CHECK(valid_to IS NULL OR valid_to >= valid_from),
    UNIQUE(player_id, team_id, provider_id, valid_from)
);
CREATE TABLE IF NOT EXISTS fixture_lineups (
    lineup_id INTEGER PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    formation TEXT,
    coach_provider_id TEXT,
    lineup_kind TEXT NOT NULL CHECK(
        lineup_kind IN (
            'confirmed_historical',
            'confirmed_timestamped',
            'predicted'
        )
    ),
    observed_at TEXT NOT NULL,
    published_at TEXT,
    CHECK(
        lineup_kind != 'confirmed_timestamped'
        OR published_at IS NOT NULL
    ),
    UNIQUE(match_id, team_id, provider_id, lineup_kind)
);
CREATE TABLE IF NOT EXISTS lineup_players (
    lineup_id INTEGER NOT NULL
        REFERENCES fixture_lineups(lineup_id) ON DELETE CASCADE,
    player_id TEXT NOT NULL REFERENCES players(player_id),
    lineup_role TEXT NOT NULL CHECK(lineup_role IN ('starter', 'substitute')),
    position TEXT,
    formation_grid TEXT,
    shirt_number INTEGER,
    PRIMARY KEY(lineup_id, player_id)
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
CREATE INDEX IF NOT EXISTS idx_odds_analysis
ON odds(bookmaker_id, market, opening_or_closing);
CREATE INDEX IF NOT EXISTS idx_matches_league_season
ON matches(league_id, season);
CREATE INDEX IF NOT EXISTS idx_provider_team_internal
ON provider_team_mapping(provider_id, internal_team_id);
CREATE INDEX IF NOT EXISTS idx_memberships_player_time
ON team_memberships(player_id, valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_lineups_match
ON fixture_lineups(match_id, team_id, lineup_kind);
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
        """Create only tables produced by a concrete pipeline stage."""
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate_players_identity(connection)
            self._migrate_odds_provider(connection)
            self._migrate_match_performance(connection)

    @staticmethod
    def _migrate_players_identity(connection: sqlite3.Connection) -> None:
        """Upgrade legacy integer player identities to canonical text UUIDs."""
        columns = {
            str(row["name"]): str(row["type"]).upper()
            for row in connection.execute("PRAGMA table_info(players)").fetchall()
        }
        if columns.get("player_id") == "TEXT":
            for name in ("date_of_birth", "nationality"):
                if name not in columns:
                    connection.execute(f"ALTER TABLE players ADD COLUMN {name} TEXT")
            return

        connection.commit()
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            connection.executescript(
                """
                CREATE TABLE players_new (
                    player_id TEXT PRIMARY KEY,
                    player_name TEXT NOT NULL,
                    date_of_birth TEXT,
                    nationality TEXT
                );
                INSERT INTO players_new (player_id, player_name)
                SELECT CAST(player_id AS TEXT), player_name FROM players;
                DROP TABLE players;
                ALTER TABLE players_new RENAME TO players;
                """
            )
        finally:
            connection.execute("PRAGMA foreign_keys = ON")

    @staticmethod
    def _migrate_match_performance(connection: sqlite3.Connection) -> None:
        """Add nullable Football-Data performance facts to legacy databases."""
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(match_results)").fetchall()
        }
        performance_columns = (
            "home_shots",
            "away_shots",
            "home_shots_on_target",
            "away_shots_on_target",
            "home_corners",
            "away_corners",
            "home_yellow_cards",
            "away_yellow_cards",
            "home_red_cards",
            "away_red_cards",
        )
        for column in performance_columns:
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE match_results ADD COLUMN {column} INTEGER"
                )

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
                INSERT INTO match_results (
                    match_id, home_goals, away_goals, result,
                    home_shots, away_shots,
                    home_shots_on_target, away_shots_on_target,
                    home_corners, away_corners,
                    home_yellow_cards, away_yellow_cards,
                    home_red_cards, away_red_cards
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id) DO UPDATE SET
                    home_goals=excluded.home_goals,
                    away_goals=excluded.away_goals,
                    result=excluded.result,
                    home_shots=excluded.home_shots,
                    away_shots=excluded.away_shots,
                    home_shots_on_target=excluded.home_shots_on_target,
                    away_shots_on_target=excluded.away_shots_on_target,
                    home_corners=excluded.home_corners,
                    away_corners=excluded.away_corners,
                    home_yellow_cards=excluded.home_yellow_cards,
                    away_yellow_cards=excluded.away_yellow_cards,
                    home_red_cards=excluded.home_red_cards,
                    away_red_cards=excluded.away_red_cards
                """,
                (
                    match_id,
                    record.home_goals,
                    record.away_goals,
                    record.result,
                    record.home_shots,
                    record.away_shots,
                    record.home_shots_on_target,
                    record.away_shots_on_target,
                    record.home_corners,
                    record.away_corners,
                    record.home_yellow_cards,
                    record.away_yellow_cards,
                    record.home_red_cards,
                    record.away_red_cards,
                ),
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
