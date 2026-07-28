"""Canonical schema definition."""

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
CREATE TABLE IF NOT EXISTS player_match_lineup_stats (
    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    lineup_role TEXT NOT NULL CHECK(lineup_role IN ('starter', 'bench')),
    position TEXT,
    minute_in REAL,
    minute_out REAL,
    minutes_played REAL NOT NULL CHECK(minutes_played >= 0),
    PRIMARY KEY(match_id, player_id, provider_id)
);
CREATE TABLE IF NOT EXISTS player_match_observations (
    observation_id INTEGER PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    provider_match_id TEXT NOT NULL,
    provider_player_id TEXT NOT NULL,
    lineup_role TEXT CHECK(lineup_role IN ('starter', 'bench')),
    position_original TEXT,
    position_normalized TEXT,
    formation_grid TEXT,
    shirt_number INTEGER,
    bench_available INTEGER CHECK(bench_available IN (0, 1)),
    minutes_played REAL CHECK(minutes_played >= 0),
    minute_in REAL CHECK(minute_in >= 0),
    minute_out REAL CHECK(minute_out >= 0),
    substitution_entry INTEGER CHECK(substitution_entry IN (0, 1)),
    substitution_exit INTEGER CHECK(substitution_exit IN (0, 1)),
    player_statistics_json TEXT,
    acquired_at TEXT NOT NULL,
    quality TEXT NOT NULL,
    source_record_id TEXT,
    UNIQUE(match_id, player_id, provider_id)
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
CREATE INDEX IF NOT EXISTS idx_player_match_stats_time
ON player_match_lineup_stats(match_id, team_id, player_id);
CREATE INDEX IF NOT EXISTS idx_player_match_observations_time
ON player_match_observations(match_id, team_id, player_id);
"""
