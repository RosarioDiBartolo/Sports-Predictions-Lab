from datetime import datetime

from football_odds.database import ResearchDatabase
from football_odds.domain import MatchRecord
from football_odds.enrichment import (
    _wikidata_venue,
    enrich_historical_weather,
    resolve_team_venues,
)


def _database(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    database.upsert_match(
        "Provider",
        MatchRecord(
            provider_match_id="match-1",
            date=datetime(2025, 1, 10, 20, 45),
            season="2425",
            league_code="I1",
            home_team="Home FC",
            away_team="Away FC",
            home_goals=1,
            away_goals=0,
            result="H",
        ),
        "Serie A",
        "Italy",
    )
    return database


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, responses):
        self.responses = iter(responses)

    def get(self, *args, **kwargs):
        return _Response(next(self.responses))


def test_resolve_venues_and_weather(tmp_path, monkeypatch):
    database = _database(tmp_path)
    monkeypatch.setattr(
        "football_odds.enrichment._wikidata_venue", lambda *args, **kwargs: None
    )
    venue_session = _Session(
        [
            [
                {
                    "type": "stadium",
                    "lat": "45.1",
                    "lon": "9.2",
                    "osm_id": 123,
                    "display_name": "Home Stadium, Italy",
                }
            ]
        ]
    )
    assert resolve_team_venues(database, session=venue_session, pause_seconds=0) == (
        1,
        0,
    )
    weather_session = _Session(
        [
            {
                "hourly": {
                    "time": ["2025-01-10T20:00", "2025-01-10T21:00"],
                    "temperature_2m": [8.0, 7.5],
                    "precipitation": [0.0, 0.2],
                    "wind_speed_10m": [5.0, 6.0],
                }
            }
        ]
    )
    assert enrich_historical_weather(database, session=weather_session) == 1
    with database.connect() as connection:
        row = connection.execute("SELECT * FROM weather_observations").fetchone()
    assert row["observed_at"] == "2025-01-10T21:00:00"
    assert row["temperature_c"] == 7.5


def test_unresolved_venue_is_recorded(tmp_path, monkeypatch):
    database = _database(tmp_path)
    monkeypatch.setattr(
        "football_odds.enrichment._wikidata_venue", lambda *args, **kwargs: None
    )
    assert resolve_team_venues(database, session=_Session([[]]), pause_seconds=0) == (
        0,
        1,
    )
    assert len(database.teams_missing_venues()) == 1


def test_wikidata_structured_venue():
    session = _Session(
        [
            {
                "search": [
                    {
                        "id": "Q1",
                        "description": "Italian association football club",
                    }
                ]
            },
            {
                "entities": {
                    "Q1": {
                        "claims": {
                            "P115": [
                                {"mainsnak": {"datavalue": {"value": {"id": "Q2"}}}}
                            ]
                        }
                    }
                }
            },
            {
                "entities": {
                    "Q2": {
                        "labels": {"en": {"value": "Test Stadium"}},
                        "claims": {
                            "P625": [
                                {
                                    "mainsnak": {
                                        "datavalue": {
                                            "value": {
                                                "latitude": 45.0,
                                                "longitude": 9.0,
                                            }
                                        }
                                    }
                                }
                            ]
                        },
                    }
                }
            },
        ]
    )
    venue = _wikidata_venue(session, "Test FC", timeout=10)
    assert venue == {
        "name": "Test Stadium",
        "latitude": 45.0,
        "longitude": 9.0,
        "source_id": "Q2",
        "club_id": "Q1",
    }
