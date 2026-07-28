from datetime import datetime

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import MatchRecord
from football_odds.ingestion.providers import statsbomb_open


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _roster(team_id, team_name, offset):
    return {
        "team_id": team_id,
        "team_name": team_name,
        "lineup": [
            {
                "player_id": offset + number,
                "player_name": f"{team_name} {number}",
                "jersey_number": number,
                "positions": [
                    {
                        "position": "Center Midfield",
                        "start_reason": "Starting XI",
                    }
                ],
            }
            for number in range(1, 12)
        ]
        + [
            {
                "player_id": offset + 12,
                "player_name": f"{team_name} 12",
                "jersey_number": 12,
                "positions": [],
            }
        ],
    }


def test_statsbomb_import_maps_aliases_caches_and_resumes(tmp_path, monkeypatch):
    database_path = tmp_path / "data" / "football_odds.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match(
        "Football-Data.co.uk",
        MatchRecord(
            provider_match_id="canonical",
            date=datetime(2018, 9, 15),
            season="1819",
            league_code="SP1",
            home_team="Ath Bilbao",
            away_team="Ath Madrid",
            home_goals=1,
            away_goals=0,
            result="H",
        ),
        "La Liga",
        "Spain",
    )
    monkeypatch.setattr(statsbomb_open, "TARGETS", {("SP1", "1819"): (11, 4)})
    matches = [
        {
            "match_id": 15978,
            "match_date": "2018-09-15",
            "home_team": {
                "home_team_id": 1,
                "home_team_name": "Athletic Club",
            },
            "away_team": {
                "away_team_id": 2,
                "away_team_name": "Atlético Madrid",
            },
        }
    ]
    lineups = [
        _roster(1, "Athletic Club", 100),
        _roster(2, "Atlético Madrid", 200),
    ]
    calls = []

    def request(url, **kwargs):
        calls.append(url)
        return Response(lineups if "/lineups/" in url else matches)

    result = statsbomb_open.import_statsbomb_open_data(
        tmp_path,
        database_path=database_path,
        request=request,
    )
    assert result.matches_imported == 1
    assert result.unresolved == 0
    assert len(calls) == 2

    resumed = statsbomb_open.import_statsbomb_open_data(
        tmp_path,
        database_path=database_path,
        request=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    assert resumed.matches_imported == 1
    assert resumed.cache_hits == 2
