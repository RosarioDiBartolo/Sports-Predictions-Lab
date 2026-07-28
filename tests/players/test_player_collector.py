from datetime import datetime

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import MatchRecord
from football_odds.players.collector import collect_api_football_player_data


def _match(provider_id, date):
    return MatchRecord(
        provider_match_id=provider_id,
        date=datetime.fromisoformat(date),
        season="2425",
        league_code="E0",
        home_team="Home FC",
        away_team="Away FC",
        home_goals=1,
        away_goals=0,
        result="H",
    )


def _lineup(team, offset):
    return {
        "team": {"id": offset, "name": team},
        "startXI": [
            {
                "player": {
                    "id": offset * 100 + number,
                    "name": f"{team} {number}",
                    "pos": "M",
                }
            }
            for number in range(1, 12)
        ],
        "substitutes": [],
    }


class Client:
    requests_made = 0

    def get(self, endpoint, **parameters):
        self.requests_made += 1
        if endpoint == "fixtures":
            assert parameters == {"league": 39, "season": 2024}
            return [
                {
                    "fixture": {"id": 900, "date": "2025-01-10T20:45:00+00:00"},
                    "league": {"id": 39},
                    "teams": {
                        "home": {"id": 10, "name": "Home FC"},
                        "away": {"id": 20, "name": "Away FC"},
                    },
                }
            ]
        assert endpoint == "fixtures/lineups"
        assert parameters == {"fixture": "900"}
        return [_lineup("Home FC", 10), _lineup("Away FC", 20)]


def test_collector_maps_then_imports_within_budget_and_resumes(tmp_path):
    database_path = tmp_path / "data" / "football_odds.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match(
        "Football-Data.co.uk",
        _match("canonical", "2025-01-10T20:45:00"),
        "Premier League",
        "England",
    )

    result = collect_api_football_player_data(
        tmp_path,
        leagues=("E0",),
        seasons=("2425",),
        request_budget=2,
        database_path=database_path,
        client=Client(),
    )

    assert result.requests_made == 2
    assert result.fixtures_mapped == 1
    assert result.lineups_imported == 1
    assert result.manifest.exists()

    resumed = collect_api_football_player_data(
        tmp_path,
        leagues=("E0",),
        seasons=("2425",),
        request_budget=2,
        database_path=database_path,
        client=Client(),
    )
    assert resumed.fixture_batches == 0
    assert resumed.lineup_attempts == 0
