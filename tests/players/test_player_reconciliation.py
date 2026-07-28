from datetime import datetime

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import MatchRecord
from football_odds.players.reconciliation import (
    normalize_team_name,
    reconcile_api_football,
    reconcile_fixtures,
)


def _record(provider_id: str) -> MatchRecord:
    return MatchRecord(
        provider_match_id=provider_id,
        date=datetime(2024, 8, 10, 20, 45),
        season="2425",
        league_code="I1",
        home_team="Internazionale",
        away_team="AC Milan",
        home_goals=1,
        away_goals=1,
        result="D",
    )


def _fixture(fixture_id: int = 900) -> dict:
    return {
        "fixture": {"id": fixture_id, "date": "2024-08-10T18:45:00+00:00"},
        "league": {"id": 135},
        "teams": {
            "home": {"id": 44, "name": "Internazionale FC"},
            "away": {"id": 45, "name": "AC Milan"},
        },
    }


def test_reconciliation_maps_unique_normalized_teams_and_is_idempotent(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    match_id = database.upsert_match(
        "Football-Data.co.uk", _record("fd-1"), "Serie A", "Italy"
    )

    first = reconcile_fixtures(
        database, [_fixture()], observed_at="2026-07-25T10:00:00+00:00"
    )
    second = reconcile_fixtures(
        database, [_fixture()], observed_at="2026-07-25T11:00:00+00:00"
    )

    assert (first.fixtures_mapped, first.already_mapped) == (1, 0)
    assert (second.fixtures_mapped, second.already_mapped) == (0, 1)
    assert normalize_team_name("Internazionale F.C.") == "internazionale"
    assert normalize_team_name("AC Milan") == "milan"
    assert normalize_team_name("AS Roma") == "roma"
    assert normalize_team_name("Hellas Verona") == "verona"
    with database.connect() as connection:
        mapped = connection.execute(
            """
            SELECT internal_match_id FROM provider_match_mapping pm
            JOIN providers p USING(provider_id)
            WHERE p.provider_name='API-Football'
            """
        ).fetchone()[0]
        assert mapped == match_id
        assert (
            connection.execute("SELECT COUNT(*) FROM provider_team_mapping").fetchone()[
                0
            ]
            == 2
        )


def test_reconciliation_does_not_guess_unresolved_team(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    database.upsert_match("Football-Data.co.uk", _record("fd-1"), "Serie A", "Italy")
    fixture = _fixture()
    fixture["teams"]["home"]["name"] = "Inter"
    result = reconcile_fixtures(
        database, [fixture], observed_at="2026-07-25T10:00:00+00:00"
    )
    assert result.fixtures_mapped == 0
    assert result.unresolved[0]["reason"] == "team_unresolved"
    with database.connect() as connection:
        assert (
            connection.execute(
                """
            SELECT COUNT(*) FROM provider_match_mapping pm
            JOIN providers p USING(provider_id)
            WHERE p.provider_name='API-Football'
            """
            ).fetchone()[0]
            == 0
        )


def test_api_wrapper_fetches_requested_competition_and_persists_mapping(tmp_path):
    database_path = tmp_path / "research.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match("Football-Data.co.uk", _record("fd-1"), "Serie A", "Italy")

    class Client:
        requests_made = 1

        def get(self, endpoint, **parameters):
            assert endpoint == "fixtures"
            assert parameters == {"league": 135, "season": 2024}
            return [_fixture()]

    result = reconcile_api_football(
        tmp_path,
        leagues=(135,),
        seasons=(2024,),
        database_path=database_path,
        client=Client(),
    )

    assert (result.fixtures_seen, result.fixtures_mapped) == (1, 1)
    assert result.requests_made == 1
