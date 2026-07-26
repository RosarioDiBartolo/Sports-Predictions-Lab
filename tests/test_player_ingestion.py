from datetime import datetime

import pytest

from football_odds.database import ResearchDatabase
from football_odds.player_coverage import ApiFootballClient
from football_odds.player_ingestion import (
    import_api_football_lineups,
    ingest_fixture_lineups,
)
from football_odds.sources import MatchRecord


def _match(provider_fixture_id: str, home: str = "Home FC") -> MatchRecord:
    return MatchRecord(
        provider_match_id=provider_fixture_id,
        date=datetime(2024, 8, 10, 20, 45),
        season="2425",
        league_code="I1",
        home_team=home,
        away_team="Away FC",
        home_goals=1,
        away_goals=0,
        result="H",
    )


def _lineup(team: str, offset: int = 0) -> dict:
    def player(number: int) -> dict:
        return {
            "player": {
                "id": offset + number,
                "name": f"Player {offset + number}",
                "number": number,
                "pos": "M",
                "grid": f"2:{number}",
            }
        }

    return {
        "team": {"id": 10 + offset, "name": team},
        "formation": "4-3-3",
        "coach": {"id": 99 + offset},
        "startXI": [player(number) for number in range(1, 12)],
        "substitutes": [player(12), player(13)],
    }


def test_ingestion_is_idempotent_and_keeps_starters_and_bench(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    database.upsert_match("API-Football", _match("900"), "Serie A", "Italy")
    payload = [_lineup("Home FC"), _lineup("Away FC", 100)]

    for observed_at in (
        "2026-07-25T10:00:00+00:00",
        "2026-07-25T11:00:00+00:00",
    ):
        assert ingest_fixture_lineups(
            database,
            provider_fixture_id="900",
            lineups=payload,
            observed_at=observed_at,
        ) == (2, 26)

    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM fixture_lineups"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM lineup_players"
        ).fetchone()[0] == 26
        roles = dict(
            connection.execute(
                "SELECT lineup_role, COUNT(*) FROM lineup_players GROUP BY lineup_role"
            ).fetchall()
        )
        assert roles == {"starter": 22, "substitute": 4}
        assert connection.execute(
            "SELECT COUNT(*) FROM provider_player_mapping"
        ).fetchone()[0] == 26


def test_same_player_can_belong_to_different_teams(tmp_path):
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    database.upsert_match("API-Football", _match("900"), "Serie A", "Italy")
    database.upsert_match(
        "API-Football", _match("901", home="New Home FC"), "Serie A", "Italy"
    )
    ingest_fixture_lineups(
        database,
        provider_fixture_id="900",
        lineups=[_lineup("Home FC"), _lineup("Away FC", 100)],
        observed_at="2024-08-11T00:00:00+00:00",
    )
    ingest_fixture_lineups(
        database,
        provider_fixture_id="901",
        lineups=[_lineup("New Home FC"), _lineup("Away FC", 200)],
        observed_at="2025-01-11T00:00:00+00:00",
    )
    with database.connect() as connection:
        memberships = connection.execute(
            """
            SELECT COUNT(DISTINCT team_id)
            FROM team_memberships tm
            JOIN provider_player_mapping ppm
              ON ppm.internal_player_id=tm.player_id
            WHERE ppm.provider_player_id='1'
            """
        ).fetchone()[0]
    assert memberships == 2


def test_fetch_wrapper_and_incomplete_payload_is_atomic(tmp_path):
    database_path = tmp_path / "data" / "football_odds.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match("API-Football", _match("900"), "Serie A", "Italy")
    payload = [_lineup("Home FC"), _lineup("Away FC", 100)]

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"errors": [], "response": payload}

    client = ApiFootballClient(
        "secret", request=lambda *args, **kwargs: Response(), minimum_interval=0
    )
    result = import_api_football_lineups(
        tmp_path,
        fixture_ids=("900",),
        database_path=database_path,
        client=client,
        observed_at="2026-07-25T10:00:00+00:00",
    )
    assert (result.fixtures, result.lineups, result.players) == (1, 2, 26)

    incomplete = [_lineup("Home FC"), _lineup("Away FC", 100)]
    incomplete[0]["startXI"].pop()
    with pytest.raises(ValueError, match="incompleta"):
        ingest_fixture_lineups(
            database,
            provider_fixture_id="900",
            lineups=incomplete,
            observed_at="later",
        )
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM lineup_players"
        ).fetchone()[0] == 26
