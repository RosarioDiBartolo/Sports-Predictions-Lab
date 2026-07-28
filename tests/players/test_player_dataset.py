import csv
import json
from datetime import datetime

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import MatchRecord
from football_odds.players.dataset import build_player_dataset
from football_odds.players.observations import ingest_fixture_lineups


def _match(match_id):
    return MatchRecord(
        provider_match_id=match_id,
        date=datetime(2025, 1, 10),
        season="2425",
        league_code="I1",
        home_team="Home FC",
        away_team="Away FC",
        home_goals=2,
        away_goals=1,
        result="H",
    )


def _lineup(team, offset=0, missing_position=False):
    def player(number):
        return {
            "player": {
                "id": offset + number,
                "name": f"{team} Player {number}",
                "pos": None if missing_position and number == 1 else "M",
            }
        }

    return {
        "team": {"id": offset + 100, "name": team},
        "startXI": [player(number) for number in range(1, 12)],
        "substitutes": [player(12), player(13)],
    }


def test_build_player_dataset_exports_training_ready_contract(tmp_path):
    database_path = tmp_path / "data" / "football_odds.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match("API-Football", _match("900"), "Serie A", "Italy")
    ingest_fixture_lineups(
        database,
        provider_fixture_id="900",
        lineups=[_lineup("Home FC"), _lineup("Away FC", 1000)],
        observed_at="2025-01-11T00:00:00+00:00",
    )

    result = build_player_dataset(tmp_path, database_path=database_path)

    assert result.training_ready == 1
    assert result.quarantined == 0
    with result.outputs["dataset"].open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert len(json.loads(row["home_starters"])) == 11
    assert len(json.loads(row["home_bench"])) == 2
    assert row["home_lineup_provider"] == "API-Football"
    census = json.loads(result.outputs["feature_census"].read_text(encoding="utf-8"))
    assert census["raw_observations"]["bench"] == 4
    assert census["fallback_contract"]["quality_indicators_required"] is True


def test_build_player_dataset_quarantines_missing_roles_and_lineups(tmp_path):
    database_path = tmp_path / "data" / "football_odds.sqlite3"
    database = ResearchDatabase(database_path)
    database.initialize()
    database.upsert_match("API-Football", _match("900"), "Serie A", "Italy")
    ingest_fixture_lineups(
        database,
        provider_fixture_id="900",
        lineups=[
            _lineup("Home FC", missing_position=True),
            _lineup("Away FC", 1000),
        ],
        observed_at="2025-01-11T00:00:00+00:00",
    )
    database.upsert_match(
        "API-Football",
        MatchRecord(
            provider_match_id="901",
            date=datetime(2025, 1, 12),
            season="2425",
            league_code="I1",
            home_team="Other Home",
            away_team="Other Away",
            home_goals=0,
            away_goals=0,
            result="D",
        ),
        "Serie A",
        "Italy",
    )

    result = build_player_dataset(tmp_path, database_path=database_path)

    assert result.training_ready == 0
    assert result.quarantined == 2
    assert result.reasons == {
        "home_starter_role_missing": 1,
        "missing_away_lineup": 1,
        "missing_home_lineup": 1,
    }
    lines = result.outputs["quarantine"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
