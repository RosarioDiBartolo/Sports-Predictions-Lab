import json
from datetime import datetime

import pytest

from football_odds.data.repository import ResearchDatabase
from football_odds.ingestion.contracts import MatchRecord
from football_odds.ingestion.providers.seriea_feed import (
    SerieAFeedClient,
    audit_seriea_feed,
    backfill_seriea_feed,
    export_seriea_feed_audit,
)


class FakeClient:
    requests_made = 0

    def seasons(self):
        return [
            {"seasonName": season, "seasonId": f"s-{index}"}
            for index, season in enumerate(("2022/2023", "2023/2024", "2024/2025"))
        ]

    def matches(self, season_id):
        return [
            {
                "matchId": f"{season_id}-m{index}",
                "matchDateUtc": f"2024-01-0{index + 1}T12:00:00Z",
            }
            for index in range(3)
        ]

    def lineup(self, season_id, match_id):
        def player(index, events=None):
            return {
                "playerId": f"{match_id}-p{index}",
                "displayName": f"Player {index}",
                "roleLabel": "Midfielder",
                "events": events or [],
            }

        def team(prefix):
            starters = [player(index) for index in range(11)]
            starters[0]["events"] = [
                {
                    "type": "substitution-out",
                    "time": 60,
                    "additionalTime": 0,
                }
            ]
            bench = [
                player(
                    20,
                    [
                        {
                            "type": "substitution-in",
                            "time": 60,
                            "additionalTime": 0,
                        }
                    ],
                )
            ]
            for item in [*starters, *bench]:
                item["playerId"] = f"{prefix}-{item['playerId']}"
            return {
                "teamId": prefix,
                "mediaName": prefix,
                "tacticalFormation": "4-3-3",
                "fielded": starters,
                "benched": bench,
            }

        return {
            "matchId": match_id,
            "home": team("Home"),
            "away": team("Away"),
        }


def test_audit_has_complete_contract_and_reconstructs_minutes():
    summary, matches, players, raw = audit_seriea_feed(FakeClient(), sample_matches=3)

    assert len(summary) == 3
    assert len(matches) == 3
    assert len(raw) == 3
    assert summary["complete_starting_xi_rate"].eq(1.0).all()
    assert summary["substitutions"].eq(2).all()
    assert set(players.loc[players["entered"], "minutes_estimated"]) == {30.0}
    assert set(players.loc[players["left"], "minutes_estimated"]) == {60.0}


def test_export_preserves_raw_evidence_without_changing_model_data(tmp_path):
    result = export_seriea_feed_audit(
        tmp_path,
        sample_matches=3,
        client=FakeClient(),
    )

    assert all(path.exists() for path in result.outputs.values())
    metadata = json.loads(result.outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["modeling_data_changed"] is False
    assert len(json.loads(result.outputs["raw"].read_text(encoding="utf-8"))) == 3


def test_http_client_contract_and_resource_paths():
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            url = calls[-1][0]
            if url.endswith("/seasons"):
                return {"seasons": [{"seasonName": "2024/2025"}]}
            if url.endswith("/matches"):
                return {"matches": [{"matchId": "m1"}]}
            return {"matchId": "m1", "home": {}, "away": {}}

    def request(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    client = SerieAFeedClient(request=request, minimum_interval=0)

    assert client.seasons()[0]["seasonName"] == "2024/2025"
    assert client.matches("s1")[0]["matchId"] == "m1"
    assert client.lineup("s1", "m1")["matchId"] == "m1"
    assert client.requests_made == 3
    assert all(call[1]["params"] == {"locale": "en-GB"} for call in calls)
    assert all("User-Agent" in call[1]["headers"] for call in calls)


def test_audit_rejects_small_sample_and_unknown_season():
    with pytest.raises(ValueError, match="almeno una partita"):
        audit_seriea_feed(FakeClient(), sample_matches=2)
    with pytest.raises(ValueError, match="non trovate"):
        audit_seriea_feed(
            FakeClient(),
            seasons=("1999/2000",),
            sample_matches=1,
        )


def test_backfill_maps_persists_minutes_and_resumes(tmp_path):
    database = ResearchDatabase(tmp_path / "data" / "football_odds.sqlite3")
    database.initialize()
    database.upsert_match(
        "Football-Data.co.uk",
        MatchRecord(
            provider_match_id="canonical-1",
            date=datetime(2024, 1, 1, 12),
            season="2223",
            league_code="I1",
            home_team="Home",
            away_team="Away",
            home_goals=1,
            away_goals=0,
            result="H",
        ),
        "Serie A",
        "Italy",
    )

    class BackfillClient(FakeClient):
        def matches(self, season_id):
            return [
                {
                    "matchId": "feed-1",
                    "matchDateUtc": "2024-01-01T12:00:00Z",
                    "home": {"teamId": "home", "mediaName": "Home"},
                    "away": {"teamId": "away", "mediaName": "Away"},
                }
            ]

    client = BackfillClient()
    result = backfill_seriea_feed(
        tmp_path,
        seasons=("2022/2023",),
        client=client,
    )
    assert (result.mapped_matches, result.imported_matches) == (1, 1)

    with database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM fixture_lineups").fetchone()[0]
            == 2
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM player_match_lineup_stats"
            ).fetchone()[0]
            == 24
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM player_match_lineup_stats WHERE minute_in=60"
            ).fetchone()[0]
            == 2
        )

    resumed = backfill_seriea_feed(
        tmp_path,
        seasons=("2022/2023",),
        client=BackfillClient(),
    )
    assert (resumed.already_complete, resumed.imported_matches) == (1, 0)
