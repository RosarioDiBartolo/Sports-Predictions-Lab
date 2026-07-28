from pathlib import Path

import pytest

from football_odds.players.coverage import (
    ApiFootballClient,
    _facts,
    _sample,
    export_api_football_coverage,
    load_env_value,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return {"errors": [], "response": self.payload}


def request(url, *, params, **kwargs):
    if url.endswith("/fixtures"):
        return Response(
            [
                {"fixture": {"id": 1, "date": "2024-01-01"}},
                {"fixture": {"id": 2, "date": "2024-06-01"}},
                {"fixture": {"id": 3, "date": "2024-12-01"}},
            ]
        )
    starters = [{"player": {"id": value, "pos": "M"}} for value in range(11)]
    bench = [{"player": {"id": 20, "pos": "F"}}]
    return Response(
        [
            {"formation": "4-3-3", "startXI": starters, "substitutes": bench},
            {"formation": "4-4-2", "startXI": starters, "substitutes": bench},
        ]
    )


def test_export_is_read_only_and_deterministic(tmp_path):
    client = ApiFootballClient("secret", request=request, minimum_interval=0)
    result = export_api_football_coverage(
        tmp_path, seasons=(2024,), sample_per_season=2, client=client
    )
    assert len(result.summary) == 5
    assert len(result.samples) == 10
    assert result.requests_made == 15
    assert result.summary["complete_starting_xi_rate"].eq(1).all()
    assert result.summary["published_at_rate"].eq(0).all()
    assert "non entrano ancora" in result.outputs["report"].read_text(encoding="utf-8")


def test_helpers_and_failures(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("API_FOOTBALL_KEY='value'\n", encoding="utf-8")
    assert load_env_value(dotenv, "API_FOOTBALL_KEY") == "value"
    assert load_env_value(Path("missing"), "X") is None
    with pytest.raises(ValueError, match="positivo"):
        _sample([], 0)
    assert _facts([])["player_id_rate"] == 0
    with pytest.raises(ValueError, match="mancante"):
        ApiFootballClient("")

    def error(*args, **kwargs):
        response = Response([])
        response.json = lambda: {"errors": {"plan": "denied"}, "response": []}
        return response

    with pytest.raises(RuntimeError, match="denied"):
        ApiFootballClient("x", request=error, minimum_interval=0).get("fixtures")
