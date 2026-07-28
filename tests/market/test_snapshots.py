from datetime import date

import pandas as pd

from football_odds.market.snapshots import collect_timestamped_odds


class Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "errors": [],
            "paging": {"current": 1, "total": 1},
            "response": [
                {
                    "league": {"id": 135, "season": 2026},
                    "fixture": {
                        "id": 10,
                        "date": "2026-07-28T20:00:00+00:00",
                    },
                    "update": "2026-07-28T12:00:00+00:00",
                    "bookmakers": [
                        {
                            "id": 1,
                            "name": "Example",
                            "bets": [
                                {
                                    "id": 1,
                                    "values": [
                                        {"value": "Home", "odd": "2.0"},
                                        {"value": "Draw", "odd": "3.2"},
                                        {"value": "Away", "odd": "4.0"},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }


def test_collect_preserves_raw_timestamped_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("API_FOOTBALL_KEY", "secret")
    result = collect_timestamped_odds(
        tmp_path,
        target=date(2026, 7, 28),
        request=lambda *args, **kwargs: Response(),
    )

    frame = pd.read_csv(result.normalized)
    assert result.fixtures == 1
    assert result.rows == 3
    assert result.complete
    assert result.pages_available == 1
    assert result.invalid_values == 0
    assert set(frame["selection"]) == {"H", "D", "A"}
    assert frame["provider_updated_at"].eq(
        "2026-07-28T12:00:00+00:00"
    ).all()
    assert frame["collected_at"].notna().all()
    assert result.raw.is_file()
    assert result.manifest.is_file()
