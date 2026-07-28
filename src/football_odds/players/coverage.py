"""Read-only coverage audit for player and lineup data."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://v3.football.api-sports.io"
LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}


def load_env_value(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    prefix = f"{name}="
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().strip("\"'")
    return None


class ApiFootballClient:
    def __init__(
        self,
        api_key: str,
        *,
        request: Callable[..., Any] = requests.get,
        minimum_interval: float = 6.1,
    ) -> None:
        if not api_key.strip():
            raise ValueError("API_FOOTBALL_KEY mancante.")
        self.api_key = api_key
        self.request = request
        self.minimum_interval = minimum_interval
        self._last_request_at: float | None = None
        self.requests_made = 0

    def get(self, endpoint: str, **parameters: object) -> list[dict[str, Any]]:
        if self._last_request_at is not None and self.minimum_interval > 0:
            wait = self.minimum_interval - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
        response = self.request(
            f"{BASE_URL}/{endpoint.lstrip('/')}",
            params=parameters,
            headers={"x-apisports-key": self.api_key},
            timeout=30,
        )
        self._last_request_at = time.monotonic()
        self.requests_made += 1
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("errors") or {}
        if errors:
            raise RuntimeError(f"API-Football {endpoint}: {errors}")
        return list(payload.get("response") or [])


@dataclass(frozen=True)
class PlayerCoverageResult:
    summary: pd.DataFrame
    samples: pd.DataFrame
    outputs: dict[str, Path]
    requests_made: int


def _sample(fixtures: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("sample_per_season deve essere positivo.")
    ordered = sorted(
        fixtures,
        key=lambda item: (
            str(item.get("fixture", {}).get("date", "")),
            int(item.get("fixture", {}).get("id", 0)),
        ),
    )
    if len(ordered) <= count:
        return ordered
    positions = (
        {round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)}
        if count > 1
        else {len(ordered) // 2}
    )
    return [ordered[index] for index in sorted(positions)]


def _facts(lineups: list[dict[str, Any]]) -> dict[str, object]:
    players: list[dict[str, Any]] = []
    starters: list[int] = []
    benches: list[int] = []
    formations = []
    for lineup in lineups:
        start = list(lineup.get("startXI") or [])
        bench = list(lineup.get("substitutes") or [])
        starters.append(len(start))
        benches.append(len(bench))
        if lineup.get("formation"):
            formations.append(str(lineup["formation"]))
        players.extend(item.get("player") or {} for item in [*start, *bench])
    total = len(players)
    return {
        "team_lineups": len(lineups),
        "complete_starting_xi": len(starters) == 2
        and all(value == 11 for value in starters),
        "bench_available": len(benches) == 2 and all(value > 0 for value in benches),
        "players": total,
        "player_id_rate": (
            sum(player.get("id") is not None for player in players) / total
            if total
            else 0.0
        ),
        "position_rate": (
            sum(bool(player.get("pos")) for player in players) / total if total else 0.0
        ),
        "formations_available": len(formations) == 2,
        # API-Football returns the final lineup, not its publication timestamp.
        "published_at_available": False,
    }


def audit_lineups(
    client: ApiFootballClient,
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024),
    leagues: dict[int, str] = LEAGUES,
    sample_per_season: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for league_id, league in leagues.items():
        for season in seasons:
            fixtures = client.get("fixtures", league=league_id, season=season)
            rows = []
            for fixture in _sample(fixtures, sample_per_season):
                fixture_data = fixture.get("fixture") or {}
                fixture_id = int(fixture_data["id"])
                row = {
                    "league_id": league_id,
                    "league": league,
                    "season": season,
                    "fixture_id": fixture_id,
                    "fixture_date": fixture_data.get("date"),
                    **_facts(client.get("fixtures/lineups", fixture=fixture_id)),
                }
                samples.append(row)
                rows.append(row)
            frame = pd.DataFrame(rows)
            rates = (
                {
                    "two_lineups_rate": float("nan"),
                    "complete_starting_xi_rate": float("nan"),
                    "bench_rate": float("nan"),
                    "player_id_rate": float("nan"),
                    "position_rate": float("nan"),
                    "formation_rate": float("nan"),
                }
                if frame.empty
                else {
                    "two_lineups_rate": float(frame["team_lineups"].eq(2).mean()),
                    "complete_starting_xi_rate": float(
                        frame["complete_starting_xi"].mean()
                    ),
                    "bench_rate": float(frame["bench_available"].mean()),
                    "player_id_rate": float(frame["player_id_rate"].mean()),
                    "position_rate": float(frame["position_rate"].mean()),
                    "formation_rate": float(frame["formations_available"].mean()),
                }
            )
            summaries.append(
                {
                    "league_id": league_id,
                    "league": league,
                    "season": season,
                    "fixtures_available": len(fixtures),
                    "fixtures_sampled": len(rows),
                    **rates,
                    "published_at_rate": 0.0,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(samples)


def export_api_football_coverage(
    project_dir: Path,
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024),
    sample_per_season: int = 1,
    client: ApiFootballClient | None = None,
) -> PlayerCoverageResult:
    key = os.getenv("API_FOOTBALL_KEY") or load_env_value(
        project_dir / ".env", "API_FOOTBALL_KEY"
    )
    active = client or ApiFootballClient(key or "")
    summary, samples = audit_lineups(
        active, seasons=seasons, sample_per_season=sample_per_season
    )
    destination = project_dir / "reports" / "player_data"
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "api_football_coverage.csv"
    samples_path = destination / "api_football_lineup_samples.csv"
    report_path = destination / "API_FOOTBALL_COVERAGE.md"
    metadata_path = destination / "api_football_coverage.meta.json"
    summary.to_csv(summary_path, index=False)
    samples.to_csv(samples_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "API-Football",
                "seasons": list(seasons),
                "sample_per_season": sample_per_season,
                "requests_made": active.requests_made,
                "modeling_data_changed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = [
        "two_lineups_rate",
        "complete_starting_xi_rate",
        "bench_rate",
        "player_id_rate",
        "position_rate",
        "formation_rate",
        "published_at_rate",
    ]
    report_path.write_text(
        "# API-Football player coverage audit\n\n"
        f"Campione: {len(samples)} partite; richieste: {active.requests_made}.\n\n"
        "| Controllo | Copertura |\n|---|---:|\n"
        + "\n".join(
            f"| {metric} | {summary[metric].mean():.1%} |" for metric in metrics
        )
        + "\n\nLe lineup non entrano ancora nel dataset modellistico. "
        "L’endpoint non espone il timestamp di pubblicazione: non è quindi "
        "ammissibile nel modello early senza raccolta prospettica timestampata.\n",
        encoding="utf-8",
    )
    return PlayerCoverageResult(
        summary,
        samples,
        {
            "summary": summary_path,
            "samples": samples_path,
            "report": report_path,
            "metadata": metadata_path,
        },
        active.requests_made,
    )
