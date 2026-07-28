"""Prospective, timestamped API-Football pre-match odds snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_URL = "https://v3.football.api-sports.io/odds"
SELECTIONS = {"Home": "H", "Draw": "D", "Away": "A"}


@dataclass(frozen=True)
class SnapshotCollection:
    raw: Path
    normalized: Path
    manifest: Path
    fixtures: int
    rows: int
    requests: int
    pages_available: int
    complete: bool
    invalid_values: int


def _api_key(project: Path) -> str:
    key = os.getenv("API_FOOTBALL_KEY")
    env_path = project / ".env"
    if not key and env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            if raw.startswith("API_FOOTBALL_KEY="):
                key = raw.split("=", 1)[1].strip().strip("\"'")
                break
    if not key:
        raise ValueError("API_FOOTBALL_KEY mancante.")
    return key


def _request_pages(
    target: date,
    key: str,
    request: Callable[..., Any],
    max_pages: int,
) -> tuple[list[dict[str, Any]], int, datetime, int]:
    collected_at = datetime.now(timezone.utc)
    responses: list[dict[str, Any]] = []
    page = 1
    total = 1
    while page <= min(total, max_pages):
        response = request(
            API_URL,
            params={"date": target.isoformat(), "bet": 1, "page": page},
            headers={"x-apisports-key": key},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("errors") or {}
        if errors:
            raise RuntimeError(f"API-Football odds: {errors}")
        responses.append(payload)
        paging = payload.get("paging") or {}
        total = int(paging.get("total") or 1)
        page += 1
    return responses, len(responses), collected_at, total


def _rows(
    responses: list[dict[str, Any]],
    collected_at: datetime,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    invalid_values = 0
    for payload in responses:
        for item in payload.get("response") or []:
            fixture = item.get("fixture") or {}
            league = item.get("league") or {}
            provider_updated_at = pd.to_datetime(
                item.get("update"), utc=True, errors="coerce"
            )
            if pd.isna(provider_updated_at):
                raise ValueError("Snapshot privo del timestamp update del provider.")
            for bookmaker in item.get("bookmakers") or []:
                for bet in bookmaker.get("bets") or []:
                    if int(bet.get("id", -1)) != 1:
                        continue
                    for value in bet.get("values") or []:
                        selection = SELECTIONS.get(str(value.get("value")))
                        if selection is None:
                            continue
                        decimal_odds = float(value["odd"])
                        if decimal_odds <= 1:
                            invalid_values += 1
                            continue
                        rows.append(
                            {
                                "provider": "API-Football",
                                "provider_fixture_id": str(fixture["id"]),
                                "fixture_kickoff": fixture["date"],
                                "league_id": league.get("id"),
                                "season": league.get("season"),
                                "bookmaker_id": bookmaker.get("id"),
                                "bookmaker": bookmaker.get("name"),
                                "market": "1X2",
                                "selection": selection,
                                "decimal_odds": decimal_odds,
                                "provider_updated_at": provider_updated_at.isoformat(),
                                "collected_at": collected_at.isoformat(),
                            }
                        )
    return pd.DataFrame(rows), invalid_values


def collect_timestamped_odds(
    project: Path,
    *,
    target: date,
    request: Callable[..., Any] = requests.get,
    max_pages: int = 3,
) -> SnapshotCollection:
    """Collect and preserve one immutable prospective odds snapshot."""
    if max_pages < 1:
        raise ValueError("max_pages deve essere positivo.")
    responses, requests_made, collected_at, pages_available = _request_pages(
        target, _api_key(project), request, max_pages
    )
    stamp = collected_at.strftime("%Y%m%dT%H%M%S%fZ")
    destination = project / "data" / "raw" / "api_football_odds"
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / f"{target.isoformat()}-{stamp}.json"
    normalized_path = destination / f"{target.isoformat()}-{stamp}.csv"
    manifest_path = destination / f"{target.isoformat()}-{stamp}.manifest.json"
    raw_bytes = json.dumps(responses, ensure_ascii=False, indent=2).encode("utf-8")
    raw_path.write_bytes(raw_bytes)
    frame, invalid_values = _rows(responses, collected_at)
    frame.to_csv(normalized_path, index=False)
    fixture_count = (
        int(frame["provider_fixture_id"].nunique()) if not frame.empty else 0
    )
    manifest_path.write_text(
        json.dumps(
            {
                "provider": "API-Football",
                "endpoint": "/odds",
                "target_date": target.isoformat(),
                "bet_id": 1,
                "market": "1X2",
                "collected_at": collected_at.isoformat(),
                "requests": requests_made,
                "pages_available": pages_available,
                "pages_collected": requests_made,
                "complete": requests_made == pages_available,
                "fixtures": fixture_count,
                "rows": len(frame),
                "invalid_values_preserved_in_raw": invalid_values,
                "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "raw": str(raw_path),
                "normalized": str(normalized_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return SnapshotCollection(
        raw=raw_path,
        normalized=normalized_path,
        manifest=manifest_path,
        fixtures=fixture_count,
        rows=len(frame),
        requests=requests_made,
        pages_available=pages_available,
        complete=requests_made == pages_available,
        invalid_values=invalid_values,
    )
