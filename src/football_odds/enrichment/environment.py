from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..data.repository import ResearchDatabase

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
WIKIDATA_URL = "https://www.wikidata.org/w/api.php"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "football-odds-lab/0.1 (research data enrichment)"


@dataclass(frozen=True)
class EnrichmentSummary:
    venues_resolved: int
    venues_unresolved: int
    weather_matches: int


def _best_stadium_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    stadiums = [
        item
        for item in results
        if item.get("type") in {"stadium", "sports_centre", "pitch"}
        or item.get("category") == "leisure"
    ]
    return stadiums[0] if stadiums else None


def _wikidata_venue(
    client: requests.Session, team_name: str, *, timeout: int
) -> dict[str, Any] | None:
    search = client.get(
        WIKIDATA_URL,
        params={
            "action": "wbsearchentities",
            "search": team_name,
            "language": "en",
            "format": "json",
            "type": "item",
            "limit": "5",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    search.raise_for_status()
    candidates = search.json().get("search", [])
    club = next(
        (
            item
            for item in candidates
            if "football club" in str(item.get("description", "")).lower()
            or "football team" in str(item.get("description", "")).lower()
        ),
        None,
    )
    if club is None:
        return None
    entity = client.get(
        WIKIDATA_URL,
        params={
            "action": "wbgetentities",
            "ids": str(club["id"]),
            "props": "claims",
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    entity.raise_for_status()
    claims = entity.json()["entities"][club["id"]].get("claims", {})
    venues = claims.get("P115", [])
    if not venues:
        return None
    venue_id = venues[0]["mainsnak"]["datavalue"]["value"]["id"]
    venue = client.get(
        WIKIDATA_URL,
        params={
            "action": "wbgetentities",
            "ids": venue_id,
            "props": "claims|labels",
            "languages": "en",
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    venue.raise_for_status()
    data = venue.json()["entities"][venue_id]
    coordinates = data.get("claims", {}).get("P625", [])
    if not coordinates:
        return None
    value = coordinates[0]["mainsnak"]["datavalue"]["value"]
    return {
        "name": data.get("labels", {}).get("en", {}).get("value", venue_id),
        "latitude": value["latitude"],
        "longitude": value["longitude"],
        "source_id": venue_id,
        "club_id": club["id"],
    }


def resolve_team_venues(
    database: ResearchDatabase,
    *,
    session: requests.Session | None = None,
    pause_seconds: float = 1.0,
    timeout: int = 30,
) -> tuple[int, int]:
    """Resolve missing home venues via Nominatim and retain provenance."""
    client = session or requests.Session()
    if session is None:
        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        client.mount("https://", HTTPAdapter(max_retries=retry))
    resolved = unresolved = 0
    for team in database.teams_missing_venues():
        query = f"{team['team_name']} football stadium, {team['country']}"
        structured = _wikidata_venue(client, str(team["team_name"]), timeout=timeout)
        if structured is not None:
            database.upsert_team_venue(
                team_id=int(team["team_id"]),
                venue_name=str(structured["name"]),
                latitude=float(structured["latitude"]),
                longitude=float(structured["longitude"]),
                source="Wikidata P115/P625",
                source_id=str(structured["source_id"]),
                display_name=f"club={structured['club_id']}",
            )
            resolved += 1
            if pause_seconds:
                time.sleep(pause_seconds)
            continue
        response = client.get(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "limit": "5"},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        result = _best_stadium_result(response.json())
        if result is None:
            database.record_unresolved_venue(int(team["team_id"]), query)
            unresolved += 1
        else:
            database.upsert_team_venue(
                team_id=int(team["team_id"]),
                venue_name=str(result.get("display_name", query)).split(",")[0],
                latitude=float(result["lat"]),
                longitude=float(result["lon"]),
                source="OpenStreetMap Nominatim",
                source_id=str(result.get("osm_id", "")),
                display_name=str(result.get("display_name", "")),
            )
            resolved += 1
        if pause_seconds:
            time.sleep(pause_seconds)
    return resolved, unresolved


def enrich_historical_weather(
    database: ResearchDatabase,
    *,
    session: requests.Session | None = None,
    timeout: int = 60,
    limit: int | None = None,
) -> int:
    """Populate match-time weather using leakage-safe historical observations."""
    client = session or requests.Session()
    inserted = 0
    matches = database.matches_missing_weather(limit=limit)
    for match in matches:
        kickoff = datetime.fromisoformat(str(match["date"]))
        response = client.get(
            OPEN_METEO_URL,
            params={
                "latitude": str(match["latitude"]),
                "longitude": str(match["longitude"]),
                "start_date": kickoff.date().isoformat(),
                "end_date": kickoff.date().isoformat(),
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "timezone": "auto",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        times = [datetime.fromisoformat(value) for value in hourly.get("time", [])]
        if not times:
            continue
        index = min(range(len(times)), key=lambda i: abs(times[i] - kickoff))
        database.upsert_weather(
            match_id=str(match["match_id"]),
            observed_at=times[index].isoformat(),
            temperature_c=float(hourly["temperature_2m"][index]),
            precipitation_mm=float(hourly["precipitation"][index]),
            wind_kph=float(hourly["wind_speed_10m"][index]),
            source="Open-Meteo ERA5",
        )
        inserted += 1
    return inserted


def run_environment_enrichment(
    database: ResearchDatabase,
    *,
    resolve_venues: bool = True,
    weather_limit: int | None = None,
) -> EnrichmentSummary:
    database.initialize()
    resolved = unresolved = 0
    if resolve_venues:
        resolved, unresolved = resolve_team_venues(database)
    weather = enrich_historical_weather(database, limit=weather_limit)
    return EnrichmentSummary(resolved, unresolved, weather)
