"""Reconcile the BeatTheBookie hourly 1X2 series to canonical matches."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio

BOOKMAKERS = (
    "Interwetten",
    "bwin",
    "bet-at-home",
    "Unibet",
    "Stan James",
    "Expekt",
    "10Bet",
    "William Hill",
    "bet365",
    "Pinnacle Sports",
    "DOXXbet",
    "Betsafe",
    "Betway",
    "888sport",
    "Ladbrokes",
    "Betclic",
    "Sportingbet",
    "myBet",
    "Betsson",
    "188BET",
    "Jetbull",
    "Paddy Power",
    "Tipico",
    "Coral",
    "SBOBET",
    "BetVictor",
    "12BET",
    "Titanbet",
    "youwin",
    "ComeOn",
    "Betadonis",
    "Betfair Sports",
)
SNAPSHOT_INDEX = 70
HOURS_BEFORE_KICKOFF = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_name(value: object) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    ascii_value = re.sub(
        r"\b(fc|cf|afc|ac|calcio|club|de|the)\b", " ", ascii_value
    )
    return re.sub(r"[^a-z0-9]", "", ascii_value)


def _source_matches(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_dir.glob("*matches.csv.gz")):
        frame = pd.read_csv(path, encoding="latin1")
        frame.columns = frame.columns.str.strip()
        frame["series_file"] = path.name.replace("_matches", "")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("BeatTheBookie match metadata are missing.")
    matches = pd.concat(frames, ignore_index=True).drop_duplicates("match_id")
    scores = matches["score"].str.extract(r"(\d+)\s*:\s*(\d+)").astype("Int64")
    matches[["home_goals", "away_goals"]] = scores
    matches["match_datetime"] = pd.to_datetime(
        matches["match_datetime"], utc=True, format="mixed"
    )
    matches["day"] = matches["match_datetime"].dt.date.astype(str)
    matches["home_key"] = matches["home_team"].map(_normalized_name)
    matches["away_key"] = matches["away_team"].map(_normalized_name)
    return matches


def _reconcile(
    canonical: pd.DataFrame,
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = canonical.copy()
    canonical["day"] = pd.to_datetime(
        canonical["date"], format="mixed"
    ).dt.date.astype(str)
    canonical["home_key"] = canonical["home_team"].map(_normalized_name)
    canonical["away_key"] = canonical["away_team"].map(_normalized_name)
    groups = {
        key: group
        for key, group in source.groupby(["day", "home_goals", "away_goals"])
    }
    accepted: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    for row in canonical.itertuples(index=False):
        key = (row.day, row.home_goals, row.away_goals)
        candidates = groups.get(key)
        if candidates is None:
            continue
        scored = []
        for source_row in candidates.itertuples(index=False):
            home_score = ratio(row.home_key, source_row.home_key)
            away_score = ratio(row.away_key, source_row.away_key)
            scored.append(
                (
                    (home_score + away_score) / 2,
                    min(home_score, away_score),
                    home_score,
                    away_score,
                    source_row,
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0]
        margin = best[0] - scored[1][0] if len(scored) > 1 else 100.0
        record = {
            "match_id": str(row.match_id),
            "source_match_id": str(best[4].match_id),
            "canonical_home_team": row.home_team,
            "canonical_away_team": row.away_team,
            "source_home_team": best[4].home_team,
            "source_away_team": best[4].away_team,
            "home_similarity": best[2],
            "away_similarity": best[3],
            "mean_similarity": best[0],
            "runner_up_margin": margin,
            "fixture_kickoff": best[4].match_datetime,
            "series_file": best[4].series_file,
        }
        if best[0] >= 78 and best[1] >= 65 and margin >= 8:
            accepted.append(record)
        elif best[0] >= 65:
            record["reason"] = "ambiguous_or_below_threshold"
            quarantined.append(record)
    return pd.DataFrame(accepted), pd.DataFrame(quarantined)


def _cutoff_rows(raw_dir: Path, reconciled: pd.DataFrame) -> pd.DataFrame:
    wanted = set(reconciled["source_match_id"])
    columns = ["match_id"]
    for bookie in range(1, len(BOOKMAKERS) + 1):
        columns.extend(
            f"{selection}_b{bookie}_{SNAPSHOT_INDEX}"
            for selection in ("home", "draw", "away")
        )
    frames = []
    for filename in sorted(reconciled["series_file"].unique()):
        path = raw_dir / filename
        for chunk in pd.read_csv(path, usecols=columns, chunksize=10_000):
            selected = chunk.loc[chunk["match_id"].astype(str).isin(wanted)]
            if not selected.empty:
                frames.append(selected)
    if not frames:
        raise ValueError("No reconciled match has an hourly odds row.")
    odds = pd.concat(frames, ignore_index=True).drop_duplicates("match_id")
    lookup = reconciled.set_index("source_match_id")
    rows: list[dict[str, object]] = []
    collected_at = datetime.now(timezone.utc).isoformat()
    for odds_row in odds.itertuples(index=False):
        source_id = str(odds_row.match_id)
        match = lookup.loc[source_id]
        kickoff = pd.Timestamp(match["fixture_kickoff"])
        snapshot_at = kickoff - pd.Timedelta(hours=HOURS_BEFORE_KICKOFF)
        for index, bookmaker in enumerate(BOOKMAKERS, start=1):
            values = {
                "H": getattr(odds_row, f"home_b{index}_{SNAPSHOT_INDEX}"),
                "D": getattr(odds_row, f"draw_b{index}_{SNAPSHOT_INDEX}"),
                "A": getattr(odds_row, f"away_b{index}_{SNAPSHOT_INDEX}"),
            }
            if any(pd.isna(value) or float(value) <= 1 for value in values.values()):
                continue
            for selection, decimal_odds in values.items():
                rows.append(
                    {
                        "provider": "BeatTheBookie",
                        "provider_fixture_id": source_id,
                        "match_id": match["match_id"],
                        "fixture_kickoff": kickoff.isoformat(),
                        "prediction_cutoff": kickoff.isoformat(),
                        "bookmaker": bookmaker,
                        "market": "1X2",
                        "selection": selection,
                        "decimal_odds": float(decimal_odds),
                        "provider_updated_at": snapshot_at.isoformat(),
                        "collected_at": collected_at,
                        "snapshot_hours_before_kickoff": HOURS_BEFORE_KICKOFF,
                        "home_similarity": match["home_similarity"],
                        "away_similarity": match["away_similarity"],
                        "runner_up_margin": match["runner_up_margin"],
                    }
                )
    return pd.DataFrame(rows)


def reconcile_beat_the_bookie(project: Path) -> Path:
    """Write a cutoff-valid snapshot plus manifest and mapping quarantine."""
    raw_dir = project / "data/raw/beat_the_bookie"
    canonical_path = project / "data/processed/modeling_features_all.csv"
    lineup_path = project / "data/processed/player_training_ready.csv"
    canonical = pd.read_csv(canonical_path)
    lineup_ids = set(
        pd.read_csv(lineup_path, usecols=["match_id"])["match_id"].astype(str)
    )
    canonical = canonical.loc[canonical["match_id"].astype(str).isin(lineup_ids)]
    source = _source_matches(raw_dir)
    reconciled, quarantine = _reconcile(canonical, source)
    rows = _cutoff_rows(raw_dir, reconciled)
    output = raw_dir / "reconciled_cutoff_snapshot.csv"
    quarantine_path = raw_dir / "reconciliation_quarantine.csv"
    manifest_path = raw_dir / "reconciled_cutoff_snapshot.manifest.json"
    rows.to_csv(output, index=False)
    quarantine.to_csv(quarantine_path, index=False)
    complete_groups = rows.groupby(["match_id", "bookmaker"])["selection"].nunique()
    source_paths = sorted(raw_dir.glob("*.csv.gz"))
    manifest_path.write_text(
        json.dumps(
            {
                "provider": "BeatTheBookie",
                "license": "CC BY-SA 4.0",
                "source": (
                    "https://www.kaggle.com/datasets/austro/"
                    "beat-the-bookie-worldwide-football-dataset"
                ),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_index": SNAPSHOT_INDEX,
                "snapshot_hours_before_kickoff": HOURS_BEFORE_KICKOFF,
                "canonical_matches_reconciled": int(rows["match_id"].nunique()),
                "complete_match_bookmaker_markets": int(complete_groups.eq(3).sum()),
                "rows": len(rows),
                "quarantined_mappings": len(quarantine),
                "inputs": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in [canonical_path, lineup_path, *source_paths]
                ],
                "output_sha256": _sha256(output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
