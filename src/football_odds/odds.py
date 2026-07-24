from __future__ import annotations

import pandas as pd

ODDS_CANDIDATES = (
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
)


def find_odds_columns(
    frame: pd.DataFrame,
    candidates: tuple[tuple[str, str, str], ...] = ODDS_CANDIDATES,
) -> tuple[str, str, str]:
    """Restituisce la prima terna 1-X-2 completa disponibile."""
    for columns in candidates:
        if all(column in frame.columns for column in columns):
            return columns
    raise ValueError("Nessuna terna completa di quote 1-X-2 disponibile.")


def remove_margin(odds: pd.DataFrame) -> pd.DataFrame:
    """Converte tre quote decimali in probabilità fair proporzionali."""
    numeric = odds.apply(pd.to_numeric, errors="coerce")
    raw = 1.0 / numeric
    overround = raw.sum(axis=1)
    fair = raw.div(overround, axis=0)
    fair.columns = ["p_home", "p_draw", "p_away"]
    fair["overround"] = overround
    fair["margin"] = overround - 1.0
    return fair


def prepare_matches(frame: pd.DataFrame) -> pd.DataFrame:
    home_col, draw_col, away_col = find_odds_columns(frame)
    identity = ["Date", "HomeTeam", "AwayTeam", "FTR", "Season", "League"]
    missing = [column for column in identity if column not in frame.columns]
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti: {missing}")

    selected = identity + [home_col, draw_col, away_col]
    clean = frame[selected].copy()
    odds_columns = [home_col, draw_col, away_col]
    clean[odds_columns] = clean[odds_columns].apply(pd.to_numeric, errors="coerce")
    valid = (
        clean["FTR"].isin(("H", "D", "A"))
        & clean[odds_columns].notna().all(axis=1)
        & (clean[odds_columns] > 1).all(axis=1)
    )
    clean = clean.loc[valid].reset_index(drop=True)
    probabilities = remove_margin(clean[odds_columns])
    clean = pd.concat([clean, probabilities], axis=1)
    clean["odds_source"] = "/".join(odds_columns)
    clean["Date"] = pd.to_datetime(clean["Date"], dayfirst=True, errors="coerce")
    return clean
