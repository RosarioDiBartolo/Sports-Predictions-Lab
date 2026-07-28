"""Strategy discovery against timestamped bookmaker markets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")
PROBABILITY_COLUMNS = (
    "probability_home",
    "probability_draw",
    "probability_away",
)
MARKET_COLUMNS = (
    "market_home_probability",
    "market_draw_probability",
    "market_away_probability",
)


@dataclass
class EdgeDiscoveryResult:
    """Frozen-rule discovery and untouched holdout evaluation."""

    selected_rule: dict[str, object]
    summary: pd.DataFrame
    season_stability: pd.DataFrame
    candidates: pd.DataFrame
    outputs: dict[str, Path]
    promoted: bool


def _bootstrap_roi(
    profits: np.ndarray, *, samples: int = 5000, seed: int = 20260725
) -> dict[str, float]:
    if samples <= 0:
        raise ValueError("samples deve essere positivo.")
    if len(profits) == 0:
        return {"roi": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = generator.choice(profits, size=len(profits), replace=True).mean()
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "roi": float(profits.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def prepare_edge_dataset(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    analytics: pd.DataFrame,
) -> pd.DataFrame:
    """Join OOS predictions to pre-match features and average closing odds."""
    required_predictions = {
        "match_id",
        "season",
        "league",
        "result",
        "model_version",
        "dataset_version",
        "prediction_cutoff",
        *PROBABILITY_COLUMNS,
    }
    required_features = {
        "match_id",
        "elo_difference",
        "home_matches_played",
        "away_matches_played",
        *MARKET_COLUMNS,
    }
    required_analytics = {
        "match_id",
        "bookmaker",
        "selection",
        "odds",
        "opening_or_closing",
        "timestamp",
    }
    for name, frame, required in (
        ("predictions", predictions, required_predictions),
        ("features", features, required_features),
        ("analytics", analytics, required_analytics),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name}: colonne mancanti {sorted(missing)}")

    candidates = analytics.loc[
        analytics["bookmaker"].eq("Market Average"),
        ["match_id", "selection", "odds", "timestamp"],
    ].dropna(subset=["odds"])
    if candidates["timestamp"].isna().any() or candidates["timestamp"].eq("").any():
        raise ValueError("Le quote candidate devono avere un timestamp verificabile.")
    candidates = candidates.merge(
        predictions[["match_id", "prediction_cutoff"]].drop_duplicates(),
        on="match_id",
        validate="many_to_one",
    )
    candidates["_timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True)
    candidates["_cutoff"] = pd.to_datetime(candidates["prediction_cutoff"], utc=True)
    available = candidates.loc[candidates["_timestamp"].le(candidates["_cutoff"])]
    available = available.sort_values("_timestamp").drop_duplicates(
        ["match_id", "selection"], keep="last"
    )
    odds = available.pivot_table(
        index="match_id", columns="selection", values="odds", aggfunc="median"
    ).rename(columns=lambda outcome: f"odds_{outcome}")
    feature_columns = [
        "match_id",
        "elo_difference",
        "home_matches_played",
        "away_matches_played",
        *MARKET_COLUMNS,
    ]
    data = predictions.merge(
        features[feature_columns], on="match_id", how="inner", validate="one_to_one"
    ).merge(odds, on="match_id", how="inner", validate="one_to_one")
    probabilities = data[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    market = data[list(MARKET_COLUMNS)].to_numpy(dtype=float)
    pick_index = probabilities.argmax(axis=1)
    labels = np.asarray(OUTCOMES)
    data["pick"] = labels[pick_index]
    data["model_probability"] = probabilities[np.arange(len(data)), pick_index]
    data["market_probability"] = market[np.arange(len(data)), pick_index]
    data["edge"] = data["model_probability"] - data["market_probability"]
    odds_matrix = data[[f"odds_{outcome}" for outcome in OUTCOMES]].to_numpy(
        dtype=float
    )
    data["odds"] = odds_matrix[np.arange(len(data)), pick_index]
    data["profit"] = np.where(data["result"].eq(data["pick"]), data["odds"] - 1.0, -1.0)
    data["absolute_elo_difference"] = data["elo_difference"].abs()
    data["experienced"] = (
        data[["home_matches_played", "away_matches_played"]].min(axis=1) >= 5
    )
    return data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["odds", "model_probability", "market_probability"]
    )


def _rule_mask(data: pd.DataFrame, rule: dict[str, object]) -> pd.Series:
    mask = (
        data["model_probability"].ge(float(rule["min_confidence"]))
        & data["edge"].ge(float(rule["min_edge"]))
        & data["absolute_elo_difference"].ge(float(rule["min_abs_elo"]))
        & data["pick"].isin(list(str(rule["picks"])))
    )
    if bool(rule["require_experience"]):
        mask &= data["experienced"]
    return mask


def _probabilistic_metrics(data: pd.DataFrame) -> dict[str, float]:
    if data.empty:
        return {
            "model_log_loss": float("nan"),
            "market_log_loss": float("nan"),
            "log_loss_delta": float("nan"),
            "model_brier": float("nan"),
            "market_brier": float("nan"),
            "brier_delta": float("nan"),
        }
    outcome_indices = dict(zip(OUTCOMES, range(3), strict=True))
    true_index = data["result"].map(outcome_indices).to_numpy()
    model = np.clip(data[list(PROBABILITY_COLUMNS)].to_numpy(float), 1e-15, 1)
    market = np.clip(data[list(MARKET_COLUMNS)].to_numpy(float), 1e-15, 1)
    observed = np.eye(3)[true_index]
    model_log_loss = float(-np.log(model[np.arange(len(data)), true_index]).mean())
    market_log_loss = float(-np.log(market[np.arange(len(data)), true_index]).mean())
    model_brier = float(np.square(model - observed).sum(axis=1).mean())
    market_brier = float(np.square(market - observed).sum(axis=1).mean())
    return {
        "model_log_loss": model_log_loss,
        "market_log_loss": market_log_loss,
        "log_loss_delta": model_log_loss - market_log_loss,
        "model_brier": model_brier,
        "market_brier": market_brier,
        "brier_delta": model_brier - market_brier,
    }


def discover_edges(
    data: pd.DataFrame,
    *,
    holdout_seasons: tuple[str, ...] | None = None,
    minimum_discovery_bets: int = 200,
    bootstrap_samples: int = 5000,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Choose on discovery data and evaluate the frozen rule once on holdout."""
    seasons = sorted(data["season"].astype(str).unique(), key=_season_key)
    holdout_seasons = holdout_seasons or tuple(seasons[-2:])
    if len(holdout_seasons) < 1 or not set(holdout_seasons).issubset(seasons):
        raise ValueError("Le stagioni holdout devono esistere nelle predizioni OOS.")
    discovery = data.loc[~data["season"].astype(str).isin(holdout_seasons)]
    holdout = data.loc[data["season"].astype(str).isin(holdout_seasons)]
    if discovery.empty or holdout.empty:
        raise ValueError("Discovery e holdout devono contenere almeno una partita.")

    rows: list[dict[str, object]] = []
    grid = product(
        (0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75),
        (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10),
        (0.0, 50.0, 100.0, 150.0, 200.0),
        ("HDA", "HA", "H", "A"),
        (False, True),
    )
    for confidence, edge, elo, picks, experience in grid:
        rule = {
            "min_confidence": confidence,
            "min_edge": edge,
            "min_abs_elo": elo,
            "picks": picks,
            "require_experience": experience,
        }
        selected = discovery.loc[_rule_mask(discovery, rule)]
        if len(selected) < minimum_discovery_bets:
            continue
        evidence = _bootstrap_roi(
            selected["profit"].to_numpy(float),
            samples=min(bootstrap_samples, 1000),
        )
        rows.append(
            {
                **rule,
                "discovery_bets": len(selected),
                "discovery_roi": evidence["roi"],
                "discovery_ci_low": evidence["ci_low"],
                "discovery_ci_high": evidence["ci_high"],
            }
        )
    if not rows:
        raise ValueError("Nessuna regola raggiunge il minimo di puntate discovery.")
    candidates = pd.DataFrame(rows).sort_values(
        ["discovery_roi", "discovery_bets"], ascending=[False, False]
    )
    best = candidates.iloc[0]
    selected_rule = {
        "min_confidence": float(best["min_confidence"]),
        "min_edge": float(best["min_edge"]),
        "min_abs_elo": float(best["min_abs_elo"]),
        "picks": str(best["picks"]),
        "require_experience": bool(best["require_experience"]),
    }
    selected_rule["discovery_seasons"] = sorted(
        discovery["season"].astype(str).unique()
    )
    selected_rule["holdout_seasons"] = list(holdout_seasons)
    selected_rule["rules_tested"] = len(candidates)

    summary_rows = []
    selected_periods = {
        "discovery": discovery.loc[_rule_mask(discovery, selected_rule)],
        "holdout": holdout.loc[_rule_mask(holdout, selected_rule)],
    }
    for period, selected in selected_periods.items():
        roi = _bootstrap_roi(
            selected["profit"].to_numpy(float), samples=bootstrap_samples
        )
        summary_rows.append(
            {
                "period": period,
                "bets": len(selected),
                "profit_units": float(selected["profit"].sum()),
                **roi,
                **_probabilistic_metrics(selected),
            }
        )
    summary = pd.DataFrame(summary_rows)

    season_rows = []
    frozen_holdout = selected_periods["holdout"]
    for season, selected in frozen_holdout.groupby(
        frozen_holdout["season"].astype(str), sort=True
    ):
        roi = _bootstrap_roi(
            selected["profit"].to_numpy(float), samples=bootstrap_samples
        )
        season_rows.append({"season": season, "bets": len(selected), **roi})
    stability = pd.DataFrame(season_rows)
    holdout_row = summary.loc[summary["period"].eq("holdout")].iloc[0]
    probabilistic_edge = bool(
        holdout_row["log_loss_delta"] < 0 or holdout_row["brier_delta"] < 0
    )
    stable_roi = bool(
        len(stability) == len(holdout_seasons) and stability["roi"].gt(0).all()
    )
    promoted = bool(
        holdout_row["bets"] > 0
        and holdout_row["ci_low"] > 0
        and probabilistic_edge
        and stable_roi
    )
    selected_rule["promoted"] = promoted
    return selected_rule, summary, stability, candidates, promoted


def export_edge_discovery(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    analytics: pd.DataFrame,
    destination: Path,
    *,
    holdout_seasons: tuple[str, ...] | None = None,
    minimum_discovery_bets: int = 200,
    bootstrap_samples: int = 5000,
) -> EdgeDiscoveryResult:
    """Run edge discovery and persist an auditable report bundle."""
    destination.mkdir(parents=True, exist_ok=True)
    data = prepare_edge_dataset(predictions, features, analytics)
    rule, summary, stability, candidates, promoted = discover_edges(
        data,
        holdout_seasons=holdout_seasons,
        minimum_discovery_bets=minimum_discovery_bets,
        bootstrap_samples=bootstrap_samples,
    )
    rule_path = destination / "selected_rule.json"
    summary_path = destination / "edge_summary.csv"
    stability_path = destination / "edge_stability_by_season.csv"
    candidates_path = destination / "discovery_candidates.csv"
    report_path = destination / "EDGE_DISCOVERY_REPORT.md"
    rule_path.write_text(json.dumps(rule, indent=2), encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    stability.to_csv(stability_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    holdout = summary.loc[summary["period"].eq("holdout")].iloc[0]
    lines = [
        "# Edge Discovery Report",
        "",
        f"Verdetto: **{'PROMOSSO' if promoted else 'NON PROMOSSO'}**.",
        "",
        "## Regola congelata",
        "",
        f"- Confidenza minima: {float(rule['min_confidence']):.1%}",
        f"- Edge minimo sul mercato: {float(rule['min_edge']):.1%}",
        f"- Differenza Elo assoluta minima: {float(rule['min_abs_elo']):.0f}",
        f"- Esiti ammessi: {rule['picks']}",
        f"- Esperienza minima richiesta: {bool(rule['require_experience'])}",
        f"- Regole valutate esclusivamente nel discovery: {rule['rules_tested']}",
        "",
        "## Risultati",
        "",
        "| Periodo | Puntate | ROI | IC 95% | Delta Log Loss | Delta Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['period']} | {int(row['bets'])} | {row['roi']:.2%} | "
            f"[{row['ci_low']:.2%}, {row['ci_high']:.2%}] | "
            f"{row['log_loss_delta']:+.4f} | {row['brier_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gate di promozione",
            "",
            f"- ROI holdout con limite inferiore IC 95% > 0: {holdout['ci_low'] > 0}",
            "- Modello migliore del mercato su Log Loss o Brier: "
            f"{holdout['log_loss_delta'] < 0 or holdout['brier_delta'] < 0}",
            "- ROI positivo in ogni stagione holdout: "
            f"{not stability.empty and stability['roi'].gt(0).all()}",
            "",
            "## Garanzie",
            "",
            "- La griglia e la regola sono selezionate senza osservare l’holdout.",
            "- La regola congelata viene valutata una sola volta sull’holdout.",
            "- Una sola puntata argmax per partita, stake fisso.",
            "- Quote Market Average closing; nessuna scelta ex post del bookmaker.",
            "- Il closing è un benchmark di ricerca, non garantisce eseguibilità live.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {
        "rule": rule_path,
        "summary": summary_path,
        "stability": stability_path,
        "candidates": candidates_path,
        "report": report_path,
    }
    return EdgeDiscoveryResult(rule, summary, stability, candidates, outputs, promoted)


def _season_key(value: object) -> int:
    season = str(value)
    if re.fullmatch(r"\d{4}/\d{2}", season):
        return int(season[:4])
    if re.fullmatch(r"\d{4}", season):
        first = int(season[:2])
        return (1900 if first >= 50 else 2000) + first
    raise ValueError(f"Formato stagione ambiguo: {season}")
