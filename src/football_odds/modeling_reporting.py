from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def modeling_diagnostics(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create interpretable checks for Elo, rest and recent-form patterns."""
    data = features.copy()
    data["home_score"] = np.select(
        [data["result"].eq("H"), data["result"].eq("D")],
        [1.0, 0.5],
        default=0.0,
    )
    data["elo_bin"] = pd.cut(
        data["elo_difference"],
        bins=[-np.inf, -150, -75, -25, 25, 75, 150, np.inf],
    )
    elo = (
        data.groupby("elo_bin", observed=True)
        .agg(
            matches=("result", "size"),
            expected_home=("elo_expected_home", "mean"),
            actual_home_score=("home_score", "mean"),
            home_win_rate=("result", lambda values: values.eq("H").mean()),
            draw_rate=("result", lambda values: values.eq("D").mean()),
        )
        .reset_index()
    )
    elo["calibration_error"] = elo["actual_home_score"] - elo["expected_home"]

    data["rest_difference"] = data["home_rest_days"] - data["away_rest_days"]
    data["rest_bin"] = pd.cut(
        data["rest_difference"],
        bins=[-np.inf, -4, -1, 1, 4, np.inf],
        labels=["home <=-5", "home -4/-2", "similar", "home +2/+4", "home >=+5"],
    )
    rest = (
        data.dropna(subset=["rest_difference"])
        .groupby("rest_bin", observed=True)
        .agg(
            matches=("result", "size"),
            home_win_rate=("result", lambda values: values.eq("H").mean()),
            home_score=("home_score", "mean"),
        )
        .reset_index()
    )

    if {"home_points_5", "away_points_5"}.issubset(data.columns):
        data["form_difference"] = data["home_points_5"] - data["away_points_5"]
        data["form_bin"] = pd.qcut(data["form_difference"], q=5, duplicates="drop")
        form = (
            data.dropna(subset=["form_difference"])
            .groupby("form_bin", observed=True)
            .agg(
                matches=("result", "size"),
                form_difference=("form_difference", "mean"),
                home_win_rate=("result", lambda values: values.eq("H").mean()),
                home_score=("home_score", "mean"),
            )
            .reset_index()
        )
    else:
        form = pd.DataFrame()

    league = (
        data.groupby("league")
        .agg(
            matches=("result", "size"),
            home_win_rate=("result", lambda values: values.eq("H").mean()),
            draw_rate=("result", lambda values: values.eq("D").mean()),
            elo_brier=(
                "elo_expected_home",
                lambda values: float(
                    np.mean((values - data.loc[values.index, "home_score"]) ** 2)
                ),
            ),
        )
        .reset_index()
    )
    completeness = data.isna().mean().rename("missing_rate").sort_values().reset_index()
    completeness.columns = ["feature", "missing_rate"]
    return {
        "elo_patterns": elo,
        "rest_patterns": rest,
        "form_patterns": form,
        "league_diagnostics": league,
        "feature_completeness": completeness,
    }


def export_modeling_report(
    features: pd.DataFrame, destination: Path
) -> dict[str, Path]:
    """Export diagnostic tables, charts and a concise Markdown report."""
    destination.mkdir(parents=True, exist_ok=True)
    diagnostics = modeling_diagnostics(features)
    outputs: dict[str, Path] = {}
    for name, table in diagnostics.items():
        path = destination / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = path

    elo = diagnostics["elo_patterns"]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        elo["expected_home"],
        elo["actual_home_score"],
        marker="o",
        label="Elo",
    )
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfetto")
    axis.set(
        title="Calibrazione Elo pre-partita",
        xlabel="Punteggio casa atteso",
        ylabel="Punteggio casa reale",
    )
    axis.legend()
    figure.tight_layout()
    chart = destination / "elo_calibration.png"
    figure.savefig(chart, dpi=160)
    plt.close(figure)
    outputs["elo_chart"] = chart

    report = destination / "MODELING_REPORT.md"
    report.write_text(_report_text(features, diagnostics), encoding="utf-8")
    outputs["report"] = report
    return outputs


def _report_text(features: pd.DataFrame, diagnostics: dict[str, pd.DataFrame]) -> str:
    complete_5 = (
        features["home_matches_played"].ge(5) & features["away_matches_played"].ge(5)
    ).mean()
    league = diagnostics["league_diagnostics"].sort_values("elo_brier")
    best = league.iloc[0]
    return (
        "# Modeling dataset report\n\n"
        f"- Partite: {len(features):,}\n"
        f"- Campionati: {features['league'].nunique()}\n"
        f"- Stagioni: {features['season'].nunique()}\n"
        f"- Entrambe le squadre con almeno 5 precedenti: {complete_5:.2%}\n"
        f"- Migliore Elo Brier: {best['league']} ({best['elo_brier']:.4f})\n\n"
        "Tutte le feature sono state lette prima dell'aggiornamento con il "
        "risultato della partita corrente. I valori mancanti iniziali sono "
        "strutturali: rappresentano squadre senza storico sufficiente.\n"
    )
