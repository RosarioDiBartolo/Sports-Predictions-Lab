"""Canonical analytics dataset and research reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from ..core.config import ODDS_RANGE_EDGES, AnalysisConfig
from ..data.repository import ResearchDatabase

matplotlib.use("Agg")

ANALYTICS_QUERY = """
SELECT
    m.match_id,
    b.bookmaker_name AS bookmaker,
    p.provider_name AS provider,
    o.market,
    o.selection,
    o.decimal_odds AS odds,
    o.implied_probability,
    r.result,
    o.margin,
    o.timestamp,
    o.opening_or_closing,
    m.season,
    l.league_name AS league,
    ht.team_name AS home_team,
    at.team_name AS away_team,
    r.home_goals,
    r.away_goals
FROM odds o
JOIN matches m ON m.match_id = o.match_id
JOIN bookmakers b ON b.bookmaker_id = o.bookmaker_id
JOIN providers p ON p.provider_id = o.provider_id
JOIN leagues l ON l.league_id = m.league_id
JOIN teams ht ON ht.team_id = m.home_team_id
JOIN teams at ON at.team_id = m.away_team_id
LEFT JOIN match_results r ON r.match_id = m.match_id
"""


def build_analytics_dataset(
    database: ResearchDatabase | Path | str,
    *,
    bin_width: float = 0.05,
) -> pd.DataFrame:
    """Build one analysis row per bookmaker selection and snapshot."""
    repository = (
        database
        if isinstance(database, ResearchDatabase)
        else ResearchDatabase(database)
    )
    repository.initialize()
    with repository.connect() as connection:
        frame = pd.read_sql_query(ANALYTICS_QUERY, connection)
    fixed = frame["opening_or_closing"].isin(["opening", "closing"])
    canonical = frame.loc[fixed].drop_duplicates(
        [
            "match_id",
            "bookmaker",
            "market",
            "selection",
            "opening_or_closing",
        ],
        keep="last",
    )
    frame = pd.concat([canonical, frame.loc[~fixed]], ignore_index=True)
    if frame.empty:
        return frame.assign(
            prediction_correct=pd.Series(dtype=bool),
            favorite=pd.Series(dtype=bool),
            favorite_won=pd.Series(dtype=bool),
            calibration_bin=pd.Series(dtype=str),
            logloss_contribution=pd.Series(dtype=float),
            brier_contribution=pd.Series(dtype=float),
            roi=pd.Series(dtype=float),
        )

    frame["prediction_correct"] = frame["selection"] == frame["result"]
    snapshot_keys = [
        "match_id",
        "bookmaker",
        "market",
        "timestamp",
        "opening_or_closing",
    ]
    maximum = frame.groupby(snapshot_keys, dropna=False)[
        "implied_probability"
    ].transform("max")
    frame["favorite"] = frame["implied_probability"].eq(maximum)
    frame["favorite_won"] = frame["favorite"] & frame["prediction_correct"]
    edges = np.append(np.arange(0, 1, bin_width), 1.0)
    frame["calibration_bin"] = pd.cut(
        frame["implied_probability"],
        bins=np.unique(edges),
        include_lowest=True,
    ).astype(str)
    actual = frame["prediction_correct"].astype(float)
    probability = frame["implied_probability"].clip(1e-15, 1 - 1e-15)
    true_probability = probability.where(frame["prediction_correct"])
    frame["logloss_contribution"] = -np.log(
        true_probability.groupby(
            [frame[key] for key in snapshot_keys], dropna=False
        ).transform("max")
    )
    squared_error = (probability - actual) ** 2
    frame["brier_contribution"] = squared_error.groupby(
        [frame[key] for key in snapshot_keys], dropna=False
    ).transform("sum")
    frame["roi"] = np.where(frame["prediction_correct"], frame["odds"] - 1.0, -1.0)
    return frame


def save_analytics_dataset(
    config: AnalysisConfig,
    database: ResearchDatabase | None = None,
) -> pd.DataFrame:
    """Build and persist the replaceable analytics layer."""
    frame = build_analytics_dataset(
        database or ResearchDatabase(config.database_path),
        bin_width=config.bin_width,
    )
    config.ensure_directories()
    frame.to_csv(config.processed_dir / "analytics_dataset.csv", index=False)
    return frame


def analyze_predictions(frame: pd.DataFrame) -> dict[str, float | int]:
    """Calculate binary selection-level research metrics."""
    if frame.empty:
        raise ValueError("Il dataset analitico è vuoto.")
    actual = frame["prediction_correct"].astype(float)
    probability = frame["implied_probability"]
    calibration = (
        frame.groupby("calibration_bin", observed=True)
        .agg(
            observations=("prediction_correct", "size"),
            predicted=("implied_probability", "mean"),
            actual=("prediction_correct", "mean"),
        )
        .reset_index()
    )
    absolute_error = (calibration["predicted"] - calibration["actual"]).abs()
    ece = np.average(absolute_error, weights=calibration["observations"])
    favorite_rows = frame[frame["favorite"]]
    return {
        "predictions": len(frame),
        "accuracy": float(favorite_rows["favorite_won"].mean()),
        "log_loss": float(frame["logloss_contribution"].mean()),
        "brier_score": float(frame["brier_contribution"].mean()),
        "calibration_error": float(actual.mean() - probability.mean()),
        "expected_calibration_error": float(ece),
        "average_overround": float(frame["margin"].mean()),
        "sharpness": float(probability.var(ddof=0)),
    }


def compare_bookmakers(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank bookmakers on the common set of available predictions."""
    bookmakers = frame["bookmaker"].nunique()
    if bookmakers > 1:
        coverage = frame.groupby(
            ["match_id", "market", "selection", "opening_or_closing"]
        )["bookmaker"].nunique()
        shared = coverage[coverage == bookmakers].index
        indexed = frame.set_index(
            ["match_id", "market", "selection", "opening_or_closing"]
        )
        common = indexed[indexed.index.isin(shared)].reset_index()
        if not common.empty:
            frame = common
    rows = []
    for bookmaker, group in frame.groupby("bookmaker", sort=True):
        rows.append({"bookmaker": bookmaker, **analyze_predictions(group)})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["rank"] = result["log_loss"].rank(method="min").astype(int)
    return result.sort_values(["rank", "bookmaker"]).reset_index(drop=True)


def analyze_odds_ranges(
    frame: pd.DataFrame,
    edges: tuple[float, ...] = ODDS_RANGE_EDGES,
) -> pd.DataFrame:
    """Aggregate calibration and flat-stake ROI by decimal-odds range."""
    data = frame.copy()
    labels = [
        f"{left:.2f}-{right:.2f}" if np.isfinite(right) else f"{left:.2f}+"
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    ]
    data["odds_range"] = pd.cut(
        data["odds"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    result = (
        data.groupby("odds_range", observed=True)
        .agg(
            predictions=("prediction_correct", "size"),
            implied_probability=("implied_probability", "mean"),
            actual_frequency=("prediction_correct", "mean"),
            roi=("roi", "mean"),
        )
        .reset_index()
    )
    result["calibration_error"] = (
        result["actual_frequency"] - result["implied_probability"]
    )
    return result


def compare_leagues(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the same metrics for every available league."""
    return pd.DataFrame(
        [
            {"league": league, **analyze_predictions(group)}
            for league, group in frame.groupby("league", sort=True)
        ]
    )


def compare_opening_closing(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare timing metrics and average probability movement."""
    rows = []
    key = ["match_id", "bookmaker", "market", "selection"]
    pivot = frame.pivot_table(
        index=key,
        columns="opening_or_closing",
        values="implied_probability",
        aggfunc="last",
    )
    movement = (
        (pivot["closing"] - pivot["opening"]).abs().mean()
        if {"opening", "closing"}.issubset(pivot.columns)
        else float("nan")
    )
    for timing, group in frame[
        frame["opening_or_closing"].isin(["opening", "closing"])
    ].groupby("opening_or_closing"):
        rows.append(
            {
                "timing": timing,
                **analyze_predictions(group),
                "mean_absolute_probability_movement": float(movement),
            }
        )
    return pd.DataFrame(rows)


def _save_histogram(
    values: pd.Series, title: str, xlabel: str, destination: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values.dropna(), bins=20, edgecolor="white")
    axis.set(title=title, xlabel=xlabel, ylabel="Osservazioni")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def export_research_report(frame: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Export tables, static figures and a portable interactive dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "bookmakers": compare_bookmakers(frame),
        "leagues": compare_leagues(frame),
        "opening_closing": compare_opening_closing(frame),
        "odds_ranges": analyze_odds_ranges(frame),
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path

    charts = {
        "probability_histogram.png": (
            frame["implied_probability"],
            "Distribuzione probabilità",
            "Probabilità implicita",
        ),
        "odds_distribution.png": (
            frame["odds"].clip(upper=10),
            "Distribuzione quote (massimo visualizzato: 10)",
            "Quota decimale",
        ),
        "margin_distribution.png": (
            frame["margin"],
            "Distribuzione margine bookmaker",
            "Margine",
        ),
    }
    for filename, (values, title, xlabel) in charts.items():
        destination = output_dir / filename
        _save_histogram(values, title, xlabel, destination)
        paths[filename] = destination

    table_guides = {
        "bookmakers": (
            "Confronto tra bookmaker",
            "Ogni riga riassume un bookmaker. Serve a confrontare quante quote "
            "ha fornito, quanto margine incorpora nelle quote e come si sono comportate "
            "le sue previsioni. Più il calibration error è vicino a zero, più la probabilità "
            "media dichiarata assomiglia alla frequenza media osservata.",
        ),
        "leagues": (
            "Confronto tra campionati",
            "Ogni riga rappresenta un campionato. La tabella aiuta a capire se le quote "
            "sono più o meno precise in competizioni diverse.",
        ),
        "opening_closing": (
            "Quote iniziali e finali",
            "Le quote iniziali sono pubblicate prima; quelle finali sono le ultime prima "
            "della partita. Confrontarle mostra come il mercato cambia idea nel tempo.",
        ),
        "odds_ranges": (
            "Risultati per fascia di quota",
            "Le partite sono raccolte in gruppi di quote simili. Così possiamo vedere "
            "se le favorite, le partite equilibrate o le sorprese hanno prodotto risultati diversi.",
        ),
    }
    sections = "\n".join(
        (
            '<section class="table-card">'
            f"<h2>{table_guides[name][0]}</h2>"
            f'<p class="explanation">{table_guides[name][1]}</p>'
            '<div class="table-scroll">'
            f"{table.to_html(index=False)}"
            "</div></section>"
        )
        for name, table in tables.items()
    )
    figures = [
        px.scatter(
            frame.groupby("calibration_bin", observed=True)
            .agg(
                predicted=("implied_probability", "mean"),
                actual=("prediction_correct", "mean"),
                observations=("prediction_correct", "size"),
            )
            .reset_index(),
            x="predicted",
            y="actual",
            size="observations",
            hover_name="calibration_bin",
            labels={
                "predicted": "Probabilità prevista",
                "actual": "Frequenza reale",
                "observations": "Numero di esempi",
            },
            title="Le previsioni mantengono le promesse?",
        ),
        px.histogram(
            frame,
            x="implied_probability",
            color="bookmaker",
            barmode="overlay",
            labels={
                "implied_probability": "Probabilità indicata dalla quota",
                "count": "Numero di pronostici",
                "bookmaker": "Bookmaker",
            },
            title="Quante volte compare ogni probabilità?",
        ),
        px.box(
            frame,
            x="bookmaker",
            y="margin",
            color="opening_or_closing",
            labels={
                "bookmaker": "Bookmaker",
                "margin": "Margine del bookmaker",
                "opening_or_closing": "Momento della quota",
            },
            title="Quanto margine trattiene ogni bookmaker?",
        ),
        px.bar(
            tables["odds_ranges"],
            x="odds_range",
            y="roi",
            labels={"odds_range": "Fascia di quota", "roi": "ROI"},
            title="Guadagno o perdita per fascia di quota",
        ),
    ]
    chart_guides = [
        {
            "title": "1. Le previsioni mantengono le promesse?",
            "what": (
                "Immagina che un bookmaker dica: «Questo risultato ha il 70% di possibilità». "
                "Se lo dice 10 volte, dovrebbe accadere circa 7 volte. Ogni pallina confronta "
                "la promessa del bookmaker con ciò che è successo davvero. In questo grafico "
                "tutti i bookmaker e tutti i momenti disponibili sono messi insieme."
            ),
            "read": (
                "Vai da sinistra a destra per leggere la probabilità promessa e dal basso "
                "verso l’alto per leggere la frequenza reale. Una pallina grande contiene più "
                "partite ed è quindi, in genere, più affidabile di una piccola."
            ),
            "meaning": (
                "Una pallina vicino alla diagonale ideale indica una previsione ben calibrata. "
                "Sopra la diagonale il risultato è accaduto più spesso del previsto; sotto è "
                "accaduto meno spesso. La distanza verticale dalla diagonale è il calibration "
                "error di quel gruppo. Non significa automaticamente che una scommessa sia conveniente."
            ),
            "example": (
                "Esempio: una pallina in posizione 0,60 sull’asse orizzontale e 0,50 su quello "
                "verticale significa: «promesso 60%, successo davvero 50 volte ogni 100». "
                "L’errore di calibrazione del gruppo è -0,10, cioè -10 punti percentuali."
            ),
        },
        {
            "title": "2. Quali probabilità usa più spesso ogni bookmaker?",
            "what": (
                "È come mettere tanti pronostici in scatole: nella scatola del 20% finiscono "
                "le possibilità vicine al 20%, in quella del 50% quelle vicine al 50%, e così via."
            ),
            "read": (
                "Più una colonna è alta, più pronostici appartengono a quella zona. I colori "
                "rappresentano bookmaker diversi e possono sovrapporsi. Passa il mouse su una "
                "colonna per vedere i numeri esatti."
            ),
            "meaning": (
                "Il grafico descrive quanto spesso compaiono favorite, eventi equilibrati o "
                "risultati poco probabili. Non misura da solo la qualità delle previsioni e una "
                "colonna alta non vuol dire «scommessa migliore»."
            ),
            "example": (
                "Esempio: molte colonne alte vicino a 0,30 significano che nel campione ci sono "
                "molti risultati ai quali è stata assegnata circa una possibilità su tre."
            ),
        },
        {
            "title": "3. Quanto margine trattiene ogni bookmaker?",
            "what": (
                "Il margine è il vantaggio matematico incorporato dal bookmaker nell’intero mercato "
                "1-X-2, non una trattenuta applicata dopo la partita. Si trasformano le tre quote "
                "in probabilità grezze con 1 ÷ quota e si sommano: la parte che supera il 100% è il margine."
            ),
            "read": (
                "Ogni scatola raccoglie molti margini. La linea dentro la scatola è il valore centrale; "
                "la scatola mostra la zona più comune; i baffi e i puntini indicano valori più lontani. "
                "I colori distinguono quote iniziali e finali."
            ),
            "meaning": (
                "Confronta soprattutto l’altezza delle scatole: più sono in basso, più il margine è "
                "ridotto. Una scatola molto alta o lunga indica che il margine cambia parecchio tra "
                "le osservazioni."
            ),
            "example": (
                "Esempio: quote 2,00, 3,40 e 4,00 danno 50% + 29,4% + 25% = 104,4%. "
                "Il 4,4% oltre il 100% è il margine. Non è la probabilità di vincere e non è "
                "una commissione prelevata dopo la partita."
            ),
        },
        {
            "title": "4. In quali fasce di quota si è guadagnato o perso?",
            "what": (
                "Il ROI confronta il denaro ottenuto con quello puntato. Ogni colonna riunisce "
                "scommesse con quote simili, dalle favorite alle sorprese."
            ),
            "read": (
                "La linea dello zero separa guadagno e perdita. Colonne sopra zero indicano un "
                "guadagno storico; colonne sotto zero una perdita. Più la colonna è lontana da zero, "
                "più grande è stato il risultato."
            ),
            "meaning": (
                "Questo è un riassunto del passato, non una promessa per il futuro. Prima di fidarti "
                "di una colonna controlla anche quante osservazioni contiene: pochi casi possono "
                "produrre un risultato molto fortunato o sfortunato."
            ),
            "example": (
                "Esempio: ROI 0,08 significa +8%, cioè circa 108 euro restituiti partendo da 100 euro "
                "puntati complessivamente; ROI -0,08 significa una perdita di circa 8 euro."
            ),
        },
    ]
    interactive = "\n".join(
        (
            '<section class="chart-card">'
            f"<h2>{guide['title']}</h2>"
            '<div class="guide-grid">'
            f"<div><h3>Che cos’è?</h3><p>{guide['what']}</p></div>"
            f"<div><h3>Come si legge?</h3><p>{guide['read']}</p></div>"
            f"<div><h3>Cosa significa?</h3><p>{guide['meaning']}</p></div>"
            f"<div><h3>Un esempio facile</h3><p>{guide['example']}</p></div>"
            "</div>"
            + pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs="cdn" if index == 0 else False,
                config={
                    "displaylogo": False,
                    "toImageButtonOptions": {"format": "png"},
                },
            )
            + "</section>"
        )
        for index, (figure, guide) in enumerate(zip(figures, chart_guides, strict=True))
    )
    html = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Football Odds Research</title>
<style>
:root{{--ink:#172033;--muted:#526074;--paper:#f4f7fb;--card:#fff;--accent:#2457d6;--line:#dbe3ee}}
*{{box-sizing:border-box}} body{{font:16px/1.6 system-ui,sans-serif;color:var(--ink);
background:var(--paper);max-width:1240px;margin:auto;padding:2rem}}
h1{{font-size:clamp(2rem,5vw,3.4rem);line-height:1.05;margin-bottom:.6rem}} h2{{line-height:1.2}}
.intro,.chart-card,.table-card{{background:var(--card);border:1px solid var(--line);
border-radius:18px;padding:clamp(1rem,3vw,2rem);margin:1.25rem 0;box-shadow:0 8px 30px #1720330a}}
.intro{{border-left:7px solid var(--accent)}} .intro strong,h3{{color:var(--accent)}}
.guide-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;margin:1rem 0}}
.guide-grid>div{{background:#f7f9fd;border-radius:12px;padding:.8rem 1rem}}
.guide-grid h3{{font-size:1rem;margin:0 0 .25rem}} .guide-grid p{{margin:0}}
.explanation{{color:var(--muted);max-width:80ch}} .table-scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{padding:.55rem;border:1px solid var(--line)}}
th{{background:#eef3fb;text-align:left}} @media(max-width:700px){{body{{padding:.75rem}}.guide-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Capire le quote, un passo alla volta</h1>
<p>Questa pagina racconta che cosa mostrano i dati delle quote calcistiche.</p></header>
<section class="intro">
<h2>Prima di iniziare</h2>
<p><strong>Una quota non è una previsione certa.</strong> È un numero che possiamo trasformare
in una probabilità. Per esempio, quota 2,00 corrisponde all’incirca al 50% prima di considerare
il margine del bookmaker.</p>
<p>Puoi passare il mouse sui grafici per vedere i valori, trascinare per ingrandire una zona,
fare doppio clic per tornare indietro e usare l’icona della macchina fotografica per salvare
uno screenshot PNG. I risultati descrivono questo insieme di dati e non garantiscono ciò che
succederà nelle prossime partite.</p>
</section>
<section class="intro">
<h2>Due parole importanti</h2>
<p><strong>Calibration error:</strong> misura la differenza tra la probabilità prevista e la
frequenza osservata. Se una previsione del 70% si verifica nel 60% dei casi, lo scarto è di
10 punti percentuali. Nelle tabelle, <em>calibration_error</em> conserva il segno: positivo
significa che il risultato è accaduto più spesso di quanto previsto; negativo significa che
è accaduto meno spesso. <em>Expected_calibration_error</em> considera invece la
distanza assoluta in ogni fascia e ne calcola la media pesata: zero è perfetto e più il numero
è basso, meglio è.</p>
<p><strong>Margine:</strong> per ogni mercato 1-X-2 calcoliamo
<code>(1/quota casa + 1/quota pareggio + 1/quota trasferta) − 1</code>. Per esempio, un risultato
di 0,05 equivale al 5%. È il vantaggio teorico incorporato nell’insieme delle quote, non una
percentuale tolta materialmente da ogni vincita.</p>
</section>
{interactive}
<section class="intro"><h2>Le tabelle, senza paura</h2>
<p>Le tabelle mostrano gli stessi dati in forma più precisa. Ogni riga è un gruppo e ogni
colonna è una misura. Se un valore manca, significa che non c’erano abbastanza informazioni
per calcolarlo: non equivale automaticamente a zero.</p></section>
{sections}</body></html>"""
    dashboard = output_dir / "dashboard.html"
    dashboard.write_text(html, encoding="utf-8")
    paths["dashboard"] = dashboard
    return paths
