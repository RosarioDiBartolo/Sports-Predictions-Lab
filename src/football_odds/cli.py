from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_LEAGUES,
    DEFAULT_SEASONS,
    AnalysisConfig,
    BackfillConfig,
    ModelingConfig,
)
from .database import ResearchDatabase
from .enrichment import run_environment_enrichment
from .pipeline import (
    run_analysis,
    run_backfill_pipeline,
    run_baseline_pipeline,
    run_fixture_prediction,
    run_hybrid_model_pipeline,
    run_modeling_pipeline,
    run_research_pipeline,
    run_sport_model_pipeline,
    run_unified_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analizza quote storiche 1-X-2.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "Esegue l'analisi MVP originale."),
        ("research", "Esegue database, analytics e report di ricerca."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--league", default="I1")
        command.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
        command.add_argument("--bin-width", type=float, default=0.05)
        command.add_argument("--refresh", action="store_true")
        command.add_argument("--project-dir", type=Path, default=Path.cwd())
    modeling = subparsers.add_parser(
        "modeling", help="Genera feature Elo e rolling anti-leakage."
    )
    modeling.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    modeling.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    modeling.add_argument("--windows", nargs="+", type=int, default=[5, 10])
    modeling.add_argument("--refresh", action="store_true")
    modeling.add_argument("--project-dir", type=Path, default=Path.cwd())
    baseline = subparsers.add_parser(
        "baselines", help="Confronta baseline walk-forward fuori campione."
    )
    baseline.add_argument("--project-dir", type=Path, default=Path.cwd())
    sport_model = subparsers.add_parser(
        "sport-model",
        help="Addestra e valuta il modello predittivo sport-only.",
    )
    sport_model.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    sport_model.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    sport_model.add_argument("--windows", nargs="+", type=int, default=[5, 10])
    sport_model.add_argument("--project-dir", type=Path, default=Path.cwd())
    hybrid_model = subparsers.add_parser(
        "hybrid-model",
        help="Valuta Dixon-Coles + gradient boosting contro il modello ufficiale.",
    )
    hybrid_model.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    hybrid_model.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    hybrid_model.add_argument("--windows", nargs="+", type=int, default=[5, 10])
    hybrid_model.add_argument("--project-dir", type=Path, default=Path.cwd())
    predict = subparsers.add_parser(
        "predict",
        help="Predice fixture future senza risultato usando il modello sport-only.",
    )
    predict.add_argument("--fixtures", type=Path, required=True)
    predict.add_argument("--model-path", type=Path)
    predict.add_argument("--output", type=Path)
    predict.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    predict.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    predict.add_argument("--windows", nargs="+", type=int, default=[5, 10])
    predict.add_argument("--project-dir", type=Path, default=Path.cwd())
    backfill = subparsers.add_parser(
        "backfill", help="Importa più leghe e stagioni nel database canonico."
    )
    backfill.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    backfill.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    backfill.add_argument("--refresh", action="store_true")
    backfill.add_argument("--project-dir", type=Path, default=Path.cwd())
    enrich = subparsers.add_parser(
        "enrich-weather", help="Risolvi stadi e aggiungi meteo storico."
    )
    enrich.add_argument("--project-dir", type=Path, default=Path.cwd())
    enrich.add_argument("--skip-venues", action="store_true")
    enrich.add_argument("--limit", type=int)
    build = subparsers.add_parser(
        "build", help="Esegue il grafo canonico della pipeline per target."
    )
    build.add_argument(
        "targets",
        nargs="+",
        choices=(
            "all",
            "ingest",
            "analytics",
            "market",
            "features",
            "baselines",
            "hybrid",
            "model",
        ),
    )
    build.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    build.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    build.add_argument("--windows", nargs="+", type=int, default=[5, 10])
    build.add_argument("--refresh", action="store_true")
    build.add_argument("--rebuild-features", action="store_true")
    build.add_argument("--project-dir", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        result = run_unified_pipeline(
            ModelingConfig(
                leagues=tuple(args.leagues),
                seasons=tuple(args.seasons),
                rolling_windows=tuple(args.windows),
                project_dir=args.project_dir.resolve(),
            ),
            targets=tuple(args.targets),
            refresh=args.refresh,
            rebuild_features=args.rebuild_features,
        )
        print(
            json.dumps(
                {
                    "completed_stages": result.completed_stages,
                    "counts": result.counts,
                    "manifest": str(result.artifacts["manifest"]),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "backfill":
        result = run_backfill_pipeline(
            BackfillConfig(
                leagues=tuple(args.leagues),
                seasons=tuple(args.seasons),
                project_dir=args.project_dir.resolve(),
            ),
            refresh=args.refresh,
        )
        print(
            json.dumps(
                {
                    "matches_processed": result.ingestion.matches,
                    "odds_selections_processed": result.ingestion.odds_selections,
                    "analytics_rows": len(result.analytics),
                    "manifest": str(result.manifest),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "enrich-weather":
        database = ResearchDatabase(
            args.project_dir.resolve() / "data" / "football_odds.sqlite3"
        )
        summary = run_environment_enrichment(
            database,
            resolve_venues=not args.skip_venues,
            weather_limit=args.limit,
        )
        print(json.dumps(summary.__dict__, indent=2, ensure_ascii=False))
        return
    if args.command == "sport-model":
        result = run_sport_model_pipeline(
            ModelingConfig(
                leagues=tuple(args.leagues),
                seasons=tuple(args.seasons),
                rolling_windows=tuple(args.windows),
                project_dir=args.project_dir.resolve(),
            )
        )
        print(
            json.dumps(
                {
                    "oos_predictions": len(result.predictions),
                    "trained_seasons": (
                        list(result.predictor.trained_seasons)
                        if result.predictor
                        else []
                    ),
                    "model": (
                        str(result.outputs["model"])
                        if "model" in result.outputs
                        else None
                    ),
                    "report": str(result.outputs["report"]),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    if args.command == "hybrid-model":
        result = run_hybrid_model_pipeline(
            ModelingConfig(
                leagues=tuple(args.leagues),
                seasons=tuple(args.seasons),
                rolling_windows=tuple(args.windows),
                project_dir=args.project_dir.resolve(),
            )
        )
        metadata = json.loads(
            result.outputs["metadata"].read_text(encoding="utf-8")
        )
        print(
            json.dumps(
                {
                    "promoted": metadata["promotion"]["promoted"],
                    "report": str(result.outputs["report"]),
                    "official_model_unchanged": True,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "predict":
        project_dir = args.project_dir.resolve()
        result = run_fixture_prediction(
            args.fixtures.resolve(),
            ModelingConfig(
                leagues=tuple(args.leagues),
                seasons=tuple(args.seasons),
                rolling_windows=tuple(args.windows),
                project_dir=project_dir,
            ),
            model_path=args.model_path.resolve() if args.model_path else None,
            output_path=args.output.resolve() if args.output else None,
        )
        print(
            json.dumps(
                {
                    "fixtures": len(result.predictions),
                    "output": str(result.output),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "modeling":
        modeling_config = ModelingConfig(
            leagues=tuple(args.leagues),
            seasons=tuple(args.seasons),
            rolling_windows=tuple(args.windows),
            project_dir=args.project_dir.resolve(),
        )
        result = run_modeling_pipeline(modeling_config, refresh=args.refresh)
        print(
            json.dumps(
                {
                    "matches": len(result.features),
                    "features": len(result.features.columns),
                    "leagues": result.features["league"].nunique(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"\nReport: {result.outputs['report']}")
        return
    if args.command == "baselines":
        result = run_baseline_pipeline(
            ModelingConfig(project_dir=args.project_dir.resolve())
        )
        print(result.metrics.groupby("model")["log_loss"].mean().sort_values())
        print(f"\nReport: {result.outputs['report']}")
        return

    config = AnalysisConfig(
        league=args.league,
        seasons=tuple(args.seasons),
        bin_width=args.bin_width,
        project_dir=args.project_dir.resolve(),
    )
    if args.command == "run":
        result = run_analysis(config, refresh=args.refresh)
        print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
        print(f"\nOutput salvati in: {config.report_dir}")
    else:
        result = run_research_pipeline(config, refresh=args.refresh)
        print(
            json.dumps(
                {
                    "matches": result.ingestion.matches,
                    "odds_selections": result.ingestion.odds_selections,
                    "analytics_rows": len(result.analytics),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"\nDashboard: {result.outputs['dashboard']}")


if __name__ == "__main__":
    main()
