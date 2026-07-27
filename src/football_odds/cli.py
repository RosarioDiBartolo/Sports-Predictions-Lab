from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_LEAGUES,
    DEFAULT_SEASONS,
    BackfillConfig,
    ModelingConfig,
)
from .confirmed_lineup import export_confirmed_lineup_model
from .database import ResearchDatabase
from .enrichment import run_environment_enrichment
from .pipeline import (
    run_backfill_pipeline,
    run_edge_discovery_pipeline,
    run_fixture_prediction,
    run_hybrid_model_pipeline,
    run_sport_model_pipeline,
    run_unified_pipeline,
)
from .player_coverage import export_api_football_coverage
from .player_ingestion import (
    backfill_api_football_lineups,
    import_api_football_lineups,
)
from .player_reconciliation import LEAGUE_CODES, reconcile_api_football
from .seriea_feed import DEFAULT_SEASONS as SERIEA_FEED_SEASONS
from .seriea_feed import backfill_seriea_feed, export_seriea_feed_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analizza quote storiche 1-X-2.")
    subparsers = parser.add_subparsers(dest="command", required=True)
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
    lineup_model = subparsers.add_parser(
        "confirmed-lineup-model",
        help="Valuta Dixon-Coles + correzione regolarizzata della lineup ufficiale.",
    )
    lineup_model.add_argument(
        "--features",
        type=Path,
        help="CSV leakage-safe con feature *_confirmed_lineup_*.",
    )
    lineup_model.add_argument("--alpha", type=float, default=25.0)
    lineup_model.add_argument("--project-dir", type=Path, default=Path.cwd())
    edge = subparsers.add_parser(
        "edge-discovery",
        help="Cerca una regola sul discovery e la valida su un holdout bloccato.",
    )
    edge.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    edge.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    edge.add_argument("--windows", nargs="+", type=int, default=[5, 10])
    edge.add_argument("--project-dir", type=Path, default=Path.cwd())
    coverage = subparsers.add_parser(
        "player-coverage",
        help="Misura la copertura lineup senza modificare il modello.",
    )
    coverage.add_argument(
        "--seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    coverage.add_argument("--sample-per-season", type=int, default=1)
    coverage.add_argument("--project-dir", type=Path, default=Path.cwd())
    player_import = subparsers.add_parser(
        "player-import",
        help="Importa lineup storiche API-Football per fixture già mappate.",
    )
    player_import.add_argument("--fixtures", nargs="+", required=True)
    player_import.add_argument("--database", type=Path)
    player_import.add_argument("--project-dir", type=Path, default=Path.cwd())
    player_backfill = subparsers.add_parser(
        "player-backfill",
        help="Importa in batch le lineup mancanti per fixture già mappate.",
    )
    player_backfill.add_argument("--league", default="I1")
    player_backfill.add_argument(
        "--seasons", nargs="+", default=["2223", "2324", "2425"]
    )
    player_backfill.add_argument("--limit", type=int)
    player_backfill.add_argument("--database", type=Path)
    player_backfill.add_argument("--project-dir", type=Path, default=Path.cwd())
    seriea_audit = subparsers.add_parser(
        "seriea-feed-audit",
        help="Verifica lineup e sostituzioni dal feed pubblico Lega Serie A.",
    )
    seriea_audit.add_argument(
        "--seasons", nargs="+", default=list(SERIEA_FEED_SEASONS)
    )
    seriea_audit.add_argument("--sample-matches", type=int, default=10)
    seriea_audit.add_argument("--project-dir", type=Path, default=Path.cwd())
    seriea_backfill = subparsers.add_parser(
        "seriea-feed-backfill",
        help="Importa lineup, sostituzioni e minuti dal feed pubblico Lega.",
    )
    seriea_backfill.add_argument(
        "--seasons", nargs="+", default=list(SERIEA_FEED_SEASONS)
    )
    seriea_backfill.add_argument("--limit", type=int)
    seriea_backfill.add_argument("--project-dir", type=Path, default=Path.cwd())
    reconcile = subparsers.add_parser(
        "player-reconcile",
        help="Riconcilia fixture API-Football con i match canonici.",
    )
    reconcile.add_argument(
        "--leagues", nargs="+", type=int, default=list(LEAGUE_CODES)
    )
    reconcile.add_argument(
        "--seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    reconcile.add_argument("--database", type=Path)
    reconcile.add_argument("--project-dir", type=Path, default=Path.cwd())
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
            "model",
            "hybrid",
            "edge",
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
        return
    if args.command == "confirmed-lineup-model":
        project_dir = args.project_dir.resolve()
        feature_path = args.features or (
            project_dir / "data" / "processed" / "confirmed_lineup_features.csv"
        )
        if not feature_path.is_absolute():
            feature_path = project_dir / feature_path
        result = export_confirmed_lineup_model(
            pd.read_csv(feature_path),
            project_dir / "reports" / "modeling" / "confirmed_lineup_model",
            alpha=args.alpha,
        )
        metadata = json.loads(
            result.outputs["metadata"].read_text(encoding="utf-8")
        )
        print(
            json.dumps(
                {
                    "oos_predictions": len(result.predictions),
                    "promoted": metadata["promotion"]["promoted"],
                    "official_model_unchanged": True,
                    "report": str(result.outputs["report"]),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
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
                    "official_model": metadata["official_model"],
                    "evaluation_gate_passed": metadata["promotion"]["promoted"],
                    "report": str(result.outputs["report"]),
                    "retired_official_model": metadata["retired_official_model"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "edge-discovery":
        result = run_edge_discovery_pipeline(
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
                    "promoted": result.promoted,
                    "selected_rule": result.selected_rule,
                    "report": str(result.outputs["report"]),
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
    if args.command == "player-coverage":
        result = export_api_football_coverage(
            args.project_dir.resolve(),
            seasons=tuple(args.seasons),
            sample_per_season=args.sample_per_season,
        )
        print(
            json.dumps(
                {
                    "requests_made": result.requests_made,
                    "sampled_fixtures": len(result.samples),
                    "report": str(result.outputs["report"]),
                    "modeling_data_changed": False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "player-import":
        project_dir = args.project_dir.resolve()
        result = import_api_football_lineups(
            project_dir,
            fixture_ids=tuple(args.fixtures),
            database_path=args.database.resolve() if args.database else None,
        )
        print(
            json.dumps(
                {
                    "fixtures": result.fixtures,
                    "lineups": result.lineups,
                    "players": result.players,
                    "requests_made": result.requests_made,
                    "modeling_features_changed": False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "player-backfill":
        project_dir = args.project_dir.resolve()
        result = backfill_api_football_lineups(
            project_dir,
            league=args.league,
            seasons=tuple(args.seasons),
            limit=args.limit,
            database_path=args.database.resolve() if args.database else None,
        )
        print(
            json.dumps(
                {
                    "eligible_fixtures": result.eligible_fixtures,
                    "already_complete": result.already_complete,
                    "attempted_fixtures": result.attempted_fixtures,
                    "imported_fixtures": result.imported_fixtures,
                    "lineups": result.lineups,
                    "players": result.players,
                    "failures": list(result.failures),
                    "requests_made": result.requests_made,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "seriea-feed-audit":
        result = export_seriea_feed_audit(
            args.project_dir.resolve(),
            seasons=tuple(args.seasons),
            sample_matches=args.sample_matches,
        )
        print(
            json.dumps(
                {
                    "sampled_matches": len(result.matches),
                    "player_rows": len(result.players),
                    "requests_made": result.requests_made,
                    "report": str(result.outputs["report"]),
                    "modeling_data_changed": False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "seriea-feed-backfill":
        result = backfill_seriea_feed(
            args.project_dir.resolve(),
            seasons=tuple(args.seasons),
            limit=args.limit,
        )
        print(
            json.dumps(
                {
                    "feed_matches": result.feed_matches,
                    "mapped_matches": result.mapped_matches,
                    "already_complete": result.already_complete,
                    "imported_matches": result.imported_matches,
                    "unresolved": list(result.unresolved),
                    "requests_made": result.requests_made,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "player-reconcile":
        project_dir = args.project_dir.resolve()
        result = reconcile_api_football(
            project_dir,
            leagues=tuple(args.leagues),
            seasons=tuple(args.seasons),
            database_path=args.database.resolve() if args.database else None,
        )
        print(
            json.dumps(
                {
                    "fixtures_seen": result.fixtures_seen,
                    "fixtures_mapped": result.fixtures_mapped,
                    "already_mapped": result.already_mapped,
                    "unresolved": len(result.unresolved),
                    "unresolved_by_reason": {
                        reason: sum(
                            row["reason"] == reason for row in result.unresolved
                        )
                        for reason in sorted(
                            {str(row["reason"]) for row in result.unresolved}
                        )
                    },
                    "requests_made": result.requests_made,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
if __name__ == "__main__":
    main()
