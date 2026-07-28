"""Single command surface for domain workflows."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odds-lab")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    commands.add_parser("ingest")
    players = commands.add_parser("players").add_subparsers(
        dest="action", required=True
    )
    collect = players.add_parser("collect")
    collect.add_argument("--request-budget", type=int, default=100)
    players.add_parser("reconcile")
    players.add_parser("dataset")
    players.add_parser("features")
    enrich = commands.add_parser("enrich").add_subparsers(dest="action", required=True)
    weather = enrich.add_parser("weather")
    weather.add_argument("--limit", type=int)
    market = commands.add_parser("market").add_subparsers(dest="action", required=True)
    market.add_parser("build")
    model = commands.add_parser("model").add_subparsers(dest="action", required=True)
    command = model.add_parser("train")
    command.add_argument("--epochs", type=int, default=80)
    command.add_argument("--embedding-dim", type=int, default=32)
    evaluate = model.add_parser("evaluate")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    predict = model.add_parser("predict")
    predict.add_argument("--fixtures", type=Path, required=True)
    strategy = commands.add_parser("strategy").add_subparsers(
        dest="action", required=True
    )
    strategy.add_parser("discover")
    strategy.add_parser("backtest")
    return parser


def _ingest(project: Path) -> None:
    from ..core.config import AnalysisConfig
    from ..data.repository import ResearchDatabase
    from ..ingestion.contracts import (
        FootballDataProvider,
        IngestionPipeline,
        load_all_seasons,
    )

    config = AnalysisConfig(project_dir=project)
    database = ResearchDatabase(config.database_path)
    IngestionPipeline(database).run(FootballDataProvider(load_all_seasons(config)))


def _players(project: Path, action: str, budget: int = 100) -> None:
    if action == "collect":
        from ..players.collector import collect_api_football_player_data

        collect_api_football_player_data(project, request_budget=budget)
    elif action == "reconcile":
        from ..players.reconciliation import reconcile_api_football

        reconcile_api_football(
            project, leagues=(39, 78, 135, 140, 61), seasons=(2022, 2023, 2024)
        )
    elif action == "dataset":
        from ..players.dataset import build_player_dataset

        build_player_dataset(project)
    else:
        from ..players.feature_store import build_player_feature_store

        build_player_feature_store(project)


def _market(project: Path) -> None:
    from ..core.config import AnalysisConfig
    from ..data.repository import ResearchDatabase
    from ..market.research import save_analytics_dataset

    config = AnalysisConfig(project_dir=project)
    save_analytics_dataset(
        config,
        ResearchDatabase(project / "data/football_odds.sqlite3"),
    )


def _model(project: Path, epochs: int, embedding_dim: int) -> None:
    from ..modeling.training import run_training

    run_training(project, epochs=epochs, embedding_dim=embedding_dim)


def main() -> None:
    args = build_parser().parse_args()
    project = args.project_dir.resolve()
    if args.command == "ingest":
        _ingest(project)
    elif args.command == "players":
        _players(project, args.action, getattr(args, "request_budget", 100))
    elif args.command == "enrich":
        from ..data.repository import ResearchDatabase
        from ..enrichment.environment import run_environment_enrichment

        run_environment_enrichment(
            ResearchDatabase(project / "data/football_odds.sqlite3"),
            weather_limit=args.limit,
        )
    elif args.command == "market":
        _market(project)
    elif args.command == "model" and args.action == "train":
        _model(project, args.epochs, args.embedding_dim)
    elif args.command == "model" and args.action == "evaluate":
        from ..modeling.frozen_evaluation import evaluate_frozen

        evaluate_frozen(args.predictions, args.reference, args.output_dir)
    elif args.command == "model":
        raise SystemExit(
            "Neural fixture prediction requires a versioned lineup tensor artifact."
        )
    elif args.command == "strategy":
        raise SystemExit(
            "Strategy commands require versioned neural predictions and cutoff odds."
        )
    elif args.command == "run":
        _ingest(project)
        _players(project, "dataset")
        _players(project, "features")
        _market(project)
        _model(project, 80, 32)


if __name__ == "__main__":
    main()
