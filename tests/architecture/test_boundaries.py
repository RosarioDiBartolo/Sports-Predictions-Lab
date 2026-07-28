"""Executable architecture constraints."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path("src/football_odds")
DOMAINS = {
    "core",
    "data",
    "ingestion",
    "players",
    "modeling",
    "market",
    "strategies",
    "enrichment",
    "cli",
}


def test_package_root_contains_no_business_modules() -> None:
    assert {path.name for path in PACKAGE.glob("*.py")} == {"__init__.py"}


def test_no_legacy_runtime_names_or_cross_domain_private_imports() -> None:
    forbidden = {"legacy", "sport_model", "hybrid_model", "confirmed_lineup_model"}
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(token in path.stem for token in forbidden)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_domain = node.module.split(".")[0]
                owner = path.relative_to(PACKAGE).parts[0]
                if imported_domain in DOMAINS and imported_domain != owner:
                    assert all(not alias.name.startswith("_") for alias in node.names)


def test_every_domain_has_local_instructions() -> None:
    for domain in DOMAINS:
        assert (PACKAGE / domain / "AGENTS.md").is_file()
