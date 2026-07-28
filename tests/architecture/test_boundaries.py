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
ALLOWED_DEPENDENCIES = {
    "core": set(),
    "data": {"core"},
    "players": {"core", "data"},
    "ingestion": {"core", "data", "players"},
    "market": {"core", "data"},
    "modeling": {"core", "data", "players", "market"},
    "strategies": {"core", "modeling", "market"},
    "enrichment": {"core", "data"},
    "cli": DOMAINS - {"cli"},
}


def _relative_target(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level:
        owner_parts = list(path.relative_to(PACKAGE).parent.parts)
        prefix = owner_parts[: len(owner_parts) - node.level + 1]
        parts = prefix + (node.module or "").split(".")
    elif node.module and node.module.startswith("football_odds."):
        parts = node.module.split(".")[1:]
    else:
        return None
    return parts[0] if parts and parts[0] in DOMAINS else None


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


def test_domain_dependencies_follow_the_documented_direction() -> None:
    for path in PACKAGE.rglob("*.py"):
        owner = path.relative_to(PACKAGE).parts[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            imported_domain = _relative_target(path, node)
            if imported_domain is None or imported_domain == owner:
                continue
            assert imported_domain in ALLOWED_DEPENDENCIES[owner], (
                f"{path}:{node.lineno}: {owner} non può dipendere da {imported_domain}"
            )


def test_dependency_matrix_rejects_representative_forbidden_edges() -> None:
    forbidden = {
        ("core", "data"),
        ("data", "ingestion"),
        ("players", "ingestion"),
        ("modeling", "strategies"),
        ("strategies", "ingestion"),
    }
    assert all(target not in ALLOWED_DEPENDENCIES[owner] for owner, target in forbidden)


def test_cli_contains_only_orchestration() -> None:
    for path in (PACKAGE / "cli").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                assert not any(
                    module.split(".")[0] in {"numpy", "pandas", "sklearn", "torch"}
                    for module in modules
                )
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not any(
                    token in node.value.upper()
                    for token in ("SELECT ", "INSERT ", "UPDATE ", "DELETE ")
                )


def test_every_domain_has_local_instructions() -> None:
    for domain in DOMAINS:
        assert (PACKAGE / domain / "AGENTS.md").is_file()
