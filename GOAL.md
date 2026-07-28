# Current project goal

Updated: 2026-07-28

## Goal

Complete the total domain refactor so Sports Predictions Lab has one clean,
scoped implementation per responsibility and only the neural player model is
operational.

## Status

In progress.

The domain layout, Markdown documentation, neural-only modeling direction and
83 functional tests are in place. Completion gates still fail on coverage,
typing and unfinished CLI workflows.

## Completion criteria

- `ruff` and `mypy` pass.
- The full `pytest` run passes with coverage at or above 90%.
- `model predict` and strategy commands execute real domain workflows rather
  than placeholders.
- Training performs the mandatory dataset preflight and produces observable,
  heartbeat-based run logs.
- Static architecture tests find no root business modules, compatibility
  wrappers, legacy model runtime or forbidden dependencies.
- The canonical database checksum and all historical data/artifacts remain
  unchanged by the code refactor.
- Production LOC is lower than before the refactor.

## Constraints

- Preserve raw/canonical data, provenance, licences, quarantine and historical
  evaluation evidence.
- No compatibility layer or obsolete runtime code.
- Bookmaker markets are the decisive model benchmark.
- Database v2 is the next separate phase; this goal does not migrate data.

## Next verification checkpoint

Implement dataset preflight and observable training-run logging, then restore
typing and coverage gates before adding another model or database feature.

## Discussion notes

- After this goal, the likely next goal is the conservative canonical database
  v2 migration described in `docs/data-platform.md`.

## Decision log

- 2026-07-28: Markdown replaced the docs application as authoritative
  documentation.
- 2026-07-28: `dixon_coles_shared_encoder_pooling` became the only operational
  model by explicit exception; bookmaker markets became the decisive benchmark.
- 2026-07-28: Code organization became domain-scoped, subtraction-first and
  incompatible with legacy Python/CLI paths by design.
- 2026-07-28: Dataset preflight and heartbeat-based training observability became
  mandatory before every training run.
