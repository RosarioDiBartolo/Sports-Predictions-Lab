# Sports Predictions Lab documentation

This directory is the authoritative project documentation.

The repository-level [current goal](../GOAL.md) is authoritative for current
priority, completion criteria and status.

Read the documents relevant to the task before planning or implementation:

- [Architecture](architecture.md): package boundaries, dependency direction and
  migration policy.
- [Data platform](data-platform.md): sources, canonical database, reconciliation,
  temporal guarantees, enrichment and governance.
- [Models](modeling/models.md): official model, baselines, evaluation and
  promotion decisions.
- [Market and strategies](market-strategies.md): bookmaker benchmark, odds,
  strategy discovery and backtesting.
- [Operations](operations.md): CLI, pipelines, artifacts and validation commands.

## Binding guarantees

- Raw assets, canonical data, provenance, licences, external identifiers,
  quarantine records and historical evaluation evidence are never discarded.
- Raw and canonical layers remain distinct.
- All predictive variants use the common out-of-sample evaluation pipeline.
- Temporal features are computed only from information available before the
  relevant cutoff.
- Bookmaker odds and their timestamps are preserved and remain available for
  strategy research and model evaluation.
- Legacy structures are migrated conservatively and are not physically removed
  until reconciliation evidence is accepted explicitly.
- User changes and changes from other tasks are not overwritten.

Reports in `reports/` are evidence produced by the pipelines. They do not
override decisions recorded here.
