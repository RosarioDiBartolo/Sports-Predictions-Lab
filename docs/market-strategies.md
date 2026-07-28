# Market and strategies

Bookmaker odds are both preserved source data and the decisive benchmark for
the official neural model.

The system retains:

- raw odds snapshots and provenance;
- opening, closing and timestamped snapshots;
- calibration and overround analysis;
- model-versus-market comparisons;
- strategy search on discovery periods;
- frozen-rule evaluation on untouched holdouts;
- reproducible backtests and settlement.

Every strategy selection must resolve to a model version, dataset version,
prediction cutoff, configuration, bookmaker and odds snapshot actually
available at that cutoff.

Strategy promotion requires out-of-sample probabilistic evidence and economic
evidence. A positive average ROI alone is insufficient; uncertainty and
season-by-season stability remain visible.
