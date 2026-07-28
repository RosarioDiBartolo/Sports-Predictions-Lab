# Documentation contract

`docs/README.md` is the authoritative documentation index. The Markdown files
linked from it are the single source of truth for architecture, pipelines,
features, commands, artifacts, dependencies, guarantees and active decisions.

Update the relevant Markdown file in the same change whenever a CLI entry
point, pipeline stage, contract, feature, output path, temporal guarantee or
validation strategy changes. A change is incomplete until documentation links
are valid and the affected tests pass.

Reports under `reports/`, README prose, presentations and legacy snapshots are
evidence or historical material, not authoritative architecture.

# Decision continuity

Before planning or implementing work, read `GOAL.md`, then start from
`docs/README.md` and read the linked domain document and guarantees. Treat
documented active decisions,
modeling constraints, temporal guarantees, validation requirements, and
promotion gates as binding project directives across chats and sessions.

Do not silently replace, weaken, bypass, or reinterpret those directives.
If a request appears to conflict with them, identify the conflict explicitly
and ask for a deliberate decision. When the user deliberately changes a
directive, update the relevant authoritative Markdown file in the same change
before treating the new decision as authoritative.

Every predictive model or model variant must use the documented common
evaluation pipeline. A deliberate user decision may override a promotion gate
only when the exception and rationale are recorded in
`docs/modeling/models.md`; evaluation evidence remains mandatory and must not
be hidden.

# Goal continuity

`GOAL.md` is the single source of truth for the current project goal. Keep
exactly one current goal with measurable completion criteria, constraints,
status and the next verification checkpoint.

Conversation about future work is welcome and does not require the user to use
formal decision language. When discussion materially changes the intended
outcome, constraints, priority or completion criteria, update `GOAL.md` in the
same turn. Record tentative ideas under discussion notes; do not silently
replace the current goal with speculation. If two discussed goals are mutually
exclusive and intent is genuinely unclear, preserve the current goal, record
the candidate and ask for the decision.

After completing, blocking or superseding a goal, update its status and
decision log before starting another goal. Plans and implementation reports
must map their work to the current goal rather than creating an undocumented
parallel objective.

# Observable training

No model training may start until the dataset preflight defined in
`docs/modeling/models.md` passes and its immutable report has been written.
Training must emit the machine-readable heartbeat and human-readable log
defined there. An agent supervising training must inspect the preflight, verify
fresh heartbeats during execution and inspect the terminal status before
describing a run as healthy or complete.

Silent long-running training is forbidden. A stale heartbeat, non-finite loss,
schema drift, leakage failure or invalid dataset check stops the run and
requires diagnosis; it must not be ignored to save time.

# Scoped ownership

Every production module must belong to exactly one domain package. Business
logic in `src/football_odds/` itself is forbidden. Read the nearest `AGENTS.md`
before editing a domain.

Dependencies are one-way: `core` is dependency-free; `data` may depend on
`core`; `ingestion` and `players` may depend on `core` and `data`; `modeling`
may depend on player and market contracts; `strategies` consumes versioned
predictions and market snapshots. The CLI may orchestrate domains but may not
contain SQL, transformations, training or evaluation logic. Importing private
symbols from another domain is forbidden.

# Subtraction-first engineering

Before adding code, search for code that can be removed, merged or simplified.
For a refactor, removed production lines must outnumber added production lines.
A feature may increase production LOC only for a new, verified responsibility
and must delete the implementation it replaces in the same change.

Compatibility wrappers, legacy aliases, duplicate APIs, speculative
abstractions, placeholder infrastructure, single-implementation factories and
barrel exports are forbidden. Do not retain unreachable code "just in case".
Files above roughly 500 lines must be split before adding behavior.

Historical data and evidence are not garbage code. Raw assets, licences,
lineage, quarantine records, model artifacts and prior evaluation results remain
preserved even when their producing runtime code is removed.
