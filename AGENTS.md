# Project workflow

Read `qis_risk_dashboard_design.md`, including the approved implementation addendum and milestone ledger, before changing methodology.

- Keep one notebook and three substantive modules: data, model, report. Numerical defaults live in validated Python dataclasses.
- The notebook contains parameters and orchestration/display only. The renderer consumes calculated results; it never re-estimates risk.
- Preserve the defined synthetic excess-return holdings, timing, missing-data, and attribution conventions. Do not normalize invalid target weights or impute prices.
- Use parallel builders with one writer per file and an independent reviewer. The primary agent owns integration, documentation, packaging, and the notebook.
- Use `uv sync --all-groups --locked`, `uv run ruff check .`, and `uv run pytest`. The report tests execute the notebook in a fresh kernel.
- Update the milestone ledger with verification evidence and commit checkpoints. Resume from verified work after interruptions.
- Commit only code, clean notebooks, documentation, tests, dependency metadata, and the two simulated CSVs. Keep real data and generated reports out of Git.
- Delivery is complete only after numerical tests, notebook execution, visual verification, performance measurement, and a verified push to private `TylerWang1996/qis-dashboards` main.
