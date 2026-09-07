# QIS Portfolio Risk Dashboard

A local notebook and offline HTML report for reviewing diversification and correlation risk in a long-only portfolio of excess-return indices. Calculations live in a small Python package, outside the notebook.

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups --locked
uv run jupyter lab qis_risk_dashboard.ipynb
```

Select the project Python kernel and run all cells. The default inputs are eight years of reproducible simulated data for ten strategies. The notebook displays the review and writes an offline report into `reports/`.

To run without opening JupyterLab:

```bash
uv run jupyter nbconvert --to notebook --execute --output qis_risk_dashboard.executed.ipynb --output-dir reports qis_risk_dashboard.ipynb
```

The source notebook stays clean. Open `reports/qis_risk_dashboard.html` in any browser; it contains all images and styles and needs no server or internet connection.

## Inputs

Both CSVs begin with a `date` column and use the same strategy identifiers. Index levels must be positive; target weights are decimal fractions (0.25 means 25%), complete, nonnegative, and sum to one. Change input paths and the explicit ordered universe in the notebook when using real data.

CSV rows declare the trading calendar. Retain missing observations as blank cells; omitted trading dates cannot be detected. An optional `calendar_complete_through` date certifies coverage through non-trading month/week boundaries. Rebalances occur at the close, and index units remain constant between rebalances.

For example, if the file ends on a Friday that is the month's final trading close, certify coverage through the calendar month-end only when that fact is known. Without that declaration, an unproven terminal month remains partial. Requests beyond certified coverage produce an unavailable report. Real-data live/backtested status and data vintage should be set explicitly; the library defaults to unknown.

The notebook exposes `ModelConfig` for numerical settings and `DisplayConfig` for chart history, pair count, and heatmap visibility. Display settings never change the estimation sample. Default publication requires 504 common daily returns and 444 lag-standardized vectors, so provide more than five years of input for a complete three-year chart window. Weekly diagnostics require 104 complete paired intervals; DR percentiles require 24 valid prior monthly snapshots.

The report includes a signed strategy risk-contribution waterfall, a correlation-clustered official matrix labeled with its configured trading-day half-life, and an expanded glossary of metrics and sensitivities. Clustering changes only the matrix's display order; every numerical result retains the model's original conventions. Contributions are ordered from largest to smallest, including negative and zero contributions, and sum to portfolio volatility.

The portfolio is a synthetic excess-return book, not funded NAV. Currency conversion occurs upstream; collateral yield, costs, financing adjustments, flows, allocation optimization, and forecast calibration are outside this version.

## Development

```bash
uv run ruff check .
uv run pytest
```

Read `qis_risk_dashboard_design.md` for methodology, acceptance checks, and build progress. Real inputs, generated HTML, executed notebook copies, and caches are ignored by Git. Only the explicitly simulated example CSVs are committed.

The implementation has 114 passing tests, including direct numerical references and a fresh-kernel notebook test that verifies exact HTML/display parity. The original independent review checked 200 random valid correlation/exposure cases against separately assembled covariance calculations. The presentation update additionally passed 60 independent matrix/portfolio cases, including 32 with negative strategy contributions.

On an AMD Ryzen 9 PRO 8945HS running Linux and Python 3.13.12, analysis of 5,000 dates × 10 strategies took **1.23 seconds** (median of three warmed runs); the complete demo notebook took **2.83 seconds** in a fresh kernel. The standalone demo HTML is approximately **325 KB**. Timings are observational, not CI thresholds.

For a local analysis timing check:

```bash
uv run python -c 'from time import perf_counter; from qis_risk.data import make_demo_data; from qis_risk.model import run_review; x=make_demo_data(periods=5000); t=perf_counter(); r=run_review(x); print(r.status, perf_counter()-t)'
```
