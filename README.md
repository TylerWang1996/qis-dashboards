# QIS portfolio risk dashboard

A local notebook and offline HTML report for reviewing diversification and correlation risk in a long-only portfolio of excess-return indices. Calculations live in a small Python package, outside the notebook.

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups --locked
uv run jupyter lab qis_risk_dashboard.ipynb
```

Select the project Python kernel and run all cells. The default inputs are eight years of reproducible simulated data for ten strategies. The notebook displays the review and writes an offline report into `reports/`.

## Inputs

Both CSVs begin with a `date` column and use the same strategy identifiers. Index levels must be positive; target weights are decimal fractions (0.25 means 25%), complete, nonnegative, and sum to one. Change input paths and the explicit ordered universe in the notebook when using real data.

CSV rows declare the trading calendar. Retain missing observations as blank cells; omitted trading dates cannot be detected. An optional `calendar_complete_through` date certifies coverage through non-trading month/week boundaries. Rebalances occur at the close, and index units remain constant between rebalances.

The portfolio is a synthetic excess-return book, not funded NAV. Currency conversion occurs upstream; collateral yield, costs, financing adjustments, flows, allocation optimization, and forecast calibration are outside this version.

## Development

```bash
uv run ruff check .
uv run pytest
```

Read `qis_risk_dashboard_design.md` for methodology, acceptance checks, and build progress. Real inputs, generated HTML, executed notebook copies, and caches are ignored by Git. Only the explicitly simulated example CSVs are committed.
