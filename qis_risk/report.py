"""Static presentation of precomputed results, shared by Jupyter and offline HTML."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import tempfile
import textwrap
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qis-matplotlib"))

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from scipy.cluster.hierarchy import leaves_list, linkage

if TYPE_CHECKING:
    from .model import ReviewResult


@dataclass(frozen=True)
class DisplayConfig:
    history_months: int = 36
    top_n_pairs: int = 5
    show_heatmap: bool = True

    def __post_init__(self):
        for name in ("history_months", "top_n_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.show_heatmap, bool):
            raise ValueError("show_heatmap must be boolean")


_CSS = """
.qis-review{--ink:#162f43;--teal:#087e8b;--muted:#5c6f7c;--rule:#dce5ea;
 color:var(--ink);background:#fff;font:15px/1.55 system-ui,-apple-system,sans-serif;
 max-width:1080px;margin:0 auto;padding:42px 44px;box-sizing:border-box}
.qis-review *{box-sizing:border-box}.qis-review h1{font-size:36px;line-height:1.15;
 letter-spacing:-1px;margin:8px 0 12px;font-weight:650}.qis-review h2{font-size:23px;
 margin:0 0 15px;line-height:1.3}.qis-review h3{font-size:16px;margin:24px 0 10px}
.qis-review p{margin:10px 0}.qis-review .eyebrow{letter-spacing:2px;font-size:11px;
 text-transform:uppercase;font-weight:750;color:var(--teal)}
.qis-review .muted,.qis-review .note{color:var(--muted);font-size:13px}
.qis-review .meta{display:flex;flex-wrap:wrap;gap:8px 24px;margin:20px 0 12px;
 font-size:12px}.qis-review .meta span{white-space:normal}
.qis-review .status{background:#f0f6f7;border-left:3px solid var(--teal);
 padding:12px 16px;font-size:13px;margin:20px 0}
.qis-review .status.failed{border-color:#a34835;background:#fff3ed}
.qis-review section{border-top:1px solid var(--rule);margin-top:32px;padding-top:26px}
.qis-review .cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
 gap:14px;margin:22px 0}.qis-review .card{padding:20px;background:#f3f7f9;
 border-top:3px solid var(--teal);border-radius:2px}
.qis-review .card-label{font-size:12px;font-weight:650;color:var(--muted)}
.qis-review .card-value{font-size:34px;font-weight:650;letter-spacing:-1px;margin:5px 0}
.qis-review .card-detail{font-size:12px;color:var(--muted)}
.qis-review .narrative{font-size:16px;max-width:900px}
.qis-review .chart{width:100%;height:auto;display:block;margin:12px 0 20px}
.qis-review .table-wrap{overflow-x:auto;margin:12px 0 18px}
.qis-review table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.qis-review th{background:#f1f5f7;color:#425c6e;font-size:11px;text-align:right;
 font-weight:650;border-bottom:1px solid var(--rule);padding:10px 9px}
.qis-review td{padding:9px;border-bottom:1px solid #e7edef;text-align:right}
.qis-review td:first-child,.qis-review th:first-child{text-align:left}
.qis-review tr.total td{font-weight:700;border-top:2px solid #a9bbc5}
.qis-review details{border:1px solid var(--rule);border-radius:4px;margin:12px 0;padding:12px 16px}
.qis-review summary{cursor:pointer;font-weight:650}.qis-review pre{white-space:pre-wrap;
 overflow-wrap:anywhere;background:#f5f7f8;padding:16px;font-size:11px;max-height:450px;overflow:auto}
.qis-review .two{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.qis-review .formula{font-family:ui-monospace,monospace;background:#f3f7f9;padding:12px}
.qis-review .glossary{display:grid;grid-template-columns:1fr 1fr;gap:0 28px}
.qis-review .glossary>div{padding:16px 0;border-bottom:1px solid var(--rule);min-width:0}
.qis-review .glossary dt{font-weight:650;font-size:14px}
.qis-review .glossary dd{margin:6px 0 0;font-size:13px;color:var(--muted)}
.qis-review .glossary .formula{display:block;color:var(--ink);overflow-wrap:anywhere;font-size:12px}
.qis-review footer{border-top:1px solid var(--rule);padding-top:20px;margin-top:32px;font-size:12px;color:var(--muted)}
@media(max-width:720px){.qis-review{padding:24px 18px}.qis-review h1{font-size:29px}
 .qis-review .cards,.qis-review .two,.qis-review .glossary{grid-template-columns:1fr}.qis-review .card-value{font-size:29px}}
@media print{.qis-review{padding:0;max-width:none}.qis-review section{break-inside:avoid}
 .qis-review details{display:block}.qis-review details>*{display:block}.qis-review pre{max-height:none}}
"""


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _fmt(value, kind="number"):
    if not _finite(value):
        return "Unavailable"
    value = float(value)
    if kind == "percent":
        return f"{100 * value:.2f}%"
    if kind == "points":
        return f"{100 * value:+.2f}"
    if kind == "signed":
        return f"{value:+.3f}"
    return f"{value:.3f}"


def _date(value):
    return "Unavailable" if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%d %b %Y")


def _column_label(column):
    """Human-readable table labels; technical keys and source values stay untouched."""
    words = str(column).replace("_", " ").split()
    acronyms = {"dr": "DR", "ewma": "EWMA", "ess": "ESS", "qis": "QIS",
                "csv": "CSV", "psd": "PSD", "i": "i", "j": "j"}
    return " ".join(acronyms.get(word.lower(), word.capitalize()) for word in words)


def _table(frame, columns=None, formats=None, index=False):
    if frame is None or frame.empty:
        return '<p class="muted">No available observations.</p>'
    frame = frame.copy()
    if index:
        frame = frame.reset_index()
    formats = formats or {}
    columns = columns or {str(c): _column_label(c) for c in frame.columns}
    available = [c for c in columns if c in frame.columns]
    head = "".join(f"<th>{escape(columns[c])}</th>" for c in available)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for c in available:
            v = row[c]
            if v is None or (np.isscalar(v) and pd.isna(v)):
                rendered = "—" if row.get("pair") in ("Other pairs", "Total") else "Unavailable"
            elif c in formats:
                rendered = _fmt(v, formats[c])
            elif isinstance(v, (float, np.floating)):
                rendered = _fmt(v)
            else:
                rendered = str(v)
                if c == "pair" and rendered == "Other pairs":
                    rendered = "Other Pairs"
            cells.append(f"<td>{escape(rendered)}</td>")
        total = str(row.get("pair", "")) == "Total"
        rows.append(f'<tr class="{"total" if total else ""}">{"".join(cells)}</tr>')
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def rank_pairs(frame: pd.DataFrame, value_column: str, top_n: int) -> pd.DataFrame:
    """Rank signed precomputed effects, and reconcile omitted rows for presentation."""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["pair"] = result["strategy_i"].astype(str) + " / " + result["strategy_j"].astype(str)
    if not all(_finite(v) for v in result[value_column]):
        return pd.DataFrame()
    result["_magnitude"] = result[value_column].abs()
    result = result.sort_values(
        ["_magnitude", "strategy_i", "strategy_j"], ascending=[False, True, True], kind="stable"
    ).drop(columns="_magnitude")
    selected = result.iloc[:top_n].copy()
    extra = [
        {"pair": "Other pairs", value_column: result.iloc[top_n:][value_column].sum()},
        {"pair": "Total", value_column: result[value_column].sum()},
    ]
    return pd.concat([selected, pd.DataFrame(extra)], ignore_index=True)


def _pairs(frame, value_column, top_n, monthly=False):
    if frame is None or frame.empty:
        return '<p class="muted">No strategy pairs.</p>'
    ranked = rank_pairs(frame, value_column, top_n)
    if ranked.empty:
        return '<p class="muted">Pair effects unavailable: the required volatility denominator is negligible.</p>'
    columns = {"pair": "Strategy Pair"}
    if monthly:
        columns.update(previous_correlation="Previous ρ", current_correlation="Current ρ", correlation_change="Δρ")
    else:
        columns["correlation"] = "Current ρ"
    columns[value_column] = "Effect (vol pts)" if monthly else "Contribution (vol pts)"
    formats = {value_column: "points", "correlation_change": "signed"}
    return _table(ranked, columns, formats)


def _figure(fig, description):
    with mpl.rc_context({"svg.fonttype": "none", "font.size": 10, "font.family": "DejaVu Sans"}):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight", facecolor="white")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="chart" alt="{escape(description)}" src="data:image/svg+xml;base64,{encoded}">'


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#dce5ea")
    ax.grid(axis="y", color="#e9eef1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#5c6f7c", labelsize=9)


def _plot_history(ax, history, column, color, scale=1, linestyle="-"):
    if column not in history:
        return
    complete = history.get("completed_month", pd.Series(True, index=history.index)).fillna(False)
    values = pd.to_numeric(history[column], errors="coerce") * scale
    ax.plot(history.index, values.where(complete), color=color, linewidth=1.8, linestyle=linestyle)
    partial = values.loc[~complete].dropna()
    if not partial.empty:
        ax.scatter(partial.index, partial, color=color, marker="D", s=25, zorder=4)


def _history_chart(history):
    fig = Figure(figsize=(10, 7.4), layout="constrained")
    axes = fig.subplots(4, 1, sharex=True)
    fields = [
        ("portfolio_volatility", "Annualized Volatility", True),
        ("diversification_ratio", "Diversification Ratio", False),
        ("effective_exposures", "Effective Standalone Exposures", False),
        ("correlation_multiplier", "Correlation Multiplier", False),
    ]
    for ax, (column, title, percent) in zip(axes, fields, strict=True):
        _style_axes(ax)
        _plot_history(ax, history, column, "#087e8b")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color="#162f43")
        if percent:
            if "zero_correlation_volatility" in history:
                _plot_history(ax, history, "zero_correlation_volatility", "#899aa7", linestyle="--")
                ax.plot([], [], "--", color="#899aa7", label="Zero-Correlation Reference")
                if "completed_month" in history and (~history.completed_month).any():
                    ax.plot([], [], "D", color="#087e8b", label="Current / Partial Month", markersize=4)
                ax.legend(loc="upper left", frameon=False, fontsize=8)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if column == "correlation_multiplier":
            ax.axhline(1, color="#aebbc4", linestyle=":", linewidth=1)
    return _figure(fig, "Aligned history of portfolio and zero-correlation volatility, diversification ratio, effective standalone exposures, and correlation multiplier")


def _conditions_chart(history):
    fig = Figure(figsize=(10, 2.6), layout="constrained")
    for ax, (field, title) in zip(fig.subplots(1, 2), [
        ("equal_risk_correlation_multiplier", "Fixed Equal-Risk Correlation Reference"),
        ("fast_slow_disagreement", "Fast − Slow Model Disagreement (vol pts)"),
    ], strict=True):
        _style_axes(ax)
        if field in history:
            scale = 100 if field == "fast_slow_disagreement" else 1
            _plot_history(ax, history, field, "#517091", scale=scale)
            if scale == 100:
                ax.axhline(0, color="#aebbc4", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=9)
        ax.tick_params(axis="x", labelrotation=25)
    return _figure(fig, "Fixed equal-risk reference and fast-versus-slow model disagreement")


def _waterfall(attribution):
    start, end = attribution["start_volatility"] * 100, attribution["end_volatility"] * 100
    effects = attribution["effects"]
    values = [start, effects["weights"] * 100, effects["standalone_volatility"] * 100, effects["correlation"] * 100, end]
    fig = Figure(figsize=(10, 3.2), layout="constrained")
    ax = fig.subplots()
    _style_axes(ax)
    running = start
    ax.bar(0, start, color="#233e54", width=0.58)
    for i in range(1, 4):
        v = values[i]
        ax.bar(i, abs(v), bottom=min(running, running + v), color="#c1844b" if v >= 0 else "#087e8b", width=0.58)
        ax.plot([i - 1 + .29, i - .29], [running, running], color="#b4c1c9", linewidth=1)
        running += v
        ax.annotate(f"{v:+.2f}", (i, max(running, running - v)), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
    ax.bar(4, end, color="#233e54", width=0.58)
    for i, v in [(0, start), (4, end)]:
        ax.annotate(f"{v:.2f}%", (i, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(range(5), ["Previous", "Weights", "Standalone Volatility", "Correlation", "Current"])
    ax.set_ylabel("Annualized Volatility (%)")
    ax.margins(y=.22)
    return _figure(fig, "Previous-to-current portfolio volatility waterfall with weights, standalone volatility, and correlation effects")


def _strategy_waterfall(strategies, portfolio_volatility):
    """Accumulate already-calculated signed Euler contributions for display only."""
    if (strategies.empty or not _finite(portfolio_volatility)
            or not all(_finite(v) for v in strategies["euler_contribution"])):
        return ('<p class="muted">Strategy risk contributions unavailable: '
                'the required portfolio volatility denominator is negligible or unavailable.</p>')
    ordered = strategies.sort_index().sort_values(
        "euler_contribution", ascending=False, kind="stable"
    )
    names = [textwrap.fill(str(name), width=32, break_long_words=True,
                           break_on_hyphens=False) for name in ordered.index]
    values = ordered["euler_contribution"].to_numpy() * 100
    total = portfolio_volatility * 100
    row_height = max(len(name.splitlines()) for name in names) * .16 + .3
    fig = Figure(figsize=(10, max(3.2, row_height * (len(names) + 1) + 1)),
                 layout="constrained")
    ax = fig.subplots()
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e9eef1", linewidth=.7)
    running = 0.0
    for i, value in enumerate(values):
        end = running + value
        color = "#c1844b" if value > 0 else "#087e8b" if value < 0 else "#899aa7"
        ax.barh(i, abs(value), left=min(running, end), height=.58, color=color)
        if value == 0:
            ax.plot(running, i, marker="|", color=color, markersize=10)
        ax.annotate(f"{value:+.2f}", (max(running, end), i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=9)
        ax.plot([end, end], [i + .29, i + 1 - .29], color="#b4c1c9", linewidth=1)
        running = end
    ax.barh(len(names), total, left=0, height=.58, color="#233e54")
    ax.annotate(f"{total:.2f}%", (total, len(names)), xytext=(7, 0),
                textcoords="offset points", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(names) + 1), [*names, "Portfolio Volatility"])
    ax.invert_yaxis()
    ax.axvline(0, color="#aebbc4", linewidth=.8)
    ax.set_xlabel("Annualized Volatility (vol pts)")
    ax.margins(x=.18, y=.06)
    return _figure(fig, "Strategy Risk Contributions: signed Euler contributions in descending "
                   "order, including zero and negative contributions, totaling portfolio volatility")


def _clustered_correlation(correlation):
    """Permute a copy of the official matrix; never alter its values or model order."""
    names = sorted(correlation.index)
    ordered = correlation.loc[names, names].copy()
    if len(names) < 2:
        return ordered
    # Inputs are model-validated. Clip only the distance calculation's roundoff;
    # keep the original correlation coefficients for the displayed heatmap.
    upper = ordered.to_numpy()[np.triu_indices(len(names), k=1)]
    distance = np.sqrt(np.clip((1 - upper) / 2, 0, 1))
    tree = linkage(distance, method="average", optimal_ordering=True)
    order = leaves_list(tree)
    return ordered.iloc[order, order].copy()


def _heatmap(correlation, half_life):
    correlation = _clustered_correlation(correlation)
    fig = Figure(figsize=(9, 6), layout="constrained")
    ax = fig.subplots()
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    names = list(correlation.index)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    if len(names) <= 12:
        for i in range(len(names)):
            for j in range(len(names)):
                value = correlation.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > .65 else "#183447")
    fig.colorbar(im, ax=ax, shrink=.8, label="Correlation")
    return _figure(fig, f"Official Correlation Matrix ({half_life:g}-Day Half-Life), "
                   "clustered by correlation, fixed color scale from minus one to plus one")


def _stress_grid(grid):
    pivot = grid.pivot(index="multiplier", columns="kappa", values="volatility").sort_index().sort_index(axis=1)
    uplift = grid.pivot(index="multiplier", columns="kappa", values="uplift").reindex_like(pivot)
    dr = grid.pivot(index="multiplier", columns="kappa", values="diversification_ratio").reindex_like(pivot)
    fig = Figure(figsize=(8.8, 3.2), layout="constrained")
    ax = fig.subplots()
    ax.imshow(pivot, cmap="Blues", aspect="auto", vmin=0, vmax=float(pivot.max().max()) * 1.5)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{_fmt(pivot.iloc[i, j], 'percent')}\n{_fmt(uplift.iloc[i, j], 'points')} vol pts", ha="center", va="center", fontsize=12, color="#162f43")
    ax.set_xticks(range(len(pivot.columns)), [f"{k:.0%} Convergence\nDR {_fmt(dr.iloc[0, j])}" for j, k in enumerate(pivot.columns)])
    ax.set_yticks(range(len(pivot)), [f"{m:.2f}× Volatility" for m in pivot.index])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    return _figure(fig, "Combined uniform volatility and correlation-convergence sensitivity grid; every uplift uses the current baseline")


def _json(value):
    return '<pre>' + escape(json.dumps(value, indent=2, default=str, ensure_ascii=False)) + '</pre>'


def _glossary():
    entries = [
        ("Portfolio Volatility", "σ = √(aᵀRa)",
         "Annualized modeled volatility of the synthetic excess-return portfolio, using "
         "closing holdings and the official correlation matrix. It measures second-moment "
         "risk, rather than all nonlinear, liquidity or tail risks."),
        ("Standalone Volatility", "sᵢ = √A × dᵢ",
         "Annualized volatility of strategy i on its own. The model annualizes its daily "
         "EWMA volatility dᵢ using A trading periods per year (252 by default)."),
        ("Standalone Risk Exposure", "aᵢ = xᵢsᵢ; S = Σᵢ aᵢ",
         "Closing portfolio weight times standalone volatility. S is the sum of these "
         "exposures before allowing for diversification; it is the numerator of DR."),
        ("Diversification Ratio", "DR = S / σ = √N_eff / M_R",
         "The sum of standalone risk exposures divided by portfolio volatility. For this "
         "nonnegative-exposure portfolio, 1 means no volatility diversification benefit "
         "and larger values mean more benefit. A decline alone does not identify a "
         "correlation change."),
        ("Effective Standalone Exposures", "N_eff = S² / Σᵢ aᵢ²",
         "The effective number of standalone risk exposures: a concentration measure, "
         "not a count of independent economic bets. It ranges from 1 to the number of "
         "positive exposures and reaches that count when the exposures are equal."),
        ("Zero-Correlation Volatility", "σ₀ = √(Σᵢ aᵢ²)",
         "Portfolio volatility if all pair correlations were zero, holding current "
         "weights and standalone volatilities fixed. This is a reference scenario, "
         "not a target or the lowest possible risk."),
        ("Correlation Uplift", "U_R = σ − σ₀",
         "The signed difference from zero-correlation volatility, shown in volatility "
         "points. A negative value means correlations lower risk relative to that reference."),
        ("Correlation Multiplier", "M_R = σ / σ₀",
         "Portfolio volatility relative to the zero-correlation reference: above 1 means "
         "higher risk, below 1 means lower risk. Its history reflects both correlation "
         "changes and changes in the live exposure mix."),
        ("Equal-Risk Correlation Reference", "M_R* = √(1ᵀR1 / N)",
         "The correlation multiplier for equal standalone risk exposures across the fixed "
         "N-strategy universe. It isolates estimated correlation changes from changes in "
         "live exposures, without implying economic causality or statistical significance."),
        ("Model Disagreement", "W = √(aᵀR_fast a) − √(aᵀR_official a)",
         "Fast-versus-slow model disagreement: the volatility difference using the fast "
         "and official correlation estimates at the same current exposures. Positive "
         "values mean the fast estimate implies more risk. This is not a calibrated alert."),
        ("Euler Risk Contribution", "RCᵢ = aᵢ(Ra)ᵢ / σ; Σᵢ RCᵢ = σ",
         "A strategy's additive contribution to current portfolio volatility. The strategy "
         "waterfall shows these signed contributions, largest first. Negative contributions "
         "reduce total risk; zero-weight strategies have zero contributions when σ is meaningful."),
        ("Additive Cross-Covariance Contribution", "C_R = 2Σᵢ<ⱼ aᵢaⱼρᵢⱼ / σ",
         "The sum of unique-pair contributions to portfolio volatility. Together with "
         "the self contribution Σᵢaᵢ² / σ, it sums to σ. This additive quantity differs "
         "from correlation uplift σ − σ₀. A current pair contribution is also distinct "
         "from its monthly correlation-driven effect."),
        ("Correlation Convergence", "Rκ = (1 − κ)R + κ11ᵀ; ρκ = ρ + κ(1 − ρ)",
         "Closes fraction κ of every correlation's distance to +1 at fixed exposures. "
         "For example, 25% convergence moves a correlation of 0.20 to "
         "0.20 + 0.25 × (1 − 0.20) = 0.40, not 0.45. These are illustrative "
         "sensitivities without assigned probabilities."),
        ("Volatility Multiplier", "σ_m,κ = mσκ; DR_m,κ = S / σκ",
         "Multiplies every standalone volatility by the same positive factor m. At a "
         "fixed convergence scenario, portfolio volatility scales by m while DR is "
         "unchanged. Every grid uplift is measured from the original current baseline."),
        ("EWMA Half-Life", "λ = 2^(−1/h)",
         "The h trading-calendar steps over which an observation's relative weight halves. "
         "An exponentially weighted moving average uses all eligible finite history; "
         "the half-life is not a rolling lookback window or an observation count. "
         "Excluded dates still consume decay steps."),
        ("Effective Sample Size", "ESS = 1 / Σᵤ ωᵤ²",
         "The effective count implied by normalized EWMA weights ω. Correlation ESS uses "
         "eligible standardized-return vectors. It is a weighting diagnostic, not the "
         "number of independent observations or a confidence measure."),
        ("Variance-Shapley Attribution", "φₖ^σ = φₖ^V / (σ_previous + σ_current)",
         "Average each block's incremental variance effect across all six orders of "
         "updating weights, standalone volatilities and correlation, then divide by "
         "the sum of endpoint volatilities. The effects reconcile to the monthly "
         "volatility change. This is an allocation convention, not proof of causality; "
         "the weights effect includes both trading and drift."),
    ]
    pieces = ['<section id="glossary"><span class="eyebrow">06 / Reference</span>'
              '<h2>Glossary</h2><p class="note">Notation: x is the closing weight vector, '
              's is annualized standalone volatility, a = x ⊙ s, and R is the official '
              'correlation matrix. One volatility point is 0.01 in decimal annualized '
              'volatility. Ratios and contributions are unavailable when their required '
              'denominators are negligible.</p><dl class="glossary">']
    for term, formula, definition in entries:
        pieces.append(f'<div><dt>{escape(term)}</dt><dd><span class="formula">'
                      f'{escape(formula)}</span>{escape(definition)}</dd></div>')
    return "".join([*pieces, "</dl></section>"])


def _appendix(result):
    current = result.current
    config = result.metadata.get("model_config", {})
    pieces = ['<section><span class="eyebrow">Technical Detail</span><h2>Model, Holdings &amp; Diagnostics</h2><details><summary>Open Technical Appendix</summary>']
    if current is not None:
        pieces += ["<h3>Holdings and Strategy Risk</h3>", _table(current.strategies, index=True, formats={
            "weight": "percent", "target_weight": "percent", "standalone_volatility": "percent",
            "risk_exposure": "points", "euler_contribution": "points", "self_contribution": "points", "cross_contribution": "points",
        }), '<p class="note">Exposure and contribution columns are annualized volatility points. Target weights are from the latest rebalance; current weights include drift.</p>', "<h3>Snapshot Observations, Effective Sample Sizes and Matrix Checks</h3>", _json(current.diagnostics)]
        pieces += ['<details><summary>All Current Strategy Pairs</summary>', _table(current.pairs, formats={"contribution": "points"}), "</details>"]
    attribution = result.attribution
    if isinstance(attribution.get("pairs"), pd.DataFrame):
        pieces += ['<details><summary>All Monthly Correlation Effects</summary>', _table(attribution["pairs"], formats={"effect": "points"}), _table(attribution.get("strategies"), index=True, formats={"effect": "points"}), "</details>"]
    for key, title in [
        ("stale_returns", "Repeated Zero-Return Diagnostics"),
        ("shrinkage_sensitivity", "Shrinkage Sensitivity"),
        ("half_life_sensitivity", "Correlation Half-Life Sensitivity"),
        ("sensitivity_pairs", "Pair Ranking and Sign Sensitivity"),
        ("weekly", "Daily versus Weekly Annualization Diagnostic"),
    ]:
        value = result.diagnostics.get(key)
        if isinstance(value, pd.DataFrame):
            percent_columns = {"portfolio_volatility", "zero_correlation_volatility", "daily_volatility", "weekly_volatility", "standalone_sum"}
            point_columns = {"contribution", "self_contribution", "cross_contribution", "correlation_uplift", "fast_slow_disagreement", "change_weights", "change_standalone_volatility", "change_correlation"}
            formats = {c: "percent" for c in percent_columns}
            formats.update({c: "points" for c in point_columns})
            indexed = key in {"weekly", "stale_returns"}
            fields = value.reset_index().columns if indexed else value.columns
            columns = {c: _column_label(c) + (" (vol pts)" if c in point_columns else " (%)" if c in percent_columns else "") for c in fields}
            pieces += [f'<details><summary>{title}</summary>', _table(value, columns, formats, index=indexed), "</details>"]
    weekly_note = (
        f'Weekly diagnostics compare paired daily and non-overlapping weekly samples over the trailing '
        f'{config.get("weekly_history_months", 36)} calendar months, with '
        f'{config.get("annualization", 252):g}/{config.get("weekly_annualization", 52):g} annualization '
        f'and at least {config.get("weekly_min_intervals", 104)} valid weekly intervals. '
        'Lag-one autocorrelations preserve calendar gaps. These are empirical diagnostics, '
        'separate from the official EWMA model.'
    )
    pieces += [f'<p class="note">{escape(weekly_note)}</p>', "<h3>Historical DR Percentile</h3>", _json(result.diagnostics.get("dr_percentile", {}))]
    pieces += ['<details><summary>Coverage, Validation and Availability</summary>', _json({k: v for k, v in result.diagnostics.items() if not isinstance(v, pd.DataFrame)}), "</details>"]
    pieces += ['<details><summary>Model Configuration and Run Provenance</summary>', _json(result.metadata), "</details>"]
    pieces += ['''<details><summary>Methodology and Interpretation</summary>
<p>Closing weights refer to a synthetic excess-return book. At each rebalance close old units first earn that day's return, then new targets set the units held through the next interval. The base value is 100; no collateral yield, financing adjustments, external flows or transaction costs are included.</p>
<p>Returns are computed from adjacent declared trading-date levels before selecting common complete vectors. Finite-history zero-mean EWMA estimates decay across excluded observations. Default volatility half-life is 60 trading days; correlation half-lives are 126 and 42. Standardization divides each return by the preceding close's EWMA daily volatility, after 60 initialization observations. Default publication requires 504 common raw observations and 444 standardized vectors.</p>
<p class="formula">a = x ⊙ s; σ² = aᵀRa; σ₀² = Σ aᵢ²; DR = Σ aᵢ / σ<br>N_eff = (Σ aᵢ)² / Σ aᵢ²; M_R = σ / σ₀; DR = √N_eff / M_R</p>
<p>The effective number of standalone risk exposures measures concentration, not independent economic bets. Zero correlation is a reference scenario. Current correlation uplift σ − σ₀ differs from the additive cross-covariance contribution 2Σᵢ&lt;ⱼ aᵢaⱼρᵢⱼ / σ. Negative contributions are retained.</p>
<p>Monthly attribution averages incremental variance effects across all six orders of updating weights, standalone volatilities and correlation. Each effect is divided by the sum of endpoint volatilities to reconcile to the volatility change. This is variance-Shapley attribution expressed in volatility units, not proof of economic causality. The weights effect includes both trading and drift.</p>
<p class="formula">Rκ = (1 − κ)R + κ11ᵀ; σ²κ = (1 − κ)σ² + κ(Σ aᵢ)²</p>
<p>Convergence closes a fraction of each correlation's distance to +1. Pair stress effects divide 2κaᵢaⱼ(1 − ρᵢⱼ) by σκ + σ. Uniform standalone-volatility scaling multiplies portfolio volatility and leaves DR unchanged. All grid uplifts use the original current baseline.</p>
<p>Unavailable values retain explicit reasons. Matrix and reconciliation failures block publication. Effective sample sizes measure normalized weighting, not independent information. Historical DR percentiles use midranks among valid completed month-ends in the preceding 36 months, exclude the current snapshot, and require 24 prior observations.</p>
<p>Historical estimates use the current input vintage; they are not point-in-time historical records without archived source data. Volatility is a second-moment risk measure and does not capture all nonlinear, liquidity or tail-dependence risks.</p>
</details></details></section>''']
    return "".join(pieces)


def render_report(result: ReviewResult, display: DisplayConfig | None = None) -> str:
    """Render one self-contained HTML document, without altering the analytical result."""
    display = display or DisplayConfig()
    portfolio = result.metadata.get("portfolio", {})
    title = portfolio.get("portfolio_id", "QIS portfolio") if isinstance(portfolio, dict) else str(portfolio)
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)} — Risk Review</title><style>{_CSS}</style></head><body><article class="qis-review">']
    parts += [f'<header><span class="eyebrow">QIS / Portfolio Review</span><h1>Diversification &amp; Correlation</h1><p class="muted">{escape(title)} · Synthetic excess-return portfolio</p><div class="meta"><span><b>As of</b> {_date(result.resolved_date)}</span><span><b>Requested</b> {"Latest input" if result.requested_date is None else _date(result.requested_date)}</span><span><b>Comparison</b> {_date(result.comparison_date)}</span></div></header>']
    coverage = result.diagnostics.get("coverage", {})
    status = "Risk model available" if result.status == "valid" else f"Risk report {result.status}"
    reasons = "; ".join(str(r) for r in result.reasons)
    parts += [f'<div class="status {"failed" if result.status == "failed" else ""}"><b>{escape(status)}</b>{": " + escape(reasons) if reasons else ""}<br>Calendar: CSV trading dates. Entirely omitted trading dates cannot be detected.</div>']
    history = result.history.copy()
    if not history.empty:
        valid = history.loc[history["status"].eq("valid")] if "status" in history else history
        if not valid.empty:
            parts += [f'<p class="note">Available model history: {_date(valid.index.min())} – {_date(valid.index.max())}. Charts request {display.history_months} months and retain unavailable periods as gaps.</p>']
        if result.resolved_date is not None:
            history = history.loc[history.index >= pd.Timestamp(result.resolved_date) - pd.DateOffset(months=display.history_months)]
    if result.current is None or result.status != "valid":
        parts += ["<h2>Coverage and Availability</h2>", _json(coverage), _glossary(), _appendix(result), "</article></body></html>"]
        return "".join(parts)
    current, previous = result.current, result.previous
    parts += [f'<div class="meta"><span><b>Closing Holdings</b> {_date(current.diagnostics.get("holdings_date", current.date))}</span><span><b>Latest Rebalance</b> {_date(current.diagnostics.get("last_rebalance"))}</span><span><b>Latest Target Weights</b> {_date(current.diagnostics.get("latest_target_date"))}</span></div>']
    metrics = current.metrics
    prior_metrics = previous.metrics if previous is not None else {}
    parts += ['<section><span class="eyebrow">01 / Current Diversification</span><h2>How Diversified Is the Portfolio Today?</h2><div class="cards">']
    for key, label, kind in [
        ("portfolio_volatility", "Portfolio Volatility", "percent"),
        ("diversification_ratio", "Diversification Ratio", "number"),
        ("correlation_uplift", "Correlation Uplift (vol pts)", "points"),
    ]:
        value = metrics.get(key)
        if key == "correlation_uplift":
            detail = f"Zero-correlation volatility: {_fmt(metrics.get('zero_correlation_volatility'), 'percent')}"
        elif _finite(value) and _finite(prior_metrics.get(key)):
            kind_delta = "points" if kind == "percent" else "signed"
            detail = f"{_fmt(value - prior_metrics[key], kind_delta)}{' vol pts' if kind == 'percent' else ''} since {_date(result.comparison_date)}"
        else:
            detail = "Monthly comparison unavailable" if _finite(value) else "Required denominator is negligible"
        parts += [f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{_fmt(value, kind)}</div><div class="card-detail">{escape(detail)}</div></div>']
    parts += [f'</div><p class="narrative">At today’s holdings and standalone volatilities, estimated portfolio volatility is <b>{_fmt(metrics.get("portfolio_volatility"), "percent")}</b>; it would be <b>{_fmt(metrics.get("zero_correlation_volatility"), "percent")}</b> with zero pairwise correlation, a difference of <b>{_fmt(metrics.get("correlation_uplift"), "points")} volatility points</b>.</p><p class="note">Zero correlation is a reference scenario, not a target or the lowest achievable risk.</p>']
    if not history.empty:
        parts += [_history_chart(history), "<h3>Correlation Conditions</h3>", _conditions_chart(history)]
    parts += ['<p class="note">The equal-risk reference holds the strategy universe fixed. Fast-versus-slow disagreement holds current exposures fixed; neither indicator is a calibrated alert.</p></section>']
    parts += ['<section><span class="eyebrow">02 / Strategy Risk</span>'
              '<h2>Strategy Risk Contributions</h2>',
              _strategy_waterfall(current.strategies, metrics.get("portfolio_volatility")),
              '<p class="note">Signed Euler contributions to current portfolio volatility, '
              'ordered from largest to smallest. Every strategy is included; negative '
              'contributions reduce total risk. Contributions are annualized volatility '
              'points, and the total is portfolio volatility.</p></section>']
    parts += [f'<section><span class="eyebrow">03 / Monthly Change</span><h2>Why Did Risk Change?</h2><p class="muted">Change since {_date(result.comparison_date)} · closing holdings through {_date(result.resolved_date)}</p>']
    attribution = result.attribution
    if attribution.get("status") == "valid" and all(_finite(v) for v in attribution.get("effects", {}).values()) and attribution.get("effects"):
        parts += [_waterfall(attribution)]
        effects = attribution["effects"]
        labels = {"weights": "Weights", "standalone_volatility": "Standalone Volatility", "correlation": "Correlation"}
        ordered = sorted(effects, key=lambda k: (-abs(effects[k]), k))
        statement = "; ".join(f"{labels[k]} {_fmt(effects[k], 'points')} vol pts" for k in ordered)
        parts += [f'<p class="narrative">Modeled changes, ranked by absolute effect: {escape(statement)}.</p>']
    else:
        parts += [f'<p class="muted">Monthly attribution unavailable: {escape(str(attribution.get("reason", "A valid comparison snapshot is required.")))}</p>']
    parts += ['<p class="note">The weights effect includes deliberate reallocation and performance-driven drift. Effects use variance-Shapley attribution expressed in volatility units.</p></section><section><span class="eyebrow">04 / Relationships</span><h2>Which Relationships Matter?</h2><h3>Current Cross-Covariance Contributions</h3>', _pairs(current.pairs, "contribution", display.top_n_pairs), '<p class="note">Total reconciles to the additive cross-covariance contribution, which differs from the zero-correlation uplift.</p><h3>Monthly Correlation Drivers</h3>']
    if attribution.get("status") == "valid":
        parts += [_pairs(attribution.get("pairs"), "effect", display.top_n_pairs, monthly=True)]
    else:
        parts += ['<p class="muted">Unavailable without a valid monthly comparison.</p>']
    if display.show_heatmap:
        half_life = result.metadata["model_config"]["corr_half_life_slow"]
        parts += [f"<h3>Official Correlation Matrix ({half_life:g}-Day Half-Life)</h3>",
                  '<p class="note">The EWMA half-life is measured in trading days, not a '
                  'rolling lookback window. Strategies are clustered by correlation; '
                  'the display order can change with the snapshot.</p>',
                  _heatmap(current.correlation, half_life)]
    parts += ['</section><section><span class="eyebrow">05 / Sensitivity</span><h2>Where Could Diversification Disappear?</h2><p>Illustrative sensitivities without assigned probabilities. Holdings and standalone volatilities stay fixed in the correlation-only scenarios.</p>']
    stress = result.stress
    if isinstance(stress.get("scenarios"), pd.DataFrame):
        parts += [_table(stress["scenarios"], {"kappa": "Convergence", "volatility": "Stressed Volatility", "uplift": "Uplift (vol pts)", "percentage_increase": "Increase", "diversification_ratio": "Stressed DR"}, {"kappa": "percent", "volatility": "percent", "uplift": "points", "percentage_increase": "percent"})]
        parts += [f'<h3>Pair Contributors to {_fmt(stress.get("pair_kappa", .25), "percent")} Convergence</h3>', _pairs(stress.get("pairs"), "contribution", display.top_n_pairs)]
    if isinstance(stress.get("grid"), pd.DataFrame) and not stress["grid"].empty:
        parts += ["<h3>Combined Volatility and Correlation Sensitivity</h3>", _stress_grid(stress["grid"]), '<p class="note">Each cell shows annualized volatility and uplift from the same current baseline. DR is shown once per correlation column because uniform volatility scaling leaves it unchanged.</p>']
    parts += ["</section>", _glossary(), _appendix(result), '<footer>Calculated at full precision; displayed values are rounded. Small displayed reconciliation differences can arise from rounding. This review describes modeled diversification and illustrative sensitivities; it does not prescribe trades.</footer></article></body></html>']
    return "".join(parts)


def render_input_error(error: Exception) -> str:
    """Present an invalid input contract without publishing a partial risk report."""
    message = escape(str(error))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>QIS Input Diagnostics</title>
<style>{_CSS}</style></head><body><article class="qis-review"><span class="eyebrow">QIS / Input Diagnostics</span>
<h1>Risk Report Unavailable</h1><div class="status failed">{message}</div>
<p>Correct the input contract and run all cells again. No PM risk estimates were published.</p>
</article></body></html>'''


def export_html(result: ReviewResult | str, path: str | Path, display: DisplayConfig | None = None) -> Path:
    """Export the same self-contained report displayed by the notebook."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = result if isinstance(result, str) else render_report(result, display)
    path.write_text(html, encoding="utf-8")
    return path.resolve()
