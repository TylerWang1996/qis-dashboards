"""Presentation contracts and a real fresh-kernel notebook acceptance run."""

import ast
import base64
import copy
import os
import sys
from html.parser import HTMLParser
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from nbclient import NotebookClient

from qis_risk import report as report_module
from qis_risk.data import DEMO_UNIVERSE, load_inputs
from qis_risk.model import ReviewResult, run_review
from qis_risk.report import (
    DisplayConfig,
    export_html,
    rank_pairs,
    render_input_error,
    render_report,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def review():
    inputs = load_inputs(
        ROOT / "data/demo/index_levels.csv", ROOT / "data/demo/rebalance_weights.csv",
        universe=DEMO_UNIVERSE,
    )
    return run_review(inputs, as_of_date="2025-12-17")


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.external = []
        self.scripts = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.scripts += 1
        if tag == "img":
            self.images.append(attrs)
        for key in ("src", "href"):
            value = attrs.get(key, "")
            if value.startswith(("http:", "https:", "//")):
                self.external.append(value)


class VisibleTextParser(HTMLParser):
    """Collect initially visible text, excluding collapsed appendix contents."""

    def __init__(self):
        super().__init__()
        self.details = 0
        self.hidden = 0
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            self.details += 1
        elif tag in {"style", "script"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag == "details":
            self.details -= 1
        elif tag in {"style", "script"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.details and not self.hidden:
            self.text.append(data)


@pytest.fixture
def captured_figures(monkeypatch):
    figures = []

    def capture(figure, description):
        figures.append((figure, description))
        return "<figure>Captured chart</figure>"

    monkeypatch.setattr(report_module, "_figure", capture)
    return figures


@pytest.fixture
def block_correlation():
    names = ["A", "B", "C", "D", "E", "F"]
    values = np.full((6, 6), .1)
    np.fill_diagonal(values, 1)
    for i, j, rho in [(0, 3, .9), (1, 4, .85), (2, 5, .8)]:
        values[i, j] = values[j, i] = rho
    return pd.DataFrame(values, index=names, columns=names)


def test_clustered_correlation_groups_blocks_and_preserves_source(block_correlation):
    source = block_correlation.loc[["F", "A", "C", "E", "B", "D"], :].copy()
    before = source.copy(deep=True)
    clustered = report_module._clustered_correlation(source)
    names = list(clustered.index)
    assert list(clustered.columns) == names
    for left, right in [("A", "D"), ("B", "E"), ("C", "F")]:
        assert abs(names.index(left) - names.index(right)) == 1
    pd.testing.assert_frame_equal(clustered.loc[before.index, before.columns], before)
    pd.testing.assert_frame_equal(source, before)
    # Alphabetical initialization makes tied cluster distances independent of input order.
    permuted = block_correlation.loc[["C", "D", "E", "B", "F", "A"],
        ["D", "C", "A", "B", "E", "F"]]
    pd.testing.assert_frame_equal(report_module._clustered_correlation(permuted), clustered)
    assert not np.shares_memory(clustered.to_numpy(), source.to_numpy())


@pytest.mark.parametrize("rho", [-.25, 0., .25, 1.])
def test_clustered_correlation_ties_are_reproducible(rho):
    names = ["zeta", "Alpha", "beta", "Delta"]
    values = np.full((4, 4), rho)
    np.fill_diagonal(values, 1)
    matrix = pd.DataFrame(values, index=names, columns=names)
    first = report_module._clustered_correlation(matrix)
    reordered = matrix.loc[names[::-1], ["Delta", "Alpha", "zeta", "beta"]]
    pd.testing.assert_frame_equal(report_module._clustered_correlation(reordered), first)
    pd.testing.assert_frame_equal(first.loc[names, names], matrix)


def test_single_strategy_correlation_remains_a_copied_matrix(captured_figures):
    matrix = pd.DataFrame([[1.0]], index=["Only Strategy"], columns=["Only Strategy"])
    clustered = report_module._clustered_correlation(matrix)
    pd.testing.assert_frame_equal(clustered, matrix)
    assert clustered is not matrix
    report_module._heatmap(matrix, 126.0)
    ax = captured_figures[0][0].axes[0]
    assert [label.get_text() for label in ax.get_xticklabels()] == ["Only Strategy"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["Only Strategy"]
    np.testing.assert_array_equal(ax.images[0].get_array(), [[1.0]])


def test_heatmap_axes_values_annotations_and_scale(block_correlation, captured_figures):
    report_module._heatmap(block_correlation, 84.5)
    figure, description = captured_figures[0]
    ax = figure.axes[0]
    names = [label.get_text() for label in ax.get_xticklabels()]
    assert names == [label.get_text() for label in ax.get_yticklabels()]
    np.testing.assert_array_equal(ax.images[0].get_array(), block_correlation.loc[names, names])
    assert ax.images[0].get_clim() == (-1, 1)
    assert len(ax.texts) == block_correlation.size
    for annotation in ax.texts:
        column, row = annotation.get_position()
        assert annotation.get_text() == f"{block_correlation.loc[names[int(row)], names[int(column)]]:.2f}"
    assert "84.5" in description
    assert "slow" not in description.lower()


def test_strategy_waterfall_preserves_signed_order_and_reconciles(captured_figures):
    strategies = pd.DataFrame({"euler_contribution": [.03, .03, 0., -.04, .05]},
        index=pd.Index(["B", "A", "Zero Exposure", "Hedge", "Largest"], name="strategy"))
    before = strategies.copy(deep=True)
    report_module._strategy_waterfall(strategies, .07)
    ax = captured_figures[0][0].axes[0]
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == ["Largest", "A", "B", "Zero Exposure", "Hedge", "Portfolio Volatility"]
    assert ax.yaxis_inverted()
    assert len(ax.patches) == 6
    starts = [0., 5., 8., 11., 11., 0.]
    ends = [5., 8., 11., 11., 7., 7.]
    for row, (bar, start, end) in enumerate(zip(ax.patches, starts, ends, strict=True)):
        actual_start, actual_end = bar.get_x(), bar.get_x() + bar.get_width()
        assert min(actual_start, actual_end) == pytest.approx(min(start, end))
        assert max(actual_start, actual_end) == pytest.approx(max(start, end))
        assert bar.get_y() + bar.get_height() / 2 == pytest.approx(row)
    annotations = " ".join(text.get_text() for text in ax.texts)
    assert all(value in annotations for value in ["+5.00", "+3.00", "+0.00", "-4.00"])
    assert "Annualized Volatility" in ax.get_xlabel()
    pd.testing.assert_frame_equal(strategies, before)
    # Tie handling remains independent of the model's configured universe order.
    report_module._strategy_waterfall(strategies.iloc[::-1], .07)
    reordered_ax = captured_figures[1][0].axes[0]
    assert [label.get_text() for label in reordered_ax.get_yticklabels()] == labels


def test_strategy_waterfall_uses_precomputed_total(captured_figures):
    # Deliberately distinguish the supplied total from the sum: the renderer must
    # consume the model result, without independently estimating or replacing it.
    strategies = pd.DataFrame({"euler_contribution": [.04, .02]}, index=["A", "B"])
    report_module._strategy_waterfall(strategies, .06000007)
    total = captured_figures[0][0].axes[0].patches[-1]
    assert total.get_x() == 0
    assert total.get_width() == pytest.approx(6.000007, abs=1e-10, rel=0)


def test_single_strategy_waterfall_has_contribution_and_total(captured_figures):
    strategies = pd.DataFrame({"euler_contribution": [.08]}, index=["Only Strategy"])
    report_module._strategy_waterfall(strategies, .08)
    ax = captured_figures[0][0].axes[0]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["Only Strategy", "Portfolio Volatility"]
    assert len(ax.patches) == 2
    assert [bar.get_width() for bar in ax.patches] == pytest.approx([8, 8])


@pytest.mark.parametrize("contributions,total", [([np.nan, np.nan], 0.),
    ([.03, np.nan], .03), ([.03, .01], np.nan)])
def test_undefined_strategy_contributions_show_reason_without_chart(contributions, total,
        captured_figures):
    strategies = pd.DataFrame({"euler_contribution": contributions}, index=["A", "B"])
    html = report_module._strategy_waterfall(strategies, total)
    assert "unavailable" in html.lower()
    assert "denominator" in html.lower() or "volatility" in html.lower()
    assert not captured_figures


def test_pair_ranking_is_signed_stable_and_reconciles():
    pairs = pd.DataFrame({
        "strategy_i": ["B", "A", "A", "C"],
        "strategy_j": ["C", "C", "B", "D"],
        "correlation": [.2, -.2, .1, .1],
        "contribution": [.02, -.02, .01, -.004],
    })
    actual = rank_pairs(pairs, "contribution", 2)
    assert list(actual.pair) == ["A / C", "B / C", "Other pairs", "Total"]
    assert actual.iloc[0].contribution == -.02
    assert actual.iloc[2].contribution == pytest.approx(.006)
    assert actual.iloc[:-1].contribution.sum() == pytest.approx(actual.iloc[-1].contribution)
    assert actual.iloc[-1].contribution == pytest.approx(pairs.contribution.sum())
    fewer = rank_pairs(pairs.iloc[:1], "contribution", 5)
    assert len(fewer) == 3
    assert fewer.iloc[1].contribution == 0


@pytest.mark.parametrize("kwargs", [{"top_n_pairs": 0}, {"history_months": -1},
    {"history_months": True}, {"show_heatmap": 1}])
def test_display_configuration_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        DisplayConfig(**kwargs)


def test_full_report_is_offline_and_carries_contract_labels(review, tmp_path):
    assert review.status == "valid"
    html = render_report(review)
    parser = AssetParser()
    parser.feed(html)
    assert parser.scripts == 0
    assert parser.external == []
    assert len(parser.images) == 6
    assert all(im.get("alt") for im in parser.images)
    for image in parser.images:
        assert image["src"].startswith("data:image/svg+xml;base64,")
        svg = base64.b64decode(image["src"].split(",", 1)[1]).decode()
        assert "<svg" in svg
        assert "<script" not in svg
    for text in [
        "How Diversified Is the Portfolio Today?", "Why Did Risk Change?",
        "Which Relationships Matter?", "Where Could Diversification Disappear?",
        "Closing Holdings", "Latest Rebalance", "Other Pairs",
        "Official Correlation Matrix (126-Day Half-Life)",
        "Portfolio Volatility", "Diversification Ratio", "Correlation Multiplier",
        "Illustrative sensitivities without assigned probabilities",
        "variance-Shapley attribution expressed in volatility units",
        "same current baseline", "Entirely omitted trading dates cannot be detected",
    ]:
        assert text in html
    assert "Official slow correlation matrix" not in html
    assert html.index("Current Diversification") < html.index("Strategy Risk Contributions")
    assert html.index("Strategy Risk Contributions") < html.index("Monthly Change")
    assert html.index("Glossary") < html.index("Open Technical Appendix")
    charts = "\n".join(base64.b64decode(im["src"].split(",", 1)[1]).decode() for im in parser.images)
    assert "Current / Partial Month" in charts
    assert "Portfolio Volatility" in charts
    assert "Standalone Volatility" in charts
    visible = VisibleTextParser()
    visible.feed(html)
    text = " ".join(visible.text)
    for term in [
        "Portfolio Volatility", "Standalone Volatility", "Standalone Risk Exposure",
        "Diversification Ratio", "Effective Standalone Exposures", "Zero-Correlation Volatility",
        "Correlation Uplift", "Correlation Multiplier", "Equal-Risk Correlation Reference",
        "Model Disagreement", "Euler Risk Contribution", "Additive Cross-Covariance Contribution",
        "Correlation Convergence", "Volatility Multiplier", "EWMA Half-Life",
        "Effective Sample Size", "Variance-Shapley Attribution",
    ]:
        assert term in text
    assert "trading days" in text
    assert "negative contributions reduce" in text.lower()
    path = export_html(html, tmp_path / "report.html")
    assert path.read_text() == html
    assert str(path).endswith("report.html")


def test_display_changes_do_not_mutate_analytical_outputs(review):
    before = copy.deepcopy(review)
    html = render_report(review, DisplayConfig(history_months=6, top_n_pairs=2, show_heatmap=False))
    parser = AssetParser()
    parser.feed(html)
    assert len(parser.images) == 5
    pd.testing.assert_frame_equal(review.history, before.history)
    pd.testing.assert_frame_equal(review.current.pairs, before.current.pairs)
    pd.testing.assert_frame_equal(review.current.correlation, before.current.correlation)
    pd.testing.assert_frame_equal(review.current.strategies, before.current.strategies)
    pd.testing.assert_frame_equal(review.attribution["pairs"], before.attribution["pairs"])
    assert review.current.metrics == before.current.metrics
    assert review.metadata == before.metadata
    changed = copy.deepcopy(review)
    changed.metadata["model_config"].update(weekly_history_months=24, weekly_min_intervals=80, annualization=260)
    changed_html = render_report(changed, DisplayConfig(show_heatmap=False))
    assert "trailing 24 calendar months, with 260/52 annualization and at least 80 valid weekly intervals" in changed_html


def test_correlation_heading_uses_resolved_half_life(review, captured_figures):
    changed = copy.deepcopy(review)
    changed.metadata["model_config"]["corr_half_life_slow"] = 84.5
    html = render_report(changed)
    assert "Official Correlation Matrix (84.5-Day Half-Life)" in html
    assert "Official Correlation Matrix (126-Day Half-Life)" not in html
    assert any("84.5" in description for _, description in captured_figures)
    before_count = len(captured_figures)
    hidden = render_report(changed, DisplayConfig(show_heatmap=False))
    assert "Official Correlation Matrix" not in hidden
    assert len(captured_figures) == before_count + 5


def test_glossary_is_expanded_and_contains_identity_and_convergence_example():
    html = report_module._glossary()
    parser = VisibleTextParser()
    parser.feed(html)
    visible = " ".join(parser.text)
    assert "Glossary" in visible
    assert "Diversification Ratio" in visible
    assert "Correlation Convergence" in visible
    assert any(identity in visible for identity in [
        "DR = √N_eff / M_R", "DR = S / σ = √N_eff / M_R",
    ])
    assert all(value in visible for value in ["0.20", "25%", "0.40"])
    assert "<details" not in html


def test_report_escapes_identifiers_without_changing_their_case(review):
    changed = copy.deepcopy(review)
    identifier = 'sTRATEGY <script>alert("x")</script>'
    original = changed.current.strategies.index[0]
    changed.current.strategies = changed.current.strategies.rename(index={original: identifier})
    changed.current.correlation = changed.current.correlation.rename(
        index={original: identifier}, columns={original: identifier})
    changed.metadata["portfolio"]["portfolio_id"] = '<script>Portfolio & title</script>'
    html = render_report(changed)
    parser = AssetParser()
    parser.feed(html)
    assert not parser.scripts
    assert "&lt;script&gt;Portfolio &amp; title&lt;/script&gt;" in html
    assert "sTRATEGY &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html
    charts = "\n".join(base64.b64decode(im["src"].split(",", 1)[1]).decode()
        for im in parser.images)
    assert 'sTRATEGY &lt;script&gt;alert("x")&lt;/script&gt;' in charts


@pytest.mark.parametrize("status", ["unavailable", "failed"])
def test_failure_reports_do_not_publish_headline_estimates(status):
    result = ReviewResult(status, ["Required levels are missing"], None, None, None,
        diagnostics={"coverage": {"common_returns": 100}})
    html = render_report(result)
    assert "Required levels are missing" in html
    assert "Coverage and Availability" in html
    assert 'class="card-value"' not in html


def test_invalid_input_summary_escapes_source_text():
    html = render_input_error(ValueError('<script>alert("bad input")</script>'))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "No PM risk estimates were published" in html


def test_notebook_remains_thin_and_committed_without_outputs():
    notebook = nbformat.read(ROOT / "qis_risk_dashboard.ipynb", as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(cells) <= 6
    for cell in cells:
        assert not cell.outputs
        assert cell.execution_count is None
        tree = ast.parse(cell.source)
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
            ast.For, ast.While, ast.ListComp, ast.DictComp, ast.BinOp)) for node in ast.walk(tree))


def test_notebook_runs_in_fresh_kernel_and_exports_offline_html(tmp_path, monkeypatch):
    notebook = nbformat.read(ROOT / "qis_risk_dashboard.ipynb", as_version=4)
    report_path = tmp_path / "dashboard.html"
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace('Path("reports/qis_risk_dashboard.html")', f"Path({str(report_path)!r})")
    # Use the same interpreter as pytest; keep Jupyter's transient state isolated.
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    client = NotebookClient(notebook, timeout=90, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    html = report_path.read_text()
    assert "Risk model available" in html
    rendered = [output.data["text/html"] for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.outputs if output.output_type in ("display_data", "execute_result")
        and "text/html" in output.data]
    assert rendered == [html]
    parser = AssetParser()
    parser.feed(html)
    assert len(parser.images) == 6
    assert not parser.external and not parser.scripts
