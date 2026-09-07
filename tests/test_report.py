"""Presentation contracts and a real fresh-kernel notebook acceptance run."""

import ast
import base64
import copy
import os
import sys
from html.parser import HTMLParser
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from nbclient import NotebookClient

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
    assert len(parser.images) == 5
    assert all(im.get("alt") for im in parser.images)
    for image in parser.images:
        assert image["src"].startswith("data:image/svg+xml;base64,")
        svg = base64.b64decode(image["src"].split(",", 1)[1]).decode()
        assert "<svg" in svg
        assert "<script" not in svg
    for text in [
        "How diversified is the portfolio today?", "Why did risk change?",
        "Which relationships matter?", "Where could diversification disappear?",
        "Closing holdings", "Latest rebalance", "Other pairs",
        "Illustrative sensitivities without assigned probabilities",
        "variance-Shapley attribution expressed in volatility units",
        "same current baseline", "Entirely omitted trading dates cannot be detected",
    ]:
        assert text in html
    charts = "\n".join(base64.b64decode(im["src"].split(",", 1)[1]).decode() for im in parser.images)
    assert "Current / partial month" in charts
    path = export_html(html, tmp_path / "report.html")
    assert path.read_text() == html
    assert str(path).endswith("report.html")


def test_display_changes_do_not_mutate_analytical_outputs(review):
    before = copy.deepcopy(review)
    html = render_report(review, DisplayConfig(history_months=6, top_n_pairs=2, show_heatmap=False))
    parser = AssetParser()
    parser.feed(html)
    assert len(parser.images) == 4
    pd.testing.assert_frame_equal(review.history, before.history)
    pd.testing.assert_frame_equal(review.current.pairs, before.current.pairs)
    pd.testing.assert_frame_equal(review.attribution["pairs"], before.attribution["pairs"])
    assert review.current.metrics == before.current.metrics
    assert review.metadata == before.metadata


@pytest.mark.parametrize("status", ["unavailable", "failed"])
def test_failure_reports_do_not_publish_headline_estimates(status):
    result = ReviewResult(status, ["Required levels are missing"], None, None, None,
        diagnostics={"coverage": {"common_returns": 100}})
    html = render_report(result)
    assert "Required levels are missing" in html
    assert "Coverage and availability" in html
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
    assert len(parser.images) == 5
    assert not parser.external and not parser.scripts
