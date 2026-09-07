"""Independent checks of the CSV, declared-calendar, and holdings contracts."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal

from qis_risk.data import (
    DEMO_UNIVERSE,
    InputData,
    InputValidationError,
    build_holdings,
    complete_period_ends,
    daily_returns,
    load_inputs,
    make_demo_data,
    resolve_dates,
    validate_values,
    write_demo_csvs,
)


def example():
    dates = pd.bdate_range("2025-01-02", periods=3)
    return InputData(
        pd.DataFrame([[100, 100], [120, 90], [132, 81]], index=dates, columns=["A", "B"]),
        pd.DataFrame([[.5, .5], [.25, .75]], index=dates[:2], columns=["A", "B"]),
        ("A", "B"),
    )


def test_deterministic_rebalance_close_old_units_then_new_units():
    inputs = example()
    book = build_holdings(inputs)
    assert_allclose(book.value, [100, 105, 99.75])
    assert_allclose(book.units, [[.5, .5], [.21875, .875], [.21875, .875]])
    assert_allclose(book.weights.iloc[1], [.25, .75])
    assert_allclose(book.pre_rebalance_weights.iloc[1], [4 / 7, 3 / 7])
    assert_allclose(book.weights.iloc[2], [28.875 / 99.75, 70.875 / 99.75])
    assert_allclose(book.returns.iloc[1:], [.05, -.05])
    assert_allclose(book.weights.sum(axis=1), 1)
    strategy_returns = daily_returns(inputs)
    assert_allclose((book.weights.shift(1) * strategy_returns).sum(axis=1).iloc[1:], [.05, -.05])
    assert (book.reasons == "").all()
    assert_allclose(book.latest_targets.iloc[-1], [.25, .75])
    assert book.last_rebalance.iloc[-1] == inputs.targets.index[-1]


def test_index_rebasing_does_not_change_percentage_results():
    inputs = example()
    scaled = InputData(inputs.levels * [7.3, .05], inputs.targets, inputs.universe)
    original, rebased = build_holdings(inputs), build_holdings(scaled)
    assert_allclose(original.value, rebased.value)
    assert_allclose(original.weights, rebased.weights)
    assert_allclose(original.returns, rebased.returns, equal_nan=True)
    assert_allclose(original.units / [7.3, .05], rebased.units)
    assert_allclose(daily_returns(inputs), daily_returns(scaled), equal_nan=True)


@pytest.mark.parametrize("row", [[-.1, 1.1], [.25, np.nan], [.25, .5], [.25, np.inf], [25, 75]])
def test_invalid_targets_rejected_without_normalization(row):
    inputs = example()
    inputs.targets.iloc[0] = row
    with pytest.raises(InputValidationError, match="target"):
        build_holdings(inputs)


def test_target_tolerance_does_not_normalize():
    inputs = example()
    inputs.targets.iloc[0] = [.5, .5 + 1e-9]
    book = build_holdings(inputs)
    assert book.weights.iloc[0, 1] == .5 + 1e-9
    assert book.units.iloc[0, 1] == .5 + 1e-9
    inputs.targets.iloc[0] = [.5, .5 + 1e-7]
    with pytest.raises(InputValidationError, match="sum"):
        build_holdings(inputs)
    build_holdings(inputs, weight_sum_atol=1e-6)


@pytest.mark.parametrize("bad", [0, -1, np.inf, -np.inf, "oops"])
def test_bad_observed_levels_rejected(bad):
    inputs = example()
    inputs.levels = inputs.levels.astype(object)
    inputs.levels.iloc[-1, 0] = bad
    with pytest.raises(InputValidationError, match="levels|observations"):
        build_holdings(inputs)


def test_missing_level_does_not_bridge_returns_and_units_recover():
    dates = pd.bdate_range("2025-01-01", periods=5)
    levels = pd.DataFrame({"A": [100, 110, np.nan, 121, 133.1], "B": [100] * 5}, index=dates)
    targets = pd.DataFrame([[.5, .5]], columns=levels.columns, index=dates[:1])
    inputs = InputData(levels, targets, ("A", "B"))
    book, returns = build_holdings(inputs), daily_returns(inputs)
    assert np.isnan(book.value.iloc[2])
    assert book.value.iloc[3] == pytest.approx(110.5)
    assert book.weights.iloc[2].isna().all()
    assert_allclose(book.units, np.full((5, 2), .5))
    assert book.returns.iloc[2:4].isna().all()
    assert returns["A"].iloc[2:4].isna().all()
    assert returns["A"].iloc[4] == pytest.approx(.1)
    assert "missing" in book.reasons.iloc[2]
    assert book.reasons.iloc[3] == ""


def test_zero_allocation_does_not_relax_required_valuation():
    inputs = example()
    inputs.targets.iloc[:] = [1, 0]
    inputs.levels = inputs.levels.astype(float)
    inputs.levels.iloc[-1, 1] = np.nan
    assert np.isnan(build_holdings(inputs).value.iloc[-1])
    inputs.levels.iloc[1, 1] = np.nan
    with pytest.raises(InputValidationError, match="rebalance"):
        build_holdings(inputs)


def test_no_holdings_inferred_before_first_allocation():
    inputs = example()
    inputs.targets = inputs.targets.iloc[1:]
    book = build_holdings(inputs)
    assert book.value.iloc[0:1].isna().all()
    assert book.units.iloc[0].isna().all()
    assert pd.isna(book.last_rebalance.iloc[0])
    assert "before first" in book.reasons.iloc[0]
    assert book.value.iloc[1] == 100
    assert np.isnan(book.returns.iloc[1])


def test_future_numeric_errors_do_not_change_historical_calculations():
    original = example()
    levels, targets = original.levels.astype(object), original.targets.astype(object)
    levels.iloc[2] = ["bad future price", -1]
    targets.iloc[1] = ["bad future weight", np.inf]
    inputs = InputData(levels, targets, original.universe)
    cutoff = original.levels.index[0]
    expected, actual = build_holdings(original, cutoff), build_holdings(inputs, cutoff)
    assert_series_equal(expected.value, actual.value)
    assert_frame_equal(expected.weights, actual.weights)
    assert_frame_equal(daily_returns(original, cutoff), daily_returns(inputs, cutoff))
    validate_values(inputs, cutoff)
    with pytest.raises(InputValidationError):
        build_holdings(inputs)


@pytest.mark.parametrize("mutation", ["duplicate_date", "unsorted", "duplicate_columns", "unknown", "missing"])
def test_global_structural_validation(mutation):
    inputs = example()
    levels = inputs.levels.copy()
    if mutation == "duplicate_date":
        levels.index = [levels.index[0], levels.index[1], levels.index[1]]
    elif mutation == "unsorted":
        levels = levels.iloc[::-1]
    elif mutation == "duplicate_columns":
        levels.columns = ["A", "A"]
    elif mutation == "unknown":
        levels["X"] = 1
    else:
        levels = levels[["A"]]
    with pytest.raises(InputValidationError):
        InputData(levels, inputs.targets, inputs.universe)


def test_rebalances_outside_calendar_rejected_globally():
    inputs = example()
    targets = inputs.targets.copy()
    targets.index = [targets.index[0], pd.Timestamp("2026-01-01")]
    with pytest.raises(InputValidationError, match="outside"):
        InputData(inputs.levels, targets, inputs.universe)


def test_ordered_universe_reorders_columns_without_sorting_identifiers():
    inputs = example()
    reordered = InputData(inputs.levels, inputs.targets, ("B", "A"))
    assert tuple(reordered.levels.columns) == ("B", "A")
    assert tuple(reordered.targets.columns) == ("B", "A")
    with pytest.raises(InputValidationError, match="duplicate"):
        InputData(inputs.levels, inputs.targets, ("A", "A"))


def test_date_resolution_and_certified_calendar_boundaries():
    inputs = example()
    requested, resolved, cutoff = resolve_dates(inputs, "2025-01-05")
    assert requested == cutoff == pd.Timestamp("2025-01-05")
    assert resolved == pd.Timestamp("2025-01-03")
    with pytest.raises(InputValidationError, match="exceeds"):
        resolve_dates(inputs, "2025-01-07")
    with pytest.raises(InputValidationError, match="precedes"):
        resolve_dates(inputs, "2025-01-01")
    inputs = InputData(inputs.levels, inputs.targets, inputs.universe, "2025-01-31")
    requested, resolved, cutoff = resolve_dates(inputs)
    assert requested is None and resolved == inputs.levels.index[-1]
    assert cutoff == pd.Timestamp("2025-01-31")
    assert complete_period_ends(inputs.levels.index, cutoff).equals(inputs.levels.index[-1:])


def test_month_completion_requires_month_boundary_and_weeks_friday_boundary():
    dates = pd.bdate_range("2025-08-01", "2025-09-04")
    assert len(complete_period_ends(dates, pd.Timestamp("2025-08-29"))) == 0
    assert complete_period_ends(dates, pd.Timestamp("2025-08-31")).tolist() == [pd.Timestamp("2025-08-29")]
    assert complete_period_ends(dates, pd.Timestamp("2025-09-04"), "W-FRI")[-1] == pd.Timestamp("2025-08-29")
    assert complete_period_ends(dates, pd.Timestamp("2025-09-05"), "W-FRI")[-1] == pd.Timestamp("2025-09-04")
    with pytest.raises(InputValidationError, match="frequency"):
        complete_period_ends(dates, dates[-1], "D")


def test_empty_calendar_periods_are_not_fabricated():
    dates = pd.DatetimeIndex(["2025-01-03", "2025-01-17", "2025-03-31"])
    assert complete_period_ends(dates, dates[-1]).tolist() == [dates[1], dates[2]]
    assert complete_period_ends(dates, pd.Timestamp("2025-01-17"), "W-FRI").tolist() == list(dates[:2])


def test_csv_header_duplicates_and_date_column_rejected(tmp_path):
    inputs = example()
    weights_path = tmp_path / "weights.csv"
    inputs.targets.to_csv(weights_path)
    levels_path = tmp_path / "levels.csv"
    levels_path.write_text("date,A,A\n2025-01-02,100,100\n")
    with pytest.raises(InputValidationError, match="duplicate"):
        load_inputs(levels_path, weights_path, universe=inputs.universe)
    levels_path.write_text("wrong,A,B\n2025-01-02,100,100\n")
    with pytest.raises(InputValidationError, match="date column"):
        load_inputs(levels_path, weights_path, universe=inputs.universe)


def test_demo_is_reproducible_complete_and_csv_round_trip(tmp_path):
    demo = make_demo_data()
    assert demo.levels.index[0] == pd.Timestamp("2018-01-01")
    assert demo.levels.index[-1] == pd.Timestamp("2025-12-31")
    assert demo.levels.shape[1] == 10
    assert demo.universe == DEMO_UNIVERSE
    assert_frame_equal(demo.levels, make_demo_data().levels)
    assert_frame_equal(demo.targets, make_demo_data().targets)
    assert (demo.levels > 0).all().all()
    assert (demo.targets == 0).any().any()
    assert_allclose(demo.targets.sum(axis=1), 1)
    levels_path, weights_path = write_demo_csvs(tmp_path)
    loaded = load_inputs(levels_path, weights_path, universe=DEMO_UNIVERSE)
    assert_allclose(loaded.levels, demo.levels, rtol=1e-10)
    assert_allclose(loaded.targets, demo.targets, rtol=1e-10)
    assert len(loaded.provenance["levels_sha256"]) == 64
    assert build_holdings(loaded).reasons.eq("").all()
    assert len(make_demo_data(periods=5000).levels) == 5000
