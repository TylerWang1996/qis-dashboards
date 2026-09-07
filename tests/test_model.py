"""Independent numerical acceptance checks for the design's Appendices B–G."""
from dataclasses import replace
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from qis_risk.data import InputData, load_inputs, make_demo_data
from qis_risk.model import (
    ModelConfig,
    ModelValidationError,
    PortfolioConfig,
    Snapshot,
    attribute_change,
    estimate_history,
    risk_metrics,
    run_review,
    shrink_correlation,
    stress_metrics,
    validate_correlation,
    weekly_diagnostic,
)


def small_config(**kwargs):
    return ModelConfig(standardization_warmup_observations=3, min_history_observations=8, **kwargs)


def snapshot(x, s, r, date="2025-01-31", config=None):
    ids = [f"s{i}" for i in range(len(x))]
    risk = risk_metrics(x, s, r, config=config, names=ids)
    return Snapshot(pd.Timestamp(date), pd.Series(x, index=ids), pd.Series(s, index=ids),
        pd.DataFrame(r, index=ids, columns=ids), pd.DataFrame(r, index=ids, columns=ids),
        risk["metrics"], risk["strategies"], risk["pairs"], risk["diagnostics"])


def direct_estimate(frame, date, c, half_life):
    """Direct finite weighted sums, with independently constructed lagged z."""
    raw = frame.loc[:date].to_numpy()
    positions = np.arange(len(raw))
    complete = np.isfinite(raw).all(axis=1)
    weights = 2 ** (-(positions[-1] - positions[complete]) / c.vol_half_life)
    weights /= weights.sum()
    vol = np.sqrt(c.annualization * (weights[:, None] * raw[complete] ** 2).sum(axis=0))
    zs, zpositions = [], []
    for t in positions:
        prior = positions[(positions < t) & complete]
        if not complete[t] or len(prior) < c.standardization_warmup_observations:
            continue
        w = 2 ** (-((t - 1) - prior) / c.vol_half_life)
        w /= w.sum()
        variance = (w[:, None] * raw[prior] ** 2).sum(axis=0)
        if (variance <= c.annualized_volatility_epsilon**2 / c.annualization).any():
            continue
        zs.append(raw[t] / np.sqrt(variance))
        zpositions.append(t)
    z = np.asarray(zs)
    zw = 2 ** (-(positions[-1] - np.asarray(zpositions)) / half_life)
    zw /= zw.sum()
    q = (z.T * zw) @ z
    corr = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
    return vol, corr, 1 / (weights @ weights), 1 / (zw @ zw), len(z)


@pytest.mark.parametrize("kwargs", [
    {"vol_half_life": 0}, {"annualization": -2}, {"corr_half_life_slow": np.inf},
    {"standardization_warmup_observations": 0}, {"min_history_observations": 60},
    {"shrinkage_alpha": 1.1}, {"pair_stress_kappa": .75},
    {"volatility_stress_multipliers": (0,)}, {"mean_model": "sample"},
    {"allow_negative_weights": True}, {"target_weight_sum": .5},
    {"correlation_matrix_atol": 0}, {"half_life_sensitivity": ((84, 0),)},
])
def test_config_rejects_unsupported_and_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_resolved_standardization_minimum_and_portfolio_metadata():
    assert ModelConfig().min_standardized_observations == 444
    assert small_config().min_standardized_observations == 5
    with pytest.raises(ValueError):
        PortfolioConfig(return_basis="total_return")
    with pytest.raises(ValueError):
        PortfolioConfig(currency="")


def test_recursive_ewma_matches_direct_reference_with_calendar_gaps():
    rng = np.random.default_rng(901)
    frame = pd.DataFrame(rng.normal(0, .008, (37, 3)), index=pd.bdate_range("2020-01-01", periods=37))
    frame.iloc[[0, 8, 9, 15, 23], 1] = np.nan
    c = small_config(vol_half_life=5, corr_half_life_slow=7, corr_half_life_fast=2)
    estimates = estimate_history(frame, c)
    for date in frame.index[14:]:
        estimate = estimates[date]
        assert estimate.status == "valid"
        for half_life in (7, 2):
            vol, corr, vess, cess, zcount = direct_estimate(frame, date, c, half_life)
            assert_allclose(estimate.volatilities, vol, atol=1e-14)
            assert_allclose(estimate.correlations[half_life], corr, atol=1e-14)
            assert_allclose(estimate.diagnostics["volatility_ess"], vess)
            assert_allclose(estimate.diagnostics["correlation_ess"][half_life], cess)
            assert estimate.diagnostics["standardized_observations"] == zcount
    compressed = estimate_history(frame.dropna(), c)[frame.index[-1]]
    assert not np.allclose(compressed.correlations[7], estimates[frame.index[-1]].correlations[7])


def test_initialization_exact_counts_and_standardization_excludes_own_return():
    rng = np.random.default_rng(421)
    frame = pd.DataFrame(rng.normal(0, .01, (505, 2)), index=pd.bdate_range("2020-01-01", periods=505))
    frame.iloc[0] = np.nan
    estimates = estimate_history(frame)
    assert estimates[frame.index[-2]].status == "unavailable"
    last = estimates[frame.index[-1]]
    assert last.status == "valid"
    assert last.diagnostics["raw_observations"] == 504
    assert last.diagnostics["standardized_observations"] == 444
    frame.iloc[61, 0] = 1.2
    changed = estimate_history(frame)
    _, expected, *_ = direct_estimate(frame, frame.index[-1], ModelConfig(), 126)
    assert_allclose(changed[frame.index[-1]].correlations[126], expected, atol=1e-14)
    assert_allclose(changed[frame.index[60]].volatilities, estimates[frame.index[60]].volatilities)


def test_raw_history_does_not_replace_standardized_history_gate():
    frame = pd.DataFrame({"a": [0.] * 15 + [.01, .02, -.01, .03]}, index=pd.bdate_range("2020-01-01", periods=19))
    result = estimate_history(frame, small_config())[frame.index[-1]]
    assert result.diagnostics["raw_observations"] == 19
    assert result.diagnostics["standardized_observations"] == 3
    assert result.status == "unavailable"


@pytest.mark.parametrize("r", [np.eye(4), np.ones((4, 4)), np.full((4, 4), -.2) + np.eye(4)*1.2])
def test_current_risk_identities_and_signed_pair_aggregation(r):
    x, s = np.array([.2, .3, 0, .5]), np.array([.1, .15, .2, .07])
    out = risk_metrics(x, s, r)
    m, strategies, pairs = out["metrics"], out["strategies"], out["pairs"]
    a = x*s
    sigma = np.sqrt(a @ r @ a)
    assert_allclose(m["portfolio_volatility"], sigma)
    assert_allclose(strategies.euler_contribution.sum(), sigma)
    assert_allclose(m["self_contribution"] + m["cross_contribution"], sigma)
    assert_allclose(pairs.contribution.sum(), m["cross_contribution"], atol=1e-15)
    assert_allclose(m["diversification_ratio"], np.sqrt(m["effective_exposures"]) / m["correlation_multiplier"])
    assert strategies.loc["2", "euler_contribution"] == 0
    for name in strategies.index:
        selected = pairs.loc[pairs.strategy_i.eq(name) | pairs.strategy_j.eq(name)]
        assert_allclose(selected.contribution.sum() / 2, strategies.loc[name, "cross_contribution"], atol=1e-15)
    if (r[np.triu_indices(4, 1)] < 0).all():
        assert (pairs.contribution <= 0).all()
        assert m["correlation_multiplier"] < 1


def test_independence_perfect_correlation_single_and_zero_weights():
    x, s = [.5, .5], [.1, .2]
    independent = risk_metrics(x, s, np.eye(2))["metrics"]
    assert independent["correlation_uplift"] == 0
    assert independent["correlation_multiplier"] == 1
    assert independent["cross_contribution"] == 0
    perfect = risk_metrics(x, s, np.ones((2, 2)))["metrics"]
    assert_allclose(perfect["portfolio_volatility"], perfect["standalone_sum"])
    assert_allclose(perfect["diversification_ratio"], 1)
    for weights, vols, matrix in [([1], [.1], [[1]]), ([1, 0], [.1, .2], np.eye(2))]:
        result = risk_metrics(weights, vols, matrix)
        for key in ("diversification_ratio", "effective_exposures", "correlation_multiplier"):
            assert result["metrics"][key] == 1
        assert result["metrics"]["cross_contribution"] == 0
    assert risk_metrics([1], [.1], [[1]])["pairs"].empty


def test_negligible_volatility_retains_valid_counterfactuals():
    out = risk_metrics([.5, .5], [.2, .2], [[1, -1], [-1, 1]])
    assert out["metrics"]["portfolio_volatility"] == 0
    for key in ("diversification_ratio", "self_contribution", "cross_contribution"):
        assert out["metrics"][key] is None
    assert out["metrics"]["correlation_multiplier"] == 0
    assert out["metrics"]["effective_exposures"] == 2
    assert out["pairs"].contribution.isna().all()
    assert out["strategies"].euler_contribution.isna().all()


def test_allowed_roundoff_clamp_is_recorded_and_material_invalid_matrix_rejected():
    r = np.array([[1, -1-1e-11], [-1-1e-11, 1]])
    result = risk_metrics([.5, .5], [1, 1], r)
    assert result["metrics"]["portfolio_volatility"] == 0
    assert result["diagnostics"]["adjustments"]
    with pytest.raises(ModelValidationError, match="not PSD"):
        risk_metrics([.5, .5], [.1, .1], [[1, -1.01], [-1.01, 1]])
    with pytest.raises(ModelValidationError, match="unit-diagonal"):
        risk_metrics([1], [.1], [[2]])


def test_shapley_matches_independent_coalition_formula_and_pair_bar():
    x0, x1 = np.array([.2, .3, .5]), np.array([.5, .1, .4])
    s0, s1 = np.array([.15, .2, .1]), np.array([.11, .17, .2])
    r0 = np.array([[1, .2, -.1], [.2, 1, .4], [-.1, .4, 1]])
    r1 = np.array([[1, .5, -.2], [.5, 1, .1], [-.2, .1, 1]])
    old, new = snapshot(x0, s0, r0), snapshot(x1, s1, r1, "2025-02-28")
    result = attribute_change(old, new)

    def value(coalition):
        a = (x1 if 0 in coalition else x0) * (s1 if 1 in coalition else s0)
        return a @ (r1 if 2 in coalition else r0) @ a

    expected = []
    for block in range(3):
        others = set(range(3)) - {block}
        effect = 0
        for size in range(3):
            for subset in combinations(others, size):
                coalition = set(subset)
                effect += factorial(size)*factorial(2-size)/factorial(3) * (value(coalition | {block}) - value(coalition))
        expected.append(effect)
    assert_allclose(list(result["variance_effects"].values()), expected, atol=1e-15)
    assert_allclose(sum(result["effects"].values()), new.metrics["portfolio_volatility"] - old.metrics["portfolio_volatility"])
    assert_allclose(result["pairs"].effect.sum(), result["effects"]["correlation"])
    assert_allclose(result["strategies"].effect.sum(), result["effects"]["correlation"])
    assert not np.allclose(new.pairs.contribution-old.pairs.contribution, result["pairs"].effect)


def test_shapley_one_zero_endpoint_and_both_zero_denominator():
    zero = snapshot([.5, .5], [.2, .2], [[1, -1], [-1, 1]])
    positive = snapshot([.5, .5], [.2, .2], np.eye(2))
    result = attribute_change(zero, positive)
    assert result["status"] == "valid"
    assert_allclose(sum(result["effects"].values()), positive.metrics["portfolio_volatility"])
    unavailable = attribute_change(zero, zero)
    assert unavailable["status"] == "unavailable"
    assert all(v is None for v in unavailable["effects"].values())
    assert all(v == 0 for v in unavailable["variance_effects"].values())


def test_stress_monotonicity_pair_reconciliation_uniform_scaling_and_limits():
    current = snapshot([.2, .3, .5], [.1, .2, .15], np.array([[1, -.1, .2], [-.1, 1, .3], [.2, .3, 1]]))
    c = replace(ModelConfig(), correlation_stress_kappa=(0, .25, .5, 1))
    result = stress_metrics(current, c)
    scenarios = result["scenarios"].set_index("kappa")
    assert (scenarios.volatility.diff().dropna() >= 0).all()
    assert_allclose(scenarios.loc[1, "volatility"], current.metrics["standalone_sum"])
    assert_allclose(scenarios.loc[1, "diversification_ratio"], 1)
    assert (result["pairs"].contribution >= 0).all()
    assert_allclose(result["pairs"].contribution.sum(), scenarios.loc[.25, "uplift"])
    for row in result["grid"].itertuples():
        assert_allclose(row.volatility, row.multiplier*scenarios.loc[row.kappa, "volatility"])
        assert_allclose(row.uplift, row.volatility-current.metrics["portfolio_volatility"])
        assert_allclose(row.diversification_ratio, scenarios.loc[row.kappa, "diversification_ratio"])
    scaled = risk_metrics(current.weights, 5*current.volatilities, current.correlation)["metrics"]
    for key in ("diversification_ratio", "effective_exposures", "correlation_multiplier"):
        assert_allclose(scaled[key], current.metrics[key])


def test_zero_baseline_stress_conversion_and_perfect_correlation():
    zero = snapshot([.5, .5], [.2, .2], [[1, -1], [-1, 1]])
    result = stress_metrics(zero)
    assert result["scenarios"].percentage_increase.isna().all()
    assert result["pair_status"] == "valid"
    assert_allclose(result["pairs"].contribution.sum(), result["scenarios"].set_index("kappa").loc[.25, "uplift"])
    perfect = snapshot([.5, .5], [.2, .2], np.ones((2, 2)))
    assert_allclose(stress_metrics(perfect)["scenarios"].uplift, 0, atol=1e-16)
    no_risk = snapshot([1], [0], [[1]])
    assert stress_metrics(no_risk)["pair_status"] == "unavailable"


def test_shrinkage_target_remains_valid_and_single_strategy_works():
    r = np.array([[1, .2, -.1], [.2, 1, .4], [-.1, .4, 1]])
    for alpha in (0, .1, .25, 1):
        shrunk = shrink_correlation(r, alpha)
        validate_correlation(shrunk)
        assert_allclose(shrunk[np.triu_indices(3, 1)].mean(), r[np.triu_indices(3, 1)].mean())
    assert_allclose(shrink_correlation(np.ones((1, 1)), .25), [[1]])


@pytest.fixture(scope="module")
def demo_review():
    inputs = make_demo_data()
    return inputs, run_review(inputs)


def test_end_to_end_history_attribution_and_diagnostic_provenance(demo_review):
    inputs, result = demo_review
    assert result.status == "valid", result.reasons
    assert result.current.date == inputs.levels.index[-1]
    assert result.previous.date.to_period("M") == result.current.date.to_period("M") - 1
    assert (result.history.status == "valid").all()
    assert result.diagnostics["dr_percentile"]["status"] == "valid"
    percentile = result.diagnostics["dr_percentile"]
    dates = percentile["reference_dates"]
    assert result.current.date not in dates
    ref = result.history.loc[dates, "diversification_ratio"]
    current = result.current.metrics["diversification_ratio"]
    expected = 100*((ref < current).sum() + .5*(ref == current).sum())/len(ref)
    assert_allclose(percentile["value"], expected)
    assert len(result.diagnostics["shrinkage_sensitivity"]) == 3
    assert len(result.diagnostics["half_life_sensitivity"]) == 3
    assert result.diagnostics["weekly"].status.eq("valid").all()
    assert set(result.diagnostics["sensitivity_pairs"].columns) >= {"rank", "baseline_rank", "rank_change", "sign_changed"}
    assert "min_standardized_observations" in result.metadata["model_config"]


def test_asof_is_invariant_to_future_prices_targets_and_invalid_numerical_values(demo_review):
    inputs, _ = demo_review
    cutoff = inputs.levels.index[1100]
    reference = run_review(inputs, as_of_date=cutoff)
    levels, targets = inputs.levels.copy(), inputs.targets.copy()
    levels.loc[levels.index > cutoff] = -100
    targets.loc[targets.index > cutoff] = np.nan
    changed = InputData(levels, targets, inputs.universe, inputs.calendar_complete_through, inputs.provenance)
    actual = run_review(changed, as_of_date=cutoff)
    assert actual.status == "valid", actual.reasons
    assert reference.current.metrics == actual.current.metrics
    assert_frame_equal(reference.history, actual.history)
    assert_frame_equal(reference.attribution["pairs"], actual.attribution["pairs"])


def test_missing_current_and_comparison_have_local_availability_effects(demo_review):
    inputs, reference = demo_review
    levels = inputs.levels.copy()
    comparison = reference.comparison_date
    targets = inputs.targets.drop([comparison, inputs.levels.index[-1]])
    levels.loc[comparison, inputs.universe[0]] = np.nan
    missing_previous = run_review(InputData(levels, targets, inputs.universe, inputs.calendar_complete_through, inputs.provenance))
    assert missing_previous.status == "valid"
    assert missing_previous.previous is None
    assert missing_previous.attribution["status"] == "unavailable"
    assert missing_previous.history.loc[comparison, "status"] == "unavailable"
    levels = inputs.levels.copy()
    levels.loc[levels.index[-1], inputs.universe[0]] = np.nan
    missing_current = run_review(InputData(levels, targets, inputs.universe, inputs.calendar_complete_through, inputs.provenance))
    assert missing_current.status == "unavailable"
    assert missing_current.current is None


def test_invalid_current_input_blocks_valid_pm_report(demo_review):
    inputs, _ = demo_review
    levels = inputs.levels.copy()
    levels.iloc[-1, 0] = -1
    result = run_review(InputData(levels, inputs.targets, inputs.universe, inputs.calendar_complete_through, inputs.provenance))
    assert result.status == "failed"
    assert result.current is None


def test_incomplete_month_is_separate_and_short_history_unavailable(demo_review):
    inputs, _ = demo_review
    result = run_review(inputs, as_of_date="2025-07-16")
    assert result.status == "valid"
    assert not result.history.loc[result.resolved_date, "completed_month"]
    assert result.comparison_date == pd.Timestamp("2025-06-30")
    limited = run_review(inputs, as_of_date=inputs.levels.index[100])
    assert limited.status == "unavailable"
    assert limited.history.empty
    assert limited.diagnostics["dr_percentile"]["status"] == "unavailable"


def test_weekly_diagnostic_uses_paired_samples_and_no_gap_compression():
    dates = pd.bdate_range("2023-01-02", "2023-05-31")
    rng = np.random.default_rng(6)
    levels = pd.DataFrame(100*np.cumprod(1+rng.normal(0, .01, (len(dates), 1)), axis=0), index=dates, columns=["a"])
    levels.loc["2023-03-15", "a"] = np.nan
    config = small_config(weekly_min_intervals=4)
    out = weekly_diagnostic(levels, pd.Timestamp("2023-05-31"), config).loc["a"]
    assert out.status == "valid"
    assert out.last_interval_end == pd.Timestamp("2023-05-26")
    raw = levels.a / levels.a.shift(1) - 1
    endpoint_dates = pd.bdate_range("2023-01-06", "2023-05-26", freq="W-FRI")
    weekly = pd.Series(np.nan, index=endpoint_dates)
    paired_daily = pd.Series(np.nan, index=dates)
    for start, end in zip(endpoint_dates[:-1], endpoint_dates[1:], strict=True):
        chain = levels.loc[start:end, "a"]
        if chain.isna().any():
            continue
        weekly.loc[end] = chain.iloc[-1] / chain.iloc[0] - 1
        mask = (dates > start) & (dates <= end)
        paired_daily.loc[dates[mask]] = raw.loc[dates[mask]]
    assert out.weekly_intervals == weekly.notna().sum()
    assert out.daily_observations == paired_daily.notna().sum()
    assert_allclose(out.daily_volatility, np.sqrt(252*(paired_daily.dropna()**2).mean()))
    assert_allclose(out.weekly_volatility, np.sqrt(52*(weekly.dropna()**2).mean()))
    assert_allclose(out.daily_autocorrelation, paired_daily.corr(paired_daily.shift(1)))
    assert_allclose(out.weekly_autocorrelation, weekly.corr(weekly.shift(1)))
    assert not np.isclose(out.weekly_autocorrelation, weekly.dropna().autocorr())


def test_weekly_empty_week_and_uncertified_friday_are_excluded():
    dates = pd.bdate_range("2023-01-02", "2023-03-31")
    levels = pd.DataFrame({"a": 100*1.001**np.arange(len(dates))}, index=dates)
    levels = levels.drop(pd.bdate_range("2023-02-06", "2023-02-10"))
    c = small_config(weekly_min_intervals=2)
    out = weekly_diagnostic(levels, "2023-03-30", c).loc["a"]
    assert out.last_interval_end == pd.Timestamp("2023-03-24")
    # Initial missing prior endpoint, omitted whole week and the following
    # interval whose preceding endpoint is missing are all explicit exclusions.
    assert out.excluded_intervals == 3


def test_future_nonnumeric_csv_values_do_not_poison_past_numeric_dtypes(tmp_path):
    inputs = make_demo_data(periods=90)
    cutoff = inputs.levels.index[65]
    reference = run_review(inputs, model=small_config(), as_of_date=cutoff)
    levels = inputs.levels.astype(object)
    levels.loc[levels.index > cutoff, inputs.universe[0]] = "future bad value"
    target = inputs.targets.astype(object)
    target.loc[target.index > cutoff, inputs.universe[0]] = "future bad target"
    levels_path, target_path = tmp_path / "levels.csv", tmp_path / "weights.csv"
    levels.to_csv(levels_path)
    target.to_csv(target_path)
    loaded = load_inputs(levels_path, target_path, universe=inputs.universe)
    result = run_review(loaded, model=small_config(), as_of_date=cutoff)
    assert result.status == "valid", result.reasons
    assert_allclose(result.current.volatilities, reference.current.volatilities)
    assert_allclose(result.current.correlation, reference.current.correlation)
    assert_allclose(result.current.weights, reference.current.weights)


def test_empty_calendar_month_is_explicit_unavailable_history_gap():
    inputs = make_demo_data(periods=700)
    missing_period = pd.Period("2019-09", freq="M")
    levels = inputs.levels.loc[inputs.levels.index.to_period("M") != missing_period]
    targets = inputs.targets.loc[inputs.targets.index.to_period("M") != missing_period]
    result = run_review(InputData(levels, targets, inputs.universe), model=small_config())
    assert result.status == "valid", result.reasons
    gap = result.history.loc["2019-09-30"]
    assert gap.status == "unavailable"
    assert pd.isna(gap.portfolio_volatility)
    assert "No declared trading close" in gap.reason


def test_weekly_samples_are_common_across_universe_when_one_strategy_is_missing():
    dates = pd.bdate_range("2023-01-02", "2023-05-31")
    rng = np.random.default_rng(7)
    levels = pd.DataFrame(100*np.cumprod(1+rng.normal(0, .01, (len(dates), 2)), axis=0), index=dates, columns=["a", "b"])
    levels.loc["2023-03-15", "a"] = np.nan
    c = small_config(weekly_min_intervals=4)
    result = weekly_diagnostic(levels, "2023-05-31", c)
    assert result.weekly_intervals.nunique() == 1
    assert result.daily_observations.nunique() == 1
    b_alone = weekly_diagnostic(levels[["b"]], "2023-05-31", c)
    assert result.loc["b", "weekly_intervals"] == b_alone.loc["b", "weekly_intervals"] - 1


def test_invalid_asof_is_diagnostic_and_outside_coverage_is_unavailable():
    inputs = make_demo_data(periods=30)
    invalid = run_review(inputs, as_of_date="not a date")
    assert invalid.status == "failed"
    assert "valid date" in invalid.reasons[0]
    outside = run_review(inputs, as_of_date=inputs.levels.index[-1] + pd.Timedelta(days=10))
    assert outside.status == "unavailable"
    assert "calendar_complete_through" in outside.reasons[0]
    assert PortfolioConfig().live_backtested == "unknown"


def test_degenerate_lag_exclusions_record_dates_and_current_degeneracy_fails():
    dates = pd.bdate_range("2020-01-01", periods=12)
    frame = pd.DataFrame({"a": [0.] * 9 + [.1, .2, .1]}, index=dates)
    estimates = estimate_history(frame, small_config())
    assert estimates[dates[8]].status == "failed"
    assert estimates[dates[-1]].diagnostics["invalid_lag_variance_dates"] == list(dates[3:10])
    assert estimates[dates[-1]].status == "unavailable"


def test_index_rebasing_preserves_complete_review_risk():
    inputs = make_demo_data(periods=100)
    c = small_config()
    first = run_review(inputs, model=c)
    scales = np.arange(1, 11)*17
    rebased = InputData(inputs.levels*scales, inputs.targets, inputs.universe)
    second = run_review(rebased, model=c)
    assert first.status == second.status == "valid"
    assert_allclose(list(first.current.metrics.values()), list(second.current.metrics.values()), atol=1e-14)
    assert_allclose(first.attribution["pairs"].effect, second.attribution["pairs"].effect, atol=1e-14)


def test_single_universe_full_pipeline_and_percentile_gate():
    inputs = make_demo_data(periods=100)
    strategy = inputs.universe[0]
    levels = inputs.levels[[strategy]]
    targets = pd.DataFrame(1., index=inputs.targets.index, columns=[strategy])
    result = run_review(InputData(levels, targets, (strategy,)), model=small_config())
    assert result.status == "valid", result.reasons
    assert result.current.pairs.empty
    assert result.attribution["pairs"].empty
    assert result.stress["pairs"].empty
    assert result.diagnostics["dr_percentile"]["count"] < 24
    assert result.diagnostics["dr_percentile"]["status"] == "unavailable"
    assert result.current.metrics["diversification_ratio"] == 1


@pytest.mark.parametrize("kwargs", [
    {"correlation_stress_kappa": (0, .25, .25)},
    {"volatility_stress_multipliers": (1, 1.25, 1.25)},
    {"shrinkage_alpha": True}, {"pair_stress_kappa": False},
    {"correlation_stress_kappa": (False, .25, .5)},
    {"volatility_stress_multipliers": (True, 1.25)},
    {"vol_half_life": "60"}, {"reconciliation_atol": False},
    {"target_weight_sum": True}, {"allow_negative_weights": 0},
    {"half_life_sensitivity": ((True, 42),)},
    {"half_life_sensitivity": (126,)},
    {"correlation_stress_kappa": None},
])
def test_config_rejects_duplicate_scenarios_booleans_and_malformed_numerics(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_percentile_uses_complete_month_buckets_including_weekend_and_holiday_ends(demo_review):
    inputs, result = demo_review
    percentile = result.diagnostics["dr_percentile"]
    assert result.current.date == pd.Timestamp("2025-12-31")
    assert percentile["count"] == 36
    assert percentile["reference_start"] == pd.Timestamp("2022-12-30")
    assert percentile["reference_period_start"] == "2022-12"
    assert percentile["reference_period_end"] == "2025-11"
    # Simulate an upstream-declared holiday by removing the final December
    # business close. The earlier certified monthly close stays in the window.
    holiday = pd.Timestamp("2022-12-30")
    levels = inputs.levels.drop(holiday)
    targets = inputs.targets.drop(holiday)
    shifted = run_review(InputData(levels, targets, inputs.universe, inputs.calendar_complete_through))
    assert shifted.status == "valid", shifted.reasons
    adjusted = shifted.diagnostics["dr_percentile"]
    assert adjusted["count"] == 36
    assert adjusted["reference_start"] == pd.Timestamp("2022-12-29")
    assert all(date.to_period("M") < shifted.current.date.to_period("M") for date in adjusted["reference_dates"])


def test_zero_portfolio_volatility_never_publishes_undefined_sensitivity_ranks():
    dates = pd.bdate_range("2020-01-01", periods=800)
    raw = .01*np.where(np.arange(799) % 2, 1, -1)
    levels = pd.DataFrame(
        np.vstack([np.full(2, 100.),
                   100*np.cumprod(np.column_stack((1+raw, 1-raw)), axis=0)]),
        index=dates, columns=["A", "B"])
    targets = pd.DataFrame(.5, index=dates[[0, -1]], columns=["A", "B"])
    result = run_review(InputData(levels, targets, ("A", "B")))
    assert result.status == "valid", result.reasons
    assert result.current.metrics["portfolio_volatility"] == 0
    pairs = result.diagnostics["sensitivity_pairs"]
    assert pairs.contribution.isna().all()
    assert pairs[["rank", "baseline_rank", "rank_change", "sign_changed"]].isna().all().all()
    assert pairs.status.eq("unavailable").all()
    assert pairs.reason.str.contains("undefined").all()
