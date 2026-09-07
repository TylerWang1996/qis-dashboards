"""Finite-history QIS risk analytics; all numerical work lives outside the notebook.

Volatility quantities and contributions are decimal annualized units. Undefined
ratios are None, never an epsilon-substituted estimate. The input calendar is the
EWMA clock, including rows excluded from the common observation sample.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import permutations
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    InputData,
    InputValidationError,
    build_holdings,
    complete_period_ends,
    daily_returns,
    resolve_dates,
    validate_values,
)


@dataclass(frozen=True)
class PortfolioConfig:
    portfolio_id: str = "QIS demonstration"
    currency: str = "USD"
    return_basis: str = "excess_return"
    valuation_close: str = "common input close"
    version: str = "1"
    live_backtested: str = "unknown"
    data_vintage: str = "unknown"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.return_basis != "excess_return":
            raise ValueError("Only excess_return is supported")


def _finite_number(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


@dataclass(frozen=True)
class ModelConfig:
    portfolio_value_basis: str = "synthetic_excess_return"
    holdings_method: str = "constant_units_between_rebalances"
    rebalance_timing: str = "close"
    target_weight_sum: float = 1.0
    allow_negative_weights: bool = False
    vol_half_life: float = 60.0
    annualization: float = 252.0
    mean_model: str = "zero"
    standardization: str = "lagged_ewma_daily_volatility"
    standardization_warmup_observations: int = 60
    corr_half_life_slow: float = 126.0
    corr_half_life_fast: float = 42.0
    ewma_weighting: str = "normalized_finite_history"
    ewma_decay_clock: str = "declared_trading_calendar"
    shrinkage_alpha: float = 0.0
    shrinkage_target: str = "constant_correlation"
    min_history_observations: int = 504
    missing_data_policy: str = "common_complete_daily_return_vectors"
    change_attribution: str = "variance_shapley"
    rho_reference: str = "fixed_equal_standalone_risk"
    correlation_stress_kappa: tuple[float, ...] = (0.0, 0.25, 0.5)
    volatility_stress_multipliers: tuple[float, ...] = (1.0, 1.25, 1.5)
    pair_stress_kappa: float = 0.25
    percentile_history_months: int = 36
    percentile_min_prior_snapshots: int = 24
    weight_sum_atol: float = 1e-8
    reconciliation_rtol: float = 1e-8
    reconciliation_atol: float = 1e-12
    correlation_matrix_atol: float = 1e-10
    annualized_volatility_epsilon: float = 1e-10
    shrinkage_sensitivity: tuple[float, ...] = (0.0, 0.10, 0.25)
    half_life_sensitivity: tuple[tuple[float, float], ...] = (
        (84.0, 28.0), (126.0, 42.0), (189.0, 63.0)
    )
    weekly_history_months: int = 36
    weekly_min_intervals: int = 104
    weekly_annualization: float = 52.0
    model_version: str = "1.0"

    @property
    def min_standardized_observations(self) -> int:
        return self.min_history_observations - self.standardization_warmup_observations

    def __post_init__(self):
        modes = {
            "portfolio_value_basis": "synthetic_excess_return",
            "holdings_method": "constant_units_between_rebalances",
            "rebalance_timing": "close", "target_weight_sum": 1.0,
            "allow_negative_weights": False, "mean_model": "zero",
            "standardization": "lagged_ewma_daily_volatility",
            "ewma_weighting": "normalized_finite_history",
            "ewma_decay_clock": "declared_trading_calendar",
            "shrinkage_target": "constant_correlation",
            "missing_data_policy": "common_complete_daily_return_vectors",
            "change_attribution": "variance_shapley",
            "rho_reference": "fixed_equal_standalone_risk",
        }
        for name, expected in modes.items():
            if getattr(self, name) != expected or (
                isinstance(expected, bool) and not isinstance(getattr(self, name), bool)
            ):
                raise ValueError(f"Unsupported {name}; expected {expected!r}")
        if not _finite_number(self.target_weight_sum):
            raise ValueError("target_weight_sum must be a finite number")
        for name in ("correlation_stress_kappa", "volatility_stress_multipliers",
                     "shrinkage_sensitivity", "half_life_sensitivity"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or not value:
                raise ValueError(f"{name} must be a nonempty sequence")
        positive = (
            "vol_half_life", "annualization", "corr_half_life_slow",
            "corr_half_life_fast", "annualized_volatility_epsilon",
            "weekly_annualization", "correlation_matrix_atol",
        )
        for name in positive:
            value = getattr(self, name)
            if not _finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_sum_atol", "reconciliation_rtol", "reconciliation_atol"):
            if not _finite_number(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "standardization_warmup_observations", "min_history_observations",
            "percentile_history_months", "percentile_min_prior_snapshots",
            "weekly_history_months", "weekly_min_intervals",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_history_observations <= self.standardization_warmup_observations:
            raise ValueError("Minimum history must exceed standardization warmup")
        fractions = (
            self.shrinkage_alpha, self.pair_stress_kappa,
            *self.correlation_stress_kappa, *self.shrinkage_sensitivity,
        )
        if any(not _finite_number(v) or not 0 <= v <= 1 for v in fractions):
            raise ValueError("Shrinkage and convergence fractions must be in [0, 1]")
        if not self.correlation_stress_kappa or self.pair_stress_kappa not in self.correlation_stress_kappa:
            raise ValueError("Configured scenarios must include the designated pair stress")
        if not self.volatility_stress_multipliers or any(
            not _finite_number(v) or v <= 0 for v in self.volatility_stress_multipliers
        ):
            raise ValueError("Volatility multipliers must be finite and positive")
        for name in ("correlation_stress_kappa", "volatility_stress_multipliers"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values")
        if not self.shrinkage_sensitivity or not self.half_life_sensitivity:
            raise ValueError("Sensitivity settings must not be empty")
        for pair in self.half_life_sensitivity:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(not _finite_number(v) or v <= 0 for v in pair):
                raise ValueError("Half-life sensitivities require positive slow/fast pairs")
        if not isinstance(self.model_version, str) or not self.model_version:
            raise ValueError("model_version must be nonempty")


@dataclass
class Estimate:
    date: pd.Timestamp
    status: str
    reason: str
    volatilities: np.ndarray
    correlations: dict[float, np.ndarray]
    diagnostics: dict[str, Any]


@dataclass
class Snapshot:
    date: pd.Timestamp
    weights: pd.Series
    volatilities: pd.Series
    correlation: pd.DataFrame
    fast_correlation: pd.DataFrame
    metrics: dict[str, float | None]
    strategies: pd.DataFrame
    pairs: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class ReviewResult:
    status: str
    reasons: list[str]
    requested_date: pd.Timestamp | None
    resolved_date: pd.Timestamp | None
    comparison_date: pd.Timestamp | None
    current: Snapshot | None = None
    previous: Snapshot | None = None
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    attribution: dict = field(default_factory=dict)
    stress: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class ModelValidationError(ValueError):
    """A materially invalid matrix, exposure, or reconciliation blocks publication."""


def validate_correlation(matrix: np.ndarray, config: ModelConfig | None = None) -> dict:
    config = config or ModelConfig()
    r = np.asarray(matrix, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not len(r) or not np.isfinite(r).all():
        raise ModelValidationError("Correlation matrix must be finite, nonempty, and square")
    symmetry = float(np.max(np.abs(r - r.T)))
    diagonal = float(np.max(np.abs(np.diag(r) - 1)))
    minimum = float(np.linalg.eigvalsh((r + r.T) / 2)[0])
    if symmetry > config.correlation_matrix_atol or diagonal > config.correlation_matrix_atol:
        raise ModelValidationError("Correlation matrix fails symmetry or unit-diagonal validation")
    if minimum < -config.correlation_matrix_atol:
        raise ModelValidationError(f"Correlation matrix is not PSD: minimum eigenvalue {minimum:g}")
    return {"symmetry_error": symmetry, "diagonal_error": diagonal, "minimum_eigenvalue": minimum}


def _quadratic(a: np.ndarray, r: np.ndarray, config: ModelConfig, adjustments: list) -> float:
    # Callers validate the matrix first. This tolerance includes the allowed PSD
    # error and a dimension-aware bound on floating-point summation error.
    value = float(a @ r @ a)
    tolerance = (
        config.correlation_matrix_atol * float(a @ a)
        + 64 * np.finfo(float).eps * float(np.abs(a).sum()) ** 2
    )
    if not np.isfinite(value) or value < -tolerance:
        raise ModelValidationError(f"Invalid portfolio quadratic form: {value}")
    if value < 0:
        adjustments.append({"kind": "negative_quadratic_clamped", "original": value, "tolerance": tolerance})
        return 0.0
    return value


def _reconcile(lhs: float, rhs: float, name: str, config: ModelConfig) -> dict:
    residual = float(lhs - rhs)
    tolerance = config.reconciliation_atol + config.reconciliation_rtol * max(abs(lhs), abs(rhs))
    if not np.isfinite(residual) or abs(residual) > tolerance:
        raise ModelValidationError(f"{name} failed reconciliation: residual {residual:g}")
    return {"residual": residual, "tolerance": tolerance, "passed": True}


def shrink_correlation(r: np.ndarray, alpha: float) -> np.ndarray:
    """Constant-correlation shrinkage; does not silently repair an invalid input."""
    n = len(r)
    mean = float(r[np.triu_indices(n, 1)].mean()) if n > 1 else 0.0
    target = np.full((n, n), mean)
    np.fill_diagonal(target, 1)
    return (1 - alpha) * r + alpha * target


def estimate_history(
    returns: pd.DataFrame,
    config: ModelConfig | None = None,
    dates: pd.DatetimeIndex | list | None = None,
    half_lives: tuple[float, ...] | None = None,
) -> dict[pd.Timestamp, Estimate]:
    """One chronological pass; store moments only at selected requested closes.

    Each accumulator has a numerator, finite normalizer, and squared-weight
    normalizer. All three decay on missing rows. Standardization precedes the
    current observation's volatility update, preventing contemporaneous leakage.
    Correlations returned here are raw; shrinkage is applied at snapshot assembly.
    """
    c = config or ModelConfig()
    selected = set(returns.index if dates is None else pd.DatetimeIndex(dates))
    hs = tuple(dict.fromkeys(half_lives or (c.corr_half_life_slow, c.corr_half_life_fast)))
    if any(not np.isfinite(h) or h <= 0 for h in hs):
        raise ValueError("Correlation half-lives must be finite and positive")
    arr = returns.to_numpy(dtype=float)
    if np.isinf(arr).any():
        raise ModelValidationError("Returns contain infinite values")
    n = arr.shape[1]
    lv = 2.0 ** (-1 / c.vol_half_life)
    lc = np.asarray([2.0 ** (-1 / h) for h in hs])
    vnum, qnum = np.zeros(n), np.zeros((len(hs), n, n))
    vden = vden2 = 0.0
    qden, qden2 = np.zeros(len(hs)), np.zeros(len(hs))
    raw_count = z_count = 0
    raw_first = raw_last = z_first = z_last = None
    excluded_lag_dates = []
    outputs = {}
    daily_epsilon2 = c.annualized_volatility_epsilon ** 2 / c.annualization
    for date, row in zip(returns.index, arr, strict=True):
        valid = bool(np.isfinite(row).all())
        prior_variance = vnum / vden if vden > 0 else np.zeros(n)
        z_valid = valid and raw_count >= c.standardization_warmup_observations and bool(
            np.all(prior_variance > daily_epsilon2)
        )
        z = row / np.sqrt(prior_variance) if z_valid else None
        if valid and raw_count >= c.standardization_warmup_observations and not z_valid:
            excluded_lag_dates.append(date)
        vnum *= lv
        vden *= lv
        vden2 *= lv * lv
        qnum *= lc[:, None, None]
        qden *= lc
        qden2 *= lc * lc
        if valid:
            vnum += row * row
            vden += 1
            vden2 += 1
            raw_count += 1
            raw_first = date if raw_first is None else raw_first
            raw_last = date
        if z_valid:
            qnum += np.outer(z, z)[None, :, :]
            qden += 1
            qden2 += 1
            z_count += 1
            z_first = date if z_first is None else z_first
            z_last = date
        if date not in selected:
            continue
        variances = vnum / vden if vden > 0 else np.full(n, np.nan)
        vols = np.sqrt(c.annualization * variances)
        diag = {
            "raw_observations": raw_count, "standardized_observations": z_count,
            "minimum_raw_observations": c.min_history_observations,
            "minimum_standardized_observations": c.min_standardized_observations,
            "invalid_lag_variance_dates": list(excluded_lag_dates),
            "invalid_lag_variance_reason": "Required preceding standalone variance was zero or negligible" if excluded_lag_dates else "",
            "raw_start": raw_first, "raw_end": raw_last,
            "standardized_start": z_first, "standardized_end": z_last,
            "volatility_ess": vden ** 2 / vden2 if vden2 else None,
            "correlation_ess": {
                h: float(qden[k] ** 2 / qden2[k]) if qden2[k] else None
                for k, h in enumerate(hs)
            },
        }
        status, reason, correlations = "valid", "", {}
        if raw_count >= c.min_history_observations and (
            not np.isfinite(variances).all() or np.any(variances <= daily_epsilon2)
        ):
            status, reason = "failed", "A current required standalone variance is zero or negligible"
        elif raw_count < c.min_history_observations or z_count < c.min_standardized_observations:
            status, reason = "unavailable", (
                f"Insufficient history: {raw_count}/{c.min_history_observations} common raw returns; "
                f"{z_count}/{c.min_standardized_observations} eligible standardized vectors"
            )
        elif not np.isfinite(variances).all() or np.any(variances <= daily_epsilon2):
            status, reason = "unavailable", "A required standalone variance is zero or negligible"
        if status == "valid":
            try:
                for k, h in enumerate(hs):
                    q = qnum[k] / qden[k]
                    diagonal = np.diag(q)
                    if not np.isfinite(q).all() or np.any(diagonal <= 0):
                        raise ModelValidationError("Standardized second moment has invalid diagonal")
                    r = q / np.sqrt(np.outer(diagonal, diagonal))
                    validate_correlation(r, c)
                    correlations[h] = r
            except ModelValidationError as exc:
                status, reason = "failed", str(exc)
        outputs[date] = Estimate(date, status, reason, vols, correlations, diag)
    return outputs


def risk_metrics(
    weights, volatilities, correlation, fast_correlation=None,
    config: ModelConfig | None = None, names: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Risk levels and exact signed contribution tables for a single snapshot."""
    c = config or ModelConfig()
    x, s, r = (np.asarray(v, dtype=float) for v in (weights, volatilities, correlation))
    if x.ndim != 1 or s.shape != x.shape or r.shape != (len(x), len(x)):
        raise ModelValidationError("Exposure and matrix shapes do not match")
    if not np.isfinite(x).all() or not np.isfinite(s).all() or (x < 0).any() or (s < 0).any():
        raise ModelValidationError("Weights and volatilities must be finite and nonnegative")
    if abs(float(x.sum()) - 1) > c.weight_sum_atol:
        raise ModelValidationError("Risk weights must sum to one")
    ids = tuple(names) if names is not None else tuple(str(i) for i in range(len(x)))
    if len(ids) != len(x) or len(set(ids)) != len(ids):
        raise ModelValidationError("Strategy identifiers do not match exposures")
    matrix = {"slow": validate_correlation(r, c)}
    adjustments = []
    a = x * s
    variance = _quadratic(a, r, c, adjustments)
    sigma = np.sqrt(variance)
    vself, total = float(a @ a), float(a.sum())
    ii, jj = np.triu_indices(len(x), 1)
    pair_variance = 2 * a[ii] * a[jj] * r[ii, jj]
    vcross = float(pair_variance.sum())
    sigma0 = np.sqrt(vself)
    eps = c.annualized_volatility_epsilon
    meaningful = sigma > eps
    reference = _quadratic(np.ones(len(x)), r, c, adjustments)
    metrics = {
        "portfolio_volatility": float(sigma), "zero_correlation_volatility": float(sigma0),
        "correlation_uplift": float(sigma - sigma0),
        "diversification_ratio": total / sigma if meaningful else None,
        "effective_exposures": total ** 2 / vself if vself > eps ** 2 else None,
        "correlation_multiplier": sigma / sigma0 if sigma0 > eps else None,
        "equal_risk_correlation_multiplier": float(np.sqrt(reference / len(x))),
        "fast_slow_disagreement": None, "standalone_sum": total, "variance": variance,
        "self_variance": vself, "cross_variance": vcross,
        "self_contribution": vself / sigma if meaningful else None,
        "cross_contribution": vcross / sigma if meaningful else None,
    }
    if fast_correlation is not None:
        rf = np.asarray(fast_correlation, dtype=float)
        if rf.shape != r.shape:
            raise ModelValidationError("Fast and slow correlation shapes differ")
        matrix["fast"] = validate_correlation(rf, c)
        metrics["fast_slow_disagreement"] = float(np.sqrt(_quadratic(a, rf, c, adjustments)) - sigma)
    ra = r @ a
    euler = a * ra / sigma if meaningful else np.full(len(x), np.nan)
    selfc = a * a / sigma if meaningful else np.full(len(x), np.nan)
    crossc = a * (ra - a) / sigma if meaningful else np.full(len(x), np.nan)
    rho = ra / sigma if meaningful else np.full(len(x), np.nan)
    rho = np.where(s > eps, rho, np.nan)
    strategies = pd.DataFrame({
        "weight": x, "standalone_volatility": s, "risk_exposure": a,
        "portfolio_correlation": rho, "euler_contribution": euler,
        "self_contribution": selfc, "cross_contribution": crossc,
    }, index=pd.Index(ids, name="strategy"))
    pairs = pd.DataFrame({
        "strategy_i": [ids[i] for i in ii], "strategy_j": [ids[j] for j in jj],
        "correlation": r[ii, jj],
        "contribution": pair_variance / sigma if meaningful else np.full(len(ii), np.nan),
    })
    reconciliations = {"variance_self_cross": _reconcile(
        float(a @ r @ a), vself + vcross, "Variance self/cross before recorded roundoff adjustment", c)}
    if meaningful:
        reconciliations.update({
            "euler": _reconcile(float(euler.sum()), sigma, "Euler", c),
            "self_cross": _reconcile(float(selfc.sum() + crossc.sum()), sigma, "Self/cross", c),
            "pairs": _reconcile(float(pairs.contribution.sum()), metrics["cross_contribution"], "Pairs", c),
        })
        half_pairs = np.zeros(len(x))
        np.add.at(half_pairs, ii, pairs.contribution.to_numpy() / 2)
        np.add.at(half_pairs, jj, pairs.contribution.to_numpy() / 2)
        for i in range(len(x)):
            _reconcile(float(half_pairs[i]), float(crossc[i]), "Strategy half-pair aggregation", c)
        if metrics["correlation_multiplier"]:
            reconciliations["concentration"] = _reconcile(
                metrics["diversification_ratio"],
                np.sqrt(metrics["effective_exposures"]) / metrics["correlation_multiplier"],
                "Concentration identity", c,
            )
    return {"metrics": metrics, "strategies": strategies, "pairs": pairs,
            "diagnostics": {"matrix": matrix, "adjustments": adjustments, "reconciliations": reconciliations}}


def attribute_change(previous: Snapshot, current: Snapshot, config: ModelConfig | None = None) -> dict:
    """Six-order variance Shapley, with the exact analytic pair decomposition."""
    c = config or ModelConfig()
    if tuple(previous.weights.index) != tuple(current.weights.index):
        raise ModelValidationError("Attribution endpoints must have the same ordered universe")
    x0, x1 = previous.weights.to_numpy(), current.weights.to_numpy()
    s0, s1 = previous.volatilities.to_numpy(), current.volatilities.to_numpy()
    r0, r1 = previous.correlation.to_numpy(), current.correlation.to_numpy()
    validate_correlation(r0, c)
    validate_correlation(r1, c)
    adjustments = []
    values = {}
    for mask in range(8):
        x = x1 if mask & 1 else x0
        s = s1 if mask & 2 else s0
        r = r1 if mask & 4 else r0
        values[mask] = _quadratic(x * s, r, c, adjustments)
    variance_effects = np.zeros(3)
    for order in permutations(range(3)):
        mask = 0
        for block in order:
            new = mask | (1 << block)
            variance_effects[block] += (values[new] - values[mask]) / 6
            mask = new
    sigma0, sigma1 = np.sqrt(values[0]), np.sqrt(values[7])
    denom = sigma0 + sigma1
    meaningful = denom > c.annualized_volatility_epsilon
    effects = variance_effects / denom if meaningful else [None] * 3
    keys = ("weights", "standalone_volatility", "correlation")
    ii, jj = np.triu_indices(len(x0), 1)
    mixture = np.zeros(len(ii))
    for a, factor in ((x0*s0, 1/3), (x1*s0, 1/6), (x0*s1, 1/6), (x1*s1, 1/3)):
        mixture += factor * a[ii] * a[jj]
    delta = r1[ii, jj] - r0[ii, jj]
    pairv = 2 * delta * mixture
    ids = tuple(previous.weights.index)
    pairs = pd.DataFrame({
        "strategy_i": [ids[i] for i in ii], "strategy_j": [ids[j] for j in jj],
        "previous_correlation": r0[ii, jj], "current_correlation": r1[ii, jj],
        "correlation_change": delta, "variance_effect": pairv,
        "effect": pairv / denom if meaningful else np.full(len(ii), np.nan),
    })
    strategyv = np.zeros(len(ids))
    np.add.at(strategyv, ii, pairv / 2)
    np.add.at(strategyv, jj, pairv / 2)
    strategies = pd.DataFrame({"variance_effect": strategyv,
        "effect": strategyv / denom if meaningful else np.full(len(ids), np.nan)},
        index=pd.Index(ids, name="strategy"))
    rec = {
        "variance": _reconcile(float(variance_effects.sum()), values[7] - values[0], "Shapley variance", c),
        "pair_variance": _reconcile(float(pairv.sum()), float(variance_effects[2]), "Pair Shapley variance", c),
    }
    if meaningful:
        rec["volatility"] = _reconcile(float(np.sum(effects)), sigma1 - sigma0, "Shapley volatility", c)
        rec["pairs"] = _reconcile(float(pairs.effect.sum()), effects[2], "Pair Shapley volatility", c)
    return {
        "status": "valid" if meaningful else "unavailable",
        "reason": "" if meaningful else "Sum of endpoint volatilities is negligible; variance effects retained",
        "start_volatility": float(sigma0), "end_volatility": float(sigma1),
        "variance_effects": dict(zip(keys, variance_effects.tolist(), strict=True)),
        "effects": dict(zip(keys, effects, strict=True)), "pairs": pairs,
        "strategies": strategies, "reconciliations": rec, "adjustments": adjustments,
    }


def stress_metrics(snapshot: Snapshot, config: ModelConfig | None = None) -> dict:
    c = config or ModelConfig()
    r = snapshot.correlation.to_numpy()
    validate_correlation(r, c)
    a = snapshot.weights.to_numpy() * snapshot.volatilities.to_numpy()
    adjustments = []
    sigma = np.sqrt(_quadratic(a, r, c, adjustments))
    total = float(a.sum())
    scenarios, grid, rec = [], [], {}
    sigma_by_kappa = {}
    for kappa in c.correlation_stress_kappa:
        rk = (1 - kappa) * r + kappa * np.ones_like(r)
        validate_correlation(rk, c)
        vk = _quadratic(a, rk, c, adjustments)
        sk = float(np.sqrt(vk))
        rec[f"variance_kappa_{kappa:g}"] = _reconcile(vk, (1-kappa)*sigma**2 + kappa*total**2, "Stress variance", c)
        sigma_by_kappa[kappa] = sk
        row = {"kappa": kappa, "volatility": sk, "uplift": sk - sigma,
            "percentage_increase": sk / sigma - 1 if sigma > c.annualized_volatility_epsilon else None,
            "diversification_ratio": total / sk if sk > c.annualized_volatility_epsilon else None}
        scenarios.append(row)
        for multiplier in c.volatility_stress_multipliers:
            grid.append({**row, "multiplier": multiplier, "volatility": multiplier * sk,
                "uplift": multiplier * sk - sigma,
                "percentage_increase": multiplier * sk / sigma - 1 if sigma > c.annualized_volatility_epsilon else None})
    kappa = c.pair_stress_kappa
    denom = sigma_by_kappa[kappa] + sigma
    ii, jj = np.triu_indices(len(a), 1)
    pairv = 2 * kappa * a[ii] * a[jj] * (1-r[ii, jj])
    valid_pairs = denom > c.annualized_volatility_epsilon
    ids = tuple(snapshot.weights.index)
    pairs = pd.DataFrame({"strategy_i": [ids[i] for i in ii], "strategy_j": [ids[j] for j in jj],
        "correlation": r[ii, jj],
        "contribution": pairv / denom if valid_pairs else np.full(len(ii), np.nan)})
    if valid_pairs:
        rec["pairs"] = _reconcile(float(pairs.contribution.sum()), sigma_by_kappa[kappa] - sigma, "Stress pairs", c)
    return {"status": "valid", "reason": "", "scenarios": pd.DataFrame(scenarios),
        "grid": pd.DataFrame(grid), "pairs": pairs, "pair_kappa": kappa,
        "pair_status": "valid" if valid_pairs else "unavailable",
        "pair_reason": "" if valid_pairs else "Sum of baseline and stressed volatility is negligible",
        "reconciliations": rec, "adjustments": adjustments}


def _snapshot(date, estimate, holdings, universe, config, slow=None, fast=None, alpha=None):
    c = config
    if estimate.status != "valid":
        return None, estimate.status, estimate.reason
    if date not in holdings.weights.index or not np.isfinite(holdings.weights.loc[date]).all():
        reason = str(holdings.reasons.get(date, "Holdings are unavailable at this close"))
        return None, "unavailable", reason or "Holdings are unavailable at this close"
    slow = c.corr_half_life_slow if slow is None else slow
    fast = c.corr_half_life_fast if fast is None else fast
    alpha = c.shrinkage_alpha if alpha is None else alpha
    r = shrink_correlation(estimate.correlations[slow], alpha)
    rf = shrink_correlation(estimate.correlations[fast], alpha)
    weights = holdings.weights.loc[date].reindex(universe)
    result = risk_metrics(weights, estimate.volatilities, r, rf, c, universe)
    strategies = result["strategies"]
    strategies["target_weight"] = holdings.latest_targets.loc[date].reindex(universe)
    strategies["units"] = holdings.units.loc[date].reindex(universe)
    last_rebalance = holdings.last_rebalance.get(date)
    diagnostics = {
        **estimate.diagnostics, **result["diagnostics"],
        "slow_correlation_ess": estimate.diagnostics["correlation_ess"][slow],
        "fast_correlation_ess": estimate.diagnostics["correlation_ess"][fast],
        "last_rebalance": last_rebalance, "latest_target_date": last_rebalance,
        "holdings_date": date, "synthetic_portfolio_value": float(holdings.value.loc[date]),
        "pre_rebalance_weights": holdings.pre_rebalance_weights.loc[last_rebalance].to_dict()
        if pd.notna(last_rebalance) and last_rebalance in holdings.pre_rebalance_weights.index else {},
    }
    snapshot = Snapshot(date, weights, pd.Series(estimate.volatilities, index=universe),
        pd.DataFrame(r, index=universe, columns=universe),
        pd.DataFrame(rf, index=universe, columns=universe), result["metrics"],
        strategies, result["pairs"], diagnostics)
    return snapshot, "valid", ""


def _autocorrelation(values: pd.Series) -> float | None:
    # Pair on the original calendar, then remove invalid pairs. Dropping first
    # would falsely make returns on either side of a gap consecutive.
    pairs = pd.concat([values, values.shift(1)], axis=1).dropna()
    if len(pairs) < 2 or (pairs.std(ddof=0) == 0).any():
        return None
    return float(np.corrcoef(pairs.to_numpy().T)[0, 1])


def weekly_diagnostic(levels: pd.DataFrame, cutoff, config: ModelConfig | None = None) -> pd.DataFrame:
    """Paired empirical daily/weekly second moments on complete weekly chains.

    A week's valuation endpoint is its final declared trading close; the calendar
    Friday boundary must nevertheless be certified complete. Each retained
    weekly return requires endpoints in consecutive week buckets and every
    intervening common daily level. Standalone strategy statistics use the same
    retained intervals across the fixed universe, without replacing the official estimator.
    """
    c = config or ModelConfig()
    cutoff = pd.Timestamp(cutoff)
    frame = levels.loc[levels.index <= cutoff]
    if frame.empty:
        return pd.DataFrame()
    periods = frame.index.to_period("W-FRI")
    boundaries = periods.end_time.normalize()
    period_endpoints = {}
    for date, boundary in zip(frame.index, boundaries, strict=True):
        if boundary <= cutoff:
            period_endpoints[boundary] = date
    full_boundaries = pd.date_range(boundaries.min(), cutoff.normalize(), freq="W-FRI")
    window_start = cutoff - pd.DateOffset(months=c.weekly_history_months)
    daily = frame.div(frame.shift(1)).sub(1)
    output = []
    for strategy in frame.columns:
        weekly = pd.Series(np.nan, index=full_boundaries, dtype=float)
        retained_daily = pd.Series(np.nan, index=frame.index, dtype=float)
        candidate_count = missing_count = 0
        retained_starts, retained_ends = [], []
        for boundary in full_boundaries:
            if boundary <= window_start:
                continue
            candidate_count += 1
            prior_boundary = boundary - pd.Timedelta(days=7)
            start, end = period_endpoints.get(prior_boundary), period_endpoints.get(boundary)
            if start is None or end is None:
                missing_count += 1
                continue
            common_chain = frame.loc[start:end]
            if len(common_chain) < 2 or not np.isfinite(common_chain).all().all() or (common_chain <= 0).any().any():
                missing_count += 1
                continue
            chain = common_chain[strategy]
            weekly.loc[boundary] = float(chain.iloc[-1] / chain.iloc[0] - 1)
            sample = daily.loc[(daily.index > start) & (daily.index <= end), strategy]
            retained_daily.loc[sample.index] = sample
            retained_starts.append(start)
            retained_ends.append(end)
        count = int(weekly.notna().sum())
        d = retained_daily.dropna()
        valid = count >= c.weekly_min_intervals
        dv = float(np.sqrt(c.annualization * np.mean(d.to_numpy() ** 2))) if valid else None
        wv = float(np.sqrt(c.weekly_annualization * np.mean(weekly.dropna().to_numpy() ** 2))) if valid else None
        output.append({
            "strategy": strategy, "status": "valid" if valid else "unavailable",
            "reason": "" if valid else f"Requires {c.weekly_min_intervals} complete weekly intervals; found {count}",
            "weekly_intervals": count, "daily_observations": len(d),
            "candidate_intervals": candidate_count, "excluded_intervals": missing_count,
            "window_start": window_start, "window_end": cutoff,
            "first_interval_start": min(retained_starts) if retained_starts else None,
            "last_interval_end": max(retained_ends) if retained_ends else None,
            "daily_volatility": dv, "weekly_volatility": wv,
            "weekly_daily_ratio": wv / dv if valid and dv > c.annualized_volatility_epsilon else None,
            "daily_autocorrelation": _autocorrelation(retained_daily) if valid else None,
            "weekly_autocorrelation": _autocorrelation(weekly) if valid else None,
        })
    return pd.DataFrame(output).set_index("strategy")


def _stale_diagnostic(returns):
    output = []
    for name in returns:
        values = returns[name].to_numpy()
        longest = current = 0
        for value in values:
            current = current + 1 if np.isfinite(value) and value == 0 else 0
            longest = max(longest, current)
        valid = np.isfinite(values)
        count = int(valid.sum())
        zero = int((values[valid] == 0).sum())
        output.append({"strategy": name, "observations": count, "zero_returns": zero,
            "zero_fraction": zero / count if count else None,
            "longest_zero_run": longest, "latest_zero_run": current,
            "missing_returns": int((~valid).sum())})
    return pd.DataFrame(output).set_index("strategy")


def _percentile(history, current, c):
    if current is None or current.metrics["diversification_ratio"] is None:
        return {"status": "unavailable", "reason": "Current diversification ratio is unavailable", "value": None, "count": 0}
    current_month = current.date.to_period("M")
    first_month = current_month - c.percentile_history_months
    last_month = current_month - 1
    window_start = first_month.start_time
    window_end = last_month.end_time.normalize()
    if history.empty:
        reference = pd.Series(dtype=float)
    else:
        reference = history.loc[
            (history.index.to_period("M") >= first_month)
            & (history.index.to_period("M") <= last_month)
            & history.completed_month & history.status.eq("valid"), "diversification_ratio"
        ].dropna()
    count = len(reference)
    valid = count >= c.percentile_min_prior_snapshots
    value = current.metrics["diversification_ratio"]
    percentile = 100 * ((reference < value).sum() + 0.5 * (reference == value).sum()) / count if valid else None
    return {"status": "valid" if valid else "unavailable",
        "reason": "" if valid else f"Requires {c.percentile_min_prior_snapshots} prior completed snapshots; found {count}",
        "value": float(percentile) if valid else None, "count": count,
        "reference_start": reference.index.min() if count else None,
        "reference_end": reference.index.max() if count else None,
        "reference_dates": list(reference.index), "window_start": window_start,
        "window_end": window_end, "reference_period_start": str(first_month),
        "reference_period_end": str(last_month), "method": "midrank"}


def _sensitivity_diagnostics(current, previous, estimates, holdings, universe, c):
    shrink_rows, half_rows, pair_rows, adjustments = [], [], [], []
    baseline = current.pairs.set_index(["strategy_i", "strategy_j"])["contribution"]
    baseline_rank = current.pairs.assign(absolute=current.pairs.contribution.abs()).sort_values(
        ["absolute", "strategy_i", "strategy_j"], ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)
    rank_map = {(row.strategy_i, row.strategy_j): rank + 1
        for rank, row in baseline_rank.iterrows() if np.isfinite(row.contribution)}
    settings = [("shrinkage", str(alpha), c.corr_half_life_slow, c.corr_half_life_fast, alpha)
        for alpha in c.shrinkage_sensitivity]
    settings += [("half_life", f"{slow:g}/{fast:g}", slow, fast, c.shrinkage_alpha)
        for slow, fast in c.half_life_sensitivity]
    for family, setting, slow, fast, alpha in settings:
        target = shrink_rows if family == "shrinkage" else half_rows
        row = {"setting": setting, "shrinkage_alpha": alpha, "slow_half_life": slow, "fast_half_life": fast}
        try:
            snap, status, reason = _snapshot(current.date, estimates[current.date], holdings, universe, c, slow, fast, alpha)
            row.update({"status": status, "reason": reason})
            if snap is None:
                target.append(row)
                continue
            row.update(snap.metrics)
            adjustments.extend(snap.diagnostics["adjustments"])
            if previous is not None:
                old, old_status, old_reason = _snapshot(previous.date, estimates[previous.date], holdings, universe, c, slow, fast, alpha)
                row["comparison_status"] = old_status
                row["comparison_reason"] = old_reason
                if old is not None:
                    attribution = attribute_change(old, snap, c)
                    row.update({f"change_{key}": value for key, value in attribution["effects"].items()})
                    adjustments.extend(attribution["adjustments"])
            ranked = snap.pairs.assign(absolute=snap.pairs.contribution.abs()).sort_values(
                ["absolute", "strategy_i", "strategy_j"], ascending=[False, True, True], kind="stable")
            for rank, pair in enumerate(ranked.itertuples(), 1):
                key = (pair.strategy_i, pair.strategy_j)
                base = baseline.loc[key]
                current_finite = bool(np.isfinite(pair.contribution))
                baseline_finite = bool(np.isfinite(base))
                finite = current_finite and baseline_finite
                current_rank = rank if current_finite else None
                base_rank = rank_map.get(key)
                unavailable = []
                if not current_finite:
                    unavailable.append("Scenario pair contribution is undefined; its rank is unavailable")
                if not baseline_finite:
                    unavailable.append("Baseline pair contribution is undefined; its rank and rank change are unavailable")
                pair_rows.append({"family": family, "setting": setting, "strategy_i": key[0],
                    "strategy_j": key[1], "rank": current_rank, "baseline_rank": base_rank,
                    "rank_change": base_rank - current_rank if finite else None,
                    "contribution": pair.contribution,
                    "sign_changed": bool(np.sign(pair.contribution) != np.sign(base)) if finite else None,
                    "status": "valid" if finite else "unavailable", "reason": "; ".join(unavailable)})
        except ModelValidationError as exc:
            row.update({"status": "failed", "reason": str(exc)})
        target.append(row)
    return {"shrinkage_sensitivity": pd.DataFrame(shrink_rows),
        "half_life_sensitivity": pd.DataFrame(half_rows), "sensitivity_pairs": pd.DataFrame(pair_rows),
        "sensitivity_adjustments": adjustments}


def run_review(
    inputs: InputData, portfolio: PortfolioConfig | None = None,
    model: ModelConfig | None = None, as_of_date=None,
) -> ReviewResult:
    """Calculate comparable endpoints, monthly history, stresses and diagnostics.

    Input failures and invalid current numerical states yield a diagnostic report.
    Missing historical endpoints suppress only their dependent outputs. No values
    beyond the resolved as-of close enter validation, holdings or estimation.
    """
    started = perf_counter()
    c, portfolio = model or ModelConfig(), portfolio or PortfolioConfig()
    requested = None
    result = ReviewResult("unavailable", [], requested, None, None)
    result.metadata = {
        "portfolio": asdict(portfolio), "model_config": {**asdict(c),
            "min_standardized_observations": c.min_standardized_observations},
        "universe": tuple(inputs.universe), "provenance": inputs.provenance,
        "calendar_convention": "CSV index is authoritative; entirely omitted trading dates cannot be detected",
        "calendar_complete_through": inputs.calendar_complete_through,
        "historical_vintage_note": "Historical estimates use the supplied data vintage, not an archived point-in-time record",
        "run_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    try:
        requested, resolved, cutoff = resolve_dates(inputs, as_of_date)
        result.requested_date, result.resolved_date = requested, resolved
        validate_values(inputs, resolved, weight_sum_atol=c.weight_sum_atol)
        levels = inputs.levels.loc[:resolved, list(inputs.universe)].apply(pd.to_numeric)
        returns = daily_returns(inputs, through=resolved)
        holdings = build_holdings(inputs, through=resolved, weight_sum_atol=c.weight_sum_atol)
        months = complete_period_ends(levels.index, cutoff, freq="M")
        previous_month = resolved.to_period("M") - 1
        candidates = months[months.to_period("M") == previous_month]
        comparison = candidates[-1] if len(candidates) else None
        result.comparison_date = comparison
        selected = months.union(pd.DatetimeIndex([resolved]))
        if comparison is not None:
            selected = selected.union(pd.DatetimeIndex([comparison]))
        hs = tuple(dict.fromkeys((c.corr_half_life_slow, c.corr_half_life_fast,
            *(h for pair in c.half_life_sensitivity for h in pair))))
        estimates = estimate_history(returns, c, selected, hs)
        common = returns.notna().all(axis=1)
        raw_candidates = returns.iloc[1:]
        excluded = raw_candidates.index[~raw_candidates.notna().all(axis=1)]
        result.diagnostics = {
            "coverage": {"calendar_start": levels.index[0], "calendar_end": levels.index[-1],
                "calendar_dates": len(levels), "common_return_observations": int(common.sum()),
                "excluded_return_dates": list(excluded), "excluded_return_count": len(excluded),
                "missing_level_count": int(levels.isna().sum().sum()),
                "common_return_fraction": float(common.iloc[1:].mean()) if len(common) > 1 else None,
                "first_common_return": returns.index[common][0] if common.any() else None,
                "last_common_return": returns.index[common][-1] if common.any() else None,
                "calendar_complete_through": inputs.calendar_complete_through,
                "holdings_start": holdings.value.first_valid_index(),
                "latest_rebalance": holdings.last_rebalance.get(resolved)},
            "stale_returns": _stale_diagnostic(returns.iloc[1:]),
            "weekly": weekly_diagnostic(levels, cutoff, c), "adjustments": [], "reconciliations": {},
            "current_estimation": estimates[resolved].diagnostics,
        }
        rows, snapshots = [], {}
        first_valid = last_valid = None
        current_state = ("unavailable", "Current endpoint was not evaluated")
        previous_reason = "No certified completed snapshot in the preceding calendar month"
        for date in selected:
            try:
                snapshot, status, reason = _snapshot(date, estimates[date], holdings, inputs.universe, c)
            except ModelValidationError as exc:
                snapshot, status, reason = None, "failed", str(exc)
            if snapshot is not None:
                if date in (resolved, comparison):
                    snapshots[date] = snapshot
                last_valid = date
                first_valid = date if first_valid is None else first_valid
                result.diagnostics["adjustments"].extend(
                    {**adjustment, "date": date} for adjustment in snapshot.diagnostics["adjustments"])
            rows.append({"date": date, "status": status, "reason": reason,
                "completed_month": date in months, **(snapshot.metrics if snapshot is not None else {})})
            if date == resolved:
                current_state = (status, reason)
            if date == comparison:
                previous_reason = reason
        # Empty whole months remain explicit unavailable chart gaps. These
        # placeholder dates are calendar boundaries, never claimed trading closes.
        declared_months = set(levels.index.to_period("M"))
        for period in pd.period_range(levels.index[0], cutoff, freq="M"):
            boundary = period.end_time.normalize()
            if boundary <= cutoff and period not in declared_months:
                rows.append({"date": boundary, "status": "unavailable",
                    "reason": "No declared trading close in this calendar month",
                    "completed_month": True})
        history = pd.DataFrame(rows).set_index("date").sort_index()
        result.history = history.loc[first_valid:] if first_valid is not None else history.iloc[:0]
        result.current, result.previous = snapshots.get(resolved), snapshots.get(comparison)
        result.status = current_state[0]
        if current_state[1]:
            result.reasons.append(current_state[1])
        result.diagnostics["coverage"]["model_history_start"] = first_valid
        result.diagnostics["coverage"]["model_history_end"] = last_valid
        result.diagnostics["dr_percentile"] = _percentile(result.history, result.current, c)
        if result.current is not None:
            result.stress = stress_metrics(result.current, c)
            result.diagnostics["reconciliations"]["current"] = result.current.diagnostics["reconciliations"]
            result.diagnostics["reconciliations"]["stress"] = result.stress["reconciliations"]
            result.diagnostics["adjustments"].extend(result.stress["adjustments"])
            result.diagnostics.update(_sensitivity_diagnostics(result.current, result.previous,
                estimates, holdings, inputs.universe, c))
            result.diagnostics["adjustments"].extend(result.diagnostics.pop("sensitivity_adjustments"))
        else:
            result.stress = {"status": "unavailable", "reason": "Current risk snapshot is unavailable"}
        if result.current is not None and result.previous is not None:
            result.attribution = attribute_change(result.previous, result.current, c)
            result.diagnostics["reconciliations"]["attribution"] = result.attribution["reconciliations"]
            result.diagnostics["adjustments"].extend(result.attribution["adjustments"])
        else:
            result.attribution = {"status": "unavailable", "reason": previous_reason
                if result.current is not None else "Current risk snapshot is unavailable"}
        result.metadata.update({"requested_date": requested, "resolved_date": resolved,
            "comparison_date": comparison, "report_cutoff": cutoff})
    except (InputValidationError, ModelValidationError) as exc:
        result.status = "unavailable" if "calendar_complete_through" in str(exc) or "precedes the first declared" in str(exc) else "failed"
        result.reasons = [str(exc)]
        result.current = None
        result.attribution = {"status": "unavailable", "reason": "Input or numerical validation failed"}
        result.stress = {"status": "unavailable", "reason": "Input or numerical validation failed"}
    result.metadata["analysis_seconds"] = perf_counter() - started
    return result
