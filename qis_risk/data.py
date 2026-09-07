"""Validated index inputs, declared-calendar returns, and a constant-unit ledger.

Only index levels are accepted. Dates present in the level file define the
trading calendar: a wholly omitted trading date cannot be detected here.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_UNIVERSE = (
    "EQ_VALUE", "EQ_MOM", "EQ_LOWVOL", "RATES_CARRY", "RATES_TREND",
    "FX_CARRY", "FX_VALUE", "COM_CARRY", "COM_TREND", "MULTI_TREND",
)


class InputValidationError(ValueError):
    """An input violates the declared structural or numeric contract."""


def _date(value: object, label: str) -> pd.Timestamp:
    if isinstance(value, (int, float, np.number)):
        raise InputValidationError(f"{label} must be a calendar date, not a numeric timestamp")
    try:
        date = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputValidationError(f"{label} must be a valid date") from exc
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise InputValidationError(f"{label} must be a timezone-free calendar date")
    return date


def _frame_structure(frame: pd.DataFrame, universe: tuple[str, ...], label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise InputValidationError(f"{label} must be a DataFrame")
    if not frame.columns.is_unique:
        raise InputValidationError(f"{label} contains duplicate strategy identifiers")
    if set(frame.columns) != set(universe):
        missing = sorted(set(universe) - set(frame.columns))
        unknown = sorted(set(frame.columns) - set(universe), key=str)
        raise InputValidationError(f"{label} strategy mismatch: missing={missing}, unknown={unknown}")
    if isinstance(frame.index, (pd.RangeIndex, pd.MultiIndex)):
        raise InputValidationError(f"{label} must be indexed by calendar dates")
    dates = pd.DatetimeIndex([_date(value, f"{label} index") for value in frame.index])
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise InputValidationError(f"{label} dates must be sorted and unique")
    result = frame.loc[:, list(universe)].copy(deep=True)
    result.index = dates.rename("date")
    return result


@dataclass
class InputData:
    """A fixed ordered universe with globally validated structure.

    Numeric checks happen at review time through the report cutoff. This keeps
    malformed future values from changing an otherwise valid historical run.
    Missing level observations are allowed; missing target weights are not.
    """

    levels: pd.DataFrame
    targets: pd.DataFrame
    universe: tuple[str, ...]
    calendar_complete_through: pd.Timestamp | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.universe, str):
            raise InputValidationError("universe must be an ordered sequence of strategy identifiers")
        self.universe = tuple(self.universe)
        if not self.universe or any(not isinstance(x, str) or not x.strip() for x in self.universe):
            raise InputValidationError("universe must contain nonempty string identifiers")
        if len(set(self.universe)) != len(self.universe):
            raise InputValidationError("universe contains duplicate identifiers")
        self.levels = _frame_structure(self.levels, self.universe, "levels")
        self.targets = _frame_structure(self.targets, self.universe, "targets")
        if self.levels.empty:
            raise InputValidationError("levels must contain at least one trading date")
        outside = self.targets.index.difference(self.levels.index)
        if len(outside):
            raise InputValidationError(f"rebalance dates outside the declared calendar: {list(outside)}")
        self.calendar_complete_through = (
            self.levels.index[-1] if self.calendar_complete_through is None
            else _date(self.calendar_complete_through, "calendar_complete_through")
        )
        if self.calendar_complete_through < self.levels.index[0]:
            raise InputValidationError("certified calendar coverage ends before the first trading date")
        self.provenance = dict(self.provenance)


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as source:
        header = next(csv.reader(source), [])
    if len(header) < 2 or header[0] != "date":
        raise InputValidationError(f"{label} CSV needs a date column followed by strategies")
    if len(set(header)) != len(header) or any(not item.strip() for item in header):
        raise InputValidationError(f"{label} CSV contains duplicate or empty column identifiers")
    try:
        return pd.read_csv(path, index_col=0)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise InputValidationError(f"could not parse {label} CSV: {exc}") from exc


def load_inputs(
    levels_csv: str | Path,
    weights_csv: str | Path,
    *,
    universe: tuple[str, ...] | list[str],
    calendar_complete_through: object = None,
) -> InputData:
    """Load the two CSVs; decimal target weights use ``0.25 = 25%``."""
    levels_path, weights_path = Path(levels_csv), Path(weights_csv)
    return InputData(
        _read_csv(levels_path, "levels"),
        _read_csv(weights_path, "targets"),
        tuple(universe),
        calendar_complete_through,
        {
            "levels_source": str(levels_path),
            "weights_source": str(weights_path),
            "levels_sha256": hashlib.sha256(levels_path.read_bytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "calendar": "CSV date index is authoritative; wholly omitted trading dates undetectable",
            "live_backtested_status": "unknown",
            "data_vintage": "unknown",
        },
    )


def _numeric(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    try:
        return frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputValidationError(f"{label} contains nonnumeric observations") from exc


def _levels_through(inputs: InputData, through: pd.Timestamp) -> pd.DataFrame:
    levels = _numeric(inputs.levels.loc[:through], "levels")
    observed = levels.to_numpy()[~levels.isna().to_numpy()]
    if not np.isfinite(observed).all() or np.any(observed <= 0):
        raise InputValidationError("observed index levels must be positive and finite")
    return levels


def validate_values(inputs: InputData, through: pd.Timestamp, weight_sum_atol: float = 1e-8) -> None:
    """Validate numeric domains and every required rebalance through a cutoff."""
    through = _date(through, "through")
    if not np.isfinite(weight_sum_atol) or weight_sum_atol < 0:
        raise InputValidationError("weight_sum_atol must be finite and nonnegative")
    levels = _levels_through(inputs, through)
    targets = _numeric(inputs.targets.loc[:through], "targets")
    if not np.isfinite(targets.to_numpy()).all() or (targets.to_numpy() < 0).any():
        raise InputValidationError("target weights must be complete, finite and nonnegative")
    if not np.allclose(targets.sum(axis=1), 1.0, atol=weight_sum_atol, rtol=0):
        raise InputValidationError("target weight rows must sum to 1 within weight_sum_atol")
    missing_rebalances = levels.loc[targets.index].isna().any(axis=1)
    if missing_rebalances.any():
        dates = [date.date().isoformat() for date in missing_rebalances.index[missing_rebalances]]
        raise InputValidationError(f"all strategy levels are required at rebalance closes: {dates}")


def daily_returns(inputs: InputData, through: object = None) -> pd.DataFrame:
    """Consecutive declared-date simple returns; gaps invalidate both adjacent returns."""
    cutoff = inputs.levels.index[-1] if through is None else _date(through, "through")
    levels = _levels_through(inputs, cutoff)
    return levels.div(levels.shift(1)).sub(1)


@dataclass
class Holdings:
    """Daily post-close holdings and availability, including before-trade weights."""

    value: pd.Series
    weights: pd.DataFrame
    units: pd.DataFrame
    returns: pd.Series
    latest_targets: pd.DataFrame
    last_rebalance: pd.Series
    pre_rebalance_weights: pd.DataFrame
    reasons: pd.Series


def build_holdings(
    inputs: InputData, through: object = None, weight_sum_atol: float = 1e-8,
) -> Holdings:
    """Mark old units before each rebalance, then establish new closing units.

    A missing valuation retains known units for recovery at a later complete
    close. All configured strategies require levels, including zero allocations.
    No synthetic portfolio or strategy return is bridged across a missing close.
    """
    cutoff = inputs.levels.index[-1] if through is None else _date(through, "through")
    validate_values(inputs, cutoff, weight_sum_atol)
    levels = _numeric(inputs.levels.loc[:cutoff], "levels")
    targets = _numeric(inputs.targets.loc[:cutoff], "targets")
    dates, columns = levels.index, levels.columns
    count, size = levels.shape
    values = np.full(count, np.nan)
    weights, units, latest = [np.full((count, size), np.nan) for _ in range(3)]
    rebalance_dates = np.full(count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    reasons = np.full(count, "Holdings unavailable before first supplied rebalance", dtype=object)
    pre = np.full(targets.shape, np.nan)
    target_positions = {date: pos for pos, date in enumerate(targets.index)}
    target_array = targets.to_numpy()
    held_units = current_target = None
    last_rebalance = np.datetime64("NaT", "ns")
    for row, (date, level) in enumerate(zip(dates, levels.to_numpy(), strict=True)):
        rebalance = target_positions.get(date)
        if held_units is None and rebalance is None:
            continue
        if held_units is None:
            value = 100.0
        elif np.isfinite(level).all():
            value = float(held_units @ level)
            if not np.isfinite(value) or value <= 0:
                raise InputValidationError(
                    f"portfolio value must be positive and finite at {date.date()}"
                )
        else:
            value = np.nan
        if rebalance is not None:
            if held_units is not None:
                pre[rebalance] = held_units * level / value
            current_target = target_array[rebalance]
            held_units = current_target * value / level
            if not np.isfinite(held_units).all():
                raise InputValidationError(f"portfolio units must be finite at {date.date()}")
            last_rebalance = date.to_datetime64()
        units[row] = held_units
        latest[row] = current_target
        rebalance_dates[row] = last_rebalance
        if not np.isfinite(value):
            reasons[row] = "Holdings valuation unavailable: missing strategy index level"
            continue
        if value <= 0:
            raise InputValidationError(f"portfolio value must be positive and finite at {date.date()}")
        values[row] = value
        weights[row] = current_target if rebalance is not None else held_units * level / value
        reasons[row] = ""
    value = pd.Series(values, index=dates, name="synthetic_excess_return_value")
    return Holdings(
        value=value,
        weights=pd.DataFrame(weights, index=dates, columns=columns),
        units=pd.DataFrame(units, index=dates, columns=columns),
        returns=value.div(value.shift(1)).sub(1).rename("synthetic_excess_return_return"),
        latest_targets=pd.DataFrame(latest, index=dates, columns=columns),
        last_rebalance=pd.Series(rebalance_dates, index=dates, name="last_rebalance"),
        pre_rebalance_weights=pd.DataFrame(pre, index=targets.index, columns=columns),
        reasons=pd.Series(reasons, index=dates, name="holdings_reason"),
    )


def resolve_dates(
    inputs: InputData, as_of_date: object = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp, pd.Timestamp]:
    """Return requested close, preceding declared close, and certified report cutoff."""
    requested = None if as_of_date is None else _date(as_of_date, "as_of_date")
    coverage = inputs.calendar_complete_through
    cutoff = coverage if requested is None else requested
    if cutoff > coverage:
        raise InputValidationError("requested date exceeds calendar_complete_through")
    available = inputs.levels.index[inputs.levels.index <= cutoff]
    if not len(available):
        raise InputValidationError("requested date precedes the first declared trading date")
    return requested, available[-1], cutoff


def complete_period_ends(
    calendar: pd.DatetimeIndex, cutoff: pd.Timestamp, freq: str = "M",
) -> pd.DatetimeIndex:
    """Last declared closes of completed calendar months or Friday-ended weeks.

    Period completion is based on the calendar boundary, not the last observed
    close. Empty periods are omitted; callers must retain period gaps when
    constructing non-overlapping consecutive-week diagnostics.
    """
    cutoff = _date(cutoff, "cutoff")
    if freq not in {"M", "W-FRI"}:
        raise InputValidationError("period frequency must be M or W-FRI")
    dates = pd.DatetimeIndex(calendar)
    dates = dates[dates <= cutoff]
    if not len(dates):
        return pd.DatetimeIndex([], name=calendar.name)
    periods = dates.to_period(freq)
    frame = pd.Series(dates, index=periods).groupby(level=0, sort=True).last()
    complete = frame.index.end_time.normalize() <= cutoff
    return pd.DatetimeIndex(frame[complete].to_numpy(), name=calendar.name)


def make_demo_data(seed: int = 42, periods: int | None = None) -> InputData:
    """Deterministic ten-strategy demonstration; weekdays are a simulated calendar.

    Default coverage is 2018–2025. ``periods=5000`` supplies a longer benchmark
    sample. Factor amplitudes and standalone scales change smoothly and by
    regime; the data are simulated, with no market-data interpretation.
    """
    if periods is not None and (not isinstance(periods, int) or isinstance(periods, bool) or periods < 2):
        raise InputValidationError("periods must be an integer of at least 2")
    dates = (
        pd.bdate_range("2018-01-01", "2025-12-31") if periods is None
        else pd.bdate_range("2018-01-01", periods=periods)
    )
    rng = np.random.default_rng(seed)
    count, size = len(dates), len(DEMO_UNIVERSE)
    t = np.arange(count - 1)
    betas = np.array([
        [.75, .10, .05], [.50, .25, .20], [.45, -.10, .05],
        [-.20, .70, .05], [-.30, -.45, .30], [.45, .15, .35],
        [.10, -.10, -.25], [.25, .05, .65], [.10, -.15, -.50],
        [-.25, -.20, -.20],
    ])
    regime = np.where((t // 420) % 3 == 1, 1.75, .8)
    common = rng.standard_normal((count - 1, 3)) * regime[:, None]
    idiosyncratic = rng.standard_normal((count - 1, size))
    scales = np.linspace(.045, .12, size) / np.sqrt(252)
    volatility = .85 + .25 * np.sin(t[:, None] / 125 + np.arange(size)[None, :])
    returns = ((common @ betas.T) + .8 * idiosyncratic) * scales * volatility + .02 / 252
    levels = np.vstack([np.full(size, 100.0), 100 * np.cumprod(1 + returns, axis=0)])
    coverage = pd.Timestamp("2025-12-31") if periods is None else dates[-1]
    monthly = complete_period_ends(dates, coverage)
    rebalances = dates[:1].union(monthly)
    targets = rng.dirichlet(np.full(size, 12.0), size=len(rebalances))
    for row in range(0, len(rebalances), 2):
        targets[row, (row // 2) % size] = 0.0
        targets[row] /= targets[row].sum()
    return InputData(
        pd.DataFrame(levels, index=dates, columns=DEMO_UNIVERSE),
        pd.DataFrame(targets, index=rebalances, columns=DEMO_UNIVERSE),
        DEMO_UNIVERSE,
        coverage,
        {
            "source": "deterministic simulation", "seed": seed,
            "calendar": "simulated Monday–Friday calendar; no exchange holiday exclusions",
            "live_backtested_status": "simulated", "data_vintage": "simulation-v1",
        },
    )


def write_demo_csvs(directory: str | Path) -> tuple[Path, Path]:
    """Write the reproducible demonstration inputs and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inputs = make_demo_data()
    levels_path, weights_path = directory / "index_levels.csv", directory / "rebalance_weights.csv"
    inputs.levels.to_csv(levels_path, float_format="%.12g", date_format="%Y-%m-%d")
    inputs.targets.to_csv(weights_path, float_format="%.12g", date_format="%Y-%m-%d")
    return levels_path, weights_path
