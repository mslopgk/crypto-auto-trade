"""Performance-metric correctness on hand-built equity curves and trade sets.

References (core/backtest/metrics.py):
- crypto 365-day annualization (constants.periods_per_year);
- Sharpe = mean(ret)/std(ret, ddof=1) * sqrt(ppy); std of a constant is 0 -> 0;
- max drawdown = max(1 - equity/running_peak);
- CAGR = (final/initial)^(1/years) - 1, years from the actual timestamps;
- win_rate / profit_factor computed from the trades' pnl column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.metrics import compute_metrics, max_drawdown
from core.constants import periods_per_year

from conftest import hourly_index


def _equity(values) -> pd.Series:
    values = np.asarray(values, dtype=np.float64)
    return pd.Series(values, index=hourly_index(len(values)), name="equity")


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {"pnl": [], "ret_pct": [], "bars_held": []}
    ).astype({"pnl": "float64", "ret_pct": "float64", "bars_held": "float64"})


# ---------------------------------------------------------- 1) constant equity

def test_constant_equity_zero_sharpe_zero_mdd():
    eq = _equity([10_000.0] * 200)
    m = compute_metrics(eq, _empty_trades(), timeframe="1h")
    assert m["sharpe"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert m["total_return"] == 0.0
    assert m["cagr"] == 0.0
    # no downside, no drawdown -> sortino / calmar collapse to 0 (finite)
    assert m["sortino"] == 0.0
    assert m["calmar"] == 0.0
    # direct max_drawdown on a flat curve is exactly zero, no time underwater
    mdd, longest = max_drawdown(eq.to_numpy())
    assert mdd == 0.0 and longest == 0


# --------------------------------------------------------- 2) crafted drawdown

def test_crafted_drawdown_exact_mdd():
    # peak 200 then trough 150 -> mdd = 1 - 150/200 = 0.25 exactly
    eq = np.array([100.0, 200.0, 150.0, 180.0])
    mdd, longest = max_drawdown(eq)
    assert mdd == pytest.approx(0.25, abs=1e-15)
    # bars 2 and 3 are below the running peak of 200 -> 2 bars underwater
    assert longest == 2

    # a deeper single-trough curve: 1 - 60/120 = 0.5
    eq2 = np.array([120.0, 60.0, 90.0, 130.0])
    mdd2, _ = max_drawdown(eq2)
    assert mdd2 == pytest.approx(0.5, abs=1e-15)

    # and through compute_metrics on the first curve (spaced over a realistic
    # horizon so the CAGR term stays finite; MDD is horizon-independent)
    idx = pd.date_range("2023-01-01", periods=len(eq), freq="90D", tz="UTC",
                        name="timestamp")
    m = compute_metrics(pd.Series(eq, index=idx, name="equity"),
                        _empty_trades(), timeframe="1h")
    assert m["max_drawdown"] == pytest.approx(0.25, abs=1e-15)


# ------------------------------------------------- 3) constant-return -> CAGR

def test_hourly_constant_return_cagr():
    r = 0.0002
    n = 1000
    eq = _equity(10_000.0 * (1.0 + r) ** np.arange(n))
    m = compute_metrics(eq, _empty_trades(), timeframe="1h")

    # years = (n-1) hours / (24*365); (final/initial)^(1/years) = (1+r)^(24*365)
    # so CAGR is independent of n and equals (1+r)^8760 - 1.
    hours_per_year = periods_per_year("1h")
    assert hours_per_year == pytest.approx(24 * 365, rel=1e-12)
    expected = (1.0 + r) ** (24 * 365) - 1.0
    assert m["cagr"] == pytest.approx(expected, rel=1e-9)
    # every bar has an identical positive return -> Sharpe is +inf-ish but the
    # per-bar std is ~0; guard only that CAGR (the headline) is right and finite.
    assert np.isfinite(m["cagr"])


# ------------------------------------------ 4) win_rate / profit_factor by hand

def test_win_rate_and_profit_factor_hand_built():
    pnl = [10.0, -5.0, 20.0, -4.0, -1.0]          # 2 wins, 3 losses
    trades = pd.DataFrame({
        "pnl": pnl,
        "ret_pct": [0.01, -0.005, 0.02, -0.004, -0.001],
        "bars_held": [3.0, 2.0, 5.0, 1.0, 4.0],
    })
    # equity only needs to be a valid non-empty curve for the other metrics
    eq = _equity(np.linspace(10_000.0, 10_020.0, 50))
    m = compute_metrics(eq, trades, timeframe="1h")

    assert m["n_trades"] == 5
    assert m["win_rate"] == pytest.approx(2 / 5)
    # gross win = 10 + 20 = 30 ; gross loss = 5 + 4 + 1 = 10 ; PF = 3.0
    assert m["profit_factor"] == pytest.approx(3.0)
    assert m["avg_trade_pnl"] == pytest.approx(np.mean(pnl))
    assert m["best_trade"] == 20.0
    assert m["worst_trade"] == -5.0


def test_profit_factor_no_losses_is_capped():
    trades = pd.DataFrame({
        "pnl": [5.0, 3.0],
        "ret_pct": [0.005, 0.003],
        "bars_held": [1.0, 1.0],
    })
    eq = _equity(np.linspace(10_000.0, 10_008.0, 10))
    m = compute_metrics(eq, trades, timeframe="1h")
    assert m["win_rate"] == 1.0
    # no losing trades -> profit factor is capped sentinel (999.0), never inf/NaN
    assert m["profit_factor"] == 999.0
