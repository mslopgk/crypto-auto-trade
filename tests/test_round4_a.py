"""Dedicated contract tests for round-4 group A strategies.

The generic parametrized sweep in ``test_strategies.py`` already covers the
SEARCHABLE members ``vov_calm_trend`` and ``volume_shock_pullback`` (stance
domain, zero warmup, determinism, no-lookahead, param-sensitivity) on the shared
random-walk fixture. This file adds:

- Full contract coverage for :class:`WickPressure`, which is SEARCHABLE=False
  because the shared driftless fixture's symmetric wicks force ``pressure``
  negative and it can never go long there. It is exercised instead on a crafted
  ASYMMETRIC-wick uptrend (long lower wicks = genuine buying pressure).
- Registry integration + grid-budget checks for all three.
- Real-BTC no-lookahead (stance AND size_frac) + backtest smoke, guarded by
  skipif when the BTC/USDT 1d cache is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.fetcher import cache_path, load_ohlcv
from core.backtest.runner import run_strategy_backtest
from core.strategies.registry import get_strategy
from core.strategies.round4_a import (VoVCalmTrend, VolumeShockPullback,
                                       WickPressure)

_BTC_1D_OK = cache_path("binance", "BTC/USDT", "1d").exists()
requires_btc = pytest.mark.skipif(not _BTC_1D_OK, reason="no BTC/USDT 1d cache")

ALL = [WickPressure, VoVCalmTrend, VolumeShockPullback]


# --------------------------------------------------------------------------- #
# crafted fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wick_uptrend_df() -> pd.DataFrame:
    """~900 daily bars, upward drift, with LONG LOWER WICKS (buyers defend the
    dip each bar) so WickPressure's pressure proxy is positive and it trades."""
    g = np.random.default_rng(21)
    n = 900
    close = 100.0 * np.exp(np.cumsum(g.normal(0.0015, 0.02, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    top = np.maximum(open_, close)
    bot = np.minimum(open_, close)
    lower = g.uniform(0.01, 0.05, n)   # big lower wick
    upper = g.uniform(0.0, 0.005, n)   # tiny upper wick
    high = top * (1.0 + upper)
    low = bot * (1.0 - lower)
    vol = g.lognormal(6.0, 0.5, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC", name="timestamp")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx).astype(np.float64)


@pytest.fixture(scope="module")
def shock_df() -> pd.DataFrame:
    """Daily frame engineered to contain volume-shock + quiet-pullback episodes
    so VolumeShockPullback enters, confirms, and exits at least once."""
    idx = pd.date_range("2021-01-01", periods=120, freq="1D", tz="UTC", name="timestamp")
    o, h, l, c, v = [], [], [], [], []
    price = 100.0
    for i in range(120):
        vol = 1000.0
        op = price
        # baseline drift up
        cl = price * 1.001
        # engineer a loud up shock every 30 bars
        if i % 30 == 10:
            cl = price * 1.05
            vol = 8000.0
        # quiet pullback two bars after the shock
        elif i % 30 in (12, 13):
            cl = price * 0.985
            vol = 300.0
        hi = max(op, cl) * 1.004
        lo = min(op, cl) * 0.996
        o.append(op); h.append(hi); l.append(lo); c.append(cl); v.append(vol)
        price = cl
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v},
                        index=idx).astype(np.float64)


# --------------------------------------------------------------------------- #
# registry + grid budget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls,name,searchable", [
    (WickPressure, "wick_pressure", False),
    (VoVCalmTrend, "vov_calm_trend", True),
    (VolumeShockPullback, "volume_shock_pullback", True),
])
def test_registered_and_budget(cls, name, searchable):
    assert get_strategy(name) is cls
    assert cls.NAME == name
    assert getattr(cls, "SEARCHABLE", True) is searchable
    assert cls.param_grid_size() <= 150


# --------------------------------------------------------------------------- #
# WickPressure dedicated contract (asymmetric-wick uptrend)
# --------------------------------------------------------------------------- #
def test_wick_pressure_trades_and_contract(wick_uptrend_df):
    strat = WickPressure(timeframe="1d")
    sig = strat.generate_signals(wick_uptrend_df)
    assert len(sig) == len(wick_uptrend_df)
    assert set(np.unique(sig)).issubset({0.0, 1.0})
    assert sig[0] == 0.0
    assert sig.sum() > 0, "long-lower-wick uptrend must produce buying pressure"
    assert np.array_equal(sig, WickPressure(timeframe="1d").generate_signals(wick_uptrend_df))


def test_wick_pressure_param_change_alters_signals(wick_uptrend_df):
    base = WickPressure(timeframe="1d").generate_signals(wick_uptrend_df)
    changed = False
    for param in ("window", "enter_thr", "trend_days", "exit_thr"):
        for v in WickPressure.PARAM_SPACE[param]:
            if v == WickPressure.DEFAULTS.get(param):
                continue
            sig = WickPressure(timeframe="1d", **{param: v}).generate_signals(wick_uptrend_df)
            if not np.array_equal(sig, base):
                changed = True
                break
        if changed:
            break
    assert changed


def test_wick_pressure_no_lookahead(wick_uptrend_df):
    full = np.asarray(WickPressure(timeframe="1d").generate_signals(wick_uptrend_df))
    n = len(wick_uptrend_df)
    for k in (n - 1, n - 50, n // 2 + 7):
        if k < 300:
            continue
        trunc = np.asarray(WickPressure(timeframe="1d").generate_signals(wick_uptrend_df.iloc[:k]))
        assert len(trunc) == k
        assert np.array_equal(full[:k], trunc), f"lookahead at k={k}"


# --------------------------------------------------------------------------- #
# VolumeShockPullback dedicated state-machine behaviour
# --------------------------------------------------------------------------- #
def test_volume_shock_enters_on_quiet_pullback(shock_df):
    strat = VolumeShockPullback(timeframe="1d", shock_z=2.0, look_days=5, hold_max=10)
    sig = strat.generate_signals(shock_df)
    assert set(np.unique(sig)).issubset({0.0, 1.0})
    assert sig[0] == 0.0
    assert sig.sum() > 0, "engineered shock+pullback must trigger at least one long"


def test_volume_shock_no_lookahead(shock_df):
    full = np.asarray(VolumeShockPullback(timeframe="1d", shock_z=2.0).generate_signals(shock_df))
    n = len(shock_df)
    for k in (n - 1, n - 7, n - 40):
        if k < 25:
            continue
        trunc = np.asarray(
            VolumeShockPullback(timeframe="1d", shock_z=2.0).generate_signals(shock_df.iloc[:k]))
        assert len(trunc) == k
        assert np.array_equal(full[:k], trunc), f"lookahead at k={k}"


# --------------------------------------------------------------------------- #
# real BTC: no-lookahead (stance + size_frac) and backtest smoke
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def btc_daily() -> pd.DataFrame:
    return load_ohlcv("binance", "BTC/USDT", "1d", refresh=False)


@requires_btc
@pytest.mark.parametrize("cls", ALL)
def test_real_btc_no_lookahead_stance_and_size(cls, btc_daily):
    """stance(df[:k]) == stance(df)[:k] AND size_frac likewise (k=n-300)."""
    k = len(btc_daily) - 300
    full = cls()
    trunc = cls()
    f_st = np.asarray(full.generate_signals(btc_daily))
    t_st = np.asarray(trunc.generate_signals(btc_daily.iloc[:k]))
    assert len(t_st) == k
    assert np.array_equal(np.nan_to_num(f_st[:k]), np.nan_to_num(t_st))
    f_sf = full.generate_size_frac(btc_daily)
    t_sf = trunc.generate_size_frac(btc_daily.iloc[:k])
    if f_sf is None:
        assert t_sf is None
    else:
        f_sf = np.asarray(f_sf); t_sf = np.asarray(t_sf)
        assert np.allclose(np.nan_to_num(f_sf[:k]), np.nan_to_num(t_sf), atol=1e-12)


@requires_btc
@pytest.mark.parametrize("cls", ALL)
def test_real_btc_backtest_smoke(cls, btc_daily):
    res = run_strategy_backtest(btc_daily, cls(), "1d", symbol="BTC/USDT")
    assert res.n_trades > 0
    assert np.isfinite(res.metrics["sharpe"])
    assert np.isfinite(res.equity.to_numpy()).all()
