"""Round-4 group B strategy tests (compression_ladder, adaptive_weekday, vshape_recovery).

The generic zoo contract (domain/length, zero warmup, determinism, no-lookahead
truncation, param sensitivity) is already swept for these classes by
tests/test_strategies.py (they register at import and are SEARCHABLE). This file
adds behaviour-specific and REAL-DATA causality checks, guarded on the BTC/USDT
1d cache being present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import ohlcv_from_close

from core.strategies.round4_b import (AdaptiveWeekday, CompressionLadder,
                                      VShapeRecovery)

CLASSES = [CompressionLadder, AdaptiveWeekday, VShapeRecovery]


# ------------------------------------------------------------ real-data guard
from core.data.fetcher import cache_path, load_ohlcv  # noqa: E402
from core.backtest.runner import run_strategy_backtest  # noqa: E402

_BTC_1D_OK = cache_path("binance", "BTC/USDT", "1d").exists()
requires_btc = pytest.mark.skipif(not _BTC_1D_OK, reason="need BTC/USDT 1d cache")


@pytest.fixture(scope="module")
def btc_daily() -> pd.DataFrame:
    return load_ohlcv("binance", "BTC/USDT", "1d", since="2018-01-01", refresh=False)


# ------------------------------------------------------------- synthetic frame
@pytest.fixture(scope="module")
def daily_df() -> pd.DataFrame:
    """~4 years of daily bars with regime variety (up, crash, recovery)."""
    g = np.random.default_rng(23)
    n = 1400
    drift = np.concatenate([
        np.full(400, 0.0015),    # bull
        np.full(250, -0.004),    # crash / deep drawdown
        np.full(250, 0.003),     # V recovery
        g.normal(0.0, 0.001, n - 900),
    ])
    close = 100.0 * np.exp(np.cumsum(drift + g.normal(0.0, 0.02, n)))
    idx = pd.date_range("2019-01-01", periods=n, freq="1D", tz="UTC", name="timestamp")
    df = ohlcv_from_close(close, g)
    return df.set_axis(idx)


# ---------------------------------------------------------------------- generic
@pytest.mark.parametrize("cls", CLASSES)
def test_domain_warmup_deterministic(cls, daily_df):
    strat = cls(timeframe="1d")
    sig = strat.generate_signals(daily_df)
    assert len(sig) == len(daily_df)
    assert set(np.unique(sig)).issubset({0.0, 1.0})
    assert sig[0] == 0.0
    assert np.array_equal(sig, cls(timeframe="1d").generate_signals(daily_df))
    assert sig.sum() > 0, f"{cls.NAME} never fires on the regime-varied fixture"


@pytest.mark.parametrize("cls", CLASSES)
def test_no_lookahead_truncation_synth(cls, daily_df):
    strat = cls(timeframe="1d")
    full = np.asarray(strat.generate_signals(daily_df))
    for k in (len(daily_df) - 1, len(daily_df) - 300, len(daily_df) // 2 + 7):
        trunc = np.asarray(cls(timeframe="1d").generate_signals(daily_df.iloc[:k]))
        assert len(trunc) == k
        assert np.array_equal(full[:k], trunc), f"{cls.NAME} lookahead at k={k}"


# ------------------------------------------------ compression_ladder specifics
def test_compression_min_score_monotone(daily_df):
    n3 = CompressionLadder(timeframe="1d", min_score=3).generate_signals(daily_df).sum()
    n4 = CompressionLadder(timeframe="1d", min_score=4).generate_signals(daily_df).sum()
    # a stricter compression requirement can only reduce (or hold) entry starts;
    # via hold_stance this bounds exposure loosely but must not increase.
    assert n4 <= n3
    assert n3 > 0


def test_compression_gate_blocks_when_below_ema(daily_df):
    """With an impossibly long EMA gate (all-NaN) there can be no entries."""
    big = CompressionLadder(timeframe="1d", gate_days=10_000)
    assert big.generate_signals(daily_df).sum() == 0.0


# --------------------------------------------------- adaptive_weekday specifics
def test_weekday_uses_tomorrow_not_today(daily_df):
    """Signal must key off (t+1).dayofweek, never a future bar's data: verified by
    the truncation test above; here we assert the min_edge floor has bite."""
    n_lo = AdaptiveWeekday(timeframe="1d", min_edge=0.0).generate_signals(daily_df).sum()
    n_hi = AdaptiveWeekday(timeframe="1d", min_edge=0.001).generate_signals(daily_df).sum()
    assert n_hi <= n_lo
    assert n_lo > 0


def test_weekday_lookback_changes_signal(daily_df):
    a = AdaptiveWeekday(timeframe="1d", lookback=60).generate_signals(daily_df)
    b = AdaptiveWeekday(timeframe="1d", lookback=180).generate_signals(daily_df)
    assert not np.array_equal(a, b)


# ----------------------------------------------------- vshape_recovery specifics
def test_vshape_requires_arm_before_entry(daily_df):
    """A very deep dd_arm that never triggers => no entries at all."""
    never = VShapeRecovery(timeframe="1d", dd_arm=0.95)
    assert never.generate_signals(daily_df).sum() == 0.0


def test_vshape_dd_arm_monotone(daily_df):
    shallow = VShapeRecovery(timeframe="1d", dd_arm=0.25).generate_signals(daily_df).sum()
    deep = VShapeRecovery(timeframe="1d", dd_arm=0.45).generate_signals(daily_df).sum()
    # a deeper arm threshold is harder to trip -> fewer or equal long-bars
    assert deep <= shallow


# ------------------------------------------------------------- real BTC checks
@requires_btc
@pytest.mark.parametrize("cls", CLASSES)
def test_real_btc_no_lookahead(cls, btc_daily):
    """stance(df[:k]) == stance(df)[:k] AND size_frac agreement on REAL BTC 1d
    (ex-holdout: last 180 days dropped), k = n-300."""
    cutoff = btc_daily.index[-1] - pd.Timedelta(days=180)
    df = btc_daily[btc_daily.index <= cutoff]
    k = len(df) - 300
    full = cls()
    trunc = cls()
    f_st = np.asarray(full.generate_signals(df))
    t_st = np.asarray(trunc.generate_signals(df.iloc[:k]))
    assert len(t_st) == k
    assert np.array_equal(f_st[:k], t_st)
    f_sf = full.generate_size_frac(df)
    t_sf = trunc.generate_size_frac(df.iloc[:k])
    if f_sf is None:
        assert t_sf is None
    else:
        assert np.allclose(np.nan_to_num(np.asarray(f_sf)[:k]),
                           np.nan_to_num(np.asarray(t_sf)), atol=1e-12)


@requires_btc
@pytest.mark.parametrize("cls", CLASSES)
def test_real_btc_backtest_runs(cls, btc_daily):
    res = run_strategy_backtest(btc_daily, cls(), "1d", symbol="BTC/USDT")
    assert res.n_trades > 0
    assert np.isfinite(res.metrics["sharpe"])
    assert np.isfinite(res.equity.to_numpy()).all()
