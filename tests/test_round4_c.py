"""Contract + causality tests for round-4 group C strategies.

Self-contained; reuses the synthetic OHLCV builders in conftest and (when the
BTC parquet is present) real BTC daily history for the funding-aligned checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import ohlcv_from_close
from core.data.fetcher import cache_path
from core.data.funding import funding_path
from core.strategies.round4_c import FundingQualityBreakout, TrendPullback

# ------------------------------------------------------------- fixtures / marks

_BTC_OK = cache_path("binance", "BTC/USDT", "1d").exists()
_FUNDING_OK = funding_path("BTC").exists()
requires_btc_funding = pytest.mark.skipif(
    not (_BTC_OK and _FUNDING_OK), reason="no BTC daily/funding parquet")


@pytest.fixture(scope="module")
def btc_daily() -> pd.DataFrame:
    from core.data.fetcher import load_ohlcv
    return load_ohlcv("binance", "BTC/USDT", "1d", since="2019-01-01", refresh=False)


@pytest.fixture(scope="module")
def uptrend_df() -> pd.DataFrame:
    """Long noisy uptrend (2500 1h bars) — exercises TrendPullback dips."""
    g = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(g.normal(0.0015, 0.02, 2500)))
    return ohlcv_from_close(close, g)


# ----------------------------------------------------------------- TrendPullback

def test_trend_pullback_domain_warmup_determinism(uptrend_df):
    s = TrendPullback(timeframe="1h", trend_days=5, mom_days=3)
    a = s.generate_signals(uptrend_df)
    b = s.generate_signals(uptrend_df)
    assert len(a) == len(uptrend_df)
    assert set(np.unique(a)).issubset({0.0, 1.0})
    assert a[0] == 0.0
    assert np.array_equal(a, b)
    assert a.sum() > 0  # actually fires on a real uptrend


@pytest.mark.parametrize("tf,trend_days,mom_days", [("1h", 5, 3), ("4h", 3, 2)])
def test_trend_pullback_no_lookahead(uptrend_df, tf, trend_days, mom_days):
    def mk():
        return TrendPullback(timeframe=tf, trend_days=trend_days, mom_days=mom_days)
    full = np.asarray(mk().generate_signals(uptrend_df))
    k = len(uptrend_df) - 300
    trunc = np.asarray(mk().generate_signals(uptrend_df.iloc[:k]))
    assert len(trunc) == k
    assert np.array_equal(full[:k], trunc), f"lookahead tf={tf}"


def test_trend_pullback_depth_band_gates_entries(uptrend_df):
    """A wider max_dip admits at least as many entries as a tighter one."""
    wide = TrendPullback(timeframe="1h", trend_days=5, mom_days=3, max_dip=0.30)
    narrow = TrendPullback(timeframe="1h", trend_days=5, mom_days=3, max_dip=0.05)
    assert wide.generate_signals(uptrend_df).sum() >= narrow.generate_signals(uptrend_df).sum()


def test_trend_pullback_requires_uptrend_context(make_df):
    """In a pure downtrend the uptrend context is never satisfied -> flat."""
    close = 100.0 * np.exp(np.cumsum(np.full(400, -0.01)))
    df = ohlcv_from_close(close)
    s = TrendPullback(timeframe="1h", trend_days=5, mom_days=3)
    assert (s.generate_signals(df) == 0.0).all()


# ------------------------------------------------------- FundingQualityBreakout

def test_fqb_missing_funding_is_flat(uptrend_df):
    assert not funding_path("NOFUND").exists()
    s = FundingQualityBreakout(symbol="NOFUND/USDT")
    assert (s.generate_signals(uptrend_df) == 0.0).all()


@requires_btc_funding
def test_fqb_domain_warmup_and_fires(btc_daily):
    s = FundingQualityBreakout(symbol="BTC/USDT")
    st = s.generate_signals(btc_daily)
    assert len(st) == len(btc_daily)
    assert set(np.unique(st)).issubset({0.0, 1.0})
    assert st[0] == 0.0
    assert st.sum() > 0


@requires_btc_funding
def test_fqb_no_lookahead(btc_daily):
    def mk():
        return FundingQualityBreakout(symbol="BTC/USDT")
    full = np.asarray(mk().generate_signals(btc_daily))
    k = len(btc_daily) - 300
    trunc = np.asarray(mk().generate_signals(btc_daily.iloc[:k]))
    assert len(trunc) == k
    assert np.array_equal(full[:k], trunc)


@requires_btc_funding
def test_fqb_filter_is_subset_of_no_filter(btc_daily):
    """z_max=999 disables the filter; any finite z_max must take a subset of
    those breakout entries (the funding gate only removes entries)."""
    off = FundingQualityBreakout(symbol="BTC/USDT", z_max=999.0).generate_signals(btc_daily)
    on = FundingQualityBreakout(symbol="BTC/USDT", z_max=1.0).generate_signals(btc_daily)
    assert on.sum() <= off.sum()
    assert off.sum() > on.sum()  # the filter actually removes some entries
